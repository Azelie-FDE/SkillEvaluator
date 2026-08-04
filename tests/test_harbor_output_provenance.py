# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for path-bound generated-output provenance."""

from __future__ import annotations

import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from skillevaluator.tier3 import output_provenance
from skillevaluator.tier3.harbor import secure_copy
from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks, stage_native_harbor_tasks
from skillevaluator.tier3.output_provenance import (
    GENERATED_OUTPUT_MARKER,
    is_generated_output_root,
    mark_generated_output_root,
)


@pytest.fixture(autouse=True)
def _isolated_output_provenance_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE",
        str(tmp_path / ".skillevaluator-state" / "output-provenance.key"),
    )


def _write_skill(tmp_path: Path, *, native: bool = False) -> Path:
    skill = tmp_path / "skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Complete the task.", "files": []}]),
        encoding="utf-8",
    )
    if native:
        task = skill / "evals" / "harbor" / "case-001"
        task.mkdir(parents=True)
        (task / "instruction.md").write_text("Complete the native task.\n", encoding="utf-8")
        (task / "task.toml").write_text(
            'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
            '[metadata]\nentry_id = "case-001"\n\n[environment]\n',
            encoding="utf-8",
        )
    return skill


def _in_skill_output(skill: Path, name: str) -> tuple[Path, Path]:
    declared_root = skill / name
    return declared_root, declared_root / "dataset"


def test_concurrent_first_key_creation_publishes_one_complete_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "generated"
    barrier = threading.Barrier(2)
    original_link = output_provenance.os.link

    def synchronized_link(source: Path, target: Path) -> None:
        barrier.wait(timeout=5)
        original_link(source, target)

    monkeypatch.setattr(output_provenance.os, "link", synchronized_link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        payloads = list(
            executor.map(
                lambda _index: output_provenance.generated_output_marker_payload(destination),
                range(2),
            )
        )

    key_path = Path(os.environ["SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"])
    assert payloads[0] == payloads[1]
    if os.name == "nt":
        assert 32 < key_path.stat().st_size <= 4096
    else:
        assert key_path.stat().st_size == 32
    assert key_path.stat().st_nlink == 1
    assert not list(key_path.parent.glob(".output-provenance.key.tmp-*"))


def test_interrupted_hardlink_publish_is_recovered(tmp_path: Path) -> None:
    key_path = Path(os.environ["SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"])
    output_provenance.generated_output_marker_payload(tmp_path / "seed")
    temporary = key_path.parent / ".output-provenance.key.tmp-interrupted"
    os.link(key_path, temporary)

    payload = output_provenance.generated_output_marker_payload(tmp_path / "generated")

    assert payload.startswith(b"SkillEvaluator generated output v2\n")
    assert not temporary.exists()
    assert key_path.stat().st_nlink == 1


def test_fixed_marker_cannot_authorize_authored_source_replacement(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    declared_root, output = _in_skill_output(skill, "fixed-marker-results")
    authored_task = output / "case-001"
    authored_task.mkdir(parents=True)
    (output / GENERATED_OUTPUT_MARKER).write_bytes(b"SkillEvaluator generated output v1\n")
    (authored_task / "SKILL.md").write_text("# Authored source\n", encoding="utf-8")
    sentinel = authored_task / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"runtime skill source|marker"):
        generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_generated_marker_is_bound_to_its_original_destination(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    original_root, original = _in_skill_output(skill, "original-results")
    copied_root, copied = _in_skill_output(skill, "copied-results")
    generate_harbor_tasks(skill, original, repo_context_exclude_paths=(original_root,))
    copied.mkdir(parents=True)
    shutil.copy2(original / GENERATED_OUTPUT_MARKER, copied / GENERATED_OUTPUT_MARKER)
    authored_task = copied / "case-001"
    authored_task.mkdir()
    (authored_task / "SKILL.md").write_text("# Authored source\n", encoding="utf-8")
    sentinel = authored_task / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"runtime skill source|marker"):
        generate_harbor_tasks(skill, copied, repo_context_exclude_paths=(copied_root,))

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


@pytest.mark.parametrize("native", [False, True])
def test_invalid_marker_blocks_in_skill_atomic_replacement(tmp_path: Path, native: bool) -> None:
    skill = _write_skill(tmp_path, native=native)
    declared_root, output = _in_skill_output(skill, "invalid-marker-results")
    output.mkdir(parents=True)
    (output / GENERATED_OUTPUT_MARKER).write_text("invalid\n", encoding="utf-8")
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    stager = stage_native_harbor_tasks if native else generate_harbor_tasks

    with pytest.raises(ValueError, match="marker"):
        stager(skill, output, repo_context_exclude_paths=(declared_root,))

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_partial_atomic_publication_preserves_signed_output_and_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _write_skill(tmp_path)
    declared_root, output = _in_skill_output(skill, "partial-results")
    generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))
    old_dataset = (output / "dataset.toml").read_bytes()
    original_rename = secure_copy._rename_no_replace
    rename_calls = 0

    def _fail_second_rename(*args: object, **kwargs: object) -> None:
        nonlocal rename_calls
        rename_calls += 1
        if rename_calls == 2:
            raise OSError("injected generated-output publish failure after old destination moved")
        original_rename(*args, **kwargs)

    monkeypatch.setattr(secure_copy, "_rename_no_replace", _fail_second_rename)
    with pytest.raises(OSError, match="generated-output publish"):
        generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))

    assert (output / "dataset.toml").read_bytes() == old_dataset
    assert is_generated_output_root(output)

    monkeypatch.setattr(secure_copy, "_rename_no_replace", original_rename)
    tasks = generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))
    assert tasks
    assert is_generated_output_root(output)


def test_external_output_retains_unmarked_overwrite_behavior(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    output = tmp_path / "external-output"
    output.mkdir()
    (output / GENERATED_OUTPUT_MARKER).write_text("invalid\n", encoding="utf-8")
    sentinel = output / "keep.txt"
    sentinel.write_text("replace\n", encoding="utf-8")

    tasks = generate_harbor_tasks(skill, output)

    assert tasks
    assert not sentinel.exists()
    assert not (output / GENERATED_OUTPUT_MARKER).exists()


def test_unsigned_v2_marker_does_not_create_private_key(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    declared_root, output = _in_skill_output(skill, "unsigned-results")
    output.mkdir(parents=True)
    (output / GENERATED_OUTPUT_MARKER).write_bytes(b"SkillEvaluator generated output v2\n" + (b"A" * 43) + b"\n")
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    key_path = Path(os.environ["SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"])

    with pytest.raises(ValueError, match="marker"):
        generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert not key_path.exists()


def test_key_override_inside_skill_is_rejected_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _write_skill(tmp_path)
    declared_root, output = _in_skill_output(skill, "key-location-results")
    key_path = skill / "private-output-key"
    monkeypatch.setenv("SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE", str(key_path))

    with pytest.raises(ValueError, match="outside evaluated and generated trees"):
        generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))

    assert not key_path.exists()
    assert not output.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_fifo_marker_and_key_are_rejected_without_blocking(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path)
    declared_root, output = _in_skill_output(skill, "fifo-results")
    output.mkdir(parents=True)
    os.mkfifo(output / GENERATED_OUTPUT_MARKER)
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match="marker"):
        generate_harbor_tasks(skill, output, repo_context_exclude_paths=(declared_root,))
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"

    key_path = Path(os.environ["SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE"])
    key_path.parent.mkdir(parents=True, mode=0o700)
    os.mkfifo(key_path)
    with pytest.raises(ValueError, match="single-link regular file"):
        mark_generated_output_root(tmp_path / "key-fifo-output")


def test_symlinked_private_marker_root_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "marker-target"
    target.mkdir()
    linked_root = tmp_path / "marker-link"
    linked_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match=r"symlink|reparse|junction"):
        output_provenance.write_generated_output_marker(linked_root, destination=tmp_path / "destination")

    assert list(target.iterdir()) == []
