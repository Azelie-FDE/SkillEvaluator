# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for per-entry eval input staging.

Ports Skill Evaluator 0.7.22 ``0d17f5e`` ("upload staged eval inputs to standard sandboxes")
into the in-process Tier 3 engine (``tier3/harbor/adapter.py``). Entries that
declare ``files`` stage only those refs; entries that omit ``files`` retain the
legacy shared ``evals/files/`` behavior. All refs retain traversal protection.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor.adapter import (
    _entry_file_refs,
    _resolve_entry_file_ref,
    _stage_task_inputs,
    generate_harbor_tasks,
)


def _make_skill(tmp_path: Path) -> tuple[Path, Path, Path]:
    skill = tmp_path / "myskill"
    (skill / "SKILL.md").parent.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n")
    evals = skill / "evals"
    files = evals / "files"
    files.mkdir(parents=True)
    (files / "global.txt").write_text("global")
    (files / "unrelated.txt").write_text("unrelated")
    data = evals / "data"
    data.mkdir()
    (data / "case1.txt").write_text("case1")
    env_dir = tmp_path / "task" / "environment"
    env_dir.mkdir(parents=True)
    return skill, evals, env_dir


class TestEntryFileRefs:
    def test_none_returns_empty(self):
        assert _entry_file_refs({"id": "t"}) == []

    def test_string_is_wrapped(self):
        assert _entry_file_refs({"files": "data/case1.txt"}) == ["data/case1.txt"]

    def test_explicit_null_returns_empty(self):
        assert _entry_file_refs({"files": None}) == []

    def test_list_is_passed_through(self):
        assert _entry_file_refs({"files": ["a/b.txt", " c/d.txt "]}) == ["a/b.txt", "c/d.txt"]

    def test_non_string_entry_rejected(self):
        with pytest.raises(ValueError, match="must be a string"):
            _entry_file_refs({"id": "t", "files": [123]})


class TestStageTaskInputs:
    def test_explicit_files_stage_only_declared_refs(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        entry = {"id": "t1", "files": ["data/case1.txt"]}
        staged = _stage_task_inputs(
            env_dir, input_files_dir=evals / "files", entry=entry, source_skill_path=skill, evals_dir=evals
        )
        assert staged is True
        paths = sorted(
            p.relative_to(env_dir / "input").as_posix() for p in (env_dir / "input").rglob("*") if p.is_file()
        )
        assert paths == ["data/case1.txt"]

    def test_explicit_file_under_shared_directory_stages_only_that_file(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1", "files": "files/global.txt"},
            source_skill_path=skill,
            evals_dir=evals,
        )
        assert staged is True
        paths = sorted(
            p.relative_to(env_dir / "input").as_posix() for p in (env_dir / "input").rglob("*") if p.is_file()
        )
        assert paths == ["global.txt"]

    def test_omitted_files_stages_entire_shared_directory(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1"},
            source_skill_path=skill,
            evals_dir=evals,
        )
        assert staged is True
        paths = sorted(
            p.relative_to(env_dir / "input").as_posix() for p in (env_dir / "input").rglob("*") if p.is_file()
        )
        assert paths == ["global.txt", "unrelated.txt"]

    @pytest.mark.parametrize("files", [[], None, "", "   "])
    def test_explicit_empty_files_stages_nothing_and_cleans_stale_input(self, tmp_path: Path, files: object):
        skill, evals, env_dir = _make_skill(tmp_path)
        input_dir = env_dir / "input"
        input_dir.mkdir()
        (input_dir / "stale.txt").write_text("stale")

        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1", "files": files},
            source_skill_path=skill,
            evals_dir=evals,
        )

        assert staged is False
        assert not input_dir.exists()

    @pytest.mark.parametrize("stale_kind", ["file", "symlink", "fifo"])
    def test_explicit_empty_files_safely_cleans_non_directory_input(self, tmp_path: Path, stale_kind: str):
        skill, evals, env_dir = _make_skill(tmp_path)
        input_path = env_dir / "input"
        symlink_target = tmp_path / "outside.txt"
        symlink_target.write_text("keep")
        if stale_kind == "file":
            input_path.write_text("stale")
        elif stale_kind == "symlink":
            input_path.symlink_to(symlink_target)
        else:
            if not hasattr(os, "mkfifo"):
                pytest.skip("FIFOs are unavailable on this platform")
            os.mkfifo(input_path)

        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1", "files": []},
            source_skill_path=skill,
            evals_dir=evals,
        )

        assert staged is False
        assert not os.path.lexists(input_path)
        assert symlink_target.read_text() == "keep"

    def test_explicit_refs_replace_stale_input(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        input_dir = env_dir / "input"
        input_dir.mkdir()
        (input_dir / "stale.txt").write_text("stale")

        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1", "files": ["data/case1.txt"]},
            source_skill_path=skill,
            evals_dir=evals,
        )

        assert staged is True
        paths = sorted(p.relative_to(input_dir).as_posix() for p in input_dir.rglob("*") if p.is_file())
        assert paths == ["data/case1.txt"]

    def test_no_inputs_returns_false(self, tmp_path: Path):
        skill = tmp_path / "myskill"
        evals = skill / "evals"
        evals.mkdir(parents=True)
        env_dir = tmp_path / "task" / "environment"
        env_dir.mkdir(parents=True)
        staged = _stage_task_inputs(
            env_dir, input_files_dir=None, entry={"id": "t"}, source_skill_path=skill, evals_dir=evals
        )
        assert staged is False


def test_generated_tasks_apply_per_entry_input_isolation(tmp_path: Path):
    skill, evals, _ = _make_skill(tmp_path)
    entries = [
        {"id": "legacy", "question": "legacy"},
        {"id": "selected", "question": "selected", "files": ["data/case1.txt"]},
        {"id": "empty", "question": "empty", "files": []},
        {"id": "null", "question": "null", "files": None},
    ]
    (evals / "evals.json").write_text(json.dumps(entries))

    task_dirs = generate_harbor_tasks(skill, tmp_path / "generated")
    tasks = {task.name: task for task in task_dirs}

    def staged_paths(case_id: str) -> list[str]:
        input_dir = tasks[case_id] / "environment" / "input"
        if not input_dir.exists():
            return []
        return sorted(path.relative_to(input_dir).as_posix() for path in input_dir.rglob("*") if path.is_file())

    assert staged_paths("legacy") == ["global.txt", "unrelated.txt"]
    assert staged_paths("selected") == ["data/case1.txt"]
    assert staged_paths("empty") == []
    assert staged_paths("null") == []

    for case_id in ("legacy", "selected"):
        dockerfile = (tasks[case_id] / "environment" / "Dockerfile").read_text()
        assert "COPY input/ /workspace/input/" in dockerfile
    for case_id in ("empty", "null"):
        dockerfile = (tasks[case_id] / "environment" / "Dockerfile").read_text()
        assert "COPY input/ /workspace/input/" not in dockerfile


@pytest.mark.parametrize("base_image", ["", "example.invalid/eval-base:latest"])
def test_generated_task_copies_explicit_ref_without_shared_files_directory(tmp_path: Path, base_image: str):
    skill, evals, _ = _make_skill(tmp_path)
    shutil.rmtree(evals / "files")
    entries = [{"id": "selected", "question": "selected", "files": "data/case1.txt"}]
    (evals / "evals.json").write_text(json.dumps(entries))

    task = generate_harbor_tasks(skill, tmp_path / "generated", base_image=base_image)[0]

    input_dir = task / "environment" / "input"
    assert [path.relative_to(input_dir).as_posix() for path in input_dir.rglob("*") if path.is_file()] == [
        "data/case1.txt"
    ]
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert "COPY input/ /workspace/input/" in dockerfile


class TestResolveEntryFileRef:
    def test_traversal_outside_evals_blocked(self, tmp_path: Path):
        skill, evals, _ = _make_skill(tmp_path)
        with pytest.raises((ValueError, FileNotFoundError)):
            _resolve_entry_file_ref(
                "../../etc/passwd", skill_path=skill, evals_dir=evals, input_files_dir=evals / "files"
            )

    def test_absolute_path_rejected(self, tmp_path: Path):
        skill, evals, _ = _make_skill(tmp_path)
        absolute_ref = str(Path(tmp_path.anchor) / "outside.txt")
        with pytest.raises(ValueError, match="relative to evals/"):
            _resolve_entry_file_ref(absolute_ref, skill_path=skill, evals_dir=evals, input_files_dir=None)

    def test_uri_scheme_rejected(self, tmp_path: Path):
        skill, evals, _ = _make_skill(tmp_path)
        with pytest.raises(ValueError, match="unsupported URI scheme"):
            _resolve_entry_file_ref("https://example.com/x", skill_path=skill, evals_dir=evals, input_files_dir=None)
