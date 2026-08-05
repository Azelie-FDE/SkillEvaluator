# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable discovery for completed Tier 3 result directories."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from skillevaluator.tier3 import results_location
from skillevaluator.tier3.results_location import external_results_root, resolve_latest_results


def _write_completed_run(root: Path, run_id: str) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps({"run_id": run_id, "agents": {}}),
        encoding="utf-8",
    )
    return run_dir


def test_latest_results_falls_back_to_newest_completed_run_without_symlink(tmp_path: Path) -> None:
    """Windows users can discover runs when creating ``latest`` is not permitted."""
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    _write_completed_run(root, "20260705_120000")
    newest = _write_completed_run(root, "20260705_130000")

    resolved = resolve_latest_results(skill_path, cli_results_dir, environ={})

    assert resolved == newest


def test_latest_results_fallback_accepts_unique_suffixed_run_ids(tmp_path: Path) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    _write_completed_run(root, "20260705_120000_111_aaaaaaaaaaaa")
    newest = _write_completed_run(root, "20260705_130000_222_bbbbbbbbbbbb")

    resolved = resolve_latest_results(skill_path, cli_results_dir, environ={})

    assert resolved == newest


def test_latest_results_same_timestamp_uses_completion_mtime(tmp_path: Path) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    lexically_later = _write_completed_run(root, "20260705_130000_999_ffffffffffff")
    completed_later = _write_completed_run(root, "20260705_130000_111_aaaaaaaaaaaa")
    os.utime(lexically_later / "result.json", ns=(100, 100))
    os.utime(completed_later / "result.json", ns=(200, 200))

    resolved = resolve_latest_results(skill_path, cli_results_dir, environ={})

    assert resolved == completed_later


def test_latest_results_same_completion_time_uses_deterministic_name_order(tmp_path: Path) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    lexically_first = _write_completed_run(root, "20260705_130000_111_aaaaaaaaaaaa")
    lexically_last = _write_completed_run(root, "20260705_130000_999_ffffffffffff")
    os.utime(lexically_first / "result.json", ns=(100, 100))
    os.utime(lexically_last / "result.json", ns=(100, 100))

    resolved = resolve_latest_results(skill_path, cli_results_dir, environ={})

    assert resolved == lexically_last


def test_publish_latest_results_is_atomic_under_concurrency(tmp_path: Path) -> None:
    publisher = getattr(results_location, "publish_latest_results", None)
    assert callable(publisher), "results_location must expose a shared latest-link publisher"
    root = tmp_path / "results"
    run_ids = [f"20260705_130000_{index:03d}_aaaaaaaaaaaa" for index in range(8)]
    for run_id in run_ids:
        (root / run_id).mkdir(parents=True)

    with ThreadPoolExecutor(max_workers=8) as executor:
        published = list(executor.map(lambda run_id: publisher(root, run_id), run_ids))
    if not any(published):  # pragma: no cover - Windows installations without symlink privileges
        pytest.skip("symlink creation is unavailable on this host")

    latest = root / "latest"
    assert latest.is_symlink()
    assert str(latest.readlink()) in run_ids
    assert not list(root.glob(".latest-*.tmp"))


def test_publish_latest_results_preserves_real_directory_and_cleans_temp_link(tmp_path: Path) -> None:
    publisher = getattr(results_location, "publish_latest_results", None)
    assert callable(publisher), "results_location must expose a shared latest-link publisher"
    root = tmp_path / "results"
    latest = root / "latest"
    latest.mkdir(parents=True)
    marker = latest / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    run_id = "20260705_130000_111_aaaaaaaaaaaa"
    (root / run_id).mkdir()

    publisher(root, run_id)

    assert latest.is_dir()
    assert not latest.is_symlink()
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(root.glob(".latest-*.tmp"))


def test_latest_results_fallback_ignores_hidden_partial_and_malformed_directories(tmp_path: Path) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    expected = _write_completed_run(root, "20260705_120000")

    _write_completed_run(root, ".20260705_180000")
    _write_completed_run(root, "_20260705_170000")
    _write_completed_run(root, "not-a-timestamp")
    (root / "20260705_160000").mkdir()
    malformed = root / "20260705_150000"
    malformed.mkdir()
    (malformed / "result.json").write_text("{not json", encoding="utf-8")
    mismatched = root / "20260705_140000"
    mismatched.mkdir()
    (mismatched / "result.json").write_text(
        json.dumps({"run_id": "20260705_999999", "agents": {}}),
        encoding="utf-8",
    )

    resolved = resolve_latest_results(skill_path, cli_results_dir, environ={})

    assert resolved == expected


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
        pytest.skip(f"symlinks unavailable on this host: {exc}")


def test_latest_results_prefers_valid_relative_latest_link(tmp_path: Path) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    selected = _write_completed_run(root, "20260705_120000")
    _write_completed_run(root, "20260705_130000")
    latest = root / "latest"
    _symlink_or_skip(latest, Path(selected.name))

    assert resolve_latest_results(skill_path, cli_results_dir, environ={}) == latest


@pytest.mark.parametrize(
    "latest_kind",
    ["directory", "dangling", "escaping", "incomplete", "legacy", "absolute", "nested-alias"],
)
def test_latest_results_ignores_invalid_latest_pointer_and_falls_back(
    tmp_path: Path,
    latest_kind: str,
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    older = _write_completed_run(root, "20260705_120000")
    expected = _write_completed_run(root, "20260705_130000")
    latest = root / "latest"

    if latest_kind == "directory":
        latest.mkdir()
    elif latest_kind == "dangling":
        _symlink_or_skip(latest, Path("missing-run"))
    elif latest_kind == "escaping":
        outside = _write_completed_run(tmp_path / "outside", "20260705_140000")
        _symlink_or_skip(latest, outside)
    elif latest_kind == "incomplete":
        (root / "20260705_140000").mkdir()
        _symlink_or_skip(latest, Path("20260705_140000"))
    elif latest_kind == "legacy":
        legacy = root / "legacy-run"
        legacy.mkdir()
        (legacy / "result.json").write_text(json.dumps({"run_id": legacy.name}), encoding="utf-8")
        _symlink_or_skip(latest, Path(legacy.name))
    elif latest_kind == "absolute":
        _symlink_or_skip(latest, older.resolve())
    else:
        (root / "alias").mkdir()
        _symlink_or_skip(latest, Path("alias") / ".." / older.name)

    assert resolve_latest_results(skill_path, cli_results_dir, environ={}) == expected


@pytest.mark.parametrize("latest_kind", ["directory", "escaping", "incomplete"])
def test_invalid_latest_without_completed_run_returns_safe_absent_path(
    tmp_path: Path,
    latest_kind: str,
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    root.mkdir(parents=True)
    latest = root / "latest"
    if latest_kind == "directory":
        latest.mkdir()
    elif latest_kind == "escaping":
        outside = _write_completed_run(tmp_path / "outside", "20260705_140000")
        _symlink_or_skip(latest, outside)
    else:
        partial = root / "20260705_140000"
        partial.mkdir()
        _symlink_or_skip(latest, Path(partial.name))

    resolved = resolve_latest_results(skill_path, cli_results_dir, environ={})

    assert resolved != latest
    assert not os.path.lexists(resolved)
    assert resolved.parent != latest
    assert not resolved.resolve(strict=False).is_relative_to((tmp_path / "outside").resolve())


def test_all_invalid_latest_entries_return_collision_checked_hidden_sentinel(tmp_path: Path) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    primary_root = external_results_root(cli_results_dir, skill_path)
    legacy_root = skill_path / "evals" / "results"
    for root in (primary_root, legacy_root):
        (root / "latest").mkdir(parents=True)

    resolved = resolve_latest_results(skill_path, cli_results_dir, environ={})

    assert resolved.parent == primary_root
    assert resolved.name.startswith(".latest-unavailable-")
    assert not os.path.lexists(resolved)


@pytest.mark.parametrize("artifact_kind", ["symlink", "fifo", "hardlink"])
def test_latest_results_fallback_rejects_unsafe_result_artifact(tmp_path: Path, artifact_kind: str) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    expected = _write_completed_run(root, "20260705_120000")
    unsafe = root / "20260705_130000"
    unsafe.mkdir()
    result_path = unsafe / "result.json"
    source = tmp_path / "result-source.json"
    source.write_text(json.dumps({"run_id": unsafe.name}), encoding="utf-8")
    try:
        if artifact_kind == "symlink":
            result_path.symlink_to(source)
        elif artifact_kind == "fifo":
            if not hasattr(os, "mkfifo"):
                pytest.skip("FIFO creation is unavailable on this host")
            os.mkfifo(result_path)
        else:
            result_path.hardlink_to(source)
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"{artifact_kind} creation is unavailable on this host: {exc}")

    assert resolve_latest_results(skill_path, cli_results_dir, environ={}) == expected


@pytest.mark.parametrize("reparse_kind", ["run-directory", "result"])
def test_latest_results_rejects_windows_reparse_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reparse_kind: str,
) -> None:
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    cli_results_dir = tmp_path / "results"
    root = external_results_root(cli_results_dir, skill_path)
    expected = _write_completed_run(root, "20260705_120000")
    unsafe = _write_completed_run(root, "20260705_130000")
    marked = unsafe if reparse_kind == "run-directory" else unsafe / "result.json"
    detect_link = results_location._path_is_link_or_reparse

    def mock_windows_reparse(path: Path, metadata: os.stat_result) -> bool:
        return path == marked or detect_link(path, metadata)

    monkeypatch.setattr(results_location, "_path_is_link_or_reparse", mock_windows_reparse)

    assert resolve_latest_results(skill_path, cli_results_dir, environ={}) == expected
