# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared result-location helpers for local skill evaluation commands."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ENV_RESULTS_DIR = "SKILLEVALUATOR_RESULTS_DIR"
_RUN_TIMESTAMP_FORMATS = (("%Y%m%d_%H%M%S", 15), ("%Y-%m-%d_%H%M%S", 17))


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def legacy_results_root(skill_path: Path) -> Path:
    """Return the historical in-skill results directory."""
    return skill_path.expanduser().resolve() / "evals" / "results"


def skill_results_name(skill_path: Path) -> str:
    """Return the directory name used under an external results root."""
    return skill_path.expanduser().resolve().name


def external_results_root(root: str | Path, skill_path: Path) -> Path:
    """Return ``<root>/<skill-name>`` for a global or CLI results root."""
    return _expand(root) / skill_results_name(skill_path)


def env_results_root(skill_path: Path, *, environ: dict[str, str] | None = None) -> Path | None:
    """Return the env-configured results root for a skill, if configured."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_RESULTS_DIR)
    if not raw:
        return None
    return external_results_root(raw, skill_path)


def resolve_results_root(
    skill_path: Path,
    cli_results_dir: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Resolve the per-skill results root for writes.

    Precedence:
    1. command ``--results-dir`` root
    2. ``SKILLEVALUATOR_RESULTS_DIR`` root
    3. legacy ``<skill>/evals/results``
    """
    if cli_results_dir is not None:
        return external_results_root(cli_results_dir, skill_path)
    configured = env_results_root(skill_path, environ=environ)
    if configured is not None:
        return configured
    return legacy_results_root(skill_path)


def iter_candidate_results_roots(
    skill_path: Path,
    cli_results_dir: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> list[Path]:
    """Return read candidates in precedence order, with legacy fallback.

    Read commands should honor the same primary resolution as write commands,
    but falling back to the legacy location avoids hiding old runs when a user
    has newly configured ``SKILLEVALUATOR_RESULTS_DIR``.
    """
    roots: list[Path] = []
    if cli_results_dir is not None:
        roots.append(external_results_root(cli_results_dir, skill_path))
        configured = env_results_root(skill_path, environ=environ)
        if configured is not None and configured not in roots:
            roots.append(configured)
    else:
        roots.append(resolve_results_root(skill_path, environ=environ))
    legacy = legacy_results_root(skill_path)
    if legacy not in roots:
        roots.append(legacy)
    return roots


def _run_timestamp(name: str) -> datetime | None:
    for timestamp_format, prefix_length in _RUN_TIMESTAMP_FORMATS:
        prefix = name[:prefix_length]
        suffix = name[prefix_length:]
        if suffix and not suffix.startswith("_"):
            continue
        try:
            return datetime.strptime(prefix, timestamp_format)  # noqa: DTZ007 -- directory names have no timezone
        except ValueError:
            continue
    return None


def _path_is_link_or_reparse(path: Path, metadata: os.stat_result) -> bool:
    """Return whether a result path is a symlink, junction, or reparse point."""
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag):
        return True
    is_junction = getattr(path, "is_junction", None)
    if not callable(is_junction):
        return False
    try:
        return bool(is_junction())
    except (OSError, RuntimeError):
        return True


def run_directory_sort_key(
    candidate: Path,
    *,
    require_completed_result: bool = False,
) -> tuple[datetime, int, str] | None:
    """Return shared newest-first ordering metadata for a result directory.

    Timestamp-shaped directories are current runs and are visible only after
    their final ``result.json`` identifies the directory's run id. Historical
    non-timestamp summary directories remain available to compare workflows.
    """
    if candidate.name.startswith((".", "_")) or candidate.name == "latest":
        return None
    try:
        candidate_metadata = candidate.lstat()
        if _path_is_link_or_reparse(candidate, candidate_metadata) or not stat.S_ISDIR(candidate_metadata.st_mode):
            return None
    except OSError:
        return None

    timestamp = _run_timestamp(candidate.name)
    is_current_run = timestamp is not None
    if not is_current_run:
        if require_completed_result:
            return None
        timestamp = datetime.min  # noqa: DTZ901 -- result directory timestamps are intentionally timezone-free

    completed = False
    completion_mtime = -1
    try:
        result_path = candidate / "result.json"
        before = result_path.lstat()
        if _path_is_link_or_reparse(result_path, before) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return None
        result = json.loads(result_path.read_text(encoding="utf-8"))
        after = result_path.lstat()
        if (
            _path_is_link_or_reparse(result_path, after)
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            return None
        completion_mtime = after.st_mtime_ns
        completed = isinstance(result, dict) and result.get("run_id") == candidate.name
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    if (is_current_run or require_completed_result) and not completed:
        return None
    if not completed:
        completion_mtime = -1
    return timestamp, completion_mtime, candidate.name


def ordered_run_directories(root: Path, *, completed_only: bool = False) -> list[Path]:
    """Return result directories in shared newest-first order."""
    try:
        candidates = root.iterdir()
    except OSError:
        return []

    ordered: list[tuple[tuple[datetime, int, str], Path]] = []
    try:
        for candidate in candidates:
            key = run_directory_sort_key(candidate, require_completed_result=completed_only)
            if key is not None:
                ordered.append((key, candidate))
    except OSError:
        return []
    ordered.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ordered]


def _newest_completed_run(root: Path) -> Path | None:
    """Return the newest complete timestamped run without relying on symlinks."""
    return next(iter(ordered_run_directories(root, completed_only=True)), None)


def publish_latest_results(root: Path, run_id: str) -> bool:
    """Atomically replace ``latest`` with a relative symlink to ``run_id``."""
    temporary = root / f".latest-{os.getpid()}-{uuid4().hex}.tmp"
    try:
        temporary.symlink_to(run_id)
        os.replace(temporary, root / "latest")  # noqa: PTH105 -- explicit atomic replacement is the contract
    except OSError:
        with suppress(OSError):
            temporary.unlink()
        return False
    return True


def resolve_latest_results(
    skill_path: Path,
    cli_results_dir: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Return the best available ``latest`` results path for read workflows."""
    roots = iter_candidate_results_roots(
        skill_path,
        cli_results_dir,
        environ=environ,
    )
    first_missing_latest: Path | None = None
    for root in roots:
        latest = root / "latest"
        if not os.path.lexists(latest) and first_missing_latest is None:
            first_missing_latest = latest
        if latest.is_symlink():
            try:
                link_target = latest.readlink()
                if link_target.is_absolute() or len(link_target.parts) != 1 or link_target.parts[0] in {"", ".", ".."}:
                    raise ValueError("latest must name one relative run directory")
                resolved_root = root.resolve(strict=True)
                target = latest.resolve(strict=True)
            except (OSError, RuntimeError, ValueError):
                pass
            else:
                if (
                    link_target.name == target.name
                    and target.parent == resolved_root
                    and run_directory_sort_key(target, require_completed_result=True) is not None
                ):
                    return latest
        fallback = _newest_completed_run(root)
        if fallback is not None:
            return fallback
    if first_missing_latest is not None:
        return first_missing_latest
    while True:
        unavailable = roots[0] / f".latest-unavailable-{uuid4().hex}"
        if not os.path.lexists(unavailable):
            return unavailable


def resolve_explicit_or_latest_results(
    skill_path: Path,
    from_results: str | Path | None = None,
    cli_results_dir: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Resolve a specific run path or the latest run for refinement/reporting."""
    if from_results is not None:
        return _expand(from_results)
    return resolve_latest_results(skill_path, cli_results_dir, environ=environ)


def git_root_for(path: Path) -> Path | None:
    """Return the containing git repo root for ``path``, if it is in a repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path.expanduser().resolve()), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return Path(root).resolve() if root else None


def gitignore_entry_for_skill_results(skill_path: Path, repo_root: Path | None = None) -> str | None:
    """Return a repo-root-relative ignore entry for a skill's generated results."""
    skill_path = skill_path.expanduser().resolve()
    repo_root = repo_root or git_root_for(skill_path)
    if repo_root is None:
        return None
    try:
        rel = legacy_results_root(skill_path).relative_to(repo_root)
    except ValueError:
        return None
    return f"/{rel.as_posix()}/"


def ensure_skill_results_gitignore(skill_path: Path) -> tuple[Path | None, str | None, bool]:
    """Ensure the legacy in-repo results directory is ignored.

    Returns ``(.gitignore path, entry, changed)``. If the skill is not inside a
    git repository, returns ``(None, None, False)``.
    """
    repo_root = git_root_for(skill_path)
    entry = gitignore_entry_for_skill_results(skill_path, repo_root=repo_root)
    if repo_root is None or entry is None:
        return None, None, False

    gitignore_path = repo_root / ".gitignore"
    try:
        text = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
    except OSError:
        return gitignore_path, entry, False

    lines = {line.strip() for line in text.splitlines()}
    normalized_lines = {line.lstrip("/") for line in lines}
    if entry in lines or entry.lstrip("/") in normalized_lines:
        return gitignore_path, entry, False

    suffix = "" if not text or text.endswith("\n") else "\n"
    gitignore_path.write_text(f"{text}{suffix}{entry}\n", encoding="utf-8")
    return gitignore_path, entry, True
