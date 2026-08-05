# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared result-location helpers for local skill evaluation commands."""

from __future__ import annotations

import json
import math
import os
import stat
import subprocess
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from skillevaluator.tier3.output_provenance import GENERATED_OUTPUT_MARKER
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot

ENV_RESULTS_DIR = "SKILLEVALUATOR_RESULTS_DIR"

_RUN_TIMESTAMP_FORMATS = (("%Y%m%d_%H%M%S", 15), ("%Y-%m-%d_%H%M%S", 17))

# A current completed run carries ``run_config.json`` plus an atomically
# written ``result.json`` whose run identity matches its directory. Historical
# runs created before run-level results instead require a usable per-agent
# summary and must predate the generated-output provenance marker. When a
# historical summary carries status or attempt coverage, those must be complete.
_RUN_COMPLETION_ARTIFACTS = ("run_config.json",)
_FINAL_RESULT_ARTIFACT = "result.json"


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
    """Parse the timestamp prefix shared by current and legacy run names."""
    for timestamp_format, prefix_length in _RUN_TIMESTAMP_FORMATS:
        prefix = name[:prefix_length]
        suffix = name[prefix_length:]
        if suffix and not suffix.startswith("_"):
            continue
        try:
            return datetime.strptime(prefix, timestamp_format)  # noqa: DTZ007 -- run names have no timezone
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


def _node_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return the fields that must remain stable while reading an artifact."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _is_single_link_regular(path: Path, metadata: os.stat_result) -> bool:
    return not _path_is_link_or_reparse(path, metadata) and stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


def _read_stable_json(
    secure_root: SecureRoot,
    path: Path,
    before: os.stat_result,
) -> tuple[object, os.stat_result, bytes] | None:
    """Read one JSON artifact through the pinned run root."""
    if not _is_single_link_regular(path, before):
        return None
    try:
        relative_path = path.absolute().relative_to(secure_root.root)
        raw, opened = secure_root.read_bytes(relative_path, before.st_size, expected=before)
        after = path.lstat()
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError, SecurePathError, ValueError):
        return None
    if (
        not _is_single_link_regular(path, after)
        or _node_fingerprint(opened) != _node_fingerprint(before)
        or _node_fingerprint(after) != _node_fingerprint(opened)
        or len(raw) != opened.st_size
    ):
        return None
    return payload, opened, raw


def _artifact_snapshot_is_unchanged(
    secure_root: SecureRoot,
    snapshot: tuple[Path, bytes, os.stat_result],
) -> bool:
    """Re-read one artifact to detect ordinary changes during validation."""
    path, expected_raw, expected_metadata = snapshot
    try:
        current_metadata = path.lstat()
    except OSError:
        return False
    observed = _read_stable_json(secure_root, path, current_metadata)
    return bool(
        observed is not None
        and observed[2] == expected_raw
        and _node_fingerprint(observed[1]) == _node_fingerprint(expected_metadata)
    )


def _legacy_run_agents(payload: object) -> tuple[str, ...] | None:
    """Return configured result-agent keys from an authentic historical config."""
    if not isinstance(payload, dict) or not isinstance(payload.get("harbor"), dict) or not payload["harbor"]:
        return None
    if payload.get("task_source") not in {"evals_json", "native_harbor"}:
        return None
    agents = payload.get("agents")
    if not isinstance(agents, dict) or not agents:
        return None
    normalized: list[str] = []
    for agent, metadata in agents.items():
        if (
            not isinstance(agent, str)
            or not agent
            or agent in {".", ".."}
            or "/" in agent
            or "\\" in agent
            or not isinstance(metadata, dict)
            or not isinstance(metadata.get("agent"), str)
            or not metadata["agent"].strip()
            or not isinstance(metadata.get("model"), str)
            or not metadata["model"].strip()
            or not isinstance(metadata.get("source"), str)
            or not metadata["source"].strip()
            or not isinstance(metadata.get("occurrence"), str)
            or not metadata["occurrence"].isdigit()
            or int(metadata["occurrence"]) < 1
        ):
            return None
        normalized.append(agent)
    return tuple(sorted(normalized))


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _legacy_pass_at_k_is_complete(
    payload: object,
    *,
    num_trials: int,
    require_scored_attempt: bool,
    expected_scored_attempts: int | None = None,
    allow_coverage_failure: bool = False,
) -> bool:
    """Validate the historical collector's complete pass-at-k structure."""
    if not isinstance(payload, dict):
        return False
    k = payload.get("k")
    threshold = payload.get("pass_threshold")
    stop_on_pass = payload.get("stop_on_pass")
    passed_cases = payload.get("passed_cases")
    failed_cases = payload.get("failed_cases")
    attempts_used = payload.get("attempts_used")
    total_cases = payload.get("total_cases")
    max_attempts = payload.get("max_attempts_possible")
    rate = payload.get("rate")
    average_attempts = payload.get("avg_attempts_used")
    if (
        not isinstance(k, int)
        or isinstance(k, bool)
        or k < 1
        or not _is_finite_number(threshold)
        or not 0 <= threshold <= 1
        or not isinstance(stop_on_pass, bool)
        or any(not _is_nonnegative_int(value) for value in (passed_cases, failed_cases, attempts_used, max_attempts))
        or not isinstance(total_cases, int)
        or isinstance(total_cases, bool)
        or total_cases < 1
        or passed_cases + failed_cases != total_cases
        or max_attempts != total_cases * k
        or (not allow_coverage_failure and attempts_used > max_attempts)
        or not _is_finite_number(rate)
        or not 0 <= rate <= 1
        or not math.isclose(rate, round(passed_cases / total_cases, 4), abs_tol=1e-9)
        or not _is_finite_number(average_attempts)
        or average_attempts < 0
        or not math.isclose(average_attempts, round(attempts_used / total_cases, 4), abs_tol=1e-9)
    ):
        return False
    extra_cases = payload.get("extra_cases")
    if (
        not isinstance(extra_cases, list)
        or any(not isinstance(case, str) or not case for case in extra_cases)
        or len(extra_cases) != len(set(extra_cases))
    ):
        return False
    extra_case_names = set(extra_cases)
    cases = payload.get("cases")
    if not isinstance(cases, dict) or not cases:
        return False
    total_attempt_rows = 0
    expected_attempt_rows = 0
    observed_passed_cases = 0
    observed_expected_cases = 0
    for case_name, case in cases.items():
        if not isinstance(case_name, str) or not isinstance(case, dict):
            return False
        case_attempts_used = case.get("attempts_used")
        attempts_skipped = case.get("attempts_skipped")
        attempts_missing = case.get("attempts_missing")
        attempts = case.get("attempts")
        if (
            not isinstance(case.get("passed"), bool)
            or not _is_nonnegative_int(case_attempts_used)
            or not _is_nonnegative_int(attempts_skipped)
            or not _is_nonnegative_int(attempts_missing)
            or not isinstance(attempts, list)
            or len(attempts) != case_attempts_used
        ):
            return False
        first_pass_attempt = case.get("first_pass_attempt")
        if first_pass_attempt is not None and (
            not isinstance(first_pass_attempt, int)
            or isinstance(first_pass_attempt, bool)
            or first_pass_attempt < 1
            or first_pass_attempt > len(attempts)
        ):
            return False
        for ordinal, attempt in enumerate(attempts, start=1):
            if (
                not isinstance(attempt, dict)
                or attempt.get("attempt") != ordinal
                or not isinstance(attempt.get("trial"), str)
                or not _is_finite_number(attempt.get("score"))
                or not isinstance(attempt.get("passed"), bool)
            ):
                return False
            score = attempt.get("score")
            if attempt["passed"] != (score >= threshold):
                return False
        passing_ordinals = [ordinal for ordinal, attempt in enumerate(attempts, start=1) if attempt["passed"]]
        expected_first_pass = passing_ordinals[0] if passing_ordinals else None
        attempt_scores = [attempt["score"] for attempt in attempts]
        expected_best_score = max(attempt_scores) if attempt_scores else None
        best_score = case.get("best_score")
        if (
            case["passed"] != bool(passing_ordinals)
            or first_pass_attempt != expected_first_pass
            or (
                (expected_best_score is None and best_score is not None)
                or (
                    expected_best_score is not None
                    and (
                        not _is_finite_number(best_score)
                        or not math.isclose(best_score, expected_best_score, abs_tol=1e-9)
                    )
                )
            )
        ):
            return False
        total_attempt_rows += len(attempts)
        if case_name not in extra_case_names:
            observed_expected_cases += 1
            observed_passed_cases += int(case["passed"])
            expected_attempt_rows += len(attempts)
            if not allow_coverage_failure and case_attempts_used + attempts_skipped + attempts_missing != k:
                return False
        elif case.get("extra_case") is not True:
            return False
    return (
        extra_case_names.issubset(cases)
        and observed_expected_cases == total_cases
        and observed_passed_cases == passed_cases
        and expected_attempt_rows == attempts_used
        and total_attempt_rows <= num_trials
        and (expected_scored_attempts is None or total_attempt_rows == expected_scored_attempts)
        and (not require_scored_attempt or total_attempt_rows > 0)
    )


def _legacy_dimensions_are_valid(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    for name, dimension in payload.items():
        if (
            not isinstance(name, str)
            or not isinstance(dimension, dict)
            or not _is_finite_number(dimension.get("score"))
        ):
            return False
        sources = dimension.get("sources")
        if not isinstance(sources, dict) or any(
            not isinstance(source, str) or not _is_finite_number(weight) for source, weight in sources.items()
        ):
            return False
    return True


def _legacy_summary_is_complete(payload: object, *, expected_agent: str) -> bool:
    """Return whether a pre-result summary satisfies an authentic era schema."""
    if not isinstance(payload, dict):
        return False
    if payload.get("agent") != expected_agent:
        return False
    if (
        not isinstance(payload.get("model"), str)
        or not payload["model"].strip()
        or not isinstance(payload.get("model_source"), str)
        or not payload["model_source"].strip()
    ):
        return False
    metric_set = payload.get("metric_set")
    metrics = payload.get("metrics")
    if not isinstance(metric_set, str) or not metric_set or not isinstance(metrics, list):
        return False
    if any(not isinstance(metric, str) or not metric for metric in metrics) or not _legacy_dimensions_are_valid(
        payload.get("dimensions")
    ):
        return False

    num_trials = payload.get("num_trials")
    if not _is_nonnegative_int(num_trials):
        return False

    completion_fields = {
        "execution_status",
        "execution_errors",
        "expected_attempts",
        "scored_attempts",
        "job_failure",
        "trial_failures",
    }
    present_completion_fields = completion_fields.intersection(payload)
    scored_attempts: int | None = None
    if present_completion_fields and present_completion_fields != completion_fields:
        return False
    if present_completion_fields:
        execution_status = payload.get("execution_status")
        execution_errors = payload.get("execution_errors")
        job_failure = payload.get("job_failure")
        trial_failures = payload.get("trial_failures")
        if (
            execution_status not in {"succeeded", "failed"}
            or not isinstance(execution_errors, list)
            or any(not isinstance(error, str) or not error for error in execution_errors)
            or not isinstance(job_failure, str)
            or not isinstance(trial_failures, list)
            or any(not isinstance(failure, dict) for failure in trial_failures)
        ):
            return False
        expected_attempts = payload.get("expected_attempts")
        scored_attempts = payload.get("scored_attempts")
        if not _is_nonnegative_int(expected_attempts) or not _is_nonnegative_int(scored_attempts):
            return False
        if execution_status == "succeeded" and (
            execution_errors
            or job_failure
            or trial_failures
            or expected_attempts < 1
            or scored_attempts != expected_attempts
        ):
            return False
        if execution_status == "failed" and not execution_errors:
            return False
    elif num_trials < 1:
        return False

    scores = payload.get("scores")
    custom_scores = payload.get("custom_scores")
    if not isinstance(scores, dict) or not isinstance(custom_scores, dict):
        return False
    for score_group in (scores, custom_scores):
        for value in score_group.values():
            if isinstance(value, bool) or not isinstance(value, int | float):
                return False
            try:
                if not math.isfinite(value):
                    return False
            except OverflowError:
                return False
    if scores and any(metric not in metrics for metric in scores):
        return False
    status = payload.get("execution_status")
    require_scored_attempt = status != "failed"
    if not _legacy_pass_at_k_is_complete(
        payload.get("pass_at_k"),
        num_trials=num_trials,
        require_scored_attempt=require_scored_attempt,
        expected_scored_attempts=scored_attempts,
        allow_coverage_failure=status == "failed",
    ):
        return False
    if status == "failed":
        return True
    return bool(scores or custom_scores) or (metric_set == "custom-only" and metrics == [])


def _legacy_completion_mtime(
    candidate: Path,
    secure_root: SecureRoot,
    snapshots: list[tuple[Path, bytes, os.stat_result]],
    expected_agents: tuple[str, ...],
) -> int | None:
    """Return completion time for an unmarked pre-result run, if safely readable."""
    marker = candidate / GENERATED_OUTPUT_MARKER
    if os.path.lexists(marker):
        # Current runs are marked before any evaluation artifact is written.
        # Any marker presence, including a corrupt marker, prevents an
        # incomplete current run from downgrading into the legacy contract.
        return None

    completion_mtimes: list[int] = []
    try:
        children = sorted(candidate.iterdir(), key=lambda path: path.name)
        agent_dirs: dict[str, Path] = {}
        for child in children:
            if child.name.startswith("_"):
                continue
            child_metadata = child.lstat()
            if _path_is_link_or_reparse(child, child_metadata):
                return None
            if not stat.S_ISDIR(child_metadata.st_mode):
                continue
            if child.name not in expected_agents:
                return None
            agent_dirs[child.name] = child

        if set(agent_dirs) != set(expected_agents):
            return None

        for agent_name in expected_agents:
            agent_dir = agent_dirs[agent_name]

            condition_dir = agent_dir / "with-skill"
            if os.path.lexists(condition_dir):
                condition_metadata = condition_dir.lstat()
                if _path_is_link_or_reparse(condition_dir, condition_metadata):
                    return None
                if not stat.S_ISDIR(condition_metadata.st_mode):
                    return None
            nested_summary = condition_dir / "summary.json"
            flat_summary = agent_dir / "summary.json"
            summary_paths = [path for path in (nested_summary, flat_summary) if os.path.lexists(path)]
            if len(summary_paths) != 1:
                return None

            baseline_dir = agent_dir / "without-skill"
            if os.path.lexists(baseline_dir):
                baseline_metadata = baseline_dir.lstat()
                if _path_is_link_or_reparse(baseline_dir, baseline_metadata) or not stat.S_ISDIR(
                    baseline_metadata.st_mode
                ):
                    return None
                baseline_summary = baseline_dir / "summary.json"
                if not os.path.lexists(baseline_summary):
                    return None
                summary_paths.append(baseline_summary)

            for summary_path in summary_paths:
                summary_metadata = summary_path.lstat()
                if not _is_single_link_regular(summary_path, summary_metadata):
                    return None
                observed = _read_stable_json(secure_root, summary_path, summary_metadata)
                if observed is None or not _legacy_summary_is_complete(observed[0], expected_agent=agent_name):
                    return None
                snapshots.append((summary_path, observed[2], observed[1]))
                completion_mtimes.append(observed[1].st_mtime_ns)
    except OSError:
        return None

    if os.path.lexists(marker):
        return None
    return max(completion_mtimes, default=None)


def run_directory_sort_key(
    candidate: Path,
    *,
    require_completed_result: bool = False,
) -> tuple[datetime, int, str] | None:
    """Return shared ordering metadata for one result directory.

    Timestamped runs are accepted when they carry either a valid final result
    matching the directory's run ID or the stricter pre-result legacy contract,
    and are ordered by parsed timestamp, completion mtime, then name. Compare
    also keeps accepting summary-only non-timestamp directories, placing them
    behind timestamped runs.
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
    if timestamp is None:
        if require_completed_result:
            return None
        return datetime.min, -1, candidate.name  # noqa: DTZ901 -- legacy names have no timezone

    try:
        snapshots: list[tuple[Path, bytes, os.stat_result]] = []
        with SecureRoot(candidate, expected=candidate_metadata) as secure_root:
            run_config_path = candidate / _RUN_COMPLETION_ARTIFACTS[0]
            run_config_metadata = run_config_path.lstat()
            observed_run_config = _read_stable_json(secure_root, run_config_path, run_config_metadata)
            if observed_run_config is None or not isinstance(observed_run_config[0], dict):
                return None
            snapshots.append((run_config_path, observed_run_config[2], observed_run_config[1]))

            result_path = candidate / _FINAL_RESULT_ARTIFACT
            try:
                result_metadata = result_path.lstat()
            except FileNotFoundError:
                expected_agents = _legacy_run_agents(observed_run_config[0])
                if expected_agents is None:
                    return None
                legacy_completion_mtime = _legacy_completion_mtime(
                    candidate,
                    secure_root,
                    snapshots,
                    expected_agents,
                )
                if legacy_completion_mtime is None:
                    return None
                completion_mtime = max(legacy_completion_mtime, observed_run_config[1].st_mtime_ns)
            else:
                observed_result = _read_stable_json(secure_root, result_path, result_metadata)
                if observed_result is None:
                    return None
                result = observed_result[0]
                if not isinstance(result, dict) or result.get("run_id") != candidate.name:
                    return None
                snapshots.append((result_path, observed_result[2], observed_result[1]))
                completion_mtime = observed_result[1].st_mtime_ns

            if not all(_artifact_snapshot_is_unchanged(secure_root, snapshot) for snapshot in snapshots):
                return None

        candidate_after = candidate.lstat()
        if _path_is_link_or_reparse(candidate, candidate_after) or _node_fingerprint(
            candidate_after
        ) != _node_fingerprint(candidate_metadata):
            return None
    except (OSError, SecurePathError):
        return None
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


def _is_completed_run_dir(path: Path) -> bool:
    """Return whether ``path`` is a *completed* timestamped run directory.

    A candidate must match the run-name timestamp pattern, carry its run
    configuration, and satisfy either the current final-result contract or the
    pre-result legacy-summary contract. This keeps the symlink-less fallback in
    :func:`resolve_latest_results` from resolving aborted/empty runs,
    ``_harbor-*`` staging dirs, ``astra-cleanup``, or the ``<plugin>-plugin-eval``
    wrapper directory (MR !29 review 59316310).
    """
    return run_directory_sort_key(path, require_completed_result=True) is not None


def is_legacy_completed_run_dir(path: Path) -> bool:
    """Return whether ``path`` satisfies the authenticated pre-result contract."""
    if _run_timestamp(path.name) is None:
        return False
    if os.path.lexists(path / _FINAL_RESULT_ARTIFACT) or os.path.lexists(path / GENERATED_OUTPUT_MARKER):
        return False
    return run_directory_sort_key(path, require_completed_result=True) is not None


def _newest_run_dir(root: Path) -> Path | None:
    """Return the newest *completed* timestamped run directory under ``root``.

    This is the cross-platform fallback for when the ``latest`` symlink was
    never created. Only completed runs are considered, with same-second runs
    ordered by their completion artifact mtime and then deterministic name.
    """
    return next(iter(ordered_run_directories(root, completed_only=True)), None)


def publish_latest_results(root: Path, run_id: str) -> bool:
    """Atomically replace ``latest`` with a relative symlink to ``run_id``."""
    temporary = root / f".latest-{os.getpid()}-{uuid4().hex}.tmp"
    try:
        temporary.symlink_to(run_id)
        os.replace(temporary, root / "latest")  # noqa: PTH105 -- atomic replacement is the contract
    except OSError:
        return False
    finally:
        with suppress(OSError):
            temporary.unlink()
    return True


def resolve_latest_results(
    skill_path: Path,
    cli_results_dir: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Return the best available ``latest`` results path for read workflows.

    Prefers each root's valid ``latest`` symlink. The link must resolve to an
    immediate, completed child of that root. When it is absent or invalid
    (including native Windows where symlink creation is not permitted), falls
    back to the newest completed timestamped run directory.
    """
    roots = iter_candidate_results_roots(
        skill_path,
        cli_results_dir,
        environ=environ,
    )
    first_absent_latest: Path | None = None
    for root in roots:
        candidate = root / "latest"
        if not os.path.lexists(candidate) and first_absent_latest is None:
            first_absent_latest = candidate
        if candidate.is_symlink():
            try:
                link_target = candidate.readlink()
                resolved_root = root.resolve(strict=True)
                target = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                pass
            else:
                if (
                    not link_target.is_absolute()
                    and len(link_target.parts) == 1
                    and link_target.name == target.name
                    and target.parent == resolved_root
                    and _is_completed_run_dir(target)
                ):
                    return candidate
        newest = _newest_run_dir(root)
        if newest is not None:
            return newest
    if first_absent_latest is not None:
        return first_absent_latest
    # Never hand an invalid existing ``latest`` entry back to callers: most
    # immediately test ``exists()`` or resolve symlinks and would otherwise
    # consume the very directory/link rejected above. A unique non-existent
    # path preserves the historical Path-returning API and its absence check.
    while True:
        unresolved = roots[0] / f".latest-unavailable-{uuid4().hex}"
        if not os.path.lexists(unresolved):
            return unresolved


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
