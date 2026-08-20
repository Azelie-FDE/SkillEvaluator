# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-as-Judge for richer Tier 3 Insights (conclusions + recommendations).

The Insights judge runs *after* the deterministic dimension judge has produced
per-dimension scores, verdicts, and explanations. It receives the canonical
Tier 3 payload and returns up to N additional, contextual conclusions and
actionable recommendations that go beyond the deterministic baselines (best
performing agent, weakest dimension, coverage expansion).

If the LLM is unavailable the judge falls back to an empty list, so the
deterministic conclusions and recommendations from
:mod:`skillevaluator.evaluation.tier3_normalizer` continue to render unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any

from skillevaluator.constants import (
    AGENT_EVAL_EVALUATORS,
    INSIGHTS_JUDGE_MAX_CONCLUSIONS,
    INSIGHTS_JUDGE_MAX_RECOMMENDATIONS,
    INSIGHTS_JUDGE_MAX_TOKENS,
    INSIGHTS_JUDGE_MODEL,
    INSIGHTS_JUDGE_TEMPERATURE,
)
from skillevaluator.inference.client import LLMClient
from skillevaluator.logging_config import get_logger

logger = get_logger(__name__)


_VALID_CONCLUSION_SEVERITIES = {"pass", "warn", "fail"}
_VALID_RECOMMENDATION_CATEGORIES = {
    "Update",
    "Add",
    "Implement",
    "Document",
    "Fix",
    "Test",
    "Improve",
    "Action",
}
_NEGATIVE_CONTROL_CLAIM_PHRASES = (
    "unintended skill application",
    "unexpected skill application",
    "unintended skill activation",
    "unexpected skill activation",
    "scope misalignment",
    "overfitting",
    "overfit",
)
_IRRELEVANT_CASE_REMOVAL_RE = re.compile(
    r"\b(?:exclude|remove|drop|delete|eliminate)\b.{0,80}"
    r"\b(?:irrelevant|unrelated|out[- ]of[- ]scope)\b.{0,80}"
    r"\b(?:case|cases|test|tests|query|queries|prompt|prompts|dataset)\b",
    re.IGNORECASE,
)


_SYSTEM_PROMPT = """\
You are an expert Skill Quality Reviewer summarising a Tier 3 (live agent)
evaluation of an Agent Skill.

You will receive a JSON payload describing:
- Skill identity (name, agents run, best performing agent, runtime)
- Five SkillEvaluator dimensions (Security, Correctness, Discoverability,
  Effectiveness, Efficiency) with score, baseline, lift, verdict, and a
  human explanation.
- Per-evaluator scores (security, skill_execution, skill_efficiency, accuracy,
  goal_accuracy, behavior_check) for the best performing agent.
- A short selection of trial outcomes, error_recovery summaries, and
  baseline pairings.
- A handful of dataset cases the agent ran against.

The runner already provides three deterministic baseline observations
(\"best performing agent\", \"weakest dimension\", and an optional
\"coverage expansion\" suggestion). DO NOT repeat those exact observations.

NEGATIVE-CONTROL INTERPRETATION:
- A dataset case whose expected_skill is null is an intentional negative
  control: the agent should complete the unrelated task without activating
  the evaluated skill.
- An explicit should_trigger value takes precedence over expected_skill, just
  as it does in the runtime grader. Otherwise, a falsey expected_skill marks a
  negative control; a case with neither routing field is unlabeled.
- A successful negative-control case is correct routing behavior. Do not infer
  unintended skill application, scope misalignment, or overfitting merely
  because the agent answered the unrelated task correctly.
- Report unintended skill application only when explicit execution evidence
  shows that the evaluated skill was read or invoked, or routing evidence for
  that case failed. A passing routing result is evidence against that warning;
  when invocation evidence is absent or ambiguous, do not issue the warning.
- Do not recommend removing an unrelated negative-control case merely because
  it is outside the skill's scope; that is what the case is designed to test.
- Any conclusion or recommendation alleging unintended skill use, scope
  misalignment, overfitting, or removal of an irrelevant case MUST include the
  affected case ids in evidence_case_ids. Such a claim is valid only when the
  supplied context shows failed routing or explicit invocation evidence for at
  least one of those negative-control cases.

Your job is to produce ADDITIONAL, contextual insights:

CONCLUSIONS (up to {max_conclusions}): observations the reviewer should know
- what worked, what regressed, where evidence points to systemic issues. Each
should be 1-2 sentences and cite a concrete signal (a metric, a case id,
an error message, or a comparison).

RECOMMENDATIONS (up to {max_recommendations}): concrete, actionable next
steps for the skill author. Each must be one short imperative sentence.
Tag each with a category from this set: {categories}.

SEVERITY: tag every conclusion and recommendation with one of:
- \"pass\"   - positive observation / low-effort recommendation
- \"warn\"   - needs attention / medium-effort recommendation
- \"fail\"   - regression or blocking issue / high-priority fix

IMPORTANT: Respond with ONLY a JSON object, no markdown, no preamble. Use
this exact schema:

{{
  \"conclusions\": [
    {{\"title\": \"<short title>\", \"message\": \"<1-2 sentences>\", \"severity\": \"pass|warn|fail\", \"evidence_case_ids\": [\"<case id>\"]}}
  ],
  \"recommendations\": [
    {{\"category\": \"<one of the allowed categories>\", \"title\": \"<short title>\", \"message\": \"<one imperative sentence>\", \"severity\": \"pass|warn|fail\", \"evidence_case_ids\": [\"<case id>\"]}}
  ]
}}
""".format(
    max_conclusions=INSIGHTS_JUDGE_MAX_CONCLUSIONS,
    max_recommendations=INSIGHTS_JUDGE_MAX_RECOMMENDATIONS,
    categories=", ".join(sorted(_VALID_RECOMMENDATION_CATEGORIES)),
)


def _summarize_dimensions(dimensions: list[dict]) -> list[dict]:
    """Project SkillEvaluator dimensions into a compact prompt-friendly shape."""
    summary: list[dict] = []
    for dim in dimensions or []:
        summary.append(
            {
                "id": dim.get("id"),
                "score": dim.get("score", dim.get("with_skill", 0.0)),
                "baseline": dim.get("baseline"),
                "lift": dim.get("lift"),
                "verdict": dim.get("verdict"),
                "explanation": (dim.get("explanation") or "")[:600],
                "evaluators": dim.get("evaluators") or [],
            }
        )
    return summary


def _summarize_evaluators(evaluators: dict) -> dict:
    """Drop any unknown evaluator fields and keep core scores."""
    out: dict = {}
    for name in AGENT_EVAL_EVALUATORS:
        entry = evaluators.get(name) if isinstance(evaluators, dict) else None
        if isinstance(entry, dict):
            out[name] = {
                "with_skill": entry.get("with_skill"),
                "baseline": entry.get("baseline"),
                "lift": entry.get("lift"),
            }
    return out


def _case_routing(case: dict) -> tuple[str, bool | None]:
    """Mirror the runtime grader's routing precedence for prompt metadata."""
    if "should_trigger" in case:
        should_trigger = bool(case.get("should_trigger"))
    elif "expected_skill" in case:
        should_trigger = bool(case.get("expected_skill"))
    else:
        return "unlabeled", None
    return ("skill_activation", True) if should_trigger else ("negative_control", False)


def _summarize_case(case: dict) -> dict:
    case_type, skill_expected_to_activate = _case_routing(case)
    expected_behavior = case.get("expected_behavior")
    if isinstance(expected_behavior, list):
        expected_behavior = [str(item)[:300] for item in expected_behavior[:5]]
    elif isinstance(expected_behavior, str):
        expected_behavior = expected_behavior[:300]
    else:
        expected_behavior = None

    return {
        "id": case.get("id"),
        "case_type": case_type,
        "skill_expected_to_activate": skill_expected_to_activate,
        "question": (case.get("question") or "")[:300],
        "ground_truth": (case.get("ground_truth") or "")[:300],
        "expected_skill": case.get("expected_skill"),
        "should_trigger": case.get("should_trigger") if "should_trigger" in case else None,
        "expected_script": case.get("expected_script"),
        "expected_behavior": expected_behavior,
    }


def _case_lookup(dataset: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for case in dataset or []:
        if not isinstance(case, dict) or case.get("id") is None:
            continue
        lookup.setdefault(str(case["id"]), case)
    return lookup


def _summarize_trials(trials: list[dict], cases_by_id: dict[str, dict] | None = None) -> list[dict]:
    """Pick a representative slice without collapsing agents or attempts."""
    if not trials:
        return []
    cases_by_id = cases_by_id or {}

    def _key(indexed_trial: tuple[int, dict]) -> tuple[float, int]:
        index, trial = indexed_trial
        overall = trial.get("overall")
        score = overall if isinstance(overall, (int, float)) else 0.5
        return score, index

    indexed_trials = [(index, trial) for index, trial in enumerate(trials) if isinstance(trial, dict)]
    sorted_trials = sorted(indexed_trials, key=_key)
    chosen = sorted_trials[:3] + sorted_trials[-3:]

    attempt_counts: dict[tuple[str, str], int] = {}
    attempts_by_index: dict[int, int | str] = {}
    for index, trial in indexed_trials:
        identity = (str(trial.get("agent") or ""), str(trial.get("entry_id") or ""))
        attempt_counts[identity] = attempt_counts.get(identity, 0) + 1
        attempts_by_index[index] = trial.get("attempt") or attempt_counts[identity]

    seen_indices: set[int] = set()
    compact: list[dict] = []
    for index, trial in chosen:
        if index in seen_indices:
            continue
        seen_indices.add(index)
        scores = trial.get("scores") or {}
        entry_id = trial.get("entry_id")
        case = cases_by_id.get(str(entry_id)) if entry_id is not None else None
        compact.append(
            {
                "agent": trial.get("agent"),
                "trial_id": trial.get("trial_id"),
                "entry_id": entry_id,
                "attempt": attempts_by_index[index],
                "overall": trial.get("overall"),
                "scores": {k: v for k, v in scores.items() if v is not None},
                "baseline_overall": trial.get("baseline_overall"),
                "lift_scores": trial.get("lift_scores") or {},
                "warnings": (trial.get("warnings") or [])[:3],
                "error_recovery": trial.get("error_recovery") or {},
                "case": _summarize_case(case) if case is not None else None,
            }
        )
        if len(compact) >= 6:
            break
    return compact


def _summarize_dataset(dataset: list[dict]) -> list[dict]:
    """Trim dataset cases to the question + ground truth (plus id)."""
    return [_summarize_case(case) for case in (dataset or [])[:5] if isinstance(case, dict)]


def _dataset_for_sampled_trials(trials: list[dict], dataset: list[dict]) -> list[dict]:
    """Keep one copy of every case joined to a sampled trial, in trial order."""
    summary: list[dict] = []
    seen: set[str] = set()
    for trial in trials:
        case = trial.get("case")
        if not isinstance(case, dict) or case.get("id") is None:
            continue
        case_id = str(case["id"])
        if case_id in seen:
            continue
        seen.add(case_id)
        summary.append(case)
    return summary or _summarize_dataset(dataset)


def _coerce_severity(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _VALID_CONCLUSION_SEVERITIES:
            return v
    return "warn"


def _coerce_category(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().capitalize()
        if v in _VALID_RECOMMENDATION_CATEGORIES:
            return v
    return "Action"


def _requires_negative_control_evidence(item: dict) -> bool:
    text = f"{item.get('title') or ''} {item.get('message') or ''}"[:2_000]
    lowered = text.casefold()
    return any(phrase in lowered for phrase in _NEGATIVE_CONTROL_CLAIM_PHRASES) or bool(
        _IRRELEVANT_CASE_REMOVAL_RE.search(text)
    )


def _negative_control_grounding(canonical: dict) -> tuple[dict[str, bool], set[str]]:
    """Return negative cases mapped to whether invocation/routing failure exists."""
    known_case_ids: set[str] = set()
    evidence: dict[str, bool] = {}
    for case in canonical.get("dataset") or []:
        if not isinstance(case, dict) or case.get("id") is None:
            continue
        case_id = str(case["id"])
        known_case_ids.add(case_id)
        _case_type, should_trigger = _case_routing(case)
        if should_trigger is False:
            evidence[case_id] = False

    for trial in canonical.get("trials") or []:
        if not isinstance(trial, dict) or trial.get("entry_id") is None:
            continue
        case_id = str(trial["entry_id"])
        if case_id not in evidence:
            continue
        scores = trial.get("scores") if isinstance(trial.get("scores"), dict) else {}
        routing_scores = [
            scores.get("skill_execution"),
            scores.get("skill_routing"),
            trial.get("skill_execution"),
            trial.get("skill_routing"),
        ]
        failed_routing = any(
            isinstance(score, int | float) and not isinstance(score, bool) and score < 1.0
            for score in routing_scores
        )
        explicit_invocation = trial.get("skill_invoked") is True or trial.get("routing_passed") is False
        if failed_routing or explicit_invocation:
            evidence[case_id] = True
    return evidence, known_case_ids


def _referenced_case_ids(item: dict, known_case_ids: set[str]) -> set[str]:
    references: set[str] = set()
    raw_references = item.get("evidence_case_ids")
    if isinstance(raw_references, list):
        references.update(str(value) for value in raw_references[:10] if value is not None)

    text = f"{item.get('title') or ''} {item.get('message') or ''}"[:2_000]
    for case_id in known_case_ids:
        if re.search(rf"(?<![\w-]){re.escape(case_id)}(?![\w-])", text, re.IGNORECASE):
            references.add(case_id)
    return references


def _unsupported_negative_control_claim(
    item: dict,
    grounding: tuple[dict[str, bool], set[str]],
) -> bool:
    evidence, known_case_ids = grounding
    if not evidence or not _requires_negative_control_evidence(item):
        return False

    references = _referenced_case_ids(item, known_case_ids)
    if not references or any(case_id not in known_case_ids for case_id in references):
        return True

    negative_references = references.intersection(evidence)
    if not negative_references:
        return False
    return not any(evidence[case_id] for case_id in negative_references)


class InsightsJudge(LLMClient):
    """LLM judge that produces additional conclusions and recommendations."""

    default_model: str = INSIGHTS_JUDGE_MODEL
    default_max_tokens: int | None = INSIGHTS_JUDGE_MAX_TOKENS
    default_temperature: float | None = INSIGHTS_JUDGE_TEMPERATURE

    def get_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def create_user_prompt(self, **kwargs: Any) -> str:
        canonical: dict = kwargs.get("canonical") or {}
        deterministic: dict = kwargs.get("deterministic") or {}
        dataset = canonical.get("dataset") or []
        trials = _summarize_trials(canonical.get("trials") or [], _case_lookup(dataset))

        prompt: dict = {
            "skill": {
                "name": canonical.get("skill_name"),
                "best_agent": canonical.get("best_agent"),
                "agents_run": canonical.get("agents_run") or [],
                "verdict": canonical.get("verdict"),
                "overall_score": canonical.get("overall_score"),
                "overall_lift": canonical.get("overall_lift"),
                "runtime_seconds": canonical.get("runtime_seconds"),
            },
            "dimensions": _summarize_dimensions(canonical.get("dimensions") or []),
            "best_agent_evaluators": _summarize_evaluators(canonical.get("evaluators") or {}),
            "trials": trials,
            "dataset": _dataset_for_sampled_trials(trials, dataset),
            "deterministic_observations": {
                "conclusions": [
                    {
                        "title": item.get("title"),
                        "message": item.get("message"),
                        "severity": item.get("severity"),
                    }
                    for item in (deterministic.get("conclusions") or [])
                ],
                "recommendations": list(deterministic.get("suggestions") or []),
            },
        }

        return (
            "Tier 3 evaluation context (already includes deterministic "
            "observations — produce ADDITIONAL insights, do not repeat):\n\n"
            + json.dumps(prompt, indent=2, default=str)
        )

    def parse_response(self, response_text: str, **kwargs: Any) -> dict:
        content = response_text.strip().lstrip("\ufeff")
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0].strip()
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")
        canonical = kwargs.get("canonical") if isinstance(kwargs.get("canonical"), dict) else {}
        grounding = _negative_control_grounding(canonical)

        conclusions: list[dict] = []
        for item in (data.get("conclusions") or [])[:INSIGHTS_JUDGE_MAX_CONCLUSIONS]:
            if not isinstance(item, dict):
                continue
            if _unsupported_negative_control_claim(item, grounding):
                continue
            title = (item.get("title") or "").strip()
            message = (item.get("message") or "").strip()
            if not title and not message:
                continue
            conclusions.append(
                {
                    "title": title or "Insight",
                    "message": message or title,
                    "severity": _coerce_severity(item.get("severity")),
                    "source": "llm",
                }
            )

        recommendations: list[dict] = []
        for item in (data.get("recommendations") or [])[:INSIGHTS_JUDGE_MAX_RECOMMENDATIONS]:
            if not isinstance(item, dict):
                continue
            if _unsupported_negative_control_claim(item, grounding):
                continue
            title = (item.get("title") or "").strip()
            message = (item.get("message") or "").strip()
            if not title and not message:
                continue
            recommendations.append(
                {
                    "title": title or message[:60],
                    "message": message or title,
                    "category": _coerce_category(item.get("category")),
                    "severity": _coerce_severity(item.get("severity")),
                    "source": "llm",
                }
            )

        return {"conclusions": conclusions, "recommendations": recommendations}

    def get_fallback_response(self, **_kwargs: Any) -> dict:
        return {"conclusions": [], "recommendations": []}


def build_insights(
    canonical: dict,
    deterministic: dict,
    *,
    use_llm: bool = True,
) -> dict:
    """Return ``{"conclusions": [...], "recommendations": [...]}`` from the LLM.

    The function never raises: when the LLM is unavailable it simply returns
    empty lists so callers can keep using their deterministic baselines.
    """
    if not use_llm:
        return {"conclusions": [], "recommendations": []}

    judge = InsightsJudge()
    return judge.process(canonical=canonical, deterministic=deterministic)


__all__ = [
    "InsightsJudge",
    "build_insights",
]
