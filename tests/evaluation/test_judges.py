# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Tier 3 LLM-as-judge modules (deterministic paths).

The dimension judge and insights judge fall back to deterministic / empty
output when no LLM is available, so these tests exercise that path without any
network access.
"""

from __future__ import annotations

import json

import pytest

from skillevaluator.constants import DIMENSION_MAPPING
from skillevaluator.evaluation import (
    InsightsJudge,
    build_insights,
    compute_dimensions,
    compute_dimensions_deterministic,
)


@pytest.fixture
def evaluators() -> dict:
    return {
        "security": {"with_skill": 0.9, "baseline": 0.8},
        "skill_execution": {"with_skill": 0.7, "baseline": 0.5},
        "skill_efficiency": {"with_skill": 0.6, "baseline": 0.6},
        "accuracy": {"with_skill": 0.8, "baseline": 0.7},
        "goal_accuracy": {"with_skill": 0.75, "baseline": 0.6},
        "behavior_check": {"with_skill": 0.85, "baseline": 0.8},
        "token_efficiency": {"with_skill": 0.5, "baseline": 0.4},
    }


class TestDimensionJudgeDeterministic:
    def test_produces_all_five_dimensions(self, evaluators: dict) -> None:
        dims = compute_dimensions_deterministic(evaluators)
        assert {d["id"] for d in dims} == set(DIMENSION_MAPPING)

    def test_scores_and_lift_present(self, evaluators: dict) -> None:
        dims = {d["id"]: d for d in compute_dimensions_deterministic(evaluators)}
        sec = dims["security"]
        assert sec["score"] == pytest.approx(0.9)
        assert sec["with_skill"] == pytest.approx(0.9)
        assert sec["baseline"] == pytest.approx(0.8)
        assert sec["lift"] == pytest.approx(0.1)
        assert sec["verdict"] == "PASS"
        assert sec["reasoning_bullets"]
        assert sec["explanation"]

    def test_verdict_thresholds(self, evaluators: dict) -> None:
        dims = {d["id"]: d for d in compute_dimensions_deterministic(evaluators)}
        # efficiency = 0.7*0.6 + 0.3*0.5 = 0.57 -> below 0.7 pass threshold
        assert dims["efficiency"]["verdict"] == "NEUTRAL"

    def test_baseline_absent_yields_none_lift(self) -> None:
        evaluators = {"security": {"with_skill": 0.9}}
        dims = {d["id"]: d for d in compute_dimensions_deterministic(evaluators)}
        assert dims["security"]["baseline"] is None
        assert dims["security"]["lift"] is None

    def test_compute_dimensions_no_llm_uses_deterministic(self, evaluators: dict) -> None:
        dims = compute_dimensions(evaluators, [], 0.1, use_llm=False)
        assert {d["id"] for d in dims} == set(DIMENSION_MAPPING)


class TestInsightsJudgeFallback:
    def test_build_insights_no_llm_returns_empty(self, evaluators: dict) -> None:
        deterministic = {"conclusions": [], "recommendations": []}
        canonical = {"dimensions": compute_dimensions_deterministic(evaluators)}
        out = build_insights(canonical, deterministic, use_llm=False)
        assert out == {"conclusions": [], "recommendations": []}


class TestInsightsJudgeNegativeControls:
    def test_prompt_marks_expected_skill_null_as_negative_control(self) -> None:
        prompt = InsightsJudge().create_user_prompt(
            canonical={
                "dataset": [
                    {
                        "id": "negative-001",
                        "question": "Convert 42 degrees Fahrenheit to Celsius.",
                        "ground_truth": "5.6",
                        "expected_skill": None,
                        "expected_behavior": [
                            "The agent did not read or invoke the evaluated skill",
                        ],
                    }
                ]
            },
            deterministic={},
        )
        context = json.loads(prompt.split("\n\n", 1)[1])

        case = context["dataset"][0]
        assert case["case_type"] == "negative_control"
        assert case["skill_expected_to_activate"] is False
        assert case["expected_skill"] is None
        assert case["expected_behavior"] == [
            "The agent did not read or invoke the evaluated skill",
        ]

    @pytest.mark.parametrize(
        ("case", "expected_type", "expected_activation"),
        [
            ({"id": "positive-001", "expected_skill": "calculator"}, "skill_activation", True),
            ({"id": "unlabeled-001"}, "unlabeled", None),
        ],
    )
    def test_prompt_preserves_positive_and_unlabeled_case_semantics(
        self,
        case: dict,
        expected_type: str,
        expected_activation: bool | None,
    ) -> None:
        prompt = InsightsJudge().create_user_prompt(
            canonical={"dataset": [case]},
            deterministic={},
        )
        context = json.loads(prompt.split("\n\n", 1)[1])

        assert context["dataset"][0]["case_type"] == expected_type
        assert context["dataset"][0]["skill_expected_to_activate"] is expected_activation

    def test_prompt_bounds_negative_control_assertions(self) -> None:
        prompt = InsightsJudge().create_user_prompt(
            canonical={
                "dataset": [
                    {
                        "id": "negative-001",
                        "expected_skill": None,
                        "expected_behavior": ["x" * 500] * 10,
                    }
                ]
            },
            deterministic={},
        )
        context = json.loads(prompt.split("\n\n", 1)[1])

        assertions = context["dataset"][0]["expected_behavior"]
        assert len(assertions) == 5
        assert all(len(assertion) == 300 for assertion in assertions)

    def test_system_prompt_requires_execution_evidence_for_negative_control_warning(self) -> None:
        prompt = InsightsJudge().get_system_prompt()

        assert "expected_skill is null" in prompt
        assert "successful negative-control" in prompt
        assert "explicit execution evidence" in prompt
        assert "invocation evidence is absent or ambiguous" in prompt
        assert "Do not recommend removing an unrelated negative-control case" in prompt
