"""Tests for the HSTG (hierarchical spatio-temporal) belief backend."""

import unittest

from bayesian_agent import (
    BayesianSkillRegistry,
    LexicalSimilarity,
    SkillBelief,
    TrajectoryEvidence,
    normalize_algorithm,
    SUPPORTED_ALGORITHMS,
)
from bayesian_agent.benchmarks.evolution import build_benchmark_skill_context
from bayesian_agent.core.algorithms.hstg import HSTGState


def _evidence(task_id, outcome, task_text, failure_mode="", context="sop_bench"):
    return TrajectoryEvidence(
        task_id=task_id,
        skill_id="benchmark/sop_bench",
        context=context,
        outcome=outcome,
        failure_mode=failure_mode,
        task_text=task_text,
        total_tokens=5000,
        turns=3,
        elapsed_seconds=30.0,
    )


class LexicalSimilarityTest(unittest.TestCase):
    def test_identical_texts_score_one(self):
        provider = LexicalSimilarity()
        self.assertAlmostEqual(provider.similarity("verify csv row output", "verify csv row output"), 1.0)

    def test_disjoint_texts_score_zero(self):
        provider = LexicalSimilarity()
        self.assertEqual(provider.similarity("csv row output", "sql insert statement"), 0.0)

    def test_partial_overlap_is_between_zero_and_one(self):
        provider = LexicalSimilarity()
        score = provider.similarity("verify csv row output cell", "verify csv column header cell")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_text_scores_zero(self):
        provider = LexicalSimilarity()
        self.assertEqual(provider.similarity("", "anything"), 0.0)


class HSTGStateTest(unittest.TestCase):
    def test_cold_start_reduces_to_global_prior(self):
        state = HSTGState()
        # No evidence at all: global Laplace prior is 0.5 and there is no
        # kernel mass, so the dynamic prior must fall back exactly to it.
        prior = state.dynamic_prior("a brand new task about csv rows")
        self.assertEqual(prior["w_local"], 0.0)
        self.assertAlmostEqual(prior["pi_success"], prior["pi_global"])
        self.assertAlmostEqual(state.predict_success(task_text="a brand new task about csv rows"), 0.5)

    def test_dissimilar_history_falls_back_to_global_prior(self):
        state = HSTGState()
        for idx in range(3):
            state.update(_evidence(f"t{idx}", "success", "sql insert statement into orders table"))
        prior = state.dynamic_prior("完全不同的中文任务描述")
        self.assertEqual(prior["kernel_mass"], 0.0)
        self.assertAlmostEqual(prior["pi_success"], prior["pi_global"])

    def test_kernel_weighting_shifts_prediction_toward_local_evidence(self):
        state = HSTGState()
        failure_text = "verify csv row expected_output cell is non-empty before finishing"
        success_text = "write one sql statement to answer.sql using task.json columns"
        for idx in range(3):
            state.update(_evidence(f"f{idx}", "failure", failure_text, failure_mode="left_expected_output_blank"))
            state.update(_evidence(f"s{idx}", "success", success_text))
        near_failures = state.predict_success(task_text=failure_text)
        near_successes = state.predict_success(task_text=success_text)
        self.assertLess(near_failures, near_successes)
        self.assertLess(near_failures, 0.5)
        self.assertGreater(near_successes, 0.5)

    def test_w_local_grows_with_kernel_mass(self):
        state = HSTGState()
        text = "verify csv row expected_output cell"
        w_values = []
        for idx in range(4):
            state.update(_evidence(f"t{idx}", "success", text))
            w_values.append(state.dynamic_prior(text)["w_local"])
        self.assertEqual(w_values, sorted(w_values))
        self.assertGreater(w_values[-1], w_values[0])

    def test_weighted_failure_support_without_task_text_equals_counts(self):
        state = HSTGState()
        for idx in range(2):
            state.update(_evidence(f"f{idx}", "failure", f"task {idx}", failure_mode="left_expected_output_blank"))
        support = state.weighted_failure_support("")
        self.assertEqual(support, {"left_expected_output_blank": 2.0})

    def test_weighted_failure_support_discounts_distant_failures(self):
        state = HSTGState()
        state.update(_evidence("f0", "failure", "sql insert into orders", failure_mode="invented_unrequested_column"))
        support = state.weighted_failure_support("verify csv row output cell")
        self.assertEqual(support.get("invented_unrequested_column", 0.0), 0.0)


class HSTGBeliefIntegrationTest(unittest.TestCase):
    def test_algorithm_is_registered(self):
        self.assertIn("hstg", SUPPORTED_ALGORITHMS)
        self.assertEqual(normalize_algorithm("hstg"), "hstg")

    def test_belief_serialization_roundtrip_preserves_predictions(self):
        belief = SkillBelief(skill_id="benchmark/sop_bench", algorithm="hstg")
        text = "verify csv row expected_output cell"
        for idx in range(2):
            belief.update(_evidence(f"f{idx}", "failure", text, failure_mode="left_expected_output_blank"))
        restored = SkillBelief.from_dict("benchmark/sop_bench", belief.to_dict(), algorithm="hstg")
        self.assertEqual(restored.algorithm, "hstg")
        self.assertEqual(len(restored.hstg.events), 2)
        self.assertAlmostEqual(
            restored.predict_success_probability(task_text=text),
            belief.predict_success_probability(task_text=text),
        )

    def test_registry_records_and_ranks_with_task_text(self):
        registry = BayesianSkillRegistry.in_memory(algorithm="hstg")
        registry.record(_evidence("t0", "success", "verify csv row output"))
        top = registry.top(context="sop_bench", task_text="verify csv row output")
        self.assertEqual(top[0].skill_id, "benchmark/sop_bench")


class KernelPatchGatingTest(unittest.TestCase):
    FAILURE_TEXT = "sop_bench row 3; order_id=A3; product_id=P9; quantity_requested=5"

    def _registry_with_failures(self, count):
        registry = BayesianSkillRegistry.in_memory(algorithm="hstg")
        for idx in range(count):
            registry.record(
                _evidence(
                    f"sop_{idx:02d}",
                    "failure",
                    self.FAILURE_TEXT,
                    failure_mode="left_expected_output_blank",
                )
            )
        return registry

    def test_single_identical_failure_activates_patch_for_similar_task(self):
        # Cold-start acceleration: one semantically near-identical failure
        # reaches the weighted threshold without waiting for a second one.
        registry = self._registry_with_failures(1)
        context = build_benchmark_skill_context("sop_bench", registry, task_text=self.FAILURE_TEXT)
        self.assertIn("left_expected_output_blank", context)

    def test_repeated_failures_activate_patch_for_related_task(self):
        # Hybrid clause: two same-mode failures on related (but not
        # identical) tasks keep count-gating behavior as long as the
        # current task retains minimal semantic relevance.
        registry = self._registry_with_failures(2)
        context = build_benchmark_skill_context(
            "sop_bench", registry, task_text="sop_bench row 7; order_id=B1; quantity_requested=2"
        )
        self.assertIn("left_expected_output_blank", context)

    def test_repeated_failures_do_not_activate_patch_for_distant_task(self):
        registry = self._registry_with_failures(3)
        context = build_benchmark_skill_context(
            "sop_bench", registry, task_text="完全无关的市场数据指标计算任务"
        )
        self.assertNotIn("left_expected_output_blank", context)

    def test_count_gating_still_applies_without_task_text(self):
        registry = self._registry_with_failures(2)
        context = build_benchmark_skill_context("sop_bench", registry)
        self.assertIn("left_expected_output_blank", context)

    def test_default_backend_keeps_count_gating(self):
        registry = BayesianSkillRegistry.in_memory()
        for idx in range(2):
            registry.record(
                _evidence(
                    f"sop_{idx:02d}",
                    "failure",
                    self.FAILURE_TEXT,
                    failure_mode="left_expected_output_blank",
                )
            )
        context = build_benchmark_skill_context("sop_bench", registry, task_text=self.FAILURE_TEXT)
        self.assertIn("left_expected_output_blank", context)


if __name__ == "__main__":
    unittest.main()
