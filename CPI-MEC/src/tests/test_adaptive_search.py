"""Pure smoke tests for the frozen adaptive-search rules.

Test rules that need no GPU or real data.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CORE_DIR = Path(__file__).resolve().parents[1] / "cpi_mec"
sys.path.insert(0, str(CORE_DIR))

from adaptive_oof import annotate_full_domain_ranks
from adaptive_search_common import (
    Candidate,
    add_skill_scores,
    backbone_groups,
    candidate_summary,
    evaluate_candidates,
    grid_candidates,
    local_bounds,
    model_batches,
    select_exact_best,
)


class AdaptiveSearchRulesTest(unittest.TestCase):
    """Verify frozen search rules."""

    def test_complete_grid_and_ablation_count(self) -> None:
        """Verify 180, 125, and 55."""
        anchors = (0.1, 0.25, 0.5, 1.0, 2.0)
        weights = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
        candidates = grid_candidates(anchors, weights, weights)
        self.assertEqual(len(candidates), 180)
        self.assertEqual(sum(candidate.eligible for candidate in candidates), 125)
        self.assertEqual(sum(not candidate.eligible for candidate in candidates), 55)

    def test_dynamic_local_bounds(self) -> None:
        """Verify local geometric bounds."""
        anchors = (0.1, 0.25, 0.5, 1.0, 2.0)
        weights = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
        bounds = local_bounds(
            Candidate(0.1, 1.0, 0.25),
            anchors,
            weights,
            weights,
        ).set_index("parameter")
        self.assertAlmostEqual(bounds.at["anchor_weight", "local_low"], 0.04)
        self.assertEqual(bounds.at["anchor_weight", "local_high"], 0.25)
        self.assertEqual(bounds.at["graphban_weight", "local_low"], 0.5)
        self.assertEqual(bounds.at["graphban_weight", "local_high"], 2.0)
        self.assertEqual(bounds.at["boltz_weight", "local_low"], 0.125)
        self.assertEqual(bounds.at["boltz_weight", "local_high"], 0.5)

        upper = local_bounds(
            Candidate(2.0, 4.0, 0.5),
            anchors,
            weights,
            weights,
        ).set_index("parameter")
        self.assertEqual(upper.at["anchor_weight", "local_high"], 4.0)
        self.assertEqual(upper.at["graphban_weight", "local_high"], 8.0)

    def test_ablation_never_wins(self) -> None:
        """Exclude zero-weight winners."""
        seed_metrics = pd.DataFrame(
            [
                {
                    "candidate_id": "formal",
                    "anchor_weight": 0.1,
                    "graphban_weight": 1.0,
                    "boltz_weight": 0.25,
                    "seed_offset": 0,
                    "selection_score_j": 0.2,
                    "global_skill": 0.2,
                    "within_skill": 0.2,
                },
                {
                    "candidate_id": "ablation",
                    "anchor_weight": 0.1,
                    "graphban_weight": 0.0,
                    "boltz_weight": 0.25,
                    "seed_offset": 0,
                    "selection_score_j": 0.9,
                    "global_skill": 0.9,
                    "within_skill": 0.9,
                },
            ]
        )
        summary = candidate_summary(seed_metrics)
        summary["eligible_for_selection"] = summary["graphban_weight"] > 0
        selected, _ = select_exact_best(summary)
        self.assertEqual(selected, Candidate(0.1, 1.0, 0.25))

    def test_equal_global_within_objective(self) -> None:
        """Verify the formal J definition."""
        scored = add_skill_scores(
            {
                "positive_prevalence": 0.4,
                "ranking_pooled_auprc": 0.7,
                "pairwise_logistic_loss": 0.5 * np.log(2.0),
            }
        )
        self.assertAlmostEqual(scored["global_skill"], 0.5)
        self.assertAlmostEqual(scored["within_skill"], 0.5)
        self.assertAlmostEqual(scored["selection_score_j"], 0.5)

    def test_full_domain_oof_ranks(self) -> None:
        """Verify full-domain ranks."""
        values = np.asarray(
            [[1.0, 3.0, 2.0], [6.0, 4.0, 5.0]],
            dtype=np.float32,
        )
        predictions = pd.DataFrame(
            {
                "entity_a": ["A", "B"],
                "entity_b": ["x", "z"],
                "label": [1, 1],
            }
        )
        ranked = annotate_full_domain_ranks(
            predictions,
            values,
            ("A", "B"),
            ("x", "y", "z"),
            "lower_is_better",
            0.5,
            2,
        )
        self.assertEqual(ranked["within_cid_rank"].tolist(), [1, 2])
        self.assertEqual(
            ranked["in_global_top0.5"].tolist(),
            [True, False],
        )

    def test_coarse_heads_group_by_shared_backbone(self) -> None:
        """Verify shared-backbone batches."""

        class FakeRunner:
            def __init__(self) -> None:
                self.calls = []

            def evaluate_grouped_candidates(
                self,
                candidates,
                phase,
                trial_numbers,
            ):
                self.calls.append((candidates, phase, trial_numbers))
                return [
                    pd.DataFrame(
                        [
                            {
                                "candidate_id": candidate.identifier,
                                **candidate.as_dict(),
                                "seed_offset": 0,
                                "global_skill": 0.1,
                                "within_skill": 0.2,
                                "selection_score_j": 0.15,
                            }
                        ]
                    )
                    for candidate in candidates
                ]

        runner = FakeRunner()
        candidates = grid_candidates(
            (0.1, 0.5),
            (0.0, 1.0),
            (0.0, 0.25),
        )
        seed_metrics, summary = evaluate_candidates(
            runner,
            candidates,
            "coarse_grid",
        )
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(seed_metrics), 8)
        self.assertEqual(len(summary), 8)

    def test_fast_backbone_batch_counts(self) -> None:
        """Verify 50 and 1 batches."""
        candidates = grid_candidates(
            (0.1, 0.25, 0.5, 1.0, 2.0),
            (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
            (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
        )
        groups = backbone_groups(candidates, range(len(candidates)))
        self.assertEqual(len(groups), 36)
        self.assertTrue(all(len(group) == 5 for group in groups))
        coarse_batches = (
            len(model_batches(groups, 8))
            * 5
            * 2
        )
        continuous_batches = len(
            model_batches(
                [(fold, seed) for seed in (0, 1) for fold in range(5)],
                10,
            )
        )
        self.assertEqual(coarse_batches, 50)
        self.assertEqual(continuous_batches, 1)

    def test_continuous_trial_batches_folds_and_seeds(self) -> None:
        """Verify mixed-seed caching."""
        path = CORE_DIR / "adaptive_search_common.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            node.name: ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        evaluate = methods["evaluate_candidate"]
        self.assertIn("for seed_offset in self.plan.seed_offsets", evaluate)
        self.assertIn("self.plan.model_batch_size", evaluate)
        self.assertIn("seed_offsets", evaluate)
        self.assertIn("_cached_supervision", methods["train_backbone_batch"])
        self.assertIn("_cached_schedules", methods["train_backbone_batch"])

    def test_mixed_seed_backbone_initialization(self) -> None:
        """Verify mixed seeds."""
        path = CORE_DIR / "adaptive_fast_backbones.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            node.name: ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        constructor = methods["__init__"]
        fit = methods["fit_fixed_weight_batch"]
        self.assertIn("torch.stack", constructor)
        self.assertIn("load_scorer_states", fit)
        self.assertIn("schedule_step", fit)
        self.assertNotIn(
            "torch.stack([schedules[value][index] for value in seeds])",
            fit,
        )

    def test_trial_feature_refresh_disables_autograd(self) -> None:
        """Do not retain backbone graphs."""
        path = CORE_DIR / "adaptive_fast_trials.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        for name in ("build_features", "refresh_trial_features"):
            decorators = {
                ast.unparse(value)
                for value in methods[name].decorator_list
            }
            self.assertIn("torch.no_grad()", decorators)

    def test_backbone_schedule_is_unique_seed_int32(self) -> None:
        """Keep unique-seed int32 plans."""
        path = CORE_DIR / "adaptive_fast_backbones.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            node.name: ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        full = methods["full_schedule"]
        fit = methods["fit_fixed_weight_batch"]
        step = methods["schedule_step"]
        self.assertIn("dtype=np.int32", full)
        self.assertIn("for value in set(seeds)", fit)
        self.assertIn("schedule_step", fit)
        self.assertIn(".long()", step)

    def test_sampling_plans_upload_once(self) -> None:
        """Verify one-shot schedule uploads."""
        backbone_path = CORE_DIR / "adaptive_fast_backbones.py"
        head_path = CORE_DIR / "adaptive_fast_heads.py"
        backbone = ast.parse(backbone_path.read_text(encoding="utf-8"))
        head = ast.parse(head_path.read_text(encoding="utf-8"))
        backbone_methods = {
            node.name: ast.unparse(node)
            for node in ast.walk(backbone)
            if isinstance(node, ast.FunctionDef)
        }
        head_methods = {
            node.name: ast.unparse(node)
            for node in ast.walk(head)
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("for epoch in range(epochs)", backbone_methods["full_schedule"])
        self.assertNotIn(".cpu()", backbone_methods["fit_fixed_weight_batch"].split("for epoch in range(1, epochs + 1):")[1].split("if retain_device:")[0])
        self.assertIn("head_schedule", head_methods["fit_candidate_batch"])
        self.assertNotIn(".cpu()", head_methods["fit_candidate_batch"].split("for epoch in range(1, epochs + 1):")[1].split("final_effective =")[0])

    def test_complete_gpu_trial_path(self) -> None:
        """Verify the complete GPU trial path."""
        path = CORE_DIR / "adaptive_search_common.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            node.name: ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        evaluate = methods["evaluate_candidate"]
        complete = methods["run_complete_trial_batch"]
        self.assertIn("self.run_complete_trial_batch", evaluate)
        self.assertIn("self.fast_trial.run_trial_batch", complete)
        self.assertIn("self._commit_heads", complete)
        self.assertIn("self._commit_backbones", complete)

    def test_formal_outer_protocol_is_five_by_five(self) -> None:
        """Verify the formal 5-by-5 protocol."""
        import yaml

        path = CORE_DIR.parents[1] / "configs" / "adaptive_pipeline.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["m1"]["outer_repeats"], 5)
        self.assertEqual(config["m1"]["outer_folds"], 5)
        self.assertEqual(config["evaluation"]["repeats"], 5)

    def test_fast_backbone_validator_uses_runner_module(self) -> None:
        """Resolve the module via runner."""
        path = CORE_DIR / "validate_adaptive_fast_backbones.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        attributes = {
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertIn(
            "runtime.runner.fast_backbone",
            attributes,
        )
        self.assertNotIn("runtime.fast_backbone", attributes)

    def test_fast_backbone_uses_high_throughput_kernels(self) -> None:
        """Verify batched fast kernels."""
        path = CORE_DIR / "adaptive_fast_backbones.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            node.name: ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("F.embedding", methods["_gather"])
        self.assertIn("torch.gather", methods["_gather"])
        self.assertIn("einsum", methods["decode_interaction"])

    def test_fast_backbone_validator_gates_selection_outputs(self) -> None:
        """Gate selection quantities only."""
        path = CORE_DIR / "validate_adaptive_fast_backbones.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            node.name: ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        gate = methods["comparison_gate"]
        self.assertIn("quantity.startswith('state_')", gate)
        self.assertIn("quantity == 'validation_residual'", gate)
        self.assertIn("return 'diagnostic'", gate)
        self.assertIn("return 'selection'", gate)
        self.assertIn("reclassify_existing(output_dir)", methods["main"])

    def test_fast_backbone_validator_has_true_throughput_mode(self) -> None:
        """Verify true throughput scaling."""
        path = CORE_DIR / "validate_adaptive_fast_backbones.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            node.name: ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        builder = methods["build_throughput_jobs"]
        batched = methods["batched_backbones"]
        main = methods["main"]
        self.assertIn("for index in range(n_models)", builder)
        self.assertIn("'requested_batch_size': batch_size", batched)
        self.assertIn("'actual_batch_size': len(chunk)", batched)
        self.assertIn("args.throughput_models < max(batch_sizes)", main)
        self.assertIn("fast_backbone_throughput_timing.csv", main)


if __name__ == "__main__":
    unittest.main()
