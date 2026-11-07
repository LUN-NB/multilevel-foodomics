"""Evaluate the complete 180-point CPI-MEC coarse grid.

All 180 settings use identical inner-OOF folds and seeds. The 55 zero-weight
settings are ablations only; the exact J maximum among 125 positive settings
is the formal coarse winner.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adaptive_search_common import (
    SCRIPT_RELEASE as OBJECTIVE_RELEASE,
    Candidate,
    SearchPlan,
    evaluate_candidates,
    grid_candidates,
    implementation_hashes,
    initialize_search,
    local_bounds,
    pareto_front,
    parse_float_values,
    parse_int_values,
    search_signature,
    select_exact_best,
    sha256,
)


SCRIPT_VERSION = "1.1.0"
SCRIPT_RELEASE = "complete_180_grid_fast_backbone_20260729"
DEFAULT_ANCHOR_VALUES = "0.1,0.25,0.5,1,2"
DEFAULT_GRAPHBAN_VALUES = "0,0.25,0.5,1,2,4"
DEFAULT_BOLTZ_VALUES = "0,0.25,0.5,1,2,4"


def main(argv: list[str] | None = None) -> None:
    """Run the complete coarse search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=("nested", "full"), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--inner-folds", default="1,2,3,4,5")
    parser.add_argument("--seed-offsets", default="0,1")
    parser.add_argument("--anchor-values", default=DEFAULT_ANCHOR_VALUES)
    parser.add_argument("--graphban-values", default=DEFAULT_GRAPHBAN_VALUES)
    parser.add_argument("--boltz-values", default=DEFAULT_BOLTZ_VALUES)
    parser.add_argument("--domain")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone-epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--head-epochs", type=int, default=120)
    parser.add_argument("--model-batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    if args.scope == "nested" and args.outer_fold is None:
        parser.error("--outer-fold is required for nested search")
    if args.scope == "full" and args.outer_fold is not None:
        parser.error("--outer-fold is not used for full-label CV")

    config_path = Path(args.config).resolve()
    core_dir = Path(args.core_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_values = parse_float_values(args.anchor_values)
    graphban_values = parse_float_values(args.graphban_values)
    boltz_values = parse_float_values(args.boltz_values)
    plan = SearchPlan(
        scope=args.scope,
        repeat=args.repeat,
        outer_fold=args.outer_fold,
        inner_folds=parse_int_values(args.inner_folds),
        seed_offsets=parse_int_values(args.seed_offsets),
        backbone_epochs=args.backbone_epochs,
        steps_per_epoch=args.steps_per_epoch,
        head_epochs=args.head_epochs,
        model_batch_size=args.model_batch_size,
    )
    signature_payload = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256(Path(__file__).resolve()),
        "objective_release": OBJECTIVE_RELEASE,
        "scope": plan.scope,
        "repeat": plan.repeat,
        "outer_fold": plan.outer_fold,
        "inner_folds": plan.inner_folds,
        "seed_offsets": plan.seed_offsets,
        "anchor_values": anchor_values,
        "graphban_values": graphban_values,
        "boltz_values": boltz_values,
        "backbone_epochs": plan.backbone_epochs,
        "steps_per_epoch": plan.steps_per_epoch,
        "head_epochs": plan.head_epochs,
        "model_batch_size": plan.model_batch_size,
        "implementation_sha256": implementation_hashes(core_dir),
        "config_sha256": sha256(config_path),
        "outer_splits_sha256": sha256(
            splits_dir / "outer_assignments.csv"
        ),
        "inner_splits_sha256": sha256(
            splits_dir / "inner_assignments.csv"
        ),
        "objective": "J=mean_seed(0.5*G+0.5*W)",
    }
    signature = search_signature(signature_payload)
    runtime = initialize_search(
        config_path,
        core_dir,
        splits_dir,
        output_dir,
        args.domain,
        args.device,
        plan,
        signature,
        "cpi_mec_coarse",
    )

    candidates = grid_candidates(
        anchor_values,
        graphban_values,
        boltz_values,
    )
    print(
        f"PLAN  coarse grid: {len(candidates)} candidates "
        "(125 formal + 55 ablation)",
        flush=True,
    )
    backbone_groups = len(graphban_values) * len(boltz_values)
    backbone_batches = (
        (backbone_groups + plan.model_batch_size - 1)
        // plan.model_batch_size
        * len(plan.inner_folds)
        * len(plan.seed_offsets)
    )
    print(
        f"PLAN  coarse backbones: {backbone_groups * len(plan.inner_folds) * len(plan.seed_offsets)} "
        f"fits in {backbone_batches} GPU model batches",
        flush=True,
    )
    seed_metrics, summary = evaluate_candidates(
        runtime.runner,
        candidates,
        "coarse_grid",
    )
    summary["ablation_only"] = (
        (summary["graphban_weight"] == 0)
        | (summary["boltz_weight"] == 0)
    )
    summary["eligible_for_selection"] = ~summary["ablation_only"]
    selected, summary = select_exact_best(summary)
    bounds = local_bounds(
        selected,
        anchor_values,
        graphban_values,
        boltz_values,
    )
    selected_row = summary.loc[summary["selected"]].iloc[0]
    selection = pd.DataFrame(
        [
            {
                "scope": plan.scope,
                "repeat": plan.repeat,
                "outer_fold": plan.outer_fold,
                "selected_candidate_id": selected.identifier,
                **selected.as_dict(),
                "selection_score_j": selected_row[
                    "mean_seed_pooled_selection_score_j"
                ],
                "global_skill": selected_row[
                    "mean_seed_pooled_global_skill"
                ],
                "within_skill": selected_row[
                    "mean_seed_pooled_within_skill"
                ],
                "pooled_auprc": selected_row[
                    "mean_seed_pooled_ranking_pooled_auprc"
                ],
                "pairwise_logistic_loss": selected_row[
                    "mean_seed_pooled_pairwise_logistic_loss"
                ],
                "selection_rule": (
                    "exact maximum J among positive coarse candidates"
                ),
            }
        ]
    )

    seed_metrics.to_csv(
        output_dir / "coarse_seed_pooled_metrics.csv",
        index=False,
    )
    summary.sort_values(
        ["eligible_for_selection", "objective_rank", "candidate_id"],
        ascending=[False, True, True],
    ).to_csv(
        output_dir / "coarse_candidate_summary.csv",
        index=False,
    )
    selection.to_csv(output_dir / "selected_parameters.csv", index=False)
    bounds.to_csv(output_dir / "local_continuous_bounds.csv", index=False)
    pareto_front(summary).to_csv(
        output_dir / "pareto_front.csv",
        index=False,
    )
    runtime.runner.save_progress()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module": Path(__file__).name,
        "module_version": SCRIPT_VERSION,
        "module_release": SCRIPT_RELEASE,
        "search_signature": signature,
        "scope": plan.scope,
        "repeat": plan.repeat,
        "outer_fold": plan.outer_fold,
        "objective": {
            "global_skill": "G=(AUPRC-prevalence)/(1-prevalence)",
            "within_skill": "W=1-pairwise_logistic_loss/ln(2)",
            "selection_score": "J=mean_seed(0.5*G+0.5*W)",
            "rule": "exact_best",
        },
        "candidate_counts": {
            "all": len(candidates),
            "formal_positive": int(summary["eligible_for_selection"].sum()),
            "ablation_only": int(summary["ablation_only"].sum()),
        },
        "selected_parameters": selected.as_dict(),
        "local_continuous_bounds": bounds.to_dict(orient="records"),
        "plan": {
            "inner_folds": plan.inner_folds,
            "seed_offsets": plan.seed_offsets,
            "backbone_epochs": plan.backbone_epochs,
            "steps_per_epoch": plan.steps_per_epoch,
            "head_epochs": plan.head_epochs,
            "model_batch_size": plan.model_batch_size,
            "backbone_gpu_batches": backbone_batches,
        },
        "dependencies": {
            runtime.base_path.name: sha256(runtime.base_path),
            runtime.fixed_path.name: sha256(runtime.fixed_path),
            runtime.fast_path.name: sha256(runtime.fast_path),
            runtime.fast_backbone_path.name: sha256(
                runtime.fast_backbone_path
            ),
        },
        "signature_payload": signature_payload,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"SELECTED {selected.identifier}", flush=True)
    print(f"PASS  coarse search output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
