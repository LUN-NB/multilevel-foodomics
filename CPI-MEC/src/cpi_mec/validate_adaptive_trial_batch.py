"""Validate serial and complete-GPU trial paths on the same candidates.

Compare predictions, G/W/J, and exact-best selection under identical inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_fast_heads import fit_candidate_batch
from adaptive_search_common import (
    Candidate,
    SearchPlan,
    add_skill_scores,
    candidate_summary,
    initialize_search,
    search_signature,
    select_exact_best,
    sha256,
)


SCRIPT_VERSION = "1.0.0"
SCRIPT_RELEASE = "serial_complete_gpu_trial_parity_20260729"
PREDICTION_COLUMNS = (
    "probability",
    "residual",
    "final_score",
    "ranking_score",
)
METRIC_COLUMNS = ("global_skill", "within_skill", "selection_score_j")


def parse_candidates(text: str) -> tuple[Candidate, ...]:
    """Parse A:G:B candidates."""
    return tuple(
        Candidate(*(float(value) for value in item.split(":")))
        for item in text.split(",")
    )


def difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return maximum absolute and relative errors."""
    absolute = np.abs(left - right)
    scale = np.maximum(np.abs(left), np.abs(right))
    relative = absolute / np.maximum(scale, np.finfo(np.float64).eps)
    return (
        float(absolute.max(initial=0.0)),
        float(relative.max(initial=0.0)),
    )


def pooled_metrics(runtime, predictions: pd.DataFrame) -> pd.DataFrame:
    """Pool folds by seed and compute J."""
    from evaluation import evaluate_predictions

    rows = []
    for seed_offset, part in predictions.groupby("seed_offset", sort=True):
        rows.append(
            {
                "candidate_id": part["candidate_id"].iloc[0],
                "anchor_weight": part["anchor_weight"].iloc[0],
                "graphban_weight": part["graphban_weight"].iloc[0],
                "boltz_weight": part["boltz_weight"].iloc[0],
                "seed_offset": int(seed_offset),
                **add_skill_scores(
                    evaluate_predictions(
                        part,
                        runtime.context.settings.ranking_temperature,
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def serial_candidate(runtime, candidate: Candidate) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate through the prior serial tail."""
    parts = []
    runner = runtime.runner
    for seed_offset in runner.plan.seed_offsets:
        state = runtime.seed_states[seed_offset]
        for inner_fold in runner.plan.inner_folds:
            train_rows, validation_rows = runner.fold_rows(inner_fold)
            job = runner.fast_backbone.BackboneJob(
                train_rows,
                candidate.graphban,
                candidate.boltz,
            )
            fitted_backbone = runner.fast_backbone.fit_fixed_weight_batch(
                runtime.context,
                runner.backbone_data,
                state["backbone"],
                state["scorer"],
                state["standardizer"],
                [job],
                runner.plan.backbone_epochs,
                runner.plan.steps_per_epoch,
                state["seed"],
                runner.fast_backbone.prepare_supervision(
                    runtime.context,
                    [job],
                ),
                {
                    state["seed"]: runner.fast_backbone.full_schedule(
                        runtime.context,
                        runner.backbone_data,
                        state["seed"],
                        runner.plan.backbone_epochs,
                        runner.plan.steps_per_epoch,
                    )
                },
            )[0]
            runtime.context.standardizer = copy.deepcopy(
                state["standardizer"]
            )
            runner.activate_backbone(fitted_backbone.state, candidate)
            fitted_head = fit_candidate_batch(
                runtime.context,
                train_rows,
                [candidate.anchor],
                True,
                state["seed"] + 5000,
                runner.plan.head_epochs,
            )[0]
            prediction, _ = runtime.base.evaluate_strategy(
                runtime.context,
                fitted_head,
                validation_rows,
                candidate.identifier,
            )
            prediction["inner_fold"] = inner_fold
            prediction["seed_offset"] = seed_offset
            prediction["candidate_id"] = candidate.identifier
            for name, value in candidate.as_dict().items():
                prediction[name] = value
            parts.append(prediction)
    predictions = pd.concat(parts, ignore_index=True)
    return predictions, pooled_metrics(runtime, predictions)


def batched_candidate(runtime, candidate: Candidate) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate through the complete GPU path."""
    from evaluation import ranking_score

    runner = runtime.runner
    pending = [
        (inner_fold, seed_offset)
        for seed_offset in runner.plan.seed_offsets
        for inner_fold in runner.plan.inner_folds
    ]
    folds = [value[0] for value in pending]
    seed_offsets = [value[1] for value in pending]
    train_validation = [runner.fold_rows(value) for value in folds]
    states = [runtime.seed_states[value] for value in seed_offsets]
    jobs = [
        runner.fast_backbone.BackboneJob(
            rows[0],
            candidate.graphban,
            candidate.boltz,
        )
        for rows in train_validation
    ]
    result = runner.fast_trial.run_trial_batch(
        runtime.context,
        runner.backbone_data,
        [state["backbone"] for state in states],
        [state["scorer"] for state in states],
        [state["standardizer"] for state in states],
        jobs,
        [state["seed"] for state in states],
        [state["seed"] + 5000 for state in states],
        [candidate.anchor for _ in states],
        [rows[1] for rows in train_validation],
        runner.plan.backbone_epochs,
        runner.plan.steps_per_epoch,
        runner.plan.head_epochs,
        runner.fast_backbone.prepare_supervision(runtime.context, jobs),
        {
            seed: runner.fast_backbone.full_schedule(
                runtime.context,
                runner.backbone_data,
                seed,
                runner.plan.backbone_epochs,
                runner.plan.steps_per_epoch,
            )
            for seed in set(state["seed"] for state in states)
        },
    )
    parts = []
    for prediction, inner_fold, seed_offset in zip(
        result.predictions,
        folds,
        seed_offsets,
    ):
        prediction["ranking_score"] = ranking_score(
            prediction["final_score"],
            runtime.context.models.architecture.primary_direction,
        )
        prediction["inner_fold"] = inner_fold
        prediction["seed_offset"] = seed_offset
        prediction["candidate_id"] = candidate.identifier
        for name, value in candidate.as_dict().items():
            prediction[name] = value
        parts.append(prediction)
    predictions = pd.concat(parts, ignore_index=True)
    return predictions, pooled_metrics(runtime, predictions)


def main(argv: list[str] | None = None) -> None:
    """Run real serial-batch validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--inner-folds", default="1,2,3,4,5")
    parser.add_argument("--seed-offsets", default="0,1")
    parser.add_argument(
        "--candidates",
        default="0.1:4:0.25,0.25:4:0.25",
    )
    parser.add_argument("--backbone-epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--domain")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    candidates = parse_candidates(args.candidates)
    inner_folds = tuple(int(value) for value in args.inner_folds.split(","))
    seed_offsets = tuple(int(value) for value in args.seed_offsets.split(","))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = search_signature(
        {
            "module_version": SCRIPT_VERSION,
            "config_sha256": sha256(Path(args.config).resolve()),
            "repeat": args.repeat,
            "outer_fold": args.outer_fold,
            "inner_folds": inner_folds,
            "seed_offsets": seed_offsets,
            "candidates": [value.as_dict() for value in candidates],
            "backbone_epochs": args.backbone_epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "head_epochs": args.head_epochs,
        }
    )
    plan = SearchPlan(
        "nested",
        args.repeat,
        args.outer_fold,
        inner_folds,
        seed_offsets,
        args.backbone_epochs,
        args.steps_per_epoch,
        args.head_epochs,
        len(inner_folds) * len(seed_offsets),
    )
    runtime = initialize_search(
        Path(args.config).resolve(),
        Path(args.core_dir).resolve(),
        Path(args.splits_dir).resolve(),
        output_dir,
        args.domain,
        args.device,
        plan,
        signature,
        "cpi_mec_trial_batch_validation",
    )

    audit = []
    serial_metrics = []
    batched_metrics = []
    for candidate in candidates:
        serial_prediction, serial_seed = serial_candidate(runtime, candidate)
        batched_prediction, batched_seed = batched_candidate(runtime, candidate)
        keys = ["seed_offset", "inner_fold", "relation_row"]
        serial_prediction = serial_prediction.sort_values(keys)
        batched_prediction = batched_prediction.sort_values(keys)
        for column in PREDICTION_COLUMNS:
            absolute, relative = difference(
                serial_prediction[column].to_numpy(np.float64),
                batched_prediction[column].to_numpy(np.float64),
            )
            audit.append(
                {
                    "candidate_id": candidate.identifier,
                    "quantity": f"prediction_{column}",
                    "max_absolute_difference": absolute,
                    "max_relative_difference": relative,
                    "passed": bool(
                        np.allclose(
                            serial_prediction[column],
                            batched_prediction[column],
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                    ),
                }
            )
        serial_seed = serial_seed.sort_values("seed_offset")
        batched_seed = batched_seed.sort_values("seed_offset")
        for column in METRIC_COLUMNS:
            absolute, relative = difference(
                serial_seed[column].to_numpy(np.float64),
                batched_seed[column].to_numpy(np.float64),
            )
            audit.append(
                {
                    "candidate_id": candidate.identifier,
                    "quantity": column,
                    "max_absolute_difference": absolute,
                    "max_relative_difference": relative,
                    "passed": bool(
                        np.allclose(
                            serial_seed[column],
                            batched_seed[column],
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                    ),
                }
            )
        serial_metrics.append(serial_seed)
        batched_metrics.append(batched_seed)

    serial_table = pd.concat(serial_metrics, ignore_index=True)
    batched_table = pd.concat(batched_metrics, ignore_index=True)
    serial_selected, _ = select_exact_best(candidate_summary(serial_table))
    batched_selected, _ = select_exact_best(candidate_summary(batched_table))
    selection_passed = serial_selected == batched_selected
    audit.append(
        {
            "candidate_id": "ALL",
            "quantity": "final_selected_candidate",
            "max_absolute_difference": 0.0,
            "max_relative_difference": 0.0,
            "passed": selection_passed,
        }
    )
    audit_table = pd.DataFrame(audit)
    audit_table.to_csv(output_dir / "trial_batch_parity_audit.csv", index=False)
    pd.concat(
        [
            serial_table.assign(execution="serial_tail"),
            batched_table.assign(execution="complete_gpu_trial"),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "trial_batch_candidate_metrics.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module_version": SCRIPT_VERSION,
        "module_release": SCRIPT_RELEASE,
        "passed": bool(audit_table["passed"].all()),
        "serial_selected": serial_selected.identifier,
        "batched_selected": batched_selected.identifier,
        "selection_passed": selection_passed,
        "absolute_tolerance": args.atol,
        "relative_tolerance": args.rtol,
        "signature": signature,
    }
    (output_dir / "trial_batch_parity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not manifest["passed"]:
        raise RuntimeError(
            "Complete GPU trial parity failed; inspect trial_batch_parity_audit.csv"
        )
    print(
        "PASS  serial/complete-GPU trial parity: "
        f"selected={serial_selected.identifier}",
        flush=True,
    )


if __name__ == "__main__":
    main()
