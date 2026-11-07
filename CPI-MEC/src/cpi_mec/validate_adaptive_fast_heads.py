"""Validate batched adaptive heads against the formal serial implementation.

Under one frozen backbone, initialization, and sampling stream, fit the full
anchor grid serially and in one batch, then compare predictions, coefficients,
and epoch losses.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_fast_heads import fit_candidate_batch
from adaptive_search_common import (
    Candidate,
    SearchPlan,
    implementation_hashes,
    initialize_search,
    parse_float_values,
    search_signature,
    sha256,
)


SCRIPT_VERSION = "1.1.0"
SCRIPT_RELEASE = "serial_batch_parity_fast_backbone_20260729"
COMPARE_COLUMNS = (
    "probability",
    "residual",
    "final_score",
    "ranking_score",
)


def difference(
    serial: np.ndarray,
    batched: np.ndarray,
) -> tuple[float, float]:
    """Return maximum absolute and relative error."""
    absolute = np.abs(serial - batched)
    scale = np.maximum(np.abs(serial), np.abs(batched))
    relative = absolute / np.maximum(scale, np.finfo(np.float64).eps)
    return float(absolute.max(initial=0.0)), float(relative.max(initial=0.0))


def main(argv: list[str] | None = None) -> None:
    """Run real-path parity validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--inner-fold", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--anchor-values",
        default="0.1,0.25,0.5,1,2",
    )
    parser.add_argument("--graphban-weight", type=float, default=1.0)
    parser.add_argument("--boltz-weight", type=float, default=0.25)
    parser.add_argument("--backbone-epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--head-epochs", type=int, default=120)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--domain")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    core_dir = Path(args.core_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchors = parse_float_values(args.anchor_values)
    signature_payload = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256(Path(__file__).resolve()),
        "repeat": args.repeat,
        "outer_fold": args.outer_fold,
        "inner_fold": args.inner_fold,
        "seed_offset": args.seed_offset,
        "anchor_values": anchors,
        "graphban_weight": args.graphban_weight,
        "boltz_weight": args.boltz_weight,
        "backbone_epochs": args.backbone_epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "head_epochs": args.head_epochs,
        "implementation_sha256": implementation_hashes(core_dir),
        "config_sha256": sha256(config_path),
    }
    signature = search_signature(signature_payload)
    plan = SearchPlan(
        scope="nested",
        repeat=args.repeat,
        outer_fold=args.outer_fold,
        inner_folds=(args.inner_fold,),
        seed_offsets=(args.seed_offset,),
        backbone_epochs=args.backbone_epochs,
        steps_per_epoch=args.steps_per_epoch,
        head_epochs=args.head_epochs,
        model_batch_size=len(anchors),
    )
    runtime = initialize_search(
        config_path,
        core_dir,
        splits_dir,
        output_dir,
        args.domain,
        args.device,
        plan,
        signature,
        "cpi_mec_fast_validation",
    )
    runner = runtime.runner
    train_rows, validation_rows = runner.fold_rows(args.inner_fold)
    representative = Candidate(
        anchors[0],
        args.graphban_weight,
        args.boltz_weight,
    )
    backbone = runner.train_backbone(
        representative,
        train_rows,
        args.inner_fold,
        args.seed_offset,
    )
    runner.activate_backbone(backbone, representative)
    head_seed = runtime.seed_states[args.seed_offset]["seed"] + 5000

    from training import fit_candidate

    serial_results = [
        fit_candidate(
            runtime.context,
            train_rows,
            learnable_coefficient=True,
            residual_anchor_weight=anchor,
            seed=head_seed,
            epochs=args.head_epochs,
        )
        for anchor in anchors
    ]
    batched_results = fit_candidate_batch(
        runtime.context,
        train_rows,
        anchors,
        learnable_coefficient=True,
        seed=head_seed,
        epochs=args.head_epochs,
    )

    audit_rows = []
    passed = True
    for anchor, serial, batched in zip(
        anchors,
        serial_results,
        batched_results,
    ):
        serial_prediction, _ = runtime.base.evaluate_strategy(
            runtime.context,
            serial,
            validation_rows,
            "serial",
        )
        batched_prediction, _ = runtime.base.evaluate_strategy(
            runtime.context,
            batched,
            validation_rows,
            "batched",
        )
        for column in COMPARE_COLUMNS:
            serial_values = serial_prediction[column].to_numpy(np.float64)
            batched_values = batched_prediction[column].to_numpy(np.float64)
            maximum_absolute, maximum_relative = difference(
                serial_values,
                batched_values,
            )
            comparison_passed = bool(
                np.allclose(
                    serial_values,
                    batched_values,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            passed &= comparison_passed
            audit_rows.append(
                {
                    "anchor_weight": anchor,
                    "quantity": f"validation_{column}",
                    "max_absolute_difference": maximum_absolute,
                    "max_relative_difference": maximum_relative,
                    "passed": comparison_passed,
                }
            )
        history_columns = [
            column
            for column in serial.history.select_dtypes(
                include=[np.number]
            ).columns
            if column != "epoch"
        ]
        for column in history_columns:
            serial_values = serial.history[column].to_numpy(np.float64)
            batched_values = batched.history[column].to_numpy(np.float64)
            maximum_absolute, maximum_relative = difference(
                serial_values,
                batched_values,
            )
            comparison_passed = bool(
                np.allclose(
                    serial_values,
                    batched_values,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            passed &= comparison_passed
            audit_rows.append(
                {
                    "anchor_weight": anchor,
                    "quantity": f"history_{column}",
                    "max_absolute_difference": maximum_absolute,
                    "max_relative_difference": maximum_relative,
                    "passed": comparison_passed,
                }
            )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "fast_head_parity_audit.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module": Path(__file__).name,
        "module_version": SCRIPT_VERSION,
        "module_release": SCRIPT_RELEASE,
        "passed": passed,
        "absolute_tolerance": args.atol,
        "relative_tolerance": args.rtol,
        "n_anchor_heads": len(anchors),
        "head_seed": head_seed,
        "signature_payload": signature_payload,
        "max_absolute_difference": float(
            audit["max_absolute_difference"].max()
        ),
        "max_relative_difference": float(
            audit["max_relative_difference"].max()
        ),
    }
    (output_dir / "fast_head_parity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(
            "Fast-head parity validation failed; inspect "
            "fast_head_parity_audit.csv"
        )
    print(
        "PASS  fast-head serial/batch parity: "
        f"{len(anchors)} anchors, "
        f"max_abs={manifest['max_absolute_difference']:.3g}, "
        f"max_rel={manifest['max_relative_difference']:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
