"""Validate fast backbones against the formal serial training path.

Validate both coarse mode (different G/B within one fold/seed) and continuous
mode (different inner folds within one seed), including states, histories,
post-head predictions, G/W/J, and coarse-candidate ordering.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from adaptive_fast_heads import fit_candidate_batch
from adaptive_search_common import (
    Candidate,
    SearchPlan,
    add_skill_scores,
    implementation_hashes,
    initialize_search,
    parse_int_values,
    search_signature,
    sha256,
)


SCRIPT_VERSION = "1.1.0"
SCRIPT_RELEASE = "true_batch_throughput_benchmark_20260729"
PREDICTION_COLUMNS = (
    "probability",
    "residual",
    "final_score",
    "ranking_score",
)
METRIC_COLUMNS = (
    "global_skill",
    "within_skill",
    "selection_score_j",
)


@dataclass(frozen=True)
class ValidationJob:
    """Store one validation fit definition."""

    scenario: str
    identifier: str
    candidate: Candidate
    inner_fold: int
    train_rows: np.ndarray
    validation_rows: np.ndarray


def parse_pairs(text: str) -> tuple[tuple[float, float], ...]:
    """Parse G:B candidates."""
    pairs = []
    for item in text.split(","):
        graphban, boltz = item.split(":")
        pairs.append((float(graphban), float(boltz)))
    return tuple(pairs)


def difference(
    serial: np.ndarray,
    batched: np.ndarray,
) -> tuple[float, float]:
    """Return maximum errors."""
    absolute = np.abs(serial - batched)
    scale = np.maximum(np.abs(serial), np.abs(batched))
    relative = absolute / np.maximum(
        scale,
        np.finfo(np.float64).eps,
    )
    return (
        float(absolute.max(initial=0.0)),
        float(relative.max(initial=0.0)),
    )


def comparison_gate(quantity: str) -> str:
    """Mark selection gates or diagnostics."""
    if quantity.startswith("state_") or quantity == "validation_residual":
        return "diagnostic"
    return "selection"


def summarize_parity(
    audit_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """Summarize gates and diagnostics."""
    audit_table = audit_table.copy()
    audit_table["passed"] = (
        audit_table["passed"]
        if audit_table["passed"].dtype == bool
        else audit_table["passed"].astype(str).str.lower().eq("true")
    )
    audit_table["gate"] = audit_table["quantity"].map(comparison_gate)
    selection = audit_table["gate"] == "selection"
    diagnostic = audit_table["gate"] == "diagnostic"
    return audit_table, {
        "passed": bool(audit_table.loc[selection, "passed"].all()),
        "selection_critical_passed": bool(
            audit_table.loc[selection, "passed"].all()
        ),
        "diagnostic_parity_passed": bool(
            audit_table.loc[diagnostic, "passed"].all()
        ),
        "selection_critical_failures": int(
            (~audit_table.loc[selection, "passed"]).sum()
        ),
        "diagnostic_failures": int(
            (~audit_table.loc[diagnostic, "passed"]).sum()
        ),
        "selection_max_absolute_difference": float(
            audit_table.loc[
                selection,
                "max_absolute_difference",
            ].max()
        ),
        "diagnostic_max_absolute_difference": float(
            audit_table.loc[
                diagnostic,
                "max_absolute_difference",
            ].max()
        ),
    }


def report_parity(summary: dict[str, float | int | bool]) -> None:
    """Report gates and diagnostics."""
    if not summary["selection_critical_passed"]:
        raise RuntimeError(
            "Fast-backbone selection-critical parity failed "
            f"({summary['selection_critical_failures']} checks); inspect "
            "fast_backbone_parity_audit.csv"
        )
    if not summary["diagnostic_parity_passed"]:
        print(
            "WARN  fast-backbone numerical diagnostics exceeded strict "
            f"tolerance: {summary['diagnostic_failures']} "
            "state/residual checks; selection-critical outputs still passed",
            flush=True,
        )


def reclassify_existing(output_dir: Path) -> None:
    """Reclassify an existing audit only."""
    audit_path = output_dir / "fast_backbone_parity_audit.csv"
    manifest_path = output_dir / "fast_backbone_parity_manifest.json"
    if not audit_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Audit-only mode requires fast_backbone_parity_audit.csv "
            "and fast_backbone_parity_manifest.json"
        )
    audit_table, summary = summarize_parity(pd.read_csv(audit_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "reclassified_utc": datetime.now(timezone.utc).isoformat(),
            "module_version": SCRIPT_VERSION,
            "module_release": SCRIPT_RELEASE,
            **summary,
        }
    )
    audit_table.to_csv(audit_path, index=False)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_parity(summary)
    print(
        "PASS  existing fast-backbone audit: "
        "selection-critical histories/predictions/metrics/ranking passed, "
        f"selection_max_abs="
        f"{summary['selection_max_absolute_difference']:.3g}",
        flush=True,
    )


def compare_values(
    audit: list[dict[str, object]],
    scenario: str,
    identifier: str,
    batch_size: int,
    quantity: str,
    serial: np.ndarray,
    batched: np.ndarray,
    atol: float,
    rtol: float,
) -> bool:
    """Compare one serial/batched quantity."""
    maximum_absolute, maximum_relative = difference(serial, batched)
    passed = bool(
        np.allclose(
            serial,
            batched,
            atol=atol,
            rtol=rtol,
        )
    )
    audit.append(
        {
            "scenario": scenario,
            "job_id": identifier,
            "model_batch_size": batch_size,
            "quantity": quantity,
            "gate": comparison_gate(quantity),
            "max_absolute_difference": maximum_absolute,
            "max_relative_difference": maximum_relative,
            "passed": passed,
        }
    )
    return passed


def build_jobs(
    runner,
    scenario: str,
    anchor: float,
    coarse_pairs: tuple[tuple[float, float], ...],
    continuous_folds: tuple[int, ...],
    graphban: float,
    boltz: float,
) -> list[ValidationJob]:
    """Build both formal batch shapes."""
    jobs = []
    if scenario in ("coarse", "both"):
        fold = continuous_folds[0]
        train, validation = runner.fold_rows(fold)
        for index, (graphban_weight, boltz_weight) in enumerate(
            coarse_pairs
        ):
            jobs.append(
                ValidationJob(
                    "coarse",
                    f"coarse_{index + 1:02d}",
                    Candidate(anchor, graphban_weight, boltz_weight),
                    fold,
                    train,
                    validation,
                )
            )
    if scenario in ("continuous", "both"):
        candidate = Candidate(anchor, graphban, boltz)
        for fold in continuous_folds:
            train, validation = runner.fold_rows(fold)
            jobs.append(
                ValidationJob(
                    "continuous",
                    f"continuous_fold_{fold:02d}",
                    candidate,
                    fold,
                    train,
                    validation,
                )
            )
    return jobs


def build_throughput_jobs(
    runner,
    scenario: str,
    n_models: int,
    anchor: float,
    coarse_pairs: tuple[tuple[float, float], ...],
    continuous_folds: tuple[int, ...],
    graphban: float,
    boltz: float,
) -> list[ValidationJob]:
    """Expand real jobs to a model count."""
    source = build_jobs(
        runner,
        scenario,
        anchor,
        coarse_pairs,
        continuous_folds,
        graphban,
        boltz,
    )
    jobs = []
    for current_scenario in ("coarse", "continuous"):
        templates = [
            job for job in source if job.scenario == current_scenario
        ]
        if not templates:
            continue
        for index in range(n_models):
            template = templates[index % len(templates)]
            jobs.append(
                ValidationJob(
                    current_scenario,
                    f"throughput_{current_scenario}_{index + 1:03d}",
                    template.candidate,
                    template.inner_fold,
                    template.train_rows,
                    template.validation_rows,
                )
            )
    return jobs


def serial_backbones(runtime, jobs, seed_offset):
    """Run serial backbone baselines."""
    state = runtime.seed_states[seed_offset]
    results = {}
    timings = []
    for job in jobs:
        runtime.context.standardizer = copy.deepcopy(state["standardizer"])
        started = time.perf_counter()
        history, summary = runtime.fixed.train_fixed_weight(
            runtime.base,
            runtime.context,
            state["backbone"],
            state["scorer"],
            job.train_rows,
            job.candidate.graphban,
            job.candidate.boltz,
            runtime.runner.plan.backbone_epochs,
            runtime.runner.plan.steps_per_epoch,
            state["seed"],
        )
        elapsed = time.perf_counter() - started
        results[job.identifier] = {
            "state": runtime.fixed.cpu_state(
                runtime.context.models.backbone
            ),
            "history": history,
            "summary": summary,
        }
        timings.append(
            {
                "scenario": job.scenario,
                "execution": "serial",
                "requested_batch_size": 1,
                "actual_batch_size": 1,
                "model_batch_size": 1,
                "n_models": 1,
                "elapsed_seconds": elapsed,
                "seconds_per_backbone": elapsed,
            }
        )
    return results, timings


def batched_backbones(runtime, jobs, seed_offset, batch_size):
    """Run backbones with one batch size."""
    state = runtime.seed_states[seed_offset]
    results = {}
    timings = []
    for scenario in ("coarse", "continuous"):
        selected = [job for job in jobs if job.scenario == scenario]
        for start in range(0, len(selected), batch_size):
            chunk = selected[start : start + batch_size]
            batch_jobs = [
                runtime.runner.fast_backbone.BackboneJob(
                    job.train_rows,
                    job.candidate.graphban,
                    job.candidate.boltz,
                )
                for job in chunk
            ]
            fitted = runtime.runner.fast_backbone.fit_fixed_weight_batch(
                runtime.context,
                runtime.runner.backbone_data,
                state["backbone"],
                state["scorer"],
                state["standardizer"],
                batch_jobs,
                runtime.runner.plan.backbone_epochs,
                runtime.runner.plan.steps_per_epoch,
                state["seed"],
            )
            elapsed = float(
                fitted[0].summary["batch_elapsed_seconds"]
            )
            timings.append(
                {
                    "scenario": scenario,
                    "execution": "fast_model_batch",
                    "requested_batch_size": batch_size,
                    "actual_batch_size": len(chunk),
                    "model_batch_size": len(chunk),
                    "n_models": len(chunk),
                    "elapsed_seconds": elapsed,
                    "seconds_per_backbone": elapsed / len(chunk),
                    "peak_cuda_memory_mb": float(
                        fitted[0].summary["peak_cuda_memory_mb"]
                    ),
                }
            )
            for job, result in zip(chunk, fitted):
                results[job.identifier] = {
                    "state": result.state,
                    "history": result.history,
                    "summary": result.summary,
                }
    return results, timings


def evaluate_state(runtime, job, state, seed_offset, head_epochs):
    """Fit the formal head and evaluate."""
    runtime.runner.activate_backbone(state, job.candidate)
    head_seed = runtime.seed_states[seed_offset]["seed"] + 5000
    fitted = fit_candidate_batch(
        runtime.context,
        job.train_rows,
        [job.candidate.anchor],
        learnable_coefficient=True,
        seed=head_seed,
        epochs=head_epochs,
    )[0]
    prediction, metrics = runtime.base.evaluate_strategy(
        runtime.context,
        fitted,
        job.validation_rows,
        job.identifier,
    )
    return prediction, add_skill_scores(metrics)


def main(argv: list[str] | None = None) -> None:
    """Run parity and speed validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--core-dir")
    parser.add_argument("--splits-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--throughput-only", action="store_true")
    parser.add_argument("--throughput-models", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--scenario",
        choices=("coarse", "continuous", "both"),
        default="both",
    )
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument(
        "--coarse-pairs",
        default=(
            "0.25:0.25,0.25:0.5,0.5:0.25,0.5:0.5,"
            "1:0.5,1:1,2:1,2:2"
        ),
    )
    parser.add_argument("--continuous-folds", default="1,2,3,4,5")
    parser.add_argument("--graphban-weight", type=float, default=1.0)
    parser.add_argument("--boltz-weight", type=float, default=0.25)
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--backbone-epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=5)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--feature-sample-size", type=int, default=4096)
    parser.add_argument("--anchor-sample-size", type=int, default=4096)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--domain")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    if args.audit_only and args.throughput_only:
        parser.error("--audit-only and --throughput-only are exclusive")
    if args.audit_only:
        reclassify_existing(output_dir)
        return
    if not all((args.config, args.core_dir, args.splits_dir)):
        parser.error(
            "--config, --core-dir, and --splits-dir are required "
            "unless --audit-only is used"
        )
    config_path = Path(args.config).resolve()
    core_dir = Path(args.core_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_pairs = parse_pairs(args.coarse_pairs)
    continuous_folds = parse_int_values(args.continuous_folds)
    batch_sizes = parse_int_values(args.batch_sizes)
    if args.throughput_only:
        if args.throughput_models < max(batch_sizes):
            parser.error(
                "--throughput-models must be at least the largest batch size"
            )
    signature_payload = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256(Path(__file__).resolve()),
        "repeat": args.repeat,
        "outer_fold": args.outer_fold,
        "seed_offset": args.seed_offset,
        "scenario": args.scenario,
        "anchor_weight": args.anchor_weight,
        "coarse_pairs": coarse_pairs,
        "continuous_folds": continuous_folds,
        "graphban_weight": args.graphban_weight,
        "boltz_weight": args.boltz_weight,
        "batch_sizes": batch_sizes,
        "throughput_only": args.throughput_only,
        "throughput_models": args.throughput_models,
        "backbone_epochs": args.backbone_epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "head_epochs": args.head_epochs,
        "feature_sample_size": args.feature_sample_size,
        "anchor_sample_size": args.anchor_sample_size,
        "implementation_sha256": implementation_hashes(core_dir),
        "config_sha256": sha256(config_path),
    }
    signature = search_signature(signature_payload)
    with config_path.open("r", encoding="utf-8") as handle:
        validation_config = yaml.safe_load(handle)
    validation_config["path_base"] = str(
        (
            config_path.parent
            / validation_config.get("path_base", ".")
        ).resolve()
    )
    validation_config["model"]["feature_sample_size"] = (
        args.feature_sample_size
    )
    validation_config["model"]["anchor_sample_size"] = (
        args.anchor_sample_size
    )
    validation_config_path = output_dir / "validation_input_config.yaml"
    with validation_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            validation_config,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    plan = SearchPlan(
        scope="nested",
        repeat=args.repeat,
        outer_fold=args.outer_fold,
        inner_folds=continuous_folds,
        seed_offsets=(args.seed_offset,),
        backbone_epochs=args.backbone_epochs,
        steps_per_epoch=args.steps_per_epoch,
        head_epochs=args.head_epochs,
        model_batch_size=max(batch_sizes),
    )
    runtime = initialize_search(
        validation_config_path,
        core_dir,
        splits_dir,
        output_dir,
        args.domain,
        args.device,
        plan,
        signature,
        "cpi_mec_fast_backbone_validation",
    )
    if args.throughput_only:
        jobs = build_throughput_jobs(
            runtime.runner,
            args.scenario,
            args.throughput_models,
            args.anchor_weight,
            coarse_pairs,
            continuous_folds,
            args.graphban_weight,
            args.boltz_weight,
        )
        timings = []
        for batch_size in batch_sizes:
            _, batch_timings = batched_backbones(
                runtime,
                jobs,
                args.seed_offset,
                batch_size,
            )
            timings.extend(batch_timings)
        timing_table = pd.DataFrame(timings)
        timing_table.to_csv(
            output_dir / "fast_backbone_throughput_timing.csv",
            index=False,
        )
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "module": Path(__file__).name,
            "module_version": SCRIPT_VERSION,
            "module_release": SCRIPT_RELEASE,
            "passed": True,
            "mode": "throughput_only",
            "models_per_scenario": args.throughput_models,
            "batch_sizes": batch_sizes,
            "signature_payload": signature_payload,
        }
        (
            output_dir / "fast_backbone_throughput_manifest.json"
        ).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "PASS  fast-backbone throughput benchmark: "
            f"{args.throughput_models} models per scenario, "
            f"requested batches={batch_sizes}",
            flush=True,
        )
        return

    jobs = build_jobs(
        runtime.runner,
        args.scenario,
        args.anchor_weight,
        coarse_pairs,
        continuous_folds,
        args.graphban_weight,
        args.boltz_weight,
    )
    serial, serial_timings = serial_backbones(
        runtime,
        jobs,
        args.seed_offset,
    )
    serial_predictions = {}
    serial_metrics = {}
    for job in jobs:
        prediction, metrics = evaluate_state(
            runtime,
            job,
            serial[job.identifier]["state"],
            args.seed_offset,
            args.head_epochs,
        )
        serial_predictions[job.identifier] = prediction
        serial_metrics[job.identifier] = metrics

    audit = []
    timings = list(serial_timings)
    for batch_size in batch_sizes:
        batched, batch_timings = batched_backbones(
            runtime,
            jobs,
            args.seed_offset,
            batch_size,
        )
        timings.extend(batch_timings)
        batch_metrics = {}
        for job in jobs:
            serial_result = serial[job.identifier]
            batch_result = batched[job.identifier]
            for name in serial_result["state"]:
                compare_values(
                    audit,
                    job.scenario,
                    job.identifier,
                    batch_size,
                    f"state_{name}",
                    serial_result["state"][name].numpy(),
                    batch_result["state"][name].numpy(),
                    args.atol,
                    args.rtol,
                )
            history_columns = [
                column
                for column in serial_result["history"].select_dtypes(
                    include=[np.number]
                ).columns
                if column != "epoch"
            ]
            for column in history_columns:
                compare_values(
                    audit,
                    job.scenario,
                    job.identifier,
                    batch_size,
                    f"history_{column}",
                    serial_result["history"][column].to_numpy(
                        np.float64
                    ),
                    batch_result["history"][column].to_numpy(
                        np.float64
                    ),
                    args.atol,
                    args.rtol,
                )
            prediction, metrics = evaluate_state(
                runtime,
                job,
                batch_result["state"],
                args.seed_offset,
                args.head_epochs,
            )
            batch_metrics[job.identifier] = metrics
            for column in PREDICTION_COLUMNS:
                compare_values(
                    audit,
                    job.scenario,
                    job.identifier,
                    batch_size,
                    f"validation_{column}",
                    serial_predictions[job.identifier][column].to_numpy(
                        np.float64
                    ),
                    prediction[column].to_numpy(np.float64),
                    args.atol,
                    args.rtol,
                )
            for column in METRIC_COLUMNS:
                compare_values(
                    audit,
                    job.scenario,
                    job.identifier,
                    batch_size,
                    f"metric_{column}",
                    np.asarray(
                        [serial_metrics[job.identifier][column]],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [metrics[column]],
                        dtype=np.float64,
                    ),
                    args.atol,
                    args.rtol,
                )

        coarse_jobs = [job for job in jobs if job.scenario == "coarse"]
        if coarse_jobs:
            serial_order = sorted(
                coarse_jobs,
                key=lambda job: (
                    -serial_metrics[job.identifier]["selection_score_j"],
                    job.candidate.identifier,
                ),
            )
            batch_order = sorted(
                coarse_jobs,
                key=lambda job: (
                    -batch_metrics[job.identifier]["selection_score_j"],
                    job.candidate.identifier,
                ),
            )
            rank_passed = [
                job.identifier for job in serial_order
            ] == [job.identifier for job in batch_order]
            audit.append(
                {
                    "scenario": "coarse",
                    "job_id": "all",
                    "model_batch_size": batch_size,
                    "quantity": "candidate_ranking",
                    "gate": comparison_gate("candidate_ranking"),
                    "max_absolute_difference": 0.0,
                    "max_relative_difference": 0.0,
                    "passed": rank_passed,
                }
            )

    audit_table = pd.DataFrame(audit)
    timing_table = pd.DataFrame(timings)
    audit_table, parity = summarize_parity(audit_table)
    audit_table.to_csv(
        output_dir / "fast_backbone_parity_audit.csv",
        index=False,
    )
    timing_table.to_csv(
        output_dir / "fast_backbone_timing.csv",
        index=False,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module": Path(__file__).name,
        "module_version": SCRIPT_VERSION,
        "module_release": SCRIPT_RELEASE,
        **parity,
        "absolute_tolerance": args.atol,
        "relative_tolerance": args.rtol,
        "n_jobs": len(jobs),
        "batch_sizes": batch_sizes,
        "signature_payload": signature_payload,
        "max_absolute_difference": float(
            audit_table["max_absolute_difference"].max()
        ),
        "max_relative_difference": float(
            audit_table["max_relative_difference"].max()
        ),
    }
    (output_dir / "fast_backbone_parity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_parity(parity)
    print(
        "PASS  fast-backbone selection-critical parity: "
        f"{len(jobs)} jobs, batches={batch_sizes}, "
        "histories/predictions/metrics/ranking passed, "
        f"selection_max_abs="
        f"{parity['selection_max_absolute_difference']:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
