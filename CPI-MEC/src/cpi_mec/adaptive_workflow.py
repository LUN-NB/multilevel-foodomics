"""Run the formal M0-M1-Final CPI-MEC adaptive workflow.

Each of the 25 M1 outer tasks runs its own complete coarse search and local
continuous TPE. Final is selected independently by five-fold OOF on all labels.
There is no Reference guard, one-SE rule, fixed fallback, Top-8, or boundary
probe. Independent backbones are executed in verified GPU model batches.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PIPELINE_VERSION = "4.3.1"
STAGES = (
    "m1-search",
    "m1",
    "final-search",
    "final",
    "metrics",
)
MODULES = {
    "coarse": "adaptive_coarse_search.py",
    "continuous": "adaptive_continuous_search.py",
    "m1": "adaptive_oof.py",
    "final": "adaptive_final.py",
    "metrics": "adaptive_metrics.py",
}


def utc_now() -> str:
    """Return current UTC time."""
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    """Compute a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    """Read a YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def verify_hashes(
    root: Path,
    expected: dict[str, str],
    label: str,
) -> None:
    """Verify frozen files."""
    mismatches = []
    for relative, expected_hash in expected.items():
        path = root / relative
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected_hash:
            mismatches.append(
                f"{relative}: expected={expected_hash}, actual={actual}"
            )
    if mismatches:
        raise RuntimeError(
            f"{label} verification failed:\n" + "\n".join(mismatches)
        )
    print(f"PASS  {label}: {len(expected)} files", flush=True)


def runtime_versions() -> dict[str, str]:
    """Read key runtime versions."""
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "optuna",
        "PyYAML",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    return versions


def verify_runtime(config: dict[str, Any]) -> dict[str, str]:
    """Verify TPE-relevant runtime."""
    versions = runtime_versions()
    python_prefix = str(config["runtime"]["expected_python_prefix"])
    optuna_version = str(config["runtime"]["expected_optuna_version"])
    if not versions["python"].startswith(python_prefix):
        raise RuntimeError(
            f"Python {python_prefix}.x is required; found {versions['python']}"
        )
    if versions["optuna"] != optuna_version:
        raise RuntimeError(
            f"Optuna {optuna_version} is required; "
            f"found {versions['optuna']}"
        )
    print(
        f"PASS  runtime: Python {versions['python']}, "
        f"Optuna {versions['optuna']}",
        flush=True,
    )
    return versions


def run_module(
    core_dir: Path,
    module: str,
    arguments: list[str],
) -> None:
    """Run one module in a subprocess."""
    path = core_dir / MODULES[module]
    print(f"RUN   {module}: {path.name}", flush=True)
    subprocess.run([sys.executable, str(path), *arguments], check=True)


def stage_range(start: str, stop: str) -> tuple[str, ...]:
    """Resolve a contiguous stage range."""
    start_index = STAGES.index(start)
    stop_index = STAGES.index(stop)
    if start_index > stop_index:
        raise ValueError("--start-stage must not follow --stop-stage")
    return STAGES[start_index : stop_index + 1]


def write_manifest(path: Path, state: dict[str, Any]) -> None:
    """Write pipeline state."""
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def training_arguments(training: dict[str, Any]) -> list[str]:
    """Build fixed training arguments."""
    return [
        "--backbone-epochs",
        str(training["backbone_epochs"]),
        "--steps-per-epoch",
        str(training["steps_per_epoch"]),
        "--head-epochs",
        str(training["head_epochs"]),
    ]


def search_arguments(
    model_config: Path,
    core_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    scope: str,
    repeat: int,
    outer_fold: int | None,
    device: str,
    search: dict[str, Any],
    training: dict[str, Any],
) -> list[str]:
    """Build coarse-search arguments."""
    arguments = [
        "--config",
        str(model_config),
        "--core-dir",
        str(core_dir),
        "--splits-dir",
        str(splits_dir),
        "--output-dir",
        str(output_dir),
        "--scope",
        scope,
        "--repeat",
        str(repeat),
        "--inner-folds",
        search["inner_folds"],
        "--seed-offsets",
        search["seed_offsets"],
        "--anchor-values",
        search["anchor_values"],
        "--graphban-values",
        search["graphban_values"],
        "--boltz-values",
        search["boltz_values"],
        "--model-batch-size",
        str(search["model_batch_size"]),
        "--device",
        device,
    ]
    if outer_fold is not None:
        arguments.extend(["--outer-fold", str(outer_fold)])
    return arguments + training_arguments(training)


def run_search_pair(
    model_config: Path,
    core_dir: Path,
    splits_dir: Path,
    search_dir: Path,
    scope: str,
    repeat: int,
    outer_fold: int | None,
    device: str,
    search: dict[str, Any],
    continuous: dict[str, Any],
    training: dict[str, Any],
) -> Path:
    """Run one search pair."""
    coarse_dir = search_dir / "01_coarse"
    continuous_dir = search_dir / "02_continuous"
    run_module(
        core_dir,
        "coarse",
        search_arguments(
            model_config,
            core_dir,
            splits_dir,
            coarse_dir,
            scope,
            repeat,
            outer_fold,
            device,
            search,
            training,
        ),
    )
    arguments = [
        "--config",
        str(model_config),
        "--core-dir",
        str(core_dir),
        "--splits-dir",
        str(splits_dir),
        "--coarse-output-dir",
        str(coarse_dir),
        "--output-dir",
        str(continuous_dir),
        "--scope",
        scope,
        "--repeat",
        str(repeat),
        "--inner-folds",
        search["inner_folds"],
        "--seed-offsets",
        search["seed_offsets"],
        "--n-trials",
        str(continuous["trials"]),
        "--n-startup-trials",
        str(continuous["startup_trials"]),
        "--n-ei-candidates",
        str(continuous["ei_candidates"]),
        "--sampler-seed",
        str(continuous["sampler_seed"]),
        "--study-name",
        continuous["study_name"],
        "--model-batch-size",
        str(continuous["model_batch_size"]),
        "--device",
        device,
    ]
    if outer_fold is not None:
        arguments.extend(["--outer-fold", str(outer_fold)])
    run_module(
        core_dir,
        "continuous",
        arguments + training_arguments(training),
    )
    return continuous_dir / "selected_parameters.csv"


def m1_keys(
    repeats: int,
    folds: int,
    repeat: int | None,
    outer_fold: int | None,
) -> list[tuple[int, int]]:
    """Return requested M1 outer folds."""
    if repeat is None:
        return [
            (repeat_id, fold)
            for repeat_id in range(1, repeats + 1)
            for fold in range(1, folds + 1)
        ]
    if outer_fold is None:
        return [(repeat, fold) for fold in range(1, folds + 1)]
    return [(repeat, outer_fold)]


def collect_m1_parameters(
    search_root: Path,
    repeats: int,
    folds: int,
    output_path: Path,
) -> pd.DataFrame:
    """Collect completed fold parameters."""
    tables = []
    for repeat in range(1, repeats + 1):
        for fold in range(1, folds + 1):
            path = (
                search_root
                / f"repeat_{repeat:02d}"
                / f"outer_{fold:02d}"
                / "02_continuous"
                / "selected_parameters.csv"
            )
            if path.is_file():
                tables.append(pd.read_csv(path))
    selected = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame()
    )
    if not selected.empty:
        selected.sort_values(["repeat", "outer_fold"]).to_csv(
            output_path,
            index=False,
        )
    return selected


def main(argv: list[str] | None = None) -> None:
    """Run the formal adaptive pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument(
        "--pipeline-config",
        default="configs/adaptive_pipeline.yaml",
    )
    parser.add_argument("--device")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--start-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--stop-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--verify-package", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.outer_fold is not None and args.repeat is None:
        parser.error("--outer-fold requires --repeat")

    root = Path(args.project_root).resolve()
    pipeline_config = Path(args.pipeline_config)
    if not pipeline_config.is_absolute():
        pipeline_config = root / pipeline_config
    config = load_config(pipeline_config)
    paths = config["paths"]
    verify_hashes(
        root,
        config["expected_pipeline_hashes"],
        "integrated pipeline modules",
    )
    if args.verify_package:
        return

    splits_dir = root / paths["splits_dir"]
    verify_hashes(
        splits_dir,
        config["expected_split_hashes"],
        "materialized CV splits",
    )
    verify_hashes(
        root,
        config["expected_input_hashes"],
        "formal model inputs",
    )
    versions = verify_runtime(config)
    if args.verify_only:
        return

    active = stage_range(args.start_stage, args.stop_stage)
    device = args.device or str(config["runtime"]["device"])
    model_config = root / paths["model_config"]
    core_dir = root / paths["core_dir"]
    output_dir = root / paths["output_dir"]
    m1_search_root = output_dir / "01_m1_search"
    m1_output = output_dir / "02_m1_matrix"
    final_search_root = output_dir / "03_final_search"
    final_output = output_dir / "04_final_matrix"
    metrics_output = output_dir / "05_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pipeline_run_manifest.json"
    state: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "protocol": config["protocol"],
        "started_utc": utc_now(),
        "status": "running",
        "runtime_versions": versions,
        "device": device,
        "requested_stages": active,
        "stages": [],
    }
    write_manifest(manifest_path, state)

    search = config["search"]
    continuous = config["continuous_tpe"]
    training = config["training"]
    m1 = config["m1"]
    final_search = config["final_search"]
    final = config["final"]
    evaluation = config["evaluation"]
    selected_m1_path = m1_search_root / "selected_parameters_by_outer_fold.csv"
    selected_final_path = final_search_root / "selected_parameters.csv"

    try:
        if "m1-search" in active:
            keys = m1_keys(
                int(m1["outer_repeats"]),
                int(m1["outer_folds"]),
                args.repeat,
                args.outer_fold,
            )
            for repeat, outer_fold in keys:
                fold_dir = (
                    m1_search_root
                    / f"repeat_{repeat:02d}"
                    / f"outer_{outer_fold:02d}"
                )
                run_search_pair(
                    model_config,
                    core_dir,
                    splits_dir,
                    fold_dir,
                    "nested",
                    repeat,
                    outer_fold,
                    device,
                    search,
                    continuous,
                    training,
                )
                selected = collect_m1_parameters(
                    m1_search_root,
                    int(m1["outer_repeats"]),
                    int(m1["outer_folds"]),
                    selected_m1_path,
                )
                print(
                    f"PROGRESS  M1 selected folds: "
                    f"{len(selected)}/{int(m1['outer_repeats']) * int(m1['outer_folds'])}",
                    flush=True,
                )
            state["stages"].append(
                {
                    "stage": "m1-search",
                    "completed_utc": utc_now(),
                    "selected_parameters": str(selected_m1_path),
                }
            )
            write_manifest(manifest_path, state)

        if "m1" in active:
            arguments = [
                "--config",
                str(model_config),
                "--core-dir",
                str(core_dir),
                "--splits-dir",
                str(splits_dir),
                "--selected-parameters",
                str(selected_m1_path),
                "--output-dir",
                str(m1_output),
                "--seed-offsets",
                m1["seed_offsets"],
                "--row-chunk-size",
                str(m1["row_chunk_size"]),
                "--global-top-fraction",
                str(evaluation["global_top_fraction"]),
                "--target-top-k",
                str(evaluation["target_top_k"]),
                "--device",
                device,
            ]
            if args.repeat is not None:
                arguments.extend(["--repeat", str(args.repeat)])
            if args.outer_fold is not None:
                arguments.extend(["--outer-fold", str(args.outer_fold)])
            run_module(
                core_dir,
                "m1",
                arguments + training_arguments(training),
            )
            state["stages"].append(
                {"stage": "m1", "completed_utc": utc_now()}
            )
            write_manifest(manifest_path, state)

        if "final-search" in active:
            final_selected_source = run_search_pair(
                model_config,
                core_dir,
                splits_dir,
                final_search_root,
                "full",
                int(final_search["split_repeat"]),
                None,
                device,
                search,
                continuous,
                training,
            )
            pd.read_csv(final_selected_source).to_csv(
                selected_final_path,
                index=False,
            )
            state["stages"].append(
                {
                    "stage": "final-search",
                    "completed_utc": utc_now(),
                    "selected_parameters": str(selected_final_path),
                }
            )
            write_manifest(manifest_path, state)

        if "final" in active:
            run_module(
                core_dir,
                "final",
                [
                    "--config",
                    str(model_config),
                    "--core-dir",
                    str(core_dir),
                    "--selected-parameters",
                    str(selected_final_path),
                    "--output-dir",
                    str(final_output),
                    "--seed-offsets",
                    final["seed_offsets"],
                    "--final-seed-offset",
                    str(final["final_seed_offset"]),
                    "--row-chunk-size",
                    str(final["row_chunk_size"]),
                    "--device",
                    device,
                    *training_arguments(training),
                ],
            )
            state["stages"].append(
                {"stage": "final", "completed_utc": utc_now()}
            )
            write_manifest(manifest_path, state)

        if "metrics" in active:
            run_module(
                core_dir,
                "metrics",
                [
                    "--matrix",
                    f"M0={root / evaluation['m0_matrix']}",
                    "--matrix",
                    (
                        "M1="
                        f"{m1_output / 'M1_nested_CV_ensemble_preZ_matrix.csv'}"
                    ),
                    "--matrix",
                    (
                        "Final="
                        f"{final_output / 'Final_full_label_fit_preZ_matrix.csv'}"
                    ),
                    "--m1-oof-summary",
                    str(
                        m1_output
                        / "outer_label_oof_metric_summary.csv"
                    ),
                    "--labels",
                    str(root / evaluation["labels"]),
                    "--descriptors",
                    str(root / evaluation["descriptors"]),
                    "--output-dir",
                    str(metrics_output),
                    "--folds",
                    str(evaluation["folds"]),
                    "--repeats",
                    str(evaluation["repeats"]),
                    "--seed",
                    str(evaluation["seed"]),
                    "--global-top-fraction",
                    str(evaluation["global_top_fraction"]),
                    "--target-top-k",
                    str(evaluation["target_top_k"]),
                ],
            )
            state["stages"].append(
                {"stage": "metrics", "completed_utc": utc_now()}
            )

        state["status"] = "completed"
        state["completed_utc"] = utc_now()
        state["outputs"] = {
            "m0": str(root / evaluation["m0_matrix"]),
            "m1": str(
                m1_output / "M1_nested_CV_ensemble_preZ_matrix.csv"
            ),
            "m1_strict_oof_metrics": str(
                m1_output / "outer_label_oof_metric_summary.csv"
            ),
            "final": str(
                final_output / "Final_full_label_fit_preZ_matrix.csv"
            ),
            "metrics": str(metrics_output),
        }
        write_manifest(manifest_path, state)
    except Exception as error:
        state["status"] = "failed"
        state["failed_utc"] = utc_now()
        state["error"] = f"{type(error).__name__}: {error}"
        write_manifest(manifest_path, state)
        raise

    print(f"PASS  Adaptive pipeline output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
