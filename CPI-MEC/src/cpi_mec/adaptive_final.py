"""Refit the automatically selected CPI-MEC method on all labels.

This module reads the final automatically selected three-parameter setting,
trains the backbone and head on every binary CPI label, and scores the complete
CID-protein domain. The resulting Final matrix is a production full-label fit,
not an OOF validation matrix. No training-label performance is reported.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2.0.0"
SCRIPT_RELEASE = "full_label_final_20260728"
PROTOCOL_VERSION = "2.0.0"
BASE_DEPENDENCY = "adaptive_training.py"
FIXED_DEPENDENCY = "adaptive_fixed_training.py"
PARAMETER_NAMES = (
    "anchor_weight",
    "graphban_weight",
    "boltz_weight",
)
CSV_FLOAT_FORMAT = "%.9g"


@dataclass(frozen=True)
class Parameters:
    """Store the automatically selected parameters."""

    anchor_weight: float
    graphban_weight: float
    boltz_weight: float

    def as_dict(self) -> dict[str, float]:
        """Return a plain parameter mapping."""
        return {
            name: float(getattr(self, name))
            for name in PARAMETER_NAMES
        }


def sha256(path: Path) -> str:
    """Compute a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    """Load a dependency module from the same directory."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_selected_parameters(path: Path) -> Parameters:
    """Read the single automatic-selection row."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one selected row in {path}")
    return Parameters(
        *(float(rows[0][name]) for name in PARAMETER_NAMES)
    )


def parse_int_values(text: str) -> tuple[int, ...]:
    """Parse non-negative seed offsets."""
    values = tuple(sorted(set(int(value) for value in text.split(","))))
    if not values or values[0] < 0:
        raise ValueError("Seed offsets must be non-empty and non-negative")
    return values


def matrix_frame(
    values: np.ndarray,
    rows: tuple[str, ...],
    columns: tuple[str, ...],
    index_name: str,
) -> pd.DataFrame:
    """Restore a wide matrix with fixed axes."""
    frame = pd.DataFrame(values, index=list(rows), columns=list(columns))
    frame.index.name = index_name
    return frame


def build_acceptance(
    pre_z: np.ndarray,
    row_z_values: np.ndarray,
    expected_shape: tuple[int, int],
    n_labels_used: int,
    n_labels_available: int,
) -> pd.DataFrame:
    """Build Final-matrix acceptance checks."""
    checks = (
        (
            "training_used_all_labels",
            n_labels_used == n_labels_available,
            f"{n_labels_used}/{n_labels_available}",
        ),
        ("pre_z_shape", pre_z.shape == expected_shape, str(pre_z.shape)),
        ("pre_z_finite", bool(np.isfinite(pre_z).all()), int(np.isfinite(pre_z).sum())),
        (
            "row_z_shape",
            row_z_values.shape == expected_shape,
            str(row_z_values.shape),
        ),
        (
            "row_z_finite",
            bool(np.isfinite(row_z_values).all()),
            int(np.isfinite(row_z_values).sum()),
        ),
        (
            "row_z_mean",
            bool(np.max(np.abs(row_z_values.mean(axis=1))) <= 5e-6),
            float(np.max(np.abs(row_z_values.mean(axis=1)))),
        ),
        (
            "row_z_population_sd",
            bool(
                np.max(
                    np.abs(row_z_values.std(axis=1, ddof=0) - 1.0)
                )
                <= 5e-6
            ),
            float(
                np.max(
                    np.abs(row_z_values.std(axis=1, ddof=0) - 1.0)
                )
            ),
        ),
    )
    return pd.DataFrame(
        {
            "check": [check[0] for check in checks],
            "passed": [check[1] for check in checks],
            "value": [check[2] for check in checks],
        }
    )


def main(argv: list[str] | None = None) -> None:
    """Run the all-label Final refit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION} ({SCRIPT_RELEASE})",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-parameters", required=True)
    parser.add_argument("--domain")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed-offsets", default="0")
    parser.add_argument(
        "--final-seed-offset",
        type=int,
        default=9_000_000,
        help="Added to the configured base seed for the all-label fit.",
    )
    parser.add_argument("--backbone-epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--head-epochs", type=int, default=120)
    parser.add_argument("--row-chunk-size", type=int, default=16)
    args = parser.parse_args(argv)

    if min(
        args.backbone_epochs,
        args.steps_per_epoch,
        args.head_epochs,
        args.row_chunk_size,
    ) <= 0:
        raise ValueError("Training and chunk counts must be positive")

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    base_path = script_dir / BASE_DEPENDENCY
    fixed_path = script_dir / FIXED_DEPENDENCY
    config_path = Path(args.config).resolve()
    core_dir = Path(args.core_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    selected_path = Path(args.selected_parameters).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = read_selected_parameters(selected_path)
    seed_offsets = parse_int_values(args.seed_offsets)

    sys.path.insert(0, str(core_dir))
    base = load_module(base_path, "cpi_mec_final_base_dependency")
    fixed = load_module(fixed_path, "cpi_mec_final_fixed_dependency")

    from evaluation import score_domain
    from training import fit_candidate, prepare_context
    from transforms import row_z

    experiment_config = base.build_experiment_config(
        config_path,
        output_dir,
    )
    context = prepare_context(
        experiment_config,
        args.domain,
        args.device,
        backbone_epochs=0,
        steps_per_epoch=args.steps_per_epoch,
    )
    all_rows = np.arange(len(context.labels), dtype=np.int64)
    final_seed = context.settings.seed + args.final_seed_offset
    seed_states = fixed.build_seed_states(
        context,
        base,
        seed_offsets,
        final_seed,
    )

    domain_rows = tuple(context.relations.domain.entity_a)
    domain_columns = tuple(context.relations.domain.entity_b)
    score_sum = np.zeros(
        (len(domain_rows), len(domain_columns)),
        dtype=np.float64,
    )
    fit_rows = []
    backbone_histories = []
    head_histories = []

    for seed_offset in seed_offsets:
        state = seed_states[seed_offset]
        context.standardizer = copy.deepcopy(state["standardizer"])
        print(
            f"START Final full-label fit seed_offset={seed_offset}",
            flush=True,
        )
        backbone_history, backbone_summary = fixed.train_fixed_weight(
            base,
            context,
            state["backbone"],
            state["scorer"],
            all_rows,
            selected.graphban_weight,
            selected.boltz_weight,
            args.backbone_epochs,
            args.steps_per_epoch,
            state["seed"],
        )
        context.backbone_effective_weights = {
            context.models.architecture.auxiliary_feature: (
                selected.graphban_weight
            ),
            context.models.architecture.sparse_features[0]: (
                selected.boltz_weight
            ),
        }
        context.domain_features = None
        base.refresh_frozen_features(context)
        head_seed = state["seed"] + 5000
        fitted = fit_candidate(
            context,
            all_rows,
            learnable_coefficient=True,
            residual_anchor_weight=selected.anchor_weight,
            seed=head_seed,
            epochs=args.head_epochs,
        )
        score_sum += score_domain(
            context,
            fitted,
            row_chunk_size=args.row_chunk_size,
        )
        fit_rows.append(
            {
                "seed_offset": seed_offset,
                "joint_seed": state["seed"],
                "head_seed": head_seed,
                **selected.as_dict(),
                "n_training_labels": len(all_rows),
                "backbone_epochs_completed": len(backbone_history),
                "backbone_selected_epoch": backbone_summary[
                    "selected_epoch"
                ],
                "head_epochs_completed": len(fitted.history),
                "primary_coefficient": fitted.primary_coefficient,
            }
        )
        backbone_history.insert(0, "seed_offset", seed_offset)
        head_history = fitted.history.copy()
        head_history.insert(0, "seed_offset", seed_offset)
        backbone_histories.append(backbone_history)
        head_histories.append(head_history)
        print(
            f"PASS  Final full-label fit seed_offset={seed_offset}",
            flush=True,
        )

    pre_z = (score_sum / len(seed_offsets)).astype(np.float32)
    row_z_values, _, _ = row_z(pre_z, "lower_is_better")
    primary_role = context.models.architecture.primary_input
    contract = context.config["inputs"][primary_role]["contract"]
    index_name = contract.get(
        "entity_a_column",
        contract.get("row_entity_column", "entity_a"),
    )
    matrix_frame(
        pre_z,
        domain_rows,
        domain_columns,
        index_name,
    ).to_csv(
        output_dir / "Final_full_label_fit_preZ_matrix.csv",
        float_format=CSV_FLOAT_FORMAT,
    )
    matrix_frame(
        row_z_values,
        domain_rows,
        domain_columns,
        index_name,
    ).to_csv(
        output_dir / "Final_full_label_fit_rowZ_matrix.csv",
        float_format=CSV_FLOAT_FORMAT,
    )
    pd.DataFrame(fit_rows).to_csv(
        output_dir / "final_fit_summary.csv",
        index=False,
    )
    pd.concat(backbone_histories, ignore_index=True).to_csv(
        output_dir / "backbone_training_history.csv",
        index=False,
    )
    pd.concat(head_histories, ignore_index=True).to_csv(
        output_dir / "head_training_history.csv",
        index=False,
    )
    acceptance = build_acceptance(
        pre_z,
        row_z_values,
        (len(domain_rows), len(domain_columns)),
        len(all_rows),
        len(context.labels),
    )
    acceptance.to_csv(
        output_dir / "matrix_acceptance.csv",
        index=False,
    )
    if not acceptance["passed"].all():
        raise RuntimeError("One or more Final matrix acceptance checks failed")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module": script_path.name,
        "module_version": SCRIPT_VERSION,
        "module_release": SCRIPT_RELEASE,
        "protocol_version": PROTOCOL_VERSION,
        "module_sha256": sha256(script_path),
        "fit_scope": "all available binary CPI labels",
        "validation_role": (
            "Production Final matrix only; training-label metrics are not "
            "reported as validation performance."
        ),
        "selected_parameters_source": str(selected_path),
        "selected_parameters_sha256": sha256(selected_path),
        "selected_parameters": selected.as_dict(),
        "n_training_labels": len(all_rows),
        "domain_shape": [len(domain_rows), len(domain_columns)],
        "configured_base_seed": context.settings.seed,
        "final_seed_offset": args.final_seed_offset,
        "seed_offsets": seed_offsets,
        "backbone_epochs": args.backbone_epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "head_epochs": args.head_epochs,
        "dependencies": {
            base_path.name: sha256(base_path),
            fixed_path.name: sha256(fixed_path),
        },
        "base_config": str(config_path),
        "base_config_sha256": sha256(config_path),
        "generated_config": str(experiment_config),
        "outputs": {
            "pre_z": "Final_full_label_fit_preZ_matrix.csv",
            "row_z": "Final_full_label_fit_rowZ_matrix.csv",
            "fit_summary": "final_fit_summary.csv",
            "acceptance": "matrix_acceptance.csv",
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS  Final matrix: {len(domain_rows)} x {len(domain_columns)}",
        flush=True,
    )
    print(f"PASS  output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
