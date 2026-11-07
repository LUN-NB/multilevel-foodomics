"""Shared training and scoring helpers for CPI-MEC parameter search.

The coarse grid and continuous TPE share one trainer, inner-OOF metrics, and
exact-best rule.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.2.0"
SCRIPT_RELEASE = "complete_gpu_trial_batch_25_outer_tasks_20260729"
BASE_DEPENDENCY = "adaptive_training.py"
FIXED_DEPENDENCY = "adaptive_fixed_training.py"
FAST_DEPENDENCY = "adaptive_fast_heads.py"
FAST_BACKBONE_DEPENDENCY = "adaptive_fast_backbones.py"
FAST_TRIAL_DEPENDENCY = "adaptive_fast_trials.py"
PARAMETER_NAMES = (
    "anchor_weight",
    "graphban_weight",
    "boltz_weight",
)
LOG2 = float(np.log(2.0))


@dataclass(frozen=True)
class Candidate:
    """Store one three-parameter candidate."""

    anchor: float
    graphban: float
    boltz: float

    @property
    def identifier(self) -> str:
        """Build a stable candidate identifier."""
        values = [
            f"{value:.12g}".replace(".", "p")
            for value in (self.anchor, self.graphban, self.boltz)
        ]
        return f"a{values[0]}_g{values[1]}_b{values[2]}"

    @property
    def eligible(self) -> bool:
        """Test formal eligibility."""
        return self.anchor > 0 and self.graphban > 0 and self.boltz > 0

    def as_dict(self) -> dict[str, float]:
        """Return the parameter mapping."""
        return {
            "anchor_weight": self.anchor,
            "graphban_weight": self.graphban,
            "boltz_weight": self.boltz,
        }


@dataclass(frozen=True)
class SearchPlan:
    """Store one search plan."""

    scope: str
    repeat: int
    outer_fold: int | None
    inner_folds: tuple[int, ...]
    seed_offsets: tuple[int, ...]
    backbone_epochs: int
    steps_per_epoch: int
    head_epochs: int
    model_batch_size: int


@dataclass(frozen=True)
class SearchRuntime:
    """Store initialized search dependencies."""

    runner: "ExperimentRunner"
    base: Any
    fixed: Any
    context: Any
    splits: Any
    generated_config: Path
    base_path: Path
    fixed_path: Path
    fast_path: Path
    fast_backbone_path: Path
    fast_trial_path: Path
    seed_states: dict[int, dict[str, Any]]


def sha256(path: Path) -> str:
    """Compute a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    """Load and register a module from a file."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_float_values(text: str) -> tuple[float, ...]:
    """Parse ordered non-negative floats."""
    values = tuple(sorted(set(float(value) for value in text.split(","))))
    if not values or values[0] < 0:
        raise ValueError("Parameter grids must be non-empty and non-negative")
    return values


def parse_int_values(text: str) -> tuple[int, ...]:
    """Parse ordered non-negative integers."""
    values = tuple(sorted(set(int(value) for value in text.split(","))))
    if not values or values[0] < 0:
        raise ValueError("Fold and seed lists must be non-empty")
    return values


def search_signature(payload: dict[str, Any]) -> str:
    """Build a stable search signature."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_progress(path: Path, signature: str) -> pd.DataFrame:
    """Load a resumable table."""
    if not path.is_file():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if set(table["search_signature"].astype(str)) != {signature}:
        raise ValueError(f"{path.name} belongs to another search")
    return table


def append_progress(path: Path, table: pd.DataFrame) -> None:
    """Append one progress batch."""
    if table.empty:
        return
    table.to_csv(
        path,
        mode="a",
        header=not path.is_file(),
        index=False,
    )


def implementation_hashes(core_dir: Path) -> dict[str, str]:
    """Hash search implementation."""
    script_dir = Path(__file__).resolve().parent
    paths = [
        script_dir / Path(__file__).name,
        script_dir / BASE_DEPENDENCY,
        script_dir / FIXED_DEPENDENCY,
        script_dir / FAST_DEPENDENCY,
        script_dir / FAST_BACKBONE_DEPENDENCY,
        script_dir / FAST_TRIAL_DEPENDENCY,
        core_dir / "models.py",
        core_dir / "training.py",
        core_dir / "evaluation.py",
    ]
    return {path.name: sha256(path) for path in paths}


def add_skill_scores(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add G, W, and J."""
    prevalence = float(metrics["positive_prevalence"])
    global_skill = (
        float(metrics["ranking_pooled_auprc"]) - prevalence
    ) / (1.0 - prevalence)
    within_skill = (
        1.0
        - float(metrics["pairwise_logistic_loss"]) / LOG2
    )
    return {
        **metrics,
        "global_skill": global_skill,
        "within_skill": within_skill,
        "selection_score_j": 0.5 * global_skill + 0.5 * within_skill,
    }


def label_oof_top_k_metrics(
    frame: pd.DataFrame,
    top_k: int = 100,
) -> dict[str, float | int]:
    """Compute label-domain Top-k audits."""
    ranked = frame.sort_values(
        "ranking_score",
        ascending=False,
        kind="stable",
    )
    top = ranked.head(min(top_k, len(ranked)))
    n_positive = int((frame["label"] == 1).sum())
    recall = int((top["label"] == 1).sum()) / n_positive
    success = []
    for _, part in frame.groupby("entity_a", sort=False):
        positives = part["label"] == 1
        if positives.any():
            within = part.sort_values(
                "ranking_score",
                ascending=False,
                kind="stable",
            ).head(min(top_k, len(part)))
            success.append(bool((within["label"] == 1).any()))
    return {
        f"label_oof_global_recall_top{top_k}": recall,
        f"label_oof_success_top{top_k}": float(np.mean(success)),
        f"label_oof_n_cid_success_top{top_k}": int(np.sum(success)),
    }


def candidate_from_row(row: pd.Series | Any) -> Candidate:
    """Restore a candidate from a table row."""
    return Candidate(
        float(row["anchor_weight"]),
        float(row["graphban_weight"]),
        float(row["boltz_weight"]),
    )


def candidate_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidates across seeds."""
    keys = [
        "candidate_id",
        "anchor_weight",
        "graphban_weight",
        "boltz_weight",
    ]
    numeric = [
        column
        for column in seed_metrics.select_dtypes(include=[np.number]).columns
        if column not in {*PARAMETER_NAMES, "seed_offset"}
    ]
    rows = []
    for values, part in seed_metrics.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        row["n_seeds"] = int(part["seed_offset"].nunique())
        for column in numeric:
            row[f"mean_seed_pooled_{column}"] = float(part[column].mean())
            row[f"sd_seed_pooled_{column}"] = float(
                part[column].std(ddof=1)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def select_exact_best(
    summary: pd.DataFrame,
) -> tuple[Candidate, pd.DataFrame]:
    """Select the exact J maximum."""
    table = summary.copy()
    if "eligible_for_selection" not in table:
        table["eligible_for_selection"] = (
            (table["anchor_weight"] > 0)
            & (table["graphban_weight"] > 0)
            & (table["boltz_weight"] > 0)
        )
    eligible = table.loc[table["eligible_for_selection"]].sort_values(
        [
            "mean_seed_pooled_selection_score_j",
            "candidate_id",
        ],
        ascending=[False, True],
        kind="stable",
    )
    selected = candidate_from_row(eligible.iloc[0])
    table["selected"] = table["candidate_id"] == selected.identifier
    rank = pd.Series(
        np.arange(1, len(eligible) + 1),
        index=eligible.index,
        dtype="Int64",
    )
    table["objective_rank"] = rank
    return selected, table


def pareto_front(summary: pd.DataFrame) -> pd.DataFrame:
    """Extract the G-W Pareto front."""
    eligible = summary.loc[summary["eligible_for_selection"]].copy()
    global_values = eligible[
        "mean_seed_pooled_global_skill"
    ].to_numpy()
    within_values = eligible[
        "mean_seed_pooled_within_skill"
    ].to_numpy()
    dominated = np.zeros(len(eligible), dtype=bool)
    for index in range(len(eligible)):
        dominated[index] = bool(
            np.any(
                (global_values >= global_values[index])
                & (within_values >= within_values[index])
                & (
                    (global_values > global_values[index])
                    | (within_values > within_values[index])
                )
            )
        )
    return eligible.loc[~dominated].sort_values(
        "mean_seed_pooled_selection_score_j",
        ascending=False,
    )


def local_interval(
    value: float,
    positive_grid: tuple[float, ...],
) -> tuple[float, float, str]:
    """Derive one local interval."""
    grid = np.asarray(positive_grid, dtype=float)
    index = int(np.flatnonzero(np.isclose(grid, value))[0])
    if index == 0:
        return value * value / grid[1], float(grid[1]), "lower_boundary"
    if index == len(grid) - 1:
        return float(grid[-2]), value * value / grid[-2], "upper_boundary"
    return float(grid[index - 1]), float(grid[index + 1]), "interior"


def local_bounds(
    selected: Candidate,
    anchor_values: tuple[float, ...],
    graphban_values: tuple[float, ...],
    boltz_values: tuple[float, ...],
) -> pd.DataFrame:
    """Build dynamic local bounds."""
    rows = []
    grids = {
        "anchor_weight": tuple(value for value in anchor_values if value > 0),
        "graphban_weight": tuple(
            value for value in graphban_values if value > 0
        ),
        "boltz_weight": tuple(value for value in boltz_values if value > 0),
    }
    for name, value in selected.as_dict().items():
        low, high, position = local_interval(value, grids[name])
        rows.append(
            {
                "parameter": name,
                "coarse_value": value,
                "coarse_position": position,
                "local_low": low,
                "local_high": high,
                "sampling_scale": "log",
            }
        )
    return pd.DataFrame(rows)


def grid_candidates(
    anchor_values: tuple[float, ...],
    graphban_values: tuple[float, ...],
    boltz_values: tuple[float, ...],
) -> list[Candidate]:
    """Enumerate the full coarse grid."""
    return [
        Candidate(anchor, graphban, boltz)
        for anchor in anchor_values
        for graphban in graphban_values
        for boltz in boltz_values
    ]


def backbone_groups(
    candidates: Iterable[Candidate],
    trial_numbers: Iterable[int],
) -> list[list[tuple[Candidate, int]]]:
    """Group candidates by G/B in order."""
    grouped: dict[
        tuple[float, float],
        list[tuple[Candidate, int]],
    ] = {}
    for candidate, trial_number in zip(candidates, trial_numbers):
        grouped.setdefault(
            (candidate.graphban, candidate.boltz),
            [],
        ).append((candidate, trial_number))
    return list(grouped.values())


def model_batches(values: list[Any], batch_size: int) -> list[list[Any]]:
    """Split ordered model jobs."""
    return [
        values[start : start + batch_size]
        for start in range(0, len(values), batch_size)
    ]


class ExperimentRunner:
    """Batch backbones and create OOF."""

    def __init__(
        self,
        base,
        fixed,
        context,
        splits,
        plan: SearchPlan,
        output_dir: Path,
        signature: str,
        seed_states: dict[int, dict[str, Any]],
        fast_backbone,
        fast_trial,
    ) -> None:
        self.base = base
        self.fixed = fixed
        self.fast_backbone = fast_backbone
        self.fast_trial = fast_trial
        self.context = context
        self.splits = splits
        self.plan = plan
        self.output_dir = output_dir
        self.signature = signature
        self.seed_states = seed_states
        if plan.model_batch_size <= 0:
            raise ValueError("model_batch_size must be positive")

        self.runs_path = output_dir / "inner_runs.csv"
        self.predictions_path = output_dir / "inner_oof_predictions.csv"
        self.backbone_path = output_dir / "backbone_fit_summary.csv"
        self.runs = load_progress(self.runs_path, signature)
        self.predictions = load_progress(self.predictions_path, signature)
        self.backbones = load_progress(self.backbone_path, signature)
        self._normalize_progress()
        self.backbone_data = fast_backbone.prepare_backbone_device_data(
            context
        )
        self._fold_cache = {
            inner_fold: self._resolve_fold_rows(inner_fold)
            for inner_fold in plan.inner_folds
        }
        self._supervision_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            Any,
        ] = {}
        self._schedule_cache: dict[
            int,
            tuple[Any, ...],
        ] = {}

    def save_progress(self) -> None:
        """Compact resumable progress."""
        for table, path in (
            (self.runs, self.runs_path),
            (self.predictions, self.predictions_path),
            (self.backbones, self.backbone_path),
        ):
            if not table.empty:
                table.to_csv(path, index=False)

    def _normalize_progress(self) -> None:
        """Clean orphan and duplicate rows."""
        if not self.runs.empty:
            self.runs = self.runs.drop_duplicates(
                ["candidate_id", "inner_fold", "seed_offset"],
                keep="last",
            ).reset_index(drop=True)
        completed = self.completed_keys()
        if not self.predictions.empty:
            key = list(
                zip(
                    self.predictions["candidate_id"].astype(str),
                    self.predictions["inner_fold"].astype(int),
                    self.predictions["seed_offset"].astype(int),
                )
            )
            self.predictions = self.predictions.loc[
                [value in completed for value in key]
            ].drop_duplicates(
                [
                    "candidate_id",
                    "inner_fold",
                    "seed_offset",
                    "relation_row",
                ],
                keep="last",
            ).reset_index(drop=True)
        if not self.backbones.empty:
            self.backbones = self.backbones.drop_duplicates(
                [
                    "inner_fold",
                    "seed_offset",
                    "graphban_weight",
                    "boltz_weight",
                ],
                keep="last",
            ).reset_index(drop=True)

    def _commit_backbones(self, rows: list[dict[str, Any]]) -> None:
        """Append backbone summaries."""
        if not rows:
            return
        frame = pd.DataFrame(rows)
        append_progress(self.backbone_path, frame)
        self.backbones = pd.concat(
            [self.backbones, frame],
            ignore_index=True,
        )

    def _commit_heads(
        self,
        run_rows: list[dict[str, Any]],
        prediction_parts: list[pd.DataFrame],
    ) -> None:
        """Write predictions before commit keys."""
        predictions = pd.concat(prediction_parts, ignore_index=True)
        runs = pd.DataFrame(run_rows)
        append_progress(self.predictions_path, predictions)
        append_progress(self.runs_path, runs)
        self.predictions = pd.concat(
            [self.predictions, predictions],
            ignore_index=True,
        )
        self.runs = pd.concat(
            [self.runs, runs],
            ignore_index=True,
        )

    def completed_keys(self) -> set[tuple[str, int, int]]:
        """Return completed fit keys."""
        if self.runs.empty:
            return set()
        return set(
            zip(
                self.runs["candidate_id"].astype(str),
                self.runs["inner_fold"].astype(int),
                self.runs["seed_offset"].astype(int),
            )
        )

    def _resolve_fold_rows(
        self,
        inner_fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return CV rows."""
        if self.plan.scope == "nested":
            outer_fold = self.plan.outer_fold
            assert outer_fold is not None
            train = self.splits.inner_train_rows(
                self.plan.repeat,
                outer_fold,
                inner_fold,
            )
            validation = self.splits.inner_validation_rows(
                self.plan.repeat,
                outer_fold,
                inner_fold,
            )
            return train, validation
        assignment = self.splits.outer.loc[
            self.splits.outer["repeat"] == self.plan.repeat
        ]
        validation = assignment.loc[
            assignment["outer_fold"] == inner_fold,
            "relation_row",
        ].to_numpy(np.int64)
        train = assignment.loc[
            assignment["outer_fold"] != inner_fold,
            "relation_row",
        ].to_numpy(np.int64)
        return train, validation

    def fold_rows(self, inner_fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Return cached inner-fold row indices."""
        return self._fold_cache[inner_fold]

    def _cached_supervision(
        self,
        jobs: list[Any],
        inner_folds: list[int],
        seed_offsets: list[int],
    ) -> Any:
        """Reuse trial-invariant supervision."""
        key = (tuple(inner_folds), tuple(seed_offsets))
        if key not in self._supervision_cache:
            self._supervision_cache[key] = (
                self.fast_backbone.prepare_supervision(
                    self.context,
                    jobs,
                )
            )
        return self._supervision_cache[key]

    def _cached_schedules(
        self,
        seeds: list[int],
    ) -> dict[int, tuple[Any, ...]]:
        """Reuse trial-invariant schedules."""
        for seed in set(seeds):
            if seed in self._schedule_cache:
                continue
            self._schedule_cache[seed] = self.fast_backbone.full_schedule(
                self.context,
                self.backbone_data,
                seed,
                self.plan.backbone_epochs,
                self.plan.steps_per_epoch,
            )
        return self._schedule_cache

    def train_backbone_batch(
        self,
        candidates: list[Candidate],
        train_rows: list[np.ndarray],
        inner_folds: list[int],
        seed_offsets: list[int],
    ) -> list[Any]:
        """Train one backbone model batch."""
        states = [self.seed_states[value] for value in seed_offsets]
        jobs = [
            self.fast_backbone.BackboneJob(
                rows,
                candidate.graphban,
                candidate.boltz,
            )
            for candidate, rows in zip(candidates, train_rows)
        ]
        results = self.fast_backbone.fit_fixed_weight_batch(
            self.context,
            self.backbone_data,
            [state["backbone"] for state in states],
            [state["scorer"] for state in states],
            [state["standardizer"] for state in states],
            jobs,
            self.plan.backbone_epochs,
            self.plan.steps_per_epoch,
            [state["seed"] for state in states],
            self._cached_supervision(jobs, inner_folds, seed_offsets),
            self._cached_schedules([state["seed"] for state in states]),
        )
        logged = set()
        if not self.backbones.empty:
            logged = set(
                zip(
                    self.backbones["inner_fold"].astype(int),
                    self.backbones["seed_offset"].astype(int),
                    self.backbones["graphban_weight"].astype(float),
                    self.backbones["boltz_weight"].astype(float),
                )
            )
        rows = []
        for candidate, inner_fold, seed_offset, result in zip(
            candidates,
            inner_folds,
            seed_offsets,
            results,
        ):
            key = (
                inner_fold,
                seed_offset,
                candidate.graphban,
                candidate.boltz,
            )
            if key in logged:
                continue
            final = result.history.iloc[-1].to_dict()
            rows.append(
                {
                    "search_signature": self.signature,
                    "inner_fold": inner_fold,
                    "seed_offset": seed_offset,
                    "graphban_weight": candidate.graphban,
                    "boltz_weight": candidate.boltz,
                    **result.summary,
                    **{
                        f"final_{name}": value
                        for name, value in final.items()
                        if name != "epoch"
                    },
                }
            )
        self._commit_backbones(rows)
        return results

    def train_backbone(
        self,
        candidate: Candidate,
        train_rows: np.ndarray,
        inner_fold: int,
        seed_offset: int,
    ) -> dict[str, Any]:
        """Train one backbone for validators."""
        return self.train_backbone_batch(
            [candidate],
            [train_rows],
            [inner_fold],
            [seed_offset],
        )[0].state

    def activate_backbone(
        self,
        state: dict[str, Any],
        candidate: Candidate,
    ) -> None:
        """Activate one backbone."""
        self.context.models.backbone.load_state_dict(state)
        for parameter in self.context.models.backbone.parameters():
            parameter.requires_grad_(False)
        self.context.backbone_effective_weights = {
            self.context.models.architecture.auxiliary_feature: (
                candidate.graphban
            ),
            self.context.models.architecture.sparse_features[0]: (
                candidate.boltz
            ),
        }
        self.base.refresh_frozen_features(self.context)

    def run_head_batch(
        self,
        candidates: list[Candidate],
        backbone_state: dict[str, Any],
        train_rows: np.ndarray,
        validation_rows: np.ndarray,
        inner_fold: int,
        seed_offset: int,
        phase: str,
        trial_numbers: list[int],
    ) -> None:
        """Fit heads sharing one backbone."""
        if not candidates or len(candidates) != len(trial_numbers):
            return
        completed = self.completed_keys()
        pending = [
            (candidate, trial_number)
            for candidate, trial_number in zip(candidates, trial_numbers)
            if (
                candidate.identifier,
                inner_fold,
                seed_offset,
            )
            not in completed
        ]
        if not pending:
            return
        representative = pending[0][0]
        print(
            f"START {phase} heads={len(pending)} "
            f"g={representative.graphban:.12g} "
            f"b={representative.boltz:.12g} "
            f"inner={inner_fold} seed={seed_offset}",
            flush=True,
        )
        refresh_started = time.perf_counter()
        self.context.standardizer = copy.deepcopy(
            self.seed_states[seed_offset]["standardizer"]
        )
        self.activate_backbone(backbone_state, representative)
        refresh_seconds = time.perf_counter() - refresh_started

        head_seed = self.seed_states[seed_offset]["seed"] + 5000
        for start in range(
            0,
            len(pending),
            self.plan.model_batch_size,
        ):
            batch = pending[start : start + self.plan.model_batch_size]
            from adaptive_fast_heads import fit_candidate_batch

            head_started = time.perf_counter()
            fitted_batch = fit_candidate_batch(
                self.context,
                train_rows,
                [candidate.anchor for candidate, _ in batch],
                learnable_coefficient=True,
                seed=head_seed,
                epochs=self.plan.head_epochs,
            )
            head_seconds = time.perf_counter() - head_started
            run_rows = []
            prediction_parts = []
            for (candidate, trial_number), fitted in zip(
                batch,
                fitted_batch,
            ):
                evaluation_started = time.perf_counter()
                prediction, metrics = self.base.evaluate_strategy(
                    self.context,
                    fitted,
                    validation_rows,
                    candidate.identifier,
                )
                evaluation_seconds = (
                    time.perf_counter() - evaluation_started
                )
                prediction = prediction.rename(
                    columns={"strategy": "candidate_id"}
                )
                prediction.insert(1, "search_signature", self.signature)
                prediction.insert(2, "phase", phase)
                prediction.insert(3, "trial_number", trial_number)
                prediction.insert(4, "inner_fold", inner_fold)
                prediction.insert(5, "seed_offset", seed_offset)
                for position, (name, value) in enumerate(
                    candidate.as_dict().items(),
                    start=6,
                ):
                    prediction.insert(position, name, value)
                final_head = fitted.history.iloc[-1].to_dict()
                run_rows.append(
                    {
                        "search_signature": self.signature,
                        "phase": phase,
                        "trial_number": trial_number,
                        "candidate_id": candidate.identifier,
                        **candidate.as_dict(),
                        "inner_fold": inner_fold,
                        "seed_offset": seed_offset,
                        "n_train": len(train_rows),
                        "n_validation": len(validation_rows),
                        "head_seed": head_seed,
                        "head_execution": "fast_model_batch",
                        "head_model_batch_size": len(batch),
                        "feature_refresh_seconds": refresh_seconds,
                        "head_batch_elapsed_seconds": head_seconds,
                        "seconds_per_head": head_seconds / len(batch),
                        "head_evaluation_seconds": evaluation_seconds,
                        "primary_coefficient": fitted.primary_coefficient,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key != "strategy"
                        },
                        **{
                            f"head_final_{name}": value
                            for name, value in final_head.items()
                            if name != "epoch"
                        },
                    }
                )
                prediction_parts.append(prediction)
                print(
                    f"PASS  {candidate.identifier}: "
                    f"AUPRC={metrics['ranking_pooled_auprc']:.6f}",
                    flush=True,
                )
            self._commit_heads(run_rows, prediction_parts)
    def evaluate_grouped_candidates(
        self,
        candidates: list[Candidate],
        phase: str,
        trial_numbers: list[int],
    ) -> list[pd.DataFrame]:
        """Batch coarse G/B groups."""
        groups = backbone_groups(candidates, trial_numbers)
        for inner_fold in self.plan.inner_folds:
            train_rows, validation_rows = self.fold_rows(inner_fold)
            for seed_offset in self.plan.seed_offsets:
                completed = self.completed_keys()
                pending_groups = [
                    group
                    for group in groups
                    if any(
                        (
                            candidate.identifier,
                            inner_fold,
                            seed_offset,
                        )
                        not in completed
                        for candidate, _ in group
                    )
                ]
                for chunk in model_batches(
                    pending_groups,
                    self.plan.model_batch_size,
                ):
                    representatives = [group[0][0] for group in chunk]
                    print(
                        f"START {phase} backbones={len(chunk)} "
                        f"inner={inner_fold} seed={seed_offset}",
                        flush=True,
                    )
                    results = self.train_backbone_batch(
                        representatives,
                        [train_rows for _ in chunk],
                        [inner_fold for _ in chunk],
                        [seed_offset for _ in chunk],
                    )
                    for group, result in zip(chunk, results):
                        self.run_head_batch(
                            [candidate for candidate, _ in group],
                            result.state,
                            train_rows,
                            validation_rows,
                            inner_fold,
                            seed_offset,
                            phase,
                            [number for _, number in group],
                        )
        return [
            self.seed_pooled_metrics(candidate)
            for candidate in candidates
        ]

    def seed_pooled_metrics(self, candidate: Candidate) -> pd.DataFrame:
        """Pool folds and compute G, W, J."""
        from evaluation import evaluate_predictions

        selected = self.predictions.loc[
            self.predictions["candidate_id"] == candidate.identifier
        ]
        rows = []
        for seed_offset in self.plan.seed_offsets:
            part = selected.loc[selected["seed_offset"] == seed_offset]
            metrics = add_skill_scores(
                evaluate_predictions(
                    part,
                    self.context.settings.ranking_temperature,
                )
            )
            metrics.update(label_oof_top_k_metrics(part))
            rows.append(
                {
                    "candidate_id": candidate.identifier,
                    **candidate.as_dict(),
                    "seed_offset": seed_offset,
                    **metrics,
                }
            )
        return pd.DataFrame(rows)

    def run_complete_trial_batch(
        self,
        candidate: Candidate,
        phase: str,
        trial_number: int,
        pending: list[tuple[int, int]],
    ) -> None:
        """Complete and commit one GPU trial."""
        from evaluation import (
            evaluate_predictions,
            ranking_score,
        )

        folds = [value[0] for value in pending]
        seed_offsets = [value[1] for value in pending]
        train_validation = [
            self.fold_rows(inner_fold)
            for inner_fold in folds
        ]
        states = [self.seed_states[value] for value in seed_offsets]
        jobs = [
            self.fast_backbone.BackboneJob(
                rows[0],
                candidate.graphban,
                candidate.boltz,
            )
            for rows in train_validation
        ]
        result = self.fast_trial.run_trial_batch(
            self.context,
            self.backbone_data,
            [state["backbone"] for state in states],
            [state["scorer"] for state in states],
            [state["standardizer"] for state in states],
            jobs,
            [state["seed"] for state in states],
            [state["seed"] + 5000 for state in states],
            [candidate.anchor for _ in states],
            [rows[1] for rows in train_validation],
            self.plan.backbone_epochs,
            self.plan.steps_per_epoch,
            self.plan.head_epochs,
            self._cached_supervision(jobs, folds, seed_offsets),
            self._cached_schedules([state["seed"] for state in states]),
        )

        backbone_rows = []
        run_rows = []
        prediction_parts = []
        n_models = len(pending)
        cpu_evaluation_started = time.perf_counter()
        for model, (
            inner_fold,
            seed_offset,
            rows,
            backbone_history,
            head_history,
            prediction,
        ) in enumerate(
            zip(
                folds,
                seed_offsets,
                train_validation,
                result.backbone_histories,
                result.head_histories,
                result.predictions,
            )
        ):
            final_backbone = backbone_history.iloc[-1].to_dict()
            backbone_rows.append(
                {
                    "search_signature": self.signature,
                    "inner_fold": inner_fold,
                    "seed_offset": seed_offset,
                    "graphban_weight": candidate.graphban,
                    "boltz_weight": candidate.boltz,
                    "selected_epoch": self.plan.backbone_epochs,
                    **final_backbone,
                    "backbone_execution": "complete_gpu_trial_batch",
                    "backbone_model_batch_size": n_models,
                    "batch_elapsed_seconds": result.backbone_seconds,
                    "seconds_per_backbone": (
                        result.backbone_seconds / n_models
                    ),
                    "peak_cuda_memory_mb": result.peak_cuda_memory_mb,
                    **{
                        f"final_{name}": value
                        for name, value in final_backbone.items()
                        if name != "epoch"
                    },
                }
            )

            prediction.insert(0, "candidate_id", candidate.identifier)
            prediction["ranking_score"] = ranking_score(
                prediction["final_score"],
                self.context.models.architecture.primary_direction,
            )
            metrics = evaluate_predictions(
                prediction,
                self.context.settings.ranking_temperature,
            )
            prediction.insert(1, "search_signature", self.signature)
            prediction.insert(2, "phase", phase)
            prediction.insert(3, "trial_number", trial_number)
            prediction.insert(4, "inner_fold", inner_fold)
            prediction.insert(5, "seed_offset", seed_offset)
            for position, (name, value) in enumerate(
                candidate.as_dict().items(),
                start=6,
            ):
                prediction.insert(position, name, value)
            final_head = head_history.iloc[-1].to_dict()
            run_rows.append(
                {
                    "search_signature": self.signature,
                    "phase": phase,
                    "trial_number": trial_number,
                    "candidate_id": candidate.identifier,
                    **candidate.as_dict(),
                    "inner_fold": inner_fold,
                    "seed_offset": seed_offset,
                    "n_train": len(rows[0]),
                    "n_validation": len(rows[1]),
                    "head_seed": states[model]["seed"] + 5000,
                    "head_execution": "complete_gpu_trial_batch",
                    "head_model_batch_size": n_models,
                    "feature_refresh_seconds": (
                        result.feature_refresh_seconds / n_models
                    ),
                    "head_batch_elapsed_seconds": (
                        result.head_seconds / n_models
                    ),
                    "seconds_per_head": result.head_seconds / n_models,
                    "head_evaluation_seconds": (
                        result.evaluation_seconds / n_models
                    ),
                    "trial_feature_refresh_elapsed_seconds": (
                        result.feature_refresh_seconds
                    ),
                    "trial_head_elapsed_seconds": result.head_seconds,
                    "trial_evaluation_elapsed_seconds": (
                        result.evaluation_seconds
                    ),
                    "trial_transfer_elapsed_seconds": result.transfer_seconds,
                    "primary_coefficient": (
                        result.primary_coefficients[model]
                    ),
                    **metrics,
                    **{
                        f"head_final_{name}": value
                        for name, value in final_head.items()
                        if name != "epoch"
                    },
                }
            )
            prediction_parts.append(prediction)

        cpu_evaluation_seconds = (
            time.perf_counter() - cpu_evaluation_started
        )
        total_evaluation_seconds = (
            result.evaluation_seconds + cpu_evaluation_seconds
        )
        total_trial_seconds = (
            result.backbone_seconds
            + result.feature_refresh_seconds
            + result.head_seconds
            + total_evaluation_seconds
            + result.transfer_seconds
        )
        for row in run_rows:
            row["head_evaluation_seconds"] = (
                total_evaluation_seconds / n_models
            )
            row["trial_gpu_prediction_elapsed_seconds"] = (
                result.evaluation_seconds
            )
            row["trial_cpu_metrics_elapsed_seconds"] = (
                cpu_evaluation_seconds
            )
            row["trial_evaluation_elapsed_seconds"] = (
                total_evaluation_seconds
            )
            row["trial_total_compute_seconds"] = total_trial_seconds
        self._commit_backbones(backbone_rows)
        self._commit_heads(run_rows, prediction_parts)
        print(
            f"PASS  {candidate.identifier}: "
            f"backbone={result.backbone_seconds:.3f}s "
            f"refresh={result.feature_refresh_seconds:.3f}s "
            f"head={result.head_seconds:.3f}s "
            f"evaluation={total_evaluation_seconds:.3f}s "
            f"transfer={result.transfer_seconds:.3f}s",
            flush=True,
        )

    def evaluate_candidate(
        self,
        candidate: Candidate,
        phase: str,
        trial_number: int,
    ) -> pd.DataFrame:
        """Batch one full trial."""
        completed = self.completed_keys()
        pending = [
            (inner_fold, seed_offset)
            for seed_offset in self.plan.seed_offsets
            for inner_fold in self.plan.inner_folds
            if (
                candidate.identifier,
                inner_fold,
                seed_offset,
            )
            not in completed
        ]
        expected = len(self.plan.inner_folds) * len(self.plan.seed_offsets)
        if len(pending) == expected and expected <= self.plan.model_batch_size:
            print(
                f"START {phase} complete-gpu-trial models={expected} "
                f"candidate={candidate.identifier}",
                flush=True,
            )
            self.run_complete_trial_batch(
                candidate,
                phase,
                trial_number,
                pending,
            )
            return self.seed_pooled_metrics(candidate)

        for batch in model_batches(pending, self.plan.model_batch_size):
            folds = [value[0] for value in batch]
            seed_offsets = [value[1] for value in batch]
            train_validation = [
                self.fold_rows(inner_fold)
                for inner_fold in folds
            ]
            print(
                f"START {phase} backbones={len(batch)} "
                f"candidate={candidate.identifier} "
                f"seeds={sorted(set(seed_offsets))}",
                flush=True,
            )
            results = self.train_backbone_batch(
                [candidate for _ in batch],
                [rows[0] for rows in train_validation],
                folds,
                seed_offsets,
            )
            for (
                inner_fold,
                seed_offset,
                rows,
                result,
            ) in zip(
                folds,
                seed_offsets,
                train_validation,
                results,
            ):
                self.run_head_batch(
                    [candidate],
                    result.state,
                    rows[0],
                    rows[1],
                    inner_fold,
                    seed_offset,
                    phase,
                    [trial_number],
                )
        return self.seed_pooled_metrics(candidate)

    def evaluate_candidate_batch(
        self,
        candidates: list[Candidate],
        phase: str,
        trial_numbers: list[int],
    ) -> list[pd.DataFrame]:
        """Complete one shared-backbone group."""
        return self.evaluate_grouped_candidates(
            candidates,
            phase,
            trial_numbers,
        )


def initialize_search(
    config_path: Path,
    core_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    domain: str | None,
    device: str,
    plan: SearchPlan,
    signature: str,
    module_tag: str,
) -> SearchRuntime:
    """Initialize one search runtime."""
    script_dir = Path(__file__).resolve().parent
    base_path = script_dir / BASE_DEPENDENCY
    fixed_path = script_dir / FIXED_DEPENDENCY
    fast_path = script_dir / FAST_DEPENDENCY
    fast_backbone_path = script_dir / FAST_BACKBONE_DEPENDENCY
    fast_trial_path = script_dir / FAST_TRIAL_DEPENDENCY
    sys.path.insert(0, str(core_dir))
    base = load_module(base_path, f"{module_tag}_base")
    fixed = load_module(fixed_path, f"{module_tag}_fixed")
    fast_backbone = load_module(
        fast_backbone_path,
        f"{module_tag}_fast_backbone",
    )
    fast_trial = load_module(
        fast_trial_path,
        f"{module_tag}_fast_trial",
    )

    from selection import audit_splits, load_materialized_splits
    from training import prepare_context

    generated_config = base.build_experiment_config(
        config_path,
        output_dir,
    )
    context = prepare_context(
        generated_config,
        domain,
        device,
        backbone_epochs=0,
        steps_per_epoch=plan.steps_per_epoch,
    )
    splits = load_materialized_splits(splits_dir)
    audit_splits(splits, context)
    outer_component = 0 if plan.outer_fold is None else plan.outer_fold * 10000
    joint_seed = (
        context.settings.seed
        + plan.repeat * 100000
        + outer_component
        + 2700
    )
    seed_states = fixed.build_seed_states(
        context,
        base,
        plan.seed_offsets,
        joint_seed,
    )
    runner = ExperimentRunner(
        base,
        fixed,
        context,
        splits,
        plan,
        output_dir,
        signature,
        seed_states,
        fast_backbone,
        fast_trial,
    )
    return SearchRuntime(
        runner,
        base,
        fixed,
        context,
        splits,
        generated_config,
        base_path,
        fixed_path,
        fast_path,
        fast_backbone_path,
        fast_trial_path,
        seed_states,
    )


def evaluate_candidates(
    runner: ExperimentRunner,
    candidates: Iterable[Candidate],
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch candidates and summarize."""
    numbered = list(enumerate(candidates))
    seed_tables = runner.evaluate_grouped_candidates(
        [candidate for _, candidate in numbered],
        phase,
        [number for number, _ in numbered],
    )
    seed_metrics = pd.concat(seed_tables, ignore_index=True)
    return seed_metrics, candidate_summary(seed_metrics)
