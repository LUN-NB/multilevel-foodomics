"""Refine one coarse winner with dynamic local continuous TPE.

Local bounds come directly from the coarse winner and adjacent positive grid
points. A boundary winner is geometrically extended by one grid step. TPE adds
64 positive candidates, then exact-best compares all of them with the coarse
winner using J.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from adaptive_search_common import (
    SCRIPT_RELEASE as OBJECTIVE_RELEASE,
    Candidate,
    SearchPlan,
    candidate_summary,
    implementation_hashes,
    initialize_search,
    pareto_front,
    parse_int_values,
    search_signature,
    select_exact_best,
    sha256,
)


SCRIPT_VERSION = "1.2.0"
SCRIPT_RELEASE = "dynamic_local_tpe_complete_gpu_trial_20260729"
PARAMETERS = (
    "anchor_weight",
    "graphban_weight",
    "boltz_weight",
)


def completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Return completed trials."""
    return study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE,),
    )


def trial_candidate(trial: optuna.trial.FrozenTrial) -> Candidate:
    """Restore a candidate from one trial."""
    return Candidate(*(float(trial.params[name]) for name in PARAMETERS))


def read_coarse(
    coarse_dir: Path,
) -> tuple[pd.Series, Candidate, dict[str, tuple[float, float]]]:
    """Read the coarse winner and local bounds."""
    row = pd.read_csv(coarse_dir / "selected_parameters.csv").iloc[0]
    candidate = Candidate(
        float(row["anchor_weight"]),
        float(row["graphban_weight"]),
        float(row["boltz_weight"]),
    )
    bounds_table = pd.read_csv(
        coarse_dir / "local_continuous_bounds.csv"
    ).set_index("parameter")
    bounds = {
        name: (
            float(bounds_table.at[name, "local_low"]),
            float(bounds_table.at[name, "local_high"]),
        )
        for name in PARAMETERS
    }
    return row, candidate, bounds


def inject_coarse(
    study: optuna.Study,
    coarse: Candidate,
    coarse_row: pd.Series,
    bounds: dict[str, tuple[float, float]],
) -> None:
    """Inject the coarse winner."""
    if study.get_trials(deepcopy=False):
        return
    distributions = {
        name: optuna.distributions.FloatDistribution(
            low=bounds[name][0],
            high=bounds[name][1],
            log=True,
        )
        for name in PARAMETERS
    }
    study.add_trial(
        optuna.trial.create_trial(
            params=dict(zip(PARAMETERS, coarse.as_dict().values())),
            distributions=distributions,
            value=float(coarse_row["selection_score_j"]),
            user_attrs={
                "source": "coarse_injected",
                "candidate_id": coarse.identifier,
                "mean_seed_pooled_global_skill": float(
                    coarse_row["global_skill"]
                ),
                "mean_seed_pooled_within_skill": float(
                    coarse_row["within_skill"]
                ),
                "mean_seed_pooled_ranking_pooled_auprc": float(
                    coarse_row["pooled_auprc"]
                ),
                "mean_seed_pooled_pairwise_logistic_loss": float(
                    coarse_row["pairwise_logistic_loss"]
                ),
            },
        )
    )


def continuous_trial_count(study: optuna.Study) -> int:
    """Count completed new continuous trials."""
    return sum(
        trial.user_attrs.get("source") == "continuous_tpe"
        for trial in completed_trials(study)
    )


def study_summary(study: optuna.Study) -> pd.DataFrame:
    """Summarize all study candidates."""
    rows = []
    for trial in completed_trials(study):
        candidate = trial_candidate(trial)
        rows.append(
            {
                "trial_number": trial.number,
                "source": trial.user_attrs["source"],
                "candidate_id": candidate.identifier,
                **candidate.as_dict(),
                "selection_score_j": float(trial.value),
                "global_skill": trial.user_attrs[
                    "mean_seed_pooled_global_skill"
                ],
                "within_skill": trial.user_attrs[
                    "mean_seed_pooled_within_skill"
                ],
                "pooled_auprc": trial.user_attrs[
                    "mean_seed_pooled_ranking_pooled_auprc"
                ],
                "pairwise_logistic_loss": trial.user_attrs[
                    "mean_seed_pooled_pairwise_logistic_loss"
                ],
            }
        )
    return pd.DataFrame(rows)


def boundary_audit(
    selected: Candidate,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Audit local-bound proximity."""
    rows = []
    for name, value in selected.as_dict().items():
        low, high = bounds[name]
        rows.append(
            {
                "parameter": name,
                "selected_value": value,
                "local_low": low,
                "local_high": high,
                "at_lower_boundary": bool(np.isclose(value, low)),
                "at_upper_boundary": bool(np.isclose(value, high)),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """Run dynamic local continuous TPE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--coarse-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=("nested", "full"), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--inner-folds", default="1,2,3,4,5")
    parser.add_argument("--seed-offsets", default="0,1")
    parser.add_argument("--n-trials", type=int, default=64)
    parser.add_argument("--n-startup-trials", type=int, default=12)
    parser.add_argument("--n-ei-candidates", type=int, default=24)
    parser.add_argument("--sampler-seed", type=int, default=20260725)
    parser.add_argument(
        "--study-name",
        default="cpi_mec_dynamic_local_tpe",
    )
    parser.add_argument("--domain")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone-epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--head-epochs", type=int, default=120)
    parser.add_argument("--model-batch-size", type=int, default=10)
    args = parser.parse_args(argv)

    if args.scope == "nested" and args.outer_fold is None:
        parser.error("--outer-fold is required for nested search")
    if args.scope == "full" and args.outer_fold is not None:
        parser.error("--outer-fold is not used for full-label CV")

    config_path = Path(args.config).resolve()
    core_dir = Path(args.core_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    coarse_dir = Path(args.coarse_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_row, coarse, bounds = read_coarse(coarse_dir)
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
        "coarse_candidate": coarse.as_dict(),
        "coarse_selection_score_j": float(
            coarse_row["selection_score_j"]
        ),
        "bounds": bounds,
        "n_trials": args.n_trials,
        "n_startup_trials": args.n_startup_trials,
        "n_ei_candidates": args.n_ei_candidates,
        "sampler_seed": args.sampler_seed,
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
        "cpi_mec_continuous",
    )

    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        n_startup_trials=args.n_startup_trials,
        n_ei_candidates=args.n_ei_candidates,
        multivariate=True,
    )
    storage = f"sqlite:///{(output_dir / 'tpe_study.sqlite3').as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        direction="maximize",
        load_if_exists=True,
    )
    stored_signature = study.user_attrs.get("search_signature")
    if stored_signature is None:
        study.set_user_attr("search_signature", signature)
    elif stored_signature != signature:
        raise ValueError("Existing study belongs to another search")
    inject_coarse(study, coarse, coarse_row, bounds)
    models_per_trial = len(plan.inner_folds) * len(plan.seed_offsets)
    batches_per_trial = (
        models_per_trial + plan.model_batch_size - 1
    ) // plan.model_batch_size
    print(
        f"PLAN  continuous backbones: {batches_per_trial} "
        f"GPU model batches per trial",
        flush=True,
    )

    def objective(trial: optuna.Trial) -> float:
        """Evaluate one continuous J."""
        candidate = Candidate(
            *(
                trial.suggest_float(
                    name,
                    bounds[name][0],
                    bounds[name][1],
                    log=True,
                )
                for name in PARAMETERS
            )
        )
        pooled = runtime.runner.evaluate_candidate(
            candidate,
            "continuous_tpe",
            trial.number,
        )
        means = pooled.mean(numeric_only=True)
        trial.set_user_attr("source", "continuous_tpe")
        trial.set_user_attr("candidate_id", candidate.identifier)
        for metric in (
            "global_skill",
            "within_skill",
            "ranking_pooled_auprc",
            "pairwise_logistic_loss",
        ):
            trial.set_user_attr(
                f"mean_seed_pooled_{metric}",
                float(means[metric]),
            )
        return float(means["selection_score_j"])

    while continuous_trial_count(study) < args.n_trials:
        study.optimize(
            objective,
            n_trials=1,
            n_jobs=1,
            gc_after_trial=False,
        )

    trials = study_summary(study)
    trials.to_csv(output_dir / "tpe_candidate_summary.csv", index=False)
    study.trials_dataframe(multi_index=False).to_csv(
        output_dir / "tpe_trials.csv",
        index=False,
    )

    coarse_seed = pd.read_csv(
        coarse_dir / "coarse_seed_pooled_metrics.csv"
    )
    coarse_seed = coarse_seed.loc[
        coarse_seed["candidate_id"] == coarse.identifier
    ]
    continuous_candidates = {
        trial_candidate(trial).identifier: trial_candidate(trial)
        for trial in completed_trials(study)
        if trial.user_attrs["source"] == "continuous_tpe"
    }
    seed_metrics = pd.concat(
        [
            coarse_seed,
            *(
                runtime.runner.seed_pooled_metrics(candidate)
                for candidate in continuous_candidates.values()
            ),
        ],
        ignore_index=True,
    )
    summary = candidate_summary(seed_metrics)
    source = trials.drop_duplicates("candidate_id").set_index(
        "candidate_id"
    )["source"]
    summary["source"] = summary["candidate_id"].map(source)
    summary["eligible_for_selection"] = True
    selected, summary = select_exact_best(summary)
    selected_row = summary.loc[summary["selected"]].iloc[0]
    selection = pd.DataFrame(
        [
            {
                "scope": plan.scope,
                "repeat": plan.repeat,
                "outer_fold": plan.outer_fold,
                "selected_candidate_id": selected.identifier,
                "selected_source": selected_row["source"],
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
                    "exact maximum J across coarse winner and all "
                    "completed continuous candidates"
                ),
            }
        ]
    )
    audit = boundary_audit(selected, bounds)

    seed_metrics.to_csv(
        output_dir / "continuous_seed_pooled_metrics.csv",
        index=False,
    )
    summary.sort_values(
        ["objective_rank", "candidate_id"]
    ).to_csv(
        output_dir / "continuous_candidate_summary.csv",
        index=False,
    )
    selection.to_csv(output_dir / "selected_parameters.csv", index=False)
    audit.to_csv(output_dir / "local_boundary_audit.csv", index=False)
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
        "coarse_parameters": coarse.as_dict(),
        "local_bounds": bounds,
        "sampler": {
            "name": "Optuna multivariate TPESampler",
            "direction": "maximize",
            "scale": "log",
            "seed": args.sampler_seed,
            "n_startup_trials": args.n_startup_trials,
            "n_ei_candidates": args.n_ei_candidates,
            "new_continuous_trials": args.n_trials,
            "coarse_winner_injected": True,
        },
        "objective": "J=mean_seed(0.5*G+0.5*W)",
        "selection_rule": (
            "exact maximum across the coarse winner and every completed "
            "continuous candidate"
        ),
        "selected_parameters": selected.as_dict(),
        "selected_source": selected_row["source"],
        "boundary_audit": audit.to_dict(orient="records"),
        "plan": {
            "inner_folds": plan.inner_folds,
            "seed_offsets": plan.seed_offsets,
            "backbone_epochs": plan.backbone_epochs,
            "steps_per_epoch": plan.steps_per_epoch,
            "head_epochs": plan.head_epochs,
            "model_batch_size": plan.model_batch_size,
            "backbone_gpu_batches_per_trial": batches_per_trial,
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
    print(f"PASS  continuous TPE output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
