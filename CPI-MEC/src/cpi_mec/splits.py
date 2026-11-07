"""Create reproducible nested grouped cross-validation assignments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import load_config
from relations import PairRelation, RelationBundle, load_relations


@dataclass
class SplitBundle:
    """Store fixed nested cross-validation assignments."""

    labels: pd.DataFrame
    outer_assignments: pd.DataFrame
    inner_assignments: pd.DataFrame
    domain_assignments: pd.DataFrame
    summary: pd.DataFrame

    def outer_test_rows(self, repeat: int, outer_fold: int) -> np.ndarray:
        """Return row IDs for one outer test set."""
        rows = self.outer_assignments
        selected = rows[(rows["repeat"] == repeat) & (rows["outer_fold"] == outer_fold)]
        return selected["relation_row"].to_numpy(dtype=int)

    def outer_train_rows(self, repeat: int, outer_fold: int) -> np.ndarray:
        """Return row IDs for one outer training set."""
        test_rows = self.outer_test_rows(repeat, outer_fold)
        return np.setdiff1d(self.labels["relation_row"].to_numpy(), test_rows)

    def inner_validation_rows(self, repeat: int, outer_fold: int, inner_fold: int) -> np.ndarray:
        """Return row IDs for one inner validation set."""
        rows = self.inner_assignments
        selected = rows[
            (rows["repeat"] == repeat)
            & (rows["outer_fold"] == outer_fold)
            & (rows["inner_fold"] == inner_fold)
        ]
        return selected["relation_row"].to_numpy(dtype=int)

    def inner_train_rows(self, repeat: int, outer_fold: int, inner_fold: int) -> np.ndarray:
        """Return row IDs for one inner training set."""
        outer_train = self.outer_train_rows(repeat, outer_fold)
        inner_validation = self.inner_validation_rows(repeat, outer_fold, inner_fold)
        return np.setdiff1d(outer_train, inner_validation)


def find_binary_relation(bundle: RelationBundle) -> tuple[PairRelation, str]:
    """Find the sole binary relation from contracts."""
    matches = []
    for relation in bundle.pair_tables.values():
        for column, rule in relation.contract.get("columns", {}).items():
            if rule.get("type") == "binary":
                matches.append((relation, column))
    if len(matches) != 1:
        raise ValueError(f"Expected one binary relation column, found {len(matches)}")
    return matches[0]


def prepare_labels(bundle: RelationBundle, config: dict[str, Any]) -> pd.DataFrame:
    """Combine binary labels with configured groups."""
    relation, label_column = find_binary_relation(bundle)
    grouping = config["cross_validation"]["grouping"]
    group_table = bundle.entity_tables[grouping["input"]].frame
    group_column = grouping["column"]

    groups = group_table[["entity_a", group_column]].rename(columns={group_column: "group"})
    labels = relation.frame[["entity_a", "entity_b", label_column]].rename(
        columns={label_column: "label"}
    )
    labels = labels.merge(groups, on="entity_a", how="left", validate="many_to_one")
    labels["label"] = pd.to_numeric(labels["label"]).astype(int)
    labels["group"] = labels["group"].astype("string").str.strip()
    labels.insert(0, "relation_row", np.arange(len(labels), dtype=int))
    if labels["group"].isna().any() or labels["group"].eq("").any():
        raise ValueError("Missing cross-validation groups for binary relations")
    return labels


def assign_folds(labels: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """Assign stratified folds while isolating groups."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.empty(len(labels), dtype=int)
    for fold, (_, validation_index) in enumerate(
        splitter.split(labels, labels["label"], labels["group"]), start=1
    ):
        folds[validation_index] = fold
    return folds


def assign_domain_entities(
    bundle: RelationBundle,
    config: dict[str, Any],
    labelled: pd.DataFrame,
    repeat: int,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Keep labelled-group folds and balance remaining domain entities."""
    grouping = config["cross_validation"]["grouping"]
    group_column = grouping["column"]
    entities = bundle.entity_tables[grouping["input"]].frame[["entity_a", group_column]].rename(
        columns={group_column: "group"}
    )
    entities["group"] = entities["group"].astype("string").str.strip()
    group_to_fold = labelled.groupby("group")["outer_fold"].first().astype(int).to_dict()
    group_sizes = entities.groupby("group", sort=False)["entity_a"].nunique()
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    for group, fold in group_to_fold.items():
        fold_sizes[fold - 1] += int(group_sizes.loc[group])

    remaining = np.asarray(sorted(set(group_sizes.index) - set(group_to_fold)), dtype=object)
    np.random.default_rng(seed).shuffle(remaining)
    for group in remaining:
        fold = int(np.argmin(fold_sizes))
        group_to_fold[group] = fold + 1
        fold_sizes[fold] += int(group_sizes.loc[group])

    assignment = entities.copy()
    assignment.insert(0, "repeat", repeat)
    assignment.insert(1, "outer_fold", assignment["group"].map(group_to_fold).astype(int))
    return assignment


def audit_assignment(frame: pd.DataFrame, fold_column: str, n_folds: int, context: str) -> None:
    """Audit row assignment, group isolation, and classes."""
    if sorted(frame[fold_column].unique()) != list(range(1, n_folds + 1)):
        raise ValueError(f"{context}: incomplete fold IDs")
    if frame.groupby("group")[fold_column].nunique().max() != 1:
        raise ValueError(f"{context}: a group crosses folds")
    if frame.groupby(fold_column)["label"].nunique().min() != 2:
        raise ValueError(f"{context}: a fold lacks one binary class")


def summarize(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    """Summarize samples, groups, and labels per fold."""
    fold_columns = ["repeat", "outer_fold"]
    if level == "inner":
        fold_columns.append("inner_fold")
    summary = (
        frame.groupby(fold_columns, sort=True)
        .agg(
            n_rows=("relation_row", "size"),
            n_groups=("group", "nunique"),
            n_positive=("label", "sum"),
        )
        .reset_index()
    )
    summary["n_negative"] = summary["n_rows"] - summary["n_positive"]
    summary["positive_prevalence"] = summary["n_positive"] / summary["n_rows"]
    summary.insert(0, "level", level)
    return summary


def build_splits(config_path: str | Path, domain_name: str | None = None) -> SplitBundle:
    """Build repeated nested grouped cross-validation."""
    config = load_config(config_path)
    bundle = load_relations(config_path, domain_name)
    labels = prepare_labels(bundle, config)

    settings = config["cross_validation"]
    repeats = int(settings["outer_repeats"])
    outer_folds = int(settings["outer_folds"])
    inner_folds = int(settings.get("inner_folds", outer_folds))
    base_seed = int(settings.get("seed", config["model"]["seed"]))

    outer_parts = []
    inner_parts = []
    domain_parts = []
    for repeat in range(1, repeats + 1):
        outer = labels.copy()
        outer["repeat"] = repeat
        outer["outer_fold"] = assign_folds(labels, outer_folds, base_seed + repeat - 1)
        audit_assignment(outer, "outer_fold", outer_folds, f"repeat {repeat} outer")
        outer_parts.append(outer)
        domain_parts.append(
            assign_domain_entities(
                bundle,
                config,
                outer,
                repeat,
                outer_folds,
                base_seed + repeat - 1,
            )
        )

        for outer_fold in range(1, outer_folds + 1):
            inner = outer.loc[outer["outer_fold"] != outer_fold].copy()
            inner["outer_fold"] = outer_fold
            inner["inner_fold"] = assign_folds(
                inner,
                inner_folds,
                base_seed + repeat * 1000 + outer_fold,
            )
            audit_assignment(
                inner,
                "inner_fold",
                inner_folds,
                f"repeat {repeat} outer fold {outer_fold} inner",
            )
            inner_parts.append(inner)

    outer_assignments = pd.concat(outer_parts, ignore_index=True)
    inner_assignments = pd.concat(inner_parts, ignore_index=True)
    domain_assignments = pd.concat(domain_parts, ignore_index=True)
    summary = pd.concat(
        [summarize(outer_assignments, "outer"), summarize(inner_assignments, "inner")],
        ignore_index=True,
    )
    return SplitBundle(labels, outer_assignments, inner_assignments, domain_assignments, summary)


def save_splits(splits: SplitBundle, output_dir: str | Path) -> None:
    """Save assignments and their audit summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_columns = ["relation_row", "entity_a", "entity_b", "label", "group"]
    splits.outer_assignments[
        ["repeat", "outer_fold", *split_columns]
    ].to_csv(output / "outer_assignments.csv", index=False)
    splits.inner_assignments[
        ["repeat", "outer_fold", "inner_fold", *split_columns]
    ].to_csv(output / "inner_assignments.csv", index=False)
    splits.domain_assignments[["repeat", "outer_fold", "entity_a", "group"]].to_csv(
        output / "domain_assignments.csv", index=False
    )
    splits.summary.to_csv(output / "split_summary.csv", index=False)


def verify_saved_splits(
    splits: SplitBundle,
    output_dir: str | Path,
) -> None:
    """Verify saved splits against current inputs and rules."""
    output = Path(output_dir)
    outer = pd.read_csv(output / "outer_assignments.csv")
    inner = pd.read_csv(output / "inner_assignments.csv")
    domain = pd.read_csv(output / "domain_assignments.csv")

    expected_outer = splits.outer_assignments[
        ["repeat", "outer_fold", "relation_row", "entity_a", "entity_b", "label", "group"]
    ]
    expected_inner = splits.inner_assignments[
        [
            "repeat",
            "outer_fold",
            "inner_fold",
            "relation_row",
            "entity_a",
            "entity_b",
            "label",
            "group",
        ]
    ]
    expected_domain = splits.domain_assignments[
        ["repeat", "outer_fold", "entity_a", "group"]
    ]
    for name, saved, expected in (
        ("outer_assignments.csv", outer, expected_outer),
        ("inner_assignments.csv", inner, expected_inner),
        ("domain_assignments.csv", domain, expected_domain),
    ):
        try:
            pd.testing.assert_frame_equal(
                saved.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_dtype=False,
            )
        except AssertionError as error:
            raise ValueError(f"{name} does not match the configured split rule") from error


def main() -> None:
    """Provide the split-generation entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--domain", help="Configured domain name; defaults to the first domain.")
    parser.add_argument("--output-dir", required=True, help="Directory for split CSV files.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing split files without replacing them.",
    )
    args = parser.parse_args()

    splits = build_splits(args.config, args.domain)
    if args.verify_only:
        verify_saved_splits(splits, args.output_dir)
    else:
        save_splits(splits, args.output_dir)
    outer = splits.outer_assignments
    inner = splits.inner_assignments
    print(f"PASS  binary relations: {len(splits.labels)} rows, {splits.labels['group'].nunique()} groups")
    print(
        f"PASS  outer splits: {outer['repeat'].nunique()} repeats x "
        f"{outer['outer_fold'].nunique()} folds"
    )
    print(f"PASS  outer group_cross_fold_max: {outer.groupby(['repeat', 'group'])['outer_fold'].nunique().max()}")
    print(f"PASS  inner folds per outer training set: {inner['inner_fold'].nunique()}")
    print(
        "PASS  inner group_cross_fold_max: "
        f"{inner.groupby(['repeat', 'outer_fold', 'group'])['inner_fold'].nunique().max()}"
    )
    print(
        f"PASS  domain assignments: {splits.domain_assignments['entity_a'].nunique()} entities x "
        f"{splits.domain_assignments['repeat'].nunique()} repeats"
    )
    action = "verified" if args.verify_only else "saved"
    print(f"PASS  split files {action}: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
