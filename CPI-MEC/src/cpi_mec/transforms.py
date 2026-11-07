"""Apply declared numeric transformations without changing relation semantics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import load_config
from relations import RelationBundle, RelationDomain, load_relations


DIRECTIONS = {"higher_is_better", "lower_is_better"}
OPERATIONS = {"identity", "row_percentile"}
MISSING_POLICIES = {"preserve", "row_mean"}


@dataclass(frozen=True)
class FittedStandardizer:
    """Store column scaling fitted on training data only."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted parameters."""
        return ((np.asarray(values) - self.mean) / self.scale).astype(np.float32)


@dataclass
class TransformedMatrix:
    """Store transformed values and the original observation mask."""

    name: str
    source_role: str
    values: pd.DataFrame
    observed: pd.DataFrame
    input_direction: str
    output_direction: str
    operation: str
    missing_policy: str

    @property
    def imputed_count(self) -> int:
        """Return the number of originally missing cells that were filled."""
        return int((~self.observed & self.values.notna()).to_numpy().sum())


@dataclass
class TransformBundle:
    """Collect declared matrix transforms in one domain."""

    domain: RelationDomain
    matrices: dict[str, TransformedMatrix]

    def get(self, name: str) -> TransformedMatrix:
        """Retrieve a transformed matrix by configured name."""
        return self.matrices[name]


def fill_row_means(values: pd.DataFrame) -> pd.DataFrame:
    """Fill missing cells with their row mean."""
    array = values.to_numpy(dtype=np.float64, copy=True)
    all_missing = np.isnan(array).all(axis=1)
    if all_missing.any():
        bad = values.index[all_missing].astype(str).tolist()
        raise ValueError(f"Cannot row-mean impute entirely missing rows: {bad}")
    row_means = np.nanmean(array, axis=1)
    rows, columns = np.where(np.isnan(array))
    array[rows, columns] = row_means[rows]
    return pd.DataFrame(array, index=values.index, columns=values.columns)


def row_percentile(values: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Compute row percentiles; larger output is always stronger."""
    require_direction(direction)
    ranked = values.rank(
        axis=1,
        method="average",
        pct=True,
        ascending=direction == "higher_is_better",
    )
    return ranked.astype(np.float32)


def stable_logit(probability: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Apply a clipped, numerically stable logit."""
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")
    values = np.asarray(probability, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if ((finite < 0) | (finite > 1)).any():
        raise ValueError("probability values must lie in [0, 1]")
    clipped = np.clip(values, epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped)).astype(np.float32)


def fit_standardizer(train: np.ndarray, minimum_scale: float = 1e-8) -> FittedStandardizer:
    """Fit column means and population SDs on training data only."""
    values = np.asarray(train, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("training features must be a finite two-dimensional array")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale < minimum_scale] = 1.0
    return FittedStandardizer(mean, scale)


def median_row_scale(values: np.ndarray) -> float:
    """Return the median population SD across rows."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("scale source must be a finite two-dimensional array")
    scale = float(np.median(array.std(axis=1, ddof=0)))
    if scale <= 0:
        raise ValueError("median row scale must be positive")
    return scale


def row_z(values: np.ndarray, input_direction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build higher-is-better row Z-scores and return source moments."""
    require_direction(input_direction)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("row-Z input must be a finite two-dimensional array")
    mean = array.mean(axis=1)
    scale = array.std(axis=1, ddof=0)
    if (scale <= 0).any():
        raise ValueError(f"Cannot row-standardize zero-variance rows: {np.flatnonzero(scale <= 0).tolist()}")
    sign = 1.0 if input_direction == "higher_is_better" else -1.0
    standardized = sign * (array - mean[:, None]) / scale[:, None]
    return standardized.astype(np.float32), mean, scale


def require_direction(direction: str) -> None:
    """Validate a score-direction declaration."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}, got {direction!r}")


def transform_matrix(
    name: str,
    bundle: RelationBundle,
    specification: dict[str, Any],
) -> TransformedMatrix:
    """Transform one matrix from one declarative specification."""
    source_role = specification["input"]
    if source_role not in bundle.matrices:
        raise ValueError(f"transforms.{name}.input is not a matrix role: {source_role}")
    operation = specification.get("operation", "identity")
    missing_policy = specification.get("missing", "preserve")
    direction = specification["input_direction"]
    require_direction(direction)
    if operation not in OPERATIONS:
        raise ValueError(f"transforms.{name}.operation must be one of {sorted(OPERATIONS)}")
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(f"transforms.{name}.missing must be one of {sorted(MISSING_POLICIES)}")

    source = bundle.matrices[source_role]
    observed = source.observed.copy()
    values = source.values.copy()
    if missing_policy == "row_mean":
        values = fill_row_means(values)
    if operation == "row_percentile":
        values = row_percentile(values, direction)
        output_direction = "higher_is_better"
    else:
        values = values.astype(np.float32)
        output_direction = direction

    return TransformedMatrix(
        name=name,
        source_role=source_role,
        values=values,
        observed=observed,
        input_direction=direction,
        output_direction=output_direction,
        operation=operation,
        missing_policy=missing_policy,
    )


def build_transforms(
    config_path: str | Path,
    domain_name: str | None = None,
) -> TransformBundle:
    """Load relations and execute all declared matrix transforms."""
    config = load_config(config_path)
    specifications = config.get("transforms")
    if not isinstance(specifications, dict) or not specifications:
        raise ValueError("Configuration must declare a non-empty transforms mapping")
    bundle = load_relations(config_path, domain_name)
    matrices = {
        name: transform_matrix(name, bundle, specification)
        for name, specification in specifications.items()
    }
    return TransformBundle(bundle.domain, matrices)


def main() -> None:
    """Validate the transform layer without writing matrices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--domain", help="Configured domain name; defaults to the first domain.")
    args = parser.parse_args()

    transformed = build_transforms(args.config, args.domain)
    print(
        f"PASS  domain {transformed.domain.name}: "
        f"{len(transformed.domain.entity_a)} x {len(transformed.domain.entity_b)}"
    )
    for name, matrix in transformed.matrices.items():
        observed = int(matrix.observed.to_numpy().sum())
        finite = int(np.isfinite(matrix.values.to_numpy()).sum())
        print(
            f"PASS  {name}: {matrix.operation}, {matrix.input_direction} -> "
            f"{matrix.output_direction}, observed {observed}, finite {finite}, "
            f"imputed {matrix.imputed_count}"
        )
    print("PASS  transform bundle")


if __name__ == "__main__":
    main()
