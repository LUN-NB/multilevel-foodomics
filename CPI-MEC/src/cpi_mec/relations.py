"""Load declared relation inputs into one domain-aligned internal representation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import load_config, validate_input_paths


@dataclass(frozen=True)
class RelationDomain:
    """Store an analysis domain and its fixed axis order."""

    name: str
    entity_a: tuple[str, ...]
    entity_b: tuple[str, ...]


@dataclass
class MatrixRelation:
    """Store an aligned matrix; NaN means unobserved."""

    role: str
    values: pd.DataFrame
    contract: dict[str, Any]

    @property
    def observation_semantics(self) -> str:
        """Read observation semantics declared by the contract."""
        return self.contract.get("observation_semantics", "continuous")

    @property
    def observed(self) -> pd.DataFrame:
        """Mark observations without treating zero as missing."""
        return self.values.notna()

    @property
    def is_complete(self) -> bool:
        """Report whether every domain cell is observed."""
        return bool(self.values.notna().to_numpy().all())


@dataclass
class PairRelation:
    """Store in-domain pairs and original evidence fields."""

    role: str
    frame: pd.DataFrame
    contract: dict[str, Any]
    source_rows: int

    @property
    def observation_semantics(self) -> str:
        """Read observation semantics declared by the contract."""
        return self.contract.get("observation_semantics", "observed_only")


@dataclass
class EntityTable:
    """Store one-axis attributes ordered by the domain."""

    role: str
    frame: pd.DataFrame
    contract: dict[str, Any]
    source_rows: int


@dataclass
class RelationBundle:
    """Collect all normalized inputs in one domain."""

    domain: RelationDomain
    matrices: dict[str, MatrixRelation]
    pair_tables: dict[str, PairRelation]
    entity_tables: dict[str, EntityTable]

    def get(self, role: str) -> MatrixRelation | PairRelation | EntityTable:
        """Retrieve an input object by its configured role."""
        for collection in (self.matrices, self.pair_tables, self.entity_tables):
            if role in collection:
                return collection[role]
        raise KeyError(role)


def read_matrix(path: Path, contract: dict[str, Any]) -> pd.DataFrame:
    """Read a wide matrix and normalize axis names."""
    entity_a_column = contract["entity_a_column"]
    frame = pd.read_csv(
        path,
        dtype={entity_a_column: "string"},
        encoding=contract.get("encoding", "utf-8-sig"),
        sep=contract.get("separator", ","),
        low_memory=False,
    )
    frame[entity_a_column] = frame[entity_a_column].str.strip()
    values = frame.set_index(entity_a_column).apply(pd.to_numeric)
    values.index.name = "entity_a"
    values.columns = values.columns.astype(str).str.strip()
    values.columns.name = "entity_b"
    return values


def read_table(path: Path, contract: dict[str, Any]) -> pd.DataFrame:
    """Read a long table and normalize entity columns."""
    entity_columns = [contract["entity_a_column"]]
    if contract["kind"] == "pair_table":
        entity_columns.append(contract["entity_b_column"])
    frame = pd.read_csv(
        path,
        dtype={column: "string" for column in entity_columns},
        encoding=contract.get("encoding", "utf-8-sig"),
        sep=contract.get("separator", ","),
        low_memory=False,
    )
    rename = {contract["entity_a_column"]: "entity_a"}
    if contract["kind"] == "pair_table":
        rename[contract["entity_b_column"]] = "entity_b"
    frame = frame.rename(columns=rename)
    frame["entity_a"] = frame["entity_a"].str.strip()
    if "entity_b" in frame:
        frame["entity_b"] = frame["entity_b"].str.strip()
    return frame


def build_domain(
    name: str,
    rule: dict[str, Any],
    config: dict[str, Any],
) -> RelationDomain:
    """Build a domain in its declared matrix order."""
    source_role = rule["from"]
    source = config["inputs"][source_role]
    values = read_matrix(source["path"], source["contract"])
    return RelationDomain(name, tuple(values.index), tuple(values.columns))


def align_matrix(values: pd.DataFrame, domain: RelationDomain) -> pd.DataFrame:
    """Strictly subset and reorder a matrix to the domain."""
    return values.loc[list(domain.entity_a), list(domain.entity_b)]


def align_pair_table(frame: pd.DataFrame, domain: RelationDomain) -> pd.DataFrame:
    """Keep only entity pairs inside the analysis domain."""
    in_domain = frame["entity_a"].isin(domain.entity_a) & frame["entity_b"].isin(domain.entity_b)
    return frame.loc[in_domain].reset_index(drop=True)


def align_entity_table(frame: pd.DataFrame, domain: RelationDomain) -> pd.DataFrame:
    """Order attributes by the domain entity_a axis."""
    return frame.set_index("entity_a").loc[list(domain.entity_a)].reset_index()


def load_relations(
    config_path: str | Path,
    domain_name: str | None = None,
) -> RelationBundle:
    """Load and normalize all inputs in one domain."""
    config = load_config(config_path)
    validate_input_paths(config)
    selected_domain = domain_name or next(iter(config["domains"]))
    domain = build_domain(selected_domain, config["domains"][selected_domain], config)

    matrices: dict[str, MatrixRelation] = {}
    pair_tables: dict[str, PairRelation] = {}
    entity_tables: dict[str, EntityTable] = {}

    for role, item in config["inputs"].items():
        contract = item["contract"]
        if contract["kind"] == "matrix":
            values = align_matrix(read_matrix(item["path"], contract), domain)
            matrices[role] = MatrixRelation(role, values, contract)
        elif contract["kind"] == "pair_table":
            frame = read_table(item["path"], contract)
            pair_tables[role] = PairRelation(
                role,
                align_pair_table(frame, domain),
                contract,
                len(frame),
            )
        else:
            frame = read_table(item["path"], contract)
            entity_tables[role] = EntityTable(
                role,
                align_entity_table(frame, domain),
                contract,
                len(frame),
            )

    return RelationBundle(domain, matrices, pair_tables, entity_tables)


def main() -> None:
    """Provide a relation-loading check entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--domain", help="Configured domain name; defaults to the first domain.")
    args = parser.parse_args()

    bundle = load_relations(args.config, args.domain)
    print(
        f"PASS  domain {bundle.domain.name}: "
        f"{len(bundle.domain.entity_a)} x {len(bundle.domain.entity_b)}"
    )
    for role, relation in bundle.matrices.items():
        observed = int(relation.values.notna().to_numpy().sum())
        print(f"PASS  {role}: matrix {relation.values.shape[0]} x {relation.values.shape[1]}, observed {observed}")
    for role, relation in bundle.pair_tables.items():
        print(f"PASS  {role}: pair_table {len(relation.frame)}/{relation.source_rows} rows in domain")
    for role, relation in bundle.entity_tables.items():
        print(f"PASS  {role}: entity_table {len(relation.frame)}/{relation.source_rows} rows in domain")
    print("PASS  relation bundle")


if __name__ == "__main__":
    main()
