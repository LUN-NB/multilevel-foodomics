"""Load and validate declarative relation-model configurations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = ("inputs", "axis_constraints", "domains", "coverage")
SUPPORTED_KINDS = {"matrix", "pair_table", "entity_table"}
SUPPORTED_PROTOCOLS = {
    "formal",
    "adaptive_phase1_dev",
    "adaptive_phase2_4gb_dev",
}


def validate_config(config: dict[str, Any]) -> None:
    """Validate the declarative configuration structure."""
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")

    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    if config.get("protocol") not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(SUPPORTED_PROTOCOLS)}")
    if not isinstance(config["inputs"], dict) or not config["inputs"]:
        raise ValueError("inputs must be a non-empty mapping")

    for role, item in config["inputs"].items():
        if not isinstance(item, dict) or "path" not in item or "contract" not in item:
            raise ValueError(f"{role}: input must declare path and contract")
        contract = item["contract"]
        if not isinstance(contract, dict) or contract.get("kind") not in SUPPORTED_KINDS:
            raise ValueError(f"{role}: unsupported or missing contract kind")
        if "entity_a_column" not in contract:
            raise ValueError(f"{role}: contract must declare entity_a_column")
        if contract["kind"] == "pair_table" and "entity_b_column" not in contract:
            raise ValueError(f"{role}: pair_table must declare entity_b_column")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the configuration and resolve input paths."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    validate_config(config)
    project_root = (path.parent / config.get("path_base", ".")).resolve()
    for item in config["inputs"].values():
        item["path"] = (project_root / item["path"]).resolve()
    config["_config_path"] = path
    config["_project_root"] = project_root
    return config


def validate_input_paths(config: dict[str, Any]) -> None:
    """Check every standardized input file."""
    invalid = [
        f"{role}: {item['path']}"
        for role, item in config["inputs"].items()
        if not item["path"].is_file()
    ]
    if invalid:
        raise FileNotFoundError("Missing configured input files:\n" + "\n".join(invalid))


def main() -> None:
    """Provide a standalone configuration check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.check_inputs:
        validate_input_paths(config)

    print(f"PASS  config: {config['_config_path']}")
    print(f"PASS  protocol: {config['protocol']}")
    print(f"PASS  declared inputs: {len(config['inputs'])}")
    if args.check_inputs:
        print("PASS  configured input files")


if __name__ == "__main__":
    main()
