"""Define generic relation models, score composition, and elementary losses.

This module contains no optimizer, sampling loop, cross-validation logic, or
file output.  Training code can therefore reuse the same mathematical model
without changing the declared relation semantics.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from config import load_config
from transforms import TransformBundle, build_transforms


DIRECTIONS = {"higher_is_better", "lower_is_better"}
DECODER_OUTPUTS = {"identity", "sigmoid"}


def inverse_softplus(value: float) -> float:
    """Return the inverse softplus of a positive scalar."""
    if value <= 0:
        raise ValueError("inverse_softplus requires a positive value")
    return math.log(math.expm1(value))


class UncertaintyLossBalancer(nn.Module):
    """Learn positive auxiliary-loss weights relative to a fixed primary task,
    while the log term prevents the optimizer from escaping by setting a
    difficult task weight to zero.
    """

    def __init__(
        self,
        initial_weights: Mapping[str, float],
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        super().__init__()
        if (
            not initial_weights
            or lower_bound <= 0
            or upper_bound <= lower_bound
            or any(value <= 0 for value in initial_weights.values())
        ):
            raise ValueError("Adaptive loss weights require positive initials and valid bounds")
        self.names = tuple(initial_weights)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.log_variance = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(-math.log(float(weight))))
                for name, weight in initial_weights.items()
            }
        )
        self.project()

    def weights(self) -> dict[str, torch.Tensor]:
        """Return current effective auxiliary-loss weights."""
        return {
            name: torch.exp(-self.log_variance[name])
            for name in self.names
        }

    def objective(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Compute the collapse-resistant auxiliary objective."""
        if set(losses) != set(self.names):
            raise ValueError("Adaptive loss names must exactly match configured names")
        weights = self.weights()
        return torch.stack(
            [
                weights[name] * losses[name] + self.log_variance[name]
                for name in self.names
            ]
        ).sum()

    @torch.no_grad()
    def project(self) -> None:
        """Project effective weights to the declared range."""
        minimum_log_variance = -math.log(self.upper_bound)
        maximum_log_variance = -math.log(self.lower_bound)
        for parameter in self.log_variance.values():
            parameter.clamp_(minimum_log_variance, maximum_log_variance)

    def detached_weights(self) -> dict[str, float]:
        """Return plain floating-point weights for audit."""
        return {
            name: float(value.detach().cpu())
            for name, value in self.weights().items()
        }


@dataclass(frozen=True)
class ModelArchitecture:
    """Store the configured model-structure mapping."""

    primary_input: str
    primary_direction: str
    coefficient_bounds: tuple[float, float]
    backbone_relations: tuple[str, ...]
    decoder_outputs: dict[str, str]
    primary_feature: str
    auxiliary_feature: str
    sparse_features: tuple[str, ...]

    @property
    def scalar_feature_names(self) -> tuple[str, ...]:
        """Return feature names before latent interactions."""
        names = [
            "auxiliary_minus_primary",
            "primary_auxiliary_centered_interaction",
            "primary_relation_reconstruction",
            "auxiliary_relation_reconstruction",
            "shared_latent_dot",
        ]
        for name in self.sparse_features:
            names.extend((f"{name}_value", f"{name}_observed"))
        return tuple(names)

    def feature_names(self, embedding_dim: int) -> tuple[str, ...]:
        """Return complete ordered model feature names."""
        latent = tuple(f"latent_interaction_{index + 1}" for index in range(embedding_dim))
        return self.scalar_feature_names + latent


class RelationalBackbone(nn.Module):
    """Represent multiple relations with shared entity embeddings and
    relation-specific diagonal decoders.
    """

    def __init__(
        self,
        n_entity_a: int,
        n_entity_b: int,
        embedding_dim: int,
        relation_names: tuple[str, ...],
        decoder_outputs: Mapping[str, str],
    ) -> None:
        super().__init__()
        if min(n_entity_a, n_entity_b, embedding_dim) <= 0:
            raise ValueError("entity counts and embedding_dim must be positive")
        if not relation_names or len(set(relation_names)) != len(relation_names):
            raise ValueError("relation_names must be non-empty and unique")
        invalid = set(decoder_outputs.values()) - DECODER_OUTPUTS
        if invalid or set(decoder_outputs) != set(relation_names):
            raise ValueError("decoder_outputs must define every relation exactly once")

        self.embedding_dim = embedding_dim
        self.relation_names = relation_names
        self.relation_index = {name: index for index, name in enumerate(relation_names)}
        self.decoder_outputs = dict(decoder_outputs)
        self.entity_a = nn.Embedding(n_entity_a, embedding_dim)
        self.entity_b = nn.Embedding(n_entity_b, embedding_dim)
        self.relation_scale = nn.Parameter(torch.ones(len(relation_names), embedding_dim))
        self.relation_bias = nn.Parameter(torch.zeros(len(relation_names)))
        nn.init.normal_(self.entity_a.weight, std=0.05)
        nn.init.normal_(self.entity_b.weight, std=0.05)

    def interaction(self, entity_a: torch.Tensor, entity_b: torch.Tensor) -> torch.Tensor:
        """Return elementwise shared latent interactions."""
        return (
            self.entity_a(entity_a) * self.entity_b(entity_b)
            / math.sqrt(self.embedding_dim)
        )

    def shared_dot(self, entity_a: torch.Tensor, entity_b: torch.Tensor) -> torch.Tensor:
        """Return the shared latent dot product."""
        return self.interaction(entity_a, entity_b).sum(dim=1)

    def decode_raw(
        self,
        entity_a: torch.Tensor,
        entity_b: torch.Tensor,
        relation: str,
    ) -> torch.Tensor:
        """Return unactivated scores for one relation."""
        return self.decode_interaction(
            self.interaction(entity_a, entity_b),
            relation,
        )

    def decode_interaction(
        self,
        interaction: torch.Tensor,
        relation: str,
    ) -> torch.Tensor:
        """Decode from cached interactions."""
        index = self.relation_index[relation]
        return (
            (interaction * self.relation_scale[index]).sum(dim=1)
            + self.relation_bias[index]
        )

    def decode(
        self,
        entity_a: torch.Tensor,
        entity_b: torch.Tensor,
        relation: str,
    ) -> torch.Tensor:
        """Decode one relation with its declared output type."""
        raw = self.decode_raw(entity_a, entity_b, relation)
        return torch.sigmoid(raw) if self.decoder_outputs[relation] == "sigmoid" else raw


class ResidualScorer(nn.Module):
    """Learn a shared residual, binary calibration, and a bounded primary-relation
    coefficient.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        coefficient_bounds: tuple[float, float] = (0.0, 1.0),
        learnable_coefficient: bool = True,
    ) -> None:
        super().__init__()
        lower, upper = coefficient_bounds
        if min(n_features, hidden_dim) <= 0 or lower > 1.0 or upper < 1.0 or lower >= upper:
            raise ValueError("invalid scorer dimensions or coefficient bounds")
        self.coefficient_bounds = coefficient_bounds
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.binary_bias = nn.Parameter(torch.zeros(()))
        self.binary_scale_raw = nn.Parameter(torch.tensor(inverse_softplus(1.0)))
        self.primary_coefficient_raw = nn.Parameter(
            torch.tensor(1.0),
            requires_grad=learnable_coefficient,
        )

    @staticmethod
    def positive_scale(raw: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained parameter to a positive scale."""
        return F.softplus(raw) + 1e-4

    def primary_coefficient(self) -> torch.Tensor:
        """Return the box-constrained primary coefficient."""
        lower, upper = self.coefficient_bounds
        return torch.clamp(self.primary_coefficient_raw, min=lower, max=upper)

    @property
    def coefficient_is_learnable(self) -> bool:
        """Report whether the primary coefficient is trainable."""
        return self.primary_coefficient_raw.requires_grad

    @torch.no_grad()
    def project_primary_coefficient(self) -> None:
        """Project the primary coefficient after optimization."""
        lower, upper = self.coefficient_bounds
        self.primary_coefficient_raw.clamp_(lower, upper)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        """Compute the shared residual written to the final matrix."""
        return self.residual_head(self.shared(features)).squeeze(1)

    def binary_logits(self, features: torch.Tensor, base_signal: torch.Tensor) -> torch.Tensor:
        """Compute calibrated logits for binary supervision."""
        return (
            self.binary_bias
            + self.positive_scale(self.binary_scale_raw) * base_signal
            + self.residual(features)
        )

    def final_score(
        self,
        primary_score: torch.Tensor,
        residual: torch.Tensor,
        correction_scale: float,
        direction: str,
    ) -> torch.Tensor:
        """Combine primary scores and residuals in the declared direction."""
        require_direction(direction)
        sign = 1.0 if direction == "higher_is_better" else -1.0
        return self.primary_coefficient() * primary_score + sign * correction_scale * residual


@dataclass
class ModelBundle:
    """Collect model structures and declarative semantics."""

    architecture: ModelArchitecture
    backbone: RelationalBackbone
    scorer: ResidualScorer
    transforms: TransformBundle
    feature_names: tuple[str, ...]


def build_pair_features(
    backbone: RelationalBackbone,
    architecture: ModelArchitecture,
    entity_a: torch.Tensor,
    entity_b: torch.Tensor,
    values: Mapping[str, torch.Tensor],
    observed: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Build generic pair features matching the frozen model."""
    required = {architecture.primary_feature, architecture.auxiliary_feature, *architecture.sparse_features}
    missing = required - set(values)
    if missing:
        raise ValueError(f"Missing pair feature values: {sorted(missing)}")
    missing_masks = set(architecture.sparse_features) - set(observed)
    if missing_masks:
        raise ValueError(f"Missing sparse observation masks: {sorted(missing_masks)}")

    primary = values[architecture.primary_feature].float()
    auxiliary = values[architecture.auxiliary_feature].float()
    interaction = backbone.interaction(entity_a, entity_b)
    primary_reconstruction = backbone.decode_interaction(
        interaction,
        architecture.primary_feature,
    )
    auxiliary_reconstruction = backbone.decode_interaction(
        interaction,
        architecture.auxiliary_feature,
    )
    if (
        backbone.decoder_outputs[architecture.primary_feature]
        == "sigmoid"
    ):
        primary_reconstruction = torch.sigmoid(primary_reconstruction)
    if (
        backbone.decoder_outputs[architecture.auxiliary_feature]
        == "sigmoid"
    ):
        auxiliary_reconstruction = torch.sigmoid(
            auxiliary_reconstruction
        )
    columns = [
        auxiliary - primary,
        (primary - 0.5) * (auxiliary - 0.5),
        primary_reconstruction,
        auxiliary_reconstruction,
        interaction.sum(dim=1),
    ]
    for name in architecture.sparse_features:
        columns.extend(
            (
                torch.nan_to_num(values[name].float(), nan=0.0),
                observed[name].float(),
            )
        )
    columns.extend(interaction.unbind(dim=1))
    return torch.stack(columns, dim=1)


def dense_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute mean squared reconstruction loss on observed cells."""
    if observed is not None:
        prediction = prediction[observed.bool()]
        target = target[observed.bool()]
    if prediction.numel() == 0:
        raise ValueError("dense reconstruction loss received no observed values")
    return F.mse_loss(prediction, target)


def pairwise_ranking_loss(
    positive_score: torch.Tensor,
    negative_score: torch.Tensor,
    direction: str,
    temperature: float = 1.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a direction-aware pairwise logistic loss."""
    require_direction(direction)
    if temperature <= 0 or positive_score.shape != negative_score.shape:
        raise ValueError("temperature must be positive and paired scores must match")
    wrong_order_margin = (
        positive_score - negative_score
        if direction == "lower_is_better"
        else negative_score - positive_score
    )
    losses = F.softplus(wrong_order_margin / temperature)
    if weight is None:
        return losses.mean()
    if weight.shape != losses.shape:
        raise ValueError("pair weights must match paired scores")
    return torch.sum(losses * weight / weight.sum())


def binary_supervision_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute pointwise binary supervision loss."""
    if logits.shape != labels.shape or logits.numel() == 0:
        raise ValueError("binary logits and labels must be non-empty and shape matched")
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def residual_anchor_loss(residual: torch.Tensor) -> torch.Tensor:
    """Penalize standardized full-domain residuals away from zero."""
    if residual.numel() == 0:
        raise ValueError("residual anchor requires at least one value")
    return residual.square().mean()


def require_direction(direction: str) -> None:
    """Validate a score direction."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}, got {direction!r}")


def parse_architecture(config: dict[str, Any], transforms: TransformBundle) -> ModelArchitecture:
    """Parse and validate the declarative model architecture."""
    specification = config.get("model", {}).get("architecture")
    if not isinstance(specification, dict):
        raise ValueError("Configuration must declare model.architecture")
    primary = specification["primary_score"]
    bounds = tuple(float(value) for value in primary.get("coefficient_bounds", [0.0, 1.0]))
    if len(bounds) != 2:
        raise ValueError("primary_score.coefficient_bounds must contain two values")
    direction = primary["direction"]
    require_direction(direction)

    relation_specs = specification["backbone_relations"]
    if not isinstance(relation_specs, dict) or not relation_specs:
        raise ValueError("model.architecture.backbone_relations must be non-empty")
    decoder_outputs = {
        name: item.get("output", "identity") for name, item in relation_specs.items()
    }
    invalid = set(decoder_outputs.values()) - DECODER_OUTPUTS
    if invalid:
        raise ValueError(f"Unsupported decoder outputs: {sorted(invalid)}")

    pair_features = specification["pair_features"]
    primary_feature = pair_features["primary"]
    auxiliary_feature = pair_features["auxiliary"]
    sparse_features = tuple(pair_features.get("sparse", []))
    referenced = {primary_feature, auxiliary_feature, *sparse_features}
    unknown = referenced - set(transforms.matrices)
    if unknown:
        raise ValueError(f"Model architecture references unknown transforms: {sorted(unknown)}")
    if not referenced.issubset(relation_specs):
        raise ValueError("Every pair-feature transform must have a backbone decoder")
    return ModelArchitecture(
        primary_input=primary["input"],
        primary_direction=direction,
        coefficient_bounds=(bounds[0], bounds[1]),
        backbone_relations=tuple(relation_specs),
        decoder_outputs=decoder_outputs,
        primary_feature=primary_feature,
        auxiliary_feature=auxiliary_feature,
        sparse_features=sparse_features,
    )


def build_models(
    config_path: str | Path,
    domain_name: str | None = None,
    learnable_coefficient: bool = True,
) -> ModelBundle:
    """Build untrained models from configuration."""
    config = load_config(config_path)
    transformed = build_transforms(config_path, domain_name)
    architecture = parse_architecture(config, transformed)
    if architecture.primary_input not in config["inputs"]:
        raise ValueError(f"Unknown primary-score input: {architecture.primary_input}")
    parameters = config["model"]
    embedding_dim = int(parameters["embedding_dim"])
    hidden_dim = int(parameters["head_hidden_dim"])
    backbone = RelationalBackbone(
        len(transformed.domain.entity_a),
        len(transformed.domain.entity_b),
        embedding_dim,
        architecture.backbone_relations,
        architecture.decoder_outputs,
    )
    feature_names = architecture.feature_names(embedding_dim)
    scorer = ResidualScorer(
        len(feature_names),
        hidden_dim,
        architecture.coefficient_bounds,
        learnable_coefficient,
    )
    return ModelBundle(architecture, backbone, scorer, transformed, feature_names)


def main() -> None:
    """Build models and run an untrained forward check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--domain", help="Configured domain name; defaults to the first domain.")
    args = parser.parse_args()

    bundle = build_models(args.config, args.domain)
    architecture = bundle.architecture
    batch_size = 8
    entity_a = torch.arange(batch_size) % bundle.backbone.entity_a.num_embeddings
    entity_b = torch.arange(batch_size) % bundle.backbone.entity_b.num_embeddings
    values = {name: torch.linspace(0.1, 0.8, batch_size) for name in architecture.backbone_relations}
    observed = {name: torch.ones(batch_size, dtype=torch.bool) for name in architecture.sparse_features}
    features = build_pair_features(
        bundle.backbone,
        architecture,
        entity_a,
        entity_b,
        values,
        observed,
    )
    residual = bundle.scorer.residual(features)
    initial = torch.linspace(-8.0, -4.0, batch_size)
    final = bundle.scorer.final_score(initial, residual, 1.0, architecture.primary_direction)
    if features.shape[1] != len(bundle.feature_names) or not torch.allclose(final, initial):
        raise RuntimeError("Untrained model invariants failed")

    print(
        f"PASS  domain {bundle.transforms.domain.name}: "
        f"{bundle.backbone.entity_a.num_embeddings} x {bundle.backbone.entity_b.num_embeddings}"
    )
    print(
        f"PASS  relational backbone: {len(architecture.backbone_relations)} relations, "
        f"embedding_dim {bundle.backbone.embedding_dim}"
    )
    print(f"PASS  pair features: {len(bundle.feature_names)}")
    print(
        f"PASS  primary coefficient: [{architecture.coefficient_bounds[0]:g}, "
        f"{architecture.coefficient_bounds[1]:g}], learnable"
    )
    print("PASS  zero-initialized residual preserves the primary score")
    print("PASS  model bundle")


if __name__ == "__main__":
    main()
