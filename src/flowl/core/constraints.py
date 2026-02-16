"""Constraint types for the flowl model graph."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from flowl._types import Array


@dataclass(frozen=True)
class DriftConstraint:
    """Spring between parent and child clouds.

    Energy = elasticity * sinkdiv(parent, child).
    """

    parent: str
    child: str
    elasticity: float = 1.0


@dataclass(frozen=True)
class CoversConstraint:
    """Anchors a cloud to observed data.

    Energy = sinkdiv(cloud, data).
    """

    cloud: str
    data: Array


@dataclass(frozen=True)
class WarpConstraint:
    """Transport-constrained mapping between clouds.

    Energy = sinkdiv(source, target) + penalty when distance > max_distance.
    """

    source: str
    target: str
    max_distance: float = float("inf")


Constraint = DriftConstraint | CoversConstraint | WarpConstraint
