"""Total energy function over constraints."""

from __future__ import annotations

import jax.numpy as jnp

from flowl._types import Array
from flowl.backends.sinkhorn import sinkdiv
from flowl.core.constraints import CoversConstraint, DriftConstraint, WarpConstraint
from flowl.core.cloud import CloudState
from flowl.core.graph import FrozenGraph


def total_energy(
    cloud_states: dict[str, CloudState],
    graph: FrozenGraph,
    epsilon: float = 0.05,
) -> Array:
    """Pure function: sum of Sinkhorn divergences over all constraints."""
    energy = jnp.float32(0.0)
    geometry = graph.geometry

    for constraint in graph.constraints:
        if isinstance(constraint, DriftConstraint):
            parent = cloud_states[constraint.parent]
            child = cloud_states[constraint.child]
            energy = energy + constraint.elasticity * sinkdiv(
                parent.positions, child.positions,
                geometry=geometry, epsilon=epsilon,
            )

        elif isinstance(constraint, CoversConstraint):
            cloud = cloud_states[constraint.cloud]
            energy = energy + sinkdiv(
                cloud.positions, constraint.data,
                geometry=geometry, epsilon=epsilon,
            )

        elif isinstance(constraint, WarpConstraint):
            source = cloud_states[constraint.source]
            target = cloud_states[constraint.target]
            div = sinkdiv(
                source.positions, target.positions,
                geometry=geometry, epsilon=epsilon,
            )
            if constraint.max_distance < float("inf"):
                penalty = jnp.maximum(div - constraint.max_distance, 0.0) ** 2
                energy = energy + div + penalty
            else:
                energy = energy + div

    return energy
