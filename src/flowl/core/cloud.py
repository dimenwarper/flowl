"""CloudState (JAX pytree) and Cloud (user handle)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from flowl._types import Array
from flowl.core.constraints import CoversConstraint, DriftConstraint
from flowl.core.graph import CloudMeta
from flowl.tracing.context import get_graph


class CloudState:
    """JAX-compatible pytree holding particle positions and weights."""

    positions: Array  # (n, dim)
    weights: Array  # (n,)
    fixed: bool

    def __init__(self, positions: Array, weights: Array, fixed: bool = False):
        self.positions = positions
        self.weights = weights
        self.fixed = fixed


def _cloud_state_flatten(cs: CloudState) -> tuple[tuple[Array, Array], bool]:
    return (cs.positions, cs.weights), cs.fixed


def _cloud_state_unflatten(fixed: bool, children: tuple[Array, Array]) -> CloudState:
    return CloudState(positions=children[0], weights=children[1], fixed=fixed)


jax.tree_util.register_pytree_node(
    CloudState,
    _cloud_state_flatten,
    _cloud_state_unflatten,
)


class Cloud:
    """User-facing handle for a cloud of particles.

    Created inside a ``with Space():`` block. Registers itself and any
    constraints into the active ModelGraph.
    """

    def __init__(self, name: str, *, n_particles: int = 100, dim: int = 1):
        self.name = name
        self.n_particles = n_particles
        self.dim = dim
        self._fixed = False
        self._initial_positions: Array | None = None

        graph = get_graph()
        graph.add_cloud(CloudMeta(name=name, n_particles=n_particles, dim=dim, fixed=False))

    # --- Factory -----------------------------------------------------------

    @classmethod
    def from_samples(cls, name: str, data: Array, *, fixed: bool = False) -> Cloud:
        """Create a cloud initialized at observed sample positions."""
        data = jnp.asarray(data)
        if data.ndim == 1:
            data = data[:, None]
        n, dim = data.shape

        graph = get_graph()
        graph.add_cloud(CloudMeta(name=name, n_particles=n, dim=dim, fixed=fixed))

        cloud = object.__new__(cls)
        cloud.name = name
        cloud.n_particles = n
        cloud.dim = dim
        cloud._fixed = fixed
        cloud._initial_positions = data
        return cloud

    # --- Constraint builders -----------------------------------------------

    def drift(self, child: Cloud, *, elasticity: float = 1.0) -> Cloud:
        """Add a drift (spring) constraint from *self* to *child*.

        Returns *child* for chaining.
        """
        graph = get_graph()
        graph.add_constraint(DriftConstraint(parent=self.name, child=child.name, elasticity=elasticity))
        return child

    def covers(self, data: Array) -> Cloud:
        """Anchor this cloud to observed data."""
        data = jnp.asarray(data)
        if data.ndim == 1:
            data = data[:, None]
        graph = get_graph()
        graph.add_constraint(CoversConstraint(cloud=self.name, data=data))
        if self._initial_positions is None:
            self._initial_positions = data
            self.n_particles = data.shape[0]
            self.dim = data.shape[1]
            # Update metadata in graph
            graph.clouds[self.name] = CloudMeta(
                name=self.name, n_particles=self.n_particles, dim=self.dim, fixed=self._fixed
            )
        return self
