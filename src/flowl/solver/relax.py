"""Main solver loop and Posterior result class."""

from __future__ import annotations

import jax.numpy as jnp

from flowl._types import Array
from flowl.core.cloud import Cloud, CloudState
from flowl.core.graph import FrozenGraph, ModelGraph
from flowl.core.space import Space
from flowl.solver.energy import total_energy
from flowl.solver.jko import make_step_fn


class Posterior:
    """Result of ``relax()``.  Provides access to solved cloud states."""

    def __init__(self, cloud_states: dict[str, CloudState], energy: float):
        self._states = cloud_states
        self._energy = energy

    def __getitem__(self, name: str) -> CloudState:
        return self._states[name]

    @property
    def energy(self) -> float:
        return self._energy

    def positions(self, name: str) -> Array:
        return self._states[name].positions

    def mean(self, name: str) -> Array:
        cs = self._states[name]
        return jnp.average(cs.positions, axis=0, weights=cs.weights)

    def std(self, name: str) -> Array:
        cs = self._states[name]
        mu = self.mean(name)
        var = jnp.average((cs.positions - mu) ** 2, axis=0, weights=cs.weights)
        return jnp.sqrt(var)


def _init_cloud_states(
    graph: ModelGraph,
    clouds: dict[str, Cloud],
) -> dict[str, CloudState]:
    """Initialize CloudState for every cloud in the graph."""
    states: dict[str, CloudState] = {}

    for name, meta in graph.clouds.items():
        cloud = clouds.get(name)

        if cloud is not None and cloud._initial_positions is not None:
            positions = cloud._initial_positions
        else:
            positions = _init_from_graph(name, graph, clouds)

        n, dim = positions.shape
        weights = jnp.ones(n) / n
        fixed = meta.fixed
        states[name] = CloudState(positions=positions, weights=weights, fixed=fixed)

    return states


def _init_from_graph(
    name: str,
    graph: ModelGraph,
    clouds: dict[str, Cloud],
) -> Array:
    """Initialize a free cloud from downstream observed data."""
    from flowl.core.constraints import DriftConstraint, CoversConstraint

    all_data = []
    visited = set()
    queue = [name]

    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)

        for c in graph.constraints:
            if isinstance(c, CoversConstraint) and c.cloud == current:
                all_data.append(c.data)
            elif isinstance(c, DriftConstraint) and c.parent == current:
                child_cloud = clouds.get(c.child)
                if child_cloud is not None and child_cloud._initial_positions is not None:
                    all_data.append(child_cloud._initial_positions)
                queue.append(c.child)

    meta = graph.clouds[name]

    if all_data:
        combined = jnp.concatenate(all_data, axis=0)
        mean = jnp.mean(combined, axis=0)
        std = jnp.std(combined, axis=0) + 1e-6
        n = meta.n_particles
        t = jnp.linspace(-1, 1, n)[:, None]
        positions = mean[None, :] + t * std[None, :]
    else:
        positions = jnp.zeros((meta.n_particles, meta.dim))

    return positions


def relax(
    space: Space,
    clouds: dict[str, Cloud] | None = None,
    *,
    lr: float = 0.01,
    epsilon: float = 0.05,
    max_steps: int = 200,
    tol: float = 1e-4,
) -> Posterior:
    """Run Wasserstein gradient flow until convergence.

    Parameters
    ----------
    space : Space
        The space context containing the model graph.
    clouds : dict, optional
        Mapping of cloud names to Cloud objects. If None, inferred from graph.
    lr : float
        Learning rate for gradient descent.
    epsilon : float
        Sinkhorn regularization parameter.
    max_steps : int
        Maximum number of JKO steps.
    tol : float
        Convergence tolerance on energy delta.
    """
    graph = space.graph
    frozen = graph.freeze()

    if clouds is None:
        clouds = {}

    cloud_states = _init_cloud_states(graph, clouds)

    # Build JIT-compiled step function
    step_fn, free_names = make_step_fn(cloud_states, frozen, lr, epsilon)

    # Extract arrays for the JIT loop
    free_positions = [cloud_states[name].positions for name in free_names]
    free_weights = [cloud_states[name].weights for name in free_names]
    fixed_weights = {
        name: cs.weights for name, cs in cloud_states.items() if cs.fixed
    }

    prev_energy = float("inf")
    for step in range(max_steps):
        new_positions, energy = step_fn(free_positions, fixed_weights, free_weights)
        e = float(energy)

        if abs(prev_energy - e) < tol:
            break
        prev_energy = e
        free_positions = new_positions

    # Reconstruct final cloud states
    final_states = {}
    for name, cs in cloud_states.items():
        if name in free_names:
            idx = free_names.index(name)
            final_states[name] = CloudState(
                positions=free_positions[idx],
                weights=cs.weights,
                fixed=cs.fixed,
            )
        else:
            final_states[name] = cs

    return Posterior(final_states, e)
