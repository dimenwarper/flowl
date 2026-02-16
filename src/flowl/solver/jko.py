"""JKO step: gradient descent on free particle positions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flowl.core.cloud import CloudState
from flowl.core.graph import FrozenGraph
from flowl.solver.energy import total_energy


def make_step_fn(
    cloud_states: dict[str, CloudState],
    graph: FrozenGraph,
    lr: float,
    epsilon: float,
):
    """Build a JIT-compiled step function for the given graph structure.

    Returns a function: free_positions_flat -> (new_free_positions_flat, energy)
    where the flat representation is a list of position arrays for free clouds.
    """
    free_names = sorted(name for name, cs in cloud_states.items() if not cs.fixed)
    fixed_data = {
        name: cs for name, cs in cloud_states.items() if cs.fixed
    }

    @jax.jit
    def step(free_positions: list[jnp.ndarray], fixed_weights: dict[str, jnp.ndarray], free_weights: list[jnp.ndarray]):
        def energy_fn(pos_list: list[jnp.ndarray]) -> jnp.ndarray:
            updated = {}
            for i, name in enumerate(free_names):
                updated[name] = CloudState(
                    positions=pos_list[i],
                    weights=free_weights[i],
                    fixed=False,
                )
            for name, cs in fixed_data.items():
                updated[name] = cs
            return total_energy(updated, graph, epsilon=epsilon)

        energy = energy_fn(free_positions)
        grads = jax.grad(energy_fn)(free_positions)
        new_positions = [p - lr * g for p, g in zip(free_positions, grads)]
        return new_positions, energy

    return step, free_names


def jko_step(
    cloud_states: dict[str, CloudState],
    graph: FrozenGraph,
    lr: float = 0.01,
    epsilon: float = 0.05,
) -> dict[str, CloudState]:
    """One gradient-descent step on free cloud positions (non-JIT version)."""
    free_names = [name for name, cs in cloud_states.items() if not cs.fixed]

    def energy_fn(free_positions: dict[str, jnp.ndarray]) -> jnp.ndarray:
        updated = {}
        for name, cs in cloud_states.items():
            if name in free_positions:
                updated[name] = CloudState(
                    positions=free_positions[name],
                    weights=cs.weights,
                    fixed=cs.fixed,
                )
            else:
                updated[name] = cs
        return total_energy(updated, graph, epsilon=epsilon)

    free_positions = {name: cloud_states[name].positions for name in free_names}
    grads = jax.grad(energy_fn)(free_positions)

    new_states = {}
    for name, cs in cloud_states.items():
        if name in grads:
            new_positions = cs.positions - lr * grads[name]
            new_states[name] = CloudState(
                positions=new_positions,
                weights=cs.weights,
                fixed=cs.fixed,
            )
        else:
            new_states[name] = cs

    return new_states
