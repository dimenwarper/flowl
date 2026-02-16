"""Tests for the solver components."""

import jax.numpy as jnp

from flowl.core.cloud import CloudState
from flowl.core.constraints import DriftConstraint, CoversConstraint
from flowl.core.graph import FrozenGraph, CloudMeta
from flowl.solver.energy import total_energy


class TestEnergy:
    def test_covers_energy_is_zero_at_data(self):
        """Energy should be near zero when cloud matches data exactly."""
        data = jnp.array([[0.0], [1.0], [2.0]])
        cs = CloudState(data, jnp.ones(3) / 3, fixed=False)
        graph = FrozenGraph(
            clouds={"a": CloudMeta("a", 3, 1, False)},
            constraints=(CoversConstraint("a", data),),
        )
        e = total_energy({"a": cs}, graph, epsilon=0.05)
        assert float(e) < 0.01

    def test_drift_energy_positive_when_apart(self):
        """Drift energy should be positive when clouds are apart."""
        p = CloudState(jnp.array([[0.0]]), jnp.array([1.0]), fixed=False)
        c = CloudState(jnp.array([[10.0]]), jnp.array([1.0]), fixed=False)
        graph = FrozenGraph(
            clouds={
                "p": CloudMeta("p", 1, 1, False),
                "c": CloudMeta("c", 1, 1, False),
            },
            constraints=(DriftConstraint("p", "c", 1.0),),
        )
        e = total_energy({"p": p, "c": c}, graph, epsilon=0.05)
        assert float(e) > 0.1
