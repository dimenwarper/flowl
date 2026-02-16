"""Tests for Cloud and CloudState."""

import jax.numpy as jnp
import pytest

from flowl import Cloud, CloudState, Space


class TestCloudState:
    def test_pytree_roundtrip(self):
        import jax

        cs = CloudState(jnp.ones((5, 2)), jnp.ones(5) / 5, fixed=False)
        leaves, treedef = jax.tree_util.tree_flatten(cs)
        cs2 = treedef.unflatten(leaves)
        assert jnp.allclose(cs2.positions, cs.positions)
        assert cs2.fixed == cs.fixed

    def test_fixed_flag(self):
        cs = CloudState(jnp.zeros((3, 1)), jnp.ones(3) / 3, fixed=True)
        assert cs.fixed is True


class TestCloud:
    def test_cloud_creates_in_space(self):
        with Space() as s:
            c = Cloud("mu", n_particles=50, dim=2)
        assert "mu" in s.graph.clouds
        assert s.graph.clouds["mu"].n_particles == 50

    def test_from_samples(self):
        data = jnp.array([1.0, 2.0, 3.0])
        with Space() as s:
            c = Cloud.from_samples("obs", data, fixed=True)
        assert c._initial_positions.shape == (3, 1)
        assert s.graph.clouds["obs"].fixed is True

    def test_drift_adds_constraint(self):
        with Space() as s:
            parent = Cloud("parent", n_particles=10, dim=1)
            child = Cloud("child", n_particles=10, dim=1)
            parent.drift(child, elasticity=2.0)
        assert len(s.graph.constraints) == 1
        assert s.graph.constraints[0].elasticity == 2.0

    def test_covers_adds_constraint(self):
        data = jnp.array([[1.0], [2.0]])
        with Space() as s:
            c = Cloud("c", n_particles=10, dim=1)
            c.covers(data)
        assert len(s.graph.constraints) == 1

    def test_no_space_raises(self):
        with pytest.raises(RuntimeError, match="No active Space"):
            Cloud("orphan")
