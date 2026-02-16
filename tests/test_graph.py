"""Tests for ModelGraph and FrozenGraph."""

import pytest

from flowl.core.graph import ModelGraph, FrozenGraph, CloudMeta
from flowl.core.constraints import DriftConstraint


class TestModelGraph:
    def test_add_cloud(self):
        g = ModelGraph()
        g.add_cloud(CloudMeta("a", 10, 2, False))
        assert "a" in g.clouds

    def test_duplicate_cloud_raises(self):
        g = ModelGraph()
        g.add_cloud(CloudMeta("a", 10, 2, False))
        with pytest.raises(ValueError, match="already exists"):
            g.add_cloud(CloudMeta("a", 10, 2, False))

    def test_add_constraint(self):
        g = ModelGraph()
        g.add_constraint(DriftConstraint("a", "b", 1.0))
        assert len(g.constraints) == 1

    def test_freeze(self):
        g = ModelGraph()
        g.add_cloud(CloudMeta("x", 5, 1, False))
        g.add_constraint(DriftConstraint("x", "y", 0.5))
        fg = g.freeze()
        assert isinstance(fg, FrozenGraph)
        assert isinstance(fg.constraints, tuple)
        # Frozen graph is independent
        g.add_cloud(CloudMeta("z", 3, 1, False))
        assert "z" not in fg.clouds
