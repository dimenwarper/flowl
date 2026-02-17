"""Tests for configurable space geometry."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from flowl.core.geometry import SpaceGeometry, resolve_geometry


class TestResolveGeometry:
    """Unit tests for the resolve_geometry factory."""

    def test_default_is_pointcloud_no_cost(self):
        geo = resolve_geometry()
        assert geo.kind == "pointcloud"
        assert geo.cost_fn is None

    def test_sq_euclidean_string(self):
        geo = resolve_geometry("sq_euclidean")
        assert geo.kind == "pointcloud"
        assert geo.cost_fn is None

    def test_euclidean_string(self):
        geo = resolve_geometry("euclidean")
        assert geo.kind == "pointcloud"
        from ott.geometry.costs import Euclidean
        assert isinstance(geo.cost_fn, Euclidean)

    def test_cosine_string(self):
        geo = resolve_geometry("cosine")
        assert geo.kind == "pointcloud"
        from ott.geometry.costs import Cosine
        assert isinstance(geo.cost_fn, Cosine)

    def test_hyperbolic_string(self):
        geo = resolve_geometry("hyperbolic")
        assert geo.kind == "pointcloud"
        from flowl.costs.hyperbolic import HyperbolicCost
        assert isinstance(geo.cost_fn, HyperbolicCost)

    def test_custom_cost_fn(self):
        from ott.geometry.costs import PNormP
        cost = PNormP(p=1.5)
        geo = resolve_geometry(cost)
        assert geo.kind == "pointcloud"
        assert geo.cost_fn is cost

    def test_graph_requires_laplacian(self):
        with pytest.raises(ValueError, match="graph_laplacian"):
            resolve_geometry("graph")

    def test_graph_with_laplacian(self):
        L = jnp.eye(3)
        geo = resolve_geometry("graph", graph_laplacian=L)
        assert geo.kind == "graph"
        assert geo.graph_laplacian is not None

    def test_grid_requires_xs(self):
        with pytest.raises(ValueError, match="grid_xs"):
            resolve_geometry("grid")

    def test_grid_with_xs(self):
        xs = (jnp.linspace(0, 1, 5),)
        geo = resolve_geometry("grid", grid_xs=xs)
        assert geo.kind == "grid"
        assert geo.grid_xs is not None

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown geometry"):
            resolve_geometry("manhattan")


class TestSpaceGeometryFlowsThrough:
    """Integration: geometry set on Space reaches the frozen graph."""

    def test_default_space_geometry(self):
        from flowl import Space
        with Space() as space:
            pass
        geo = space.graph.geometry
        assert geo.kind == "pointcloud"
        assert geo.cost_fn is None

    def test_euclidean_space_geometry(self):
        from flowl import Space
        with Space(geometry="euclidean") as space:
            pass
        geo = space.graph.geometry
        assert geo.kind == "pointcloud"
        from ott.geometry.costs import Euclidean
        assert isinstance(geo.cost_fn, Euclidean)

    def test_geometry_on_frozen_graph(self):
        from flowl import Space
        with Space(geometry="cosine") as space:
            pass
        frozen = space.graph.freeze()
        assert frozen.geometry.kind == "pointcloud"
        from ott.geometry.costs import Cosine
        assert isinstance(frozen.geometry.cost_fn, Cosine)


class TestDifferentGeometriesDifferentEnergies:
    """Different cost functions should yield different Sinkhorn divergences."""

    def test_euclidean_vs_cosine_sinkdiv(self):
        from flowl.backends.sinkhorn import sinkdiv

        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = jnp.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])

        geo_euc = resolve_geometry("euclidean")
        geo_cos = resolve_geometry("cosine")

        div_euc = float(sinkdiv(x, y, geometry=geo_euc))
        div_cos = float(sinkdiv(x, y, geometry=geo_cos))

        # Both should be non-negative
        assert div_euc >= 0
        assert div_cos >= 0
        # They should differ
        assert div_euc != pytest.approx(div_cos, abs=1e-6)

    def test_custom_pnorm_cost(self):
        from ott.geometry.costs import PNormP
        from flowl.backends.sinkhorn import sinkdiv

        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        y = jnp.array([[0.5, 0.5], [1.0, 0.0]])

        geo_default = resolve_geometry()
        geo_p15 = resolve_geometry(PNormP(p=1.5))

        div_default = float(sinkdiv(x, y, geometry=geo_default))
        div_p15 = float(sinkdiv(x, y, geometry=geo_p15))

        assert div_default >= 0
        assert div_p15 >= 0

    def test_hyperbolic_vs_euclidean_sinkdiv(self):
        from flowl.backends.sinkhorn import sinkdiv

        # Points well inside the unit ball
        x = jnp.array([[0.3, 0.1], [0.0, 0.4], [0.2, 0.2]])
        y = jnp.array([[0.1, 0.3], [0.4, 0.0], [0.0, 0.2]])

        geo_euc = resolve_geometry("euclidean")
        geo_hyp = resolve_geometry("hyperbolic")

        div_euc = float(sinkdiv(x, y, geometry=geo_euc))
        div_hyp = float(sinkdiv(x, y, geometry=geo_hyp))

        assert div_euc >= 0
        assert div_hyp >= 0
        assert div_euc != pytest.approx(div_hyp, abs=1e-6)
