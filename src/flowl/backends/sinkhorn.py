"""OTT-JAX wrapper for Sinkhorn divergence and OT solvers."""

from __future__ import annotations

import jax.numpy as jnp
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear import sinkhorn
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from flowl._types import Array
from flowl.core.geometry import SpaceGeometry


def _make_geom(
    x: Array,
    y: Array,
    geometry: SpaceGeometry | None,
    epsilon: float,
) -> PointCloud:
    """Construct the OTT-JAX geometry object from a SpaceGeometry config."""
    if geometry is None or geometry.kind == "pointcloud":
        cost_fn = geometry.cost_fn if geometry is not None else None
        return PointCloud(x, y, cost_fn=cost_fn, epsilon=epsilon)

    if geometry.kind == "graph":
        from ott.geometry.graph import Graph

        return Graph.from_graph(
            geometry.graph_laplacian,
            epsilon=epsilon,
        )

    if geometry.kind == "grid":
        from ott.geometry.grid import Grid

        return Grid(
            *geometry.grid_xs,
            epsilon=epsilon,
        )

    raise ValueError(f"Unsupported geometry kind: {geometry.kind!r}")


def sinkdiv(
    x: Array,
    y: Array,
    *,
    geometry: SpaceGeometry | None = None,
    epsilon: float = 0.05,
) -> Array:
    """Compute the Sinkhorn divergence between two point clouds.

    Returns a scalar JAX array.
    """
    cost_fn = geometry.cost_fn if geometry is not None else None
    div, _ = sinkhorn_divergence(
        PointCloud,
        x,
        y,
        cost_fn=cost_fn,
        epsilon=epsilon,
    )
    return div


def solve_ot(
    x: Array,
    y: Array,
    *,
    geometry: SpaceGeometry | None = None,
    epsilon: float = 0.05,
) -> sinkhorn.SinkhornOutput:
    """Solve the OT problem between x and y, returning the full output."""
    geom = _make_geom(x, y, geometry, epsilon)
    solver = sinkhorn.Sinkhorn()
    return solver(sinkhorn.linear_problem.LinearProblem(geom))
