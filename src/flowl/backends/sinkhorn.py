"""OTT-JAX wrapper for Sinkhorn divergence and OT solvers."""

from __future__ import annotations

import jax.numpy as jnp
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear import sinkhorn
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from flowl._types import Array


def sinkdiv(
    x: Array,
    y: Array,
    *,
    epsilon: float = 0.05,
) -> Array:
    """Compute the Sinkhorn divergence between two point clouds.

    Returns a scalar JAX array.
    """
    div, _ = sinkhorn_divergence(
        PointCloud,
        x,
        y,
        epsilon=epsilon,
    )
    return div


def solve_ot(
    x: Array,
    y: Array,
    *,
    epsilon: float = 0.05,
) -> sinkhorn.SinkhornOutput:
    """Solve the OT problem between x and y, returning the full output."""
    geom = PointCloud(x, y, epsilon=epsilon)
    solver = sinkhorn.Sinkhorn()
    return solver(sinkhorn.linear_problem.LinearProblem(geom))
