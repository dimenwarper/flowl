"""Displacement interpolation via OT coupling."""

from __future__ import annotations

import jax.numpy as jnp

from flowl._types import Array
from flowl.backends.sinkhorn import solve_ot
from flowl.core.geometry import SpaceGeometry


def displacement_interpolation(
    x: Array,
    y: Array,
    t: float = 0.5,
    *,
    geometry: SpaceGeometry | None = None,
    epsilon: float = 0.05,
) -> Array:
    """Compute displacement interpolation at time *t* between x and y."""
    out = solve_ot(x, y, geometry=geometry, epsilon=epsilon)
    coupling = out.matrix  # (n, m)
    # Normalize rows to get transport plan per source particle
    row_sums = coupling.sum(axis=1, keepdims=True)
    transport = coupling / jnp.maximum(row_sums, 1e-12)
    # Barycentric projection: target for each source point
    target_positions = transport @ y  # (n, dim)
    return (1 - t) * x + t * target_positions
