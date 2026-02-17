"""Morph: trajectory between two clouds via displacement interpolation."""

from __future__ import annotations

import jax.numpy as jnp

from flowl._types import Array
from flowl.backends.interpolation import displacement_interpolation
from flowl.core.cloud import CloudState
from flowl.core.geometry import SpaceGeometry


class Morph:
    """Displacement interpolation trajectory between two cloud states."""

    def __init__(
        self,
        source: CloudState,
        target: CloudState,
        *,
        geometry: SpaceGeometry | None = None,
        epsilon: float = 0.05,
    ):
        self._source = source
        self._target = target
        self._geometry = geometry
        self._epsilon = epsilon

    def at(self, t: float) -> Array:
        """Return interpolated positions at time t in [0, 1]."""
        return displacement_interpolation(
            self._source.positions,
            self._target.positions,
            t=t,
            geometry=self._geometry,
            epsilon=self._epsilon,
        )
