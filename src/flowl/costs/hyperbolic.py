"""Hyperbolic (Poincaré disk) distance cost function."""

from __future__ import annotations

import jax.numpy as jnp
import jax.tree_util as jtu
from ott.geometry.costs import CostFn


@jtu.register_pytree_node_class
class HyperbolicCost(CostFn):
    """Poincaré disk distance cost function.

    Computes the hyperbolic distance between two points inside the unit disk::

        d(x, y) = arccosh(1 + 2 * ||x - y||² / ((1 - ||x||²)(1 - ||y||²)))

    Args:
        ridge: Ridge regularization to avoid division by zero near
            the boundary of the disk where ``||x|| -> 1``.
    """

    def __init__(self, ridge: float = 1e-8):
        super().__init__()
        self._ridge = ridge

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        diff_sq = jnp.sum((x - y) ** 2)
        x_sq = jnp.sum(x ** 2)
        y_sq = jnp.sum(y ** 2)
        denom = (1.0 - x_sq) * (1.0 - y_sq) + self._ridge
        arg = 1.0 + 2.0 * diff_sq / denom
        return jnp.arccosh(jnp.maximum(arg, 1.0 + 1e-7))

    @classmethod
    def _padder(cls, dim: int) -> jnp.ndarray:
        return jnp.zeros((1, dim))

    def tree_flatten(self):  # noqa: D102
        return (), (self._ridge,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):  # noqa: D102
        del children
        return cls(*aux_data)
