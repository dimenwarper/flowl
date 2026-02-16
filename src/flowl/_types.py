"""Type aliases for flowl."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

Array = jnp.ndarray
"""JAX array type alias."""

PyTree = Any
"""Generic JAX pytree."""
