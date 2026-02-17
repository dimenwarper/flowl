"""Space geometry configuration for optimal transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flowl._types import Array


@dataclass(frozen=True)
class SpaceGeometry:
    """Describes the metric structure of a Space.

    Parameters
    ----------
    kind : str
        One of ``"pointcloud"``, ``"graph"``, ``"grid"``.
    cost_fn : CostFn | None
        OTT-JAX cost function for pointcloud geometries.
    graph_laplacian : Array | None
        Laplacian matrix for graph geometries.
    grid_xs : tuple[Array, ...] | None
        Marginal coordinate vectors for grid geometries.
    """

    kind: str
    cost_fn: Any | None = None
    graph_laplacian: Array | None = None
    grid_xs: tuple[Array, ...] | None = None


_COST_SHORTCUTS: dict[str, str] = {
    "sq_euclidean": "SqEuclidean",
    "euclidean": "Euclidean",
    "cosine": "Cosine",
}


def resolve_geometry(geometry: Any = None, **kwargs: Any) -> SpaceGeometry:
    """Build a :class:`SpaceGeometry` from user-facing arguments.

    Parameters
    ----------
    geometry
        A string shortcut (``"euclidean"``, ``"cosine"``, ``"sq_euclidean"``,
        ``"hyperbolic"``, ``"graph"``, ``"grid"``), an OTT-JAX ``CostFn``
        instance, or *None* for the default squared-Euclidean cost.
    **kwargs
        Extra arguments forwarded to :class:`SpaceGeometry` (e.g.
        ``graph_laplacian`` or ``grid_xs``).
    """
    if geometry is None or geometry == "sq_euclidean":
        return SpaceGeometry(kind="pointcloud", cost_fn=None)

    if isinstance(geometry, str):
        if geometry == "hyperbolic":
            from flowl.costs.hyperbolic import HyperbolicCost

            return SpaceGeometry(kind="pointcloud", cost_fn=HyperbolicCost())

        if geometry in _COST_SHORTCUTS:
            from ott.geometry import costs as ott_costs

            cost_cls = getattr(ott_costs, _COST_SHORTCUTS[geometry])
            return SpaceGeometry(kind="pointcloud", cost_fn=cost_cls())

        if geometry == "graph":
            laplacian = kwargs.get("graph_laplacian")
            if laplacian is None:
                raise ValueError(
                    "geometry='graph' requires a graph_laplacian keyword argument."
                )
            return SpaceGeometry(kind="graph", graph_laplacian=laplacian)

        if geometry == "grid":
            grid_xs = kwargs.get("grid_xs")
            if grid_xs is None:
                raise ValueError(
                    "geometry='grid' requires a grid_xs keyword argument."
                )
            return SpaceGeometry(kind="grid", grid_xs=tuple(grid_xs))

        raise ValueError(
            f"Unknown geometry string: {geometry!r}. "
            f"Supported: {sorted(list(_COST_SHORTCUTS) + ['graph', 'grid', 'hyperbolic'])}."
        )

    # Assume it's an OTT-JAX CostFn instance
    return SpaceGeometry(kind="pointcloud", cost_fn=geometry)
