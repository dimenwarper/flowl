"""Space context manager for model tracing."""

from __future__ import annotations

from typing import Any

from flowl.core.geometry import SpaceGeometry, resolve_geometry
from flowl.core.graph import ModelGraph
from flowl.tracing.context import push_graph, pop_graph


class Space:
    """Context manager that opens a model-tracing scope.

    Usage::

        with Space() as space:
            mu = Cloud("mu")
            ...

        graph = space.graph  # ModelGraph built during the block

    Parameters
    ----------
    geometry
        A string shortcut (``"euclidean"``, ``"cosine"``, ``"sq_euclidean"``,
        ``"graph"``, ``"grid"``), an OTT-JAX ``CostFn`` instance, or *None*
        for the default squared-Euclidean cost.
    **kwargs
        Forwarded to :func:`resolve_geometry` (e.g. ``graph_laplacian``).
    """

    def __init__(self, geometry: Any = None, **kwargs: Any) -> None:
        geo = resolve_geometry(geometry, **kwargs)
        self._graph = ModelGraph(geometry=geo)

    @property
    def graph(self) -> ModelGraph:
        return self._graph

    def __enter__(self) -> Space:
        push_graph(self._graph)
        return self

    def __exit__(self, *exc: object) -> None:
        pop_graph()
