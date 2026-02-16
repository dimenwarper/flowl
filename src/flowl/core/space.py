"""Space context manager for model tracing."""

from __future__ import annotations

from flowl.core.graph import ModelGraph
from flowl.tracing.context import push_graph, pop_graph


class Space:
    """Context manager that opens a model-tracing scope.

    Usage::

        with Space() as space:
            mu = Cloud("mu")
            ...

        graph = space.graph  # ModelGraph built during the block
    """

    def __init__(self) -> None:
        self._graph = ModelGraph()

    @property
    def graph(self) -> ModelGraph:
        return self._graph

    def __enter__(self) -> Space:
        push_graph(self._graph)
        return self

    def __exit__(self, *exc: object) -> None:
        pop_graph()
