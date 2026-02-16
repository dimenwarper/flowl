"""Thread-local graph stack for model tracing."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowl.core.graph import ModelGraph

_local = threading.local()


def _get_stack() -> list[ModelGraph]:
    if not hasattr(_local, "graph_stack"):
        _local.graph_stack = []
    return _local.graph_stack


def push_graph(graph: ModelGraph) -> None:
    """Push a graph onto the thread-local stack."""
    _get_stack().append(graph)


def pop_graph() -> ModelGraph:
    """Pop and return the top graph from the stack."""
    stack = _get_stack()
    if not stack:
        raise RuntimeError("No active graph context.")
    return stack.pop()


def get_graph() -> ModelGraph:
    """Return the current active graph, or raise if none."""
    stack = _get_stack()
    if not stack:
        raise RuntimeError(
            "No active Space context. Wrap your model code in `with Space():`."
        )
    return stack[-1]
