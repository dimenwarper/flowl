"""Model graph: mutable during tracing, frozen before solve."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from flowl.core.constraints import Constraint
from flowl.core.geometry import SpaceGeometry


class CloudMeta(NamedTuple):
    """Metadata for a cloud in the graph."""

    name: str
    n_particles: int
    dim: int
    fixed: bool


@dataclass
class ModelGraph:
    """Mutable graph built during tracing inside a Space context."""

    clouds: dict[str, CloudMeta] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    geometry: SpaceGeometry = field(default_factory=lambda: SpaceGeometry(kind="pointcloud"))

    def add_cloud(self, meta: CloudMeta) -> None:
        if meta.name in self.clouds:
            raise ValueError(f"Cloud '{meta.name}' already exists in graph.")
        self.clouds[meta.name] = meta

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def freeze(self) -> FrozenGraph:
        return FrozenGraph(
            clouds=dict(self.clouds),
            constraints=tuple(self.constraints),
            geometry=self.geometry,
        )


@dataclass(frozen=True)
class FrozenGraph:
    """Immutable graph snapshot passed to the solver."""

    clouds: dict[str, CloudMeta]
    constraints: tuple[Constraint, ...]
    geometry: SpaceGeometry = field(default_factory=lambda: SpaceGeometry(kind="pointcloud"))
