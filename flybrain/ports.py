"""Typed port descriptors.

A :class:`Port` is the contract between a circuit and whatever is embodying it.
It carries a name, a direction, the expected value shape/dtype, the physical
units and a human-readable description.

Ports are frozen (hashable) so they can be used directly as dictionary keys::

    fly.set_input(sensory.VISUAL_FIELD, frame)
    yaw = fly.get_output(motor.YAW)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

__all__ = ["Port", "Direction"]

Direction = Literal["in", "out"]


@dataclass(frozen=True)
class Port:
    """A typed sensory input or motor output of a circuit.

    Attributes:
        name: Unique identifier, e.g. ``"VISUAL_FIELD"``.
        direction: ``"in"`` for sensory ports, ``"out"`` for motor ports.
        dtype: Expected element dtype, e.g. ``"float32"``.
        shape: Expected shape. ``None`` for scalars. Entries may be strings
            (``"H"``, ``"W"``) to denote dimensions fixed at runtime rather
            than statically.
        units: Physical/non-dimensional units of the value. See the package
            README for the full unit and sign conventions.
        description: What the value means.
    """

    name: str
    direction: Direction
    dtype: str = "float32"
    shape: Optional[Tuple[object, ...]] = None
    units: str = "dimensionless"
    description: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<Port {self.name} ({self.direction}) [{self.units}]>"
