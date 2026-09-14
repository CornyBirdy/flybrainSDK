"""Circuit interface and registry.

A circuit is a self-contained piece of fly neuroscience with typed ports. The
:class:`FlyBrain` container knows nothing about what is inside one; it only
routes values between ports. Adding a new circuit (looming/escape, central
complex, ...) means implementing this interface and registering it.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from ..ports import Port

__all__ = ["Circuit", "register_circuit", "get_circuit_class", "available_circuits"]


class Circuit(abc.ABC):
    """Base class for a neural circuit with typed sensory/motor ports."""

    #: Registry key, e.g. ``"optic_lobe"``.
    name: str = "circuit"

    @property
    @abc.abstractmethod
    def inputs(self) -> Tuple[Port, ...]:
        """Sensory ports this circuit consumes."""

    @property
    @abc.abstractmethod
    def outputs(self) -> Tuple[Port, ...]:
        """Motor ports this circuit produces."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Return the circuit to its resting state."""

    @abc.abstractmethod
    def set_input(self, port: Port, value: Any) -> None:
        """Stage a value on an input port for the next :meth:`step`."""

    @abc.abstractmethod
    def step(self, dt: float) -> None:
        """Advance the circuit dynamics by ``dt`` seconds."""

    @abc.abstractmethod
    def get_output(self, port: Port) -> float:
        """Read the current value of an output port."""

    def input_shape(self, port: Port) -> Tuple[int, ...]:
        """Return the natural input resolution for ``port``.

        Raises:
            KeyError: If the port is not an input of this circuit.
        """
        raise KeyError(port)


_REGISTRY: Dict[str, Callable[..., Circuit]] = {}


def register_circuit(name: str) -> Callable[[type], type]:
    """Class decorator registering a circuit under ``name``."""

    def decorator(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_circuit_class(name: str) -> Callable[..., Circuit]:
    """Look up a registered circuit class by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown circuit {name!r}. Available: {sorted(_REGISTRY)}"
        ) from None


def available_circuits() -> Sequence[str]:
    """Names of all registered circuits."""
    return tuple(sorted(_REGISTRY))
