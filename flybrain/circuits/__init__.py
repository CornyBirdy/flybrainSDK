"""Neural circuits exposed by the flybrain SDK.

Stage 1 ships the optic-lobe / optomotor steering circuit. Later stages add
further circuits behind the same :class:`~flybrain.circuits.base.Circuit`
interface.
"""

from .base import Circuit, available_circuits, get_circuit_class, register_circuit
from .optic_lobe import OpticLobeCircuit, OpticLobeConfig

__all__ = [
    "Circuit",
    "register_circuit",
    "get_circuit_class",
    "available_circuits",
    "OpticLobeCircuit",
    "OpticLobeConfig",
]
