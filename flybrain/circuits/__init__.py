"""Neural circuits exposed by the flybrain SDK.

Two circuits ship today, behind the same
:class:`~flybrain.circuits.base.Circuit` interface and usable together:

``optic_lobe``
    The pretrained flyvis optic lobe with an HS-cell steering read-out.
``male_cns``
    The MaleCNS v1.0 connectome as a leaky integrate-and-fire model.

Importing this package registers both but constructs neither, so neither
flyvis nor the MaleCNS loader is imported until a circuit is instantiated.
"""

from .base import Circuit, available_circuits, get_circuit_class, register_circuit
from .male_cns import MaleCNSCircuit, MaleCNSConfig
from .optic_lobe import OpticLobeCircuit, OpticLobeConfig

__all__ = [
    "Circuit",
    "register_circuit",
    "get_circuit_class",
    "available_circuits",
    "OpticLobeCircuit",
    "OpticLobeConfig",
    "MaleCNSCircuit",
    "MaleCNSConfig",
]
