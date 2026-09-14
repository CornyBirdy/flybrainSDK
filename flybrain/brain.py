"""The :class:`FlyBrain` container.

FlyBrain owns a set of circuits and routes typed values between them and the
outside world. It deliberately knows nothing about neuroscience or about
flyvis: adding a circuit does not change this file.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .circuits.base import Circuit, get_circuit_class
from .ports import Port

__all__ = ["FlyBrain"]


class FlyBrain:
    """A fly brain assembled from one or more circuits.

    Args:
        circuits: Names of circuits to instantiate, e.g. ``["optic_lobe"]``.
            Circuit instances may also be passed directly.
        configs: Optional per-circuit config objects, keyed by circuit name.
        dt: Default integration step in seconds, used when :meth:`step` is
            called without an explicit ``dt``. 1/60 s matches a 60 fps host.

    Example:
        >>> from flybrain import FlyBrain, sensory, motor
        >>> fly = FlyBrain(circuits=["optic_lobe"])          # doctest: +SKIP
        >>> fly.set_input(sensory.VISUAL_FIELD, frame)       # doctest: +SKIP
        >>> fly.step()                                       # doctest: +SKIP
        >>> yaw = fly.get_output(motor.YAW)                  # doctest: +SKIP
    """

    def __init__(
        self,
        circuits: Sequence[Any] = ("optic_lobe",),
        configs: Optional[Dict[str, Any]] = None,
        dt: float = 1.0 / 60.0,
    ) -> None:
        if isinstance(circuits, str):
            circuits = (circuits,)
        configs = configs or {}

        self.dt = float(dt)
        self.circuits: Dict[str, Circuit] = {}
        for entry in circuits:
            if isinstance(entry, Circuit):
                self.circuits[entry.name] = entry
                continue
            cls = get_circuit_class(entry)
            config = configs.get(entry)
            self.circuits[entry] = cls(config) if config is not None else cls()

        self._input_routes: Dict[str, List[Circuit]] = {}
        self._output_routes: Dict[str, Circuit] = {}
        for circuit in self.circuits.values():
            for port in circuit.inputs:
                self._input_routes.setdefault(port.name, []).append(circuit)
            for port in circuit.outputs:
                self._output_routes[port.name] = circuit

        self._steps = 0

    # -- public API ----------------------------------------------------------

    def set_input(self, port: Port, value: Any) -> None:
        """Stage ``value`` on ``port`` for the next :meth:`step`.

        Raises:
            KeyError: If no loaded circuit consumes ``port``.
        """
        targets = self._input_routes.get(port.name)
        if not targets:
            raise KeyError(
                f"No loaded circuit accepts input port {port.name!r}. "
                f"Available inputs: {[p.name for p in self.input_ports]}"
            )
        for circuit in targets:
            circuit.set_input(port, value)

    def step(self, dt: Optional[float] = None) -> None:
        """Advance every circuit by ``dt`` seconds (default :attr:`dt`)."""
        step_dt = self.dt if dt is None else float(dt)
        for circuit in self.circuits.values():
            circuit.step(step_dt)
        self._steps += 1

    def get_output(self, port: Port) -> float:
        """Read the current value of ``port``.

        Raises:
            KeyError: If no loaded circuit produces ``port``.
        """
        circuit = self._output_routes.get(port.name)
        if circuit is None:
            raise KeyError(
                f"No loaded circuit produces output port {port.name!r}. "
                f"Available outputs: {[p.name for p in self.output_ports]}"
            )
        return circuit.get_output(port)

    def reset(self) -> None:
        """Return every circuit to its resting state."""
        for circuit in self.circuits.values():
            circuit.reset()
        self._steps = 0

    # -- introspection -------------------------------------------------------

    @property
    def input_ports(self) -> Tuple[Port, ...]:
        """Every sensory port exposed by the loaded circuits."""
        return tuple(
            dict.fromkeys(p for c in self.circuits.values() for p in c.inputs)
        )

    @property
    def output_ports(self) -> Tuple[Port, ...]:
        """Every motor port exposed by the loaded circuits."""
        return tuple(
            dict.fromkeys(p for c in self.circuits.values() for p in c.outputs)
        )

    @property
    def steps(self) -> int:
        """Number of times :meth:`step` has been called since the last reset."""
        return self._steps

    def input_shape(self, port: Port) -> Tuple[int, ...]:
        """Natural resolution for ``port``, e.g. ``(391, 391)`` for vision.

        Rendering at this size avoids an internal resize before the frame is
        sampled onto the photoreceptor lattice.
        """
        for circuit in self._input_routes.get(port.name, []):
            return circuit.input_shape(port)
        raise KeyError(port)

    def describe(self) -> str:
        """A human-readable summary of the loaded circuits and their ports."""
        lines = [f"FlyBrain(dt={self.dt:.4f}s, circuits={list(self.circuits)})"]
        for name, circuit in self.circuits.items():
            lines.append(f"  [{name}]")
            for port in circuit.inputs:
                lines.append(f"    in  {port.name:<16s} {port.units}")
            for port in circuit.outputs:
                lines.append(f"    out {port.name:<16s} {port.units}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<FlyBrain circuits={list(self.circuits)} steps={self._steps}>"
