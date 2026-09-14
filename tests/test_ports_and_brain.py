"""Container and port-plumbing tests. These do not need flyvis."""

import numpy as np
import pytest

from flybrain import FlyBrain, motor, sensory
from flybrain.circuits.base import Circuit, register_circuit
from flybrain.ports import Port

DUMMY_IN = Port("DUMMY_IN", "in", units="test")
DUMMY_OUT = Port("DUMMY_OUT", "out", units="test")


@register_circuit("_dummy")
class _DummyCircuit(Circuit):
    """Minimal circuit used to exercise FlyBrain without loading a model."""

    def __init__(self, config=None):
        self.config = config
        self.value = 0.0
        self.staged = 0.0
        self.dts = []

    @property
    def inputs(self):
        return (DUMMY_IN,)

    @property
    def outputs(self):
        return (DUMMY_OUT,)

    def reset(self):
        self.value = 0.0
        self.staged = 0.0
        self.dts = []

    def set_input(self, port, value):
        self.staged = float(value)

    def step(self, dt):
        self.dts.append(dt)
        self.value += self.staged * dt

    def get_output(self, port):
        return self.value

    def input_shape(self, port):
        return (4, 4)


def test_port_is_hashable_and_frozen():
    assert Port("A", "in") == Port("A", "in")
    assert {DUMMY_IN: 1}[DUMMY_IN] == 1
    with pytest.raises(Exception):
        DUMMY_IN.name = "other"


def test_brain_routes_values_and_integrates():
    fly = FlyBrain(circuits=["_dummy"], dt=0.5)
    fly.set_input(DUMMY_IN, 2.0)
    fly.step()
    assert fly.get_output(DUMMY_OUT) == pytest.approx(1.0)
    fly.step(0.25)
    assert fly.get_output(DUMMY_OUT) == pytest.approx(1.5)
    assert fly.steps == 2


def test_reset_clears_state():
    fly = FlyBrain(circuits=["_dummy"])
    fly.set_input(DUMMY_IN, 1.0)
    fly.step()
    fly.reset()
    assert fly.get_output(DUMMY_OUT) == 0.0
    assert fly.steps == 0


def test_unknown_ports_raise_keyerror():
    fly = FlyBrain(circuits=["_dummy"])
    with pytest.raises(KeyError):
        fly.set_input(sensory.VISUAL_FIELD, np.zeros((4, 4)))
    with pytest.raises(KeyError):
        fly.get_output(motor.YAW)


def test_unknown_circuit_name_raises():
    with pytest.raises(KeyError):
        FlyBrain(circuits=["does_not_exist"])


def test_introspection_lists_ports():
    fly = FlyBrain(circuits=["_dummy"])
    assert fly.input_ports == (DUMMY_IN,)
    assert fly.output_ports == (DUMMY_OUT,)
    assert fly.input_shape(DUMMY_IN) == (4, 4)
    assert "DUMMY_IN" in fly.describe()


def test_motor_ports_declare_sign_convention():
    # The demo and any downstream consumer depend on these being documented.
    assert "steer RIGHT" in motor.YAW.description
    assert "POSITIVE" in motor.YAW.description
    assert motor.YAW.direction == "out"
    assert sensory.VISUAL_FIELD.direction == "in"
