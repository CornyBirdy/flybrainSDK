"""Tests for the MaleCNS leaky integrate-and-fire circuit.

These lock in the result the whole circuit rests on - that activating the
sweet-sensing gustatory neurons drives the proboscis-extension motor neuron
MN9, that bitter co-activation suppresses it, and that a degree-preserving
shuffle of the connectome abolishes it - plus the port contract around it.

They need the connectome cache and skip cleanly without it:

    python -m flybrain.malecns download

The model is stochastic and the simulation is slow (see the module docstring
of ``flybrain/circuits/male_cns.py``), so durations here are the shortest
that still separate the conditions, and the thresholds are deliberately loose.
The point is the sign and the ordering of the effects, not their magnitude.
"""

import numpy as np
import pytest

from flybrain import FlyBrain, motor, sensory
from flybrain.circuits import MaleCNSCircuit, MaleCNSConfig
from flybrain.circuits.male_cns import INPUT_POPULATIONS, OUTPUT_NEURONS

from .conftest import mean_rate, requires_connectome

pytestmark = requires_connectome

#: Drive rate for the gustatory populations. Shiu et al. use 100-200 Hz.
DRIVE_HZ = 150.0

#: Simulated seconds per condition. Long enough to separate the conditions,
#: short enough that the file runs in about a minute.
DURATION = 0.5


# -- the sanity check ------------------------------------------------------


@pytest.fixture(scope="module")
def sugar_response(male_cns):
    """MN9_L rate with the sugar GRNs driven, and with nothing driven."""
    baseline = mean_rate(male_cns, motor.MN9_L, {}, DURATION)
    sugar = mean_rate(male_cns, motor.MN9_L, {sensory.SUGAR_GRN: DRIVE_HZ}, DURATION)
    return baseline, sugar


def test_sugar_drives_mn9(sugar_response):
    """Sweet GRNs must drive the proboscis-extension motor neuron.

    This is the load-bearing result. If it fails, nothing else the circuit
    produces is meaningful.
    """
    baseline, sugar = sugar_response
    assert baseline == 0.0, (
        f"the model has zero basal firing rate by construction, but the "
        f"undriven network put MN9_L at {baseline:.2f} Hz"
    )
    assert sugar > 1.0, (
        f"sugar activation should drive MN9_L well above silence, got "
        f"{sugar:.2f} Hz"
    )


def test_bitter_suppresses_the_sugar_response(male_cns, sugar_response):
    """Co-activating bitter GRNs must shut the sugar response down."""
    _, sugar = sugar_response
    both = mean_rate(
        male_cns,
        motor.MN9_L,
        {sensory.SUGAR_GRN: DRIVE_HZ, sensory.BITTER_GRN: DRIVE_HZ},
        DURATION,
    )
    assert both < sugar, (
        f"bitter co-activation should suppress MN9_L, but it went from "
        f"{sugar:.2f} Hz to {both:.2f} Hz"
    )


def test_bitter_alone_does_not_drive_mn9(male_cns):
    """Bitter taste should not produce proboscis extension."""
    bitter = mean_rate(male_cns, motor.MN9_L, {sensory.BITTER_GRN: DRIVE_HZ}, DURATION)
    assert bitter == 0.0, f"bitter alone put MN9_L at {bitter:.2f} Hz"


def test_silent_input_produces_no_motor_output(male_cns):
    """With every population silent, nothing downstream may fire.

    The model has no basal firing rate, so this is exact, not approximate.
    """
    male_cns.reset()
    for port in male_cns.inputs:
        male_cns.set_input(port, 0.0)
    male_cns.step(0.05)

    for port in male_cns.outputs:
        assert male_cns.get_output(port) == 0.0, f"{port.name} fired with no input"
    assert male_cns.spike_counts.sum() == 0, (
        f"{int(male_cns.spike_counts.sum())} spikes with no input; the model "
        "is supposed to be silent at rest"
    )


def test_shuffling_the_connectome_abolishes_the_result(male_cns):
    """A degree-preserving rewiring must destroy the sugar to MN9 pathway.

    A result that survives this shuffle would be a property of the simulator
    rather than of the wiring, and the whole circuit would be worthless. The
    shuffle keeps every neuron's in-degree, out-degree, synapse counts and
    transmitter signs, and destroys only the pairing of who talks to whom.

    The control is not perfectly matched - a randomly rewired graph also
    propagates less activity overall, see NOTES.md - so the assertion checks
    both that MN9 goes quiet AND that the shuffled network is still firing,
    which is what makes the first half mean anything.
    """
    from flybrain.malecns import LIFNetwork, shuffle_preserving_degree

    sugar = np.concatenate(
        [
            male_cns.connectome.select(type=t)
            for t in INPUT_POPULATIONS[sensory.SUGAR_GRN.name]
        ]
    )
    mn9 = male_cns.population(motor.MN9_L)

    shuffled = shuffle_preserving_degree(male_cns.connectome.weights, seed=7)
    net = LIFNetwork(shuffled, seed=0)
    net.set_poisson_drive(sugar, np.full(sugar.size, DRIVE_HZ))
    net.step(DURATION)
    rates = net.rates()

    assert rates[mn9].mean() == 0.0, (
        f"the sugar to MN9 pathway survived a degree-preserving shuffle "
        f"({rates[mn9].mean():.2f} Hz), so it is an artefact of the simulator"
    )
    assert rates.sum() > 100.0, (
        "the shuffled network is silent, which makes the control vacuous - "
        "MN9 would be quiet whatever the wiring"
    )


# -- port contract ---------------------------------------------------------


def test_both_circuits_expose_disjoint_ports(male_cns):
    """male_cns must not collide with optic_lobe's ports.

    FlyBrain routes by port name, so a collision would silently shadow one
    circuit's output with the other's.
    """
    optic_lobe_ports = {
        motor.YAW.name,
        motor.HS_LEFT.name,
        motor.HS_RIGHT.name,
        motor.FLOW_ASYMMETRY.name,
        sensory.VISUAL_FIELD.name,
    }
    mine = {p.name for p in male_cns.inputs} | {p.name for p in male_cns.outputs}
    assert not (mine & optic_lobe_ports)


def test_ports_route_through_flybrain(male_cns):
    """The public three-call API must work for this circuit like any other."""
    fly = FlyBrain(circuits=[male_cns], dt=1 / 60)
    assert sensory.SUGAR_GRN in fly.input_ports
    assert motor.MN9_L in fly.output_ports

    fly.reset()
    fly.set_input(sensory.SUGAR_GRN, DRIVE_HZ)
    fly.step()
    fly.step(1 / 30)
    assert isinstance(fly.get_output(motor.MN9_L), float)

    with pytest.raises(KeyError):
        fly.get_output(motor.YAW)  # optic_lobe is not loaded


def test_every_port_maps_to_real_neurons(male_cns):
    """No port may be wired to an empty population.

    A port with no neurons behind it would read a constant zero and look like
    a working read-out.
    """
    for port in male_cns.outputs:
        assert male_cns.population(port).size > 0, port.name
    for port_name in INPUT_POPULATIONS:
        port = getattr(sensory, port_name)
        assert male_cns.population(port).size > 0, port_name


def test_mn9_bodies_are_the_documented_ones(male_cns):
    """Guard the identity of the read-out neuron against a cache rebuild."""
    bodies = male_cns.connectome.neurons.bodyId.to_numpy()
    assert bodies[male_cns.population(motor.MN9_L)].tolist() == [10331]
    assert bodies[male_cns.population(motor.MN9_R)].tolist() == [16949]


def test_sugar_population_is_lb3b_and_lb3c(male_cns):
    """The sugar port must drive exactly the Gr64f-matching subtypes.

    Named explicitly because MaleCNS carries no modality annotation, so this
    assignment is literature-derived and worth pinning down.
    """
    types = male_cns.connectome.neurons.type.to_numpy()
    driven = set(types[male_cns.population(sensory.SUGAR_GRN)])
    assert driven == {"LB3b", "LB3c"}


def test_rejects_bad_input(male_cns):
    """Negative rates, wrong shapes and unknown ports must all raise."""
    with pytest.raises(ValueError):
        male_cns.set_input(sensory.SUGAR_GRN, -1.0)
    with pytest.raises(ValueError):
        male_cns.set_input(sensory.SUGAR_GRN, np.zeros(3))
    with pytest.raises(KeyError):
        male_cns.set_input(sensory.VISUAL_FIELD, 0.0)
    with pytest.raises(KeyError):
        male_cns.get_output(motor.YAW)
    with pytest.raises(ValueError):
        male_cns.step(0.0)


def test_visual_coupling_is_off_by_default(male_cns):
    """T4T5_DRIVE must be opt-in, and must say why when refused."""
    assert sensory.T4T5_DRIVE not in male_cns.inputs
    with pytest.raises(KeyError, match="hypothesis"):
        male_cns.set_input(sensory.T4T5_DRIVE, 10.0)


def test_reset_returns_the_circuit_to_rest(male_cns):
    """After reset, no spikes and no staged drive survive."""
    male_cns.reset()
    male_cns.set_input(sensory.SUGAR_GRN, DRIVE_HZ)
    male_cns.step(0.05)
    assert male_cns.spike_counts.sum() > 0

    male_cns.reset()
    assert male_cns.spike_counts.sum() == 0
    assert all(male_cns.get_output(p) == 0.0 for p in male_cns.outputs)
    male_cns.step(0.05)
    assert male_cns.spike_counts.sum() == 0, "drive survived a reset"


def test_config_is_a_dataclass_of_constants():
    """Every tunable lives in the config, not buried in a method."""
    config = MaleCNSConfig()
    assert config.min_synapses == 1
    assert config.readout_tau > 0
    assert config.enable_visual_coupling is False
    assert MaleCNSCircuit.name == "male_cns"


def test_output_ports_cover_every_declared_readout(male_cns):
    """OUTPUT_NEURONS and the outputs property must not drift apart."""
    assert {p.name for p in male_cns.outputs} == set(OUTPUT_NEURONS)
