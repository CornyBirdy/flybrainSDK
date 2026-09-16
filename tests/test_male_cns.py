"""Tests for the MaleCNS leaky integrate-and-fire circuit.

These lock in the result the whole circuit rests on - that activating the
sweet-sensing gustatory neurons drives the proboscis-extension motor neuron
MN9, that bitter co-activation suppresses it, and that a degree-preserving
shuffle of the connectome abolishes it - plus the port contract around it.

They need the connectome cache and skip cleanly without it:

    python -m flybrain.malecns download

The model is stochastic and the simulation is slow (see the module docstring
of ``flybrain/circuits/male_cns.py``), so durations here are the shortest
that still separate the conditions. The thresholds are *not* loose: an audit
(``VERIFICATION.md`` Part E) permuted the postsynaptic column of the weight
table, destroying 99.5% of the wiring, and every test in this file still
passed, because the load-bearing assertion was ``sugar > 1.0`` Hz over a 0.5 s
window in which a single spike reads as 2.0 Hz. Every quantitative bound here
now carries a comment giving the measured value it guards and the measured
noise floor it has to clear.
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
#: short enough that the file runs in about half a minute.
#:
#: Kept at 0.5 s deliberately. One spike in this window reads as 2.0 Hz, so
#: single-spike quantisation is coarse -- but it only *dominates* when the
#: threshold sits within a spike or two of the noise floor, which was the
#: original defect and is fixed by SUGAR_FLOOR_HZ below rather than by a
#: longer window. Measured on this graph (MN9_L, 150 Hz sugar drive):
#:
#:   0.5 s, seeds 0-14: 22-52 Hz (min 22.0, mean 33.5)
#:   1.0 s, seeds 0-4:  25-34 Hz
#:
#: Doubling DURATION tightens the spread but takes the sugar-driven condition
#: from 2.6 s to 9.6 s of wall time, and there are four such conditions in
#: this file: ~19 s -> ~45 s, on a full suite that already runs ~17 minutes.
#: The 0.5 s spread is already resolved 11 spikes deep at its minimum, so the
#: extra wall time buys margin we do not need. If you lengthen it, the floor
#: below can be raised in step.
DURATION = 0.5

#: Floor that MN9_L must clear under sugar drive, in Hz.
#:
#: This is the one number in this file the whole circuit rests on, so it is
#: sized against both sides of the measurement rather than against zero:
#:
#:   real connectome, 150 Hz sugar, 0.5 s, seeds 0-14:      22-52 Hz
#:   postsynaptic column permuted (99.5% of the wiring gone):
#:     the audit's permutation, seeds 0-4:                     0-4 Hz
#:     a second, independent permutation, seeds 0-9:            0 Hz
#:
#: So 10 Hz sits 2.5x above the highest rate any destroyed connectome has
#: produced over 15 measured seeds, and 2.2x below the lowest rate the real
#: one has. The previous bound was 1.0 Hz, which a single accidental spike
#: (2.0 Hz at this DURATION) cleared -- so a fully randomised connectome
#: passed every test in this file. See VERIFICATION.md Parts E and 7.
SUGAR_FLOOR_HZ = 10.0

#: Fraction of the real network's total activity the degree-shuffled null must
#: still carry for its silence at MN9 to mean anything.
#:
#: Measured: shuffling drops whole-network activity from 75,238 Hz to
#: 8,530-9,404 Hz across seeds 7-9, i.e. to 0.11-0.13 of the real graph. The
#: previous bound was an absolute ``rates.sum() > 100.0``, which is 0.0013 of
#: the real graph -- about 90x below the value it guards, and satisfied by a
#: connectome that had been destroyed rather than shuffled.
SHUFFLED_ACTIVITY_FRACTION = 0.02


# -- the sanity check ------------------------------------------------------


@pytest.fixture(scope="module")
def sugar_response(male_cns):
    """MN9_L rate driven and undriven, plus the driven network's total activity.

    The total is read straight after the sugar run, while the circuit still
    holds that run's spike counts, so it costs no extra simulation. The
    shuffle control needs it as the reference its own activity is measured
    against - see SHUFFLED_ACTIVITY_FRACTION.
    """
    baseline = mean_rate(male_cns, motor.MN9_L, {}, DURATION)
    sugar = mean_rate(male_cns, motor.MN9_L, {sensory.SUGAR_GRN: DRIVE_HZ}, DURATION)
    net_activity = float(male_cns.mean_rates().sum())
    return baseline, sugar, net_activity


def test_sugar_drives_mn9(sugar_response):
    """Sweet GRNs must drive the proboscis-extension motor neuron.

    This is the load-bearing result. If it fails, nothing else the circuit
    produces is meaningful - which is why the floor is SUGAR_FLOOR_HZ and not
    a token value above zero. See that constant for the measurements either
    side of it, and VERIFICATION.md Part E for what a token value let through.
    """
    baseline, sugar, _ = sugar_response
    assert baseline == 0.0, (
        f"the model has zero basal firing rate by construction, but the "
        f"undriven network put MN9_L at {baseline:.2f} Hz"
    )
    assert sugar > SUGAR_FLOOR_HZ, (
        f"sugar activation should drive MN9_L to 22-52 Hz on this graph, got "
        f"{sugar:.2f} Hz, which is below the {SUGAR_FLOOR_HZ:.1f} Hz floor. A "
        f"connectome with its wiring destroyed produces 0-4 Hz here, so a rate "
        f"this low means the pathway is gone, not merely weak"
    )


def test_bitter_suppresses_the_sugar_response(male_cns, sugar_response):
    """Co-activating bitter GRNs must shut the sugar response down.

    Measured, MN9_L under sugar + bitter: exactly 0.00 Hz in every seed tried
    (0-4 here, and four further seed families in VERIFICATION.md Part D),
    against 22-52 Hz under sugar alone. The bound is therefore a tenth of the
    sugar response rather than the bare ``both < sugar`` it replaces: the
    claim is that bitter abolishes the response, and ``both < sugar`` would
    also be satisfied by a 5% reduction.
    """
    _, sugar, _ = sugar_response
    both = mean_rate(
        male_cns,
        motor.MN9_L,
        {sensory.SUGAR_GRN: DRIVE_HZ, sensory.BITTER_GRN: DRIVE_HZ},
        DURATION,
    )
    assert both <= 0.1 * sugar, (
        f"bitter co-activation should abolish the MN9_L response, not merely "
        f"reduce it: it went from {sugar:.2f} Hz to {both:.2f} Hz, which is "
        f"{100 * both / sugar:.0f}% of the sugar response rather than the 0% "
        f"measured on this graph"
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


def test_shuffling_the_connectome_abolishes_the_result(male_cns, sugar_response):
    """A random rewiring must destroy the sugar to MN9 pathway.

    A result that survives this shuffle would be a property of the simulator
    rather than of the wiring, and the whole circuit would be worthless. The
    shuffle keeps every edge's presynaptic neuron, synapse count and
    transmitter sign and permutes the postsynaptic slots, so it destroys
    exactly the pairing of who talks to whom. It does not preserve either
    degree sequence exactly - see ``shuffle_preserving_degree`` - and it also
    drops total network activity about 8x, so it is not a clean
    single-variable null and NOTES.md says so.

    Every assertion here is a CONTRAST against the real graph, not an
    absolute bound, because absolute bounds made this control vacuous: on a
    connectome whose wiring had already been destroyed, MN9 was still 0 Hz
    under shuffling and the network still summed to 8,748 Hz, so both halves
    passed while testing nothing (VERIFICATION.md Part E, mutation 3). A
    shuffle can only be shown to abolish a result if the unshuffled graph is
    first shown to produce one, so this test asserts that too.
    """
    from flybrain.malecns import LIFNetwork, shuffle_preserving_degree

    _, real_mn9, real_activity = sugar_response

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

    # First half of the contrast: there has to be a result to abolish. Without
    # this, "MN9 is quiet after shuffling" is equally true of a connectome that
    # was already noise, which is exactly how this control passed an audit
    # mutation that destroyed 99.5% of the wiring.
    assert real_mn9 > SUGAR_FLOOR_HZ, (
        f"the unshuffled graph only put MN9_L at {real_mn9:.2f} Hz, below the "
        f"{SUGAR_FLOOR_HZ:.1f} Hz floor, so there is no result for the shuffle "
        f"to abolish and this control cannot mean anything. Fix "
        f"test_sugar_drives_mn9 first"
    )
    # Second half: and the shuffle has to abolish it.
    assert rates[mn9].mean() == 0.0, (
        f"the sugar to MN9 pathway survived a random rewiring "
        f"({rates[mn9].mean():.2f} Hz against {real_mn9:.2f} Hz on the real "
        f"graph), so it is an artefact of the simulator, not of the wiring"
    )
    # Third: the shuffled network must still be carrying real traffic, or MN9's
    # silence is just a dead network. Measured as a fraction of the real
    # graph's own activity rather than an absolute floor, because an absolute
    # floor is satisfied by a destroyed connectome too.
    floor = SHUFFLED_ACTIVITY_FRACTION * real_activity
    assert rates.sum() > floor, (
        f"the shuffled network carries {rates.sum():,.0f} Hz against the real "
        f"graph's {real_activity:,.0f} Hz, i.e. "
        f"{rates.sum() / real_activity:.4f} of it, below the "
        f"{SHUFFLED_ACTIVITY_FRACTION:.2f} this control needs (measured: "
        f"0.11-0.13). The null has gone silent, so MN9 would be quiet whatever "
        f"the wiring and this control is vacuous"
    )


#: Draw seeds for the random-gustatory null. Fixed so the test is
#: deterministic, and chosen by index (1000, 1001, ...) rather than by result.
RANDOM_GUSTATORY_SEEDS = (1000, 1001, 1002, 1003, 1004)

#: Factor by which sugar must beat the median random-gustatory draw.
#:
#: Measured at this protocol (MN9_L, 150 Hz, 0.5 s, LIF seed 0): sugar 22.00 Hz;
#: the five draws above give 8, 4, 0, 4, 0 Hz, median 4.00, so the real margin
#: is 5.5x. 3x is the bound, leaving room for the median to move by one spike.
SUGAR_SPECIFICITY_FACTOR = 3.0


def test_sugar_beats_a_random_gustatory_population(male_cns, sugar_response):
    """Sugar must drive MN9 harder than an arbitrary gustatory set does.

    The shuffle control rules out "the simulator lights up MN9 for any input".
    It does not rule out "the real wiring lights up MN9 for many inputs", and
    that one is partly true, so the margin is asserted here rather than
    remembered. This is the control an audit added and NOTES.md had never run
    (VERIFICATION.md Finding 4).

    Read the bound for what it is: a QUANTITATIVE margin over the typical
    gustatory set, not a categorical claim of sugar-specificity. Two things
    make that distinction necessary, both measured on this graph:

    * The null is heavy-tailed. Over 24 draws the median is 3.00 Hz, but
      5 of 24 exceeded sugar's own 22.00 Hz and one reached 156.00 Hz with the
      network in near-global runaway. This test therefore pins the median of
      five fixed draws; it would be false as a statement about every draw.
    * MaleCNS annotates 1,428 gustatory neurons and most of them are limbs:
      768 leg bristle, 385 wing bristle, only 163 labellar. So a random
      gustatory draw is mostly not a taste input to the proboscis at all.

    Bitter and water do give exactly zero, so specificity exists in some
    directions - see the bitter tests. It is just not the whole story.
    """
    _, sugar, _ = sugar_response

    neurons = male_cns.connectome.neurons
    gustatory = np.flatnonzero((neurons["cell_class"] == "gustatory").to_numpy())
    sugar_population = set(male_cns.population(sensory.SUGAR_GRN).tolist())
    pool = np.array([i for i in gustatory if i not in sugar_population])
    assert pool.size > 1000, f"gustatory pool is only {pool.size} neurons"

    n = len(sugar_population)
    mn9 = male_cns.population(motor.MN9_L)
    draws = []
    for draw_seed in RANDOM_GUSTATORY_SEEDS:
        index = np.sort(
            np.random.default_rng(draw_seed).choice(pool, size=n, replace=False)
        )
        male_cns.reset()
        male_cns.network.set_poisson_drive(index, np.full(index.size, DRIVE_HZ))
        male_cns.network.step(DURATION)
        draws.append(float(male_cns.network.rates()[mn9].mean()))

    median = float(np.median(draws))

    # Guard against the vacuous form: if sugar itself is not driving MN9, a
    # margin over the null says nothing.
    assert sugar > SUGAR_FLOOR_HZ, (
        f"sugar only put MN9_L at {sugar:.2f} Hz, so there is no margin to "
        f"measure. Fix test_sugar_drives_mn9 first"
    )
    assert sugar > SUGAR_SPECIFICITY_FACTOR * median, (
        f"sugar put MN9_L at {sugar:.2f} Hz against a median of {median:.2f} Hz "
        f"for {n} random gustatory neurons (draws: "
        f"{', '.join(f'{d:.2f}' for d in draws)}), a margin of "
        f"{sugar / median if median else float('inf'):.1f}x. The sugar to MN9 "
        f"result is only as specific as this margin, and it has fallen below "
        f"{SUGAR_SPECIFICITY_FACTOR:.0f}x (measured: 5.5x)"
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


#: Network-wide spikes in a 0.05 s sugar-driven step. Measured: 1,184. The
#: bound exists so that "zero spikes after a reset" means something - a step
#: that produced one spike would make it vacuous the same way an absolute
#: floor made the shuffle control vacuous.
DRIVEN_SPIKES_FLOOR = 100


def test_reset_returns_the_circuit_to_rest(male_cns):
    """After reset, no spikes and no staged drive survive."""
    male_cns.reset()
    male_cns.set_input(sensory.SUGAR_GRN, DRIVE_HZ)
    male_cns.step(0.05)
    driven = int(male_cns.spike_counts.sum())
    assert driven > DRIVEN_SPIKES_FLOOR, (
        f"a 0.05 s sugar-driven step produced only {driven} spikes network-wide "
        f"(measured: 1,184), so the zero-after-reset assertions below would "
        f"pass whether or not reset works"
    )

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
