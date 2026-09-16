"""Tests for the opt-in flyvis to MaleCNS coupling.

These test the MAPPING, not the biology. The coupling is a hypothesis -
flyvis and MaleCNS are different animals in different coordinate frames, and
nothing establishes a correspondence between their cells - so what is
asserted here is that the bridge does what its own docstring says it does,
is impossible to invoke by accident, and is self-consistent under a mirror.

Nothing here licenses the conclusion that a real fly's HS cells behave this
way. Read the module docstring of ``flybrain/circuits/coupling.py`` first.

Needs both models and skips cleanly without either.
"""

import numpy as np
import pytest

from flybrain import motor, sensory
from flybrain.circuits import MaleCNSConfig
from flybrain.circuits.coupling import CouplingConfig, OpticLobeToMaleCNS

from .conftest import panorama, requires_connectome, requires_weights

pytestmark = [requires_weights, requires_connectome]

PAN_W = 780
SHIFT = 5
DT = 1 / 60
FRAMES = 25

#: Fraction of the moving-scene T4/T5 drive that a STATIC scene must still
#: stage, for ``test_static_scene_still_produces_drive`` to be guarding the
#: flaw it documents rather than merely its complete removal.
#:
#: Measured ratio: **1.0067** -- a still image stages very slightly MORE T4/T5
#: drive than a drifting one. It is identical across MaleCNS seeds (67.86 vs
#: 67.41 Hz at seeds 0, 1 and 2) because the staged drive comes from flyvis,
#: which is deterministic here; only the downstream spiking is stochastic. So
#: this bound is not guarding a noisy quantity and can sit close to the value.
#:
#: The previous bound was 0.5, which is half the measured ratio: a change that
#: cut the pedestal by 40% -- a large improvement to the mapping -- would have
#: passed silently, so the test caught a complete fix and not an improvement.
#: See ``VERIFICATION.md`` Finding 10.
PEDESTAL_FRACTION = 0.9


@pytest.fixture(scope="module")
def coupled(fly, male_cns):
    """An optic lobe, a coupling-enabled MaleCNS circuit, and a bridge."""
    from flybrain.circuits import MaleCNSCircuit

    eye = fly.circuits["optic_lobe"]
    cns = MaleCNSCircuit(MaleCNSConfig(enable_visual_coupling=True, seed=0))
    return eye, cns, OpticLobeToMaleCNS(eye, cns, CouplingConfig())


def drift(eye, cns, bridge, shift_per_frame, frame_size):
    """Run a drifting panorama through the coupled pair.

    Returns:
        ``(HSE_L - HSE_R, total staged drive in Hz)``.
    """
    img = panorama(frame_size[0], PAN_W)
    cols = np.arange(frame_size[1])
    eye.reset()
    cns.reset()
    for i in range(FRAMES):
        eye.set_input(sensory.VISUAL_FIELD, img[:, (cols - shift_per_frame * i) % PAN_W])
        eye.step(DT)
        bridge.transfer()
        cns.step(DT)
    asymmetry = cns.get_output(motor.HSE_L) - cns.get_output(motor.HSE_R)
    return asymmetry, sum(bridge.last_rates().values())


def test_coupling_refuses_a_circuit_that_did_not_opt_in(fly, male_cns):
    """The default MaleCNS circuit must not be couplable."""
    eye = fly.circuits["optic_lobe"]
    with pytest.raises(ValueError, match="hypothesis"):
        OpticLobeToMaleCNS(eye, male_cns)


def test_coupling_rejects_the_wrong_circuit_types(fly, coupled):
    """Arguments are checked, so a swapped pair fails loudly."""
    eye, cns, _ = coupled
    with pytest.raises(TypeError):
        OpticLobeToMaleCNS(cns, cns)
    with pytest.raises(TypeError):
        OpticLobeToMaleCNS(eye, eye)


def test_flybrain_does_not_couple_the_circuits_by_default(male_cns):
    """Loading both circuits together must leave them independent.

    The two circuits coexist; that is a requirement. They must not start
    talking to each other because they happen to be in the same FlyBrain.
    """
    from flybrain import FlyBrain

    both = FlyBrain(circuits=["optic_lobe", male_cns])
    assert sensory.T4T5_DRIVE not in both.input_ports
    assert sensory.VISUAL_FIELD in both.input_ports
    assert motor.YAW in both.output_ports
    assert motor.MN9_L in both.output_ports


def test_transfer_stages_drive_on_the_male_cns_t4t5_cells(coupled, frame_size):
    """After a step and a transfer, MaleCNS T4/T5 cells must actually fire."""
    eye, cns, bridge = coupled
    eye.reset()
    cns.reset()
    eye.set_input(sensory.VISUAL_FIELD, panorama(frame_size[0], frame_size[1]))
    eye.step(DT)
    rates = bridge.transfer()

    assert set(rates) == set(cns.t4t5_populations)
    assert max(rates.values()) > 0.0, "the bridge staged no drive at all"
    assert all(0.0 <= r <= CouplingConfig().max_hz for r in rates.values())

    cns.step(0.05)
    driven = cns.connectome.select(type=["T4a", "T4b", "T5a", "T5b"])
    assert cns.spike_counts[driven].sum() > 0


def test_transfer_needs_a_stepped_optic_lobe(coupled):
    """Transferring before the optic lobe has any activity must raise."""
    eye, _, bridge = coupled
    eye.reset()
    with pytest.raises(RuntimeError):
        bridge.transfer()


def test_mirroring_the_motion_reverses_the_hs_asymmetry(coupled, frame_size):
    """Opposite drift directions must give opposite HS asymmetries.

    This is the substantive property of the mapping: under it, the real
    MaleCNS HS cells inherit a direction-dependent left/right imbalance whose
    sign tracks the optic lobe's own. It replicates across MaleCNS seeds
    (+30 Hz vs -27 Hz, see NOTES.md).

    It is a statement about the mapping, not about flies. In particular the
    total drive is nearly the same for a static scene (see
    ``test_static_scene_still_produces_drive``), so this asymmetry rides on a
    large motion-independent pedestal.
    """
    eye, cns, bridge = coupled
    rightward, _ = drift(eye, cns, bridge, +SHIFT, frame_size)
    leftward, _ = drift(eye, cns, bridge, -SHIFT, frame_size)

    assert rightward > 5.0, f"expected HSE_L > HSE_R for rightward drift, got {rightward:+.1f} Hz"
    assert leftward < -5.0, f"expected HSE_R > HSE_L for leftward drift, got {leftward:+.1f} Hz"


def test_static_scene_still_produces_drive(coupled, frame_size):
    """Document the mapping's main flaw rather than letting it hide.

    The drive is derived from flyvis activity relative to a GREY screen, so a
    static textured scene is already a large deviation and produces almost as
    much T4/T5 drive as a moving one -- measured, 100.7% as much. The bridge is
    therefore not a motion signal; the motion information is only in its
    left/right imbalance, which is a ±17-29 Hz asymmetry riding on a
    motion-independent pedestal, i.e. under 10% of the signal. This test exists
    so that the flaw is visible in the suite and so that anyone who fixes it
    finds out here.

    See PEDESTAL_FRACTION for why the bound is 0.9 and not the 0.5 it was.
    """
    eye, cns, bridge = coupled
    _, moving_drive = drift(eye, cns, bridge, +SHIFT, frame_size)
    static_asymmetry, static_drive = drift(eye, cns, bridge, 0, frame_size)

    assert static_drive > PEDESTAL_FRACTION * moving_drive, (
        f"the static-scene pedestal has shrunk: static drive is "
        f"{static_drive / moving_drive:.3f} of moving drive, against a measured "
        f"1.007 and a bound of {PEDESTAL_FRACTION}. If that was deliberate, "
        f"update this test and the coupling docstring - the flaw it documents "
        f"has been partly fixed"
    )
    assert abs(static_asymmetry) < 15.0, (
        f"a static scene should be roughly left/right symmetric, got "
        f"{static_asymmetry:+.1f} Hz"
    )
