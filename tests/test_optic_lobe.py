"""Tests for the pretrained optic-lobe circuit.

These lock in the properties the rest of the SDK and any downstream consumer
rely on: the sign convention of YAW, the bilateral symmetry of the read-out,
and input handling. They need the pretrained flyvis weights.
"""

import numpy as np
import pytest

from flybrain import FlyBrain, motor, sensory
from flybrain.circuits import OpticLobeConfig
from flybrain.circuits.optic_lobe import LEFTWARD_TYPES, RIGHTWARD_TYPES

from .conftest import panorama, requires_weights

pytestmark = requires_weights

PAN_W = 780
SHIFT = 5
CYCLE = PAN_W // SHIFT
DT = 1 / 60


def drift(fly, height, panorama_img, shift_per_frame, n_cycles=2):
    """Slide a wrapping panorama; + shifts scene content RIGHT on screen.

    The panorama is traversed a whole number of times, so any fixed bias from
    the texture is identical for both directions and cancels on subtraction.
    Returns the mean (HS_LEFT, HS_RIGHT, YAW) over the final traversal.
    """
    fly.reset()
    cols = np.arange(fly.input_shape(sensory.VISUAL_FIELD)[1])
    out = []
    for i in range(n_cycles * CYCLE):
        frame = panorama_img[:, (cols - shift_per_frame * i) % PAN_W]
        fly.set_input(sensory.VISUAL_FIELD, frame)
        fly.step(DT)
        out.append((
            fly.get_output(motor.HS_LEFT),
            fly.get_output(motor.HS_RIGHT),
            fly.get_output(motor.YAW),
        ))
    return np.array(out[CYCLE:]).mean(0)


@pytest.fixture(scope="module")
def rotation_response(fly, frame_size):
    """HS/YAW motion terms for a body rotation to the LEFT."""
    img = panorama(frame_size[0], PAN_W)
    plus = drift(fly, frame_size[0], img, +SHIFT)   # scene right == yaw LEFT
    minus = drift(fly, frame_size[0], img, -SHIFT)  # scene left  == yaw RIGHT
    return 0.5 * (plus - minus)


def test_yaw_opposes_body_rotation(rotation_response):
    """A body rotation must produce the OPPOSITE-signed steering command.

    This is the whole contract of motor.YAW: it is already corrective, so a
    consumer can feed it straight into a turn rate.
    """
    _, _, yaw = rotation_response
    # Body yawed LEFT, so the corrective command must be to steer RIGHT (+).
    assert yaw > 0, f"expected positive YAW for a leftward body rotation, got {yaw}"


def test_hs_cells_are_antisymmetric_under_rotation(rotation_response):
    """Rotation drives the two HS cells in opposite directions."""
    hs_left, hs_right, _ = rotation_response
    assert hs_right > 0 > hs_left
    # YAW is defined as HS_RIGHT - HS_LEFT.
    assert hs_right - hs_left > 0


def test_symmetric_scene_gives_zero_yaw(fly, frame_size):
    """A left/right symmetric world must not command a turn.

    The bilateral construction (the network is run on the frame and its mirror)
    makes this exact rather than approximate.
    """
    height, width = frame_size
    img = panorama(height, width, seed=3)
    # Exactly mirror-symmetric for any width, odd included.
    symmetric = np.minimum(img, img[:, ::-1])
    assert np.array_equal(symmetric, symmetric[:, ::-1])

    fly.reset()
    for _ in range(30):
        fly.set_input(sensory.VISUAL_FIELD, symmetric)
        fly.step(DT)
    assert fly.get_output(motor.YAW) == pytest.approx(0.0, abs=1e-6)
    assert fly.get_output(motor.FLOW_ASYMMETRY) == pytest.approx(0.0, abs=1e-6)


def test_mirroring_the_world_flips_the_command(fly, frame_size):
    """Mirroring the input must exactly negate YAW."""
    height, width = frame_size
    img = panorama(height, PAN_W, seed=5)
    cols = np.arange(width)

    def run(mirror):
        fly.reset()
        vals = []
        for i in range(60):
            frame = img[:, (cols - SHIFT * i) % PAN_W]
            if mirror:
                frame = frame[:, ::-1]
            fly.set_input(sensory.VISUAL_FIELD, frame)
            fly.step(DT)
            vals.append(fly.get_output(motor.YAW))
        return np.array(vals[30:])

    assert run(False) == pytest.approx(-run(True), abs=1e-6)


def test_uint8_and_float_frames_agree(fly, frame_size):
    """uint8 [0,255] and float [0,1] frames are the same stimulus."""
    height, width = frame_size
    img = panorama(height, width, seed=7)

    def run(frame):
        fly.reset()
        for _ in range(12):
            fly.set_input(sensory.VISUAL_FIELD, frame)
            fly.step(DT)
        return fly.get_output(motor.YAW)

    as_uint8 = (img * 255).astype(np.uint8)
    assert run(img) == pytest.approx(run(as_uint8), abs=2e-3)


def test_rgb_and_greyscale_agree(fly, frame_size):
    """A grey RGB frame matches the equivalent single-channel frame."""
    height, width = frame_size
    img = panorama(height, width, seed=11)
    rgb = np.repeat(img[..., None], 3, axis=2)

    def run(frame):
        fly.reset()
        for _ in range(12):
            fly.set_input(sensory.VISUAL_FIELD, frame)
            fly.step(DT)
        return fly.get_output(motor.YAW)

    assert run(img) == pytest.approx(run(rgb), abs=1e-6)


def test_reset_is_repeatable(fly, frame_size):
    """The same stimulus after reset gives the same answer."""
    height, width = frame_size
    img = panorama(height, width, seed=13)

    def run():
        fly.reset()
        for _ in range(10):
            fly.set_input(sensory.VISUAL_FIELD, img)
            fly.step(DT)
        return fly.get_output(motor.YAW)

    assert run() == pytest.approx(run(), abs=1e-9)


def test_step_without_input_raises(fly):
    fly.reset()
    with pytest.raises(RuntimeError, match="No VISUAL_FIELD input"):
        fly.step(DT)


def test_bad_frame_shape_raises(fly):
    with pytest.raises(ValueError):
        fly.set_input(sensory.VISUAL_FIELD, np.zeros((10, 10, 5)))
    with pytest.raises(ValueError):
        fly.set_input(sensory.VISUAL_FIELD, np.zeros((10,)))


def test_direction_constants_match_flyvis_ground_truth():
    """Our horizontal T4/T5 assignment must match flyvis's published tuning."""
    from flyvis.utils.groundtruth_utils import preferred_directions

    for cell_type in RIGHTWARD_TYPES:
        assert preferred_directions[cell_type] == 0, cell_type
    for cell_type in LEFTWARD_TYPES:
        assert preferred_directions[cell_type] == 180, cell_type


def test_config_is_honoured(fly, frame_size):
    """yaw_gain and yaw_clip scale and bound the command."""
    from flybrain import FlyBrain

    height, width = frame_size
    img = panorama(height, PAN_W, seed=17)
    cols = np.arange(width)

    def run(brain):
        brain.reset()
        vals = []
        for i in range(60):
            brain.set_input(sensory.VISUAL_FIELD, img[:, (cols - SHIFT * i) % PAN_W])
            brain.step(DT)
            vals.append(brain.get_output(motor.YAW))
        return np.abs(vals[30:]).max()

    clipped = FlyBrain(
        circuits=["optic_lobe"],
        configs={"optic_lobe": OpticLobeConfig(yaw_gain=1e6, yaw_clip=0.25)},
    )
    assert run(clipped) <= 0.25 + 1e-9
