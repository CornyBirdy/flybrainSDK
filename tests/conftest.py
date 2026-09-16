"""Shared fixtures.

Tests that need a downloaded model are skipped rather than failed when it is
unavailable, so the fast tests still run anywhere. There are two such models:
the pretrained flyvis weights (``optic_lobe``) and the MaleCNS connectome
cache (``male_cns``).
"""

import numpy as np
import pytest


def _weights_available() -> bool:
    """Whether optic_lobe can actually run: torch, flyvis AND the weights."""
    try:
        import torch  # noqa: F401

        import flyvis
    except Exception:
        return False
    return (flyvis.results_dir / "flow" / "0000").exists()


requires_weights = pytest.mark.skipif(
    not _weights_available(),
    reason=(
        "optic_lobe unavailable: needs torch and flyvis "
        "('pip install flybrain[optic_lobe]') and the pretrained weights "
        "('flyvis download-pretrained')"
    ),
)


@pytest.fixture(scope="session")
def fly():
    """A FlyBrain with the optic lobe loaded once for the whole session."""
    pytest.importorskip("flyvis")
    if not _weights_available():
        pytest.skip("pretrained flyvis weights not available")
    from flybrain import FlyBrain

    return FlyBrain(circuits=["optic_lobe"])


@pytest.fixture(scope="session")
def frame_size(fly):
    from flybrain import sensory

    return fly.input_shape(sensory.VISUAL_FIELD)


def panorama(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Band-limited random texture in [0, 1], seamless in x."""
    rng = np.random.default_rng(seed)
    img = rng.random((height, width)).astype(np.float32)
    k = np.hanning(17).astype(np.float32)
    k /= k.sum()
    img = np.apply_along_axis(
        lambda m: np.convolve(np.r_[m[-17:], m, m[:17]], k, "same")[17:-17], 1, img
    )
    img = np.apply_along_axis(lambda m: np.convolve(m, k, "same"), 0, img)
    return ((img - img.min()) / np.ptp(img)).astype(np.float32)


def _connectome_available() -> bool:
    """Whether male_cns can actually run: its stack AND its cache.

    Both halves matter. ``flybrain.malecns`` imports pandas, pyarrow and scipy
    lazily inside the functions that use them, so it imports fine without them
    and only fails later -- and the cache lives in ``~/.cache`` and outlives any
    virtualenv. So a numpy-only install with a cache left over from a previous
    environment passes a cache-only check and then errors out 15 times in the
    fixture instead of skipping. Both circuits are extras over a numpy-only
    core (see ``pyproject.toml``), which makes that an ordinary situation rather
    than an exotic one.
    """
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
        import scipy.sparse  # noqa: F401

        from flybrain.malecns import cache_available
    except Exception:
        return False
    return cache_available()


requires_connectome = pytest.mark.skipif(
    not _connectome_available(),
    reason=(
        "male_cns unavailable: needs pandas, pyarrow and scipy "
        "('pip install flybrain[male_cns]') and a built connectome cache "
        "('python -m flybrain.malecns download')"
    ),
)


@pytest.fixture(scope="session")
def male_cns():
    """A MaleCNSCircuit loaded once for the whole session.

    Loading the connectome costs a second or two and about a gigabyte, so it
    is emphatically not per-test.
    """
    if not _connectome_available():
        pytest.skip("MaleCNS connectome cache not built")
    from flybrain.circuits import MaleCNSCircuit

    return MaleCNSCircuit()


def mean_rate(circuit, port, drive, duration, reset=True):
    """Drive ``circuit`` for ``duration`` seconds and return ``port``'s rate.

    Uses the cumulative, unsmoothed mean rate rather than the smoothed
    read-out, so a test result does not depend on ``readout_tau``.

    Args:
        circuit: A MaleCNSCircuit.
        port: The motor port to measure.
        drive: Mapping of sensory port to firing rate in Hz.
        duration: Seconds of simulated time.
        reset: Whether to reset the circuit first.

    Returns:
        Mean firing rate over the run, in Hz.
    """
    if reset:
        circuit.reset()
    for sensory_port, rate in drive.items():
        circuit.set_input(sensory_port, rate)
    circuit.step(duration)
    return float(circuit.mean_rates()[circuit.population(port)].mean())
