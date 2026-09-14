"""Shared fixtures.

Tests that need the pretrained network are skipped rather than failed when
flyvis or its weights are unavailable, so the fast tests still run anywhere.
"""

import numpy as np
import pytest


def _weights_available() -> bool:
    try:
        import flyvis
    except Exception:
        return False
    return (flyvis.results_dir / "flow" / "0000").exists()


requires_weights = pytest.mark.skipif(
    not _weights_available(),
    reason="pretrained flyvis weights not available; run 'flyvis download-pretrained'",
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
