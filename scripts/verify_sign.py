"""Empirically verify the YAW sign convention against the pretrained weights.

Pure yaw rotation is simulated by sliding a band-limited random panorama across
the fly's view. The panorama wraps and is traversed COMPLETELY in both
directions, so any fixed left/right bias from the texture itself is identical in
the two runs and cancels when they are differenced:

    YAW(scene right) = bias + motion
    YAW(scene left)  = bias - motion

The test is that the motion term has the sign of a corrective optomotor
response: a body rotation to the right must produce a NEGATIVE YAW (steer left).

Run:  python scripts/verify_sign.py
"""
import numpy as np

from flybrain import FlyBrain, motor, sensory
from flybrain.circuits import OpticLobeConfig

PAN_W = 780          # panorama width in px; wraps
SHIFT = 5            # px/frame -> full traversal in PAN_W/SHIFT = 156 frames
CYCLE = PAN_W // SHIFT
N = 2 * CYCLE        # two full traversals; the first is discarded as transient
DT = 1 / 60
SMOOTH = 17          # px; band-limits the texture to ~1.5 facets


def band_limited_panorama(height, width, seed=0):
    """Random texture with structure at the photoreceptor scale, in [0, 1]."""
    rng = np.random.default_rng(seed)
    img = rng.random((height, width)).astype(np.float32)
    kern = np.hanning(SMOOTH).astype(np.float32)
    kern /= kern.sum()
    # Wrap-around blur along x keeps the panorama seamless.
    img = np.apply_along_axis(
        lambda m: np.convolve(np.r_[m[-SMOOTH:], m, m[:SMOOTH]], kern, "same")[
            SMOOTH:-SMOOTH
        ],
        1, img,
    )
    img = np.apply_along_axis(lambda m: np.convolve(m, kern, "same"), 0, img)
    return ((img - img.min()) / np.ptp(img)).astype(np.float32)


def main():
    fly = FlyBrain(
        circuits=["optic_lobe"],
        # unity gain, no clipping: we want the raw HS difference
        configs={"optic_lobe": OpticLobeConfig(yaw_gain=1.0, yaw_clip=None)},
    )
    H, W = fly.input_shape(sensory.VISUAL_FIELD)
    panorama = band_limited_panorama(H, PAN_W)
    cols = np.arange(W)

    def run(shift_per_frame):
        """shift_per_frame > 0 slides scene content to the RIGHT on screen.

        Screen column j shows panorama column j - off, so a feature sits at
        j = p + off: increasing ``off`` moves it RIGHT across the screen.
        """
        fly.reset()
        trace = []
        for i in range(N):
            off = int(shift_per_frame * i)
            frame = panorama[:, (cols - off) % PAN_W]
            fly.set_input(sensory.VISUAL_FIELD, frame)
            fly.step(DT)
            trace.append((
                fly.get_output(motor.HS_LEFT),
                fly.get_output(motor.HS_RIGHT),
                fly.get_output(motor.YAW),
            ))
        return np.array(trace[CYCLE:]).mean(0)  # second traversal only

    print(f"frame {H}x{W}, panorama {PAN_W}px, {SHIFT}px/frame, "
          f"{N} frames/condition\n")

    l_plus, r_plus, y_plus = run(+SHIFT)   # scene -> right == body yaws LEFT
    l_minus, r_minus, y_minus = run(-SHIFT)  # scene -> left  == body yaws RIGHT

    print(f"  scene slides RIGHT (body yaws LEFT ):"
          f"  HS_L={l_plus:+.5f}  HS_R={r_plus:+.5f}  YAW={y_plus:+.5f}")
    print(f"  scene slides LEFT  (body yaws RIGHT):"
          f"  HS_L={l_minus:+.5f}  HS_R={r_minus:+.5f}  YAW={y_minus:+.5f}")

    bias = 0.5 * (y_plus + y_minus)
    motion = 0.5 * (y_plus - y_minus)
    hs_l_motion = 0.5 * (l_plus - l_minus)
    hs_r_motion = 0.5 * (r_plus - r_minus)

    print(f"\n  texture bias (common mode) : {bias:+.5f}")
    print(f"  motion term  (differential): {motion:+.5f}")
    print(f"    HS_LEFT  motion term     : {hs_l_motion:+.5f}"
          "   (body yaws left -> left field streams back-to-front -> negative)")
    print(f"    HS_RIGHT motion term     : {hs_r_motion:+.5f}"
          "   (body yaws left -> right field streams front-to-back -> positive)")

    ok = motion > 0
    print(f"\n  body yaws LEFT  -> YAW {'+' if motion > 0 else '-'}ve "
          f"(steer right, corrective)")
    print(f"  body yaws RIGHT -> YAW {'-' if motion > 0 else '+'}ve "
          f"(steer left,  corrective)")
    print("\nSIGN CONVENTION:", "VERIFIED" if ok else "MISMATCH")
    if ok:
        print(f"raw |YAW| motion term at {SHIFT}px/frame: {motion:.5f}")
        print(f"-> yaw_gain ~{1.0 / abs(motion):.0f} maps this to full scale")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
