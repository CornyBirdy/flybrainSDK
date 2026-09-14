"""Closed-loop demo: a virtual car steered by a pretrained fly optic lobe.

Each tick renders a first-person view of a striped corridor, hands it to the
fly brain through the public port API, reads the steering command back out, and
moves the car. Nothing here imports flyvis or torch: the only contact with the
neuroscience is ``set_input`` / ``step`` / ``get_output``.

    python demo/run_demo.py                # closed + open loop, writes outputs
    python demo/run_demo.py --live         # live matplotlib window
    python demo/run_demo.py --duration 12  # shorter run

Outputs (in --outdir, default demo/output):
    corridor_demo.gif        animation: fly's view, top-down track, telemetry
    corridor_telemetry.png   static summary comparing closed and open loop
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.corridor import (  # noqa: E402
    CarConfig,
    CarState,
    Corridor,
    CorridorConfig,
    step_car,
)
from flybrain import FlyBrain, motor, sensory  # noqa: E402

# ---------------------------------------------------------------------------
# Tunable constants. Everything the loop's behaviour depends on lives here.
# ---------------------------------------------------------------------------

#: Simulation tick, seconds. The brain integrates with its own finer substeps.
DT = 1.0 / 60.0

#: Wall-clock duration of the run, seconds.
DURATION = 24.0

#: Forward speed, m/s, and corridor half-width, m. Their RATIO sets how much
#: translational flow the walls produce (speed / half_width rad/s). Keep it
#: well below the rotation rates you want the fly to resolve, or the expansion
#: flow swamps the rotation signal the HS cells are reading.
SPEED = 0.5
HALF_WIDTH = 3.0

#: Turn rate, rad/s, commanded by a full-scale (+/-1) steering signal.
MAX_TURN_RATE = np.deg2rad(60.0)

#: Weight on motor.YAW, the HS left/right difference. This is the optomotor
#: term: it opposes body ROTATION. YAW already carries the corrective sign.
YAW_WEIGHT = 1.0

#: Weight on -motor.FLOW_ASYMMETRY, the left/right speed balance. This is the
#: centering term: it steers away from whichever side is streaming faster, and
#: so corrects lateral and heading offsets, which YAW alone does not.
#: Set to 0.0 to see the optomotor term in isolation.
CENTERING_WEIGHT = 8.0

#: Rotational gusts: {time in seconds: heading kick in degrees}. These are the
#: "nudges" the fly has to recover from.
GUSTS: Dict[float, float] = {4.0: 18.0, 10.0: -22.0, 16.0: 15.0}

#: Where the car starts: off-centre, so the run also shows it recentering.
START_X = 1.2

#: Appearance of the corridor.
CORRIDOR = CorridorConfig(
    half_width=HALF_WIDTH,
    stripe_period=1.0,
    shade_falloff=20.0,
    max_depth=80.0,
)

#: Downsampled size of the stored view frames, to keep the animation small.
VIEW_THUMB = 150
#: Keep every Nth frame in the animation.
ANIM_STRIDE = 3


@dataclass
class Telemetry:
    """Per-tick record of one run."""

    label: str
    t: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    z: List[float] = field(default_factory=list)
    heading: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    asymmetry: List[float] = field(default_factory=list)
    steer: List[float] = field(default_factory=list)
    views: List[np.ndarray] = field(default_factory=list)
    crashed_at: Optional[float] = None

    def as_arrays(self) -> Dict[str, np.ndarray]:
        return {
            k: np.asarray(v)
            for k, v in vars(self).items()
            if isinstance(v, list) and k != "views"
        }


def simulate(
    fly: FlyBrain,
    world: Corridor,
    closed_loop: bool,
    duration: float,
    keep_views: bool,
) -> Telemetry:
    """Run one closed- or open-loop pass down the corridor.

    Args:
        fly: The brain. Only its public port API is used.
        world: The corridor renderer.
        closed_loop: If False the steering command is ignored, for comparison.
        duration: Run length in seconds.
        keep_views: Whether to store downsampled frames for the animation.

    Returns:
        A :class:`Telemetry` record of the run.
    """
    fly.reset()
    car = CarState(x=START_X, z=0.0, heading=0.0)
    car_cfg = CarConfig(speed=SPEED, max_turn_rate=MAX_TURN_RATE)
    tel = Telemetry(label="closed loop" if closed_loop else "open loop")

    n_ticks = int(round(duration / DT))
    for i in range(n_ticks):
        t = i * DT

        frame = world.render(car.x, car.z, car.heading)

        # ---- the entire brain interface ----
        fly.set_input(sensory.VISUAL_FIELD, frame)
        fly.step(DT)
        yaw = fly.get_output(motor.YAW)
        asym = fly.get_output(motor.FLOW_ASYMMETRY)
        # ------------------------------------

        steer = YAW_WEIGHT * yaw - CENTERING_WEIGHT * asym
        steer = float(np.clip(steer, -1.0, 1.0))
        if not closed_loop:
            steer = 0.0

        car = step_car(car, steer, DT, car_cfg)

        for gust_t, gust_deg in GUSTS.items():
            if abs(t - gust_t) < DT / 2:
                car = CarState(car.x, car.z, car.heading + np.deg2rad(gust_deg))

        tel.t.append(t)
        tel.x.append(car.x)
        tel.z.append(car.z)
        tel.heading.append(np.rad2deg(car.heading))
        tel.yaw.append(yaw)
        tel.asymmetry.append(asym)
        tel.steer.append(steer)
        if keep_views and i % ANIM_STRIDE == 0:
            step = max(1, frame.shape[0] // VIEW_THUMB)
            tel.views.append(frame[::step, ::step, 0].copy())

        if abs(car.x) >= HALF_WIDTH:
            tel.crashed_at = t
            break

    return tel


def summarise(tel: Telemetry) -> str:
    """One-line summary of a run."""
    a = tel.as_arrays()
    end = "hit wall at %.1fs" % tel.crashed_at if tel.crashed_at else "completed"
    return (
        f"{tel.label:12s} {end:20s} "
        f"final x={a['x'][-1]:+.2f} m  max|x|={np.abs(a['x']).max():.2f} m  "
        f"RMS heading={np.sqrt((a['heading'] ** 2).mean()):.1f} deg"
    )


def plot_telemetry(runs: List[Telemetry], path: str) -> None:
    """Write the static comparison figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    colours = {"closed loop": "#1b7837", "open loop": "#b2182b"}

    for tel in runs:
        a = tel.as_arrays()
        c = colours.get(tel.label, "#555555")
        axes[0].plot(a["t"], a["x"], color=c, label=tel.label)
        axes[1].plot(a["t"], a["heading"], color=c, label=tel.label)
        if tel.label == "closed loop":
            axes[2].plot(a["t"], a["yaw"], color="#2166ac", label="YAW (optomotor)")
            axes[2].plot(
                a["t"], -CENTERING_WEIGHT * a["asymmetry"],
                color="#d6604d", label=f"-{CENTERING_WEIGHT:g} x FLOW_ASYMMETRY",
            )
            axes[2].plot(a["t"], a["steer"], color="k", lw=1.4, label="steer")

    for ax in axes:
        for gust_t, gust_deg in GUSTS.items():
            ax.axvline(gust_t, color="#999999", ls=":", lw=1)
        ax.grid(alpha=0.25)

    axes[0].axhline(HALF_WIDTH, color="k", lw=1.2)
    axes[0].axhline(-HALF_WIDTH, color="k", lw=1.2)
    axes[0].axhline(0, color="#999999", lw=0.8)
    axes[0].set_ylabel("lateral offset x  [m]")
    axes[0].set_title(
        "Fly optic lobe steering a corridor (dotted lines = rotational gusts)"
    )
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].axhline(0, color="#999999", lw=0.8)
    axes[1].set_ylabel("heading  [deg]")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].axhline(0, color="#999999", lw=0.8)
    axes[2].set_ylabel("steering signals")
    axes[2].set_xlabel("time  [s]")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate(tel: Telemetry, path: Optional[str], live: bool) -> None:
    """Render the three-panel animation, to ``path`` and/or a live window."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    a = tel.as_arrays()
    idx = np.arange(0, len(a["t"]), ANIM_STRIDE)[: len(tel.views)]

    fig = plt.figure(figsize=(11, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.6], hspace=0.45, wspace=0.2)
    ax_view = fig.add_subplot(grid[:, 0])
    ax_map = fig.add_subplot(grid[0, 1])
    ax_sig = fig.add_subplot(grid[1, 1])

    im = ax_view.imshow(tel.views[0], cmap="gray", vmin=0, vmax=255)
    ax_view.set_title("what the fly sees (VISUAL_FIELD)", fontsize=10)
    ax_view.set_xticks([])
    ax_view.set_yticks([])

    z_max = max(a["z"].max(), 1.0)
    ax_map.set_xlim(0, z_max)
    ax_map.set_ylim(-HALF_WIDTH * 1.05, HALF_WIDTH * 1.05)
    ax_map.axhline(HALF_WIDTH, color="k", lw=2)
    ax_map.axhline(-HALF_WIDTH, color="k", lw=2)
    ax_map.axhline(0, color="#bbbbbb", lw=0.8, ls="--")
    (track,) = ax_map.plot([], [], color="#1b7837", lw=1.6)
    (dot,) = ax_map.plot([], [], "o", color="#1b7837", ms=7)
    ax_map.set_ylabel("x  [m]")
    ax_map.set_title("top-down track", fontsize=10)

    ax_sig.set_xlim(0, a["t"][-1])
    lo = min(a["steer"].min(), a["yaw"].min(), -0.5)
    hi = max(a["steer"].max(), a["yaw"].max(), 0.5)
    ax_sig.set_ylim(lo * 1.15, hi * 1.15)
    ax_sig.axhline(0, color="#bbbbbb", lw=0.8)
    for gust_t in GUSTS:
        ax_sig.axvline(gust_t, color="#999999", ls=":", lw=1)
    (l_yaw,) = ax_sig.plot([], [], color="#2166ac", lw=1.3, label="YAW")
    (l_steer,) = ax_sig.plot([], [], color="k", lw=1.3, label="steer")
    ax_sig.legend(loc="upper right", fontsize=8)
    ax_sig.set_xlabel("time  [s]")

    def update(k):
        j = idx[k]
        im.set_data(tel.views[k])
        track.set_data(a["z"][: j + 1], a["x"][: j + 1])
        dot.set_data([a["z"][j]], [a["x"][j]])
        l_yaw.set_data(a["t"][: j + 1], a["yaw"][: j + 1])
        l_steer.set_data(a["t"][: j + 1], a["steer"][: j + 1])
        ax_view.set_xlabel(
            f"t={a['t'][j]:5.1f}s   x={a['x'][j]:+.2f} m   "
            f"heading={a['heading'][j]:+.1f}deg",
            fontsize=9,
        )
        return im, track, dot, l_yaw, l_steer

    anim = animation.FuncAnimation(
        fig, update, frames=len(tel.views), interval=1000 * DT * ANIM_STRIDE,
        blit=False,
    )

    if path:
        fps = max(1, int(round(1.0 / (DT * ANIM_STRIDE))))
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=72)
        print(f"wrote {path}")
    if live:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--outdir", default=os.path.join("demo", "output"))
    parser.add_argument("--live", action="store_true", help="show a live window")
    parser.add_argument("--no-anim", action="store_true", help="skip the gif")
    parser.add_argument(
        "--no-open-loop", action="store_true", help="skip the comparison run"
    )
    args = parser.parse_args()

    if not args.live:
        import matplotlib

        matplotlib.use("Agg")

    os.makedirs(args.outdir, exist_ok=True)

    fly = FlyBrain(circuits=["optic_lobe"])
    height, width = fly.input_shape(sensory.VISUAL_FIELD)
    world = Corridor((height, width), CORRIDOR)

    print(fly.describe())
    print(
        f"\ncorridor half-width {HALF_WIDTH} m, speed {SPEED} m/s "
        f"(wall flow {np.rad2deg(SPEED / HALF_WIDTH):.0f} deg/s), "
        f"gusts at {sorted(GUSTS)} s"
    )
    print(f"steer = {YAW_WEIGHT:g}*YAW - {CENTERING_WEIGHT:g}*FLOW_ASYMMETRY\n")

    runs = [simulate(fly, world, True, args.duration, keep_views=True)]
    print(summarise(runs[0]))
    if not args.no_open_loop:
        runs.append(simulate(fly, world, False, args.duration, keep_views=False))
        print(summarise(runs[1]))

    telemetry_path = os.path.join(args.outdir, "corridor_telemetry.png")
    plot_telemetry(runs, telemetry_path)
    print(f"wrote {telemetry_path}")

    if not args.no_anim or args.live:
        animate(
            runs[0],
            None if args.no_anim else os.path.join(args.outdir, "corridor_demo.gif"),
            args.live,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
