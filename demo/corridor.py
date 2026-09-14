"""A textured corridor and a virtual car, for the closed-loop demo.

This module is deliberately ignorant of flybrain: it renders images and moves a
body. The demo script is what connects the two, and it does so only through the
public FlyBrain port API. Nothing here imports flyvis or torch.

Geometry
--------
The corridor runs along +z. The walls are vertical planes at x = +/- half_width.
The car sits at (x, z) with a heading measured from the +z axis, positive
turning to the right (towards +x). The camera is equirectangular: screen column
j maps linearly to a bearing, which suits a wide field of view and the fly's
near-uniform angular sampling much better than a pinhole projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["CorridorConfig", "Corridor", "CarConfig", "CarState", "step_car"]


@dataclass
class CorridorConfig:
    """Appearance and geometry of the corridor.

    Attributes:
        half_width: Distance in metres from the centreline to each wall.
        fov_deg: Horizontal field of view in degrees.
        stripe_period: Length in metres of one light+dark stripe pair.
        wall_height: Wall height in metres, centred on the camera.
        max_depth: Distance in metres beyond which walls fade to background.
        shade_falloff: Distance in metres over which brightness halves.
        stripe_dark / stripe_light: Luminance of the two stripe phases.
        background: Luminance of floor and ceiling (untextured, so they carry
            no optic flow and the demo is driven by the walls alone).
    """

    half_width: float = 2.0
    fov_deg: float = 150.0
    stripe_period: float = 0.8
    wall_height: float = 2.4
    max_depth: float = 60.0
    shade_falloff: float = 12.0
    stripe_dark: float = 0.10
    stripe_light: float = 0.95
    background: float = 0.45


class Corridor:
    """Renders a first-person view of a striped corridor.

    Args:
        size: ``(height, width)`` of the rendered image in pixels.
        config: Appearance and geometry. Defaults to :class:`CorridorConfig`.
    """

    def __init__(
        self, size: Tuple[int, int], config: CorridorConfig | None = None
    ) -> None:
        self.height, self.width = int(size[0]), int(size[1])
        self.config = config or CorridorConfig()

        fov = np.deg2rad(self.config.fov_deg)
        # Bearing of each screen column relative to the heading; + is right.
        self._bearings = np.linspace(
            -fov / 2, fov / 2, self.width, dtype=np.float64
        )
        self._px_per_rad = self.width / fov
        self._rows = np.arange(self.height, dtype=np.float64)
        self._cy = (self.height - 1) / 2.0

    def render(self, x: float, z: float, heading: float) -> np.ndarray:
        """Render the view from ``(x, z)`` facing ``heading``.

        Args:
            x: Lateral position in metres; 0 is the centreline, + is right.
            z: Position along the corridor in metres.
            heading: Heading in radians from +z; + turns right.

        Returns:
            An ``(H, W, 3)`` uint8 RGB image.
        """
        cfg = self.config
        phi = heading + self._bearings
        sx = np.sin(phi)
        cz = np.cos(phi)

        # Each ray hits exactly one wall, whichever side it points towards.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_right = (cfg.half_width - x) / sx
            t_left = (-cfg.half_width - x) / sx
        t = np.where(sx > 0, t_right, t_left)
        t = np.where(np.abs(sx) < 1e-9, np.inf, t)
        t = np.where(t > 0, t, np.inf)
        t = np.minimum(t, cfg.max_depth)

        z_hit = z + t * cz

        # Stripe phase along the corridor, and distance shading.
        stripe = np.floor(z_hit / cfg.stripe_period) % 2.0
        luminance = np.where(stripe > 0.5, cfg.stripe_light, cfg.stripe_dark)
        shade = 1.0 / (1.0 + t / cfg.shade_falloff)
        wall = cfg.background + (luminance - cfg.background) * shade

        # Equirectangular vertical extent of the wall, in pixels.
        half_px = np.arctan((cfg.wall_height / 2.0) / t) * self._px_per_rad

        on_wall = np.abs(self._rows[:, None] - self._cy) <= half_px[None, :]
        img = np.where(on_wall, wall[None, :], cfg.background)

        img = np.clip(img, 0.0, 1.0)
        rgb = (img[..., None] * 255.0).astype(np.uint8)
        return np.repeat(rgb, 3, axis=2)


@dataclass
class CarConfig:
    """Kinematics of the virtual car.

    Attributes:
        speed: Forward speed in metres per second.
        max_turn_rate: Turn rate in rad/s produced by a full-scale (+/-1)
            steering command.
    """

    speed: float = 3.0
    max_turn_rate: float = np.deg2rad(90.0)


@dataclass
class CarState:
    """Pose of the virtual car."""

    x: float = 0.0
    z: float = 0.0
    heading: float = 0.0


def step_car(
    state: CarState, steer: float, dt: float, config: CarConfig
) -> CarState:
    """Advance the car one tick under a steering command.

    Args:
        state: Current pose.
        steer: Steering command, + is right, nominally in [-1, 1].
        dt: Timestep in seconds.
        config: Kinematic constants.

    Returns:
        The new :class:`CarState`.
    """
    steer = float(np.clip(steer, -1.0, 1.0))
    heading = state.heading + steer * config.max_turn_rate * dt
    return CarState(
        x=state.x + config.speed * np.sin(heading) * dt,
        z=state.z + config.speed * np.cos(heading) * dt,
        heading=heading,
    )
