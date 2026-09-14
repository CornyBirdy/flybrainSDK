"""Sensory (input) ports.

Every port here is something the outside world pushes *into* the brain via
:meth:`flybrain.FlyBrain.set_input`.
"""

from __future__ import annotations

from ..ports import Port

__all__ = ["VISUAL_FIELD", "ALL"]


VISUAL_FIELD = Port(
    name="VISUAL_FIELD",
    direction="in",
    dtype="float32",
    shape=("H", "W", 3),
    units="luminance, normalised to [0, 1]",
    description=(
        "First-person view of the world as an RGB image, shape (H, W, 3), or a "
        "greyscale image of shape (H, W). uint8 arrays in [0, 255] and float "
        "arrays in [0, 1] are both accepted and normalised internally. The image "
        "is converted to luminance and resampled onto the fly's 721-facet "
        "hexagonal photoreceptor lattice. Image +x is the fly's RIGHT, image +y "
        "is DOWN. Use FlyBrain.input_shape(VISUAL_FIELD) to get the resolution "
        "that avoids resampling."
    ),
)

#: All sensory ports defined by the SDK.
ALL = (VISUAL_FIELD,)
