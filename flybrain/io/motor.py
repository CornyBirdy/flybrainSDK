"""Motor (output) ports.

Every port here is something the brain produces, read back out via
:meth:`flybrain.FlyBrain.get_output`.

Sign conventions are stated from the fly's own point of view and are verified
against the pretrained model by ``tests/test_sign_convention.py``.
"""

from __future__ import annotations

from ..ports import Port

__all__ = ["YAW", "HS_LEFT", "HS_RIGHT", "FLOW_ASYMMETRY", "ALL"]


YAW = Port(
    name="YAW",
    direction="out",
    dtype="float32",
    shape=None,
    units="dimensionless steering command, nominally [-1, 1]",
    description=(
        "Horizontal steering command from the optic-flow (optomotor) pathway. "
        "POSITIVE means steer RIGHT (clockwise seen from above); NEGATIVE means "
        "steer LEFT. Computed as HS_RIGHT - HS_LEFT, scaled by the circuit's "
        "yaw_gain and clipped to yaw_clip. It is already a CORRECTIVE signal: "
        "an unintended rotation of the body to the right produces a negative "
        "YAW, which steers left and nulls the rotation. Feed it straight into a "
        "turn rate; no sign flip is needed."
    ),
)

HS_LEFT = Port(
    name="HS_LEFT",
    direction="out",
    dtype="float32",
    shape=None,
    units="dimensionless, model membrane potential (arbitrary flyvis units)",
    description=(
        "Pooled activity of the LEFT horizontal-system-like cell: the "
        "front-to-back minus back-to-front motion signal, averaged over the "
        "left half of the visual field. Positive means the left visual field is "
        "streaming front-to-back (as during forward flight, or a leftward body "
        "rotation). Diagnostic output; YAW is the steering signal."
    ),
)

HS_RIGHT = Port(
    name="HS_RIGHT",
    direction="out",
    dtype="float32",
    shape=None,
    units="dimensionless, model membrane potential (arbitrary flyvis units)",
    description=(
        "Pooled activity of the RIGHT horizontal-system-like cell: the "
        "front-to-back minus back-to-front motion signal, averaged over the "
        "right half of the visual field. Mirror image of HS_LEFT. Diagnostic "
        "output; YAW is the steering signal."
    ),
)

FLOW_ASYMMETRY = Port(
    name="FLOW_ASYMMETRY",
    direction="out",
    dtype="float32",
    shape=None,
    units="dimensionless, model membrane potential (arbitrary flyvis units)",
    description=(
        "Left/right imbalance in horizontal-motion MAGNITUDE, ignoring "
        "direction: how hard the horizontal T4/T5 columns are driven on the "
        "right minus the same on the left. POSITIVE means the right side of "
        "the scene is streaming faster, which in a corridor means the right "
        "wall is nearer. Near zero under pure rotation, because rotation "
        "sweeps both sides equally fast. This is the speed-balance "
        "('centering') cue described by Srinivasan et al. for bees, and is a "
        "DIFFERENT computation from the HS direction difference in YAW, which "
        "stabilises rotation only. A corrective turn steers AWAY from the "
        "faster side, i.e. is proportional to -FLOW_ASYMMETRY. Not used by "
        "YAW; combine the two yourself if you want both course and position "
        "held."
    ),
)

#: All motor ports defined by the SDK.
ALL = (YAW, HS_LEFT, HS_RIGHT, FLOW_ASYMMETRY)
