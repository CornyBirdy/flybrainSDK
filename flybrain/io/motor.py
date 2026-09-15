"""Motor (output) ports.

Every port here is something the brain produces, read back out via
:meth:`flybrain.FlyBrain.get_output`.

Sign conventions are stated from the fly's own point of view and are verified
against the pretrained model by ``tests/test_sign_convention.py``.
"""

from __future__ import annotations

from ..ports import Port

__all__ = [
    "YAW",
    "HS_LEFT",
    "HS_RIGHT",
    "FLOW_ASYMMETRY",
    "MN9_L",
    "MN9_R",
    "DNa02_L",
    "DNa02_R",
    "MDN_L",
    "MDN_R",
    "DNp09_L",
    "DNp09_R",
    "HSE_L",
    "HSE_R",
    "ALL",
]


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

# --------------------------------------------------------------------------
# MaleCNS descending and motor neuron read-outs (male_cns circuit)
#
# These are not commands and not gains. Each one is the modelled firing rate
# of one named neuron in the MaleCNS v1.0 connectome, in Hz, smoothed over
# the circuit's readout_tau. The port is named after the neuron so that there
# is never any doubt which cell it is; what the host does with a descending
# neuron's rate is the host's business.
#
# Every one of these is a LIF read-out, so all the assumptions in
# flybrain/malecns/dataset.py and lif.py apply: uniform synaptic weight,
# uniform membrane constants, sign inferred from a predicted transmitter, no
# neuromodulation, no gap junctions. Absolute rates are not calibrated
# against recordings; differences between conditions are the meaningful part.
# --------------------------------------------------------------------------


def _rate_port(name: str, description: str) -> Port:
    """A firing-rate output port for one named MaleCNS neuron."""
    return Port(
        name=name,
        direction="out",
        dtype="float32",
        shape=None,
        units="firing rate, Hz",
        description=description,
    )


MN9_L = _rate_port(
    "MN9_L",
    "Firing rate of the LEFT MN9 (bodyId 10331), the motor neuron whose "
    "contraction extends the rostrum - the largest segment of the proboscis. "
    "This is the standard read-out for proboscis extension and the one Shiu "
    "et al. (2024) use. Driving SUGAR_GRN raises it; co-driving BITTER_GRN "
    "silences it.",
)

MN9_R = _rate_port(
    "MN9_R",
    "Firing rate of the RIGHT MN9 (bodyId 16949). USE WITH CARE: this body is "
    "flagged 'RT Hard to trace' in MaleCNS v1.0 and has evidently lost inputs "
    "that MN9_L retains - it responds roughly 20x more weakly to the same "
    "sugar drive. The asymmetry is a reconstruction artefact, not biology. "
    "Prefer MN9_L, and read the two separately rather than averaging them.",
)

DNa02_L = _rate_port(
    "DNa02_L",
    "Firing rate of the LEFT DNa02, a descending neuron whose activity is "
    "associated with ipsilateral turning during walking. Read-out only: the "
    "SDK does not convert it into a steering command, because the mapping "
    "from DN rate to body turn rate is not in the connectome.",
)

DNa02_R = _rate_port(
    "DNa02_R",
    "Firing rate of the RIGHT DNa02. Mirror of DNa02_L; the left/right "
    "difference is the quantity usually of interest.",
)

MDN_L = _rate_port(
    "MDN_L",
    "Firing rate of the LEFT MDN (moonwalker descending neuron), associated "
    "with backward walking. MaleCNS annotates 4 MDN bodies, 2 per side; this "
    "port is the mean rate of the left pair.",
)

MDN_R = _rate_port(
    "MDN_R",
    "Firing rate of the RIGHT MDN. Mean of the right pair.",
)

DNp09_L = _rate_port(
    "DNp09_L",
    "Firing rate of the LEFT DNp09, a descending neuron associated with "
    "stopping and freezing.",
)

DNp09_R = _rate_port(
    "DNp09_R",
    "Firing rate of the RIGHT DNp09.",
)

HSE_L = _rate_port(
    "HSE_L",
    "Firing rate of the LEFT HSE, an equatorial horizontal-system tangential "
    "cell of the lobula plate. Worth knowing: HS cells are REAL NEURONS in "
    "MaleCNS (HSE, HSN, HSS, two of each), whereas the optic_lobe circuit has "
    "to model them because flyvis stops at the columnar types. They are only "
    "meaningful here if something is driving T4T5_DRIVE.",
)

HSE_R = _rate_port(
    "HSE_R",
    "Firing rate of the RIGHT HSE. See HSE_L.",
)

#: All motor ports defined by the SDK.
ALL = (
    YAW,
    HS_LEFT,
    HS_RIGHT,
    FLOW_ASYMMETRY,
    MN9_L,
    MN9_R,
    DNa02_L,
    DNa02_R,
    MDN_L,
    MDN_R,
    DNp09_L,
    DNp09_R,
    HSE_L,
    HSE_R,
)
