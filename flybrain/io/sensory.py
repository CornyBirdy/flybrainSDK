"""Sensory (input) ports.

Every port here is something the outside world pushes *into* the brain via
:meth:`flybrain.FlyBrain.set_input`.
"""

from __future__ import annotations

from ..ports import Port

__all__ = [
    "VISUAL_FIELD",
    "SUGAR_GRN",
    "BITTER_GRN",
    "WATER_GRN",
    "HIGH_SALT_GRN",
    "T4T5_DRIVE",
    "ALL",
]


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

# --------------------------------------------------------------------------
# MaleCNS gustatory inputs (male_cns circuit)
#
# Each port drives a named population of gustatory receptor neurons at a
# firing rate in Hz, the way an optogenetic experiment would. The value is a
# LEVEL, not an event: it persists across steps until you set it again, and
# 0.0 means the population is silent.
#
# The types behind each port are stated because they matter. MaleCNS carries
# no taste-modality annotation at all - only anatomical type names - so every
# mapping below comes from the literature, not from the release:
#
#   Schlegel et al., Nature 634, 139-152 (2024), FlyWire annotations, for the
#   LB1 = bitter / LB3 = sugar-water division;
#   Tastekin et al., "From Sensory Detection to Motor Action: The
#   Comprehensive Drosophila Taste-Feeding Connectome", bioRxiv
#   2025.08.25.671814 / Cell (2026), for the split of LB3 into subtypes,
#   matched to Gr64f-, Gr33a-, ppk28- and Ir7c-GAL4 projection patterns.
# --------------------------------------------------------------------------

SUGAR_GRN = Port(
    name="SUGAR_GRN",
    direction="in",
    dtype="float32",
    shape=None,
    units="firing rate, Hz",
    description=(
        "Drive to the sweet-sensing labellar gustatory receptor neurons: "
        "MaleCNS types LB3b and LB3c, 34 neurons. Tastekin et al. matched "
        "both to the Gr64f-GAL4 projection pattern. This is the appetitive "
        "taste input; driving it at 100-200 Hz produces the proboscis-"
        "extension response read out on MN9_L / MN9_R. Note their Figure 2F: "
        "MaleCNS reconstruction was hardest for LB3a and LB3b, so this "
        "population may be slightly under-reconstructed."
    ),
)

BITTER_GRN = Port(
    name="BITTER_GRN",
    direction="in",
    dtype="float32",
    shape=None,
    units="firing rate, Hz",
    description=(
        "Drive to the bitter / aversive labellar gustatory receptor neurons: "
        "MaleCNS types LB1a-LB1e, 57 neurons. LB1a-d match Gr33a-GAL4, which "
        "labels all bitter-sensing GRNs; LB1e is Ir94e-like and clusters with "
        "them. Co-activating this with SUGAR_GRN suppresses MN9. Caveat: 6 of "
        "the 57 (all of LB1b) have an 'unclear' transmitter prediction and so "
        "have no modelled output, making the effective population 51."
    ),
)

WATER_GRN = Port(
    name="WATER_GRN",
    direction="in",
    dtype="float32",
    shape=None,
    units="firing rate, Hz",
    description=(
        "Drive to the water-sensing labellar GRNs: MaleCNS type LB3a, 17 "
        "neurons, matched to ppk28-GAL4. Appetitive in the animal, but in "
        "this model it does not drive MN9."
    ),
)

HIGH_SALT_GRN = Port(
    name="HIGH_SALT_GRN",
    direction="in",
    dtype="float32",
    shape=None,
    units="firing rate, Hz",
    description=(
        "Drive to the high-salt / aversive labellar GRNs: MaleCNS type LB3d, "
        "26 neurons, matched to Ir7c-GAL4 and ppk23-GAL4. READ THE WARNING: "
        "in the animal these are glutamatergic and are thought to inhibit "
        "appetitive circuits, but MaleCNS v1.0 predicts all 26 to be "
        "CHOLINERGIC (mean confidence 0.71, no ground truth), so in this "
        "model they excite the feeding pathway and drive MN9 harder than "
        "sugar does. The port is exposed because the discrepancy is worth "
        "seeing, not because the behaviour is right."
    ),
)

T4T5_DRIVE = Port(
    name="T4T5_DRIVE",
    direction="in",
    dtype="float32",
    shape=("n_types",),
    units="firing rate, Hz, per T4/T5 subtype",
    description=(
        "Drive to the MaleCNS T4/T5 motion-detector populations, as eight "
        "rates in the order (T4a, T4b, T4c, T4d, T5a, T5b, T5c, T5d), or a "
        "mapping from subtype name to rate. A scalar drives all eight "
        "equally. Exists so that another circuit's motion estimate can be "
        "injected here - see flybrain.circuits.coupling, which is an "
        "explicitly opt-in HYPOTHESIS and not a measured correspondence."
    ),
)

#: All sensory ports defined by the SDK.
ALL = (
    VISUAL_FIELD,
    SUGAR_GRN,
    BITTER_GRN,
    WATER_GRN,
    HIGH_SALT_GRN,
    T4T5_DRIVE,
)
