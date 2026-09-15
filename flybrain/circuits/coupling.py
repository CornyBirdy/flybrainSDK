"""Driving MaleCNS from flyvis. THIS IS A HYPOTHESIS, NOT A RESULT.

Read this section before using anything in this module.

There is no principled mapping between flyvis cell types and MaleCNS neurons.
The two models are built from **different animals**, imaged in **different EM
volumes**, reconstructed in **different coordinate frames**, and one of them
is a sex the other is not. Nothing in either dataset establishes that a given
flyvis column corresponds to a given MaleCNS cell. What this module implements
is a guess that seemed defensible, wrapped so that it is impossible to invoke
by accident:

* it is off unless you pass ``MaleCNSConfig(enable_visual_coupling=True)``;
* you then have to construct :class:`OpticLobeToMaleCNS` yourself and call
  :meth:`~OpticLobeToMaleCNS.transfer` on every frame;
* ``FlyBrain(circuits=["optic_lobe", "male_cns"])`` does **not** couple the
  two. Both circuits run, independently, exactly as they do alone.

Why bother at all
-----------------
Because the two models are complementary in a specific and interesting way.
flyvis stops at the columnar types: it has T4/T5 motion detectors whose
parameters were *fitted against neural recordings*, but it has no HS cells,
which is why ``optic_lobe`` has to model the HS read-out rather than simulate
it. MaleCNS has the real HS cells - HSE, HSN and HSS, two of each, with their
measured wiring - but no fitted dynamics and no photoreceptors that see
anything. Bridging T4/T5 to T4/T5 is the one place the seam is narrow enough
to be worth probing.

Everything that is assumed, in order of how much it should worry you
--------------------------------------------------------------------
1. **Retinotopy is thrown away.** flyvis has 721 columns per subtype; MaleCNS
   has roughly 840 cells per subtype per side. This module does not attempt a
   column-to-cell correspondence, because none is known. It pools flyvis
   activity to one number per subtype per lobe and drives every MaleCNS cell
   of that subtype at the same rate. A real HS cell integrates over a
   *spatial* pattern of T4/T5 input; this destroys exactly that pattern, so
   the HS read-outs downstream cannot show direction selectivity that depends
   on where in the visual field the motion is.
2. **flyvis activity is not a firing rate.** It is a dimensionless model
   membrane potential in arbitrary units. Converting it to a Poisson rate in
   Hz needs a gain and an offset, and both are invented here
   (:class:`CouplingConfig`). There is no measurement that fixes them.
3. **Subtype letters are assumed to mean the same thing in both datasets.**
   flyvis ships measured preferred directions: T4a/T5a leftward, T4b/T5b
   rightward, T4c/T5c upward, T4d/T5d downward. MaleCNS uses the same Janelia
   naming convention, so a-to-a and b-to-b is the natural reading - but it is
   a naming convention, not a measurement of preferred direction in this
   animal's cells.
4. **Chirality is assumed.** ``optic_lobe`` simulates two lobes by running
   flyvis on the frame and on its mirror image, so its "left lobe" sees a
   mirrored world in its own retinotopic frame. MaleCNS's left lobe is a real
   left lobe. Mapping flyvis eye 0 to MaleCNS ``_R`` and eye 1 to ``_L`` is
   consistent under that mirroring, but it has not been checked against
   anything.
5. **Rectification.** flyvis activity is a deviation that can go negative;
   a firing rate cannot. Negative deviations are clipped to zero, which is a
   half-wave rectification nobody measured.

What would make this real
-------------------------
A column-to-cell registration between the flyvis lattice and the MaleCNS
optic lobe, and a calibration of flyvis units against measured T4/T5 firing
rates. Neither exists here. Until they do, treat any behaviour that comes out
of this module as a property of these assumptions, not of either connectome.

Usage
-----
::

    from flybrain import sensory
    from flybrain.circuits import MaleCNSCircuit, MaleCNSConfig, OpticLobeCircuit
    from flybrain.circuits.coupling import OpticLobeToMaleCNS

    eye = OpticLobeCircuit()
    cns = MaleCNSCircuit(MaleCNSConfig(enable_visual_coupling=True))
    bridge = OpticLobeToMaleCNS(eye, cns)

    eye.set_input(sensory.VISUAL_FIELD, frame)
    eye.step(dt)
    bridge.transfer()      # explicit, every frame
    cns.step(dt)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..io import sensory
from .male_cns import T4T5_SUBTYPES, MaleCNSCircuit
from .optic_lobe import LEFT_EYE, RIGHT_EYE, OpticLobeCircuit

__all__ = ["CouplingConfig", "OpticLobeToMaleCNS", "SUBTYPE_MAP"]

#: flyvis cell type -> MaleCNS cell type. Identity, on the assumption that the
#: Janelia subtype letters mean the same thing in both datasets. See
#: assumption 3 in the module docstring; this is the weakest link that is
#: cheap to change.
SUBTYPE_MAP: Dict[str, str] = {subtype: subtype for subtype in T4T5_SUBTYPES}


@dataclass
class CouplingConfig:
    """Invented constants converting flyvis activity into a firing rate.

    None of these come from a measurement. They exist so that the guess is
    visible and adjustable rather than buried in a method.

    Attributes:
        gain_hz_per_unit: Hz of Poisson drive per unit of flyvis activity
            deviation. flyvis activity is dimensionless and typically of
            order 0.01-0.1 for a textured moving scene, so 1000 puts a
            strongly driven subtype in the tens of Hz. Pure invention.
        baseline_hz: Rate applied when a subtype's deviation is zero. Real
            T4/T5 cells have a nonzero baseline; this model's neurons have
            none, so leaving it at 0 keeps the "silent input, silent output"
            guarantee intact. Raising it breaks that guarantee deliberately.
        max_hz: Upper clip on the drive, in Hz. Guards against a bright
            transient injecting an implausible rate.
        rectify: Clip negative deviations to zero rather than taking their
            magnitude. See assumption 5.
        subtypes: Which subtypes to couple. The default is the horizontal
            pair only - a, b - because those are the ones the HS cells
            integrate, and coupling the vertical pair as well adds drive with
            no read-out to show for it. Pass all eight to couple everything.
    """

    gain_hz_per_unit: float = 1000.0
    baseline_hz: float = 0.0
    max_hz: float = 200.0
    rectify: bool = True
    subtypes: Tuple[str, ...] = ("T4a", "T4b", "T5a", "T5b")


class OpticLobeToMaleCNS:
    """An explicit, opt-in bridge from flyvis T4/T5 activity to MaleCNS T4/T5.

    Args:
        optic_lobe: A stepped :class:`~flybrain.circuits.optic_lobe.OpticLobeCircuit`.
        male_cns: A :class:`~flybrain.circuits.male_cns.MaleCNSCircuit`
            constructed with ``enable_visual_coupling=True``.
        config: Invented conversion constants.

    Raises:
        TypeError: If either circuit is of the wrong kind.
        ValueError: If the MaleCNS circuit has not opted in, or if a
            configured subtype has no mapping.
    """

    def __init__(
        self,
        optic_lobe: OpticLobeCircuit,
        male_cns: MaleCNSCircuit,
        config: CouplingConfig | None = None,
    ) -> None:
        if not isinstance(optic_lobe, OpticLobeCircuit):
            raise TypeError(f"expected an OpticLobeCircuit, got {type(optic_lobe)}")
        if not isinstance(male_cns, MaleCNSCircuit):
            raise TypeError(f"expected a MaleCNSCircuit, got {type(male_cns)}")
        if not male_cns.config.enable_visual_coupling:
            raise ValueError(
                "This coupling is a hypothesis, not a measured "
                "correspondence, so it has to be asked for: construct the "
                "circuit as MaleCNSCircuit(MaleCNSConfig("
                "enable_visual_coupling=True)). Read the docstring of "
                "flybrain.circuits.coupling first."
            )
        self.config = config or CouplingConfig()
        unmapped = set(self.config.subtypes) - set(SUBTYPE_MAP)
        if unmapped:
            raise ValueError(
                f"No MaleCNS counterpart configured for {sorted(unmapped)}. "
                f"Known: {sorted(SUBTYPE_MAP)}."
            )
        self.optic_lobe = optic_lobe
        self.male_cns = male_cns
        self._last_rates: Dict[str, float] = {}

    def transfer(self) -> Dict[str, float]:
        """Read flyvis T4/T5 activity and stage it as MaleCNS drive.

        Call this after the optic lobe has stepped and before the MaleCNS
        circuit steps. Nothing calls it for you.

        Returns:
            The staged rate in Hz for each MaleCNS population, keyed by
            ``"<subtype>_<side>"``, e.g. ``"T4b_R"``. Useful for seeing what
            the invented gain is actually producing.

        Raises:
            RuntimeError: If the optic lobe has not been stepped since its
                last reset.
        """
        config = self.config
        # Every per-side population is named, including the ones this
        # transfer leaves at zero, so a subtype dropping out of
        # config.subtypes silences it rather than freezing its last rate.
        rates: Dict[str, float] = {
            population: 0.0 for population in self.male_cns.t4t5_populations
        }

        for flyvis_type in config.subtypes:
            male_cns_type = SUBTYPE_MAP[flyvis_type]
            # [RIGHT_EYE, LEFT_EYE]: deviation from the grey-screen rest state.
            deviation = self.optic_lobe.type_activity(flyvis_type)
            for eye, side in ((RIGHT_EYE, "R"), (LEFT_EYE, "L")):
                value = float(deviation[eye])
                value = max(value, 0.0) if config.rectify else abs(value)
                rate = config.baseline_hz + config.gain_hz_per_unit * value
                rates[f"{male_cns_type}_{side}"] = float(
                    np.clip(rate, 0.0, config.max_hz)
                )

        self.male_cns.set_input(sensory.T4T5_DRIVE, rates)
        self._last_rates = rates
        return rates

    def last_rates(self) -> Dict[str, float]:
        """The rates staged by the most recent :meth:`transfer`."""
        return dict(self._last_rates)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"<OpticLobeToMaleCNS subtypes={list(self.config.subtypes)} "
            f"gain={self.config.gain_hz_per_unit} Hz/unit (INVENTED)>"
        )
