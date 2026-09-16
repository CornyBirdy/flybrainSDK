"""MaleCNS circuit: the whole fly central nervous system as a spiking model.

This module is the *only* place in the SDK that knows the MaleCNS loader
(:mod:`flybrain.malecns`) exists, exactly as
:mod:`flybrain.circuits.optic_lobe` is the only place that knows about flyvis.

What the underlying model is
----------------------------
**MaleCNS v1.0** (HHMI Janelia FlyEM / Google Research / Cambridge
Connectomics, CC-BY): a serial-section EM reconstruction of the central
nervous system of a male *Drosophila melanogaster*, covering brain, optic
lobes and ventral nerve cord. The **release** holds

    164,587 neurons
     25,563,197 connections
    124,025,046 synapses

and the **graph this circuit actually runs on** is a 96% subset of it:

    164,587 neurons
     24,539,704 connections
    120,793,200 synapses

Every neuron survives; 1,023,493 connections (4.0%) do not. They are dropped
because their presynaptic neuron is predicted dopaminergic, octopaminergic,
serotonergic or ``unclear``, and this SDK gives those neurons a sign of zero
rather than guessing one -- see ``NT_SIGN`` in :mod:`flybrain.malecns.dataset`
and assumption 4 below. So the release figures describe the data and the
subset figures describe the model, and any claim about the simulation should
quote the second set. (An earlier version of this docstring quoted the release
figures and called them "the whole traced connectome, not a subset". It is a
subset -- see ``VERIFICATION.md`` Finding 2.)

On top of that wiring runs a **leaky integrate-and-fire** model in the style
of Shiu et al., "A leaky integrate-and-fire computational model based on the
connectome of the entire adult *Drosophila* brain reveals insights into
sensorimotor processing", *Nature* 634, 210-219 (2024). A spike shifts the
downstream membrane potential in proportion to the number of synapses
between the two cells, with the sign set by the presynaptic neuron's
predicted neurotransmitter.

Measured versus assumed
-----------------------
The distinction matters more here than in ``optic_lobe``, because flyvis's
parameters were *fitted against neural recordings* while these were not.

MEASURED, from the release: the wiring; the synapse count of each connection;
the neurotransmitter prediction for each neuron; cell type and side.

ASSUMED, by this SDK:

* **Synapse count is proportional to synaptic strength.** The connectome does
  not measure strength.
* **Every synapse contributes the same 0.275 mV.** A free parameter of Shiu
  et al., chosen to make firing rates physiological. Not a measurement.
* **Glutamate is inhibitory** (GluCl-alpha; Liu & Wilson 2013), along with
  GABA and histamine; acetylcholine is excitatory; monoaminergic and
  unclassified neurons are given no output at all.
* **One set of membrane constants for all 164,587 neurons.**
* No gap junctions, no neuromodulation, no dendritic processing, zero basal
  firing rate.

What was verified
-----------------
Activating the sweet-sensing labellar GRNs (types LB3b, LB3c) at 150 Hz
drives MN9, the rostrum protractor, to 13.6 +/- 2.3 Hz against a silent
baseline; co-activating the bitter GRNs (LB1a-e) returns it to zero; and a
random rewiring of the connectome abolishes the effect across six seeds.
``tests/test_male_cns.py`` asserts all three, and an independent audit
reproduced all three on fresh seeds (``VERIFICATION.md`` Part D).

**How specific this is, stated at the strength the evidence supports.** Bitter
and water give exactly 0.00 Hz, so MN9 is not simply driven by any input. But
MN9 is *not* a sugar-specific read-out: 34 *random* gustatory neurons also
drive it, at a median of 3.0 Hz against sugar's 22.0 at the test suite's
protocol, so sugar wins by a measured factor rather than categorically; the
null is heavy-tailed, with 5 of 24 draws exceeding sugar. ``LB4b`` gets within
a factor of 2. And the effect is mostly ``LB3c`` (18.4 Hz driven alone), not
``LB3b`` (4.5 Hz). See ``NOTES.md`` §5 and ``VERIFICATION.md`` Finding 4.

See ``NOTES.md`` for the full log, including two results that do *not* look
right.

Cost
----
This circuit does **not** run at 30 fps and is not intended to. The reference
model integrates at 0.1 ms, so one 1/60 s host frame is 167 internal steps
over all 164,587 neurons. Measured cost, on a 4-core Xeon @ 2.80 GHz with no
GPU: **4.6-15.5 s of wall time per second of simulated time**, i.e.
**4.0-13.1 fps** at a 1/60 s frame and **2.0-6.6 fps** at 1/30 s, against
~30 ms/step for ``optic_lobe``. Cost rises with how much of the network is
firing. Resident set during a run is 0.64-0.72 GB; the high-water mark is the
one-off cache build, at 2.52 GB. :meth:`MaleCNSCircuit.step` accepts the same
``dt`` as any other circuit; it will simply take longer than real time to
return.

Building the cache with ``min_synapses=5`` drops the graph to 6.1 M edges and
a 49 MB cache but is **not** faster -- measured at 15.30 s per simulated
second under sugar drive against the full model's 15.03 s. The bottleneck is
the dense per-step passes over 164,587 neurons, not the sparse scatter, so
pruning edges cannot help. See ``NOTES.md`` §7.

Getting the data
----------------
Once, before first use::

    python -m flybrain.malecns download

566 MB from a public bucket, no account and no token. There is no synthetic
fallback: if the data is missing the circuit raises rather than inventing a
connectome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..io import motor, sensory
from ..ports import Port
from .base import Circuit, register_circuit

logger = logging.getLogger(__name__)

__all__ = [
    "MaleCNSCircuit",
    "MaleCNSConfig",
    "INPUT_POPULATIONS",
    "OUTPUT_NEURONS",
    "T4T5_SUBTYPES",
]


# --------------------------------------------------------------------------
# Cell-type constants
#
# MaleCNS v1.0 carries no taste-modality annotation - only anatomical type
# names - so every assignment below comes from the literature and is cited.
# The port docstrings in flybrain/io/sensory.py carry the full quotations.
# --------------------------------------------------------------------------

#: Sensory port name -> the MaleCNS ``type`` values it drives.
#:
#: LB1a-d match Gr33a-GAL4 (all bitter GRNs) and LB1e clusters with them;
#: LB3b and LB3c match Gr64f-GAL4 (sweet); LB3a matches ppk28-GAL4 (water);
#: LB3d matches Ir7c-/ppk23-GAL4 (high salt). Tastekin et al., bioRxiv
#: 2025.08.25.671814 / Cell (2026), consistent with the bitter / sugar-water
#: division in the FlyWire annotations of Schlegel et al., Nature 634,
#: 139-152 (2024).
INPUT_POPULATIONS: Dict[str, Tuple[str, ...]] = {
    sensory.SUGAR_GRN.name: ("LB3b", "LB3c"),
    sensory.BITTER_GRN.name: ("LB1a", "LB1b", "LB1c", "LB1d", "LB1e"),
    sensory.WATER_GRN.name: ("LB3a",),
    sensory.HIGH_SALT_GRN.name: ("LB3d",),
}

#: Motor port name -> the MaleCNS ``instance`` values it reads out.
#:
#: ``instance`` rather than ``type`` because it carries the side suffix, and
#: side is exactly what these ports distinguish.
OUTPUT_NEURONS: Dict[str, Tuple[str, ...]] = {
    motor.MN9_L.name: ("MN9_L",),
    motor.MN9_R.name: ("MN9_R",),
    motor.DNa02_L.name: ("DNa02_L",),
    motor.DNa02_R.name: ("DNa02_R",),
    motor.MDN_L.name: ("MDN_L",),
    motor.MDN_R.name: ("MDN_R",),
    motor.DNp09_L.name: ("DNp09_L",),
    motor.DNp09_R.name: ("DNp09_R",),
    motor.HSE_L.name: ("HSE_L",),
    motor.HSE_R.name: ("HSE_R",),
}

#: The eight T4/T5 subtypes, in the order T4T5_DRIVE expects them.
T4T5_SUBTYPES: Tuple[str, ...] = (
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
)


@dataclass
class MaleCNSConfig:
    """Tunable constants for :class:`MaleCNSCircuit`.

    Attributes:
        cache_root: Directory holding the built connectome cache. ``None``
            uses ``$FLYBRAIN_MALECNS_DIR`` or the per-user cache directory.
        min_synapses: Connections below this synapse count are excluded from
            the graph. 1, the default, keeps every traced connection and
            matches Shiu et al. Raising it gives a smaller and faster but
            explicitly different model, and the sanity checks have only been
            run at 1.
        seed: Seed for the Poisson drive. The model is stochastic; two runs
            with the same seed match, two with different seeds do not.
        readout_tau: Time constant, in seconds, of the exponential smoothing
            applied to output firing rates. Raw per-frame spike counts on a
            single neuron are far too noisy to read directly: at 60 fps a
            20 Hz neuron emits 0 or 1 spikes per frame. 0.1 s is a
            compromise between responsiveness and legibility, and is a
            READ-OUT choice only - it does not touch the dynamics.
        lif: Membrane and synapse constants. Defaults to Shiu et al. (2024).
        enable_visual_coupling: Whether to accept the T4T5_DRIVE input port.
            Off by default. Driving MaleCNS T4/T5 cells from another model's
            motion estimate is a HYPOTHESIS, not a measured correspondence -
            see :mod:`flybrain.circuits.coupling`.
    """

    cache_root: Optional[str] = None
    min_synapses: int = 1
    seed: int = 0
    readout_tau: float = 0.1
    lif: Optional[Any] = None
    enable_visual_coupling: bool = False


@register_circuit("male_cns")
class MaleCNSCircuit(Circuit):
    """The MaleCNS v1.0 connectome as a spiking circuit with typed ports.

    Args:
        config: Tunable constants. Defaults to :class:`MaleCNSConfig`.

    Raises:
        ImportError: If pandas, pyarrow or scipy are not installed.
        RuntimeError: If the connectome cache has not been built.
    """

    def __init__(self, config: Optional[MaleCNSConfig] = None) -> None:
        self.config = config or MaleCNSConfig()
        dataset, lif = _import_loader()

        cache_config = dataset.CacheConfig(
            **(
                {"root": self.config.cache_root}
                if self.config.cache_root is not None
                else {}
            ),
            min_synapses=self.config.min_synapses,
        )
        self.connectome = dataset.load_cache(cache_config)
        logger.info(
            "Loaded MaleCNS v1.0: %d neurons, %d connections",
            self.connectome.n_neurons,
            self.connectome.weights.nnz,
        )

        self._lif_params = self.config.lif or lif.LIFParams()
        self.network = lif.LIFNetwork(
            self.connectome.weights, params=self._lif_params, seed=self.config.seed
        )

        self._input_index = self._resolve_inputs()
        self._output_index = self._resolve_outputs()
        self._t4t5_index = self._resolve_t4t5()

        self._drive: Dict[str, np.ndarray] = {
            name: np.zeros(len(index), dtype=np.float64)
            for name, index in self._input_index.items()
        }
        self._t4t5_drive: Dict[str, float] = {}
        self._rate: Dict[str, float] = {}
        self._drive_dirty = True
        self.reset()

    # -- construction helpers ------------------------------------------------

    def _resolve_inputs(self) -> Dict[str, np.ndarray]:
        """Matrix indices of each driveable sensory population.

        Raises:
            RuntimeError: If a population is absent from the cache, which
                means the cache was built from a different release.
        """
        index: Dict[str, np.ndarray] = {}
        for port_name, types in INPUT_POPULATIONS.items():
            found = self.connectome.select(type=list(types))
            if found.size == 0:
                raise RuntimeError(
                    f"No neurons of type {list(types)} in the cached "
                    f"connectome, so input port {port_name!r} cannot be "
                    "wired. Rebuild the cache with "
                    "'python -m flybrain.malecns download --force'."
                )
            index[port_name] = found
        return index

    def _resolve_outputs(self) -> Dict[str, np.ndarray]:
        """Matrix indices of each read-out neuron.

        Raises:
            RuntimeError: If a read-out neuron is absent from the cache.
        """
        index: Dict[str, np.ndarray] = {}
        for port_name, instances in OUTPUT_NEURONS.items():
            found = self.connectome.select(instance=list(instances))
            if found.size == 0:
                raise RuntimeError(
                    f"No neuron with instance {list(instances)} in the cached "
                    f"connectome, so output port {port_name!r} cannot be "
                    "wired. Rebuild the cache with "
                    "'python -m flybrain.malecns download --force'."
                )
            index[port_name] = found
        return index

    def _resolve_t4t5(self) -> Dict[str, np.ndarray]:
        """Matrix indices of each T4/T5 population, per subtype AND per side.

        Keyed by ``instance`` (``"T4b_R"``) rather than by ``type``, because
        the optional visual coupling drives the two lobes separately.
        """
        return {
            f"{subtype}_{side}": self.connectome.select(
                instance=f"{subtype}_{side}"
            )
            for subtype in T4T5_SUBTYPES
            for side in ("L", "R")
        }

    # -- Circuit interface ---------------------------------------------------

    @property
    def inputs(self) -> Tuple[Port, ...]:
        ports = (
            sensory.SUGAR_GRN,
            sensory.BITTER_GRN,
            sensory.WATER_GRN,
            sensory.HIGH_SALT_GRN,
        )
        if self.config.enable_visual_coupling:
            ports += (sensory.T4T5_DRIVE,)
        return ports

    @property
    def outputs(self) -> Tuple[Port, ...]:
        return (
            motor.MN9_L,
            motor.MN9_R,
            motor.DNa02_L,
            motor.DNa02_R,
            motor.MDN_L,
            motor.MDN_R,
            motor.DNp09_L,
            motor.DNp09_R,
            motor.HSE_L,
            motor.HSE_R,
        )

    def input_shape(self, port: Port) -> Tuple[int, ...]:
        """Natural shape of a sensory port's value.

        Raises:
            KeyError: If the port is not an input of this circuit.
        """
        if port.name in self._drive:
            return ()
        if port.name == sensory.T4T5_DRIVE.name and self.config.enable_visual_coupling:
            return (len(T4T5_SUBTYPES),)
        raise KeyError(port)

    def reset(self) -> None:
        """Return every neuron to rest and clear all drive and read-outs."""
        self.network.reset()
        self.network.rng = np.random.default_rng(self.config.seed)
        for drive in self._drive.values():
            drive.fill(0.0)
        self._t4t5_drive = {}
        self._drive_dirty = True
        self._rate = {port.name: 0.0 for port in self.outputs}

    def set_input(self, port: Port, value: Any) -> None:
        """Set the drive on a sensory population, in Hz.

        Unlike the optic lobe's VISUAL_FIELD, this is a LEVEL: it persists
        across steps until changed, and zero is a legitimate value meaning
        "this population is silent". Nothing needs to be staged before
        :meth:`step`.

        Args:
            port: One of the circuit's input ports.
            value: Firing rate in Hz. A scalar drives the whole population
                equally; an array of the population's length drives each
                neuron separately. For T4T5_DRIVE, also accepts a mapping
                from subtype name to rate.

        Raises:
            KeyError: If the port is not an input of this circuit.
            ValueError: If the value is negative or the wrong length.
        """
        if port.name in self._drive:
            target = self._drive[port.name]
            rates = np.asarray(value, dtype=np.float64)
            if rates.ndim == 0:
                target.fill(float(rates))
            elif rates.shape == target.shape:
                target[:] = rates
            else:
                raise ValueError(
                    f"{port.name} expects a scalar rate or {target.size} "
                    f"rates, got shape {rates.shape}"
                )
            if (target < 0).any():
                raise ValueError(f"{port.name} rate must be >= 0 Hz, got {value!r}")
            self._drive_dirty = True
            return

        if port.name == sensory.T4T5_DRIVE.name:
            if not self.config.enable_visual_coupling:
                raise KeyError(
                    "T4T5_DRIVE is off by default because driving MaleCNS "
                    "T4/T5 cells from another model is a hypothesis, not a "
                    "measured correspondence. Enable it deliberately with "
                    "MaleCNSConfig(enable_visual_coupling=True) and read "
                    "flybrain/circuits/coupling.py first."
                )
            self._set_t4t5_drive(value)
            self._drive_dirty = True
            return

        raise KeyError(f"{self.name} has no input port {port.name!r}")

    def step(self, dt: float) -> None:
        """Integrate the LIF state for ``dt`` seconds and update the read-outs.

        ``dt`` is subdivided into internal steps of ``lif.dt`` (0.1 ms), so a
        60 fps host runs 167 internal steps per call. This is the honest cost
        of the reference model, and it is why this circuit runs at 2-13 fps
        rather than 30 - see the module docstring.

        Args:
            dt: Duration to integrate, in seconds.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if self._drive_dirty:
            self._apply_drive()
            self._drive_dirty = False

        spikes = self.network.step(dt)

        # Exponentially smooth the instantaneous rate. A single neuron emits
        # 0 or 1 spikes in a 1/60 s frame, so the unsmoothed value is
        # unusable; the smoothing is a read-out choice and does not feed back
        # into the dynamics.
        alpha = 1.0 - float(np.exp(-dt / self.config.readout_tau))
        for port_name, index in self._output_index.items():
            instantaneous = float(spikes[index].mean()) / dt
            self._rate[port_name] += alpha * (instantaneous - self._rate[port_name])

    def get_output(self, port: Port) -> float:
        """Smoothed firing rate of the port's neuron, in Hz.

        Raises:
            KeyError: If the port is not an output of this circuit.
        """
        try:
            return self._rate[port.name]
        except KeyError:
            raise KeyError(f"{self.name} has no output port {port.name!r}") from None

    # -- internals -----------------------------------------------------------

    def _set_t4t5_drive(self, value: Any) -> None:
        """Stage drive on the T4/T5 populations, by subtype or by side.

        Args:
            value: A scalar driving all eight subtypes on both sides; eight
                rates in the order of :data:`T4T5_SUBTYPES`, again both
                sides; or a mapping whose keys are either subtype names
                (``"T4b"``, both sides) or instance names (``"T4b_R"``, one
                side). A mapping omits nothing silently: unnamed populations
                are set to zero.

        Raises:
            KeyError: If a mapping key is neither a T4/T5 subtype nor one of
                its per-side instances.
            ValueError: If an array is the wrong length, or a rate is negative.
        """
        if isinstance(value, Mapping):
            per_population: Dict[str, float] = {}
            for key, rate in value.items():
                if key in T4T5_SUBTYPES:
                    per_population[f"{key}_L"] = float(rate)
                    per_population[f"{key}_R"] = float(rate)
                elif key in self._t4t5_index:
                    per_population[key] = float(rate)
                else:
                    raise KeyError(
                        f"{key!r} is neither a T4/T5 subtype nor one of its "
                        f"instances. Expected any of {list(T4T5_SUBTYPES)} or "
                        f"{sorted(self._t4t5_index)}."
                    )
        else:
            rates = np.asarray(value, dtype=np.float64)
            if rates.ndim == 0:
                rates = np.full(len(T4T5_SUBTYPES), float(rates))
            elif rates.shape != (len(T4T5_SUBTYPES),):
                raise ValueError(
                    f"T4T5_DRIVE expects a scalar, {len(T4T5_SUBTYPES)} rates "
                    f"in the order {list(T4T5_SUBTYPES)}, or a mapping; got "
                    f"shape {rates.shape}"
                )
            per_population = {
                f"{subtype}_{side}": float(rate)
                for subtype, rate in zip(T4T5_SUBTYPES, rates)
                for side in ("L", "R")
            }

        negative = {k: v for k, v in per_population.items() if v < 0}
        if negative:
            raise ValueError(f"T4T5_DRIVE rates must be >= 0 Hz, got {negative}")
        self._t4t5_drive = per_population

    def _apply_drive(self) -> None:
        """Push the staged rates into the LIF network's Poisson input."""
        index_parts = []
        rate_parts = []
        for port_name, index in self._input_index.items():
            rates = self._drive[port_name]
            active = rates > 0
            if active.any():
                index_parts.append(index[active])
                rate_parts.append(rates[active])

        for population, rate in self._t4t5_drive.items():
            index = self._t4t5_index[population]
            if rate > 0 and index.size:
                index_parts.append(index)
                rate_parts.append(np.full(index.size, rate))

        if index_parts:
            self.network.set_poisson_drive(
                np.concatenate(index_parts), np.concatenate(rate_parts)
            )
        else:
            self.network.set_poisson_drive(
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
            )

    # -- introspection -------------------------------------------------------

    def population(self, port: Port) -> np.ndarray:
        """Matrix indices of the neurons behind a port.

        Useful for reaching past the ports into the raw model, e.g. to read
        the rate of a neuron the SDK does not expose.

        Raises:
            KeyError: If the port belongs to neither direction.
        """
        if port.name in self._input_index:
            return self._input_index[port.name]
        if port.name in self._output_index:
            return self._output_index[port.name]
        if port.name == sensory.T4T5_DRIVE.name:
            return np.concatenate(list(self._t4t5_index.values()))
        raise KeyError(f"{self.name} has no port {port.name!r}")

    @property
    def t4t5_populations(self) -> Tuple[str, ...]:
        """Names of the per-side T4/T5 populations T4T5_DRIVE accepts."""
        return tuple(sorted(self._t4t5_index))

    @property
    def spike_counts(self) -> np.ndarray:
        """Spikes emitted by every neuron since the last reset (read-only)."""
        return self.network.spike_counts

    def mean_rates(self) -> np.ndarray:
        """Mean firing rate of every neuron since the last reset, in Hz.

        Unsmoothed and cumulative, unlike :meth:`get_output`. This is the
        quantity the published trial-based experiments report.
        """
        return self.network.rates()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"<MaleCNSCircuit neurons={self.connectome.n_neurons} "
            f"t={self.network.elapsed:.3f}s>"
        )


# --------------------------------------------------------------------------
# import helper - keep the heavy dependency lazy and the error message useful
# --------------------------------------------------------------------------


def _import_loader() -> Tuple[Any, Any]:
    """Import the MaleCNS loader, or explain what is missing.

    Raises:
        ImportError: If the scientific stack the loader needs is absent.
    """
    try:
        from ..malecns import dataset, lif
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "flybrain's male_cns circuit needs pandas, pyarrow and scipy. "
            "Install them with 'pip install flybrain[male_cns]', then build "
            "the connectome cache once with "
            "'python -m flybrain.malecns download'."
        ) from exc
    return dataset, lif
