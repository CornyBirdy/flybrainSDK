"""Leaky integrate-and-fire dynamics over the MaleCNS connectome.

Together with :mod:`flybrain.malecns.dataset` this is the MaleCNS "loader"
that :mod:`flybrain.circuits.male_cns` wraps. Nothing else in the SDK imports
it. It depends on NumPy alone.

The model
---------
This reproduces the model of Shiu et al., "A leaky integrate-and-fire
computational model based on the connectome of the entire adult *Drosophila*
brain reveals insights into sensorimotor processing", *Nature* 634, 210-219
(2024), as published in https://github.com/philshiu/Drosophila_brain_model
(``model.py``, ``default_params``). Every constant below is theirs; the
citations are the ones given in their source.

    dv/dt = (v_0 - v + g) / t_mbr     (frozen while refractory)
    dg/dt = -g / tau                  (frozen while refractory)
    spike when v > v_th, then v <- v_rst, g <- 0, refractory for t_rfc
    a presynaptic spike adds w to the postsynaptic g, after a delay t_dly

with ``w = w_syn * sign(transmitter) * n_synapses``.

What is measured and what is assumed
------------------------------------
MEASURED: ``n_synapses`` for every connection, and the transmitter prediction
that sets its sign.

ASSUMED, and these are the load-bearing assumptions:
  * every synapse of every neuron has the same weight ``w_syn`` = 0.275 mV.
    This is a free parameter fitted by Shiu et al. so that the model's
    firing rates are physiological; it is NOT a measurement. Real synaptic
    strengths vary by orders of magnitude and the connectome does not
    contain them.
  * one set of passive membrane constants for all 164,587 neurons
    (Kakaria & de Bivort 2017), regardless of cell type or size.
  * no gap junctions (the EM segmentation does not resolve them), no
    neuromodulation, no dendritic processing, zero basal firing rate.

Deviation from the reference implementation
-------------------------------------------
Shiu et al. run this in Brian 2. This is a direct NumPy/SciPy integrator so
that the SDK can advance it one host frame at a time without a Brian network
object, and so that it has no compiler dependency. It uses the exact
(exponential-Euler) solution of the linear subsystem, which is what Brian's
``method='linear'`` does, at the same 0.1 ms internal step. Two known
differences from the Brian model are noted at :meth:`LIFNetwork.step`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LIFParams:
    """Membrane and synapse constants. Defaults are Shiu et al. (2024).

    Attributes:
        v_0: Resting potential, volts. Kakaria & de Bivort 2017.
        v_rst: Reset potential after a spike, volts.
        v_th: Spike threshold, volts.
        t_mbr: Membrane time constant, seconds (2 nF x 10 MOhm).
        tau: Synaptic conductance time constant, seconds.
            Juergensen et al. 2021.
        t_rfc: Absolute refractory period, seconds. Lazar et al. 2021.
        t_dly: Axonal/synaptic delay, seconds. Paul et al. 2015.
        w_syn: Voltage step contributed by ONE synapse, volts. Free
            parameter of Shiu et al.; not measured.
        f_poi: Scale factor on ``w_syn`` for the injected Poisson drive. 250
            makes every Poisson event suprathreshold, i.e. a stimulated
            neuron fires as a Poisson process at the requested rate.
        dt: Internal integration step, seconds. Brian 2's default.
    """

    v_0: float = -52e-3
    v_rst: float = -52e-3
    v_th: float = -45e-3
    t_mbr: float = 20e-3
    tau: float = 5e-3
    t_rfc: float = 2.2e-3
    t_dly: float = 1.8e-3
    w_syn: float = 0.275e-3
    f_poi: float = 250.0
    dt: float = 0.1e-3


class LIFNetwork:
    """LIF dynamics over a signed connectome.

    Args:
        weights: Signed synapse counts as a scipy CSR matrix, rows
            presynaptic. Element ``[i, j]`` is added to neuron ``j``'s
            conductance (times ``w_syn``) when ``i`` spikes.
        params: Model constants.
        seed: Seed for the Poisson drive.

    Raises:
        ValueError: If ``weights`` is not square.
    """

    def __init__(
        self,
        weights,
        params: Optional[LIFParams] = None,
        seed: int = 0,
    ) -> None:
        if weights.shape[0] != weights.shape[1]:
            raise ValueError(f"weights must be square, got {weights.shape}")
        self.params = params or LIFParams()
        self.n = weights.shape[0]

        weights = weights.tocsr()
        self._indptr = weights.indptr.astype(np.int64)
        self._indices = weights.indices.astype(np.int32)
        # Pre-multiply by w_syn once, so the hot loop is a pure gather-and-add.
        self._data = (weights.data.astype(np.float32) * self.params.w_syn).astype(
            np.float32
        )

        # Exact solution of the linear subsystem over one dt. See module
        # docstring: g decays with tau, v relaxes to v_0 with t_mbr while
        # being driven by g.
        p = self.params
        self._decay_g = float(np.exp(-p.dt / p.tau))
        self._decay_v = float(np.exp(-p.dt / p.t_mbr))
        # v-response to a conductance that is itself decaying.
        self._g_to_v = float(
            (1.0 / p.t_mbr) / (1.0 / p.t_mbr - 1.0 / p.tau)
        ) * (self._decay_g - self._decay_v)

        self._delay_steps = max(1, int(round(p.t_dly / p.dt)))
        self._refractory_steps = int(round(p.t_rfc / p.dt))

        self.rng = np.random.default_rng(seed)
        self._poisson_index = np.empty(0, dtype=np.int64)
        self._poisson_rate = np.empty(0, dtype=np.float64)
        self._silenced = np.zeros(self.n, dtype=bool)

        self.v = np.empty(self.n, dtype=np.float32)
        self.g = np.empty(self.n, dtype=np.float32)
        self._scratch = np.empty(self.n, dtype=np.float32)
        self._refractory = np.empty(self.n, dtype=np.int32)
        self._pending = np.empty((self._delay_steps, self.n), dtype=np.float32)
        self._cursor = 0
        self.spike_counts = np.empty(self.n, dtype=np.int64)
        self.elapsed = 0.0
        self.reset()

    # -- state ---------------------------------------------------------------

    def reset(self) -> None:
        """Return every neuron to rest and clear the spike counters."""
        self.v.fill(self.params.v_0)
        self.g.fill(0.0)
        self._refractory.fill(0)
        self._pending.fill(0.0)
        self._cursor = 0
        self.spike_counts.fill(0)
        self.elapsed = 0.0

    def set_poisson_drive(self, index: Sequence[int], rate_hz: Sequence[float]) -> None:
        """Drive neurons with independent Poisson input, as optogenetics would.

        Reproduces Brian's ``PoissonInput(target_var='v')`` in the reference
        implementation: each event adds ``w_syn * f_poi`` (68.75 mV) to the
        membrane potential, which is far suprathreshold, and the refractory
        period of a driven neuron is suppressed. The net effect is a Poisson
        spike train at ``rate_hz``.

        Args:
            index: Matrix indices of the neurons to drive.
            rate_hz: Rate for each of them, in Hz. Scalar-broadcastable.

        Raises:
            ValueError: If the two arguments have different lengths.
        """
        index = np.asarray(index, dtype=np.int64)
        rate = np.broadcast_to(np.asarray(rate_hz, dtype=np.float64), index.shape)
        if index.ndim != 1:
            raise ValueError("index must be one-dimensional")
        self._poisson_index = index
        self._poisson_rate = np.array(rate, dtype=np.float64)

    def silence(self, index: Sequence[int]) -> None:
        """Silence neurons: they may still spike, but their output is muted.

        This is the ``neu_slnc`` manipulation of the reference implementation
        (all outgoing synapse weights set to zero), implemented as a spike
        mask so it can be toggled without rebuilding the matrix.
        """
        self._silenced.fill(False)
        self._silenced[np.asarray(index, dtype=np.int64)] = True

    # -- integration ---------------------------------------------------------

    def step(self, dt: float) -> np.ndarray:
        """Advance the network by ``dt`` seconds.

        ``dt`` is subdivided into internal steps of ``params.dt``; a host
        calling at 60 fps therefore runs 167 internal steps per call.

        Two deviations from the Brian reference, both in the reference's own
        grey areas: spikes are emitted at the end of an internal step rather
        than mid-step, and a neuron driven by Poisson input can also be
        pushed over threshold by its synaptic input (in Brian the Poisson
        term is added to ``v`` identically, so this matches; what differs is
        only that we apply all events of a step at once).

        Args:
            dt: Duration to integrate, in seconds.

        Returns:
            Number of spikes each neuron emitted during this call.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        p = self.params
        n_sub = max(1, int(round(dt / p.dt)))
        before = self.spike_counts.copy()

        poisson_kick = np.float32(p.w_syn * p.f_poi)
        has_poisson = self._poisson_index.size > 0
        if has_poisson:
            lam = self._poisson_rate * p.dt

        v, g, tmp = self.v, self.g, self._scratch
        for _ in range(n_sub):
            # Refractory neurons hold both v and g frozen, as in the Brian
            # model where 'unless refractory' gates both ODEs. They are a
            # small minority, so save-update-restore beats a masked write.
            frozen = np.flatnonzero(self._refractory > 0)
            v_held, g_held = v[frozen], g[frozen]

            # Exact linear update, in place and allocation-free:
            #   v <- v_0 + (v - v_0) * decay_v + g * g_to_v
            #   g <- g * decay_g
            np.multiply(g, self._g_to_v, out=tmp)
            np.subtract(v, p.v_0, out=v)
            np.multiply(v, self._decay_v, out=v)
            np.add(v, tmp, out=v)
            np.add(v, p.v_0, out=v)
            np.multiply(g, self._decay_g, out=g)

            v[frozen] = v_held
            g[frozen] = g_held

            # Synaptic input released delay_steps ago lands now. It reaches
            # refractory neurons too: in Brian, 'unless refractory' gates the
            # ODE but not the on_pre statement.
            row = self._pending[self._cursor]
            g += row
            row.fill(0.0)

            if has_poisson:
                events = self.rng.poisson(lam)
                fired = events > 0
                if fired.any():
                    idx = self._poisson_index[fired]
                    v[idx] += poisson_kick * events[fired].astype(np.float32)
                    self._refractory[idx] = 0

            self._refractory -= 1

            # A refractory neuron sits at v_rst, which is below threshold, and
            # its v is frozen, so the threshold test needs no refractory mask.
            # The Poisson branch above clears the refractory flag before its
            # kick, so driven neurons are not excluded either.
            spiked = np.flatnonzero(v > p.v_th)
            if spiked.size:
                v[spiked] = p.v_rst
                g[spiked] = 0.0
                self._refractory[spiked] = self._refractory_steps
                self.spike_counts[spiked] += 1
                emitting = spiked[~self._silenced[spiked]]
                if emitting.size:
                    self._deliver(emitting)

            self._cursor = (self._cursor + 1) % self._delay_steps
            self.elapsed += p.dt

        return self.spike_counts - before

    def _deliver(self, spiked: np.ndarray) -> None:
        """Add the outgoing weights of ``spiked`` into the delay buffer."""
        starts = self._indptr[spiked]
        counts = self._indptr[spiked + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        # Flatten the CSR row slices of the spiking neurons into one gather.
        offsets = np.repeat(starts - np.cumsum(counts) + counts, counts)
        positions = offsets + np.arange(total, dtype=np.int64)
        target = self._pending[(self._cursor + self._delay_steps - 1) % self._delay_steps]
        np.add.at(target, self._indices[positions], self._data[positions])

    # -- read-out ------------------------------------------------------------

    def rates(self, index: Optional[Sequence[int]] = None) -> np.ndarray:
        """Mean firing rate since :meth:`reset`, in Hz."""
        if self.elapsed <= 0:
            return np.zeros(self.n if index is None else len(index), dtype=np.float64)
        counts = self.spike_counts if index is None else self.spike_counts[index]
        return counts / self.elapsed

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"<LIFNetwork n={self.n} edges={self._data.size} "
            f"t={self.elapsed * 1e3:.1f}ms>"
        )


def run_trials(
    weights,
    drive: Dict[int, float],
    duration: float = 1.0,
    n_trials: int = 5,
    params: Optional[LIFParams] = None,
    silenced: Sequence[int] = (),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run repeated activation trials and return per-neuron firing rates.

    Args:
        weights: Signed connectome, CSR.
        drive: Matrix index -> Poisson rate in Hz.
        duration: Trial length, seconds.
        n_trials: Independent repetitions.
        params: Model constants.
        silenced: Matrix indices whose output is muted.
        seed: Base RNG seed; trial ``k`` uses ``seed + k``.

    Returns:
        ``(mean_rate, std_rate)``, each of shape ``(n_neurons,)``, in Hz.
    """
    rates = np.empty((n_trials, weights.shape[0]), dtype=np.float64)
    net = LIFNetwork(weights, params=params, seed=seed)
    index = np.fromiter(drive.keys(), dtype=np.int64, count=len(drive))
    rate_hz = np.fromiter(drive.values(), dtype=np.float64, count=len(drive))
    for trial in range(n_trials):
        net.rng = np.random.default_rng(seed + trial)
        net.reset()
        net.set_poisson_drive(index, rate_hz)
        if len(silenced):
            net.silence(silenced)
        net.step(duration)
        rates[trial] = net.rates()
    return rates.mean(0), rates.std(0)
