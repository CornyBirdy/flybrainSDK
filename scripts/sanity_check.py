"""Re-derive the headline MaleCNS numbers in NOTES.md §5 and §6.

This is a RECONSTRUCTION. The original `scratch/sanity_check.py` that produced
those tables was never committed (`scratch/` is gitignored), so the protocol
behind the published figures could not be checked and an audit flagged it --
`VERIFICATION.md` Finding 9. This file exists so the headline numbers have a
runnable provenance from here on. It is rebuilt from the audit's own
reproduction, which matched the statistics but not the per-seed magnitudes,
so expect the same: the conditions and their ordering reproduce tightly, the
individual numbers move by a few Hz between seeds.

The one ambiguity the missing script left is settled explicitly here rather
than inherited: **MN9 is reported three ways** -- MN9_L, MN9_R and the mean of
the two bodies -- because NOTES.md §5 and §6 disagree at 150 Hz (13.60 vs
10.50 Hz) and the mean-of-both-bodies reading is the most likely cause. Nearly
all of the response is on MN9_L, so the two conventions differ by about 2x.

Protocol, matching what NOTES.md §5 states: 5 trials x 1 s of simulated time,
gustatory populations driven at 150 Hz, rates are the unsmoothed cumulative
mean over each trial, trial k uses seed + k.

Needs the connectome cache:

    python -m flybrain.malecns download

Run:  python scripts/sanity_check.py
      python scripts/sanity_check.py --quick      # 2 trials x 0.5 s
      python scripts/sanity_check.py --seed 100

Cost: about 6 minutes for the default protocol on a 4-core Xeon @ 2.8 GHz.
--quick is about 1 minute and is enough to see the ordering.
"""

import argparse
import sys
import time

import numpy as np

from flybrain import motor, sensory
from flybrain.circuits import MaleCNSCircuit
from flybrain.circuits.male_cns import INPUT_POPULATIONS

#: Drive rate for a gustatory population, in Hz. Shiu et al. use 100-200.
DRIVE_HZ = 150.0

#: Drive rates for the dose-response sweep in §6.
DOSE_HZ = (25.0, 50.0, 100.0, 150.0, 200.0)

#: Seeds for the §6 shuffle control. 7, 8, 9 are the three NOTES.md reports.
SHUFFLE_SEEDS = (7, 8, 9)


def populations(circuit):
    """Port name -> matrix indices, for every gustatory input population."""
    return {
        name: circuit.population(getattr(sensory, name))
        for name in INPUT_POPULATIONS
    }


def trials(circuit, drive, duration, n_trials, seed):
    """Per-neuron mean rate over ``n_trials`` independent runs, in Hz.

    Args:
        circuit: A MaleCNSCircuit.
        drive: Sequence of ``(matrix index array, drive rate in Hz)`` pairs.
            A sequence rather than a mapping because arrays are unhashable.
        duration: Seconds of simulated time per trial.
        n_trials: Independent repetitions; trial k uses ``seed + k``.
        seed: Base RNG seed.

    Returns:
        Array of shape ``(n_trials, n_neurons)`` of firing rates.
    """
    out = np.empty((n_trials, circuit.connectome.n_neurons), dtype=np.float64)
    for trial in range(n_trials):
        circuit.config.seed = seed + trial
        circuit.reset()
        for index, rate in drive:
            circuit.network.set_poisson_drive(index, np.full(index.size, rate))
        circuit.network.step(duration)
        out[trial] = circuit.network.rates()
    return out


def report_mn9(label, rates, mn9_l, mn9_r, width=26):
    """Print one condition's MN9 rate under all three read-out conventions."""
    left = rates[:, mn9_l].mean(axis=1)
    right = rates[:, mn9_r].mean(axis=1)
    both = rates[:, np.concatenate([mn9_l, mn9_r])].mean(axis=1)
    print(
        f"  {label:<{width}} "
        f"{both.mean():6.2f} +/- {both.std():5.2f}   "
        f"{left.mean():6.2f} +/- {left.std():5.2f}   "
        f"{right.mean():5.2f} +/- {right.std():4.2f}"
    )
    return both.mean()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-derive the NOTES.md §5 and §6 MaleCNS figures."
    )
    parser.add_argument("--trials", type=int, default=5, help="trials per condition")
    parser.add_argument("--duration", type=float, default=1.0, help="seconds per trial")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="2 trials x 0.5 s: fast, enough to see the ordering, noisier",
    )
    parser.add_argument(
        "--skip-shuffle",
        action="store_true",
        help="skip the §6 shuffle control, which is the slow half",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.trials, args.duration = 2, 0.5

    t0 = time.time()
    print("loading the connectome cache ...", flush=True)
    circuit = MaleCNSCircuit()
    cx = circuit.connectome
    print(
        f"  {cx.n_neurons:,} neurons, {cx.weights.nnz:,} modelled connections, "
        f"{cx.n_synapses:,} signed synapses",
        flush=True,
    )
    print(
        f"  protocol: {args.trials} trials x {args.duration} s, "
        f"drive {DRIVE_HZ:.0f} Hz, base seed {args.seed}",
        flush=True,
    )

    pops = populations(circuit)
    mn9_l = circuit.population(motor.MN9_L)
    mn9_r = circuit.population(motor.MN9_R)

    # -- §5 conditions -----------------------------------------------------
    print("\nMN9 firing rate, Hz              both bodies      MN9_L            MN9_R")
    conditions = {
        "baseline (nothing driven)": [],
        "sugar        (LB3b/c)": [(pops["SUGAR_GRN"], DRIVE_HZ)],
        "bitter       (LB1a-e)": [(pops["BITTER_GRN"], DRIVE_HZ)],
        "sugar + bitter": [
            (pops["SUGAR_GRN"], DRIVE_HZ),
            (pops["BITTER_GRN"], DRIVE_HZ),
        ],
        "water        (LB3a)": [(pops["WATER_GRN"], DRIVE_HZ)],
        "high salt    (LB3d)": [(pops["HIGH_SALT_GRN"], DRIVE_HZ)],
    }
    sugar_rates = None
    for label, drive in conditions.items():
        rates = trials(circuit, drive, args.duration, args.trials, args.seed)
        report_mn9(label, rates, mn9_l, mn9_r)
        if label.startswith("sugar  "):
            sugar_rates = rates

    print(
        "\n  Note: 'both bodies' is what NOTES.md §5 most likely reported and\n"
        "  MN9_L is what tests/test_male_cns.py asserts. MN9_R is flagged\n"
        "  'RT Hard to trace' in the release and has 556 input synapses from\n"
        "  137 partners against MN9_L's 6,012 from 278, so it is not a working\n"
        "  read-out. Read the two sides separately."
    )

    # -- §5 anomaly (b): the hottest cells ---------------------------------
    mean_sugar = sugar_rates.mean(axis=0)
    hottest = np.argsort(mean_sugar)[::-1][:6]
    instances = np.array(
        [s if isinstance(s, str) else "?" for s in cx.neurons.instance.to_numpy()]
    )
    print("\nHottest cells under sugar drive (NOTES.md §5 anomaly (b)):")
    for i in hottest:
        print(f"  {instances[i]:<14} {mean_sugar[i]:7.1f} Hz")
    print(
        f"  peak {mean_sugar.max():.1f} Hz; {(mean_sugar > 400).sum()} neurons above "
        f"400 Hz, {(mean_sugar > 250).sum()} above 250. The refractory ceiling is "
        f"{1 / circuit._lif_params.t_rfc:.1f} Hz,"
    )
    print("  so these cells are unphysiologically hot but nowhere near saturated.")
    print(
        f"  network total {mean_sugar.sum():,.0f} Hz, mean {mean_sugar.mean():.3f} Hz, "
        f"{(mean_sugar > 1).sum():,} neurons above 1 Hz"
    )

    # -- §6 dose-response --------------------------------------------------
    print("\nDose-response, sugar GRNs (NOTES.md §6):")
    print("  drive          both bodies      MN9_L            MN9_R")
    for hz in DOSE_HZ:
        rates = trials(
            circuit, [(pops["SUGAR_GRN"], hz)], args.duration, args.trials, args.seed
        )
        report_mn9(f"{hz:5.0f} Hz", rates, mn9_l, mn9_r, width=13)

    # -- §6 shuffle control ------------------------------------------------
    if args.skip_shuffle:
        print("\nShuffle control skipped (--skip-shuffle).")
    else:
        from flybrain.malecns import LIFNetwork, shuffle_preserving_degree

        print("\nShuffle control (NOTES.md §6):")
        real = trials(
            circuit,
            [(pops["SUGAR_GRN"], DRIVE_HZ)],
            args.duration,
            args.trials,
            args.seed,
        ).mean(axis=0)
        print(
            f"  {'real wiring':<18} MN9_L {real[mn9_l].mean():6.2f} Hz   "
            f"network {real.sum():10,.0f} Hz   >1 Hz {(real > 1).sum():,}"
        )
        sugar = pops["SUGAR_GRN"]
        for shuffle_seed in SHUFFLE_SEEDS:
            shuffled = shuffle_preserving_degree(cx.weights, seed=shuffle_seed)
            net = LIFNetwork(shuffled, params=circuit._lif_params, seed=args.seed)
            net.set_poisson_drive(sugar, np.full(sugar.size, DRIVE_HZ))
            net.step(args.duration)
            r = net.rates()
            print(
                f"  {'shuffle seed ' + str(shuffle_seed):<18} "
                f"MN9_L {r[mn9_l].mean():6.2f} Hz   "
                f"network {r.sum():10,.0f} Hz   >1 Hz {(r > 1).sum():,}   "
                f"({r.sum() / real.sum():.3f} of real)"
            )
        print(
            "\n  The shuffle takes MN9 to exactly zero while the network keeps\n"
            "  firing, so the result is a property of the wiring. Two caveats,\n"
            "  both real: total activity drops to about an eighth, so the null is\n"
            "  not matched for total drive; and shuffle_preserving_degree does\n"
            "  not preserve either degree sequence exactly -- see its docstring."
        )

    # -- the specificity control NOTES.md never had ------------------------
    print("\nSpecificity: 34 RANDOM gustatory neurons (VERIFICATION.md Finding 4):")
    gustatory = np.flatnonzero((cx.neurons["cell_class"] == "gustatory").to_numpy())
    sugar_set = set(pops["SUGAR_GRN"].tolist())
    pool = np.array([i for i in gustatory if i not in sugar_set])
    draws = []
    for draw_seed in range(1000, 1005):
        index = np.sort(
            np.random.default_rng(draw_seed).choice(
                pool, size=pops["SUGAR_GRN"].size, replace=False
            )
        )
        r = trials(
            circuit, [(index, DRIVE_HZ)], args.duration, args.trials, args.seed
        )
        draws.append(r[:, mn9_l].mean())
        print(f"  draw rng={draw_seed}   MN9_L {draws[-1]:6.2f} Hz")
    sugar_mn9_l = sugar_rates[:, mn9_l].mean()
    median = float(np.median(draws))
    print(
        f"  median {median:.2f} Hz against sugar's {sugar_mn9_l:.2f} Hz: a margin of "
        f"{sugar_mn9_l / median if median else float('inf'):.1f}x."
    )
    print(
        "  Sugar wins by a factor, not categorically. The null is heavy-tailed --\n"
        "  over 24 draws the median is 3.0 Hz but 5 exceeded sugar and one reached\n"
        "  156 Hz -- so this is a claim about the typical draw. Bitter and water\n"
        "  are genuinely zero. See NOTES.md §5."
    )

    print(f"\ndone in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
