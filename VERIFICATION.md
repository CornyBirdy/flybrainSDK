# VERIFICATION — adversarial audit of the `male_cns` work

An independent re-derivation of the claims in `NOTES.md`, run against the real
MaleCNS v1.0 release on a fresh environment by a session that did not write the
code.

Status: Parts A–E complete. Parts F–G in progress (marked `[PENDING]`).

---

## 0. READ THIS FIRST — the machine is not the machine in the brief

The brief specifies Windows 11, Ryzen 5 3600 (6c/12t), 32 GB DDR4, RTX 5070 Ti.
**This audit did not run on that machine.** It ran in a remote Linux container:

```
Linux 6.18.44-fc-v33 x86_64                   (not Windows 11)
Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores (not Ryzen 5 3600, 6c/12t)
15 GB RAM                                     (not 32 GB)
no GPU                                        (not an RTX 5070 Ti)
```

This is the *same class of machine the audited session used* (`NOTES.md` §12:
"a 4 vCPU / 15 GB Linux container"). Consequences:

* Every Windows-specific question is **unanswered**. See §4.
* Part F is a re-measurement of the original platform, not a port to the target.
* "Confirm peak RAM stays sane on 32 GB" was tested against **15 GB** — a
  strictly harder test, which passed.

---

## 1. Verdict

`[PENDING — written last]`

---

## 2. Confirmed

### Part A — the data

All three tables downloaded from the public bucket, tokenless.

| file | bytes | SHA256 |
|---|---:|---|
| `body-annotations-…-minconf-0.5.feather` | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| `body-neurotransmitters-…feather` | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| `connectome-weights-…-traced-only.feather` | 508,025,642 | `9b3beab17bad5f618be3f2c02d3139a8d07b822565919c013f1e5506d93e604b` |

565,791,790 B = 565.8 MB total. `NOTES.md` §1 says 14.5 / 43.3 / 508 MB, "566 MB".
Exact match. (`NOTES.md` records no hashes; the above are a new baseline.)

**Schemas** were read straight from the files. Every column named in `NOTES.md`
§1 exists with the stated type. Nothing missing, nothing renamed.

**No strength column — verified across the whole release, not just the three
files used.** I fetched the Arrow schema header of all **11** files by HTTP range
request (~31 GB of data, ~1 MB read). No column matching
`strength|conductance|gmax|efficacy|amplitude|norm` exists anywhere. The nearest
is `synweight` in `body-stats`, which is a per-body total synapse count, not a
synaptic strength. **Assumption 1's premise holds.**

**`weight` is a synapse count — confirmed against an independent table.** Cross-
checked the traced-only edge list against `body-stats` (778 MB, downloaded for
this purpose; the audited session never downloaded it):

```
sum(weight) into a body  <=  body-stats `post`   for 164,587 / 164,587 bodies (100%)
correlation                                       0.998809
totals: 124,025,046 (traced-only)  vs  130,412,255 (all partners) = 95.1%
```

The inequality holds for every body and the shortfall is exactly what
"traced-only is a subset of all partners" predicts. `weight` is a synapse count.

**Graph size, derived from the raw feather before reading any claim:**

| | derived | `NOTES.md` |
|---|---:|---:|
| neurons | **164,587** | 164,587 |
| connections | **25,563,197** | 25,563,197 |
| synapses | **124,025,046** | 124,025,046 |

`weight` is int64, min 1, max 2591, no nulls; 25,563,197 unique `(pre,post)`
pairs with zero duplicates; 101 self-edges.

**Neurotransmitter breakdown, re-derived independently of the SDK's own join:**

| transmitter | mine | `NOTES.md` |
|---|---:|---:|
| acetylcholine | 103,701 | 103,701 |
| glutamate | 29,296 | 29,296 |
| GABA | 22,055 | 22,055 |
| histamine | 5,904 | 5,904 |
| unclear | 3,090 | 3,090 |
| dopamine | 392 | 392 |
| octopamine | 101 | 101 |
| serotonin | 48 | 48 |

Exact, all eight. NT table has 1,835,518 rows as claimed; 73 graph bodies are
absent from it and fall back to `unclear`. Ground-truth labels: **83,495** within
the graph (`NOTES.md` says "~85 k" — fair).

Cache build: **24.1 s, peak RSS 2.52 GB**, output 196,977,241 B = **197 MB** as
claimed. 1,023,493 connections dropped as unsigned — matching §2 assumption 4
exactly — leaving **24,539,704 connections / 120,793,200 synapses**.

### Part B — the neuron identifications

**The body IDs are NOT hardcoded.** `INPUT_POPULATIONS` and `OUTPUT_NEURONS`
(`flybrain/circuits/male_cns.py`) hold *type* and *instance* strings, resolved
against the annotation table at load time via `Connectome.select()`. The only
numeric body IDs in the source are in a docstring (`flybrain/io/motor.py:132,141`)
and in a test assertion. This is the right design, and I re-derived the mapping
myself rather than trusting it.

**MN9** — derived from the annotation table by `type == "MN9"`:

```
 bodyId type instance subclass superclass somaSide      statusLabel
  10331  MN9    MN9_L       pm   cb_motor        L   Roughly traced
  16949  MN9    MN9_R       pm   cb_motor        R  RT Hard to trace
```

Exactly two bodies, `10331`/`16949`, type `MN9`, superclass `cb_motor`. Confirmed.
The `statusLabel` asymmetry §3 mentions is real.

**GRN populations** present with the claimed counts: LB3b 11 + LB3c 23 = **34**
sugar; LB1a–e 11+6+16+5+19 = **57** bitter; LB3a **17** water; LB3d **26** salt.
All are `class=gustatory`, `subclass=labellar bristle`, `superclass=cb_sensory`.
MaleCNS annotates **1,428** gustatory neurons, as §3 says, and a grep of every
string column for `sugar|bitter|Gr64|Gr66|Gr33|ppk28|ppk23|Ir7c|sweet|water|salt`
returns **nothing** — §3's claim that the release carries no modality label holds.

**LB1b transmitters** — all **6** LB1b neurons have `consensus_nt = unclear`
(confidences 0.486–0.684), hence `nt_sign = 0` and no modelled output. Confirmed;
the effective bitter set is 51, not 57.

**AN13B002 is the strongest output target of LB3b** — computed, not taken on
faith. Per postsynaptic body: `30088 AN13B002_L` with 238 synapses, rank 1. Per
postsynaptic type: `AN13B002` 427 synapses, rank 1. Confirmed.

**MN9's left/right asymmetry has a concrete cause** I verified independently:
MN9_L has **6,012** input synapses from 278 partners; MN9_R has **556** from 137.
An 11× reconstruction deficit, consistent with `RT Hard to trace`.

**Hop distance** sugar → MN9: 0 direct edges, MN9 first reached at hop 2.
Confirmed. One-hop count is **465** excluding the source cells, **498** including
them; §5 says 499 (they counted the 34 sources, 33 of which are themselves
one-hop targets). A definitional difference, not an error.

### Part C — the model

**All nine shared LIF constants are identical** to `philshiu/Drosophila_brain_model`
`default_params`, parsed programmatically from their `model.py` after cloning:

| | SDK | philshiu |
|---|---|---|
| `v_0`, `v_rst` | −0.052 V | −0.052 V |
| `v_th` | −0.045 V | −0.045 V |
| `t_mbr` | 0.02 s | 0.02 s |
| `tau` | 0.005 s | 0.005 s |
| `t_rfc` | 0.0022 s | 0.0022 s |
| `t_dly` | 0.0018 s | 0.0018 s |
| `w_syn` | 0.000275 V | 0.000275 V |
| `f_poi` | 250 | 250 |

**Zero undocumented deviations.** The SDK's `dt = 0.1 ms` has no counterpart in
`default_params` because the reference never overrides `defaultclock.dt` — Brian 2's
default is 0.1 ms, so this matches too. Mechanisms also match: `on_pre='g += w'`
with `delay=t_dly`; reset `v=v_rst; g=0`; `(unless refractory)` on *both* ODEs;
`PoissonInput(target_var='v', weight=w_syn*f_poi)` with the target's refractory
set to zero; edges signed as `Excitatory × Connectivity × w_syn`. The SDK's only
listed protocol difference is `n_trials=5` vs the reference's `n_run=30`, which
`NOTES.md` §5 discloses ("5 trials × 1 s").

**The integrator does what it says** — checked on hand-built tiny networks:

* Poisson drive tracks the requested rate: 10→9.72, 25→24.48, 50→49.64,
  100→99.76, 150→147.36, 200→196.96, 300→291.28 Hz (5 seeds each). `NOTES.md`
  §4's "150 Hz asked, 143 Hz fired" is within this spread.
* Undriven neurons in the same network fire **exactly zero**.
* A single +200-synapse excitatory edge takes the target 0 → 98.2 Hz; adding a
  coincident −200-synapse inhibitory source drops it to 25.6 Hz.
* Edge sign flows through: `+1 → 95.0 Hz`, `−1 → 0.00 Hz`, `0 → 0.00 Hz`.
* The exponential-Euler coefficients match the analytic solution exactly, and one
  0.1 ms step agrees with a 20,000×-oversampled forward-Euler reference to
  **4.3e-12 V** in v and **2.0e-11 V** in g. §4 claims <1e-9 V; mine is tighter.

**Glutamate is negative and it is load-bearing.** `NT_SIGN["glutamate"] == -1` as
shipped. Rebuilding the cache with `+1` flips 29,296 neurons and changes
**4,797,639 of 24,539,704 edges (19.6%)**; inhibitory edges fall 9,802,165 →
5,004,526. The effect on dynamics is drastic:

| | net activity | neurons >1 Hz | MN9 | max rate |
|---|---:|---:|---:|---:|
| glutamate −1 (shipped) | 83,646 Hz | 2,381 | 14.17 Hz | 330 Hz |
| glutamate +1 (flipped) | **9,556,156 Hz** | **77,661** | 24.50 Hz | 416 Hz |

114× the network activity, 33× the active neurons — near-global runaway. The sign
map is unambiguously wired into the integrator. **But note the asymmetry: MN9
itself only moves 14.17 → 24.50 Hz.** It does not go quiet. See Finding 7.

### Part D — the results

All conditions reproduced with **fresh seeds**, 150 Hz drive, MN9 = mean of the
two bodies.

**§5 conditions, four independent seed families:**

| condition | seed 0 | 100 | 200 | 300 | `NOTES.md` |
|---|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sugar (LB3b/c) | 14.70 | 11.17 | 12.00 | 17.17 | 13.60 ± 2.31 |
| bitter (LB1a–e) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sugar + bitter | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| water (LB3a) | 0.00 | — | — | — | 0.00 |
| high salt (LB3d) | 15.70 | 16.83 | 17.33 | 19.50 | 16.80 ± 1.59 |

Sugar → MN9 across seeds: **13.76 ± 2.73 Hz** vs their **13.60 ± 2.31**.

Baseline is *exactly* silent — 0 spikes network-wide, not merely 0 at MN9.
Bitter suppression is genuine rather than a global shutdown: under
`sugar + bitter` the whole network is **more** active (87,508 Hz) than under
sugar alone (83,618 Hz) while MN9 sits at exactly zero.

**Dose-response**, 3 trials, two seeds — monotonic at both, cleaner than the
original (which dipped at 50 Hz):

| drive | seed 0 | seed 500 | `NOTES.md` |
|---|---:|---:|---:|
| 25 Hz | 3.00 | 3.83 | 3.67 |
| 50 Hz | 3.17 | 3.50 | 2.83 |
| 100 Hz | 6.50 | 5.83 | 9.83 |
| 150 Hz | 14.17 | 14.17 | 10.50 |
| 200 Hz | 25.17 | 29.33 | 28.17 |

**Left/right split** — 95.5 / 96.6 / 100.0 / 95.1 % of the response is on MN9_L
across the four seed families. Confirmed, and explained by the 11× input deficit
above.

**Shuffle control** — reproduces almost exactly, plus three fresh shuffle seeds
the original never ran:

| | MN9 mine | MN9 theirs | net mine | net theirs | >1 Hz mine | theirs |
|---|---:|---:|---:|---:|---:|---:|
| real wiring | 14.17 | 10.50 | 83,646 | 79,463 | 2,381 | 2,632 |
| shuffle 7 | **0.00** | 0.00 | 9,418 | 9,249 | 352 | 337 |
| shuffle 8 | **0.00** | 0.00 | 8,488 | 8,368 | 301 | 303 |
| shuffle 9 | **0.00** | 0.00 | 9,159 | 9,038 | 343 | 343 |
| shuffle 41 (new) | **0.00** | — | 9,687 | — | 401 | — |
| shuffle 42 (new) | **0.00** | — | 8,679 | — | 339 | — |
| shuffle 43 (new) | **0.00** | — | 9,404 | — | 383 | — |

MN9 is exactly zero in **all six** shuffles. The activity drop the notes honestly
flag is **8.8×** (they said "about 9-fold"). Their caveat that the null is not
degree-matched for total drive is correct and, if anything, understated — see
Finding 2.

**Anomaly (a) — LB3d out-drives sugar — reproduces in 4/4 seeds.** The stated
cause also checks out: all 26 LB3d neurons are predicted cholinergic in the
release, so a cell the literature calls glutamatergic/aversive acts excitatory here.

**Anomaly (b) — VNC saturation — reproduces in substance but is overstated.**
Top cells under sugar drive: `MN11D` 320.8, `INXXX137` 315.0, `MNad64` 302.4,
`MNx01` 268.6 Hz — the same cell types §5 names, same order of magnitude. Network
stays sparse: 83,618 Hz total, 0.508 Hz mean, 2,480 neurons above 1 Hz (they said
79,463 / ~2,600). See Finding 4 for where the wording fails.

### Part E — do the tests actually test anything

Five mutations, each applied to a clean tree, cache rebuilt where relevant, then
`pytest tests/test_male_cns.py` (15 tests), then reverted. Mutant caches were
built in a separate directory so the pristine cache was never touched.

| # | mutation | result | tests that caught it |
|---|---|---|---|
| 1 | zero all transmitter signs | **2 failed**, 13 passed | `test_sugar_drives_mn9` (0.00 Hz), `test_bitter_suppresses…` |
| 2 | glutamate → excitatory | **2 failed**, 13 passed | `test_bitter_alone_does_not_drive_mn9` (144 Hz), `test_bitter_suppresses…` (54→154 Hz) |
| 3 | permute postsynaptic column | **15 passed — SUITE FULLY GREEN** | **none** |
| 4 | swap sugar/bitter GRN sets | **4 failed**, 11 passed | both sugar tests, both bitter tests, `test_sugar_population_is_lb3b_and_lb3c` |
| 5 | MN9 → two arbitrary motor neurons | **3 failed**, 12 passed | `test_sugar_drives_mn9`, `test_bitter_suppresses…`, `test_mn9_bodies_are_the_documented_ones` |

Final `git status` after the run: **clean**. No tracked file was left modified.

**Mutation 3 is the finding.** Permuting `body_post` before caching destroys the
connectome — I verified only **0.502%** of `(pre, post)` pairs survive
(123,263 of 24,539,704) — and the suite does not notice. I measured why:

```
MN9_L on the PERMUTED graph, the suite's own protocol (0.5 s, 150 Hz):
  seed 0:  2.00 Hz      seed 3:  0.00 Hz
  seed 1:  2.00 Hz      seed 4:  4.00 Hz
  seed 2:  2.00 Hz
MN9_L on the REAL graph, same protocol:
  seed 0: 22.00 Hz   seed 1: 36.00 Hz   seed 2: 26.00 Hz
```

At `DURATION = 0.5 s`, **one spike = 2.0 Hz**, and the assertion is `sugar > 1.0`.
So a single accidental spike on a randomised graph passes the test that the
suite's own docstring calls "the load-bearing result". It passes at seed 0 (the
default) and would have *failed* at seed 3. The real effect is 11–18× larger than
the threshold, so there is ample room for a stricter bound.

`test_shuffling_the_connectome_abolishes_the_result` also passes under mutation 3,
because shuffling an already-random graph still yields MN9 = 0 and
`rates.sum() = 8,748 > 100`. The shuffle test is vacuous once the base graph is
destroyed — it cannot distinguish "wiring matters" from "there is no wiring".

**On assertions recorded from previous runs:** I checked for this specifically.
The thresholds are all loose and structural (`== 0.0`, `> 1.0`, `< sugar`,
`> 100.0`), not magnitudes copied from a run — which is good practice and means
they do not pass by construction. The two literal-value tests,
`test_mn9_bodies_are_the_documented_ones` (`[10331]`, `[16949]`) and
`test_sugar_population_is_lb3b_and_lb3c`, pin values derived from the *data*, and
I re-derived both from the annotation table independently. They are legitimate
regression pins, and mutations 4 and 5 show they fire. The problem is not
tautological assertions; it is that the one quantitative threshold is far too low.

---

## 3. Did not reproduce

* **Exact per-seed magnitudes.** The statistical picture reproduces tightly, but
  individual numbers do not match bit-for-bit at nominally identical seeds
  (e.g. dose-response seed 0: mine 3.00/3.17/6.50/14.17/25.17 vs theirs
  3.67/2.83/9.83/10.50/28.17). Most likely cause: the script that produced §5/§6,
  `scratch/sanity_check.py`, was never committed (Finding 5), so its exact
  protocol — trial count, seeding, whether MN9 is the mean of both bodies — cannot
  be checked. Their own §5 and §6 disagree at 150 Hz (13.60 vs 10.50), so some of
  this is internal to the original run.
* **`NOTES.md` §10's grep transcript.** It shows `grep -rl "flybrain.malecns\|from
  ..malecns" flybrain/` returning one line. The real command returns six. The
  underlying architectural claim is nonetheless **true**: exactly one real import
  exists (`flybrain/circuits/male_cns.py:573`); the other hits are docstrings and
  the package's own files.
* **`NOTES.md` §10's test counts.** See Finding 3.

---

## 4. Could not check

* **Everything Windows-specific.** Path handling, `FLYBRAIN_MALECNS_DIR` on
  Windows, `pyarrow`/`scipy` Windows wheels, the `Path.home()/.cache` default on
  Windows. This container is Linux. *Needed:* a Windows 11 host.
* **Everything GPU-specific**, including the claim that "a GPU port is plausible".
  No GPU present. *Needed:* the RTX 5070 Ti host.
* **Performance on the target hardware** (Ryzen 5 3600, 32 GB) and on the i5-7200U
  mini-PC. Part F measures a 4-core Xeon at 2.8 GHz.
* **The Tastekin et al. quotations in `NOTES.md` §3.** The preprint *exists* and I
  confirmed its identity — DOI `10.1101/2025.08.25.671814`, title "From Sensory
  Detection to Motor Action: The Comprehensive *Drosophila* Taste-Feeding
  Connectome", authors Tastekin, de Haan Vicente, Beresford, Morris, Beckett,
  Schlegel, Gkantia, FlyEM Project Team, Cambridge Connectomics Group, Marin,
  Costa, Jefferis, Ribeiro, dated 2025-08-25 — via OpenAlex and Europe PMC
  (`PPR1072256`). **But I could not read the full text**, so the four verbatim
  quotations underpinning the sugar/bitter/water/salt assignment are unverified.
  bioRxiv returned HTTP 429 to every route (plain and browser-UA), Europe PMC has
  no full text or supplementary files for this preprint, and the only open-access
  location OpenAlex lists is the blocked bioRxiv PDF. `NOTES.md` §8 reports the
  same block, so this is consistent rather than suspicious — but it means the
  **single most load-bearing external citation in the whole SDK is taken on
  trust.** *Needed:* a network path to bioRxiv, or institutional access to the
  *Cell* version.
* **Whether the FlyWire cross-tab in §3(a) is reproducible.** I did not clone
  `flyconnectome/flywire_annotations`. I did verify the MaleCNS side: `flywireType`
  lumps LB3, LB3a, LB3b, LB3c, LB3d and LB4b all under `LB3`, so that column can
  only support the coarse LB1-vs-LB3 split — exactly as §3 says, which is why it
  leans on Tastekin et al. for the subtype split. The structure of the argument is
  sound; its key premise is the unverified quote above.
* **`min_synapses=5` behaviour** — `[PENDING, Part F]`.
* **The coupling pedestal** — `[PENDING, Part G]`.

---

## 5. Findings

`[PENDING — written last]`

---

## 6. Environment

| | |
|---|---|
| OS | Linux 6.18.44-fc-v33, x86_64, glibc 2.39 |
| CPU | Intel Xeon @ 2.80 GHz, 4 cores |
| RAM | 15 GB |
| GPU | none |
| Python | 3.11.15 (GCC 13.3.0) |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| pyarrow | 25.0.1 |
| scipy | 1.17.1 |
| pytest | 9.1.1 |
| torch / torchvision | 2.14.0+cpu / 0.29.0+cpu |
| flyvis | 1.2.0 |
| BLAS | scipy-openblas 0.3.31 |

Nothing was pre-installed; the venv was built from scratch. `NOTES.md` §7 reports
"NumPy 2.4 / SciPy 1.17", matching the major versions used here.

Timings observed so far: cache build 24.1 s (peak RSS 2.52 GB); full suite
**40 passed in 1013 s** (under CPU contention); `tests/test_male_cns.py` alone
15–25 s; suite with no connectome cache: 22 skipped cleanly, instantly.
