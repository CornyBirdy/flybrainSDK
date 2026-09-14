# MaleCNS working notes

Stage-1 log for adding a second circuit (`male_cns`) to flybrainSDK. What the
data actually is, what was verified, and what could not be.

Written as work happened, so it records dead ends as well as results.

---

## 1. What the dataset actually is

I did not trust anything I thought I knew about this release; it postdates my
training data. Everything below was checked against a live server today.

**Release.** `male-cns:v1.0`. Two dates are floating around and both are real:
the data went public on **8 June 2026** (that is the `segment property update`
timestamp neuPrint reports, and the date the download page went up), and the
**Cell** paper describing it appeared on **3 September 2026**. The brief said
3 September; the data has in fact been downloadable since June, which is why
`neuprint-python`, `navis` and the natverse packages already support it.

**Where it lives.** `male-cns:v1.0` is served from both
`neuprint.janelia.org` and `neuprint-cns.janelia.org` — confirmed by
`GET /api/dbmeta/datasets` on each, which also still lists `male-cns:v0.9`
(June 2025), `manc`, `hemibrain` and `optic-lobe`. The neuPrint REST API needs
a personal bearer token, so it is a bad dependency for an SDK that has to
install unattended.

The bulk tables need no token. They are public, CC-BY, in a Google Cloud
Storage bucket:

```
gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
```

Eleven files, 30 GB total, Arrow **feather** format (not parquet, not CSV).
Listing them via the JSON API gave the real sizes, which differ a little from
the download page:

| file | size | what it is |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14.5 MB | cell types, classes, sides |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43.3 MB | NT prediction per neuron |
| `body-stats-male-cns-v1.0-minconf-0.5.feather` | 778 MB | per-neuron synapse stats |
| `connectome-weights-...-minconf-0.5.feather` | 1051 MB | every connection |
| `connectome-weights-...-traced-only.feather` | 508 MB | connections between traced bodies |
| `connectome-weights-...-significant-only.feather` | 502 MB | thresholded variant |
| `syn-points-...` | 13.1 GB | individual synapse coordinates |
| `syn-partners-...` (3 variants) | 3–6.8 GB each | synapse-level partner pairs |
| `tbar-neurotransmitters-male-cns-v1.0.feather` | 2.7 GB | NT probability per T-bar |

**Three of those eleven are enough**, 566 MB total: annotations,
neurotransmitters, and the *traced-only* weights. I used traced-only rather
than the full weights table because it restricts the graph to proofread
bodies rather than including unproofread fragments.

**Schemas, as they actually are.** No guessing — these came from the files.

```
connectome-weights ...traced-only:  body_pre:int64  body_post:int64
                                    weight:int64  type_pre:string  type_post:string
body-annotations:                   36 columns; the useful ones are bodyId, type,
                                    instance, class, subclass, superclass, somaSide,
                                    rootSide, receptorType, statusLabel, flywireType
body-neurotransmitters:             body:int64  predicted_nt  predicted_nt_confidence
                                    ground_truth  celltype_predicted_nt  consensus_nt
```

Two things worth knowing that I would not have guessed:

* `weight` is a **synapse count**, not a strength. There is no strength column
  anywhere in the release.
* the neurotransmitter table has **1,835,518 rows** — one per segmented body,
  including tiny fragments — of which only ~164 k matter. 1.67 M of those rows
  are `unclear`. Restricted to the traced graph the picture is much better
  (see below). `consensus_nt` already backs a low-confidence per-body call off
  onto the cell-type-level prediction, so it is the column to use.

**Size of the graph I built** (traced-only, all connections of ≥1 synapse):

```
164,587 neurons
 25,563,197 connections
124,025,046 synapses
```

which matches the release's headline "~166,000 neurons, ~125M synapses".

Neurotransmitters over those 164,587 neurons — far better coverage than the
raw table suggests:

| transmitter | neurons |
|---|---|
| acetylcholine | 103,701 |
| glutamate | 29,296 |
| GABA | 22,055 |
| histamine | 5,904 |
| unclear | 3,090 |
| dopamine | 392 |
| octopamine | 101 |
| serotonin | 48 |

**Package support.** `neuprint-python`, `neuprintr` and `navis` all address
`male-cns:v1.0` today, and `natverse/malecns` exists specifically for it. All
of them go through the token-gated REST API. `fafbseg` is FlyWire/FAFB and does
not cover MaleCNS. The SDK therefore talks to the bucket directly over plain
HTTPS and depends on nothing but `pyarrow`, `pandas`, `scipy` and `numpy`.

---

## 2. Measured vs assumed

This is the part that matters.

**Measured, and taken straight from the release**

* which neuron contacts which neuron;
* how many synapses each of those contacts has;
* the neurotransmitter prediction for each neuron, and for ~85 k neurons a
  ground-truth transmitter label;
* cell type, class, soma side, entry nerve.

**Assumed, by me, and each one is a choice**

1. **Synapse count is proportional to synaptic strength.** The connectome does
   not measure strength. Same convention as Shiu et al.
2. **`w_syn` = 0.275 mV per synapse, for every synapse in the animal.** This is
   a free parameter that Shiu et al. fitted so that model firing rates come out
   physiological. It is not a measurement.
3. **Transmitter sign.** ACh → excitatory. GABA → inhibitory. **Glutamate →
   inhibitory**, because in the fly central brain glutamate acts mainly on the
   GluCl-α chloride channel (Liu & Wilson 2013, *J. Neurosci.* 33:10659) — the
   opposite of the vertebrate convention, and the single most consequential
   sign choice here: it flips 29,296 of 164,587 neurons. Histamine →
   inhibitory (ort/HisCl1).
4. **Dopamine, octopamine, serotonin and `unclear` neurons get sign 0**, i.e.
   they receive input and can spike but have no modelled output. This is a
   deliberate departure from Shiu et al., who forced every edge to ±1. It
   removes 1,023,493 of 25,563,197 connections (3.2 M of 124 M synapses),
   leaving 120,793,200 signed synapses. An explicit zero seemed better than a
   guess, but it does mean the model has no neuromodulation at all — see the
   LB3d problem in §5.
5. **One set of passive membrane constants for all 164,587 neurons**, from
   Kakaria & de Bivort 2017, regardless of cell type or size.
6. **No gap junctions** (EM does not resolve them), no dendritic processing,
   zero basal firing rate.

---

## 3. Which neurons are the sugar and bitter GRNs

This took the longest and is the part most likely to be got wrong by
assumption, so it is worth spelling out.

MaleCNS annotates 1,428 gustatory neurons but **carries no taste-modality
label at all** — I grepped every string column for `sugar`, `bitter`, `Gr64`,
`Gr66`, `ppk28`, `water`, `salt` and got nothing. What it has is anatomical
type names: `LB1a`–`LB4b` (labellar bristle), `PhG*` (pharyngeal), `LgLG*`
(leg), `WG*` (wing), `*_tpGRN` (taste peg).

Two independent sources pin the modalities down.

**(a) FlyWire.** Schlegel et al. 2024's annotation table
(`flyconnectome/flywire_annotations`, Supplemental file 1) *does* carry a
modality in `cell_sub_class`. Cross-tabulating it against `cell_type` gives an
unambiguous mapping, and MaleCNS's `flywireType` column uses the same
nomenclature:

```
LB1a,LB1d  LB1b  LB1c  LB1e   -> bitter        (65 neurons)
LB2a-b  LB2c  LB4a           -> low-salt      (19)
LB2d  LB3                    -> sugar/water   (129)
claw_tpGRN  dorsal_tpGRN     -> taste peg     (71)
```

**(b) The taste-feeding connectome paper**, which splits FlyWire's lumped
`LB3` into subtypes and does it on *both* FlyWire and MaleCNS: Tastekin,
de Haan Vicente, Beresford, Morris, Beckett, Schlegel, Costa, Jefferis &
Ribeiro, "From Sensory Detection to Motor Action: The Comprehensive
*Drosophila* Taste-Feeding Connectome", bioRxiv 2025.08.25.671814 (published in
*Cell*, 2026). Verbatim:

> "Both LB3b and LB3c projection patterns matched the Gr64f-GAL4 positive
> neurons and are, thus, likely to correspond to sweet-sensing GRNs, mediating
> appetitive responses"

> "They best matched the Gr33a-GAL4 driver projection pattern which is
> expressed in all bitter-sensing GRNs … Thus, we propose LB1a-d to be
> bitter-sensing subtypes" … "[LB1e] cluster together with canonical
> bitter-sensing GRNs, sharing specific downstream partners … making LB1 a
> broad aversive type"

> "The LB3a subtype morphology matched the projection pattern of ppk28-GAL4
> positive neurons, which are supposed to be water-sensing GRNs"

> "Morphologically, the LB3d GRNs match the glutamatergic Ir7c-GAL4 and
> ppk23-Gal4 positive neurons, which are involved in high salt avoidance"

So, in MaleCNS v1.0 body counts:

| role | types | neurons |
|---|---|---|
| sugar | LB3b, LB3c | 34 |
| bitter / aversive | LB1a–LB1e | 57 |
| water | LB3a | 17 |
| high salt (aversive) | LB3d | 26 |

**Caveats the paper itself flags, and one I found.**

* Their Figure 2F: "The challenges faced while reconstructing the maleCNS
  lbGRNs have affected mostly the LB3a and LB3b subtypes." So the sugar set
  here is probably slightly under-reconstructed.
* 6 of the 57 bitter GRNs (all of `LB1b`) have `consensus_nt = unclear` — the
  cell-type-level call is 0.49 confidence, genuinely ambiguous — so under
  assumption 4 above they have no modelled output. The bitter set is
  effectively 51 neurons, not 57.

**Independent corroboration that LB3b really is the sugar type.** Without
looking for it, the strongest output target of LB3b in my built graph is
`AN13B002`. The paper, in a completely different section about wing GRNs, says:
"A key downstream partner of WG2 is Dandelion, an AN typed as AN13B002 in the
maleCNS … Dandelion is also a key downstream neuron of LB3b GRNs". That the
graph reproduces their strongest-partner claim is a decent check that I loaded
and signed the connectome correctly.

**The read-out neuron.** MN9 is annotated directly: type `MN9`, superclass
`cb_motor`, two bodies — `10331` (`MN9_L`) and `16949` (`MN9_R`). MN9 extends
the rostrum, the largest segment of the proboscis; it is the standard read-out
for proboscis extension and the one Shiu et al. use. Note `MN9_R` is flagged
`RT Hard to trace` while `MN9_L` is `Roughly traced`, and this shows up in the
results as a strong left/right asymmetry.

---

## 4. The model

Reimplemented from `philshiu/Drosophila_brain_model` (`model.py`,
`default_params`) — the repository accompanying Shiu et al., *Nature* 634,
210–219 (2024). Cloned it and read it rather than working from the paper text.

```
dv/dt = (v_0 - v + g) / t_mbr      frozen while refractory
dg/dt = -g / tau                   frozen while refractory
spike when v > v_th; then v <- v_rst, g <- 0, refractory t_rfc
a presynaptic spike adds w to the postsynaptic g after delay t_dly
w = w_syn * sign(transmitter) * n_synapses
```

v_0 = v_rst = −52 mV, v_th = −45 mV, t_mbr = 20 ms, tau = 5 ms, t_rfc = 2.2 ms,
t_dly = 1.8 ms, w_syn = 0.275 mV, internal dt = 0.1 ms.

**Deviation: no Brian 2.** Shiu et al. build a Brian 2 network and run it as a
batch job. The SDK has to advance the model one host frame at a time, and I did
not want a code-generation/compiler dependency, so this is a direct
NumPy/SciPy integrator. It uses the exact (exponential-Euler) solution of the
linear subsystem, which is what Brian's `method='linear'` does, at the same
0.1 ms step. Checked against a 2000×-oversampled forward Euler reference:
agreement to <1e-9 V in both v and g after one step.

**Deviation: Poisson drive.** Brian's `PoissonInput(target_var='v')` with
weight `w_syn * f_poi` = 68.75 mV against a 7 mV threshold means every event
is suprathreshold, and the reference sets the refractory period of driven
neurons to zero. Reproduced exactly, so a driven neuron fires as a Poisson
process at the requested rate. Verified: a neuron asked for 150 Hz fired at
143 Hz over 5 s.

**Verified separately:** an excitatory connection raises the postsynaptic rate
(0 → 35.5 Hz) and adding an inhibitory one lowers it (35.5 → 10.5 Hz).

---

## 5. Results

`scratch/sanity_check.py`, 5 trials × 1 s, GRNs driven at 150 Hz.

```
MN9 firing rate
  baseline (nothing driven)             0.00 +/- 0.00 Hz
  sugar        (LB3b/c)                13.60 +/- 2.31 Hz
  bitter       (LB1a-e)                 0.00 +/- 0.00 Hz
  sugar + bitter                        0.00 +/- 0.00 Hz
  water        (LB3a)                   0.00 +/- 0.00 Hz
  high salt    (LB3d)                  16.80 +/- 1.59 Hz
```

**Sugar drives MN9: reproduced.** 13.6 Hz against a silent baseline. MN9 is
two synaptic hops from the sugar GRNs in this graph (0 direct synapses,
499 neurons at one hop, MN9 reached at two), consistent with the published
sugar → premotor → MN9 arc.

**Bitter suppresses it: reproduced.** Co-activating the bitter GRNs takes MN9
from 13.6 Hz to complete silence. Bitter alone does not drive MN9 either.

**Shuffle control: passes.** See §6.

**Two results that do not look right, and I am not going to paper over them.**

*(a) LB3d (high salt) drives MN9 harder than sugar does.* LB3d should be
aversive. The reason is visible in the data: LB3d's strongest targets
(`GNG038`, `GNG042`) are the same neurons LB3c drives, and MaleCNS predicts
LB3d to be **cholinergic** (26/26 neurons, mean confidence 0.71, no ground
truth) — whereas the paper's whole argument is that LB3d is *glutamatergic*
(Ir7c+) and therefore "might inhibit the activity of neurons also in
attractive circuits via glutamatergic signalling". The model inherits the
release's NT prediction, so a neuron the literature calls inhibitory acts
excitatory here. This is a real limitation of connectome-only modelling, not
a bug in the integrator, and it is exactly the kind of thing that makes
sign assumptions load-bearing.

*(b) A handful of VNC motor neurons saturate.* The most strongly driven cells
under sugar activation are `MN11D` (342 Hz), `MNx01` (289 Hz), `MNad64`
(282 Hz) — close to the 455 Hz ceiling the 2.2 ms refractory imposes. Shiu et
al. modelled the brain only; MaleCNS includes the ventral nerve cord, and the
extra recurrent loops appear to run away. Network-wide the model is still
sparse (79,463 Hz summed over 164,587 neurons ≈ 0.5 Hz mean, ~2,600 neurons
above 1 Hz), so this is a local instability, not global runaway — but any
downstream use of VNC motor rates should treat those numbers as unphysiological.

---

## 6. Shuffle control

Rewired the graph preserving every neuron's in-degree and out-degree (each
edge keeps its presynaptic neuron, so out-degree, synapse count and
transmitter sign are untouched; postsynaptic slots are permuted). What that
destroys is exactly what the connectome measures: who talks to whom.

```
                              MN9      whole network      neurons > 1 Hz
  real wiring, sugar        10.50 Hz     79,463 Hz            2,632
  shuffle seed 7             0.00 Hz      9,249 Hz              337
  shuffle seed 8             0.00 Hz      8,368 Hz              303
  shuffle seed 9             0.00 Hz      9,038 Hz              343
```

MN9 goes to exactly zero under every shuffle, while the network keeps firing.
So the sugar -> MN9 result is a property of the wiring, not of the simulator.

Reported honestly: shuffling also drops total network activity about 9-fold.
The null is therefore not perfectly matched — a random graph with the same
degrees propagates less activity overall, so some of MN9's silence is a
general loss of drive rather than a specific loss of the sugar pathway. The
control still rules out the failure mode it was there to rule out (a simulator
that lights up MN9 for any input), but it is not a clean single-variable
comparison and I am not going to claim it is.

A second, independent specificity check that does not have that problem —
MN9's response is graded with sugar drive rate, on the same wiring:

```
   25 Hz drive -> MN9  3.67 Hz   (L  7.0, R 0.3)
   50 Hz drive -> MN9  2.83 Hz   (L  5.7, R 0.0)
  100 Hz drive -> MN9  9.83 Hz   (L 19.0, R 0.7)
  150 Hz drive -> MN9 10.50 Hz   (L 21.0, R 0.0)
  200 Hz drive -> MN9 28.17 Hz   (L 54.0, R 2.3)
```

Monotonic apart from the 50 Hz point, which is within the noise of 3 trials.
Note the left/right split: **essentially the whole response is MN9_L**. MN9_R
is flagged `RT Hard to trace` in the release and evidently lost the inputs
that matter. Anything reading MN9 should read the two sides separately and
should not treat MN9_R as a working read-out.

The point of running three seeds and reporting total network activity is that
a shuffle which merely *silences* the network would make the control vacuous.

---

## 7. Cost

Benchmarked in this container: Intel Xeon @ 2.10 GHz, 4 vCPU, 15 GB RAM,
NumPy 2.4 / SciPy 1.17, single-threaded in the hot loop. **Not** the RTX 5070 Ti
workstation — expect a Ryzen 5 3600 to be meaningfully faster, but the shape of
the number will not change.

Wall time per second of simulated biological time, whole 164,587-neuron
network at the reference 0.1 ms internal step:

| condition | wall / sim-second |
|---|---|
| nothing driven (network silent) | 3.3 s |
| water GRNs driven | 5.0 s |
| bitter GRNs driven | 5.7 s |
| sugar + bitter | 7.8 s |
| sugar GRNs driven | 8.9 s |
| high-salt GRNs driven | 11.1 s |

**So it does not run in real time, and it will not hit 30 fps.** A 1/60 s host
frame costs 55–185 ms of wall time depending on how much of the network is
firing; a 1/30 s frame costs 110–370 ms. That is **3–9 frames per second**,
against the optic lobe's ~30 ms/step. Cost scales with activity, because the
dense part of the step is fixed but the spike scatter is not.

This is inherent to the reference model rather than to the implementation: at
dt = 0.1 ms, one 1/60 s frame is 167 internal steps, each touching all 164,587
neurons. Raising dt would change the model.

Memory: the built cache is a 197 MB `.npz` (24.5 M signed edges, float32 data +
int32 indices); resident set during a run is about 1.2 GB, dominated by the CSR
matrix plus the 18 × 164,587 float32 delay buffer.

Nothing here uses the GPU. The inner loop is a sparse gather-scatter over a few
hundred spiking rows per 0.1 ms step plus half a dozen dense passes over
164,587 floats; it is memory-bandwidth bound, not FLOP bound, so a GPU port is
plausible but is not a one-line change.

**The 24/7 mini PC (i5-7200U, 2 cores, 16 GB).** It will hold the data — 16 GB
is enough — but at roughly half the cores and lower clocks it would be slower
still. A reduced mode is cheap to add later (build the cache with
`min_synapses=5`, which drops 24.5 M edges to ~6.2 M) but I have not measured
it and am not claiming it works.

---

## 8. Dead ends and things that fought back

* **bioRxiv blocks this container.** `www.biorxiv.org` returns Cloudflare 1015
  / HTTP 429 for the `/content/...full`, `/full.pdf` and `.source.xml` routes.
  The PDF is reachable through the path Semantic Scholar advertises
  (`/content/biorxiv/early/2025/08/25/...full.pdf`) — that one worked first
  try. Europe PMC's `fulltextRepo` link for the same preprint returned
  `{"error":"PDF link has expired or is invalid"}`.
* **cell.com returns 403** to this container, so the published version was not
  readable; the preprint was.
* **No neuPrint token**, so nothing in this work goes through the REST API.
  Everything is from the public bucket. That is a feature for the SDK.
* **`pypdf` crashed on import** — Debian's `cryptography` 41 against the
  system `_cffi_backend`; fixed by `pip install --upgrade cffi`.
* `body-stats` (778 MB) and the `syn-points`/`syn-partners` tables (3–13 GB)
  were never downloaded. Nothing in a LIF model needs synapse coordinates.

## 9. Reference implementations read first

* `philshiu/Drosophila_brain_model` — the Shiu et al. paper repo. Cloned and
  read; `model.py` `default_params` is the source of every constant in §4, and
  `Connectivity_783.parquet` showed how they sign edges (a strict ±1
  `Excitatory` column, no zeros — see assumption 4 for where I differ).
* `flyconnectome/flywire_annotations` — Schlegel et al. 2024. Cloned; this is
  where the GRN modality mapping in §3 comes from.
* `vshapenko/flypoke` and `eonsystemspbc/fly-brain` were listed in the brief
  but not needed once the Shiu repo was in hand.
