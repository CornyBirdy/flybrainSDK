# flybrain port reference

This is the contract between the fly neuroscience and whatever is embodying it.
If you are picking this SDK up in a later session, read this file and you should
not need to relearn the biology or reread the flyvis source.

Two circuits ship, and they can be loaded together:

| circuit | model | ports |
|---|---|---|
| `optic_lobe` | pretrained flyvis optic lobe, parameters **fitted against neural recordings** | `VISUAL_FIELD` in; `YAW`, `HS_LEFT`, `HS_RIGHT`, `FLOW_ASYMMETRY` out |
| `male_cns` | MaleCNS v1.0 connectome as a leaky integrate-and-fire model, parameters **assigned by convention** | gustatory drive in Hz in; descending/motor neuron firing rates out |

Most of this file is about `optic_lobe`, which came first.
[Jump to `male_cns`](#the-male_cns-circuit) for the second one. The difference
in the "parameters" column above is the single most important thing to carry
between them and is spelled out there.

---

## The API, in full

```python
from flybrain import FlyBrain, sensory, motor

fly = FlyBrain(circuits=["optic_lobe"])

height, width = fly.input_shape(sensory.VISUAL_FIELD)   # (391, 391)

fly.set_input(sensory.VISUAL_FIELD, frame)   # numpy RGB or greyscale image
fly.step(dt)                                 # dt in seconds; default 1/60
yaw = fly.get_output(motor.YAW)              # signed float, + = steer right

fly.reset()                                  # back to resting state
print(fly.describe())                        # loaded circuits and their ports
```

That is the whole interface. `FlyBrain` routes values to whichever loaded
circuit declares the port, so adding circuits later does not change these calls.

---

## Ports: `optic_lobe`

### Inputs

| Port | Shape | Units | Meaning |
|---|---|---|---|
| `sensory.VISUAL_FIELD` | `(H, W, 3)` or `(H, W)` | luminance in `[0, 1]` | First-person view of the world. |

`uint8` arrays in `[0, 255]` and float arrays in `[0, 1]` are both accepted and
normalised internally; RGB is collapsed to luminance with BT.601 weights.
**Image `+x` is the fly's right, image `+y` is down.**

`fly.input_shape(sensory.VISUAL_FIELD)` returns `(391, 391)`, the smallest frame
that covers the photoreceptor lattice without resampling. Other sizes work but
are resized first.

### Outputs

| Port | Units | Meaning |
|---|---|---|
| `motor.YAW` | dimensionless, nominally `[-1, 1]` | Steering command. **Positive = steer right.** |
| `motor.HS_LEFT` | model membrane potential | Left HS cell: front-to-back motion in the left visual field. |
| `motor.HS_RIGHT` | model membrane potential | Right HS cell: front-to-back motion in the right visual field. |
| `motor.FLOW_ASYMMETRY` | model membrane potential | Left/right imbalance in flow *speed*. Positive = right side streaming faster (right wall nearer). |

`YAW` is the steering signal. The other three are diagnostics and building
blocks; `FLOW_ASYMMETRY` is genuinely useful and is discussed below.

---

## What HS cells are

The fly's optic lobe is a retinotopic stack: ~721 hexagonal facets feed
photoreceptors R1–R8, then the lamina (L1–L5), then the medulla, and finally
**T4** and **T5** — small-field, direction-selective motion detectors. T4 reports
moving brightness increments, T5 moving decrements. Each comes in four subtypes,
one per cardinal direction. flyvis ships the measured tuning in
`flyvis.utils.groundtruth_utils.preferred_directions`:

| subtype | preferred direction | contrast |
|---|---|---|
| T4a / T5a | 180° — leftward | ON / OFF |
| T4b / T5b | 0° — rightward | ON / OFF |
| T4c / T5c | 90° — upward | ON / OFF |
| T4d / T5d | 270° — downward | ON / OFF |

T4/T5 cells see only a couple of facets each. **HS (horizontal system) cells**
are the next stage: large-field tangential neurons in the lobula plate, each
summing the horizontally-tuned T4/T5 columns across an entire half of the visual
field. A fly has a handful per side (HSN, HSE, HSS). They are among the
best-characterised neurons in any brain, and they drive the optomotor response —
the reflex that keeps an insect flying straight.

A right HS cell depolarises for **front-to-back** motion across the right eye and
hyperpolarises for back-to-front. The left HS cell is its mirror image.

### Why this SDK *models* HS cells rather than simulating them

**flyvis does not contain HS cells.** Its connectome is the 65 columnar cell
types of the optic lobe, and it stops at T4/T5 and the Tm/TmY outputs. HS cells
are downstream of that.

So this SDK adds the one documented step that HS cells perform — spatial
integration of horizontally-tuned T4/T5 over a hemifield. The HS output is
therefore a **linear read-out of real pretrained activity**, not a second
learned model. Nothing is invented: the direction assignment comes from flyvis's
own ground-truth table, and the pooling geometry from the connectome's hex
coordinates.

### Two optic lobes out of one network

A fly has two mirror-symmetric optic lobes. flyvis provides one, and its cell
types are trained independently, so T4a and T4b are *not* exact mirror images.
Splitting a single lattice down the middle therefore produces a biased read-out:
a perfectly symmetric scene gives `HS_LEFT != HS_RIGHT`, which would show up as
a phantom steering command.

This wrapper instead evaluates the network on a **batch of two**: the frame as
seen by the right lobe, and the horizontally mirrored frame as seen by the left
lobe. Both are read out with the *identical* formula. `HS_RIGHT - HS_LEFT` is
then exactly antisymmetric under mirroring the input, so a symmetric world gives
exactly zero. `tests/test_optic_lobe.py` asserts both properties. Batching keeps
the second lobe nearly free.

All read-outs are **deviations from the resting state**, captured on a grey
screen during `reset()`. T4/T5 subtypes have different resting potentials, so
without this the constant offset dominates the motion signal.

---

## Why left-minus-right is a steering signal

Write `HS_R` and `HS_L` for the two front-to-back signals. Then:

```
YAW = yaw_gain * (HS_RIGHT - HS_LEFT)
```

Consider the two things that make a scene move:

**Rotation.** If the body yaws right, the whole world sweeps left across both
eyes. That is back-to-front on the right eye (`HS_R` falls) and front-to-back on
the left (`HS_L` rises). `HS_R - HS_L` goes **negative**, which commands a turn
to the **left** — cancelling the unintended rotation. The signal is already
corrective; feed it straight into a turn rate with no sign flip.

**Forward translation.** Flying straight ahead makes the world stream outward
from a point of expansion: front-to-back on *both* sides at once. `HS_R` and
`HS_L` both rise, and the difference cancels them.

So the difference is a **rotation detector that is blind to forward motion**.
That is exactly what a course stabiliser needs, and it is why the left-minus-right
form matters rather than either cell alone.

### Sign convention, stated once

* `YAW > 0` → steer **right** (clockwise seen from above).
* `YAW < 0` → steer **left**.
* An unintended rotation produces a `YAW` that **opposes** it.

Verified against the pretrained weights by `scripts/verify_sign.py` and locked in
by `tests/test_optic_lobe.py::test_yaw_opposes_body_rotation`.

### Scale

`YAW` is `yaw_gain` (default 10) times the raw HS difference, clipped to
`yaw_clip` (default ±1). The raw difference depends on scene contrast and
distance, so **retune `yaw_gain` for your own world**. Measured in the demo
corridor:

| stimulus | raw `HS_RIGHT - HS_LEFT` |
|---|---|
| body rotating 20 °/s | ∓0.011 |
| body rotating 40 °/s | ∓0.021 |
| body rotating 160 °/s | ∓0.096 |
| symmetric scene, straight and centred | 0.000 (exact) |

---

## The limit you must know about: YAW stabilises rotation, not position

`YAW` has no position reference, and its response to *translational* flow
asymmetries is **syn-directional** — it turns toward whichever side is streaming
faster. Measured in the demo corridor (positive = commands a right turn):

| situation | raw `YAW` | effect |
|---|---|---|
| rotating right | negative | **corrective** ✓ |
| offset toward the right wall | positive | steers further right ✗ |
| heading 10° to the right | positive | steers further right ✗ |

This is a real property of HS-driven optomotor behaviour, not a modelling bug:
the reflex stabilises *course*, and insects use a separate mechanism to stay
centred. A `YAW`-only controller in a corridor holds its heading against gusts
but slowly diverges in position.

### The fix: `motor.FLOW_ASYMMETRY`

`FLOW_ASYMMETRY` compares horizontal motion *magnitude* between the two sides,
ignoring direction — the speed-balance or "centering" cue described by
Srinivasan and colleagues for bees. Nearer surfaces sweep past faster, so:

* offset toward the right wall → positive,
* heading to the right → positive,
* pure rotation → the two sides sweep equally fast, so it carries no *net*
  position information.

Crucially, `FLOW_ASYMMETRY` has the **same** sign as `YAW` for translational
errors and the **opposite** sign for rotation. So

```python
steer = YAW - CENTERING_WEIGHT * FLOW_ASYMMETRY
```

cancels `YAW`'s destabilising translational term while *reinforcing* its
rotation correction. The demo uses `CENTERING_WEIGHT = 8.0`. Both terms are
optic flow from the same T4/T5 population; no non-visual state is involved.

With `CENTERING_WEIGHT = 0` the car holds heading but drifts into a wall in
~10 s; with it set to 8 the car recentres from a 1.2 m offset and rides out
three rotational gusts over 24 s.

---

## Time, and calling `step`

Pass the real frame duration to `step(dt)`. Internally the circuit subdivides it
so no Euler step exceeds `max_substep_dt` (default 1/200 s) — flyvis is unstable
above 1/50 s. Holding a frame constant across substeps is a zero-order hold, so
a slow render loop stays numerically sound; it just gives the fly less temporal
detail.

State persists across `step` calls. Call `reset()` between independent episodes,
or the previous episode's motion adaptation leaks in.

---

## Performance: `optic_lobe`

~30 ms per step on 2 CPU cores at 391×391 with both lobes, so roughly 30 fps
without a GPU. Set `FLYBRAIN_DEVICE=cuda` (or `OpticLobeConfig(device="cuda")`)
to use a GPU; CPU is the automatic fallback. The network is small — 734 free
parameters over 45,669 cells — so CUDA helps less than you might expect.

---

## Configuration: `optic_lobe`

```python
from flybrain import FlyBrain
from flybrain.circuits import OpticLobeConfig

fly = FlyBrain(
    circuits=["optic_lobe"],
    configs={"optic_lobe": OpticLobeConfig(yaw_gain=25.0, device="cuda")},
)
```

Fields: `ensemble`, `member`, `checkpoint`, `device`, `max_substep_dt`,
`warmup_seconds`, `grey_level`, `hemifield_deadzone`, `yaw_gain`, `yaw_clip`.
See the docstring in `flybrain/circuits/optic_lobe.py`.

By default the SDK loads the 50-network pretrained ensemble `flow/0000`, sorts
it by validation loss and runs the best member. `member=0` skips the ensemble
scan and loads that network directly, which is faster to start.

---

## The `male_cns` circuit

The whole central nervous system of a male *Drosophila* — brain, optic lobes
and ventral nerve cord — as a spiking model over real measured wiring.

```python
from flybrain import FlyBrain, sensory, motor

fly = FlyBrain(circuits=["male_cns"])
fly.set_input(sensory.SUGAR_GRN, 150.0)      # Hz, like an optogenetic drive
fly.step(1 / 60)
mn9 = fly.get_output(motor.MN9_L)            # Hz, proboscis extension
```

Build the connectome cache once before first use — 566 MB from a public
CC-BY bucket, no account and no token:

```bash
pip install -e ".[male_cns]"
python -m flybrain.malecns download
```

If the cache is missing the circuit raises and tells you this. It never
invents a connectome.

### What is measured and what is assumed

This distinction matters far more here than for `optic_lobe`, because flyvis's
parameters were **fitted against neural recordings** and these were not.

**Measured** — straight out of MaleCNS v1.0 (HHMI Janelia FlyEM / Google
Research / Cambridge Connectomics, CC-BY):

- which neuron contacts which neuron: 164,587 neurons, 25,563,197 connections;
- how many synapses each connection has: 124,025,046 in total;
- a neurotransmitter prediction per neuron, with ground truth for ~85,000;
- cell type, class, soma side and entry nerve.

**Assumed** — every one of these is a choice this SDK makes, not a datum:

| assumption | source | consequence if wrong |
|---|---|---|
| synapse count ∝ synaptic strength | convention, Shiu et al. 2024 | all relative weights are wrong |
| every synapse is worth **0.275 mV** | free parameter fitted by Shiu et al.; not a measurement | firing rates scale arbitrarily |
| **glutamate is inhibitory** (GluCl-α; Liu & Wilson 2013) | fly-specific pharmacology, opposite to the vertebrate convention | flips the sign of 29,296 of 164,587 neurons |
| GABA and histamine inhibitory, ACh excitatory | standard | — |
| dopamine/octopamine/serotonin/unclear neurons have **no output** | our choice; Shiu et al. instead force every edge to ±1 | drops 1.02 M of 25.56 M connections; no neuromodulation at all |
| one set of membrane constants for all 164,587 neurons | Kakaria & de Bivort 2017 | ignores cell size and type |
| no gap junctions, no dendritic processing, zero basal rate | EM cannot resolve them | — |

And one more that is easy to miss: **MaleCNS carries no taste-modality
annotation at all.** The mapping from `SUGAR_GRN` to cell types LB3b and LB3c
comes from the literature — Tastekin et al. matched both to the Gr64f-GAL4
projection pattern — not from the release. The same goes for every gustatory
port. The full quotations are in the port docstrings in `io/sensory.py`.

### The sanity check, and its result

Activating the sweet-sensing labellar GRNs should extend the proboscis; adding
bitter should stop it. That is the standard test for this class of model and
the reason to believe anything else it says.

```
MN9_L firing rate, 5 trials x 1 s, GRNs driven at 150 Hz

  baseline (nothing driven)        0.00 +/- 0.00 Hz
  sugar        (LB3b, LB3c)       13.60 +/- 2.31 Hz     PASS
  bitter       (LB1a-e)            0.00 +/- 0.00 Hz
  sugar + bitter                   0.00 +/- 0.00 Hz     PASS (suppressed)
```

**Shuffle control.** Rewiring the connectome at random while preserving every
neuron's in-degree, out-degree, synapse counts and transmitter signs takes MN9
to exactly 0.00 Hz, across three seeds, while the network keeps firing. So the
result is a property of the wiring rather than of the simulator. Reported
honestly: the shuffle also cuts total network activity about ninefold, so the
null is not perfectly matched. `NOTES.md` has the numbers.

`tests/test_male_cns.py` asserts all of this.

### Two results that do not look right

Both are left visible rather than tuned away.

- **`HIGH_SALT_GRN` (LB3d) drives MN9 harder than sugar does**, at 16.8 Hz.
  It should be aversive. The cause is in the data: LB3d shares its strongest
  targets with the sugar GRNs, and MaleCNS predicts all 26 LB3d neurons to be
  **cholinergic** (confidence 0.71, no ground truth) where the literature calls
  them glutamatergic and therefore inhibitory. The model inherits the release's
  prediction. This is what a load-bearing sign assumption looks like when it
  breaks.
- **A handful of VNC motor neurons saturate** near the 455 Hz ceiling the
  refractory period imposes (`MN11D` 342 Hz, `MNx01` 289 Hz). Shiu et al.
  modelled the brain only; MaleCNS adds the ventral nerve cord and its extra
  recurrent loops run away. Network-wide the model is still sparse (~0.5 Hz
  mean), so this is local, but do not trust VNC motor rates.

Also: `MN9_R` is flagged `RT Hard to trace` in the release and responds about
20× more weakly than `MN9_L` to the same drive. That is a reconstruction
artefact. Read the two sides separately; do not average them.

### Ports

Inputs — each is a **level** in Hz that persists across steps until changed,
not an event. Zero means silent, and nothing needs staging before `step()`.

| Port | MaleCNS types | n | Identified by |
|---|---|---|---|
| `sensory.SUGAR_GRN` | LB3b, LB3c | 34 | Gr64f-GAL4 projection match |
| `sensory.BITTER_GRN` | LB1a–LB1e | 57 | Gr33a-GAL4; LB1e clusters with them |
| `sensory.WATER_GRN` | LB3a | 17 | ppk28-GAL4 |
| `sensory.HIGH_SALT_GRN` | LB3d | 26 | Ir7c-/ppk23-GAL4 — see the warning above |
| `sensory.T4T5_DRIVE` | T4a–d, T5a–d | ~13,700 | opt-in only, see coupling below |

6 of the 57 bitter GRNs (all of LB1b) have an `unclear` transmitter prediction
and so have no modelled output; the effective population is 51.

Outputs — the modelled firing rate of one named neuron, in Hz, exponentially
smoothed over `readout_tau` (0.1 s by default; a single neuron emits 0 or 1
spikes per 1/60 s frame, so the raw value is unusable). These are **read-outs,
not commands**: unlike `motor.YAW` there is no gain, no clipping and no sign
convention, because the mapping from a descending neuron's rate to a body
movement is not in the connectome.

| Port | Neuron |
|---|---|
| `motor.MN9_L` / `MN9_R` | MN9, the rostrum protractor — proboscis extension |
| `motor.DNa02_L` / `DNa02_R` | DNa02, associated with ipsilateral turning |
| `motor.MDN_L` / `MDN_R` | moonwalker DN, associated with backward walking |
| `motor.DNp09_L` / `DNp09_R` | DNp09, associated with stopping and freezing |
| `motor.HSE_L` / `HSE_R` | HSE, an equatorial HS tangential cell |

Absolute rates are not calibrated against recordings. Differences between
conditions are the meaningful part.

Worth noting: **HS cells are real neurons in MaleCNS** — HSE, HSN and HSS, two
of each, with their measured wiring — whereas `optic_lobe` has to
[model the HS read-out](#why-this-sdk-models-hs-cells-rather-than-simulating-them)
because flyvis stops at the columnar types. Two further caveats on those ports:
in the animal HS cells are **graded, largely non-spiking** neurons, and this LIF
model forces them to spike; and they only mean anything if something is driving
`T4T5_DRIVE`.

### Cost: this circuit does not run at 30 fps

Measured on 4 vCPU of a 2.10 GHz Xeon, single-threaded in the hot loop:

| condition | wall time per second of simulated time |
|---|---|
| nothing driven | 3.3 s |
| water GRNs driven | 5.0 s |
| bitter GRNs driven | 5.7 s |
| sugar + bitter | 7.8 s |
| sugar GRNs driven | 8.9 s |
| high-salt GRNs driven | 11.1 s |
| coupled to `optic_lobe` (below) | 12.0 s |

So a 1/60 s frame costs 55–200 ms and a 1/30 s frame costs 110–400 ms:
**3–9 fps**, against `optic_lobe`'s ~30 ms/step. Cost rises with how much of
the network is firing.

This is the reference model's cost, not an implementation defect. Shiu et al.
integrate at 0.1 ms, so one 1/60 s frame is 167 internal steps over all 164,587
neurons. Raising `dt` would be a different model. `step(dt)` accepts the same
`dt` as any other circuit and is callable at the same cadence; it simply takes
longer than real time to return. Nothing uses the GPU: the inner loop is a
sparse gather-scatter plus a few dense passes, so it is memory-bandwidth bound.

Memory: about 1.2 GB resident, from a 197 MB cached sparse matrix plus the
delay buffer. A `min_synapses=5` build cuts 24.5 M connections to ~6.2 M and
would be much cheaper, but **the sanity checks have only been run at
`min_synapses=1`**, so it is not offered as a validated mode.

### Configuration

```python
from flybrain import FlyBrain
from flybrain.circuits import MaleCNSConfig

fly = FlyBrain(
    circuits=["male_cns"],
    configs={"male_cns": MaleCNSConfig(seed=3, readout_tau=0.05)},
)
```

Fields: `cache_root`, `min_synapses`, `seed`, `readout_tau`, `lif`,
`enable_visual_coupling`. The model is stochastic — the Poisson drive is seeded
— so two runs with the same `seed` match and two with different seeds do not.

---

## Coupling the two circuits: a hypothesis, not a result

`FlyBrain(circuits=["optic_lobe", "male_cns"])` works and is supported. It runs
both circuits independently. It does **not** connect them, and that is
deliberate.

There is no principled mapping between flyvis cell types and MaleCNS neurons.
Different animals, different EM volumes, different coordinate frames, different
sexes. `flybrain/circuits/coupling.py` implements one anyway, because the two
models are complementary in one specific place — flyvis has T4/T5 cells whose
dynamics were fitted to recordings but no HS cells; MaleCNS has real HS cells
but no fitted dynamics — and it is wrapped so it cannot be invoked by accident:

```python
from flybrain.circuits import MaleCNSCircuit, MaleCNSConfig, OpticLobeCircuit
from flybrain.circuits.coupling import OpticLobeToMaleCNS

eye = OpticLobeCircuit()
cns = MaleCNSCircuit(MaleCNSConfig(enable_visual_coupling=True))   # opt in
bridge = OpticLobeToMaleCNS(eye, cns)

eye.set_input(sensory.VISUAL_FIELD, frame)
eye.step(dt)
bridge.transfer()        # explicit, every frame; nothing calls it for you
cns.step(dt)
```

What it assumes, worst first: **retinotopy is discarded** (flyvis's 721 columns
per subtype are pooled to one number and every MaleCNS cell of that subtype is
driven equally, so nothing downstream can be selective for *where* motion is);
**flyvis activity is not a firing rate** (it is a dimensionless model membrane
potential, and the gain converting it to Hz is invented); **subtype letters are
assumed to mean the same thing in both datasets**; **chirality is assumed**; and
negative deviations are half-wave rectified.

What it actually produces, 30 frames of a drifting panorama, 3 MaleCNS seeds:

```
                                 optic_lobe YAW   T4/T5 drive   MaleCNS HSE_L - HSE_R
  scene drifts right (yaw left)        +0.117       52.5 Hz            +30.3 Hz
  scene drifts left  (yaw right)       -0.118       56.4 Hz            -27.0 Hz
  static scene                         +0.024       55.0 Hz             -4.8 Hz
```

The real MaleCNS HS cells do inherit a direction-dependent left/right
imbalance, with the sign tracking the optic lobe's own, and it replicates
across seeds. **But look at the middle column**: a static scene stages almost
exactly as much T4/T5 drive as a moving one, because the drive comes from
flyvis activity relative to a *grey* screen and a static textured scene is
already a large deviation. The bridge is therefore not a motion signal; the
motion is only in its left/right imbalance, riding on a large
motion-independent pedestal, and the HS cells are saturated at ~300 Hz.

`tests/test_coupling.py` asserts the mirror-reversal and also asserts the
static-scene pedestal, so that the flaw stays visible.

Making this real would need a column-to-cell registration between the flyvis
lattice and the MaleCNS optic lobe, and a calibration of flyvis units against
measured T4/T5 rates. Neither exists.

---

## Adding a circuit

1. Declare ports in `flybrain/io/sensory.py` / `motor.py`.
2. Subclass `Circuit` in `flybrain/circuits/`, implement `inputs`, `outputs`,
   `reset`, `set_input`, `step`, `get_output`, and decorate with
   `@register_circuit("name")`.
3. Import it in `flybrain/circuits/__init__.py`.
4. Add tests that skip cleanly when your model's data is not downloaded, as
   `tests/test_optic_lobe.py` and `tests/test_male_cns.py` both do.

`FlyBrain(circuits=[...])` picks it up; `brain.py` does not change, and it did
not change when `male_cns` was added either.

Keep every heavy import inside the circuit module: `optic_lobe.py` is the only
module that imports flyvis, and `male_cns.py` is the only one that imports
`flybrain.malecns`. That boundary is the point of the SDK. Both do it lazily,
inside the constructor, so `import flybrain` pulls in neither torch nor scipy.

If your circuit is built on data rather than on fitted parameters, say in its
docstring and in this file which parts are **measured** and which are
**assumed**, the way the `male_cns` section does. Do not let a convention
graduate into a fact by being written down without a source.
