# flybrain port reference

This is the contract between the fly neuroscience and whatever is embodying it.
If you are picking this SDK up in a later session, read this file and you should
not need to relearn the biology or reread the flyvis source.

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

## Ports

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

## Performance

~30 ms per step on 2 CPU cores at 391×391 with both lobes, so roughly 30 fps
without a GPU. Set `FLYBRAIN_DEVICE=cuda` (or `OpticLobeConfig(device="cuda")`)
to use a GPU; CPU is the automatic fallback. The network is small — 734 free
parameters over 45,669 cells — so CUDA helps less than you might expect.

---

## Configuration

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

## Adding a circuit (stage 2 and beyond)

1. Declare ports in `flybrain/io/sensory.py` / `motor.py`.
2. Subclass `Circuit` in `flybrain/circuits/`, implement `inputs`, `outputs`,
   `reset`, `set_input`, `step`, `get_output`, and decorate with
   `@register_circuit("name")`.
3. Import it in `flybrain/circuits/__init__.py`.

`FlyBrain(circuits=[...])` picks it up; `brain.py` does not change. Keep every
flyvis/torch import inside the circuit module, as `optic_lobe.py` does — that
boundary is the point of the SDK.
