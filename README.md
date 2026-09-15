# flybrain

A connectome-constrained fly brain SDK. It exposes real fly neural circuits as
**typed sensory inputs and motor outputs**, so a project can embody the fly
brain in a car, a drone, a game or a robot without touching the neuroscience.

Two circuits ship, and they can be loaded together:

- **`optic_lobe`** — the pretrained [flyvis](https://github.com/TuragaLab/flyvis)
  model of the *Drosophila* optic lobe, with an HS-cell read-out that turns
  visual motion into a steering command.
- **`male_cns`** — the **MaleCNS v1.0** connectome (164,587 neurons,
  124M synapses, brain + optic lobes + ventral nerve cord) as a leaky
  integrate-and-fire model, exposing gustatory drive and descending/motor
  neuron firing rates.

```python
from flybrain import FlyBrain, sensory, motor

fly = FlyBrain(circuits=["optic_lobe"])
fly.set_input(sensory.VISUAL_FIELD, frame)   # numpy RGB image
fly.step()
yaw = fly.get_output(motor.YAW)              # + = steer right, - = steer left
```

```python
fly = FlyBrain(circuits=["optic_lobe", "male_cns"])
fly.set_input(sensory.SUGAR_GRN, 150.0)      # Hz, like an optogenetic drive
fly.step()
mn9 = fly.get_output(motor.MN9_L)            # Hz, proboscis extension
```

**[Read `flybrain/README.md`](flybrain/README.md)** for the full port reference,
the sign conventions, what each circuit does and does not stabilise, and — for
`male_cns` especially — which of its numbers are measured and which are assumed.

---

## What is actually running

### `optic_lobe`

Real pretrained weights, no stand-in vision model:

- **flyvis** (Lappalainen et al., *Nature* 2024, "Connectome-constrained networks
  predict neural activity across the fly visual system") — 65 columnar cell
  types, 45,669 cells, wired from the FIB-25/FIB-19 connectome. Photoreceptors
  R1–R8 → lamina → medulla → T4/T5 motion detectors.
- The published 50-network ensemble, trained to predict optic flow from natural
  movies. Nothing here is trained or fine-tuned; the SDK sorts the ensemble by
  validation loss and runs the best member.
- HS (horizontal system) cells are **not** in flyvis — they are lobula-plate
  tangential cells downstream of it. This SDK adds the one documented step they
  perform: pooling the horizontally-tuned T4/T5 columns over each half of the
  visual field. Direction tuning comes from flyvis's own published table, not
  from assumption.

### `male_cns`

Real measured wiring, no generated graph — but, unlike `optic_lobe`, **the
dynamics on top of that wiring are assigned by convention rather than fitted
to recordings.** That difference is the most important thing to know about this
circuit, so it gets its own table below.

- **MaleCNS v1.0** (HHMI Janelia FlyEM / Google Research / Cambridge
  Connectomics; data public June 2026, paper in *Cell* 3 September 2026;
  CC-BY). 164,587 neurons, 25,563,197 connections, 124,025,046 synapses,
  covering brain, optic lobes and ventral nerve cord.
- A **leaky integrate-and-fire** model in the style of Shiu et al.,
  *Nature* 634, 210–219 (2024): a spike shifts the downstream membrane
  potential in proportion to the number of synapses between the two cells,
  with the sign set by the presynaptic neuron's predicted transmitter.
- Data comes from the public bucket, not the token-gated neuPrint API, so it
  installs unattended. **There is no synthetic fallback**: if the download
  fails or the cache is missing, the circuit raises and tells you how to get
  the data.

#### Measured vs assumed

| Measured, from the release | Assumed, by this SDK |
|---|---|
| which neuron contacts which neuron | synapse count ∝ synaptic strength — the connectome does not measure strength |
| how many synapses each connection has | every synapse is worth **0.275 mV**, a free parameter Shiu et al. fitted, not a measurement |
| a transmitter prediction per neuron (ground truth for ~85,000) | that **glutamate is inhibitory** (GluCl-α; Liu & Wilson 2013) — opposite to the vertebrate convention, and it flips 29,296 of 164,587 neurons |
| cell type, class, soma side, entry nerve | that monoaminergic and unclassified neurons have **no output** (drops 1.0M of 25.6M connections; no neuromodulation at all) |
| | one set of membrane constants for all 164,587 neurons |
| | no gap junctions, no dendritic processing, zero basal firing rate |

The release also carries **no taste-modality annotation whatsoever**. Every
gustatory port's cell types come from the literature, cited in
`flybrain/io/sensory.py`: `SUGAR_GRN` is types LB3b and LB3c because Tastekin
et al. matched both to the Gr64f-GAL4 projection pattern, not because MaleCNS
says "sugar".

#### The sanity check

Activating the sweet-sensing labellar GRNs should extend the proboscis, and
bitter should stop it. That is the standard test for this class of model.

```
MN9_L firing rate, 5 trials x 1 s, GRNs driven at 150 Hz

  baseline (nothing driven)        0.00 +/- 0.00 Hz
  sugar        (LB3b, LB3c)       13.60 +/- 2.31 Hz     PASS
  sugar + bitter (LB1a-e)          0.00 +/- 0.00 Hz     PASS (suppressed)
```

Rewiring the connectome at random while preserving every neuron's degree,
synapse counts and transmitter signs takes MN9 to exactly 0.00 Hz across three
seeds, while the network keeps firing — so the result comes from the wiring,
not from the simulator. The shuffle also cuts total activity about ninefold, so
the null is not perfectly matched; `NOTES.md` says so and gives the numbers.

**Two results that do not look right**, left visible rather than tuned away:
`HIGH_SALT_GRN` drives MN9 *harder* than sugar, because MaleCNS predicts those
neurons cholinergic where the literature calls them glutamatergic; and a few
ventral-nerve-cord motor neurons saturate. Both are documented in
[`flybrain/README.md`](flybrain/README.md#two-results-that-do-not-look-right).

#### Cost: `male_cns` does not run at 30 fps

Measured on 4 vCPU of a 2.10 GHz Xeon:

| condition | wall time per second of simulated time |
|---|---|
| nothing driven | 3.3 s |
| sugar GRNs driven | 8.9 s |
| high-salt GRNs driven | 11.1 s |
| coupled to `optic_lobe` | 12.0 s |

A 1/60 s frame costs **55–200 ms**: 3–9 fps, against `optic_lobe`'s ~30 ms/step.
This is the reference model's cost, not an implementation defect — Shiu et al.
integrate at 0.1 ms, so one frame is 167 internal steps over 164,587 neurons.
`step(dt)` is callable at the same cadence as any other circuit; it just takes
longer than real time to return. About 1.2 GB resident. Nothing uses the GPU.

---

## Install

Requires Python 3.9+ and PyTorch (CPU or CUDA).

```bash
python -m venv .venv && source .venv/bin/activate

# PyTorch first, matching your hardware - see https://pytorch.org
pip install torch torchvision            # or the CUDA index-url for a GPU

pip install -e .                         # installs flybrain + flyvis
flyvis download-pretrained               # ~3 MB of pretrained weights, once
```

For the `male_cns` circuit, additionally:

```bash
pip install -e ".[male_cns]"             # pandas, pyarrow, scipy
python -m flybrain.malecns download      # 566 MB of connectome, once
```

The MaleCNS tables are public and CC-BY; no account and no token. The cache
lands in `~/.cache/flybrain/malecns` unless `FLYBRAIN_MALECNS_DIR` says
otherwise, and is never re-downloaded on import.

`flyvis download-pretrained` unpacks into flyvis's data directory. To keep the
weights somewhere else, set `FLYVIS_ROOT_DIR` before both the download and any
run.

<details>
<summary>If <code>antlr4-python3-runtime</code> fails to build</summary>

An omegaconf dependency ships a legacy `setup.py` that trips over Debian-patched
distutils. Building inside a clean virtualenv fixes it; if it still fails:

```bash
SETUPTOOLS_USE_DISTUTILS=local pip install "antlr4-python3-runtime==4.9.3"
```
</details>

### GPU

CUDA is used automatically when available. Force it either way with
`FLYBRAIN_DEVICE=cuda` / `=cpu`, or `OpticLobeConfig(device=...)`. The network is
small (734 free parameters), so a GPU helps less than its size suggests — about
30 ms/step on two CPU cores, i.e. real time at 30 fps.

---

## The demo

A virtual car drives down a striped corridor steered **only** by optic flow from
the fly brain. Each tick renders a first-person view, pushes it through
`set_input` / `step` / `get_output`, and turns the car by the result.

```bash
python demo/run_demo.py            # writes demo/output/
python demo/run_demo.py --live     # live matplotlib window
```

Outputs `corridor_demo.gif` (fly's view, top-down track, live telemetry) and
`corridor_telemetry.png` (closed vs open loop).

The car starts 1.2 m off-centre and is hit by rotational gusts at t = 4, 10 and
16 s. With the loop closed it recentres and rides out every gust; with the loop
open it drifts into the wall.

```
closed loop  completed   final x=+0.04 m  max|x|=1.20 m  RMS heading=8.2 deg
open loop    completed   final x=+2.68 m  max|x|=2.68 m  RMS heading=11.2 deg
```

All gains live in a constants block at the top of `demo/run_demo.py`
(`SPEED`, `HALF_WIDTH`, `MAX_TURN_RATE`, `YAW_WEIGHT`, `CENTERING_WEIGHT`,
`GUSTS`). Nothing is buried mid-function.

> The demo's steering law is `YAW - CENTERING_WEIGHT * FLOW_ASYMMETRY`. Both
> terms are optic flow from the same T4/T5 population. `YAW` alone stabilises
> *rotation* but not *position* — a real property of HS-driven optomotor
> behaviour, explained in
> [the port reference](flybrain/README.md#the-limit-you-must-know-about-yaw-stabilises-rotation-not-position).

---

## Layout

```
flybrain/
  brain.py              FlyBrain: set_input / step / get_output, port routing
  ports.py              Port, the typed descriptor
  io/sensory.py         VISUAL_FIELD, SUGAR_GRN, BITTER_GRN, WATER_GRN, ...
  io/motor.py           YAW, HS_*, FLOW_ASYMMETRY, MN9_*, DNa02_*, MDN_*, ...
  circuits/base.py      Circuit interface + registry
  circuits/optic_lobe.py  the only module that imports flyvis
  circuits/male_cns.py    the only module that imports the MaleCNS loader
  circuits/coupling.py    opt-in flyvis -> MaleCNS bridge; a hypothesis
  malecns/dataset.py    MaleCNS download, caching, transmitter signs
  malecns/lif.py        the LIF integrator (NumPy only)
  README.md             port reference and neuroscience notes
demo/
  corridor.py           corridor renderer + car kinematics (no flybrain import)
  run_demo.py           the closed loop; uses only the public API
scripts/verify_sign.py  re-measures the sign convention from the weights
tests/                  container tests (fast) + circuit tests (need weights)
```

The boundary is deliberate: **the demo never reaches into flyvis**, and the
circuit never knows what is embodying it. Adding `male_cns` did not change
`brain.py`, `ports.py` or `base.py` by a line.

---

## Tests

```bash
pytest tests/ -q
```

Container tests run anywhere. Circuit tests need their model downloaded and skip
cleanly without it — the flyvis weights for `optic_lobe`, the connectome cache
for `male_cns`, both for the coupling.

`optic_lobe` tests assert the YAW sign convention, exact mirror antisymmetry,
zero command on a symmetric scene, and agreement between uint8/float and
RGB/greyscale inputs. `male_cns` tests assert the sugar → MN9 result, the
bitter suppression, the degree-preserving shuffle control, and that a silent
input produces no motor output at all.

```bash
python scripts/verify_sign.py     # re-derive the sign convention from scratch
```

---

## Roadmap

`optic_lobe` and `male_cns` ship. Later circuits — looming/escape,
central-complex heading control — go behind the same interface: new ports, same
three calls. See the last section of
[`flybrain/README.md`](flybrain/README.md) for how to add one.

The honest open problem is **coupling**. `flybrain/circuits/coupling.py`
bridges flyvis T4/T5 activity into MaleCNS T4/T5 cells, which is interesting
because MaleCNS has the real HS cells that flyvis lacks. It is opt-in,
off by default, and is **a hypothesis rather than a result**: the two models
are different animals in different coordinate frames, retinotopy is discarded
by the mapping, and the gain converting flyvis units into Hz is invented.
Under it, the real MaleCNS HS cells do show a direction-dependent left/right
imbalance that replicates across seeds — but a static scene produces almost as
much drive as a moving one, so it is not really a motion signal. Read the
module docstring before using it.

## Credits

The pretrained optic-lobe model is the work of the Turaga lab: Lappalainen,
J.K., Tschopp, F.D., Prakhya, S., et al. "Connectome-constrained networks
predict neural activity across the fly visual system." *Nature* 634, 1132–1140
(2024).

The MaleCNS v1.0 connectome is the work of HHMI Janelia FlyEM with Google
Research and the University of Cambridge, released under CC-BY. The LIF model
follows Shiu, P.K., Sterne, G.R., Spiller, N., et al. "A leaky integrate-and-
fire computational model based on the connectome of the entire adult
*Drosophila* brain reveals insights into sensorimotor processing." *Nature*
634, 210–219 (2024), whose reference implementation is at
[philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model).
Gustatory cell-type identities come from Schlegel et al., *Nature* 634, 139–152
(2024) and Tastekin et al., *Cell* (2026) / bioRxiv 2025.08.25.671814.

This SDK is a wrapper around all of that and claims none of that work.
`NOTES.md` records what was verified here and what was not.
