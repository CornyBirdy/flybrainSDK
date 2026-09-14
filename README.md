# flybrain

A connectome-constrained fly brain SDK. It exposes real fly neural circuits as
**typed sensory inputs and motor outputs**, so a project can embody the fly
brain in a car, a drone, a game or a robot without touching the neuroscience.

Stage 1 (this release) ships the **optic-flow steering circuit**: the pretrained
[flyvis](https://github.com/TuragaLab/flyvis) model of the *Drosophila* optic
lobe, with an HS-cell read-out that turns visual motion into a steering command.

```python
from flybrain import FlyBrain, sensory, motor

fly = FlyBrain(circuits=["optic_lobe"])
fly.set_input(sensory.VISUAL_FIELD, frame)   # numpy RGB image
fly.step()
yaw = fly.get_output(motor.YAW)              # + = steer right, - = steer left
```

**[Read `flybrain/README.md`](flybrain/README.md)** for the full port reference,
the sign conventions, and what the circuit does and does not stabilise.

---

## What is actually running

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
  io/sensory.py         VISUAL_FIELD
  io/motor.py           YAW, HS_LEFT, HS_RIGHT, FLOW_ASYMMETRY
  circuits/base.py      Circuit interface + registry
  circuits/optic_lobe.py  the only module that imports flyvis
  README.md             port reference and neuroscience notes
demo/
  corridor.py           corridor renderer + car kinematics (no flybrain import)
  run_demo.py           the closed loop; uses only the public API
scripts/verify_sign.py  re-measures the sign convention from the weights
tests/                  container tests (fast) + circuit tests (need weights)
```

The boundary is deliberate: **the demo never reaches into flyvis**, and the
circuit never knows what is embodying it.

---

## Tests

```bash
pytest tests/ -q
```

Container tests run anywhere. Circuit tests need the pretrained weights and skip
cleanly without them. They assert the properties downstream code depends on: the
YAW sign convention, exact mirror antisymmetry, zero command on a symmetric
scene, and agreement between uint8/float and RGB/greyscale inputs.

```bash
python scripts/verify_sign.py     # re-derive the sign convention from scratch
```

---

## Roadmap

Stage 1 is the optic-flow steering circuit. Later stages add looming/escape and
central-complex heading control as further circuits behind the same interface —
`FlyBrain(circuits=["optic_lobe", "looming"])`, new ports, same three calls.
See the last section of [`flybrain/README.md`](flybrain/README.md) for how to add
one.

## Credits

The pretrained model is the work of the Turaga lab: Lappalainen, J.K., Tschopp,
F.D., Prakhya, S., et al. "Connectome-constrained networks predict neural
activity across the fly visual system." *Nature* 634, 1132–1140 (2024).
This SDK is a wrapper around it and claims none of that work.
