"""Optic lobe circuit: a pretrained, connectome-constrained fly visual system.

This module is the *only* place in the SDK that knows flyvis exists. It wraps
the pretrained networks of Lappalainen et al. (2024), "Connectome-constrained
networks predict neural activity across the fly visual system" (Nature), as
published in the `flyvis <https://github.com/TuragaLab/flyvis>`_ package.

What the underlying model is
----------------------------
flyvis simulates 65 columnar cell types of the *Drosophila* optic lobe
(R1-R8 photoreceptors -> lamina -> medulla -> T4/T5 motion detectors and the
Tm/TmY output types), 45,669 cells in total, wired according to the FIB-25 /
FIB-19 connectome. Parameters are shared per cell type and per synapse type, so
a network has only ~700 free parameters; the pretrained ensemble was optimised
to predict optic flow from natural movies.

Why there is an HS *model* here rather than an HS *cell*
--------------------------------------------------------
flyvis stops at the columnar types. HS (horizontal system) cells are large-field
tangential cells of the lobula plate and are **not** in the flyvis connectome.
They are, however, well characterised as spatial integrators of exactly the
signals flyvis does produce: each HS cell sums the output of the horizontally
tuned T4/T5 columns over one half of the visual field. This module implements
that integration step, so the HS output is a documented linear read-out of real
pretrained activity rather than a second learned model.

Direction tuning is not guessed. flyvis ships the measured preferred directions
in ``flyvis.utils.groundtruth_utils.preferred_directions``:

===========  =================  ========
subtype      preferred motion   contrast
===========  =================  ========
T4a / T5a    180 deg (leftward) ON / OFF
T4b / T5b    0 deg (rightward)  ON / OFF
T4c / T5c    90 deg (upward)    ON / OFF
T4d / T5d    270 deg (downward) ON / OFF
===========  =================  ========

so the horizontal pair is (a, b) and the vertical pair (c, d) is ignored here.
``scripts/verify_sign.py`` re-measures this from the loaded weights.

Two optic lobes from one network
--------------------------------
A fly has two mirror-symmetric optic lobes; flyvis provides one. Its cell types
are trained independently, so T4a and T4b are *not* exact mirror images of one
another, and a single lattice split down the middle gives a left/right readout
with a built-in bias: a perfectly mirror-symmetric scene produces HS_LEFT !=
HS_RIGHT.

This wrapper therefore evaluates the network on a batch of two: the frame as
seen by the right eye, and the horizontally mirrored frame as seen by the left
eye. Both eyes are read out with the *identical* formula, so HS_RIGHT - HS_LEFT
is exactly antisymmetric under mirroring the input and is guaranteed to be zero
for any mirror-symmetric scene. Batching keeps the cost of the second lobe
close to free.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..io import motor, sensory
from ..ports import Port
from .base import Circuit, register_circuit

logger = logging.getLogger(__name__)

__all__ = [
    "OpticLobeCircuit",
    "OpticLobeConfig",
    "RIGHTWARD_TYPES",
    "LEFTWARD_TYPES",
]


# --------------------------------------------------------------------------
# Cell-type constants (from flyvis ground truth, see module docstring)
# --------------------------------------------------------------------------

#: T4/T5 subtypes whose preferred direction is 0 deg, i.e. rightward on the image.
RIGHTWARD_TYPES: Tuple[str, ...] = ("T4b", "T5b")

#: T4/T5 subtypes whose preferred direction is 180 deg, i.e. leftward on the image.
LEFTWARD_TYPES: Tuple[str, ...] = ("T4a", "T5a")

#: Batch row of the right optic lobe (frame as-is).
RIGHT_EYE = 0
#: Batch row of the left optic lobe (horizontally mirrored frame).
LEFT_EYE = 1
#: Number of optic lobes simulated per step.
N_EYES = 2

#: Luminance weights (ITU-R BT.601) used to collapse RGB to photoreceptor drive.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class OpticLobeConfig:
    """Tunable constants for :class:`OpticLobeCircuit`.

    Attributes:
        ensemble: flyvis ensemble directory, relative to the flyvis results dir.
        member: Which ensemble member to run. ``"best"`` picks the member with
            the lowest validation loss; an ``int`` picks by index.
        checkpoint: Which training checkpoint to restore. ``"best"`` uses the
            ensemble's own best-checkpoint rule.
        device: ``"cuda"``, ``"cpu"`` or ``None`` to auto-detect.
        max_substep_dt: Upper bound on the internal Euler integration step, in
            seconds. Each :meth:`OpticLobeCircuit.step` is subdivided so that no
            internal step exceeds this. flyvis is unstable above 1/50 s.
        warmup_seconds: Duration of the grey-screen settling period used to
            build the resting state on :meth:`OpticLobeCircuit.reset`.
        grey_level: Luminance of that grey screen, in [0, 1].
        hemifield_deadzone: Columns with ``|v| <= hemifield_deadzone`` are
            excluded from both hemifield pools, so the frontal midline does not
            contribute to the left/right difference.
        yaw_gain: Scale factor from the raw HS difference to the YAW command.
        yaw_clip: YAW is clipped to +/- this value. ``None`` disables clipping.
    """

    ensemble: str = "flow/0000"
    member: Any = "best"
    checkpoint: str = "best"
    device: Optional[str] = None
    max_substep_dt: float = 1.0 / 200.0
    warmup_seconds: float = 1.0
    grey_level: float = 0.5
    hemifield_deadzone: int = 1
    #: Scales the raw HS difference into the nominal [-1, 1] YAW range. The
    #: default puts a ~150 deg/s body rotation at full scale for a textured
    #: scene; the raw signal is contrast- and distance-dependent, so retune it
    #: for your own world. scripts/verify_sign.py reports the raw magnitudes.
    yaw_gain: float = 10.0
    yaw_clip: Optional[float] = 1.0


@register_circuit("optic_lobe")
class OpticLobeCircuit(Circuit):
    """Pretrained fly optic lobe with an HS-cell steering read-out.

    Args:
        config: Tunable constants. Defaults to :class:`OpticLobeConfig`.

    Raises:
        ImportError: If flyvis is not installed.
        RuntimeError: If the pretrained weights have not been downloaded.
    """

    def __init__(self, config: Optional[OpticLobeConfig] = None) -> None:
        self.config = config or OpticLobeConfig()
        self._torch = _import_torch()
        flyvis = _import_flyvis()

        self.device = _resolve_device(self.config.device, self._torch)
        # flyvis creates tensors via the torch default device; pin it so that
        # the box filter, stimulus buffer and network all agree.
        self._torch.set_default_device(self.device)

        self._network = self._load_network(flyvis)
        self._network.eval()
        self._network.to(self.device)

        from flyvis.datasets.rendering import BoxEye

        self._eye = BoxEye()
        self._frame_size = (
            int(self._eye.min_frame_size[0]),
            int(self._eye.min_frame_size[1]),
        )

        self._connectome = self._network.connectome
        self._pool_index = self._build_hemifield_index()

        self._staged_hex: Optional[Any] = None
        self._state: Optional[Any] = None
        self._baseline: Optional[np.ndarray] = None
        self._params: Optional[Any] = None
        self._activity: Optional[np.ndarray] = None
        self._outputs: Dict[str, float] = {}
        self.reset()

    # -- construction helpers ------------------------------------------------

    def _load_network(self, flyvis: Any) -> Any:
        """Load one pretrained network out of the pretrained ensemble."""
        results_dir = flyvis.results_dir
        if not (results_dir / self.config.ensemble).exists():
            raise RuntimeError(
                f"Pretrained flyvis weights not found at "
                f"{results_dir / self.config.ensemble}.\n"
                "Download them once with:\n"
                "    flyvis download-pretrained\n"
                "and/or point FLYVIS_ROOT_DIR at the directory holding "
                "'results/'."
            )

        member = self.config.member
        if member == "best":
            ensemble = flyvis.Ensemble(self.config.ensemble)
            # Sorts members by validation loss, lowest first.
            ensemble.sort()
            name = ensemble.names[0]
            self.ensemble_size = len(ensemble)
            logger.info(
                "Loaded flyvis ensemble %s (%d members); using best member %s",
                self.config.ensemble,
                self.ensemble_size,
                name,
            )
            network_view = ensemble[name]
        else:
            name = f"{self.config.ensemble}/{int(member):03d}"
            self.ensemble_size = 1
            network_view = flyvis.NetworkView(name)

        self.network_name = name
        return network_view.init_network(checkpoint=self.config.checkpoint)

    def _build_hemifield_index(self) -> Dict[str, np.ndarray]:
        """Cell indices of each horizontal type in the outer half of the lobe.

        flyvis lays the retina out on hexagonal (u, v) axial coordinates.
        ``flyvis.utils.hex_utils.hex_to_pixel`` maps them to pixels with
        ``x = 3/2 * v``, so **v is the azimuth axis**: v > 0 is the right half
        of the image, v < 0 the left half. Columns within
        ``hemifield_deadzone`` of the midline are dropped so the frontal
        midline does not contribute.

        Only the right half is needed: the left HS cell is obtained by running
        this same pooling on the mirrored frame (see the module docstring), not
        by pooling the other half of the lattice.
        """
        v_all = np.asarray(self._connectome.nodes.v[:]).reshape(-1)
        dead = self.config.hemifield_deadzone

        index: Dict[str, np.ndarray] = {}
        for cell_type in RIGHTWARD_TYPES + LEFTWARD_TYPES:
            cells = np.asarray(
                self._connectome.nodes.layer_index[cell_type][:]
            ).reshape(-1)
            index[cell_type] = cells[v_all[cells] > dead]
        return index

    # -- Circuit interface ---------------------------------------------------

    @property
    def inputs(self) -> Tuple[Port, ...]:
        return (sensory.VISUAL_FIELD,)

    @property
    def outputs(self) -> Tuple[Port, ...]:
        return (motor.YAW, motor.HS_LEFT, motor.HS_RIGHT, motor.FLOW_ASYMMETRY)

    def input_shape(self, port: Port) -> Tuple[int, ...]:
        if port is sensory.VISUAL_FIELD or port.name == sensory.VISUAL_FIELD.name:
            return self._frame_size
        raise KeyError(port)

    def reset(self) -> None:
        """Rebuild the resting state by showing the model a grey screen."""
        torch = self._torch
        net = self._network
        with torch.no_grad():
            net.clamp()
            self._params = net._param_api()
            self._state = net.steady_state(
                t_pre=self.config.warmup_seconds,
                dt=self.config.max_substep_dt,
                batch_size=N_EYES,
                value=self.config.grey_level,
            )
            # Resting activity of every cell on a grey screen. T4/T5 subtypes
            # have different resting potentials, so the HS pools are read out
            # as deviations from rest rather than as raw activity.
            self._baseline = self._state.nodes.activity.detach().cpu().numpy()
        self._staged_hex = None
        self._activity = None
        self._outputs = {p.name: 0.0 for p in self.outputs}

    def set_input(self, port: Port, value: Any) -> None:
        if port.name != sensory.VISUAL_FIELD.name:
            raise KeyError(f"{self.name} has no input port {port.name!r}")
        self._staged_hex = self._frame_to_hex(value)

    def step(self, dt: float) -> None:
        """Integrate the network for ``dt`` seconds holding the frame constant.

        Args:
            dt: Wall-clock duration of the frame, in seconds.

        Raises:
            RuntimeError: If no frame has been staged since the last reset.
        """
        if self._staged_hex is None:
            raise RuntimeError(
                "No VISUAL_FIELD input staged; call set_input() before step()."
            )
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        torch = self._torch
        net = self._network
        n_sub = max(1, int(np.ceil(dt / self.config.max_substep_dt)))
        sub_dt = dt / n_sub

        with torch.no_grad():
            # Expand the hex frame onto the photoreceptor cells of the full
            # 45,669-cell state vector.
            net.stimulus.zero(N_EYES, 1)
            net.stimulus.add_input(self._staged_hex)
            x_t = net.stimulus()[:, 0]

            state = self._state
            for _ in range(n_sub):
                state = net._next_state(self._params, state, x_t, sub_dt)
            self._state = state
            self._activity = state.nodes.activity.detach().cpu().numpy()

        self._outputs = self._read_out(self._activity)

    def get_output(self, port: Port) -> float:
        try:
            return self._outputs[port.name]
        except KeyError:
            raise KeyError(f"{self.name} has no output port {port.name!r}") from None

    # -- internals -----------------------------------------------------------

    def _frame_to_hex(self, frame: Any) -> Any:
        """Convert an RGB/greyscale frame to a (1, 1, 1, 721) hex-lattice tensor."""
        torch = self._torch
        arr = np.asarray(frame)

        if arr.ndim == 3:
            if arr.shape[-1] == 3:
                arr = arr.astype(np.float32) @ _LUMA
            elif arr.shape[-1] == 4:
                arr = arr[..., :3].astype(np.float32) @ _LUMA
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError(
                    f"VISUAL_FIELD expects 1, 3 or 4 channels, got shape {arr.shape}"
                )
        elif arr.ndim != 2:
            raise ValueError(
                f"VISUAL_FIELD expects (H, W) or (H, W, C), got shape {arr.shape}"
            )

        arr = arr.astype(np.float32)
        # uint8 images, and float images that were never normalised, arrive in
        # [0, 255]; the photoreceptors want [0, 1].
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        # Eye 0 is the right lobe (frame as-is); eye 1 is the left lobe, which
        # sees the mirrored world and is read out with the same formula.
        stereo = np.stack([arr, arr[:, ::-1]])
        movie = torch.tensor(
            np.ascontiguousarray(stereo), dtype=torch.float32, device=self.device
        )[:, None]
        with torch.no_grad():
            return self._eye(movie)

    def _read_out(self, activity: np.ndarray) -> Dict[str, float]:
        """Integrate T4/T5 columns into HS-like cells and a steering command.

        Args:
            activity: Activity of shape ``(N_EYES, n_cells)``; row 0 is the
                right lobe, row 1 the left lobe viewing the mirrored frame.
        """
        activity = activity - self._baseline

        def front_to_back(eye: int) -> float:
            """Front-to-back minus back-to-front motion, outer half of one lobe.

            Front-to-back on the outer (temporal) half of a lobe is rightward
            image motion in that lobe's own frame of reference. Applying this
            to the mirrored frame yields the contralateral HS cell.
            """
            row = activity[eye]

            def pool(cell_types: Sequence[str]) -> float:
                return float(
                    np.mean([
                        row[self._pool_index[ct]].mean()
                        for ct in cell_types
                    ])
                )

            return pool(RIGHTWARD_TYPES) - pool(LEFTWARD_TYPES)

        def motion_energy(eye: int) -> float:
            """Non-directional horizontal motion magnitude in one lobe.

            Sums how hard the horizontal T4/T5 columns are driven, ignoring
            which way they say the scene is moving. Nearer surfaces sweep past
            faster and so raise this, which is what makes the left/right
            difference a proximity cue rather than a rotation cue.
            """
            row = activity[eye]
            return float(
                np.mean([
                    np.abs(row[self._pool_index[ct]]).mean()
                    for ct in RIGHTWARD_TYPES + LEFTWARD_TYPES
                ])
            )

        hs_right = front_to_back(RIGHT_EYE)
        hs_left = front_to_back(LEFT_EYE)

        raw_yaw = hs_right - hs_left
        yaw = raw_yaw * self.config.yaw_gain
        if self.config.yaw_clip is not None:
            yaw = float(np.clip(yaw, -self.config.yaw_clip, self.config.yaw_clip))

        # Speed balance: how much horizontal motion energy each side carries,
        # regardless of its direction. Zero under pure rotation (both sides
        # sweep equally fast), positive when the right side is nearer.
        flow_asymmetry = motion_energy(RIGHT_EYE) - motion_energy(LEFT_EYE)

        return {
            motor.YAW.name: float(yaw),
            motor.HS_LEFT.name: float(hs_left),
            motor.HS_RIGHT.name: float(hs_right),
            motor.FLOW_ASYMMETRY.name: float(flow_asymmetry),
        }

    # -- introspection -------------------------------------------------------

    def type_activity(self, cell_type: str) -> np.ndarray:
        """Mean activity of one flyvis cell type in each lobe.

        Reported as a deviation from the grey-screen resting state, in the
        same arbitrary flyvis units as HS_LEFT and HS_RIGHT, pooled over every
        column of the type. Purely a read-out: it changes nothing and is not
        used by YAW.

        Args:
            cell_type: A flyvis cell type, e.g. ``"T4b"`` or ``"Mi1"``.

        Returns:
            Array of shape ``(2,)``: ``[RIGHT_EYE, LEFT_EYE]``, where the left
            entry is this lobe's response to the mirrored frame.

        Raises:
            KeyError: If the type is not in the flyvis connectome.
            RuntimeError: If :meth:`step` has not been called since the last
                reset, so there is no activity to report.
        """
        if self._activity is None:
            raise RuntimeError(
                "No activity yet; call set_input() and step() before "
                "type_activity()."
            )
        try:
            cells = np.asarray(
                self._connectome.nodes.layer_index[cell_type][:]
            ).reshape(-1)
        except Exception as exc:
            raise KeyError(
                f"{cell_type!r} is not a flyvis cell type. flyvis models 65 "
                "columnar types of the optic lobe; HS and other lobula-plate "
                "tangential cells are not among them."
            ) from exc
        deviation = self._activity - self._baseline
        return np.array(
            [float(deviation[eye][cells].mean()) for eye in (RIGHT_EYE, LEFT_EYE)]
        )

    @property
    def activity(self) -> Optional[np.ndarray]:
        """Raw activity, shape (2, 45669), after the last step (read-only)."""
        return self._activity

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"<OpticLobeCircuit network={getattr(self, 'network_name', '?')} "
            f"device={self.device}>"
        )


# --------------------------------------------------------------------------
# import helpers - keep the hard dependency lazy and the error message useful
# --------------------------------------------------------------------------


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "flybrain's optic_lobe circuit needs PyTorch. Install it from "
            "https://pytorch.org (CPU or CUDA build both work)."
        ) from exc
    return torch


def _import_flyvis() -> Any:
    try:
        import flyvis
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "flybrain's optic_lobe circuit needs flyvis. Install it with "
            "'pip install flyvis', then download the pretrained weights with "
            "'flyvis download-pretrained'."
        ) from exc
    return flyvis


def _resolve_device(requested: Optional[str], torch: Any) -> Any:
    """Pick a torch device, honouring FLYBRAIN_DEVICE then CUDA availability."""
    name = requested or os.environ.get("FLYBRAIN_DEVICE")
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
