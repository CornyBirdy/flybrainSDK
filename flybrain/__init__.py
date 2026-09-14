"""flybrain - a connectome-constrained fly brain SDK.

Exposes fly neural circuits as typed sensory inputs and motor outputs, so that
the same neuroscience can be embodied in a car, a drone, a game or a robot
without the host application touching the models underneath.

Stage 1 ships ``optic_lobe``: the pretrained flyvis optic lobe with an
HS-cell steering read-out.

    from flybrain import FlyBrain, sensory, motor

    fly = FlyBrain(circuits=["optic_lobe"])
    fly.set_input(sensory.VISUAL_FIELD, frame)   # numpy RGB image
    fly.step()
    yaw = fly.get_output(motor.YAW)              # + = steer right

See ``flybrain/README.md`` for the port reference and sign conventions.
"""

from .brain import FlyBrain
from .circuits import available_circuits, register_circuit
from .io import motor, sensory
from .ports import Port

__version__ = "0.1.0"

__all__ = [
    "FlyBrain",
    "sensory",
    "motor",
    "Port",
    "available_circuits",
    "register_circuit",
    "__version__",
]
