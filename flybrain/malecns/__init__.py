"""The MaleCNS loader: connectome caching plus LIF dynamics.

This subpackage stands in the same relation to
:mod:`flybrain.circuits.male_cns` as the external ``flyvis`` package does to
:mod:`flybrain.circuits.optic_lobe`: it is the model, and exactly one circuit
module imports it. Nothing else in the SDK touches it.

Build the cache once, 566 MB, before the circuit can be used::

    python -m flybrain.malecns download

:mod:`~flybrain.malecns.dataset` documents where the data comes from and,
importantly, which parts of it are measured and which are assumed.
"""

from .dataset import (
    BUCKET_URL,
    NT_SIGN,
    CacheConfig,
    Connectome,
    build_cache,
    cache_available,
    default_cache_root,
    load_cache,
    shuffle_preserving_degree,
)
from .lif import LIFNetwork, LIFParams, run_trials

__all__ = [
    "BUCKET_URL",
    "NT_SIGN",
    "CacheConfig",
    "Connectome",
    "LIFNetwork",
    "LIFParams",
    "build_cache",
    "cache_available",
    "default_cache_root",
    "load_cache",
    "run_trials",
    "shuffle_preserving_degree",
]
