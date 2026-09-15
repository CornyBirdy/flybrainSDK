"""Fetching and caching the MaleCNS v1.0 connectome.

This module and :mod:`flybrain.malecns.lif` are the MaleCNS "loader" that
:mod:`flybrain.circuits.male_cns` wraps, in the same way that
:mod:`flybrain.circuits.optic_lobe` wraps flyvis. Nothing else in the SDK
imports either of them.

Where the data comes from
-------------------------
MaleCNS v1.0 (HHMI Janelia FlyEM / Google Research / Cambridge Connectomics),
licensed CC-BY. The "flat connectome" bulk tables live in a public Google
Cloud Storage bucket and need no account and no token::

    gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/
    https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/

Three of the eleven files there, 566 MB in total, are enough to build a
leaky integrate-and-fire model. The other eight are synapse coordinates and
per-T-bar predictions, which this model does not use.

The same data is served by the neuPrint REST API as dataset ``male-cns:v1.0``
(both ``neuprint.janelia.org`` and ``neuprint-cns.janelia.org``), and
``neuprint-python``, ``neuprintr``, ``navis`` and ``natverse/malecns`` all
address it. They all require a personal bearer token, so the SDK uses the
bucket instead: it installs and downloads unattended.

Measured vs assumed
-------------------
MEASURED, straight out of the release:

* which neuron contacts which neuron, and with how many synapses;
* the per-neuron neurotransmitter prediction, plus a ground-truth
  transmitter label for roughly 85,000 of the neurons.

ASSUMED by this module, and each one is a choice, not a datum:

* the sign of each neurotransmitter -- see :data:`NT_SIGN`;
* that synapse count is proportional to synaptic strength. The connectome
  does not measure strength; there is no strength column anywhere in the
  release. This is the convention of Shiu et al. (2024).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "BUCKET_URL",
    "NT_SIGN",
    "CacheConfig",
    "Connectome",
    "build_cache",
    "cache_available",
    "default_cache_root",
    "load_cache",
    "shuffle_preserving_degree",
]

#: Public, tokenless, CC-BY location of the MaleCNS v1.0 flat-connectome tables.
BUCKET_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
)

#: Cell types, classes and sides. 14.5 MB.
ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
#: One predicted transmitter per segmented body. 43.3 MB.
NEUROTRANSMITTERS_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
#: Connections between traced (proofread) bodies only, not fragments. 508 MB.
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"

#: Environment variable overriding where the cache is kept.
CACHE_ENV_VAR = "FLYBRAIN_MALECNS_DIR"

#: Sign given to each predicted neurotransmitter.
#:
#: THIS IS A MODELLING CHOICE. The connectome predicts which transmitter a
#: neuron releases; it does not measure the effect on the postsynaptic cell.
#:
#: ``acetylcholine`` +1
#:     The excitatory transmitter of the fly CNS.
#: ``gaba`` -1
#:     Rdl chloride channel.
#: ``glutamate`` -1
#:     Glutamate is INHIBITORY at most fly central synapses, acting on the
#:     GluCl-alpha chloride channel (Liu & Wilson 2013, *J. Neurosci.*
#:     33:10659). This is the opposite of the vertebrate convention and is
#:     the single most consequential sign choice in the model: it flips
#:     29,296 of 164,587 neurons.
#: ``histamine`` -1
#:     ort / HisCl1 chloride channels (photoreceptors, some mechanosensors).
#: ``dopamine``, ``octopamine``, ``serotonin``, ``unclear`` 0
#:     Modulatory or unknown. These neurons still integrate their input and
#:     still spike; they simply have no modelled output. Shiu et al. instead
#:     force every edge to +/-1; an explicit zero is preferred here to a
#:     guess. It costs 1,023,493 of 25,563,197 connections. The consequence
#:     is that the model contains no neuromodulation whatsoever.
NT_SIGN: Dict[str, int] = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 0,
    "octopamine": 0,
    "serotonin": 0,
    "unclear": 0,
}

#: Columns kept from the 36-column annotation table.
_ANNOTATION_COLUMNS = (
    "bodyId",
    "type",
    "instance",
    "class",
    "subclass",
    "superclass",
    "somaSide",
    "rootSide",
    "receptorType",
    "statusLabel",
    "flywireType",
)


def default_cache_root() -> Path:
    """Where the cache lives: ``$FLYBRAIN_MALECNS_DIR`` or a per-user cache dir."""
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "flybrain" / "malecns"


@dataclass
class CacheConfig:
    """Where the cache lives and how the graph is built.

    Attributes:
        root: Directory holding both the downloaded release tables and the
            built cache. Defaults to :func:`default_cache_root`.
        min_synapses: Drop connections with fewer synapses than this. 1 keeps
            every traced connection, matching Shiu et al. (2024). Raising it
            builds a smaller, faster, and explicitly different model.
        nt_sign: Transmitter to sign map. See :data:`NT_SIGN`.
    """

    root: Path = field(default_factory=default_cache_root)
    min_synapses: int = 1
    nt_sign: Dict[str, int] = field(default_factory=lambda: dict(NT_SIGN))

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()

    @property
    def matrix_path(self) -> Path:
        """Cached sparse connectome."""
        return self.root / f"malecns-v1.0-connectome-min{self.min_synapses}.npz"

    @property
    def neurons_path(self) -> Path:
        """Cached neuron metadata, in matrix-index order."""
        return self.root / f"malecns-v1.0-neurons-min{self.min_synapses}.parquet"


@dataclass
class Connectome:
    """A built, cached MaleCNS graph.

    Attributes:
        weights: Signed synapse counts as a CSR matrix of shape ``(n, n)``.
            Rows are presynaptic: ``weights[i, j]`` is
            ``sign(transmitter of i) * n_synapses(i -> j)``.
        neurons: One row per neuron, in matrix-index order. Columns include
            ``bodyId``, ``type``, ``instance``, ``cell_class``, ``subclass``,
            ``superclass``, ``soma_side``, ``nt`` and ``nt_sign``.
    """

    weights: "object"  # scipy.sparse.csr_matrix; typed loosely to keep import lazy
    neurons: "object"  # pandas.DataFrame

    @property
    def n_neurons(self) -> int:
        """Number of neurons in the graph."""
        return int(self.weights.shape[0])

    @property
    def n_synapses(self) -> int:
        """Total synapses, ignoring sign."""
        return int(np.abs(self.weights.data).sum())

    def select(self, **criteria: object) -> np.ndarray:
        """Matrix indices of neurons matching every ``column=value`` criterion.

        A value may be a scalar or any iterable of accepted values, so
        ``select(type=["LB3b", "LB3c"])`` returns both types.

        Raises:
            KeyError: If a named column is not in the neuron table.
        """
        mask = np.ones(len(self.neurons), dtype=bool)
        for column, value in criteria.items():
            if column not in self.neurons.columns:
                raise KeyError(
                    f"No neuron column {column!r}. "
                    f"Available: {list(self.neurons.columns)}"
                )
            col = self.neurons[column]
            if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
                mask &= (col == value).to_numpy()
            else:
                mask &= col.isin(list(value)).to_numpy()
        return np.flatnonzero(mask)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"<Connectome male-cns:v1.0 neurons={self.n_neurons} "
            f"connections={self.weights.nnz}>"
        )


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------


def _download(name: str, root: Path, chunk: int = 1 << 22) -> Path:
    """Fetch one release table into ``root`` unless it is already there.

    Args:
        name: File name within :data:`BUCKET_URL`.
        root: Destination directory.
        chunk: Read size, in bytes.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If the download fails. There is deliberately no
            synthetic fallback: a fly brain SDK that invents its own
            connectome would be worse than one that refuses to start.
    """
    import urllib.error
    import urllib.request

    root.mkdir(parents=True, exist_ok=True)
    dest = root / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    url = f"{BUCKET_URL}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {name}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as fh:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while True:
                block = response.read(chunk)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if total:
                    print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="")
        print()
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the MaleCNS v1.0 table {name}: {exc}\n"
            f"  source: {url}\n"
            "These tables are public (CC-BY) and need no account or token. If "
            "this machine has no direct internet access, copy them in by hand:\n"
            f"    gsutil -m cp "
            f"'gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/{name}' "
            f"{root}/"
        ) from exc
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------
# build / load
# --------------------------------------------------------------------------


def build_cache(config: Optional[CacheConfig] = None, force: bool = False) -> Connectome:
    """Download the release tables and write the sparse cache. Runs once.

    Args:
        config: Cache location and graph options.
        force: Rebuild even when the cache files already exist.

    Returns:
        The built connectome.

    Raises:
        ImportError: If pyarrow, pandas or scipy are missing.
        RuntimeError: If a download fails, or the release reports a
            transmitter with no entry in ``config.nt_sign``.
    """
    import pandas as pd
    import pyarrow.feather as feather
    import scipy.sparse as sp

    config = config or CacheConfig()
    if not force and config.matrix_path.exists() and config.neurons_path.exists():
        return load_cache(config)

    annotations = _download(ANNOTATIONS_FILE, config.root)
    transmitters = _download(NEUROTRANSMITTERS_FILE, config.root)
    connections = _download(WEIGHTS_FILE, config.root)

    print("  reading connections", flush=True)
    edges = feather.read_table(
        connections, columns=["body_pre", "body_post", "weight"]
    ).to_pandas()
    if config.min_synapses > 1:
        edges = edges[edges.weight >= config.min_synapses]

    bodies = np.union1d(
        edges.body_pre.to_numpy(), edges.body_post.to_numpy()
    ).astype(np.int64)
    row_of_body = pd.Series(np.arange(len(bodies), dtype=np.int64), index=bodies)
    print(f"  {len(bodies):,} neurons, {len(edges):,} connections", flush=True)

    print("  reading annotations and neurotransmitters", flush=True)
    neurons = _neuron_table(annotations, transmitters, bodies, config.nt_sign)

    rows = row_of_body.loc[edges.body_pre.to_numpy()].to_numpy()
    cols = row_of_body.loc[edges.body_post.to_numpy()].to_numpy()
    signed = (
        edges.weight.to_numpy(dtype=np.float32)
        * neurons.nt_sign.to_numpy(dtype=np.float32)[rows]
    )

    # Connections out of modulatory or unclassified neurons carry no sign and
    # are therefore not modelled. See NT_SIGN.
    keep = signed != 0.0
    print(
        f"  {int((~keep).sum()):,} connections not modelled "
        "(presynaptic transmitter modulatory or unclear)",
        flush=True,
    )

    weights = sp.csr_matrix(
        (signed[keep], (rows[keep], cols[keep])), shape=(len(bodies), len(bodies))
    )
    weights.sum_duplicates()

    config.root.mkdir(parents=True, exist_ok=True)
    sp.save_npz(config.matrix_path, weights, compressed=False)
    neurons.to_parquet(config.neurons_path, index=False)
    print(
        f"  cached {config.matrix_path.name} "
        f"({config.matrix_path.stat().st_size / 1e6:.0f} MB)",
        flush=True,
    )
    return Connectome(weights, neurons)


def _neuron_table(
    annotations: Path,
    transmitters: Path,
    bodies: np.ndarray,
    nt_sign: Dict[str, int],
):
    """Join annotations and transmitter predictions onto the graph's bodies.

    Raises:
        RuntimeError: If the release names a transmitter with no sign.
    """
    import pandas as pd
    import pyarrow.feather as feather

    ann = feather.read_table(
        annotations, columns=list(_ANNOTATION_COLUMNS)
    ).to_pandas().set_index("bodyId").reindex(bodies)

    # consensus_nt already backs a low-confidence per-body call off onto the
    # cell-type-level prediction, so it is the column to trust.
    nt = (
        feather.read_table(
            transmitters,
            columns=[
                "body",
                "consensus_nt",
                "predicted_nt_confidence",
                "ground_truth",
            ],
        )
        .to_pandas()
        .drop_duplicates("body")
        .set_index("body")
        .reindex(bodies)
    )

    table = pd.DataFrame(
        {
            "bodyId": bodies,
            "type": ann["type"].to_numpy(),
            "instance": ann["instance"].to_numpy(),
            "cell_class": ann["class"].to_numpy(),
            "subclass": ann["subclass"].to_numpy(),
            "superclass": ann["superclass"].to_numpy(),
            "soma_side": ann["somaSide"].to_numpy(),
            "root_side": ann["rootSide"].to_numpy(),
            "receptor_type": ann["receptorType"].to_numpy(),
            "status": ann["statusLabel"].astype(object).to_numpy(),
            "flywire_type": ann["flywireType"].to_numpy(),
            "nt": nt["consensus_nt"].fillna("unclear").to_numpy(),
            "nt_confidence": nt["predicted_nt_confidence"].to_numpy(),
            "nt_is_ground_truth": nt["ground_truth"].notna().to_numpy(),
        }
    )
    unknown = set(table.nt.unique()) - set(nt_sign)
    if unknown:
        raise RuntimeError(
            "MaleCNS reports neurotransmitters this build has no sign for: "
            f"{sorted(unknown)}. Add them to NT_SIGN deliberately rather than "
            "letting them default to excitatory or to zero."
        )
    table["nt_sign"] = table.nt.map(nt_sign).astype(np.int8)
    return table


def cache_available(config: Optional[CacheConfig] = None) -> bool:
    """Whether :func:`load_cache` would succeed without downloading anything."""
    config = config or CacheConfig()
    return config.matrix_path.exists() and config.neurons_path.exists()


def load_cache(config: Optional[CacheConfig] = None) -> Connectome:
    """Load the cached connectome. Never downloads.

    Args:
        config: Cache location and graph options.

    Returns:
        The cached connectome.

    Raises:
        ImportError: If pandas or scipy are missing.
        RuntimeError: If the cache has not been built.
    """
    import pandas as pd
    import scipy.sparse as sp

    config = config or CacheConfig()
    if not cache_available(config):
        raise RuntimeError(
            f"No MaleCNS v1.0 connectome cache in {config.root}.\n"
            "Build it once (566 MB download, about a minute) with:\n"
            "    python -m flybrain.malecns download\n"
            f"or point {CACHE_ENV_VAR} at a directory that already holds one."
        )
    return Connectome(
        sp.load_npz(config.matrix_path).tocsr(),
        pd.read_parquet(config.neurons_path),
    )


def shuffle_preserving_degree(weights, seed: int = 0):
    """Rewire a connectome at random, preserving every neuron's degree.

    Each connection keeps its presynaptic neuron -- so out-degree, synapse
    count and transmitter sign are all untouched -- and the postsynaptic
    slots are permuted, leaving the in-degree sequence untouched too. What is
    destroyed is exactly the pairing, who talks to whom, which is the only
    thing the connectome actually measures.

    This is the null model for any claim of the form "the connectome produces
    behaviour X": a result that survives this shuffle is a property of the
    simulator, not of the wiring.

    Args:
        weights: Signed connectome, any scipy sparse format.
        seed: RNG seed.

    Returns:
        A CSR matrix with the same shape, degrees and edge values.
    """
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    coo = weights.tocoo()
    cols = coo.col.copy()
    rng.shuffle(cols)
    out = sp.coo_matrix((coo.data, (coo.row, cols)), shape=weights.shape).tocsr()
    out.sum_duplicates()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m flybrain.malecns`` entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m flybrain.malecns",
        description="Download and cache the MaleCNS v1.0 connectome.",
    )
    parser.add_argument(
        "command",
        choices=["download", "info"],
        help="'download' builds the cache; 'info' summarises an existing one",
    )
    parser.add_argument("--force", action="store_true", help="rebuild the cache")
    parser.add_argument(
        "--min-synapses",
        type=int,
        default=1,
        help="drop connections below this synapse count (default 1: keep all)",
    )
    args = parser.parse_args(argv)

    config = CacheConfig(min_synapses=args.min_synapses)
    print(f"MaleCNS v1.0 cache: {config.root}")
    cx = build_cache(config, force=args.force) if args.command == "download" else load_cache(config)

    print(
        f"  {cx.n_neurons:,} neurons, {cx.weights.nnz:,} modelled connections, "
        f"{cx.n_synapses:,} signed synapses"
    )
    print(cx.neurons.nt.value_counts().to_string())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
