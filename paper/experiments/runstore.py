"""Content-addressed result store for reconstruction experiments.

Motivation: 2D long runs and 3D GPU runs are expensive (hours, overnight), and
the existing scripts emit only PNG + Markdown -- so re-plotting or re-analysing
anything required a full re-run. This module persists the raw arrays plus
enough provenance to audit a run later, and skips recomputation unless the
methodology actually changed.

A run is keyed by a hash of its config dict. Change the config (iteration
count, tolerances, geometry, seed, ...) and you get a new key, so results
never silently mix across methodology changes. Re-run with an unchanged config
and `load_or_run` returns the stored arrays instead of recomputing.

Layout, under paper/experiments/results/<experiment>/:
    <key>.npz      arrays (curves, volumes, snapshots)
    <key>.json     manifest: config, environment, timings, array index
    <key>.partial  present while a run is in flight; removed on success

Usage:
    store = RunStore("2d_longrun")
    res = store.load_or_run(cfg, compute_fn)          # dict of arrays
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=str(Path(__file__).resolve().parents[2]),
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"],
                                   cwd=str(Path(__file__).resolve().parents[2]),
                                   capture_output=True, text=True, timeout=10)
            return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return "unknown"


def _pkg_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "absent"


def environment():
    """Everything needed to explain a numerical difference between machines."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "numba": _pkg_version("numba"),
        "astra": _pkg_version("astra"),
        "git_commit": _git_commit(),
    }
    try:                                     # GPU identity, when ASTRA is in play
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            env["gpu"] = out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return env


# --------------------------------------------------------------------------
# Config hashing
# --------------------------------------------------------------------------
def _canonical(obj):
    """JSON-stable form. Numpy scalars/arrays are reduced so that a config
    carrying e.g. an array of snapshot iterations still hashes consistently."""
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


# Bump when solver SEMANTICS change, to invalidate every stored result.
# The store keys on config, not on code, so a bug fix in the solver does NOT
# invalidate the cache on its own -- stale results would be served silently and
# look like fresh ones. Anything that changes the numbers a given config
# produces must bump this.
#
#   1  initial
#   2  2026-08-14: single-channel's scaled tolerance was computed before the
#      noise calibration while two-channel's was computed after, so with
#      eps_mode="noise" single-channel ran with a ~137x tighter data constraint
#      than two-channel. Every noise statistic produced before this is void.
SOLVER_VERSION = 2


def config_key(cfg):
    payload = {"__solver_version__": SOLVER_VERSION, **_canonical(cfg)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
class RunStore:
    def __init__(self, experiment, root=None):
        self.dir = (Path(root) if root else RESULTS) / experiment
        self.dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment

    def paths(self, key):
        return (self.dir / f"{key}.npz", self.dir / f"{key}.json",
                self.dir / f"{key}.partial")

    def exists(self, cfg):
        npz, man, _ = self.paths(config_key(cfg))
        return npz.exists() and man.exists()

    # -- flatten/unflatten -------------------------------------------------
    @staticmethod
    def _flatten(result):
        """np.savez wants a flat namespace. Nested dicts keyed by iteration
        (the snapshot dicts) are flattened to 'snaps_single/00100' etc."""
        flat, index = {}, {}
        for k, v in result.items():
            if isinstance(v, dict):
                index[k] = sorted(int(i) for i in v)
                for i, arr in v.items():
                    flat[f"{k}/{int(i):06d}"] = np.asarray(arr)
            else:
                flat[k] = np.asarray(v)
        return flat, index

    @staticmethod
    def _unflatten(npz, index):
        out = {}
        for k in npz.files:
            if "/" in k:
                grp, i = k.split("/", 1)
                out.setdefault(grp, {})[int(i)] = npz[k]
            else:
                out[k] = npz[k]
        for grp in index:
            out.setdefault(grp, {})
        return out

    # -- main entry point --------------------------------------------------
    def load_or_run(self, cfg, compute_fn, force=False, label=None):
        """Return stored arrays for `cfg`, or run `compute_fn(cfg)` and store.

        `compute_fn` must return a dict whose values are arrays, scalars, or
        {int: array} snapshot dicts. Anything non-numeric should be put in the
        config instead, where it participates in the hash.
        """
        key = config_key(cfg)
        npz_p, man_p, part_p = self.paths(key)
        tag = label or key

        if npz_p.exists() and man_p.exists() and not force:
            man = json.loads(man_p.read_text())
            with np.load(npz_p, allow_pickle=False) as z:
                data = self._unflatten(z, man.get("array_index", {}))
            print(f"[runstore] hit  {self.experiment}/{tag} ({key}) "
                  f"-- {man.get('elapsed_s', 0):.0f}s saved")
            return data

        if part_p.exists():
            print(f"[runstore] warning: {tag} has a stale .partial marker "
                  f"(previous run died); recomputing")
        part_p.write_text(json.dumps({"started": time.time(), "config":
                                      _canonical(cfg)}, indent=2))
        print(f"[runstore] miss {self.experiment}/{tag} ({key}) -- computing")

        t0 = time.time()
        try:
            result = compute_fn(cfg)
        except BaseException:
            part_p.unlink(missing_ok=True)     # never leave a half-claimed key
            raise
        elapsed = time.time() - t0

        flat, index = self._flatten(result)
        # Write through a file handle: np.savez_compressed appends '.npz' to a
        # path that lacks it, which would defeat the atomic rename below.
        tmp = npz_p.with_name(npz_p.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **flat)
        os.replace(tmp, npz_p)                 # atomic: no truncated .npz
        man_p.write_text(json.dumps({
            "experiment": self.experiment, "key": key, "label": label,
            "solver_version": SOLVER_VERSION,
            "config": _canonical(cfg), "environment": environment(),
            "elapsed_s": elapsed, "finished_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "array_index": index,
            "arrays": {k: list(np.asarray(v).shape) for k, v in flat.items()},
        }, indent=2))
        part_p.unlink(missing_ok=True)
        print(f"[runstore] saved {npz_p.name} ({npz_p.stat().st_size/1e6:.1f} MB, "
              f"{elapsed:.0f}s)")
        return result

    def manifests(self):
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except Exception:
                pass
        return out
