"""Two runs of the pipeline under different PYTHONHASHSEED values must be byte-identical
(ROSC handoff, defect 1), and h5 barcode order must equal cells.parquet order in every punch."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import h5py
import pandas as pd
import pytest

from synth import PUNCHES, make_bundle

ROOT = Path(__file__).resolve().parents[1]
COMPARED = ["cell_feature_matrix.h5", "cells.parquet", "cell_boundaries.parquet",
            "nucleus_boundaries.parquet", "transcripts.parquet", "metrics_summary.csv"]


def run_pipeline(bundle: Path, coords: Path, out: Path, hashseed: str, extra=()) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    cmd = [sys.executable, str(ROOT / "Xen_TMA_pipeline.py"), "--xenium_bundle_path", str(bundle),
           "--cell_stats_dir", str(coords), "--output_dir", str(out), "--n_processes", "1", *extra]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("det")
    info = make_bundle(root, seed=11)
    outs = []
    for seed in ("1", "4242"):
        out = root / f"out_{seed}"
        r = run_pipeline(info["bundle"], info["coords"], out, seed, extra=["--skip_zarr"])
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "PIPELINE COMPLETED SUCCESSFULLY" in r.stdout
        outs.append(out)
    return info, outs


def test_outputs_byte_identical_across_hash_seeds(two_runs):
    _, (a, b) = two_runs
    assert sha(a / "region_metrics.csv") == sha(b / "region_metrics.csv")
    for pid in PUNCHES:
        for name in COMPARED:
            fa, fb = a / "outs" / f"punch_{pid}" / name, b / "outs" / f"punch_{pid}" / name
            assert fa.exists(), fa
            assert sha(fa) == sha(fb), f"{name} differs between runs for punch {pid}"


def test_h5_barcodes_follow_cells_parquet_order(two_runs):
    info, (a, _) = two_runs
    parent_order = {cid: i for i, cid in enumerate(info["cells_order"])}
    for pid in PUNCHES:
        d = a / "outs" / f"punch_{pid}"
        cells = pd.read_parquet(d / "cells.parquet")["cell_id"].tolist()
        with h5py.File(d / "cell_feature_matrix.h5") as h:
            barcodes = [b.decode() for b in h["matrix/barcodes"][:]]
            assert int(h["matrix/shape"][1]) == len(cells)
        assert barcodes == cells, f"punch {pid}: h5 order != cells.parquet order"
        # and that order is the parent cells.parquet order, not the parent h5 (sorted) order
        assert [parent_order[c] for c in cells] == sorted(parent_order[c] for c in cells)
        assert cells != sorted(cells), "fixture no longer distinguishes parquet order from sorted h5 order"
        assert len(cells) == info["cells_in_punch"][pid]


def test_h5_matrix_values_match_parent(two_runs):
    info, (a, _) = two_runs
    with h5py.File(info["bundle"] / "cell_feature_matrix.h5") as h:
        pb = [b.decode() for b in h["matrix/barcodes"][:]]
        pdata, pind, pptr = h["matrix/data"][:], h["matrix/indices"][:], h["matrix/indptr"][:]
    pidx = {c: i for i, c in enumerate(pb)}
    d = a / "outs" / f"punch_{next(iter(PUNCHES))}"
    with h5py.File(d / "cell_feature_matrix.h5") as h:
        cb = [b.decode() for b in h["matrix/barcodes"][:]]
        cdata, cind, cptr = h["matrix/data"][:], h["matrix/indices"][:], h["matrix/indptr"][:]
    for j, cid in enumerate(cb):
        i = pidx[cid]
        assert list(cdata[cptr[j]:cptr[j + 1]]) == list(pdata[pptr[i]:pptr[i + 1]])
        assert list(cind[cptr[j]:cptr[j + 1]]) == list(pind[pptr[i]:pptr[i + 1]])


def test_verify_catches_reordered_h5(tmp_path):
    """Must-fire: a punch whose h5 barcodes are not in cells.parquet order is rejected."""
    from XeniumSplitter import XeniumDataSplitter

    info = make_bundle(tmp_path, seed=3)
    out = tmp_path / "out"
    r = run_pipeline(info["bundle"], info["coords"], out, "0", extra=["--run_light"])
    assert r.returncode == 0, r.stdout[-2000:]
    pid = next(iter(PUNCHES))
    h5p = out / "outs" / f"punch_{pid}" / "cell_feature_matrix.h5"
    with h5py.File(h5p, "r+") as h:
        bc = h["matrix/barcodes"][:]
        del h["matrix/barcodes"]
        h["matrix"].create_dataset("barcodes", data=bc[::-1])  # same set, reversed order, no NAs
    s = XeniumDataSplitter(str(info["bundle"]), str(out / "all_punch_coordinates.csv"), str(out / "outs"))
    s.punch_ids = list(PUNCHES)
    with pytest.raises(RuntimeError, match="barcode order differs"):
        s.verify_punch_outputs()
