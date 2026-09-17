"""Synthetic Xenium parent bundle for tests.

Writes the files the splitter reads (cells.parquet, cell_feature_matrix.h5, cell_boundaries.parquet,
nucleus_boundaries.parquet, transcripts.parquet, experiment.xenium) with the column schema of a
Xenium Onboard Analysis 6.1 bundle, plus one Xenium Explorer coordinates export per punch.

Two properties matter for the tests:
- cells.parquet rows are in a random order and the h5 barcodes are in a different (sorted) order,
  so any code that joins the two positionally fails loudly.
- everything is a function of the seed, so two generated bundles are identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ALPHABET = "abcdefghijklmnop"
REGION = 1000.0  # um, square region
# punch_id -> (x_min, x_max, y_min, y_max); hyphenated ids exercise the punch_id fallback
PUNCHES = {
    "6-1-Liver": (100.0, 450.0, 100.0, 450.0),
    "6-2-Liver": (550.0, 900.0, 100.0, 450.0),
}


def _cell_ids(n: int, rng: np.random.Generator) -> list[str]:
    ids: set[str] = set()
    while len(ids) < n:
        ids.add("".join(rng.choice(list(ALPHABET), 8)) + "-1")
    return sorted(ids)


def make_bundle(root: Path, n_cells: int = 300, n_genes: int = 50, seed: int = 7,
                long_polygon_cells: int = 0) -> dict:
    """Write a parent bundle under root/bundle and coordinates under root/coords. Returns a summary."""
    rng = np.random.default_rng(seed)
    bundle = root / "bundle"
    coords_dir = root / "coords"
    bundle.mkdir(parents=True, exist_ok=True)
    coords_dir.mkdir(parents=True, exist_ok=True)

    ids_sorted = _cell_ids(n_cells, rng)
    x = rng.uniform(0, REGION, n_cells)
    y = rng.uniform(0, REGION, n_cells)
    area = rng.uniform(20, 300, n_cells).round(3)
    genes = [f"Gene{i:03d}" for i in range(n_genes)]

    # counts matrix, cells in SORTED id order (the h5 order)
    counts = rng.poisson(0.6, size=(n_genes, n_cells)).astype(np.int32)
    transcript_counts = counts.sum(axis=0)

    # cells.parquet in a RANDOM order, different from the h5
    perm = rng.permutation(n_cells)
    cells = pd.DataFrame({
        "cell_id": np.array(ids_sorted)[perm],
        "x_centroid": x[perm], "y_centroid": y[perm],
        "transcript_counts": transcript_counts[perm].astype(np.int64),
        "control_probe_counts": 0, "genomic_control_counts": 0, "control_codeword_counts": 0,
        "unassigned_codeword_counts": 0, "deprecated_codeword_counts": 0,
        "total_counts": transcript_counts[perm].astype(np.int64),
        "cell_area": area[perm], "nucleus_area": (area[perm] * 0.2).round(3),
        "nucleus_count": 1, "segmentation_method": "Segmented by boundary stain",
    })
    cells.to_parquet(bundle / "cells.parquet", index=False)

    # h5 in 10x format, barcodes sorted (a different order from cells.parquet)
    with h5py.File(bundle / "cell_feature_matrix.h5", "w") as h:
        h.attrs["filetype"] = "matrix"
        h.attrs["version"] = 2
        m = h.create_group("matrix")
        m.create_dataset("barcodes", data=np.array(ids_sorted, dtype="S10"))
        indptr = [0]
        data, indices = [], []
        for c in range(n_cells):
            nz = np.nonzero(counts[:, c])[0]
            indices.extend(nz.tolist())
            data.extend(counts[nz, c].tolist())
            indptr.append(len(data))
        m.create_dataset("data", data=np.array(data, dtype=np.int32))
        m.create_dataset("indices", data=np.array(indices, dtype=np.int64))
        m.create_dataset("indptr", data=np.array(indptr, dtype=np.int64))
        m.create_dataset("shape", data=np.array([n_genes, n_cells], dtype=np.int32))
        f = m.create_group("features")
        f.create_dataset("_all_tag_keys", data=np.array([b"genome"], dtype="S6"))
        f.create_dataset("feature_type", data=np.array([b"Gene Expression"] * n_genes, dtype="S25"))
        f.create_dataset("genome", data=np.array([b"mm10"] * n_genes, dtype="S7"))
        f.create_dataset("id", data=np.array([f"ENSMUSG{i:011d}".encode() for i in range(n_genes)], dtype="S31"))
        f.create_dataset("name", data=np.array([g.encode() for g in genes], dtype="S31"))

    # boundaries: squares, optionally a few long rings (defect 2)
    def ring(cx, cy, r, n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return cx + r * np.cos(t), cy + r * np.sin(t)

    rows_c, rows_n = [], []
    long_ids = set(ids_sorted[:long_polygon_cells])
    for cid, cx, cy, a in zip(ids_sorted, x, y, area):
        r = float(np.sqrt(a / np.pi))
        n = 40 if cid in long_ids else 4
        vx, vy = ring(cx, cy, r, n)
        rows_c.append(pd.DataFrame({"cell_id": cid, "vertex_x": vx.astype(np.float32),
                                    "vertex_y": vy.astype(np.float32), "label_id": 0}))
        vx, vy = ring(cx, cy, r * 0.45, 4)
        rows_n.append(pd.DataFrame({"cell_id": cid, "vertex_x": vx.astype(np.float32),
                                    "vertex_y": vy.astype(np.float32), "label_id": 0}))
    pd.concat(rows_c, ignore_index=True).to_parquet(bundle / "cell_boundaries.parquet", index=False)
    pd.concat(rows_n, ignore_index=True).to_parquet(bundle / "nucleus_boundaries.parquet", index=False)

    # transcripts: those counted in the matrix, placed near their cell, plus unassigned ones
    tx = []
    for c, cid in enumerate(ids_sorted):
        for g in np.nonzero(counts[:, c])[0]:
            k = int(counts[g, c])
            tx.append(pd.DataFrame({
                "cell_id": cid, "feature_name": genes[g],
                "x_location": (x[c] + rng.normal(0, 2, k)).astype(np.float32),
                "y_location": (y[c] + rng.normal(0, 2, k)).astype(np.float32),
            }))
    n_un = 200
    tx.append(pd.DataFrame({
        "cell_id": "UNASSIGNED", "feature_name": rng.choice(genes, n_un),
        "x_location": rng.uniform(0, REGION, n_un).astype(np.float32),
        "y_location": rng.uniform(0, REGION, n_un).astype(np.float32),
    }))
    t = pd.concat(tx, ignore_index=True)
    nt = len(t)
    t.insert(0, "transcript_id", np.arange(nt, dtype=np.uint64) + 281_000_000_000_000)
    t["overlaps_nucleus"] = rng.integers(0, 2, nt).astype(np.uint8)
    t["z_location"] = np.float32(15.0)
    t["qv"] = rng.uniform(10, 40, nt).astype(np.float32)
    t["fov_name"] = "A1"
    t["nucleus_distance"] = np.where(t["overlaps_nucleus"] == 1, -1.0, rng.uniform(0, 30, nt)).astype(np.float32)
    t["codeword_index"] = rng.integers(0, 20000, nt).astype(np.int32)
    t["codeword_category"] = "predesigned_gene"
    t["is_gene"] = True
    t = t[["transcript_id", "cell_id", "overlaps_nucleus", "feature_name", "x_location", "y_location",
           "z_location", "qv", "fov_name", "nucleus_distance", "codeword_index", "codeword_category", "is_gene"]]
    t.to_parquet(bundle / "transcripts.parquet", index=False)

    (bundle / "experiment.xenium").write_text(json.dumps({
        "major_version": 6, "minor_version": 1, "patch_version": 0,
        "run_name": "synthetic", "region_name": "Liver", "preservation_method": "ffpe",
        "num_cells": n_cells, "region_area": REGION * REGION, "cassette_name": "TMA0",
        "slide_id": "0000000", "panel_name": "synthetic_panel", "panel_design_id": "SYNTH",
        "panel_predesigned_id": "none", "pixel_size": 0.2125,
        "analysis_sw_version": "xenium-4.0.2.2",
    }, indent=4))
    (bundle / "gene_panel.json").write_text(json.dumps({"payload": {"targets": [{"type": {"data": {"name": g}}} for g in genes]}}))

    # coordinates exports, exactly as Xenium Explorer writes them (CRLF, two header lines)
    for pid, (x0, x1, y0, y1) in PUNCHES.items():
        lines = [f"#Selection names: {pid}", f"#Areas (µm^2): {(x1 - x0) * (y1 - y0):.1f}",
                 "Selection,X,Y,Class,Color"]
        for vx, vy in [(x1, y1), (x1, y0), (x0, y0), (x0, y1), (x1, y1)]:
            lines.append(f"{pid},{vx},{vy},Unclassified,#c2c2c2")
        (coords_dir / f"{pid}_coordinates.csv").write_bytes(("\r\n".join(lines)).encode("utf-8"))

    in_punch = {pid: int(((x >= b[0]) & (x <= b[1]) & (y >= b[2]) & (y <= b[3])).sum()) for pid, b in PUNCHES.items()}
    return {"bundle": bundle, "coords": coords_dir, "ids_sorted": ids_sorted, "cells_in_punch": in_punch,
            "cells_order": cells["cell_id"].tolist()}
