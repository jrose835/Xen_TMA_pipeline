from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from run_metrics import parse_metadata_from_header, process_region_metrics
from synth import PUNCHES, make_bundle


def _write(tmp_path, header, name="x_coordinates.csv"):
    p = tmp_path / name
    body = "Selection,X,Y,Class,Color\r\nA,1,2,Unclassified,#c2c2c2\r\n"
    p.write_bytes((header + body).encode("utf-8"))
    return p


def test_plural_header_from_xenium_explorer(tmp_path):
    meta = parse_metadata_from_header(_write(tmp_path, "#Selection names: 24-2-Liver\r\n#Areas (µm^2): 3414683.5\r\n"))
    assert meta["selection_name"] == "24-2-Liver"
    assert meta["punch_id"] == "24-2-Liver"  # no '_' components: the name is the id
    assert meta["region_area"] == 3414683.5
    assert meta["treatment"] is None and meta["tissue"] is None


def test_singular_header_and_underscore_components(tmp_path):
    meta = parse_metadata_from_header(_write(tmp_path, "#Selection name: D3_RO_SP_1\r\n#Area (µm^2): 100.0\r\n"))
    assert (meta["punch_id"], meta["treatment"], meta["tissue"], meta["replicate"]) == ("D3", "RO", "SP", "1")


def test_densities_are_per_mm2(tmp_path):
    info = make_bundle(tmp_path, seed=5)
    cells = pd.read_parquet(info["bundle"] / "cells.parquet")
    tx = pd.read_parquet(info["bundle"] / "transcripts.parquet")
    pid, (x0, x1, y0, y1) = next(iter(PUNCHES.items()))
    area_um2 = (x1 - x0) * (y1 - y0)
    metrics = []
    process_region_metrics(cells, tx, x0, x1, y0, y1, {"punch_id": pid, "region_area": area_um2},
                           str(info["bundle"]), 20, metrics)
    m = metrics[0]
    n = info["cells_in_punch"][pid]
    assert m["num_cells"] == n
    assert m["cells_per_mm2"] == pytest.approx(n / (area_um2 / 1e6))
    assert m["decoded_transcripts_per_mm2"] == pytest.approx(m["total_high_qv_transcripts"] / (area_um2 / 1e6))
    assert "cells_per_100um2" not in m
    assert m["path"] == str(info["bundle"])


def test_real_vendor_scale_sanity():
    """25,720 cells over 11,065,936 um^2 is 2,324 per mm^2 (0.2324 per 100 um^2, as the vendor reports)."""
    metrics = []
    cells = pd.DataFrame({"cell_id": [f"c{i}" for i in range(25720)], "x_centroid": 1.0, "y_centroid": 1.0,
                          "cell_area": 100.0, "transcript_counts": 1})
    tx = pd.DataFrame({"cell_id": ["UNASSIGNED"], "x_location": [1.0], "y_location": [1.0], "is_gene": [True],
                       "qv": [30.0], "feature_name": ["G"], "nucleus_distance": [5.0]})
    process_region_metrics(cells, tx, 0, 2, 0, 2, {"punch_id": "L", "region_area": 11065936.21}, "b", 20, metrics)
    assert metrics[0]["cells_per_mm2"] == pytest.approx(2324.25, rel=1e-3)
    assert metrics[0]["cells_per_mm2"] / 1e4 == pytest.approx(0.2324, rel=1e-3)
