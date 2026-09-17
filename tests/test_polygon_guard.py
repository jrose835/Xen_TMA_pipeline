"""Defect 2: polygons over the 25-vertex cap are refused unless truncation is requested explicitly."""

from __future__ import annotations

import pandas as pd
import pytest

from parquet_to_zarr import MAX_VERTICES, build_polygon_arrays
from synth import make_bundle
from test_determinism import run_pipeline


def _bounds(n_verts_by_cell):
    rows = []
    for cid, n in n_verts_by_cell.items():
        rows.append(pd.DataFrame({"cell_id": cid, "vertex_x": range(n), "vertex_y": range(n)}))
    return pd.concat(rows, ignore_index=True)


def test_within_cap_passes():
    v, n, idx = build_polygon_arrays(_bounds({"a": 4, "b": MAX_VERTICES}), ["a", "b"])
    assert list(n) == [4, MAX_VERTICES]


def test_over_cap_refused_by_default():
    with pytest.raises(ValueError, match="Refusing to truncate"):
        build_polygon_arrays(_bounds({"a": 4, "b": MAX_VERTICES + 1}), ["a", "b"])


def test_over_cap_truncates_only_when_allowed(capsys):
    v, n, idx = build_polygon_arrays(_bounds({"a": 4, "b": 40}), ["a", "b"], allow_truncation=True)
    assert list(n) == [4, MAX_VERTICES]
    assert "WARNING" in capsys.readouterr().out


def test_pipeline_zarr_step_refuses_long_polygons(tmp_path):
    info = make_bundle(tmp_path, seed=9, long_polygon_cells=5)
    r = run_pipeline(info["bundle"], info["coords"], tmp_path / "out", "0")
    assert r.returncode != 0
    assert "Refusing to truncate" in r.stdout


def test_pipeline_zarr_step_with_explicit_truncation(tmp_path):
    info = make_bundle(tmp_path, seed=9, long_polygon_cells=5)
    r = run_pipeline(info["bundle"], info["coords"], tmp_path / "out", "0", extra=["--allow_polygon_truncation"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "truncating to the first" in r.stdout
    assert (tmp_path / "out" / "outs" / "punch_6-1-Liver" / "cells.zarr.zip").exists()
