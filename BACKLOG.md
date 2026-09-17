# Backlog

Defects from the ROSC_MTA audit (`ROSC_MTA2/setup/Claude/HANDOFF_XeniumSplitter.md` on BMESEQ,
2026-08-05) not yet fixed here. Defects 1, 2 (guard), 3 and the defect-4 de-duplication key were
addressed in v0.2.0 (2026-09-17). None of these regenerate the frozen ROSC_MTA split outputs.

| # | defect | status | notes |
|---|---|---|---|
| 2 | polygons over 25 vertices truncated to a sector | **guarded**: refused by default, `--allow_polygon_truncation` opts in with a warning | real fix: resample the ring to 25 vertices by arc length (`scipy.interpolate` on the closed contour), log the original vertex-count distribution; verify shoelace area ratio ~ 1.0 against `cell_boundaries.parquet`. Needed before any Proseg bundle is split for Xenium Explorer use. |
| 4 | one coords dir shared across two bundles, later run overwrote the earlier | de-dup key fixed to `(punch_id, path)` | still to do: refuse when two invocations would write the same `outs/punch_<id>` from different parent bundles (write a `parent_bundle` marker file per punch and compare on re-run). The `run_Proseg.sh` wrapper lives in ROSC_MTA, not here. |
| 5 | transcripts.zarr FOV indices are global while the name list is punch-local | open | pass `original_zarr=` through from `Xen_TMA_pipeline.py` (documented intent) or re-index `fov_index` into the punch-local list; make the docstring match whichever is chosen. |
| 6 | zarr outputs never regenerated on re-run; "exists" treated as "current" | open | compare a content hash of the source parquet (stored in zarr attrs) and regenerate when stale. |
| 7 | pipeline never verified its own outputs | **partly done**: `verify_punch_outputs()` checks h5/parquet order and shape, boundary id subsets, cell uniqueness | still to do: `cells.zarr.zip` row order == `cells.parquet` order and the md5 `cell_id` relationship; per-cell `cell_area`/centroids `allclose` to the parent; polygon area ratio (defect 2); make morphology crop failures and empty punches fatal or explicitly reported instead of `PIPELINE COMPLETED SUCCESSFULLY`; `tifffile` optional in one script and required in another. |
| 8 | zarr `cell_id` is an md5 prefix with 32-bit collisions | open, mitigated downstream | implement 10x's reversible encoding or widen to uint64; coordinate with ROSC_MTA2's md5 assertion, which will start failing on purpose. |
| - | `experiment.xenium` copied unmodified (`num_cells`, `region_name` are whole-region values) | open | rewrite the per-punch fields. |
| - | morphology crop origin recorded nowhere in the punch bundle; `masks/homogeneous_transform` has zero translation | open | write the crop origin into the bundle and the transform. |
| - | `gene_offset` indexes an ordering never written; density `rows` attr 4x; `uint16` feature indices | open | low priority, not exercised by current tools. |
| - | `nucleus_centroid_x/y` filled with the cell centroid | open | documented approximation. |
| - | `transcripts_parquet_to_zarr.py` uses `Group.create_dataset`, removed in zarr 3.2 | open | port to `create_array`; until then `zarr<3.2` is pinned. |
