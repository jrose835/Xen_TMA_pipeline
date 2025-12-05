# Xen_TMA_pipeline
A processing pipeline for tissue microarrays run on the Xenium spatial transcriptomics platform

This pipeline splits Xenium data based on spatial coordinates only.
It filters cells and transcripts that fall within the coordinate bounds of each punch region, and can also filter by cell size or minimum transcript number.

## Examples:
  #### Using individual coordinate files from cell_stats_dir
  ```
  python Xen_TMA_pipeline.py \\
    --xenium_bundle_path /path/to/xenium/output \\
    --cell_stats_dir /path/to/cell_stats \\
    --output_dir /path/to/output
  ```

  #### Using pre-existing consolidated coordinates file
  ```
  python Xen_TMA_pipeline.py \\
    --xenium_bundle_path /path/to/xenium/output \\
    --cell_stats_dir /path/to/cell_stats \\
    --coordinates_file /path/to/all_punch_coordinates.csv \\
    --output_dir /path/to/output
  ```
  #### With quality filters
  ```
  python Xen_TMA_pipeline.py \\
    --xenium_bundle_path /path/to/xenium/output \\
    --cell_stats_dir /path/to/cell_stats \\
    --output_dir /path/to/output \\
    --min_transcripts 10 \\
    --min_area 50 \\
    --qv_threshold 25 \\
    --n_processes 8
  ```

  #### With zarr conversion for creating cells.zarr.zip files
  ```
  python Xen_TMA_pipeline.py \\
    --xenium_bundle_path /path/to/xenium/output \\
    --cell_stats_dir /path/to/cell_stats \\
    --output_dir /path/to/output \\
    --create_zarr
  ```
  Use `--overwrite_zarr` to overwrite existing cells.zarr.zip files. Standalone conversion is also available via `python scripts/parquet_to_zarr.py --input-dir /path/to/outs`.