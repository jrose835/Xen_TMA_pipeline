# Xen_TMA_pipeline
A processing pipeline for tissue microarrays run on the Xenium spatial transcriptomics platform

This pipeline splits Xenium data based on spatial coordinates only.
It filters cells and transcripts that fall within the coordinate bounds of each punch region, and can also filter by cell size or minimum transcript number.

## Features

The pipeline performs the following steps:

1. **QC Metrics Calculation** - Generates region-specific quality control metrics
2. **Data Splitting** - Splits Xenium data files by punch regions based on coordinates. 

## Requirements

- Python 3.8+
- Required packages: pandas, numpy, h5py, zarr, pyarrow
- Optional: tifffile (for morphology image cropping)

## Outputs

Each punch subdirectory contains all files required for Xenium Explorer and downstream analysis with Xenium Ranger.

### Main output directory
- `region_metrics.csv` - QC metrics for each punch region
- `all_punch_coordinates.csv` - Consolidated punch coordinates (if using cell_stats_dir)
- `all_cell_stats.csv` - Aggregated cell statistics (if using cell_stats_dir)

### outs/ subdirectory
- `punch_cell_summary.csv` - Summary of cells per punch

### Each punch_*/ subdirectory contains:

**Core Xenium files:**
- `cell_feature_matrix.h5` - Cell-gene expression matrix
- `cells.parquet` - Cell metadata
- `cell_boundaries.parquet` - Cell boundary polygons
- `nucleus_boundaries.parquet` - Nucleus boundary polygons
- `transcripts.parquet` - Transcript locations and assignments
- `experiment.xenium` - Experiment metadata

**Additional files (unless --run_light is specified):**
- `metrics_summary.csv` - 10x Xenium format metrics
- `morphology.ome.tif` - Cropped morphology image
- `morphology_focus/` - Cropped autofocus images
- `gene_panel.json` - Gene panel definition
- `analysis_summary.html` - Analysis summary
- `cells.zarr.zip` - Xenium Explorer compatible cell data
- `transcripts.zarr.zip` - Xenium Explorer compatible transcript data

## How coordinates are consumed

The splitter works on axis-aligned bounding boxes, one per punch. There are two ways to supply
them:

1. `--cell_stats_dir DIR`: a directory of **Xenium Explorer selection exports**, one file per
   punch, named `<anything>_coordinates.csv`. Each file is read for its header and its vertex
   list; the bounding box is the min and max of the `X` and `Y` columns. Both header variants
   Xenium Explorer writes are accepted: `#Selection name:` (single selection) and
   `#Selection names:` (multi-selection export), followed by `#Area (µm^2):` or `#Areas (µm^2):`.
   CRLF line endings are fine. The selection name becomes the `punch_id`; if it has three or
   four `_`-separated parts they are read as `punch_id_treatment_tissue[_replicate]`, otherwise
   (for example `24-2-Liver`) the whole name is the `punch_id` and the other fields are empty.
   A matching `<prefix>_cells_stats.csv` export is aggregated into `all_cell_stats.csv` when
   present but is not required. The consolidated `all_punch_coordinates.csv` is **written** to
   the output directory as a derived file.
2. `--coordinates_file FILE`: a previously written `all_punch_coordinates.csv` (columns
   `punch_id, treatment, tissue, replicate, selection_name, region_area, x_min, x_max, y_min,
   y_max, coordinate_file, path`). Rows whose `path` is not the current bundle are ignored.

Polygons are not honoured: a non-rectangular selection is reduced to its bounding box. Draw
rectangles.

## Reproducibility (v0.2.0)

Two runs of identical code on identical inputs produce byte-identical `cell_feature_matrix.h5`,
`cells.parquet`, boundary and transcript parquets and metrics, regardless of `PYTHONHASHSEED`.
In every punch the h5 barcode order equals the `cells.parquet` `cell_id` order, and both follow
the parent `cells.parquet` order. `verify_punch_outputs()` checks this after every split and
fails the run otherwise. Densities in `region_metrics.csv` are per mm² (`cells_per_mm2`,
`nuclear_transcripts_per_mm2`, `decoded_transcripts_per_mm2`); the per-punch 10x-format
`metrics_summary.csv` derives its `*_per_100um2` columns from them. Polygons with more than 25
vertices are refused at zarr conversion unless `--allow_polygon_truncation` is given. Open
items are in `BACKLOG.md`; the pinned environment used for ROSC is `envs/XenSplitter.lock.yml`.

Install: `pip install git+https://github.com/jrose835/Xen_TMA_pipeline@v0.2.0` provides the
`xen-tma-pipeline` command. Tests: `pip install -e .[test] && pytest`.

## Usage

### Basic usage with coordinate files from cell_stats_dir
```bash
python Xen_TMA_pipeline.py \
  --xenium_bundle_path /path/to/xenium/output \
  --cell_stats_dir /path/to/cell_stats \
  --output_dir /path/to/output
```

### Using pre-existing consolidated coordinates file
```bash
python Xen_TMA_pipeline.py \
  --xenium_bundle_path /path/to/xenium/output \
  --coordinates_file /path/to/all_punch_coordinates.csv \
  --output_dir /path/to/output
```

### With quality filters
```bash
python Xen_TMA_pipeline.py \
  --xenium_bundle_path /path/to/xenium/output \
  --cell_stats_dir /path/to/cell_stats \
  --output_dir /path/to/output \
  --min_transcripts 10 \
  --min_area 50 \
  --qv_threshold 25 \
  --n_processes 8
```

### Light mode (skip additional processing and zarr conversion)
```bash
python Xen_TMA_pipeline.py \
  --xenium_bundle_path /path/to/xenium/output \
  --cell_stats_dir /path/to/cell_stats \
  --output_dir /path/to/output \
  --run_light
```

## Command Line Arguments

### Required Arguments
| Argument | Description |
|----------|-------------|
| `--xenium_bundle_path` | Path to Xenium bundle output directory |
| `--output_dir` | Path to output directory for split data and metrics |

### Coordinate Input (one required)
| Argument | Description |
|----------|-------------|
| `--cell_stats_dir` | Path to directory containing cell_stats and coordinates CSV files |
| `--coordinates_file` | Path to all_punch_coordinates.csv file |

### Optional Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--qv_threshold` | 20 | QV threshold for high quality transcripts |
| `--n_processes` | 4 | Number of processes for parallel data splitting |
| `--min_transcripts` | None | Minimum transcript count per cell (cells below this will be blacklisted) |
| `--min_area` | None | Minimum cell area (cells below this will be blacklisted) |
| `--skip_zarr` | False | Skip creating cells.zarr.zip and transcripts.zarr.zip files |
| `--overwrite_zarr` | False | Overwrite existing zarr.zip files |
| `--run_light` | False | Skip metrics_summary, morphology cropping, auxiliary files, and zarr conversion |

## Standalone Scripts

### Zarr conversion
Convert cells.parquet to cells.zarr.zip:
```bash
python scripts/parquet_to_zarr.py --input-dir /path/to/outs --verbose
```

Convert transcripts.parquet to transcripts.zarr.zip:
```bash
python scripts/transcripts_parquet_to_zarr.py --input-dir /path/to/outs --verbose
```

Use `--overwrite` to overwrite existing zarr files.

## License

See LICENSE file for details.
