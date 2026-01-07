#!/usr/bin/env python3
"""
Convert cells.parquet files from XenSplitter output to cells.zarr.zip format
required by Xenium Explorer and SEGGER_EXPLORER module.

This module can be used standalone or imported by the main pipeline.

Usage (standalone):
    python parquet_to_zarr.py --input-dir ./outs --verbose
    python parquet_to_zarr.py --input-dir ./outs --subdir my_sample

The script will automatically find all subdirectories containing cells.parquet files.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from numcodecs import Blosc
from zarr.storage import LocalStore


MAX_VERTICES = 25  # Maximum polygon vertices (stored as 50 values: x1,y1,x2,y2,...)
PIXEL_SIZE = 0.2125  # µm per pixel for morphology images


def convert_cell_id_to_numeric(cell_ids: pd.Series) -> np.ndarray:
    """
    Convert string cell_ids to numeric uint32 format.

    Uses a hash-based approach to generate unique numeric IDs from string IDs.
    Returns array of shape (N, 2) with [numeric_id, dataset_suffix=1].
    """
    numeric_ids = []
    for cell_id in cell_ids:
        # Use first 8 hex chars of MD5 hash, convert to int
        hash_val = int(hashlib.md5(str(cell_id).encode()).hexdigest()[:8], 16)
        numeric_ids.append(hash_val)

    # Create (N, 2) array with [cell_id, dataset_suffix=1]
    result = np.zeros((len(cell_ids), 2), dtype=np.uint32)
    result[:, 0] = numeric_ids
    result[:, 1] = 1  # dataset suffix
    return result


def build_polygon_arrays(boundaries_df: pd.DataFrame, cell_ids: list) -> tuple:
    """
    Convert boundary parquet (one row per vertex) to fixed-size polygon arrays.

    Args:
        boundaries_df: DataFrame with cell_id, vertex_x, vertex_y columns
        cell_ids: Ordered list of cell_ids to match

    Returns:
        vertices: (N, 50) float32 array with [x1,y1,x2,y2,...,x25,y25]
        num_vertices: (N,) int32 array with actual vertex count per cell
        cell_index: (N,) uint32 array mapping polygon to cell index
    """
    n_cells = len(cell_ids)
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}

    # Initialize arrays
    vertices = np.zeros((n_cells, MAX_VERTICES * 2), dtype=np.float32)
    num_vertices = np.zeros(n_cells, dtype=np.int32)
    cell_index = np.arange(n_cells, dtype=np.uint32)

    # Group vertices by cell_id
    if len(boundaries_df) > 0:
        grouped = boundaries_df.groupby('cell_id')

        for cell_id, group in grouped:
            if cell_id not in cell_id_to_idx:
                continue

            idx = cell_id_to_idx[cell_id]
            coords = group[['vertex_x', 'vertex_y']].values

            # Limit to MAX_VERTICES
            n_verts = min(len(coords), MAX_VERTICES)
            num_vertices[idx] = n_verts

            # Flatten coordinates: [x1, y1, x2, y2, ...]
            flat_coords = coords[:n_verts].flatten()
            vertices[idx, :len(flat_coords)] = flat_coords

    return vertices, num_vertices, cell_index


def build_cell_summary(cells_df: pd.DataFrame) -> np.ndarray:
    """
    Build cell_summary array from cells.parquet data.

    Array columns (in order):
        0: cell_centroid_x
        1: cell_centroid_y
        2: cell_area
        3: nucleus_centroid_x (use cell centroid if not available)
        4: nucleus_centroid_y
        5: nucleus_area
        6: z_level (default 0.0)
        7: nucleus_count
    """
    n_cells = len(cells_df)
    summary = np.zeros((n_cells, 8), dtype=np.float64)

    # Cell centroids
    summary[:, 0] = cells_df['x_centroid'].values
    summary[:, 1] = cells_df['y_centroid'].values

    # Cell area
    summary[:, 2] = cells_df['cell_area'].values

    # Nucleus centroids (use cell centroid as approximation)
    summary[:, 3] = cells_df['x_centroid'].values
    summary[:, 4] = cells_df['y_centroid'].values

    # Nucleus area (fill NaN with 0)
    summary[:, 5] = cells_df['nucleus_area'].fillna(0).values

    # Z-level (default 0)
    summary[:, 6] = 0.0

    # Nucleus count
    summary[:, 7] = cells_df['nucleus_count'].values

    return summary


def get_morphology_shape(punch_dir: Path) -> tuple:
    """
    Get the shape of the morphology image in a punch directory.

    Returns:
        tuple: (height, width) of the morphology image
    """
    morph_path = punch_dir / 'morphology.ome.tif'
    if not morph_path.exists():
        return None

    with tifffile.TiffFile(morph_path) as tif:
        # Return (height, width) to match numpy array convention
        return (tif.pages[0].imagelength, tif.pages[0].imagewidth)


def create_masks_group(root: zarr.Group, temp_dir: str, mask_shape: tuple) -> None:
    """
    Create the masks group with homogeneous_transform and empty mask arrays.

    XeniumRanger requires a masks group in cells.zarr.zip containing:
    - homogeneous_transform: 4x4 float32 transformation matrix
    - 0: nucleus mask array (empty, filled with zeros)
    - 1: cell mask array (empty, filled with zeros)

    Args:
        root: zarr root group
        temp_dir: path to temp directory for direct file creation
        mask_shape: (height, width) of mask arrays
    """
    # Create masks group
    masks_group = root.create_group('masks')

    # Create homogeneous_transform array (4x4 float32)
    # Scale factor converts from mask pixel coordinates to micron coordinates
    scale = 1.0 / PIXEL_SIZE  # ~4.705882
    transform_data = np.array([
        [scale, 0.0, 0.0, 0.0],
        [0.0, scale, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Create homogeneous_transform using zarr API
    # Note: zarr v3 infers dtype from data, so don't specify dtype when using data param
    masks_group.create_array(
        'homogeneous_transform',
        data=transform_data,
        chunks=(4, 4),
        compressor=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    )

    # Create empty mask arrays (0 = nucleus, 1 = cell)
    # Use chunking similar to original Xenium format
    chunk_y = min(1699, mask_shape[0])
    chunk_x = min(755, mask_shape[1])

    for mask_idx in [0, 1]:
        # Create directory for mask zarr array metadata
        mask_dir = Path(temp_dir) / 'masks' / str(mask_idx)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # Write .zarray metadata file (empty array, no chunks written = all zeros)
        zarray_meta = {
            "chunks": [chunk_y, chunk_x],
            "compressor": {
                "blocksize": 0,
                "clevel": 1,
                "cname": "zstd",
                "id": "blosc",
                "shuffle": 1
            },
            "dtype": "<u4",
            "fill_value": 0,
            "filters": None,
            "order": "C",
            "shape": list(mask_shape),
            "zarr_format": 2
        }
        with open(mask_dir / '.zarray', 'w') as f:
            json.dump(zarray_meta, f, indent=4)


def create_zarr_store(output_path: Path, cells_df: pd.DataFrame,
                      cell_bounds_df: pd.DataFrame, nuc_bounds_df: pd.DataFrame,
                      mask_shape: tuple = None, verbose: bool = False) -> None:
    """
    Create cells.zarr.zip file with required structure for Xenium Explorer.

    Args:
        output_path: Path to output cells.zarr.zip file
        cells_df: DataFrame with cell data
        cell_bounds_df: DataFrame with cell boundary vertices
        nuc_bounds_df: DataFrame with nucleus boundary vertices
        mask_shape: Optional (height, width) tuple for mask arrays. If provided,
                    creates the masks group required by XeniumRanger.
        verbose: Print progress information
    """
    n_cells = len(cells_df)
    cell_ids = cells_df['cell_id'].tolist()

    if verbose:
        print(f"    Creating zarr store with {n_cells} cells...")

    # Build arrays
    cell_id_array = convert_cell_id_to_numeric(cells_df['cell_id'])
    cell_summary = build_cell_summary(cells_df)

    # Build polygon arrays for cells
    cell_vertices, cell_num_verts, cell_idx = build_polygon_arrays(cell_bounds_df, cell_ids)

    # Build polygon arrays for nuclei
    nuc_vertices, nuc_num_verts, nuc_idx = build_polygon_arrays(nuc_bounds_df, cell_ids)

    # Create zarr store in temp directory first, then zip (avoids duplicate entries with ZipStore)
    temp_dir = tempfile.mkdtemp(prefix='cells_zarr_')
    store = LocalStore(temp_dir)
    root = zarr.open(store, mode='w', zarr_format=2)

    # Set root attributes
    root.attrs['major_version'] = 6
    root.attrs['minor_version'] = 2
    root.attrs['name'] = 'CellSegmentationDataset'
    root.attrs['number_cells'] = n_cells
    root.attrs['spatial_units'] = 'microns'
    root.attrs['polygon_set_names'] = ['nucleus', 'cell']
    root.attrs['polygon_set_display_names'] = ['Nucleus boundaries', 'Cell boundaries']
    root.attrs['polygon_set_descriptions'] = ['Nucleus segmentation', 'Cell segmentation']
    root.attrs['segmentation_methods'] = [{'method': 'XenSplitter'}]

    # Create cell_id array
    root['cell_id'] = cell_id_array

    # Create cell_summary array with required column metadata
    root['cell_summary'] = cell_summary
    root['cell_summary'].attrs['column_names'] = [
        'cell_centroid_x', 'cell_centroid_y', 'cell_area',
        'nucleus_centroid_x', 'nucleus_centroid_y', 'nucleus_area',
        'z_level', 'nucleus_count'
    ]
    root['cell_summary'].attrs['column_descriptions'] = [
        'Cell centroid in X', 'Cell centroid in Y', 'Cell area',
        'Nucleus centroid in X', 'Nucleus centroid in Y', 'Nucleus area',
        'z_level', 'Nucleus count'
    ]

    # Create polygon_sets group for nuclei (index 0)
    nuc_group = root.create_group('polygon_sets/0')
    nuc_group['vertices'] = nuc_vertices
    nuc_group['num_vertices'] = nuc_num_verts
    nuc_group['cell_index'] = nuc_idx
    nuc_group['method'] = np.zeros(n_cells, dtype='uint32')

    # Create polygon_sets group for cells (index 1)
    cell_group = root.create_group('polygon_sets/1')
    cell_group['vertices'] = cell_vertices
    cell_group['num_vertices'] = cell_num_verts
    cell_group['cell_index'] = cell_idx
    cell_group['method'] = np.zeros(n_cells, dtype='uint32')

    # Create masks group (required by XeniumRanger)
    if mask_shape is not None:
        create_masks_group(root, temp_dir, mask_shape)
        if verbose:
            print(f"    Created masks group with shape {mask_shape}")

    # Close store and create clean zip file
    store.close()

    # Remove existing output and create fresh zip
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(str(output_path), 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root_path, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root_path, file)
                arc_name = os.path.relpath(file_path, temp_dir)
                zf.write(file_path, arc_name)

    # Clean up temp directory
    shutil.rmtree(temp_dir)

    if verbose:
        print(f"    Created {output_path}")


def has_required_files(directory: Path) -> bool:
    """Check if a directory contains the required cells.parquet file."""
    return (directory / 'cells.parquet').exists()


def find_valid_directories(input_dir: Path, verbose: bool = False) -> list:
    """
    Find all subdirectories that contain cells.parquet files.

    Returns list of Path objects for valid directories.
    """
    valid_dirs = []

    # Check all immediate subdirectories
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir() and has_required_files(subdir):
            valid_dirs.append(subdir)
        elif subdir.is_dir() and verbose:
            print(f"    Skipping {subdir.name}: missing cells.parquet")

    return valid_dirs


def process_directory(target_dir: Path, overwrite: bool = False,
                      verbose: bool = False) -> bool:
    """
    Process a single directory and create cells.zarr.zip.

    Returns True if successful, False otherwise.
    """
    output_path = target_dir / 'cells.zarr.zip'

    # Check if output exists
    if output_path.exists() and not overwrite:
        if verbose:
            print(f"    Skipping {target_dir.name}: cells.zarr.zip already exists")
        return True

    # Check for required input files
    cells_path = target_dir / 'cells.parquet'
    cell_bounds_path = target_dir / 'cell_boundaries.parquet'
    nuc_bounds_path = target_dir / 'nucleus_boundaries.parquet'

    if not cells_path.exists():
        print(f"    Error: cells.parquet not found in {target_dir}")
        return False

    if not cell_bounds_path.exists():
        print(f"    Warning: cell_boundaries.parquet not found in {target_dir}")
        cell_bounds_df = pd.DataFrame(columns=['cell_id', 'vertex_x', 'vertex_y'])
    else:
        cell_bounds_df = pd.read_parquet(cell_bounds_path)

    if not nuc_bounds_path.exists():
        print(f"    Warning: nucleus_boundaries.parquet not found in {target_dir}")
        nuc_bounds_df = pd.DataFrame(columns=['cell_id', 'vertex_x', 'vertex_y'])
    else:
        nuc_bounds_df = pd.read_parquet(nuc_bounds_path)

    # Load cells data
    cells_df = pd.read_parquet(cells_path)

    if verbose:
        print(f"    Loaded {len(cells_df)} cells from {cells_path.name}")

    # Get morphology shape for masks (if morphology.ome.tif exists)
    mask_shape = get_morphology_shape(target_dir)
    if mask_shape is None and verbose:
        print(f"    Warning: morphology.ome.tif not found, masks group will not be created")

    # Create zarr store
    try:
        create_zarr_store(output_path, cells_df, cell_bounds_df, nuc_bounds_df,
                          mask_shape=mask_shape, verbose=verbose)
        return True
    except Exception as e:
        print(f"    Error creating zarr store: {e}")
        return False


def convert_all_punches(outs_dir: str, overwrite: bool = False, verbose: bool = False) -> tuple:
    """
    Convert all directories in outs_dir containing cells.parquet to zarr format.

    This is the main entry point for pipeline integration.

    Args:
        outs_dir: Path to the outs directory containing subdirectories with cells.parquet
        overwrite: If True, overwrite existing cells.zarr.zip files
        verbose: If True, print detailed progress

    Returns:
        tuple: (success_count, total_count)
    """
    outs_path = Path(outs_dir)

    if not outs_path.exists():
        raise FileNotFoundError(f"Output directory not found: {outs_dir}")

    # Find directories with cells.parquet
    target_dirs = find_valid_directories(outs_path, verbose)

    if not target_dirs:
        print(f"  No directories with cells.parquet found in {outs_dir}")
        return (0, 0)

    print(f"  Converting {len(target_dirs)} directories to cells.zarr.zip format...")

    success_count = 0
    for target_dir in target_dirs:
        if verbose:
            print(f"  Processing {target_dir.name}...")
        if process_directory(target_dir, overwrite, verbose):
            success_count += 1
        elif not verbose:
            print(f"    Failed: {target_dir.name}")

    return (success_count, len(target_dirs))


def main():
    """CLI entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description='Convert cells.parquet files to cells.zarr.zip format'
    )
    parser.add_argument('--input-dir', required=True, type=Path,
                        help='Directory containing subdirectories with cells.parquet files')
    parser.add_argument('--subdir', type=str, default=None,
                        help='Process only a specific subdirectory by name')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing cells.zarr.zip files')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Find directories to process
    if args.subdir:
        # Process a specific subdirectory
        target_dir = input_dir / args.subdir
        if not target_dir.exists():
            print(f"Error: Subdirectory not found: {target_dir}")
            return 1
        if not has_required_files(target_dir):
            print(f"Error: {target_dir} does not contain cells.parquet")
            return 1
        target_dirs = [target_dir]
    else:
        # Find all subdirectories with required files
        if args.verbose:
            print(f"Scanning {input_dir} for directories with cells.parquet...")
        target_dirs = find_valid_directories(input_dir, args.verbose)
        if not target_dirs:
            print(f"Error: No subdirectories with cells.parquet found in {input_dir}")
            return 1

    print(f"Found {len(target_dirs)} directories to process...")

    success_count = 0
    for target_dir in target_dirs:
        print(f"\nProcessing {target_dir.name}...")
        if process_directory(target_dir, args.overwrite, args.verbose):
            success_count += 1

    print(f"\nCompleted: {success_count}/{len(target_dirs)} directories processed successfully")
    return 0 if success_count == len(target_dirs) else 1


if __name__ == '__main__':
    exit(main())
