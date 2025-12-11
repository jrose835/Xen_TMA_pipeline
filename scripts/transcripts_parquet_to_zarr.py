#!/usr/bin/env python3
"""
Convert transcripts.parquet files from XenSplitter output to transcripts.zarr.zip format
required by Xenium Explorer.

This module can be used standalone or imported by the main pipeline.

Usage (standalone):
    python transcripts_parquet_to_zarr.py --input-dir ./outs --verbose
    python transcripts_parquet_to_zarr.py --input-dir ./outs --subdir punch_1a
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from zarr.storage import ZipStore


# Grid parameters for spatial binning (matching 10x Xenium format)
GRID_SPACING = 256.0  # microns per grid cell


def build_gene_mappings(transcripts_df: pd.DataFrame) -> tuple:
    """
    Build gene name lists and mappings from transcript data.

    Returns:
        gene_names: sorted list of unique gene/feature names
        gene_index_map: dict mapping gene name to index
        codeword_gene_names: list of gene names ordered by codeword_index
        codeword_gene_mapping: list mapping codeword index to gene index
        codeword_count: number of unique codewords
    """
    # Get unique feature names and sort them
    gene_names = sorted(transcripts_df['feature_name'].unique().tolist())
    gene_index_map = {name: idx for idx, name in enumerate(gene_names)}

    # Build codeword mappings
    codeword_df = transcripts_df[['codeword_index', 'feature_name']].drop_duplicates()
    codeword_df = codeword_df.sort_values('codeword_index')

    codeword_gene_names = []
    codeword_gene_mapping = []
    codeword_count = int(codeword_df['codeword_index'].max()) + 1

    # Create mapping for each codeword
    codeword_to_gene = dict(zip(codeword_df['codeword_index'], codeword_df['feature_name']))
    for cw_idx in range(codeword_count):
        if cw_idx in codeword_to_gene:
            gene_name = codeword_to_gene[cw_idx]
            codeword_gene_names.append(gene_name)
            codeword_gene_mapping.append(gene_index_map[gene_name])
        else:
            # Handle missing codewords
            codeword_gene_names.append(f'UnassignedCodeword_{cw_idx:04d}')
            codeword_gene_mapping.append(len(gene_names))

    return gene_names, gene_index_map, codeword_gene_names, codeword_gene_mapping, codeword_count


def build_category_arrays(transcripts_df: pd.DataFrame, gene_names: list,
                          codeword_count: int) -> tuple:
    """
    Build category boolean arrays for genes and codewords.

    Categories (bit positions):
        0: is_gene (predesigned gene)
        1: blank/control codeword
        2: negative control probe
        3: unassigned codeword
        4-7: reserved
    """
    n_genes = len(gene_names)

    # Gene category array (n_genes, 8) bool
    gene_category = np.zeros((n_genes, 8), dtype=bool)

    # Codeword category array (codeword_count, 8) bool
    codeword_category = np.zeros((codeword_count, 8), dtype=bool)

    # Build category info from the dataframe
    category_df = transcripts_df[['feature_name', 'codeword_index', 'codeword_category', 'is_gene']].drop_duplicates()

    gene_index_map = {name: idx for idx, name in enumerate(gene_names)}

    for _, row in category_df.iterrows():
        gene_name = row['feature_name']
        cw_idx = int(row['codeword_index'])
        cw_cat = str(row['codeword_category']).lower()
        is_gene = bool(row['is_gene'])

        if gene_name in gene_index_map:
            gene_idx = gene_index_map[gene_name]

            # Set is_gene flag
            gene_category[gene_idx, 0] = is_gene

            # Set category flags based on codeword_category
            if 'blank' in cw_cat or 'control' in cw_cat:
                gene_category[gene_idx, 1] = True
            if 'negcontrol' in cw_cat.replace('_', '').replace('-', ''):
                gene_category[gene_idx, 2] = True
            if 'unassigned' in cw_cat:
                gene_category[gene_idx, 3] = True

        # Set codeword category
        if 0 <= cw_idx < codeword_count:
            codeword_category[cw_idx, 0] = is_gene
            if 'blank' in cw_cat or 'control' in cw_cat:
                codeword_category[cw_idx, 1] = True
            if 'negcontrol' in cw_cat.replace('_', '').replace('-', ''):
                codeword_category[cw_idx, 2] = True
            if 'unassigned' in cw_cat:
                codeword_category[cw_idx, 3] = True

    return gene_category, codeword_category


def build_grid_groups(transcripts_df: pd.DataFrame, gene_index_map: dict,
                      n_genes: int, verbose: bool = False) -> dict:
    """
    Organize transcripts into spatial grid groups.

    Returns dict mapping grid key (e.g., "5,10") to arrays for that grid cell.
    """
    # Calculate grid coordinates for each transcript
    transcripts_df = transcripts_df.copy()
    transcripts_df['grid_x'] = (transcripts_df['x_location'] / GRID_SPACING).astype(int)
    transcripts_df['grid_y'] = (transcripts_df['y_location'] / GRID_SPACING).astype(int)
    transcripts_df['grid_key'] = transcripts_df['grid_x'].astype(str) + ',' + transcripts_df['grid_y'].astype(str)

    # Map feature names to gene indices
    transcripts_df['gene_idx'] = transcripts_df['feature_name'].map(gene_index_map)

    grids = {}
    grouped = transcripts_df.groupby('grid_key')

    for grid_key, group in grouped:
        n_transcripts = len(group)

        # Build arrays for this grid cell
        # id: (N, 2) uint32 - [transcript_id_low, transcript_id_high or dataset_suffix]
        transcript_ids = group['transcript_id'].values.astype(np.uint64)
        id_array = np.zeros((n_transcripts, 2), dtype=np.uint32)
        id_array[:, 0] = (transcript_ids & 0xFFFFFFFF).astype(np.uint32)
        id_array[:, 1] = ((transcript_ids >> 32) & 0xFFFFFFFF).astype(np.uint32)

        # uuid: (N, 2) uint32 - same as id for unique identifier
        uuid_array = id_array.copy()

        # valid: (N, 1) uint8 - all transcripts are valid (1)
        valid_array = np.ones((n_transcripts, 1), dtype=np.uint8)

        # status: (N, 1) uint8 - 0 for normal transcripts
        status_array = np.zeros((n_transcripts, 1), dtype=np.uint8)

        # location: (N, 3) float32 - [x, y, z]
        location_array = np.column_stack([
            group['x_location'].values,
            group['y_location'].values,
            group['z_location'].values
        ]).astype(np.float32)

        # gene_identity: (N, 1) uint16 - gene index
        gene_identity = group['gene_idx'].values.astype(np.uint16).reshape(-1, 1)

        # quality_score: (N, 1) float32 - qv score
        quality_score = group['qv'].values.astype(np.float32).reshape(-1, 1)

        # codeword_identity: (N, 2) uint16 - [codeword_index, secondary (65535 for none)]
        cw_identity = np.zeros((n_transcripts, 2), dtype=np.uint16)
        cw_identity[:, 0] = group['codeword_index'].values.astype(np.uint16)
        cw_identity[:, 1] = 65535  # No secondary codeword

        # gene_offset: (n_genes+1, 4) uint32 - offset array for gene-based access
        # This is a sparse index for quickly finding transcripts by gene
        # Format: [start_idx, count, reserved1, reserved2] per gene
        gene_offset = np.zeros((n_genes + 1, 4), dtype=np.uint32)

        # Sort by gene index for building offset array
        sorted_indices = np.argsort(group['gene_idx'].values)
        sorted_genes = group['gene_idx'].values[sorted_indices]

        # Build gene offset index
        current_gene = -1
        start_idx = 0
        for i, gene_idx in enumerate(sorted_genes):
            if gene_idx != current_gene:
                if current_gene >= 0:
                    gene_offset[current_gene, 0] = start_idx
                    gene_offset[current_gene, 1] = i - start_idx
                current_gene = gene_idx
                start_idx = i
        # Handle last gene
        if current_gene >= 0:
            gene_offset[current_gene, 0] = start_idx
            gene_offset[current_gene, 1] = len(sorted_genes) - start_idx

        grids[grid_key] = {
            'id': id_array,
            'uuid': uuid_array,
            'valid': valid_array,
            'status': status_array,
            'location': location_array,
            'gene_identity': gene_identity,
            'quality_score': quality_score,
            'codeword_identity': cw_identity,
            'gene_offset': gene_offset
        }

    if verbose:
        print(f"    Created {len(grids)} grid cells")

    return grids


def create_zarr_store(output_path: Path, transcripts_df: pd.DataFrame,
                      verbose: bool = False) -> None:
    """
    Create transcripts.zarr.zip file with required structure for Xenium Explorer.
    """
    n_transcripts = len(transcripts_df)

    if verbose:
        print(f"    Creating zarr store with {n_transcripts} transcripts...")

    # Build gene and codeword mappings
    gene_names, gene_index_map, codeword_gene_names, codeword_gene_mapping, codeword_count = \
        build_gene_mappings(transcripts_df)

    n_genes = len(gene_names)

    if verbose:
        print(f"    Found {n_genes} unique genes/features, {codeword_count} codewords")

    # Build category arrays
    gene_category, codeword_category = build_category_arrays(
        transcripts_df, gene_names, codeword_count
    )

    # Get unique FOV names
    fov_names = sorted(transcripts_df['fov_name'].dropna().unique().tolist())

    # Build spatial grids
    grids = build_grid_groups(transcripts_df, gene_index_map, n_genes, verbose)

    # Calculate spatial extent for density metrics
    x_min = transcripts_df['x_location'].min()
    x_max = transcripts_df['x_location'].max()
    y_min = transcripts_df['y_location'].min()
    y_max = transcripts_df['y_location'].max()

    density_spacing = 20.0  # microns
    density_x_count = int(np.ceil((x_max - x_min) / density_spacing)) + 1
    density_y_count = int(np.ceil((y_max - y_min) / density_spacing)) + 1

    # Create zarr store (use zarr_format=2 for compatibility with zarr-python 2.x)
    store = ZipStore(str(output_path), mode='w')
    root = zarr.open(store, mode='w', zarr_format=2)

    # Set root attributes
    root.attrs['major_version'] = 5
    root.attrs['minor_version'] = 0
    root.attrs['name'] = 'RnaDataset'
    root.attrs['number_rnas'] = n_transcripts
    root.attrs['number_genes'] = n_genes
    root.attrs['spatial_units'] = 'micron'
    root.attrs['coordinate_space'] = 'global_micron'
    root.attrs['data_format'] = 0

    # Gene mappings
    root.attrs['gene_names'] = gene_names
    root.attrs['gene_index_map'] = gene_index_map
    root.attrs['codeword_count'] = codeword_count
    root.attrs['codeword_gene_names'] = codeword_gene_names
    root.attrs['codeword_gene_mapping'] = codeword_gene_mapping

    # FOV info
    root.attrs['fov_names'] = fov_names

    # Density metrics
    root.attrs['metrics_density_x_origin'] = float(x_min)
    root.attrs['metrics_density_x_spacing'] = density_spacing
    root.attrs['metrics_density_x_count'] = density_x_count
    root.attrs['metrics_density_y_origin'] = float(y_min)
    root.attrs['metrics_density_y_spacing'] = density_spacing
    root.attrs['metrics_density_y_count'] = density_y_count

    # Create category arrays
    root['gene_category'] = gene_category
    root['codeword_category'] = codeword_category

    # Create grid groups
    grids_group = root.create_group('grids')
    level_group = grids_group.create_group('0')  # Level 0 is the base resolution

    for grid_key, arrays in grids.items():
        grid_group = level_group.create_group(grid_key)
        for arr_name, arr_data in arrays.items():
            grid_group[arr_name] = arr_data

    # Close store
    store.close()

    if verbose:
        print(f"    Created {output_path}")


def has_required_files(directory: Path) -> bool:
    """Check if a directory contains the required transcripts.parquet file."""
    return (directory / 'transcripts.parquet').exists()


def find_valid_directories(input_dir: Path, verbose: bool = False) -> list:
    """
    Find all subdirectories that contain transcripts.parquet files.

    Returns list of Path objects for valid directories.
    """
    valid_dirs = []

    # Check all immediate subdirectories
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir() and has_required_files(subdir):
            valid_dirs.append(subdir)
        elif subdir.is_dir() and verbose:
            print(f"    Skipping {subdir.name}: missing transcripts.parquet")

    return valid_dirs


def process_directory(target_dir: Path, overwrite: bool = False,
                      verbose: bool = False) -> bool:
    """
    Process a single directory and create transcripts.zarr.zip.

    Returns True if successful, False otherwise.
    """
    output_path = target_dir / 'transcripts.zarr.zip'

    # Check if output exists
    if output_path.exists() and not overwrite:
        if verbose:
            print(f"    Skipping {target_dir.name}: transcripts.zarr.zip already exists")
        return True

    # Check for required input file
    transcripts_path = target_dir / 'transcripts.parquet'

    if not transcripts_path.exists():
        print(f"    Error: transcripts.parquet not found in {target_dir}")
        return False

    # Load transcripts data
    transcripts_df = pd.read_parquet(transcripts_path)

    if verbose:
        print(f"    Loaded {len(transcripts_df)} transcripts from {transcripts_path.name}")

    # Create zarr store
    try:
        create_zarr_store(output_path, transcripts_df, verbose)
        return True
    except Exception as e:
        print(f"    Error creating zarr store: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_all_transcripts(outs_dir: str, overwrite: bool = False, verbose: bool = False) -> tuple:
    """
    Convert all directories in outs_dir containing transcripts.parquet to zarr format.

    This is the main entry point for pipeline integration.

    Args:
        outs_dir: Path to the outs directory containing subdirectories with transcripts.parquet
        overwrite: If True, overwrite existing transcripts.zarr.zip files
        verbose: If True, print detailed progress

    Returns:
        tuple: (success_count, total_count)
    """
    outs_path = Path(outs_dir)

    if not outs_path.exists():
        raise FileNotFoundError(f"Output directory not found: {outs_dir}")

    # Find directories with transcripts.parquet
    target_dirs = find_valid_directories(outs_path, verbose)

    if not target_dirs:
        print(f"  No directories with transcripts.parquet found in {outs_dir}")
        return (0, 0)

    print(f"  Converting {len(target_dirs)} directories to transcripts.zarr.zip format...")

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
        description='Convert transcripts.parquet files to transcripts.zarr.zip format'
    )
    parser.add_argument('--input-dir', required=True, type=Path,
                        help='Directory containing subdirectories with transcripts.parquet files')
    parser.add_argument('--subdir', type=str, default=None,
                        help='Process only a specific subdirectory by name')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing transcripts.zarr.zip files')
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
            print(f"Error: {target_dir} does not contain transcripts.parquet")
            return 1
        target_dirs = [target_dir]
    else:
        # Find all subdirectories with required files
        if args.verbose:
            print(f"Scanning {input_dir} for directories with transcripts.parquet...")
        target_dirs = find_valid_directories(input_dir, args.verbose)
        if not target_dirs:
            print(f"Error: No subdirectories with transcripts.parquet found in {input_dir}")
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
