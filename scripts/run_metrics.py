import os
import pandas as pd
import numpy as np

def parse_metadata_from_header(csv_path):
    """Parse metadata from the header of the coordinates CSV."""
    with open(csv_path, 'r') as f:
        lines = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith('#'):
                f.seek(pos)
                break
            lines.append(line.strip())
    meta = {}
    for line in lines:
        if line.startswith('#Selection name:'):
            meta['selection_name'] = line.split(':', 1)[1].strip()
        elif line.startswith('#Area'):
            meta['region_area'] = float(line.split(':', 1)[1].strip())
    # Parse selection_name into punch_id, treatment, tissue, replicate
    if 'selection_name' in meta:
        parts = meta['selection_name'].split('_')
        if len(parts) == 4:
            meta['punch_id'], meta['treatment'], meta['tissue'], meta['replicate'] = parts
        elif len(parts) == 3:
            meta['punch_id'], meta['treatment'], meta['tissue'] = parts
            meta['replicate'] = "1"
            print(f"Warning: No replicate found in selection_name '{meta['selection_name']}' in {csv_path}, defaulting replicate to '1'.")
        else:
            meta['punch_id'] = meta['treatment'] = meta['tissue'] = meta['replicate'] = None
            print(f"Warning: Could not parse selection_name '{meta['selection_name']}' in {csv_path}. Setting all fields to None.")
    else:
        meta['punch_id'] = meta['treatment'] = meta['tissue'] = meta['replicate'] = None
        print(f"Warning: No selection_name found in header of {csv_path}. Setting all fields to None.")
    return meta

def process_region_metrics(cells_df, transcripts_df, x_min, x_max, y_min, y_max,
                         meta, xenium_bundle_path, qv_threshold, metrics):
    """Process metrics for a single punch region."""
    # Filter cells by coordinates (bounding box)
    filtered_cells = cells_df[
        cells_df["x_centroid"].between(x_min, x_max) &
        cells_df["y_centroid"].between(y_min, y_max)
    ]
    
    # Filter transcripts by coordinates (bounding box)
    filtered_transcripts = transcripts_df[
        transcripts_df["x_location"].between(x_min, x_max) &
        transcripts_df["y_location"].between(y_min, y_max)
    ]
    
    # Filter transcripts by QV
    high_qv_transcripts = filtered_transcripts[filtered_transcripts['qv'] > qv_threshold]

    # Calculate metrics
    num_cells = len(filtered_cells)
    cell_areas = filtered_cells['cell_area'] if 'cell_area' in filtered_cells.columns else pd.Series(dtype=float)
    min_cell_area = cell_areas.min() if not cell_areas.empty else np.nan
    median_cell_area = cell_areas.median() if not cell_areas.empty else np.nan
    max_cell_area = cell_areas.max() if not cell_areas.empty else np.nan
    total_cell_area = cell_areas.sum() if not cell_areas.empty else np.nan

    total_transcripts = len(filtered_transcripts)
    total_high_qv_transcripts = len(high_qv_transcripts)

    # Number of transcripts in cells (cell_id != "UNASSIGNED")
    transcripts_in_cells = high_qv_transcripts[high_qv_transcripts['cell_id'] != "UNASSIGNED"]
    total_transcripts_in_cells = len(transcripts_in_cells)

    # Percent transcripts in cells
    percent_transcripts_in_cells = (
        total_transcripts_in_cells / total_high_qv_transcripts
        if total_high_qv_transcripts > 0 else np.nan
    )

    # percent empty cells: cells with zero transcripts assigned
    if 'transcript_counts' in filtered_cells.columns and num_cells > 0:
        num_empty_cells = len(filtered_cells[filtered_cells['transcript_counts'] == 0])
        percent_empty_cells = num_empty_cells/num_cells
    else:
        num_empty_cells = np.nan
        percent_empty_cells = np.nan

    # percent transcripts assigned
    if total_high_qv_transcripts > 0:
        percent_transcripts_assigned = total_transcripts_in_cells / total_high_qv_transcripts
    else:
        percent_transcripts_assigned = np.nan

    # Median transcripts per cell
    if 'transcript_counts' in filtered_cells.columns:
        median_transcripts_per_cell = filtered_cells["transcript_counts"].median()
    else:
        median_transcripts_per_cell = np.nan

    # Predesigned vs Custom transcript metrics
    predesigned_transcripts = high_qv_transcripts[high_qv_transcripts['feature_name'].str.startswith('BLANK-') == False]
    custom_transcripts = high_qv_transcripts[high_qv_transcripts['feature_name'].str.startswith('BLANK-')]
    
    total_predesigned_transcripts = len(predesigned_transcripts)
    total_custom_transcripts = len(custom_transcripts)
    
    predesigned_transcripts_in_cells = predesigned_transcripts[predesigned_transcripts['cell_id'] != "UNASSIGNED"]
    custom_transcripts_in_cells = custom_transcripts[custom_transcripts['cell_id'] != "UNASSIGNED"]
    
    total_predesigned_transcripts_in_cells = len(predesigned_transcripts_in_cells)
    total_custom_transcripts_in_cells = len(custom_transcripts_in_cells)
    
    # Fraction of predesigned and custom transcripts decoded at Q20+
    fraction_predesigned_transcripts_decoded_q20 = (
        total_predesigned_transcripts / len(filtered_transcripts[filtered_transcripts['feature_name'].str.startswith('BLANK-') == False])
        if len(filtered_transcripts[filtered_transcripts['feature_name'].str.startswith('BLANK-') == False]) > 0 else np.nan
    )
    fraction_custom_transcripts_decoded_q20 = (
        total_custom_transcripts / len(filtered_transcripts[filtered_transcripts['feature_name'].str.startswith('BLANK-')])
        if len(filtered_transcripts[filtered_transcripts['feature_name'].str.startswith('BLANK-')]) > 0 else np.nan
    )

    # Nuclear transcripts
    nuclear_transcripts = high_qv_transcripts[high_qv_transcripts.get('nucleus_distance', pd.Series(dtype=float)) <= 0] if 'nucleus_distance' in high_qv_transcripts.columns else pd.DataFrame()
    total_nuclear_transcripts = len(nuclear_transcripts)
    
    # Nuclear transcript density per 100um2
    nuclear_transcripts_per_100um2 = (
        total_nuclear_transcripts / (meta.get('region_area', 0) / 1e6) * 100
        if meta.get('region_area', 0) > 0 else np.nan
    )
    
    # Decoded transcripts per 100um2
    decoded_transcripts_per_100um2 = (
        total_high_qv_transcripts / (meta.get('region_area', 0) / 1e6) * 100
        if meta.get('region_area', 0) > 0 else np.nan
    )
    
    # Cells per 100um2
    cells_per_100um2 = (
        num_cells / (meta.get('region_area', 0) / 1e6) * 100
        if meta.get('region_area', 0) > 0 else np.nan
    )

    # Gene counts per cell
    if 'cell_id' in transcripts_in_cells.columns and len(transcripts_in_cells) > 0:
        genes_per_cell = transcripts_in_cells.groupby('cell_id')['feature_name'].nunique()
        median_genes_per_cell = genes_per_cell.median()
        
        predesigned_genes_per_cell = predesigned_transcripts_in_cells.groupby('cell_id')['feature_name'].nunique()
        median_predesigned_genes_per_cell = predesigned_genes_per_cell.median() if len(predesigned_transcripts_in_cells) > 0 else np.nan
        
        custom_genes_per_cell = custom_transcripts_in_cells.groupby('cell_id')['feature_name'].nunique()
        median_custom_genes_per_cell = custom_genes_per_cell.median() if len(custom_transcripts_in_cells) > 0 else np.nan
        
        # Transcripts per cell by type
        predesigned_transcripts_per_cell = predesigned_transcripts_in_cells.groupby('cell_id').size()
        median_predesigned_transcripts_per_cell = predesigned_transcripts_per_cell.median() if len(predesigned_transcripts_in_cells) > 0 else np.nan
        
        custom_transcripts_per_cell = custom_transcripts_in_cells.groupby('cell_id').size()
        median_custom_transcripts_per_cell = custom_transcripts_per_cell.median() if len(custom_transcripts_in_cells) > 0 else np.nan
    else:
        median_genes_per_cell = np.nan
        median_predesigned_genes_per_cell = np.nan
        median_custom_genes_per_cell = np.nan
        median_predesigned_transcripts_per_cell = np.nan
        median_custom_transcripts_per_cell = np.nan

    metrics.append({
        'punch_id': meta.get('punch_id'),
        'treatment': meta.get('treatment'),
        'tissue': meta.get('tissue'),
        'replicate': meta.get('replicate'),
        'region_area': meta.get('region_area'),
        'num_cells': num_cells,
        'min_cell_area': min_cell_area,
        'median_cell_area': median_cell_area,
        'max_cell_area': max_cell_area,
        'total_cell_area': total_cell_area,
        'cells_per_100um2': cells_per_100um2,
        'total_transcripts': total_transcripts,
        'total_high_qv_transcripts': total_high_qv_transcripts,
        'fraction_transcripts_decoded_q20': total_high_qv_transcripts / total_transcripts if total_transcripts > 0 else np.nan,
        'total_predesigned_transcripts': total_predesigned_transcripts,
        'total_custom_transcripts': total_custom_transcripts,
        'fraction_predesigned_transcripts_decoded_q20': fraction_predesigned_transcripts_decoded_q20,
        'fraction_custom_transcripts_decoded_q20': fraction_custom_transcripts_decoded_q20,
        'total_nuclear_transcripts': total_nuclear_transcripts,
        'nuclear_transcripts_per_100um2': nuclear_transcripts_per_100um2,
        'decoded_transcripts_per_100um2': decoded_transcripts_per_100um2,
        'total_transcripts_in_cells': total_transcripts_in_cells,
        'percent_transcripts_in_cells': percent_transcripts_in_cells,
        'fraction_transcripts_assigned': percent_transcripts_assigned,
        'total_empty_cells': num_empty_cells,
        'percent_empty_cells': percent_empty_cells,
        'fraction_empty_cells': percent_empty_cells,
        'median_genes_per_cell': median_genes_per_cell,
        'median_predesigned_genes_per_cell': median_predesigned_genes_per_cell,
        'median_custom_genes_per_cell': median_custom_genes_per_cell,
        'median_transcripts_per_cell': median_transcripts_per_cell,
        'median_predesigned_transcripts_per_cell': median_predesigned_transcripts_per_cell,
        'median_custom_transcripts_per_cell': median_custom_transcripts_per_cell,
    })

def main(
    xenium_bundle_path,
    cell_stats_dir=None,
    output_dir="./",
    qv_threshold=20,
    coordinates_file=None
):
    # Load Xenium bundle files
    cells_path = os.path.join(xenium_bundle_path, 'cells.parquet')
    transcripts_path = os.path.join(xenium_bundle_path, 'transcripts.parquet')
    print("Loading files")
    cells_df = pd.read_parquet(cells_path)
    transcripts_df = pd.read_parquet(transcripts_path)

    # Prepare output dataframes
    metrics = []
    all_cell_stats = []
    all_coordinates = []
    
    # Determine source of coordinates
    if coordinates_file:
        # Use pre-existing coordinates file
        print(f"Using coordinates from: {coordinates_file}")
        coords_df = pd.read_csv(coordinates_file)
        
        # Process each punch from the coordinates file
        for idx, row in coords_df.iterrows():
            print(f"Processing punch_id: {row['punch_id']}")
            
            # Extract bounds
            x_min, x_max = row['x_min'], row['x_max']
            y_min, y_max = row['y_min'], row['y_max']
            
            # Get metadata from the coordinates file
            meta = {
                'punch_id': row.get('punch_id'),
                'treatment': row.get('treatment'),
                'tissue': row.get('tissue'),
                'replicate': row.get('replicate'),
                'selection_name': row.get('selection_name'),
                'region_area': row.get('region_area')
            }
            
            # Process metrics for this punch
            process_region_metrics(
                cells_df, transcripts_df, x_min, x_max, y_min, y_max,
                meta, xenium_bundle_path, qv_threshold, metrics
            )
        
        # Don't create new all_coordinates when using existing file
        save_coordinates = False
        
    elif cell_stats_dir:
        # Use individual coordinate files from directory
        coord_files = [f for f in os.listdir(cell_stats_dir) if f.endswith('_coordinates.csv')]
        
        for coord_file in coord_files:
            print(f"Starting {coord_file}")
            coord_path = os.path.join(cell_stats_dir, coord_file)
            meta = parse_metadata_from_header(coord_path)

            # Load coordinates from coordinates.csv (after header)
            coords_df = pd.read_csv(coord_path, comment='#')
            x_min, x_max = coords_df["X"].min(), coords_df["X"].max()
            y_min, y_max = coords_df["Y"].min(), coords_df["Y"].max()

            # Store punch coordinates for consolidated output
            punch_coords = {
                'punch_id': meta.get('punch_id'),
                'treatment': meta.get('treatment'),
                'tissue': meta.get('tissue'),
                'replicate': meta.get('replicate'),
                'selection_name': meta.get('selection_name'),
                'region_area': meta.get('region_area'),
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'coordinate_file': coord_file,
                'path': xenium_bundle_path
            }
            all_coordinates.append(punch_coords)

            # Find matching cell_stats file
            prefix = coord_file.replace('_coordinates.csv', '')
            cell_stats_file = prefix + '_cells_stats.csv'
            cell_stats_path = os.path.join(cell_stats_dir, cell_stats_file)
            if os.path.exists(cell_stats_path):
                # Read cell_stats CSV (skip header lines) and append meta columns
                cell_stats_df = pd.read_csv(cell_stats_path, comment='#')
                for key, value in meta.items():
                    cell_stats_df[key] = value
                cell_stats_df["path"] = xenium_bundle_path
                all_cell_stats.append(cell_stats_df)

            # Process metrics for this punch
            process_region_metrics(
                cells_df, transcripts_df, x_min, x_max, y_min, y_max,
                meta, xenium_bundle_path, qv_threshold, metrics
            )
            print(f"Completed {coord_file}")
        
        save_coordinates = True
    else:
        raise ValueError("Either cell_stats_dir or coordinates_file must be provided")

    # Create output directory and save all files
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(output_dir, "region_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    # Save aggregated cell stats (only if using cell_stats_dir)
    if all_cell_stats:
        all_cell_stats_df = pd.concat(all_cell_stats, ignore_index=True)
        cell_stats_path = os.path.join(output_dir, "all_cell_stats.csv")
        all_cell_stats_df.to_csv(cell_stats_path, index=False)
        print(f"Saved aggregated cell stats to {cell_stats_path}")

    # Save consolidated punch coordinates (only if using cell_stats_dir)
    if save_coordinates and all_coordinates:
        coordinates_df = pd.DataFrame(all_coordinates)
        coordinates_path = os.path.join(output_dir, "all_punch_coordinates.csv")
        coordinates_df.to_csv(coordinates_path, index=False)
        print(f"Saved consolidated punch coordinates to {coordinates_path}")
        print(f"Coordinates file contains {len(coordinates_df)} punch regions")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calculate Xenium QC metrics")
    parser.add_argument('--xenium_bundle_path', required=True, help='Path to Xenium bundle output folder')
    parser.add_argument('--cell_stats_dir', help='Path to directory with cell_stats and coordinates CSVs')
    parser.add_argument('--coordinates_file', help='Path to all_punch_coordinates.csv file')
    parser.add_argument('--output_dir', default="./", help='Path to directory for output files')
    parser.add_argument('--qv_threshold', type=int, default=20, help='QV threshold for high quality transcripts')
    args = parser.parse_args()
    
    if not args.cell_stats_dir and not args.coordinates_file:
        parser.error("Either --cell_stats_dir or --coordinates_file must be provided")
    
    main(
        xenium_bundle_path=args.xenium_bundle_path,
        cell_stats_dir=args.cell_stats_dir,
        output_dir=args.output_dir,
        qv_threshold=args.qv_threshold,
        coordinates_file=args.coordinates_file
    )