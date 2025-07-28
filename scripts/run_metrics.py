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


def main(
    xenium_bundle_path,
    cell_stats_dir,
    output_dir,
    qv_threshold=20
):
    # Load Xenium bundle files
    cells_path = os.path.join(xenium_bundle_path, 'cells.parquet')
    transcripts_path = os.path.join(xenium_bundle_path, 'transcripts.parquet')
    print("Loading files")
    cells_df = pd.read_parquet(cells_path)
    transcripts_df = pd.read_parquet(transcripts_path)

    # Prepare output dataframe
    metrics = []
    all_cell_stats = []

    # Find all coordinate files
    coord_files = [f for f in os.listdir(cell_stats_dir) if f.endswith('_coordinates.csv')]
    for coord_file in coord_files:
        print(f"Starting {coord_file}")
        coord_path = os.path.join(cell_stats_dir, coord_file)
        meta = parse_metadata_from_header(coord_path)

        # Load coordinates from coordinates.csv (after header)
        coords_df = pd.read_csv(coord_path, comment='#')
        x_min, x_max = coords_df["X"].min(), coords_df["X"].max()
        y_min, y_max = coords_df["Y"].min(), coords_df["Y"].max()

        # Find matching cell_stats file
        prefix = coord_file.replace('_coordinates.csv', '')
        cell_stats_file = prefix + '_cells_stats.csv'
        cell_stats_path = os.path.join(cell_stats_dir, cell_stats_file)
        if not os.path.exists(cell_stats_path):
            print(f"Warning: No cell_stats file for {coord_file}")
            continue

        # Read cell_stats CSV (skip header lines) and append meta columns
        cell_stats_df = pd.read_csv(cell_stats_path, comment='#')
        for key, value in meta.items():
            cell_stats_df[key] = value
        cell_stats_df["path"] = xenium_bundle_path
        all_cell_stats.append(cell_stats_df)

        # Filter cells by coordinates (bounding box of coords_df)
        filtered_cells = cells_df[
            cells_df["x_centroid"].between(x_min, x_max) &
            cells_df["y_centroid"].between(y_min, y_max)
        ]
        # Filter transcripts by coordinates (bounding box of coords_df)
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

        # percent empty cells: cells with zero transcripts assigned (cell_id != "UNASSIGNED")
        if 'transcript_counts' in filtered_cells.columns and num_cells > 0:
            num_empty_cells = len(filtered_cells[filtered_cells['transcript_counts'] == 0])
            percent_empty_cells = num_empty_cells/num_cells
        else:
            num_empty_cells = np.nan
            percent_empty_cells = np.nan

        # percent transcripts assigned: transcripts with a cell_id != "UNASSIGNED"
        if total_high_qv_transcripts > 0:
            percent_transcripts_assigned = total_transcripts_in_cells / total_high_qv_transcripts
        else:
            percent_transcripts_assigned = np.nan

        # Median transcripts per cell (only for cells in filtered_cells)
        if 'transcript_counts' in filtered_cells.columns:
            median_transcripts_per_cell = filtered_cells["transcript_counts"].median()
        else:
            median_transcripts_per_cell = np.nan

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
            'total_transcripts' : total_transcripts,
            'total_high_qv_transcripts': total_high_qv_transcripts,
            'total_transcripts_in_cells': total_transcripts_in_cells,
            'percent_transcripts_in_cells': percent_transcripts_in_cells,
            'total_empty_cells' : num_empty_cells,
            'percent_empty_cells': percent_empty_cells,
            'median_transcripts_per_cell': median_transcripts_per_cell,
        })
        print(f"Completed {coord_file}")

    os.makedirs(output_dir, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(output_dir, "region_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    if all_cell_stats:
        all_cell_stats_df = pd.concat(all_cell_stats, ignore_index=True)
        cell_stats_path = os.path.join(output_dir, "all_cell_stats.csv")
        all_cell_stats_df.to_csv(cell_stats_path, index=False)
        print(f"Saved aggregated cell stats to {cell_stats_path}")
    else:
        print("No cell_stats files found to aggregate.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calculate Xenium QC metrics")
    parser.add_argument('--xenium_bundle_path', required=True, help='Path to Xenium bundle output folder')
    parser.add_argument('--cell_stats_dir', required=True, help='Path to directory with cell_stats and coordinates CSVs')
    parser.add_argument('--output_dir', default="./", help='Path to directory for output files')
    parser.add_argument('--qv_threshold', type=int, default=20, help='QV threshold for high quality transcripts')
    args = parser.parse_args()
    main(
        xenium_bundle_path=args.xenium_bundle_path,
        cell_stats_dir=args.cell_stats_dir,
        output_dir=args.output_dir,
        qv_threshold=args.qv_threshold
    )