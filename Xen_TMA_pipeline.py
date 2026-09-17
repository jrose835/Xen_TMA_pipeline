#!/usr/bin/env python3
"""
Main executable script for the Xenium TMA processing pipeline (Coordinate-based).

This script orchestrates the complete workflow:
1. Run run_metrics.py to generate subregion-specific QC metrics and consolidated coordinate files
2. Run XeniumSplitter.py to split the Xenium data by punch subregions using coordinate-based filtering

Author: jrose
Date: 30Jul2025
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import pandas as pd
import numpy as np

# Add the scripts directory to the Python path so we can import our modules
script_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_dir))

try:
    from run_metrics import main as run_metrics_main
    from XeniumSplitter import XeniumDataSplitter
    from parquet_to_zarr import convert_all_punches
    from transcripts_parquet_to_zarr import convert_all_transcripts
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure run_metrics.py, XeniumSplitter.py, parquet_to_zarr.py, and transcripts_parquet_to_zarr.py are in the scripts/ directory")
    sys.exit(1)


def create_cell_blacklist(xenium_bundle_path, min_transcripts=None, min_area=None):
    """
    Create a blacklist of cell_ids based on transcript counts and cell area thresholds.
    
    Args:
        xenium_bundle_path: Path to Xenium bundle directory
        min_transcripts: Minimum transcript count threshold (optional)
        min_area: Minimum cell area threshold (optional)
        
    Returns:
        set: Set of cell_ids to blacklist
    """
    if min_transcripts is None and min_area is None:
        return set()
    
    print("Creating cell blacklist based on quality filters...")
    
    # Load cells data
    cells_path = os.path.join(xenium_bundle_path, 'cells.parquet')
    if not os.path.exists(cells_path):
        raise FileNotFoundError(f"cells.parquet not found: {cells_path}")
    
    cells_df = pd.read_parquet(cells_path)
    blacklist = set()
    
    # Filter by transcript count
    if min_transcripts is not None:
        if 'transcript_counts' not in cells_df.columns:
            print("Warning: transcript_counts column not found in cells.parquet, skipping transcript filter")
        else:
            low_transcript_cells = cells_df[cells_df['transcript_counts'] < min_transcripts]['cell_id']
            blacklist.update(low_transcript_cells.astype(str))
            print(f"Blacklisted {len(low_transcript_cells)} cells with < {min_transcripts} transcripts")
    
    # Filter by cell area
    if min_area is not None:
        if 'cell_area' not in cells_df.columns:
            print("Warning: cell_area column not found in cells.parquet, skipping area filter")
        else:
            small_area_cells = cells_df[cells_df['cell_area'] < min_area]['cell_id']
            blacklist.update(small_area_cells.astype(str))
            print(f"Blacklisted {len(small_area_cells)} cells with < {min_area} area")
    
    print(f"Total cells blacklisted: {len(blacklist)}")
    return blacklist


def validate_paths(args):
    """Validate that all required input paths exist."""
    required_paths = [
        (args.xenium_bundle_path, "Xenium bundle directory"),
    ]
    
    # Check cell_stats_dir or coordinates_file
    if args.coordinates_file:
        required_paths.append((args.coordinates_file, "Coordinates file"))
    elif args.cell_stats_dir:
        required_paths.append((args.cell_stats_dir, "Cell stats/coordinates directory"))
    
    for path, description in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{description} not found: {path}")
    
    # Check for required files in xenium bundle
    xenium_files = [
        "cell_feature_matrix.h5",
        "cells.parquet", 
        "nucleus_boundaries.parquet",
        "cell_boundaries.parquet",
        "transcripts.parquet",
        "experiment.xenium"
    ]
    
    missing_files = []
    for file in xenium_files:
        file_path = Path(args.xenium_bundle_path) / file
        if not file_path.exists():
            missing_files.append(file)
    
    if missing_files:
        raise FileNotFoundError(f"Required files missing from Xenium bundle: {missing_files}")


def run_metrics_step(args, cell_blacklist):
    """Run the metrics calculation step."""
    print("=" * 60)
    print("STEP 1: Running QC metrics calculation")
    print("=" * 60)
    
    try:
        # Determine if we're using coordinates file or cell_stats_dir
        if args.coordinates_file and not args.cell_stats_dir:
            # Use coordinates file for metrics
            print(f"Using coordinates file for metrics: {args.coordinates_file}")
            run_metrics_main(
                xenium_bundle_path=args.xenium_bundle_path,
                cell_stats_dir=None,  # Signal to use coordinates file
                output_dir=args.output_dir,
                qv_threshold=args.qv_threshold,
                coordinates_file=args.coordinates_file,
                cell_blacklist=cell_blacklist
            )
        else:
            # Use cell_stats_dir for metrics
            print(f"Using cell_stats_dir for metrics: {args.cell_stats_dir}")
            run_metrics_main(
                xenium_bundle_path=args.xenium_bundle_path,
                cell_stats_dir=args.cell_stats_dir,
                output_dir=args.output_dir,
                qv_threshold=args.qv_threshold,
                cell_blacklist=cell_blacklist
            )
        
        print("✓ Metrics calculation completed successfully")
        
        # Verify output files were created
        expected_files = ["region_metrics.csv"]
        
        # all_punch_coordinates.csv is derived from the per-punch exports; all_cell_stats.csv
        # only exists when *_cells_stats.csv exports accompany them, so it is not required.
        if args.cell_stats_dir:
            expected_files.append("all_punch_coordinates.csv")
        
        missing_outputs = []
        for file in expected_files:
            file_path = Path(args.output_dir) / file
            if not file_path.exists():
                missing_outputs.append(file)
        
        if missing_outputs:
            raise FileNotFoundError(f"Expected output files not created: {missing_outputs}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error in metrics calculation: {e}")
        return False


def run_splitter_step(args, cell_blacklist):
    """Run the coordinate-based data splitting step."""
    print("\n" + "=" * 60)
    print("STEP 2: Splitting Xenium data by array regions (coordinate-based)")
    print("=" * 60)

    try:
        # Determine coordinates file path
        if args.coordinates_file:
            coordinates_file = args.coordinates_file
        else:
            # Use the file generated by step 1
            coordinates_file = Path(args.output_dir) / "all_punch_coordinates.csv"

        if not os.path.exists(coordinates_file):
            raise FileNotFoundError(f"Coordinates file not found: {coordinates_file}")

        # Filter coordinates to only those matching the current bundle.
        # When multiple pipeline runs share an output directory, the accumulated
        # all_punch_coordinates.csv may contain rows from different bundles.
        # Splitting all rows against the current bundle would overwrite earlier
        # outputs with data from the wrong TMA.
        coords_df = pd.read_csv(coordinates_file)
        bundle_path_str = str(args.xenium_bundle_path)
        if 'path' in coords_df.columns:
            matched = coords_df[coords_df['path'].astype(str) == bundle_path_str]
            if len(matched) < len(coords_df):
                n_skipped = len(coords_df) - len(matched)
                print(f"Filtering coordinates to current bundle: keeping {len(matched)}/{len(coords_df)} rows ({n_skipped} from other bundles skipped)")
                if len(matched) == 0:
                    raise ValueError(
                        f"No coordinates match the current bundle path '{bundle_path_str}'. "
                        f"Bundle paths in coordinates file: {coords_df['path'].unique().tolist()}"
                    )
                # Write filtered coordinates to a temp file for the splitter
                filtered_coords_file = Path(args.output_dir) / "all_punch_coordinates_filtered.csv"
                matched.to_csv(filtered_coords_file, index=False)
                coordinates_file = filtered_coords_file

        # Create outs subdirectory for split data
        outs_dir = Path(args.output_dir) / "outs"
        outs_dir.mkdir(exist_ok=True)

        # Load region metrics if available (for metrics_summary.csv generation)
        region_metrics_df = None
        region_metrics_path = Path(args.output_dir) / "region_metrics.csv"
        if region_metrics_path.exists():
            region_metrics_df = pd.read_csv(region_metrics_path)

        # Initialize the splitter with coordinate-based approach
        splitter = XeniumDataSplitter(
            xenium_dir=args.xenium_bundle_path,
            punch_coordinates_file=str(coordinates_file),
            output_base_dir=str(outs_dir),
            cell_blacklist=cell_blacklist
        )

        # Define experiment file modifications
        current_date = datetime.now().strftime('%Y-%m-%d')
        experiment_mods = None  # No modifications to experiment file

        # Run the splitting with specified number of processes
        splitter.split_all_data(
            experiment_modifications=experiment_mods,
            n_processes=args.n_processes,
            run_light=args.run_light,
            region_metrics_df=region_metrics_df
        )

        print("✓ Data splitting completed successfully")
        return True

    except Exception as e:
        print(f"✗ Error in data splitting: {e}")
        return False


def run_zarr_conversion_step(args):
    """Run the parquet to zarr conversion step for Xenium Explorer compatibility."""
    print("\n" + "=" * 60)
    print("STEP 3: Converting parquet files to zarr format")
    print("=" * 60)

    try:
        # Get the outs directory containing subdirectories
        outs_dir = Path(args.output_dir) / "outs"

        if not outs_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {outs_dir}")

        all_success = True

        # Convert cells.parquet to cells.zarr.zip
        print("\n  Converting cells.parquet files to cells.zarr.zip...")
        cells_success, cells_total = convert_all_punches(
            outs_dir=str(outs_dir),
            overwrite=args.overwrite_zarr,
            verbose=True,
            allow_truncation=args.allow_polygon_truncation
        )

        if cells_success == cells_total:
            print(f"  ✓ Cells zarr conversion completed ({cells_success}/{cells_total} directories)")
        else:
            print(f"  ✗ Cells zarr conversion partially failed ({cells_success}/{cells_total} directories)")
            all_success = False

        # Convert transcripts.parquet to transcripts.zarr.zip
        print("\n  Converting transcripts.parquet files to transcripts.zarr.zip...")
        transcripts_success, transcripts_total = convert_all_transcripts(
            outs_dir=str(outs_dir),
            overwrite=args.overwrite_zarr,
            verbose=True
        )

        if transcripts_success == transcripts_total:
            print(f"  ✓ Transcripts zarr conversion completed ({transcripts_success}/{transcripts_total} directories)")
        else:
            print(f"  ✗ Transcripts zarr conversion partially failed ({transcripts_success}/{transcripts_total} directories)")
            all_success = False

        if all_success:
            print(f"\n✓ All zarr conversions completed successfully")
            return True
        else:
            print(f"\n✗ Some zarr conversions failed")
            return False

    except Exception as e:
        print(f"✗ Error in zarr conversion: {e}")
        return False


def main():
    """Main function to orchestrate the entire pipeline."""
    parser = argparse.ArgumentParser(
        description="Xenium TMA processing pipeline - Coordinate-based splitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This pipeline splits Xenium data based on spatial coordinates only.
It filters cells and transcripts that fall within the coordinate bounds of each punch region.

Examples:
  # Using individual coordinate files from cell_stats_dir
  python Xen_TMA_pipeline.py \\
    --xenium_bundle_path /path/to/xenium/output \\
    --cell_stats_dir /path/to/cell_stats \\
    --output_dir /path/to/output

  # Using pre-existing consolidated coordinates file
  python Xen_TMA_pipeline.py \\
    --xenium_bundle_path /path/to/xenium/output \\
    --cell_stats_dir /path/to/cell_stats \\
    --coordinates_file /path/to/all_punch_coordinates.csv \\
    --output_dir /path/to/output

  # With quality filters
  python Xen_TMA_pipeline.py \\
    --xenium_bundle_path /path/to/xenium/output \\
    --cell_stats_dir /path/to/cell_stats \\
    --output_dir /path/to/output \\
    --min_transcripts 10 \\
    --min_area 50 \\
    --qv_threshold 25 \\
    --n_processes 8
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--xenium_bundle_path',
        required=True,
        help='Path to Xenium bundle output directory (contains cell_feature_matrix.h5, cells.parquet, etc.)'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Path to output directory for split data and metrics'
    )
    
    # Coordinate input options (one of these is required)
    coord_group = parser.add_mutually_exclusive_group(required=True)
    coord_group.add_argument(
        '--cell_stats_dir',
        help='Path to directory containing cell_stats and coordinates CSV files'
    )
    coord_group.add_argument(
        '--coordinates_file',
        help='Path to all_punch_coordinates.csv file (will be used for both metrics and splitting)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--qv_threshold',
        type=int,
        default=20,
        help='QV threshold for high quality transcripts (default: 20)'
    )
    parser.add_argument(
        '--n_processes',
        type=int,
        default=4,
        help='Number of processes for parallel data splitting (default: 4)'
    )
    
    # Quality filter arguments
    parser.add_argument(
        '--min_transcripts',
        type=int,
        help='Minimum transcript count per cell (cells below this will be blacklisted)'
    )
    parser.add_argument(
        '--min_area',
        type=float,
        help='Minimum cell area (cells below this will be blacklisted)'
    )

    # Zarr conversion arguments
    parser.add_argument(
        '--skip_zarr',
        action='store_true',
        help='Skip creating cells.zarr.zip and transcripts.zarr.zip files'
    )
    parser.add_argument(
        '--overwrite_zarr',
        action='store_true',
        help='Overwrite existing zarr.zip files (default: skip if exists)'
    )

    parser.add_argument(
        '--allow_polygon_truncation',
        action='store_true',
        help='cells.zarr.zip stores at most 25 vertices per polygon. By default a bundle with '
             'longer boundaries (Proseg segmentation) is refused; this keeps the first 25 vertices '
             'and warns, which stores a sector of each such cell rather than the cell (see BACKLOG.md)'
    )

    # Light mode argument
    parser.add_argument(
        '--run_light',
        action='store_true',
        help='Skip metrics_summary.csv, morphology cropping, auxiliary files, and zarr conversion'
    )

    args = parser.parse_args()
    
    # Print pipeline information
    print("Xenium TMA Processing Pipeline - Coordinate-based Splitting")
    print("=" * 60)
    print(f"Xenium bundle path: {args.xenium_bundle_path}")
    if args.cell_stats_dir:
        print(f"Cell stats directory: {args.cell_stats_dir}")
    else:
        print(f"Using coordinates file: {args.coordinates_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"QV threshold: {args.qv_threshold}")
    print(f"Number of processes: {args.n_processes}")
    
    # Print quality filters
    if args.min_transcripts or args.min_area:
        print("Quality filters:")
        if args.min_transcripts:
            print(f"  Minimum transcripts per cell: {args.min_transcripts}")
        if args.min_area:
            print(f"  Minimum cell area: {args.min_area}")
    else:
        print("No quality filters applied")
    print(f"Create zarr files: {not args.skip_zarr}")
    print(f"Run light mode: {args.run_light}")
    print("=" * 60)
    
    # Validate input paths
    try:
        validate_paths(args)
        print("✓ Input validation passed")
    except FileNotFoundError as e:
        print(f"✗ Input validation failed: {e}")
        sys.exit(1)
    
    # Create cell blacklist based on quality filters
    try:
        cell_blacklist = create_cell_blacklist(
            xenium_bundle_path=args.xenium_bundle_path,
            min_transcripts=args.min_transcripts,
            min_area=args.min_area
        )
    except Exception as e:
        print(f"✗ Error creating cell blacklist: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    success = False
    try:
        # Step 1: Run metrics calculation
        if not run_metrics_step(args, cell_blacklist):
            print("Pipeline failed at metrics calculation step")
            sys.exit(1)
        
        # Step 2: Run coordinate-based data splitting
        if not run_splitter_step(args, cell_blacklist):
            print("Pipeline failed at data splitting step")
            sys.exit(1)

        # Step 3: Convert parquet files to zarr format (skipped if --skip_zarr or --run_light)
        if not args.skip_zarr and not args.run_light:
            if not run_zarr_conversion_step(args):
                print("Pipeline failed at zarr conversion step")
                sys.exit(1)

        success = True
        
    except KeyboardInterrupt:
        print("\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error in pipeline: {e}")
        sys.exit(1)
    
    if success:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Output directory: {args.output_dir}")
        print("Files created:")
        print("  - region_metrics.csv (QC metrics for each punch)")
        if args.cell_stats_dir:
            print("  - all_punch_coordinates.csv (consolidated punch coordinates)")
            print("  - all_cell_stats.csv (aggregated cell statistics)")
        print("  - outs/punch_*/  (individual punch data directories)")
        print("  - outs/punch_cell_summary.csv (summary of cells per punch)")
        if not args.run_light:
            print("  - outs/*/metrics_summary.csv (10x format metrics for each punch)")
            print("  - outs/*/morphology.ome.tif (cropped morphology images)")
            print("  - outs/*/morphology_focus/ (cropped autofocus images)")
            print("  - outs/*/gene_panel.json (gene panel definition)")
            print("  - outs/*/analysis_summary.html (analysis summary)")
        if not args.skip_zarr and not args.run_light:
            print("  - outs/*/cells.zarr.zip (Xenium Explorer compatible cell files)")
            print("  - outs/*/transcripts.zarr.zip (Xenium Explorer compatible transcript files)")

        if cell_blacklist:
            print(f"\nQuality filters applied:")
            print(f"  - {len(cell_blacklist)} cells blacklisted and removed from outputs")
            print(f"  - Transcripts from blacklisted cells reassigned to 'UNASSIGNED'")

        print("\nNote: Data was split using coordinate-based filtering only.")
        print("Cells and transcripts were assigned to punches based on their spatial location.")
        if not args.skip_zarr and not args.run_light:
            print("Zarr files (cells and transcripts) created for Xenium Explorer compatibility.")
        if args.run_light:
            print("Run light mode enabled: metrics_summary, morphology cropping, auxiliary files, and zarr conversion skipped.")
        print("=" * 60)
    

if __name__ == "__main__":
    main()