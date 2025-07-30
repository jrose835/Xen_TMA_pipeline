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

# Add the scripts directory to the Python path so we can import our modules
script_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_dir))

try:
    from run_metrics import main as run_metrics_main
    from XeniumSplitter import XeniumDataSplitter
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure run_metrics.py and XeniumSplitter.py are in the scripts/ directory")
    sys.exit(1)


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


def run_metrics_step(args):
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
                coordinates_file=args.coordinates_file
            )
        else:
            # Use cell_stats_dir for metrics
            print(f"Using cell_stats_dir for metrics: {args.cell_stats_dir}")
            run_metrics_main(
                xenium_bundle_path=args.xenium_bundle_path,
                cell_stats_dir=args.cell_stats_dir,
                output_dir=args.output_dir,
                qv_threshold=args.qv_threshold
            )
        
        print("✓ Metrics calculation completed successfully")
        
        # Verify output files were created
        expected_files = ["region_metrics.csv"]
        
        # Only expect all_punch_coordinates.csv if using cell_stats_dir
        if args.cell_stats_dir:
            expected_files.extend(["all_punch_coordinates.csv", "all_cell_stats.csv"])
        
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
        
    try:
        run_metrics_main(
            xenium_bundle_path=args.xenium_bundle_path,
            cell_stats_dir=args.cell_stats_dir,
            output_dir=args.output_dir,
            qv_threshold=args.qv_threshold
        )
        print("✓ Metrics calculation completed successfully")
        
        # Verify output files were created
        expected_files = ["region_metrics.csv"]
        
        # Only expect all_punch_coordinates.csv and all_cell_stats.csv when using cell_stats_dir
        if args.cell_stats_dir:
            expected_files.extend(["all_punch_coordinates.csv", "all_cell_stats.csv"])
        
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


def run_splitter_step(args):
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
        
        # Create outs subdirectory for split data
        outs_dir = Path(args.output_dir) / "outs"
        outs_dir.mkdir(exist_ok=True)
        
        # Initialize the splitter with coordinate-based approach
        splitter = XeniumDataSplitter(
            xenium_dir=args.xenium_bundle_path,
            punch_coordinates_file=str(coordinates_file),
            output_base_dir=str(outs_dir)
        )
        
        # Define experiment file modifications
        current_date = datetime.now().strftime('%Y-%m-%d')
        experiment_mods = None  # No modifications to experiment file
        
        # Run the splitting with specified number of processes
        splitter.split_all_data(
            experiment_modifications=experiment_mods,
            n_processes=args.n_processes
        )
        
        print("✓ Data splitting completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error in data splitting: {e}")
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

  # With custom settings
  python Xen_TMA_pipeline.py \\
    --xenium_bundle_path /path/to/xenium/output \\
    --cell_stats_dir /path/to/cell_stats \\
    --output_dir /path/to/output \\
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
    print("=" * 60)
    
    # Validate input paths
    try:
        validate_paths(args)
        print("✓ Input validation passed")
    except FileNotFoundError as e:
        print(f"✗ Input validation failed: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    success = False
    try:
        # Step 1: Run metrics calculation
        if not run_metrics_step(args):
            print("Pipeline failed at metrics calculation step")
            sys.exit(1)
        
        # Step 2: Run coordinate-based data splitting
        if not run_splitter_step(args):
            print("Pipeline failed at data splitting step")
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
        print("\nNote: Data was split using coordinate-based filtering only.")
        print("Cells and transcripts were assigned to punches based on their spatial location.")
        print("=" * 60)
    

if __name__ == "__main__":
    main()