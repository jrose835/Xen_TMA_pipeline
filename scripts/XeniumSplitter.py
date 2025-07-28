import pandas as pd
import numpy as np
import h5py
import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Set

class XeniumDataSplitter:
    def __init__(self, xenium_dir: str, cell_stats_file: str, output_base_dir: str):
        """
        Initialize the Xenium data splitter.
        
        Args:
            xenium_dir: Path to directory containing Xenium files
            cell_stats_file: Path to all_cell_stats.csv file
            output_base_dir: Base directory for output subdirectories
        """
        self.xenium_dir = Path(xenium_dir)
        self.cell_stats_file = Path(cell_stats_file)
        self.output_base_dir = Path(output_base_dir)
        
        # File paths
        self.h5_file = self.xenium_dir / "cell_feature_matrix.h5"
        self.cells_file = self.xenium_dir / "cells.parquet"
        self.cell_boundaries_file = self.xenium_dir / "cell_boundaries.parquet"
        self.nucleus_boundaries_file = self.xenium_dir / "nucleus_boundaries.parquet"
        self.experiment_file = self.xenium_dir / "experiment.xenium"
        
        # Data containers
        self.cell_stats = None
        self.punch_ids = None
        self.cell_id_to_punch = None
        
    def load_metadata(self):
        """Load cell statistics and create punch_id mappings."""
        print("Loading cell statistics...")
        self.cell_stats = pd.read_csv(self.cell_stats_file)
        
        # Get unique punch_ids
        self.punch_ids = self.cell_stats['punch_id'].unique()
        print(f"Found {len(self.punch_ids)} unique punch_ids: {self.punch_ids}")
        
        # Add subdir column
        self.cell_stats['subdir'] = self.cell_stats['punch_id'].apply(
        lambda pid: str(self.output_base_dir / f"punch_{pid}")
        )
    
        # Save the updated cell_stats with subdir info
        updated_stats_file = self.output_base_dir / "all_cell_stats_processed.csv"
        self.cell_stats.to_csv(updated_stats_file, index=False)
        print(f"Updated cell_stats with subdir column saved to: {updated_stats_file}")

        # Create mapping from cell_id to punch_id
        # self.cell_id_to_punch = dict(zip(self.cell_stats['cell_id'], self.cell_stats['punch_id']))
        # ^AI garbage not used in later parts of function
        
    def create_output_directories(self):
        """Create output directories for each punch_id."""
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        for punch_id in self.punch_ids:
            punch_dir = self.output_base_dir / f"punch_{punch_id}"
            punch_dir.mkdir(exist_ok=True)
            print(f"Created directory: {punch_dir}")
    
    def split_h5_matrix(self):
        """Split the HDF5 cell feature matrix by punch_id."""
        print("Splitting cell feature matrix...")
        
        with h5py.File(self.h5_file, 'r') as f:
            # Read the matrix structure
            matrix_group = f['matrix']
            features = matrix_group['features']
            barcodes = matrix_group['barcodes'][:]
            data = matrix_group['data'][:]
            indices = matrix_group['indices'][:]
            indptr = matrix_group['indptr'][:]
            shape = matrix_group['shape'][:]
            
            # Convert barcodes to cell_ids (assuming they match)
            cell_ids = [barcode.decode('utf-8') if isinstance(barcode, bytes) else str(barcode) for barcode in barcodes]
            
            for punch_id in self.punch_ids:
                print(f"Processing punch_id: {punch_id}")
                
                # Get cell indices for this punch
                punch_cell_ids = self.cell_stats[self.cell_stats['punch_id'] == punch_id]['Cell ID'].tolist()
                punch_cell_indices = [i for i, cell_id in enumerate(cell_ids) if cell_id in punch_cell_ids]
                
                if not punch_cell_indices:
                    print(f"Warning: No cells found for punch_id {punch_id}")
                    continue
                
                # Create new matrix for this punch
                output_file = self.output_base_dir / f"punch_{punch_id}" / "cell_feature_matrix.h5"
                
                with h5py.File(output_file, 'w') as out_f:
                    matrix_out = out_f.create_group('matrix')
                    
                    # Copy features (unchanged)
                    features_out = matrix_out.create_group('features')
                    for key in features.keys():
                        features_out.create_dataset(key, data=features[key][:])
                    
                    # Filter barcodes and data
                    punch_barcodes = [barcodes[i] for i in punch_cell_indices]
                    matrix_out.create_dataset('barcodes', data=punch_barcodes)
                    
                    # Extract sparse matrix data for selected cells
                    new_data = []
                    new_indices = []
                    new_indptr = [0]
                    
                    for cell_idx in punch_cell_indices:
                        start_idx = indptr[cell_idx]
                        end_idx = indptr[cell_idx + 1]
                        
                        cell_data = data[start_idx:end_idx]
                        cell_indices = indices[start_idx:end_idx]
                        
                        new_data.extend(cell_data)
                        new_indices.extend(cell_indices)
                        new_indptr.append(len(new_data))
                    
                    matrix_out.create_dataset('data', data=new_data)
                    matrix_out.create_dataset('indices', data=new_indices)
                    matrix_out.create_dataset('indptr', data=new_indptr)
                    matrix_out.create_dataset('shape', data=[shape[0], len(punch_cell_indices)])
                
                print(f"Saved matrix for punch {punch_id}: {len(punch_cell_indices)} cells")
    
    def split_parquet_files(self):
        """Split parquet files by punch_id."""
        parquet_files = [
            ('cells.parquet', self.cells_file),
            ('cell_boundaries.parquet', self.cell_boundaries_file),
            ('nucleus_boundaries.parquet', self.nucleus_boundaries_file)
        ]
        
        for filename, filepath in parquet_files:
            if not filepath.exists():
                print(f"Warning: {filename} not found, skipping...")
                continue
                
            print(f"Splitting {filename}...")
            df = pd.read_parquet(filepath)
            
            for punch_id in self.punch_ids:
                # Get cell_ids for this punch
                punch_cell_ids = set(self.cell_stats[self.cell_stats['punch_id'] == punch_id]['Cell ID'])
                
                # Filter dataframe
                if 'cell_id' in df.columns:
                    punch_df = df[df['cell_id'].isin(punch_cell_ids)]
                else:
                    print(f"Warning: No cell_id column found in {filename}")
                    continue
                
                if len(punch_df) == 0:
                    print(f"Warning: No data found for punch_id {punch_id} in {filename}")
                    continue
                
                # Save filtered data
                output_file = self.output_base_dir / f"punch_{punch_id}" / filename
                punch_df.to_parquet(output_file, index=False)
                print(f"Saved {filename} for punch {punch_id}: {len(punch_df)} rows")
    
    def create_experiment_files(self, modifications: Dict = None):
        """
        Create modified experiment.xenium files for each punch.
        
        Args:
            modifications: Dictionary of modifications to apply to the experiment file
        """
        print("Creating experiment.xenium files...")
        
        # Load original experiment file
        with open(self.experiment_file, 'r') as f:
            experiment_data = json.load(f)
        
        for punch_id in self.punch_ids:
            # Make a copy of the experiment data
            punch_experiment = experiment_data.copy()
            
            # Apply modifications if provided
            if modifications:
                for key, value in modifications.items():
                    if callable(value):
                        # If value is a function, call it with punch_id
                        punch_experiment[key] = value(punch_id)
                    else:
                        punch_experiment[key] = value
            
            # Add punch_id identifier
            punch_experiment['punch_id'] = punch_id
            
            # Get cell count for this punch
            punch_cell_count = len(self.cell_stats[self.cell_stats['punch_id'] == punch_id])
            if 'cell_count' in punch_experiment:
                punch_experiment['cell_count'] = punch_cell_count
            
            # Save modified experiment file
            output_file = self.output_base_dir / f"punch_{punch_id}" / "experiment.xenium"
            with open(output_file, 'w') as f:
                json.dump(punch_experiment, f, indent=2)
            
            print(f"Saved experiment.xenium for punch {punch_id}")
    
    def split_all_data(self, experiment_modifications: Dict = None):
        """
        Main method to split all data files by punch_id.
        
        Args:
            experiment_modifications: Dictionary of modifications for experiment.xenium files
        """
        print("Starting Xenium data splitting...")
        
        # Load metadata and create mappings
        self.load_metadata()
        
        # Create output directories
        self.create_output_directories()
        
        # Split each file type
        #self.split_h5_matrix()
        self.split_parquet_files()
        #self.create_experiment_files(experiment_modifications)
        
        print(f"\nData splitting completed! Output saved to: {self.output_base_dir}")
        print(f"Created {len(self.punch_ids)} subdirectories for punch_ids: {list(self.punch_ids)}")

# Example usage
if __name__ == "__main__":
    # Define paths
    xenium_directory = "/data-raid/Xenium/prime/20250709__200006__XP1_TMA_Mouse_11_Organ_070925_/output-XETG00215__0054040__Region_1__20250709__200021"  # Update this path
    cell_stats_file = "/data-raid/Xenium/prime/20250709__200006__XP1_TMA_Mouse_11_Organ_070925_/QC_metrics/test/all_cell_stats.csv"  # Update this path
    output_directory = "/data-raid/Xenium/prime/20250709__200006__XP1_TMA_Mouse_11_Organ_070925_/XenSplitter/test"  # Update this path
    
    # Initialize splitter
    splitter = XeniumDataSplitter(
        xenium_dir=xenium_directory,
        cell_stats_file=cell_stats_file,
        output_base_dir=output_directory
    )
    
    # Define experiment file modifications (optional)
    experiment_mods = {
        'split_by': 'punch_id',
        'processing_date': '2024-01-01',  # Example modification
        'sample_name': lambda punch_id: f"sample_punch_{punch_id}"  # Dynamic modification
    }
    
    # Run the splitting
    splitter.split_all_data(experiment_modifications=experiment_mods)
