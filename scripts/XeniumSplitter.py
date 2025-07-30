import pandas as pd
import numpy as np
import h5py
import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Set, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
from datetime import datetime


class XeniumDataSplitter:
    def __init__(self, xenium_dir: str, punch_coordinates_file: str, output_base_dir: str):
        """
        Initialize the Xenium data splitter for coordinate-based splitting.
        
        Args:
            xenium_dir: Path to directory containing Xenium files
            punch_coordinates_file: Path to all_punch_coordinates.csv file
            output_base_dir: Base directory for output subdirectories
        """
        self.xenium_dir = Path(xenium_dir)
        self.punch_coordinates_file = Path(punch_coordinates_file)
        self.output_base_dir = Path(output_base_dir)
        
        # File paths
        self.h5_file = self.xenium_dir / "cell_feature_matrix.h5"
        self.cells_file = self.xenium_dir / "cells.parquet"
        self.cell_boundaries_file = self.xenium_dir / "cell_boundaries.parquet"
        self.nucleus_boundaries_file = self.xenium_dir / "nucleus_boundaries.parquet"
        self.transcripts_file = self.xenium_dir / "transcripts.parquet"
        self.experiment_file = self.xenium_dir / "experiment.xenium"
        
        # Data containers
        self.punch_coordinates_df = None
        self.punch_ids = None
        self.punch_coordinates = {}  # Store coordinates for each punch_id
        self.cells_df = None  # Store cells data for filtering
        
    def load_cells_data(self):
        """Load cells data for coordinate-based filtering."""
        print("Loading cells data...")
        self.cells_df = pd.read_parquet(self.cells_file)
        print(f"Loaded {len(self.cells_df)} cells")
        
    def create_output_directories(self):
        """Create output directories for each punch_id."""
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        for punch_id in self.punch_ids:
            punch_dir = self.output_base_dir / f"punch_{punch_id}"
            punch_dir.mkdir(exist_ok=True)
            print(f"Created directory: {punch_dir}")
    
    def get_cells_in_punch(self, punch_id: str) -> pd.DataFrame:
        """Get cells within the coordinate bounds of a punch."""
        coords = self.punch_coordinates[punch_id]
        
        # Filter cells by coordinates
        cells_in_punch = self.cells_df[
            self.cells_df["x_centroid"].between(coords['x_min'], coords['x_max']) &
            self.cells_df["y_centroid"].between(coords['y_min'], coords['y_max'])
        ]
        
        return cells_in_punch
    
    def split_h5_matrix_optimized(self, n_processes=None):
        """
        Split the HDF5 cell feature matrix by punch_id using coordinate-based filtering.
        
        Args:
            n_processes: Number of processes to use. If None, uses CPU count.
        """
        print("Splitting cell feature matrix (coordinate-based)...")
        
        if n_processes is None:
            n_processes = min(mp.cpu_count(), len(self.punch_ids))
        
        with h5py.File(self.h5_file, 'r') as f:
            # Read the matrix structure once
            matrix_group = f['matrix']
            features = matrix_group['features']
            barcodes = matrix_group['barcodes'][:]
            data = matrix_group['data'][:]
            indices = matrix_group['indices'][:]
            indptr = matrix_group['indptr'][:]
            shape = matrix_group['shape'][:]
            
            # Convert barcodes to cell_ids
            cell_ids = [barcode.decode('utf-8') if isinstance(barcode, bytes) else str(barcode) 
                       for barcode in barcodes]
            cell_id_to_index = {cell_id: i for i, cell_id in enumerate(cell_ids)}
            
            # Prepare features data
            features_data = {}
            for key in features.keys():
                features_data[key] = features[key][:]
        
        # Prepare punch data for parallel processing
        punch_tasks = []
        for punch_id in self.punch_ids:
            # Get cells in this punch based on coordinates
            cells_in_punch = self.get_cells_in_punch(punch_id)
            punch_cell_ids = set(cells_in_punch['cell_id'].astype(str))
            
            # Get indices for cells that exist in the matrix
            punch_cell_indices = [cell_id_to_index[cell_id] 
                                for cell_id in punch_cell_ids 
                                if cell_id in cell_id_to_index]
            
            if punch_cell_indices:
                output_file = self.output_base_dir / f"punch_{punch_id}" / "cell_feature_matrix.h5"
                punch_tasks.append({
                    'punch_id': punch_id,
                    'cell_indices': punch_cell_indices,
                    'output_file': output_file,
                    'barcodes': barcodes,
                    'data': data,
                    'indices': indices,
                    'indptr': indptr,
                    'shape': shape,
                    'features_data': features_data
                })
        
        # Process punches
        if n_processes > 1:
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                futures = [executor.submit(self._process_single_punch, task) for task in punch_tasks]
                
                for future in futures:
                    result = future.result()
                    print(f"Completed punch {result['punch_id']}: {result['cell_count']} cells")
        else:
            for task in punch_tasks:
                result = self._process_single_punch(task)
                print(f"Completed punch {result['punch_id']}: {result['cell_count']} cells")
    
    @staticmethod
    def _process_single_punch(task):
        """Process a single punch in a separate process."""
        punch_id = task['punch_id']
        cell_indices = task['cell_indices']
        output_file = task['output_file']
        barcodes = task['barcodes']
        data = task['data']
        indices = task['indices']
        indptr = task['indptr']
        shape = task['shape']
        features_data = task['features_data']
        
        # Create new matrix for this punch
        with h5py.File(output_file, 'w') as out_f:
            matrix_out = out_f.create_group('matrix')
            
            # Copy features
            features_out = matrix_out.create_group('features')
            for key, data_array in features_data.items():
                features_out.create_dataset(key, data=data_array)
            
            # Filter barcodes
            punch_barcodes = [barcodes[i] for i in cell_indices]
            matrix_out.create_dataset('barcodes', data=punch_barcodes)
            
            # Extract sparse matrix data for selected cells
            total_elements = sum(indptr[idx + 1] - indptr[idx] for idx in cell_indices)
            
            new_data = np.empty(total_elements, dtype=data.dtype)
            new_indices = np.empty(total_elements, dtype=indices.dtype)
            new_indptr = np.zeros(len(cell_indices) + 1, dtype=indptr.dtype)
            
            current_pos = 0
            for i, cell_idx in enumerate(cell_indices):
                start_idx = indptr[cell_idx]
                end_idx = indptr[cell_idx + 1]
                cell_size = end_idx - start_idx
                
                new_data[current_pos:current_pos + cell_size] = data[start_idx:end_idx]
                new_indices[current_pos:current_pos + cell_size] = indices[start_idx:end_idx]
                current_pos += cell_size
                new_indptr[i + 1] = current_pos
            
            matrix_out.create_dataset('data', data=new_data)
            matrix_out.create_dataset('indices', data=new_indices)
            matrix_out.create_dataset('indptr', data=new_indptr)
            matrix_out.create_dataset('shape', data=[shape[0], len(cell_indices)])
        
        return {'punch_id': punch_id, 'cell_count': len(cell_indices)}
    
    def split_parquet_files(self):
        """Split parquet files by punch_id using coordinate-based filtering."""
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
                # Get cells in this punch
                cells_in_punch = self.get_cells_in_punch(punch_id)
                punch_cell_ids = set(cells_in_punch['cell_id'].astype(str))
                
                # Filter dataframe
                if 'cell_id' in df.columns:
                    punch_df = df[df['cell_id'].astype(str).isin(punch_cell_ids)]
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
    
    def split_transcripts_file(self):
        """Split transcripts.parquet file by punch_id using coordinate-based filtering."""
        if not self.transcripts_file.exists():
            print("Warning: transcripts.parquet not found, skipping...")
            return
        
        print("Splitting transcripts.parquet...")
        transcripts_df = pd.read_parquet(self.transcripts_file)
        
        for punch_id in self.punch_ids:
            coords = self.punch_coordinates[punch_id]
            
            # Filter transcripts by coordinates
            punch_transcripts = transcripts_df[
                transcripts_df["x_location"].between(coords['x_min'], coords['x_max']) &
                transcripts_df["y_location"].between(coords['y_min'], coords['y_max'])
            ]
            
            if len(punch_transcripts) == 0:
                print(f"Warning: No transcripts found for punch_id {punch_id}")
                continue
            
            # Save filtered transcripts
            output_file = self.output_base_dir / f"punch_{punch_id}" / "transcripts.parquet"
            punch_transcripts.to_parquet(output_file, index=False)
            print(f"Saved transcripts.parquet for punch {punch_id}: {len(punch_transcripts)} transcripts")
    
    def create_experiment_files(self, modifications: Dict = None):
        """Copy experiment.xenium files for each punch without modifications."""
        print("Copying experiment.xenium files...")
        
        # Load original experiment file
        with open(self.experiment_file, 'r') as f:
            experiment_data = json.load(f)
        
        for punch_id in self.punch_ids:
            # Copy the experiment data without modifications
            output_file = self.output_base_dir / f"punch_{punch_id}" / "experiment.xenium"
            with open(output_file, 'w') as f:
                json.dump(experiment_data, f, indent=2)
            
            print(f"Copied experiment.xenium for punch {punch_id}")
    
    def create_cell_stats_summary(self):
        """Create a summary of cells per punch based on coordinate filtering."""
        print("Creating cell stats summary...")
        
        summary_data = []
        for punch_id in self.punch_ids:
            cells_in_punch = self.get_cells_in_punch(punch_id)
            
            summary_data.append({
                'punch_id': punch_id,
                'treatment': self.punch_coordinates[punch_id].get('treatment'),
                'tissue': self.punch_coordinates[punch_id].get('tissue'),
                'replicate': self.punch_coordinates[punch_id].get('replicate'),
                'num_cells': len(cells_in_punch),
                'x_min': self.punch_coordinates[punch_id]['x_min'],
                'x_max': self.punch_coordinates[punch_id]['x_max'],
                'y_min': self.punch_coordinates[punch_id]['y_min'],
                'y_max': self.punch_coordinates[punch_id]['y_max'],
                'region_area': self.punch_coordinates[punch_id].get('region_area'),
                'subdir': str(self.output_base_dir / f"punch_{punch_id}")
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_base_dir / "punch_cell_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved cell summary to: {summary_file}")
        
        return summary_df
    
    def split_all_data(self, experiment_modifications: Dict = None, n_processes=4):
        """
        Main method to split all data files by punch_id using coordinate-based filtering.
        
        Args:
            experiment_modifications: Dictionary of modifications for experiment.xenium files (ignored)
            n_processes: Number of CPU used for parrallel process H5 matrix splitting
        """
        print("Starting Xenium data splitting (coordinate-based)...")
        
        # Load coordinates from the provided file
        print(f"Loading punch coordinates from: {self.punch_coordinates_file}")
        if not self.punch_coordinates_file.exists():
            raise FileNotFoundError(f"Coordinates file not found: {self.punch_coordinates_file}")
        
        self.punch_coordinates_df = pd.read_csv(self.punch_coordinates_file)
        
        # Get unique punch_ids
        self.punch_ids = self.punch_coordinates_df['punch_id'].unique()
        print(f"Found {len(self.punch_ids)} unique punch_ids: {self.punch_ids}")
        
        # Create coordinate mappings
        for _, row in self.punch_coordinates_df.iterrows():
            punch_id = row['punch_id']
            self.punch_coordinates[punch_id] = {
                'x_min': row['x_min'],
                'x_max': row['x_max'],
                'y_min': row['y_min'],
                'y_max': row['y_max'],
                'treatment': row.get('treatment'),
                'tissue': row.get('tissue'),
                'replicate': row.get('replicate'),
                'region_area': row.get('region_area')
            }
            print(f"Loaded coordinates for punch_id {punch_id}: x=[{row['x_min']:.1f}, {row['x_max']:.1f}], y=[{row['y_min']:.1f}, {row['y_max']:.1f}]")
        
        # Load cells data
        self.load_cells_data()
        
        # Create output directories
        self.create_output_directories()
        
        # Split each file type

        self.split_h5_matrix_optimized(n_processes=n_processes)
        self.split_parquet_files()
        self.split_transcripts_file()
        self.create_experiment_files(experiment_modifications)
        
        # Create summary
        summary_df = self.create_cell_stats_summary()
        
        print(f"\nData splitting completed! Output saved to: {self.output_base_dir}")
        print(f"Created {len(self.punch_ids)} subdirectories for punch_ids: {list(self.punch_ids)}")
        print(f"\nCell distribution:")
        for _, row in summary_df.iterrows():
            print(f"  Punch {row['punch_id']}: {row['num_cells']} cells")


# Example usage
if __name__ == "__main__":
    # Define paths
    xenium_directory = "/path/to/xenium/output"
    punch_coordinates_file = "/path/to/all_punch_coordinates.csv"
    output_directory = "/path/to/output"
    
    # Initialize splitter
    splitter = XeniumDataSplitter(
        xenium_dir=xenium_directory,
        punch_coordinates_file=punch_coordinates_file,
        output_base_dir=output_directory
    )
    
    # Run the splitting
    splitter.split_all_data(n_processes=10)