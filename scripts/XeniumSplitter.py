import pandas as pd
import numpy as np
import h5py
import json
import os
from pathlib import Path
import shutil
import tempfile
import pickle
from typing import Dict, List, Set, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
from datetime import datetime

# Optional imports for OME-TIFF handling
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class XeniumDataSplitter:
    def __init__(self, xenium_dir: str, punch_coordinates_file: str, output_base_dir: str, cell_blacklist: Set[str] = None):
        """
        Initialize the Xenium data splitter for coordinate-based splitting.
        
        Args:
            xenium_dir: Path to directory containing Xenium files
            punch_coordinates_file: Path to all_punch_coordinates.csv file
            output_base_dir: Base directory for output subdirectories
            cell_blacklist: Set of cell_ids to blacklist (optional)
        """
        self.xenium_dir = Path(xenium_dir)
        self.punch_coordinates_file = Path(punch_coordinates_file)
        self.output_base_dir = Path(output_base_dir)
        self.cell_blacklist = cell_blacklist or set()
        
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
        
        # Apply blacklist filtering
        if self.cell_blacklist:
            pre_blacklist_count = len(self.cells_df)
            self.cells_df = self.cells_df[~self.cells_df['cell_id'].astype(str).isin(self.cell_blacklist)]
            blacklisted_count = pre_blacklist_count - len(self.cells_df)
            print(f"Filtered out {blacklisted_count} blacklisted cells, {len(self.cells_df)} cells remaining")
        
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
        Uses file-based communication to avoid multiprocessing memory limits.
        
        Args:
            n_processes: Number of processes to use. If None, uses CPU count.
        """
        print("Splitting cell feature matrix (coordinate-based)...")
        
        if n_processes is None:
            n_processes = min(mp.cpu_count(), len(self.punch_ids))
        
        # Create temporary directory for shared data
        temp_dir = Path(tempfile.mkdtemp(prefix="xenium_split_"))
        
        try:
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
            
            # Save large arrays to temporary files instead of passing through queue
            shared_data_file = temp_dir / "shared_data.pkl"
            shared_data = {
                'barcodes': barcodes,
                'data': data,
                'indices': indices,
                'indptr': indptr,
                'shape': shape,
                'features_data': features_data,
                'cell_id_to_index': cell_id_to_index
            }
            
            with open(shared_data_file, 'wb') as f:
                pickle.dump(shared_data, f)
            
            # Prepare punch tasks (now only small data through queue)
            punch_tasks = []
            for punch_id in self.punch_ids:
                # Get cells in this punch based on coordinates (already filtered by blacklist)
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
                        'output_file': str(output_file),
                        'shared_data_file': str(shared_data_file)  # Only pass file path
                    })
            
            # Process punches in parallel
            if n_processes > 1:
                with ProcessPoolExecutor(max_workers=n_processes) as executor:
                    futures = [executor.submit(self._process_single_punch_from_file, task) for task in punch_tasks]
                    
                    for future in futures:
                        result = future.result()
                        print(f"Completed punch {result['punch_id']}: {result['cell_count']} cells")
            else:
                for task in punch_tasks:
                    result = self._process_single_punch_from_file(task)
                    print(f"Completed punch {result['punch_id']}: {result['cell_count']} cells")
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @staticmethod
    def _process_single_punch_from_file(task):
        """Process a single punch by loading shared data from file."""
        punch_id = task['punch_id']
        cell_indices = task['cell_indices']
        output_file = task['output_file']
        shared_data_file = task['shared_data_file']
        
        # Load shared data from file
        with open(shared_data_file, 'rb') as f:
            shared_data = pickle.load(f)
        
        barcodes = shared_data['barcodes']
        data = shared_data['data']
        indices = shared_data['indices']
        indptr = shared_data['indptr']
        shape = shared_data['shape']
        features_data = shared_data['features_data']
        
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
            
            # Apply blacklist to the dataframe if it has cell_id column
            if self.cell_blacklist and 'cell_id' in df.columns:
                pre_blacklist_count = len(df)
                df = df[~df['cell_id'].astype(str).isin(self.cell_blacklist)]
                blacklisted_count = pre_blacklist_count - len(df)
                if blacklisted_count > 0:
                    print(f"  Filtered {blacklisted_count} blacklisted cells from {filename}")
            
            for punch_id in self.punch_ids:
                # Get cells in this punch (already blacklist-filtered)
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
        
        # Apply blacklist to transcripts: reassign blacklisted cell transcripts to 'UNASSIGNED'
        if self.cell_blacklist:
            blacklisted_transcript_mask = transcripts_df['cell_id'].astype(str).isin(self.cell_blacklist)
            transcripts_df.loc[blacklisted_transcript_mask, 'cell_id'] = 'UNASSIGNED'
            if blacklisted_transcript_mask.sum() > 0:
                print(f"  Reassigned {blacklisted_transcript_mask.sum()} transcripts from blacklisted cells to 'UNASSIGNED'")
        
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

    def write_metrics_summary(self, punch_id: str, region_metrics: Dict = None):
        """
        Write a metrics_summary.csv file to a punch directory in 10x Xenium format.

        Args:
            punch_id: The punch ID
            region_metrics: Dictionary containing metrics for this punch (optional)
        """
        punch_dir = self.output_base_dir / f"punch_{punch_id}"
        coords = self.punch_coordinates[punch_id]

        # Load experiment metadata from original Xenium bundle
        if self.experiment_file.exists():
            with open(self.experiment_file, 'r') as f:
                experiment_data = json.load(f)
        else:
            experiment_data = {}

        # Extract run metadata
        run_name = experiment_data.get('run_name', '')
        cassette_name = experiment_data.get('cassette_name', '')

        # Create punch-specific region name
        treatment = coords.get('treatment', '')
        tissue = coords.get('tissue', '')
        replicate = coords.get('replicate', '')
        region_name = f"{punch_id}_{treatment}_{tissue}_{replicate}".strip('_')

        # Get panel information
        panel_name = experiment_data.get('panel', {}).get('name', '')
        panel_design_id = experiment_data.get('panel', {}).get('design_id', '')
        predesigned_panel_id = experiment_data.get('panel', {}).get('predesigned_id', '')

        # Use region_metrics if provided, otherwise use coords
        metrics = region_metrics if region_metrics else {}

        # Build metrics dictionary in 10x format
        metrics_summary = {
            'run_name': run_name,
            'cassette_name': cassette_name,
            'region_name': region_name,
            'panel_name': panel_name,
            'panel_design_id': panel_design_id,
            'predesigned_panel_id': predesigned_panel_id,
            'region_area': metrics.get('region_area', coords.get('region_area', '')),
            'total_cell_area': metrics.get('total_cell_area', ''),
            'total_high_quality_decoded_transcripts': metrics.get('total_high_qv_transcripts', ''),
            'fraction_transcripts_decoded_q20': metrics.get('fraction_transcripts_decoded_q20', ''),
            'fraction_predesigned_transcripts_decoded_q20': '',
            'fraction_custom_transcripts_decoded_q20': '',
            'nuclear_transcripts_per_100um2': metrics.get('nuclear_transcripts_per_100um2', ''),
            'decoded_transcripts_per_100um2': metrics.get('decoded_transcripts_per_100um2', ''),
            'adjusted_negative_control_probe_rate': '',
            'adjusted_negative_control_codeword_rate': '',
            'adjusted_genomic_control_probe_rate': '',
            'negative_control_probe_counts_per_control_per_cell': '',
            'genomic_control_probe_counts_per_control_per_cell': '',
            'estimated_number_of_false_positive_transcripts_per_cell': '',
            'estimated_number_of_false_positive_transcripts_per_cell_including_genomic_counts': '',
            'num_cells_detected': metrics.get('num_cells', ''),
            'fraction_empty_cells': metrics.get('percent_empty_cells', ''),
            'cells_per_100um2': metrics.get('cells_per_100um2', ''),
            'fraction_transcripts_assigned': metrics.get('percent_transcripts_in_cells', ''),
            'median_genes_per_cell': metrics.get('median_genes_per_cell', ''),
            'median_predesigned_genes_per_cell': '',
            'median_custom_genes_per_cell': '',
            'median_transcripts_per_cell': metrics.get('median_transcripts_per_cell', ''),
            'median_predesigned_transcripts_per_cell': '',
            'median_custom_transcripts_per_cell': '',
            'thickness_transcripts_high_quality': '',
            'stain_definition': '',
            'segmented_cell_stain_frac': '',
            'segmented_cell_boundary_frac': '',
            'segmented_cell_interior_frac': '',
            'segmented_cell_nuc_expansion_frac': '',
            'segmented_cell_boundary_count': '',
            'segmented_cell_interior_count': '',
            'segmented_cell_nuc_expansion_count': '',
            'fraction_of_ambiguous_nucleus_mask_pixels': '',
            'fraction_of_ambiguous_cell_mask_pixels': '',
            'fraction_of_nucleus_polygons_removed': '',
            'fraction_of_cell_polygons_removed': '',
            'fraction_of_nuclei_without_cell': '',
            'number_of_cell_non_simple_polygons': '',
            'number_of_cell_multi_polygons': '',
            'number_of_nucleus_non_simple_polygons': '',
            'number_of_nucleus_multi_polygons': '',
            'segmented_cell_imported_frac': '',
            'segmented_cell_imported_count': ''
        }

        # Create DataFrame and save
        metrics_df = pd.DataFrame([metrics_summary])
        output_path = punch_dir / "metrics_summary.csv"
        metrics_df.to_csv(output_path, index=False)
        print(f"Saved metrics_summary.csv for punch {punch_id}")

    def write_all_metrics_summaries(self, region_metrics_df: pd.DataFrame = None):
        """
        Write metrics_summary.csv files to all punch directories.

        Args:
            region_metrics_df: DataFrame containing metrics for all regions
        """
        print("Writing metrics_summary.csv files...")

        for punch_id in self.punch_ids:
            # Get metrics for this punch if provided
            punch_metrics = None
            if region_metrics_df is not None:
                metrics_row = region_metrics_df[region_metrics_df['punch_id'] == punch_id]
                if len(metrics_row) == 0:
                    try:
                        metrics_row = region_metrics_df[region_metrics_df['punch_id'] == int(punch_id)]
                    except (ValueError, TypeError):
                        pass

                if len(metrics_row) > 0:
                    punch_metrics = metrics_row.iloc[0].to_dict()

            self.write_metrics_summary(punch_id, punch_metrics)

    def _crop_ome_tiff(self, input_path: Path, output_path: Path,
                       x_min: float, x_max: float, y_min: float, y_max: float) -> bool:
        """
        Crop an OME-TIFF file to specified coordinate bounds.

        Args:
            input_path: Path to input OME-TIFF file
            output_path: Path for output cropped OME-TIFF file
            x_min, x_max, y_min, y_max: Coordinates in microns

        Returns:
            True if successful, False otherwise
        """
        if not HAS_TIFFFILE:
            return False

        # Fixed pixel size for Xenium: 0.2125 microns/pixel
        px_size = 0.2125

        try:
            # Convert micron coordinates to pixel coordinates
            px_x_min = int(x_min / px_size)
            px_x_max = int(x_max / px_size)
            px_y_min = int(y_min / px_size)
            px_y_max = int(y_max / px_size)

            # Read the TIFF file - use pages directly to avoid OME series issues
            with tifffile.TiffFile(input_path) as tif:
                # Check image dimensions from first page
                page = tif.pages[0]
                img_height, img_width = page.shape[:2]

                # Clamp coordinates to image bounds
                px_x_min = max(0, px_x_min)
                px_x_max = min(img_width, px_x_max)
                px_y_min = max(0, px_y_min)
                px_y_max = min(img_height, px_y_max)

                if px_x_max <= px_x_min or px_y_max <= px_y_min:
                    print(f"  Warning: Invalid crop region for {input_path.name}")
                    return False

                # Get number of pages/z-slices
                n_pages = len(tif.pages)

                # Read and crop each page individually to preserve Z-stack structure
                cropped_pages = []
                for p in tif.pages:
                    page_data = p.asarray()
                    cropped_page = page_data[px_y_min:px_y_max, px_x_min:px_x_max]
                    cropped_pages.append(cropped_page)

                # Stack into (Z, Y, X) array
                cropped = np.stack(cropped_pages, axis=0)

            # Calculate new dimensions
            crop_height = px_y_max - px_y_min
            crop_width = px_x_max - px_x_min
            n_z = cropped.shape[0]

            # Write cropped image with proper OME-TIFF structure for Xenium
            # Xenium morphology uses XYZCT dimension order with Z-slices
            try:
                tifffile.imwrite(
                    output_path,
                    cropped,
                    photometric='minisblack',
                    compression='zlib',
                    tile=(256, 256) if min(crop_height, crop_width) >= 256 else None,
                    metadata={
                        'axes': 'ZYX',
                        'PhysicalSizeX': px_size,
                        'PhysicalSizeY': px_size,
                        'PhysicalSizeZ': 3.0,  # Xenium default Z spacing
                        'PhysicalSizeXUnit': 'um',
                        'PhysicalSizeYUnit': 'um',
                        'PhysicalSizeZUnit': 'um',
                        'Channel': {'Name': 'DAPI'},
                    }
                )
            except Exception:
                # Fallback: write without compression
                tifffile.imwrite(
                    output_path,
                    cropped,
                    photometric='minisblack',
                    metadata={
                        'axes': 'ZYX',
                        'PhysicalSizeX': px_size,
                        'PhysicalSizeY': px_size,
                        'PhysicalSizeZ': 3.0,
                        'PhysicalSizeXUnit': 'um',
                        'PhysicalSizeYUnit': 'um',
                        'PhysicalSizeZUnit': 'um',
                        'Channel': {'Name': 'DAPI'},
                    }
                )

            print(f"  Cropped {input_path.name} to {crop_width}x{crop_height} pixels ({n_z} z-slices)")
            return True

        except Exception as e:
            print(f"  Error cropping {input_path.name}: {e}")
            return False

    def _crop_morphology_focus(self, input_path: Path, output_path: Path,
                               x_min: float, x_max: float, y_min: float, y_max: float) -> bool:
        """
        Crop a morphology_focus OME-TIFF file (single channel 2D image).

        Args:
            input_path: Path to input OME-TIFF file
            output_path: Path for output cropped OME-TIFF file
            x_min, x_max, y_min, y_max: Coordinates in microns

        Returns:
            True if successful, False otherwise
        """
        if not HAS_TIFFFILE:
            return False

        px_size = 0.2125

        try:
            # Convert micron coordinates to pixel coordinates
            px_x_min = int(x_min / px_size)
            px_x_max = int(x_max / px_size)
            px_y_min = int(y_min / px_size)
            px_y_max = int(y_max / px_size)

            # Read the single-page TIFF file directly (avoid OME series parsing)
            with tifffile.TiffFile(input_path) as tif:
                page = tif.pages[0]
                img_height, img_width = page.shape[:2]

                # Clamp coordinates
                px_x_min = max(0, px_x_min)
                px_x_max = min(img_width, px_x_max)
                px_y_min = max(0, px_y_min)
                px_y_max = min(img_height, px_y_max)

                if px_x_max <= px_x_min or px_y_max <= px_y_min:
                    print(f"  Warning: Invalid crop region for {input_path.name}")
                    return False

                # Read just the first page (single channel image)
                data = page.asarray()
                cropped = data[px_y_min:px_y_max, px_x_min:px_x_max]

            # Get channel name from filename (morphology_focus_0000 -> channel 0)
            channel_names = ['DAPI', 'ATP1A1/CD45/E-Cadherin', '18S', 'alphaSMA/Vimentin']
            try:
                channel_idx = int(input_path.stem.split('_')[-1])
                channel_name = channel_names[channel_idx] if channel_idx < len(channel_names) else f'Channel_{channel_idx}'
            except (ValueError, IndexError):
                channel_name = 'Unknown'

            crop_height = px_y_max - px_y_min
            crop_width = px_x_max - px_x_min

            # Write as 2D OME-TIFF
            try:
                tifffile.imwrite(
                    output_path,
                    cropped,
                    photometric='minisblack',
                    compression='zlib',
                    tile=(256, 256) if min(crop_height, crop_width) >= 256 else None,
                    metadata={
                        'axes': 'YX',
                        'PhysicalSizeX': px_size,
                        'PhysicalSizeY': px_size,
                        'PhysicalSizeXUnit': 'um',
                        'PhysicalSizeYUnit': 'um',
                        'Channel': {'Name': channel_name},
                    }
                )
            except Exception:
                tifffile.imwrite(
                    output_path,
                    cropped,
                    photometric='minisblack',
                    metadata={
                        'axes': 'YX',
                        'PhysicalSizeX': px_size,
                        'PhysicalSizeY': px_size,
                        'PhysicalSizeXUnit': 'um',
                        'PhysicalSizeYUnit': 'um',
                        'Channel': {'Name': channel_name},
                    }
                )

            print(f"  Cropped {input_path.name} to {crop_width}x{crop_height} pixels")
            return True

        except Exception as e:
            print(f"  Error cropping {input_path.name}: {e}")
            return False

    def crop_morphology_files(self, punch_id: str):
        """
        Crop morphology.ome.tif and morphology_focus/*.ome.tif files for a punch region.

        Args:
            punch_id: The punch ID to process
        """
        if not HAS_TIFFFILE:
            print(f"  Warning: tifffile not installed, skipping morphology cropping for punch {punch_id}")
            return

        punch_dir = self.output_base_dir / f"punch_{punch_id}"
        coords = self.punch_coordinates[punch_id]
        x_min, x_max = coords['x_min'], coords['x_max']
        y_min, y_max = coords['y_min'], coords['y_max']

        # Crop main morphology.ome.tif (Z-stack)
        morphology_file = self.xenium_dir / "morphology.ome.tif"
        if morphology_file.exists():
            output_file = punch_dir / "morphology.ome.tif"
            self._crop_ome_tiff(morphology_file, output_file, x_min, x_max, y_min, y_max)

        # Crop morphology_focus files (single channel 2D images)
        morphology_focus_dir = self.xenium_dir / "morphology_focus"
        if morphology_focus_dir.exists() and morphology_focus_dir.is_dir():
            output_focus_dir = punch_dir / "morphology_focus"
            output_focus_dir.mkdir(exist_ok=True)

            focus_files = sorted(morphology_focus_dir.glob("*.ome.tif"))
            for focus_file in focus_files:
                output_file = output_focus_dir / focus_file.name
                self._crop_morphology_focus(focus_file, output_file, x_min, x_max, y_min, y_max)

    def crop_all_morphology_files(self, n_processes: int = 4):
        """
        Crop morphology files for all punch regions with parallel processing.

        Optimized approach:
        1. Load each source image once into memory
        2. Crop for all punches in parallel

        Args:
            n_processes: Number of parallel processes to use
        """
        if not HAS_TIFFFILE:
            print("Warning: tifffile not installed, skipping all morphology cropping")
            return

        print("Cropping morphology files...")
        px_size = 0.2125

        # Prepare punch data for parallel processing
        punch_data = []
        for punch_id in self.punch_ids:
            coords = self.punch_coordinates[punch_id]
            punch_dir = self.output_base_dir / f"punch_{punch_id}"

            # Convert coordinates to pixels
            px_x_min = int(coords['x_min'] / px_size)
            px_x_max = int(coords['x_max'] / px_size)
            px_y_min = int(coords['y_min'] / px_size)
            px_y_max = int(coords['y_max'] / px_size)

            punch_data.append({
                'punch_id': punch_id,
                'punch_dir': str(punch_dir),
                'px_x_min': px_x_min,
                'px_x_max': px_x_max,
                'px_y_min': px_y_min,
                'px_y_max': px_y_max,
            })

        # Process main morphology.ome.tif
        morphology_file = self.xenium_dir / "morphology.ome.tif"
        if morphology_file.exists():
            print(f"  Loading morphology.ome.tif into memory...")
            self._crop_morphology_parallel(
                str(morphology_file),
                punch_data,
                'morphology.ome.tif',
                is_zstack=True,
                n_processes=n_processes
            )

        # Process morphology_focus files
        morphology_focus_dir = self.xenium_dir / "morphology_focus"
        if morphology_focus_dir.exists() and morphology_focus_dir.is_dir():
            focus_files = sorted(morphology_focus_dir.glob("*.ome.tif"))

            for focus_file in focus_files:
                print(f"  Loading {focus_file.name} into memory...")
                # Create output subdirectories
                for pd in punch_data:
                    output_focus_dir = Path(pd['punch_dir']) / "morphology_focus"
                    output_focus_dir.mkdir(exist_ok=True)

                self._crop_morphology_parallel(
                    str(focus_file),
                    punch_data,
                    f"morphology_focus/{focus_file.name}",
                    is_zstack=False,
                    n_processes=n_processes
                )

    def _crop_morphology_parallel(self, input_path: str, punch_data: list,
                                   output_filename: str, is_zstack: bool,
                                   n_processes: int = 4):
        """
        Load image once and crop for all punches in parallel.

        Args:
            input_path: Path to source image
            punch_data: List of dicts with punch_id, punch_dir, and pixel coordinates
            output_filename: Output filename (relative to punch_dir)
            is_zstack: Whether this is a Z-stack (morphology.ome.tif) or 2D (morphology_focus)
            n_processes: Number of parallel processes
        """
        px_size = 0.2125

        try:
            # Load the full image into memory once
            with tifffile.TiffFile(input_path) as tif:
                page = tif.pages[0]
                img_height, img_width = page.shape[:2]

                if is_zstack:
                    # Read all Z-slices
                    print(f"    Reading {len(tif.pages)} z-slices...")
                    full_data = np.stack([p.asarray() for p in tif.pages], axis=0)
                else:
                    # Single 2D image
                    full_data = page.asarray()

            print(f"    Image loaded: {full_data.shape}, cropping {len(punch_data)} regions...")

            # Get channel name for morphology_focus files
            channel_name = 'DAPI'
            if not is_zstack:
                channel_names = ['DAPI', 'ATP1A1/CD45/E-Cadherin', '18S', 'alphaSMA/Vimentin']
                try:
                    channel_idx = int(Path(input_path).stem.split('_')[-1])
                    channel_name = channel_names[channel_idx] if channel_idx < len(channel_names) else f'Channel_{channel_idx}'
                except (ValueError, IndexError):
                    pass

            # Prepare tasks for parallel processing
            tasks = []
            for pd in punch_data:
                # Clamp coordinates to image bounds
                px_x_min = max(0, pd['px_x_min'])
                px_x_max = min(img_width, pd['px_x_max'])
                px_y_min = max(0, pd['px_y_min'])
                px_y_max = min(img_height, pd['px_y_max'])

                if px_x_max <= px_x_min or px_y_max <= px_y_min:
                    print(f"    Warning: Invalid crop region for punch {pd['punch_id']}")
                    continue

                output_path = Path(pd['punch_dir']) / output_filename

                tasks.append({
                    'punch_id': pd['punch_id'],
                    'output_path': str(output_path),
                    'px_x_min': px_x_min,
                    'px_x_max': px_x_max,
                    'px_y_min': px_y_min,
                    'px_y_max': px_y_max,
                    'is_zstack': is_zstack,
                    'channel_name': channel_name,
                    'px_size': px_size,
                })

            # Process in parallel using ThreadPoolExecutor (better for I/O bound tasks)
            # We use threads here since the data is already in memory
            from concurrent.futures import ThreadPoolExecutor

            def crop_and_save(task):
                try:
                    # Crop the data
                    if task['is_zstack']:
                        cropped = full_data[:, task['px_y_min']:task['px_y_max'],
                                           task['px_x_min']:task['px_x_max']]
                    else:
                        cropped = full_data[task['px_y_min']:task['px_y_max'],
                                           task['px_x_min']:task['px_x_max']]

                    crop_height = task['px_y_max'] - task['px_y_min']
                    crop_width = task['px_x_max'] - task['px_x_min']

                    # Write the cropped image
                    if task['is_zstack']:
                        metadata = {
                            'axes': 'ZYX',
                            'PhysicalSizeX': task['px_size'],
                            'PhysicalSizeY': task['px_size'],
                            'PhysicalSizeZ': 3.0,
                            'PhysicalSizeXUnit': 'um',
                            'PhysicalSizeYUnit': 'um',
                            'PhysicalSizeZUnit': 'um',
                            'Channel': {'Name': 'DAPI'},
                        }
                    else:
                        metadata = {
                            'axes': 'YX',
                            'PhysicalSizeX': task['px_size'],
                            'PhysicalSizeY': task['px_size'],
                            'PhysicalSizeXUnit': 'um',
                            'PhysicalSizeYUnit': 'um',
                            'Channel': {'Name': task['channel_name']},
                        }

                    try:
                        tifffile.imwrite(
                            task['output_path'],
                            cropped,
                            photometric='minisblack',
                            compression='zlib',
                            tile=(256, 256) if min(crop_height, crop_width) >= 256 else None,
                            metadata=metadata
                        )
                    except Exception:
                        # Fallback without compression
                        tifffile.imwrite(
                            task['output_path'],
                            cropped,
                            photometric='minisblack',
                            metadata=metadata
                        )

                    return {'punch_id': task['punch_id'], 'success': True,
                            'size': f"{crop_width}x{crop_height}"}
                except Exception as e:
                    return {'punch_id': task['punch_id'], 'success': False, 'error': str(e)}

            # Execute in parallel
            with ThreadPoolExecutor(max_workers=n_processes) as executor:
                results = list(executor.map(crop_and_save, tasks))

            # Report results
            success_count = sum(1 for r in results if r['success'])
            print(f"    Cropped {success_count}/{len(tasks)} regions successfully")

            for r in results:
                if not r['success']:
                    print(f"    Failed punch {r['punch_id']}: {r.get('error', 'unknown error')}")

        except Exception as e:
            print(f"  Error processing {input_path}: {e}")

    def copy_auxiliary_files(self, punch_id: str):
        """
        Copy gene_panel.json and analysis_summary.html to a punch directory.

        Args:
            punch_id: The punch ID to process
        """
        punch_dir = self.output_base_dir / f"punch_{punch_id}"
        files_to_copy = ['gene_panel.json', 'analysis_summary.html']

        for filename in files_to_copy:
            source_file = self.xenium_dir / filename
            dest_file = punch_dir / filename

            if source_file.exists():
                try:
                    shutil.copy2(source_file, dest_file)
                except Exception as e:
                    print(f"  Error copying {filename} for punch {punch_id}: {e}")

    def copy_all_auxiliary_files(self):
        """Copy auxiliary files to all punch directories."""
        print("Copying auxiliary files (gene_panel.json, analysis_summary.html)...")
        for punch_id in self.punch_ids:
            self.copy_auxiliary_files(punch_id)
            print(f"  Copied files for punch {punch_id}")
    
    def split_all_data(self, experiment_modifications: Dict = None, n_processes=4,
                       run_light: bool = False, region_metrics_df: pd.DataFrame = None):
        """
        Main method to split all data files by punch_id using coordinate-based filtering.

        Args:
            experiment_modifications: Dictionary of modifications for experiment.xenium files (ignored)
            n_processes: Number of CPU used for parallel process H5 matrix splitting
            run_light: If True, skip metrics_summary, morphology cropping, and auxiliary file copying
            region_metrics_df: DataFrame containing metrics for all regions (used for metrics_summary.csv)
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

        # Load cells data (with blacklist filtering applied)
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

        # Additional processing (skip if run_light mode)
        if not run_light:
            # Write metrics_summary.csv to each punch directory
            self.write_all_metrics_summaries(region_metrics_df)

            # Crop morphology files (use all available processes for speed)
            self.crop_all_morphology_files(n_processes=n_processes)

            # Copy auxiliary files
            self.copy_all_auxiliary_files()

        print(f"\nData splitting completed! Output saved to: {self.output_base_dir}")
        print(f"Created {len(self.punch_ids)} subdirectories for punch_ids: {list(self.punch_ids)}")
        print(f"\nCell distribution:")
        for _, row in summary_df.iterrows():
            print(f"  Punch {row['punch_id']}: {row['num_cells']} cells")

        if run_light:
            print("\nNote: Running in light mode - metrics_summary, morphology cropping, and auxiliary files skipped")


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