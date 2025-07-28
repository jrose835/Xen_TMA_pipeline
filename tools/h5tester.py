import h5py
import numpy as np
import pandas as pd

def inspect_h5_file(filename):
    """
    Comprehensive inspection of an HDF5 file
    """
    print(f"Inspecting HDF5 file: {filename}")
    print("=" * 50)
    
    try:
        with h5py.File(filename, 'r') as f:
            print(f"File keys (top-level groups/datasets): {list(f.keys())}")
            print()
            
            # Recursively explore the file structure
            def explore_group(name, obj):
                print(f"{'  ' * name.count('/')}{name}")
                if isinstance(obj, h5py.Dataset):
                    print(f"{'  ' * (name.count('/') + 1)}Type: Dataset")
                    print(f"{'  ' * (name.count('/') + 1)}Shape: {obj.shape}")
                    print(f"{'  ' * (name.count('/') + 1)}Dtype: {obj.dtype}")
                    print(f"{'  ' * (name.count('/') + 1)}Size: {obj.size} elements")
                    
                    # Show attributes if any
                    if obj.attrs:
                        print(f"{'  ' * (name.count('/') + 1)}Attributes:")
                        for attr_name, attr_value in obj.attrs.items():
                            print(f"{'  ' * (name.count('/') + 2)}{attr_name}: {attr_value}")
                    
                    # Show a sample of the data if it's small enough
                    if obj.size < 100:
                        print(f"{'  ' * (name.count('/') + 1)}Data preview:")
                        print(f"{'  ' * (name.count('/') + 2)}{obj[...]}")
                    elif len(obj.shape) <= 2:
                        print(f"{'  ' * (name.count('/') + 1)}Data preview (first few elements):")
                        if len(obj.shape) == 1:
                            print(f"{'  ' * (name.count('/') + 2)}{obj[:min(10, obj.shape[0])]}")
                        else:
                            print(f"{'  ' * (name.count('/') + 2)}{obj[:min(5, obj.shape[0]), :min(5, obj.shape[1])]}")
                    
                elif isinstance(obj, h5py.Group):
                    print(f"{'  ' * (name.count('/') + 1)}Type: Group")
                    if obj.attrs:
                        print(f"{'  ' * (name.count('/') + 1)}Attributes:")
                        for attr_name, attr_value in obj.attrs.items():
                            print(f"{'  ' * (name.count('/') + 2)}{attr_name}: {attr_value}")
                
                print()
            
            print("File Structure:")
            print("-" * 20)
            f.visititems(explore_group)
            
    except Exception as e:
        print(f"Error reading file: {e}")

def load_dataset(filename, dataset_path):
    """
    Load a specific dataset from the HDF5 file
    """
    try:
        with h5py.File(filename, 'r') as f:
            dataset = f[dataset_path]
            data = dataset[...]
            print(f"Loaded dataset '{dataset_path}'")
            print(f"Shape: {data.shape}")
            print(f"Type: {type(data)}")
            return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def convert_to_pandas(filename, dataset_path, row_names=None, col_names=None):
    """
    Convert a 2D dataset to a pandas DataFrame
    """
    data = load_dataset(filename, dataset_path)
    if data is not None and len(data.shape) == 2:
        df = pd.DataFrame(data)
        
        if row_names is not None:
            df.index = row_names
        if col_names is not None:
            df.columns = col_names
            
        return df
    else:
        print("Data is not 2D or couldn't be loaded")
        return None

# Main inspection
if __name__ == "__main__":
    filename = "/data-raid/Xenium/prime/20250709__200006__XP1_TMA_Mouse_11_Organ_070925_/XenSplitter/test/punch_1a/cell_feature_matrix.h5"
    
    # Basic inspection
    inspect_h5_file(filename)
    
    # Example: Load a specific dataset (you'll need to adjust the path based on inspection results)
    # data = load_dataset(filename, "matrix/data")  # Adjust path as needed
    
    # Example: Convert to pandas DataFrame
    # df = convert_to_pandas(filename, "matrix/data")
    # print(df.head())