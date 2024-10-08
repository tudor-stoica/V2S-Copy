import h5py

def print_h5_structure(file_path):
    def print_attrs(name, obj):
        print(f"{name}: {obj}")
        for key, val in obj.attrs.items():
            print(f"    Attribute - {key}: {val}")
    
    with h5py.File(file_path, 'r') as f:
        print("HDF5 file structure:")
        f.visititems(print_attrs)

# Path to your .h5 file
file_path = 'weight/beta/No0_map1-01-0.7765.h5'
print_h5_structure(file_path)