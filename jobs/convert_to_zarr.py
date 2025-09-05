import os
from pathlib import Path
import wfdb
import zarr

# Example usage:
base_directory = ''
zarr_directory = ''

# Function to convert MIMIC .hea files to Zarr format
def hea_signals_to_zarr(hea_file_path, write_data_dir, overwrite=False):
    hea_file_path = Path(hea_file_path)
    write_data_dir = Path(write_data_dir)

    record = wfdb.rdrecord(str(hea_file_path)[:-4])

    if 'ABP' not in record.sig_name:
        print(f"Skipping {hea_file_path} as it does not contain 'ABP' signal")
        return None

    sampling_frequency = record.fs
    signal_names = record.sig_name
    signal_units = record.units

    store = zarr.DirectoryStore(str(write_data_dir / hea_file_path.stem) + '.zarr')
    root_grp = zarr.group(store, overwrite=overwrite)

    for i, name in enumerate(signal_names):
        signal_data = record.p_signal[:, i]
        root_grp[name] = zarr.array(signal_data)
        root_grp[name].attrs['units'] = signal_units[i]
        root_grp[name].attrs['sampling_frequency'] = sampling_frequency
        duration_seconds = signal_data.shape[0] / sampling_frequency
        root_grp.attrs['Duration'] = duration_seconds

    zarr.consolidate_metadata(store)

    return root_grp

def process_all_folders(base_directory, write_data_dir, overwrite=False):
    folder_prefixes = [f"p0{i}" for i in range(0, 10)]  # Folder names from p00 to p09

    for folder_prefix in folder_prefixes:
        folder_path = os.path.join(base_directory, folder_prefix)

        # Walk through each folder (p00, p01, ..., p09)
        for folder_name, _, filenames in os.walk(folder_path):
            for hea_file in filenames:
                if hea_file.endswith('.hea') and hea_file[:-4] + 'n.hea' in filenames:  # Check for .hea files
                    hea_file_path = os.path.join(folder_name, hea_file)
                    if not os.path.exists(hea_file_path):
                        print(f"File {hea_file_path} not found. Skipping.")
                        continue
                    
                    print(f"Processing HEA file: {hea_file_path}")

                    try:
                        # Convert HEA signals to Zarr format
                        _ = hea_signals_to_zarr(hea_file_path, write_data_dir, overwrite)
                    except FileNotFoundError as e:
                        print(f"Error processing {hea_file_path}: {e}")
                    except Exception as e:
                        print(f"Unexpected error processing {hea_file_path}: {e}")


if __name__ == "__main__":
    process_all_folders(base_directory, zarr_directory, overwrite=False)