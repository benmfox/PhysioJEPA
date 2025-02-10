import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
import wfdb
import zarr

# Parameters for filtering and processing ABP signals
MAX_SIGNAL_MAX = 250
MIN_SIGNAL_MAX = 30
fs = 125  # Sampling frequency in Hz
MBP_THRESHOLD = 65  # Mean Blood Pressure threshold in mmHg
SBP_THRESHOLD = 90  # Systolic Blood Pressure threshold in mmHg

# Function to check if ABP signal meets the criteria
def is_valid_abp_signal(abp_signal):
    if np.nanmax(abp_signal) > MAX_SIGNAL_MAX or np.nanmax(abp_signal) < MIN_SIGNAL_MAX:
        return False
    if len(abp_signal) / fs < 60:
        return False
    return True

# Function to preprocess and filter ABP signals
def preprocess_abp_signal(abp_signal):
    good_segment_mask = np.ones_like(abp_signal, dtype=bool)
    good_segment_mask[abp_signal < 0] = False

    flat_line_mask = np.abs(np.diff(abp_signal, prepend=abp_signal[0])) < 1
    flat_line_samples = int(2 * fs)
    flat_segments = np.convolve(flat_line_mask, np.ones(flat_line_samples), 'valid') >= flat_line_samples
    extended_flat_segments = np.concatenate([flat_segments, np.zeros(len(abp_signal) - len(flat_segments), dtype=bool)])
    good_segment_mask[extended_flat_segments] = False

    abp_signal_filtered = np.copy(abp_signal)
    abp_signal_filtered[~good_segment_mask] = np.nan

    systolic_peaks, _ = find_peaks(abp_signal_filtered, distance=50)
    diastolic_troughs, _ = find_peaks(-abp_signal_filtered, distance=50)

    return abp_signal_filtered, systolic_peaks, diastolic_troughs

# Function to process ABP data and return DataFrame
def process_abp_signal(hea_file_path, abp_signal_filtered, systolic_peaks, diastolic_troughs):
    systolic_timestamps = systolic_peaks / fs
    diastolic_timestamps = diastolic_troughs / fs
    sbp_values = abp_signal_filtered[systolic_peaks]
    dbp_values = abp_signal_filtered[diastolic_troughs]

    min_length = min(len(sbp_values), len(dbp_values))
    if min_length == 0:
        print(f"No valid SBP or DBP values found for file: {hea_file_path}")
        return pd.DataFrame()

    sbp_values = sbp_values[:min_length]
    dbp_values = dbp_values[:min_length]
    systolic_timestamps = systolic_timestamps[:min_length]

    mbp_values = (1/3) * sbp_values + (2/3) * dbp_values

    bp_df = pd.DataFrame({
        'Timestamp (s)': systolic_timestamps,
        'SBP (mmHg)': sbp_values,
        'DBP (mmHg)': dbp_values,
        'MBP (mmHg)': mbp_values
    })

    bp_df['Minute'] = np.floor(bp_df['Timestamp (s)'] / 60).astype(int)

    if bp_df['Minute'].max() < 1:
        print(f"Skipping file {hea_file_path} - Less than 1 minute of data.")
        return pd.DataFrame()

    minute_avg_bp_df = bp_df.groupby('Minute').agg({
        'SBP (mmHg)': 'mean',
        'DBP (mmHg)': 'mean',
        'MBP (mmHg)': 'mean'
    }).reset_index()

    total_minutes = np.arange(0, bp_df['Minute'].max() + 1)
    all_minutes_df = pd.DataFrame({'Minute': total_minutes})
    minute_avg_bp_df = all_minutes_df.merge(minute_avg_bp_df, on='Minute', how='left').fillna(0)

    filtered_out_seconds = np.isnan(abp_signal_filtered).astype(int)
    filtered_out_minutes = np.floor(np.arange(len(filtered_out_seconds)) / fs / 60).astype(int)
    filtered_out_df = pd.DataFrame({'Minute': filtered_out_minutes, 'Filtered_Out': filtered_out_seconds})
    filtered_duration_per_minute = filtered_out_df.groupby('Minute')['Filtered_Out'].sum() / fs

    minute_avg_bp_df['hypotension'] = 0
    minute_avg_bp_df.loc[minute_avg_bp_df['Minute'].isin(filtered_duration_per_minute.index[filtered_duration_per_minute > 20]), 'hypotension'] = 2

    low_bp_condition = (minute_avg_bp_df['MBP (mmHg)'] <= MBP_THRESHOLD) | (minute_avg_bp_df['SBP (mmHg)'] <= SBP_THRESHOLD)
    minute_avg_bp_df['LowBP_Consecutive'] = (low_bp_condition & low_bp_condition.shift(fill_value=False)).astype(int)

    minute_avg_bp_df.loc[(minute_avg_bp_df['hypotension'] == 0) & (minute_avg_bp_df['LowBP_Consecutive'] == 1), 'hypotension'] = 1
    minute_avg_bp_df = minute_avg_bp_df.drop(columns=['LowBP_Consecutive'])

    # Convert 'Minute' to 'Time Stamp (seconds)'
    minute_avg_bp_df['Time Stamp (seconds)'] = minute_avg_bp_df['Minute'] * 60

    # Drop the 'Minute' column
    minute_avg_bp_df = minute_avg_bp_df.drop(columns=['Minute'])

    # Extract subject_id from folder name (e.g., p000177)
    subject_id = Path(hea_file_path).parent.name

    # Extract date-time from file name (e.g., p000177-2125-11-29-12-35)
    file_name = Path(hea_file_path).stem
    date_time_part = file_name.split('-')[1:]  # Extract date-time components after the subject_id part
    date_time_str = f"{date_time_part[1]}/{date_time_part[2]}/{date_time_part[0]} {date_time_part[3]}:{date_time_part[4]}"

    # Add 'subject_id' and 'date' columns
    minute_avg_bp_df['subject_id'] = subject_id
    minute_avg_bp_df['date'] = date_time_str

    return minute_avg_bp_df

# Function to convert .hea files to Zarr format
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
    root_grp = zarr.group(store, overwrite=True)

    for i, name in enumerate(signal_names):
        signal_data = record.p_signal[:, i]
        root_grp[name] = zarr.array(signal_data)
        root_grp[name].attrs['units'] = signal_units[i]
        root_grp[name].attrs['sampling_frequency'] = sampling_frequency

    zarr.consolidate_metadata(store)

    return root_grp

def process_all_folders(base_directory, write_data_dir, output_csv, overwrite=False):
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to store the combined data
    folder_prefixes = [f"p0{i}" for i in range(0, 3)]  # Folder names from p00 to p09

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
                        root_grp = hea_signals_to_zarr(hea_file_path, write_data_dir, overwrite)

                        if root_grp is not None and 'ABP' in root_grp:
                            abp_signal = root_grp['ABP'][:]
                            if is_valid_abp_signal(abp_signal):
                                abp_signal_filtered, systolic_peaks, diastolic_troughs = preprocess_abp_signal(abp_signal)
                                minute_avg_bp_df = process_abp_signal(hea_file_path, abp_signal_filtered, systolic_peaks, diastolic_troughs)
                                combined_df = pd.concat([combined_df, minute_avg_bp_df], ignore_index=True)
                            else:
                                print(f"Invalid ABP signal for file {hea_file_path}")
                        else:
                            print(f"File {hea_file_path} did not have a valid ABP signal.")
                    except FileNotFoundError as e:
                        print(f"Error processing {hea_file_path}: {e}")
                    except Exception as e:
                        print(f"Unexpected error processing {hea_file_path}: {e}")

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined DataFrame saved to {output_csv}")

# Example usage:
base_directory = ''
zarr_directory = ''
output_csv = ''

process_all_folders(base_directory, zarr_directory, output_csv)


# Path to the directory containing Zarr files
directory = ''

# Iterate through each .zarr file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".zarr"):
        filepath = os.path.join(directory, filename)
        
        # Open the .zarr file
        store = zarr.DirectoryStore(filepath)
        root_grp = zarr.group(store)
        
        try:
            # Try to access the 'ABP' key
            waveform_data = root_grp['ABP']
            
            # Calculate the duration of the waveform data in seconds
            duration_seconds = waveform_data.shape[0] / 125  
            
            # Add the 'Duration' attribute to the root group
            root_grp.attrs['Duration'] = duration_seconds
            
            print(f"Added 'Duration' attribute to {filename}: {duration_seconds} seconds")
        
        except KeyError:
            # If 'ABP' key is not found, print the filename and continue to the next file
            print(f"No 'ABP' key found in {filename}. Skipping...")
