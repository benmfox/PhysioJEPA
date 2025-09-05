import numpy as np
import pandas as pd
from pathlib import Path
import zarr
import glob
import os
import tqdm
import multiprocessing as mp
import time
from physiojepa.signal import butterworth, preprocess_abp_signal, resample_waveform


# Parameters for filtering and processing ABP signals
ABP_CHANNEL = 'ABP' 

CALIBRATION = 1.0  # Calibration factor for ABP signal
BACKUP_FREQUENCY = 125 # 125 for mimic
SAMPLE_FREQUENCY_KEY = 'sampling_frequency' # 'sampling_frequency' for mimic

MBP_THRESHOLD = 65  # Mean Blood Pressure threshold in mmHg
SBP_THRESHOLD = 90  # Systolic Blood Pressure threshold in mmHg
SHOCK_INDEX_CUTOFF = 0.9

zarr_files = glob.glob('*.zarr')
output_csv = 'hypotension_si_labels_mimic.csv.gz'


# Function to process ABP data and return DataFrame
def process_abp_signal(hea_file_path, abp_signal, timestamp, fs_in, fs_out=125, calibration=CALIBRATION):
    
    abp_signal = abp_signal*calibration
    if fs_in != fs_out:
        abp_signal = resample_waveform(abp_signal, fs_in=fs_in, fs_out=fs_out, is_spo2=False)
    median_sys, median_dias, median_maps, median_hr = preprocess_abp_signal(abp_signal, fs_out)
    if ~np.isnan(median_sys) and ~np.isnan(median_maps):
        hypotension = int((median_maps <= MBP_THRESHOLD) | (median_sys <= SBP_THRESHOLD))
    else:
        hypotension = np.nan
    if ~np.isnan(median_sys) and ~np.isnan(median_hr):
        shock_index = median_hr / median_sys
        shock_index_label = int(shock_index >= SHOCK_INDEX_CUTOFF)
    else:
        shock_index = np.nan
        shock_index_label = np.nan
    
    bp_df = pd.DataFrame([{
        'Timestamp (seconds)': timestamp,
        'SBP (mmHg)': median_sys,
        'DBP (mmHg)': median_dias,
        'MBP (mmHg)': median_maps,
        'hr_bpm': median_hr,
        'hypotension':hypotension,
        'shock_index':shock_index,
        'shock_index_label':shock_index_label
    }])

    # Extract date-time from file name (e.g., p000177-2125-11-29-12-35)
    file_name = Path(hea_file_path).stem
    bp_df['file_name'] = file_name
    bp_df['file_path'] = hea_file_path
    file_name_split = file_name.split('-')
    subject_id = file_name_split[0]
    date_time_part = file_name_split[1:]  # Extract date-time components after the subject_id part
    date_time_str = f"{date_time_part[1]}/{date_time_part[2]}/{date_time_part[0]} {date_time_part[3]}:{date_time_part[4]}"

    # Add 'subject_id' and 'date' columns
    bp_df['subject_id'] = subject_id
    bp_df['date'] = date_time_str

    return bp_df


def process_zarr_file(zarr_file, step_seconds=60):
    # Convert HEA signals to Zarr format
    root_grp = zarr.open(zarr_file, 'r')
    all_bp_dfs = []
    if root_grp is not None and ABP_CHANNEL in root_grp:
        abp_signal = root_grp[ABP_CHANNEL][:]
        abp_freq = root_grp[ABP_CHANNEL].attrs.get(SAMPLE_FREQUENCY_KEY, BACKUP_FREQUENCY)
        for idx, start_idx in enumerate(range(0, len(abp_signal), int(abp_freq*step_seconds))):
            end_idx = start_idx + int(abp_freq*step_seconds)
            timestamp_seconds = start_idx / abp_freq
            try:
                minute_avg_bp_df = process_abp_signal(hea_file_path=zarr_file, abp_signal=abp_signal[start_idx:end_idx], timestamp=timestamp_seconds, fs_in=abp_freq, fs_out=125)
                all_bp_dfs.append(minute_avg_bp_df)
            except Exception as e:
                print(f"Error processing segment {idx} of file {zarr_file}: {e}")
                all_bp_dfs.append(pd.DataFrame())
                raise e
        return pd.concat(all_bp_dfs, axis=0)
    else:
        print(f"No ABP signal for file {zarr_file}")
        return None
    
if __name__ == '__main__':
    print(f'Beginning MP Job with {mp.cpu_count()} processes')
    start_time = time.time()
    
    with mp.Pool(mp.cpu_count()) as pool:
        # Use imap_unordered instead of map_async for better error handling
        results = list(tqdm.tqdm(pool.imap_unordered(process_zarr_file, zarr_files),
                                 total=len(zarr_files), desc="Processing files"))
    
    # Write the header
    df = pd.DataFrame(columns=['Timestamp (seconds)', 'SBP (mmHg)', 'DBP (mmHg)', 'MBP (mmHg)', 'hr_bpm','hypotension','shock_index','shock_index_label',\
        'file_name', 'file_path', 'subject_id', 'date'])
    df.to_csv(output_csv, index=False, compression='gzip')
    
    # Write successful results
    for df in tqdm.tqdm(results, desc='Writing data'):
        if df is not None:
            try:
                df.to_csv(output_csv, index=False, mode='a', header=False, compression='gzip')
            except Exception as e:
                print(f"Error writing results to CSV: {e}")
                continue
    
    print('Job Completed')
    print(f"--- {time.time() - start_time} seconds ---")