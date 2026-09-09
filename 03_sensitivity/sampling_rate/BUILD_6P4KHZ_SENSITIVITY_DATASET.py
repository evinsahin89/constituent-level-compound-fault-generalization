"""Anti-aliased 6.4-kHz dataset preparation for sampling-rate sensitivity analysis.

A separate 6.4-kHz copy of the native 12.8-kHz MCC5-THU recordings is created
with polyphase anti-aliased resampling. The original recordings are not
modified. Input layout, channel count, output length, duration, and basic signal
statistics are validated during processing.

The generated recordings are used by the independently rerun 6.4-kHz feature
extraction and crossed evaluation.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
ROOT = Path('D:\\yay\u0131nlar\\Multi-mode Fault Diagnosis Datasets of Three-phase')
SOURCE_DIRS = [ROOT / 'MCC5-THU Motor_speed_circulation', ROOT / 'MCC5-THU Motor_torque_circulation']
SENSITIVITY_ROOT = ROOT / 'SAMPLING_RATE_SENSITIVITY'
OUT_ROOT = SENSITIVITY_ROOT / 'RAW_6P4KHZ'
QC_DIR = SENSITIVITY_ROOT / 'QC'
OUT_ROOT.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)
FS_NATIVE = 12800
FS_TARGET = 6400
UP = 1
DOWN = 2
EXPECTED_SIGNAL_CHANNELS = 8
ALLOWED_TOTAL_COLUMNS = {8, 9}
OVERWRITE = False

def is_numeric_series(series):
    """
    Returns True if every value can be interpreted as numeric.
    """
    x = pd.to_numeric(series, errors='coerce')
    return bool(x.notna().all())

def row_is_numeric(row):
    """
    Returns True if every element in a row is numeric.
    """
    x = pd.to_numeric(row, errors='coerce')
    return bool(x.notna().all())

def detect_header(path):
    """
    Determine whether a CSV has a header.

    Strategy
    --------
    Read the first few rows with header=None.

    If row 0 is fully numeric:
        assume no header.

    If row 0 contains text but subsequent rows are numeric:
        assume header.

    This avoids the common pandas issue where a headerless numeric first row
    is accidentally interpreted as column names.
    """
    preview = pd.read_csv(path, header=None, nrows=4)
    if preview.empty:
        raise RuntimeError(f'Empty CSV file:\n{path}')
    first_numeric = row_is_numeric(preview.iloc[0])
    if first_numeric:
        return False
    if len(preview) >= 2:
        second_numeric = row_is_numeric(preview.iloc[1])
        if second_numeric:
            return True
    raise RuntimeError(f'\nCould not safely determine whether the CSV contains a header.\nFile:\n{path}\n\nFirst rows:\n{preview.to_string(index=False, header=False)}')

def read_raw_csv(path):
    """
    Read MCC5-THU raw CSV.

    Returns
    -------
    df : DataFrame
    has_header : bool
    """
    has_header = detect_header(path)
    if has_header:
        df = pd.read_csv(path, header=0)
    else:
        df = pd.read_csv(path, header=None)
        df.columns = [f'col_{i + 1}' for i in range(df.shape[1])]
    n_columns = df.shape[1]
    if n_columns not in ALLOWED_TOTAL_COLUMNS:
        raise RuntimeError(f'\nUnexpected number of columns.\nFile:\n{path}\nExpected 8 signal columns or 1 time/index + 8 signal columns.\nFound: {n_columns} columns\nColumns: {list(df.columns)}')
    return (df, has_header)

def name_suggests_time_or_index(column_name):
    """
    Check common time/index column-name patterns.
    """
    name = str(column_name).strip().lower()
    exact_hints = {'time', 'timestamp', 'sample', 'samples', 'sample_index', 'sampleindex', 'index', 'idx', 't', 'time_s', 'time_sec', 'time_seconds', 'time(s)', 'seconds', 'sec'}
    if name in exact_hints:
        return True
    if 'time' in name:
        return True
    if 'sample' in name:
        return True
    return False

def monotonic_uniformity_score(series):
    """
    Test whether a numeric series behaves like a time/sample-index vector.

    Returns
    -------
    score : float or None

    Lower score = more uniformly increasing.

    None means it is not a credible time/index candidate.
    """
    x = pd.to_numeric(series, errors='coerce')
    if x.isna().any():
        return None
    arr = x.to_numpy(dtype=np.float64)
    if len(arr) < 3:
        return None
    diff = np.diff(arr)
    if not np.all(diff > 0):
        return None
    mean_step = float(np.mean(diff))
    if mean_step <= 0:
        return None
    std_step = float(np.std(diff))
    score = std_step / abs(mean_step)
    if score > 0.001:
        return None
    return score

def identify_time_column(df, path):
    """
    Identify the optional ninth time/sample-index column.

    For 8-column files:
        returns None.

    For 9-column files:
        first tries column-name hints,
        then tests numerical monotonicity/uniform spacing.

    The script stops if the ninth column cannot be identified safely.
    """
    if df.shape[1] == EXPECTED_SIGNAL_CHANNELS:
        return None
    if df.shape[1] != EXPECTED_SIGNAL_CHANNELS + 1:
        raise RuntimeError(f'Unsupported column layout in:\n{path}')
    named_candidates = [col for col in df.columns if name_suggests_time_or_index(col)]
    if len(named_candidates) == 1:
        candidate = named_candidates[0]
        score = monotonic_uniformity_score(df[candidate])
        if score is not None:
            return candidate
    candidates = []
    for col in df.columns:
        score = monotonic_uniformity_score(df[col])
        if score is not None:
            candidates.append((col, score))
    if len(candidates) == 1:
        return candidates[0][0]
    if len(candidates) > 1:
        named = [x for x in candidates if name_suggests_time_or_index(x[0])]
        if len(named) == 1:
            return named[0][0]
        candidates = sorted(candidates, key=lambda x: x[1])
        best_col, best_score = candidates[0]
        second_score = candidates[1][1]
        if best_score == 0 and second_score > 0:
            return best_col
        if best_score > 0 and second_score > best_score * 100:
            return best_col
    preview = df.head(5)
    raise RuntimeError(f'\nThe file contains 9 columns, but the time/sample-index column could not be identified safely.\n\nFile:\n{path}\n\nColumns:\n{list(df.columns)}\n\nFirst five rows:\n{preview.to_string(index=False)}')

def split_time_and_signals(df, path):
    time_col = identify_time_column(df, path)
    if time_col is None:
        signal_df = df.copy()
        if signal_df.shape[1] != EXPECTED_SIGNAL_CHANNELS:
            raise RuntimeError(f'Expected {EXPECTED_SIGNAL_CHANNELS} signal channels but found {signal_df.shape[1]} in:\n{path}')
        return (None, None, signal_df)
    time_values = pd.to_numeric(df[time_col], errors='coerce')
    if time_values.isna().any():
        raise RuntimeError(f'\nInvalid values in detected time/index column.\nFile:\n{path}\nColumn: {time_col}')
    signal_df = df.drop(columns=[time_col])
    if signal_df.shape[1] != EXPECTED_SIGNAL_CHANNELS:
        raise RuntimeError(f'\nUnexpected signal-channel count after removing the time/index column.\nFile:\n{path}\nTime column: {time_col}\nRemaining channels: {signal_df.shape[1]}')
    return (time_col, time_values.to_numpy(dtype=np.float64), signal_df)

def validate_numeric_signals(signal_df, path):
    """
    Convert all signal channels to float64 and reject invalid cells.
    """
    converted = pd.DataFrame(index=signal_df.index)
    for col in signal_df.columns:
        converted[col] = pd.to_numeric(signal_df[col], errors='coerce')
    bad_count = int(converted.isna().sum().sum())
    if bad_count > 0:
        bad_by_column = converted.isna().sum()
        bad_by_column = bad_by_column[bad_by_column > 0]
        raise RuntimeError(f'\nNon-numeric or missing signal samples detected.\nFile:\n{path}\nInvalid cells: {bad_count}\nBy column:\n{bad_by_column.to_string()}')
    return converted.astype(np.float64)

def downsample_signals(signal_matrix):
    """
    Anti-aliased polyphase resampling.

    Parameters
    ----------
    signal_matrix:
        samples x 8 signal channels

    Returns
    -------
    y:
        downsampled samples x 8 channels
    """
    y = resample_poly(signal_matrix, up=UP, down=DOWN, axis=0, padtype='line')
    return y

def create_target_time_vector(time_values, n_output):
    """
    Create the corresponding 6.4-kHz time/index coordinate.

    For exact factor-of-two decimation, every second original coordinate is
    preferred.

    If source length creates an unusual mismatch, interpolation is used only
    for the coordinate column.
    """
    if time_values is None:
        return None
    direct = time_values[::DOWN]
    if len(direct) == n_output:
        return direct
    old_positions = np.arange(len(time_values), dtype=np.float64)
    new_positions = np.linspace(0, len(time_values) - 1, n_output)
    return np.interp(new_positions, old_positions, time_values)

def channel_rms(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.mean(x ** 2, axis=0))

def channel_std(x):
    return np.std(np.asarray(x, dtype=np.float64), axis=0)

def channel_mean(x):
    return np.mean(np.asarray(x, dtype=np.float64), axis=0)

def reconstruct_output_dataframe(original_df, signal_df, downsampled_signals, time_col, time_out):
    """
    Restore the original column order.
    """
    signal_out = pd.DataFrame(downsampled_signals, columns=signal_df.columns)
    if time_col is None:
        return signal_out
    out_df = pd.DataFrame(index=np.arange(len(signal_out)))
    for col in original_df.columns:
        if col == time_col:
            out_df[col] = time_out
        else:
            out_df[col] = signal_out[col].to_numpy()
    return out_df

def process_one_file(source_file, source_root, destination_root):
    relative = source_file.relative_to(source_root)
    destination = destination_root / source_root.name / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not OVERWRITE):
        return {'source_file': str(source_file), 'output_file': str(destination), 'status': 'SKIPPED_EXISTS'}
    df, has_header = read_raw_csv(source_file)
    original_total_columns = df.shape[1]
    time_col, time_values, signal_df = split_time_and_signals(df, source_file)
    numeric_signals = validate_numeric_signals(signal_df, source_file)
    x = numeric_signals.to_numpy(dtype=np.float64)
    if x.shape[1] != EXPECTED_SIGNAL_CHANNELS:
        raise RuntimeError(f'Expected {EXPECTED_SIGNAL_CHANNELS} signal channels, found {x.shape[1]} in:\n{source_file}')
    n_in = len(x)
    duration_native_s = n_in / FS_NATIVE
    rms_native = channel_rms(x)
    std_native = channel_std(x)
    mean_native = channel_mean(x)
    y = downsample_signals(x)
    n_out = len(y)
    duration_target_s = n_out / FS_TARGET
    expected_n_out = int(np.ceil(n_in * FS_TARGET / FS_NATIVE))
    if abs(n_out - expected_n_out) > 1:
        raise RuntimeError(f'\nUnexpected output length after resampling.\nFile:\n{source_file}\nInput samples   : {n_in}\nOutput samples  : {n_out}\nExpected approx.: {expected_n_out}')
    duration_difference_s = duration_target_s - duration_native_s
    maximum_allowed_duration_error = 2.0 / FS_TARGET
    if abs(duration_difference_s) > maximum_allowed_duration_error:
        raise RuntimeError(f'\nDuration mismatch after downsampling.\nFile:\n{source_file}\nNative duration: {duration_native_s:.9f} s\nTarget duration: {duration_target_s:.9f} s\nDifference     : {duration_difference_s:.9f} s')
    rms_target = channel_rms(y)
    std_target = channel_std(y)
    mean_target = channel_mean(y)
    time_out = create_target_time_vector(time_values, n_out)
    out_df = reconstruct_output_dataframe(original_df=df, signal_df=signal_df, downsampled_signals=y, time_col=time_col, time_out=time_out)
    out_df.to_csv(destination, index=False, header=has_header)
    row = {'source_file': str(source_file), 'output_file': str(destination), 'status': 'OK', 'has_header': has_header, 'original_total_columns': original_total_columns, 'time_or_index_column': str(time_col) if time_col is not None else '', 'n_signal_channels': x.shape[1], 'n_samples_native': n_in, 'n_samples_6p4khz': n_out, 'duration_native_s': duration_native_s, 'duration_6p4khz_s': duration_target_s, 'duration_difference_s': duration_difference_s}
    for j in range(EXPECTED_SIGNAL_CHANNELS):
        channel_number = j + 1
        row[f'channel_{channel_number}_mean_native'] = mean_native[j]
        row[f'channel_{channel_number}_mean_6p4k'] = mean_target[j]
        row[f'channel_{channel_number}_std_native'] = std_native[j]
        row[f'channel_{channel_number}_std_6p4k'] = std_target[j]
        row[f'channel_{channel_number}_rms_native'] = rms_native[j]
        row[f'channel_{channel_number}_rms_6p4k'] = rms_target[j]
        row[f'channel_{channel_number}_rms_ratio'] = rms_target[j] / rms_native[j] if rms_native[j] != 0 else np.nan
    return row

def audit_csv_layouts(files):
    """
    Inspect every source CSV before full resampling.

    This catches unexpected file structures early.
    """
    rows = []
    print('\nAuditing CSV layouts...')
    for i, path in enumerate(files, start=1):
        df, has_header = read_raw_csv(path)
        time_col = identify_time_column(df, path)
        rows.append({'file': str(path), 'has_header': has_header, 'n_columns': df.shape[1], 'time_or_index_column': str(time_col) if time_col is not None else '', 'n_signal_columns': df.shape[1] - (1 if time_col is not None else 0)})
        if i % 50 == 0 or i == len(files):
            print(f'  audited {i:3d} / {len(files)}')
    audit = pd.DataFrame(rows)
    audit_path = QC_DIR / 'CSV_LAYOUT_AUDIT.csv'
    audit.to_csv(audit_path, index=False, encoding='utf-8-sig')
    bad = audit[audit['n_signal_columns'] != EXPECTED_SIGNAL_CHANNELS]
    if not bad.empty:
        raise RuntimeError('\nCSV layout audit failed.\nAt least one file does not contain exactly 8 identifiable signal channels.\n\n' + bad.to_string(index=False))
    print('CSV layout audit: PASS')
    print(f'Audit saved to:\n{audit_path}')
    return audit

def main():
    print('\n' + '=' * 120)
    print('MCC5-THU SAMPLING-RATE SENSITIVITY PREPARATION')
    print('12.8 kHz -> 6.4 kHz')
    print('=' * 120)
    print(f'Native sampling rate : {FS_NATIVE} Hz')
    print(f'Target sampling rate : {FS_TARGET} Hz')
    print(f'Resampling ratio     : {UP}/{DOWN}')
    print(f'Output root          : {OUT_ROOT}')
    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            raise FileNotFoundError(f'\nSource directory not found:\n{source_dir}')
    files = []
    for source_dir in SOURCE_DIRS:
        found = sorted(source_dir.rglob('*.csv'))
        files.extend(found)
    if not files:
        raise RuntimeError('No CSV files were found in the raw source directories.')
    print(f'\nRaw CSV files found: {len(files)}')
    if len(files) != 288:
        print(f'\nWARNING: Expected 288 physical recordings but found {len(files)} CSV files.')
    layout_audit = audit_csv_layouts(files)
    print('\nLayout summary:')
    print(layout_audit[['has_header', 'n_columns', 'time_or_index_column', 'n_signal_columns']].value_counts(dropna=False).to_string())
    rows = []
    start = time.perf_counter()
    print('\n' + '=' * 120)
    print('DOWNSAMPLING RAW RECORDINGS')
    print('=' * 120)
    for i, source_file in enumerate(files, start=1):
        source_root = next((root for root in SOURCE_DIRS if source_file.is_relative_to(root)))
        print(f'[{i:03d}/{len(files):03d}] {source_file.name}')
        row = process_one_file(source_file=source_file, source_root=source_root, destination_root=OUT_ROOT)
        rows.append(row)
    elapsed_seconds = time.perf_counter() - start
    qc = pd.DataFrame(rows)
    qc_path = QC_DIR / 'DOWNSAMPLING_12P8_TO_6P4_QC.csv'
    qc.to_csv(qc_path, index=False, encoding='utf-8-sig')
    config = {'analysis': 'sampling_rate_sensitivity', 'native_sampling_rate_hz': FS_NATIVE, 'target_sampling_rate_hz': FS_TARGET, 'resampling_method': 'scipy.signal.resample_poly', 'up': UP, 'down': DOWN, 'anti_aliasing': True, 'expected_signal_channels': EXPECTED_SIGNAL_CHANNELS, 'raw_csv_files_found': len(files), 'source_directories': [str(x) for x in SOURCE_DIRS], 'output_root': str(OUT_ROOT), 'qc_file': str(qc_path), 'elapsed_seconds': elapsed_seconds, 'elapsed_minutes': elapsed_seconds / 60.0}
    config_path = QC_DIR / 'sampling_sensitivity_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    ok = int((qc['status'] == 'OK').sum())
    skipped = int((qc['status'] == 'SKIPPED_EXISTS').sum())
    print('\n' + '=' * 120)
    print('FINISHED')
    print('=' * 120)
    print(f'Files found      : {len(files)}')
    print(f'Processed        : {ok}')
    print(f'Skipped existing : {skipped}')
    print(f'Elapsed time     : {elapsed_seconds / 60.0:.2f} min')
    print('\nQC output:')
    print(qc_path)
    print('\nConfiguration:')
    print(config_path)
    print('\n6.4-kHz shadow dataset:')
    print(OUT_ROOT)
    print('\n' + '=' * 120)
    print('NEXT STEP')
    print('=' * 120)
    print('Run the SAME physics-guided feature-extraction pipeline using the 6.4-kHz shadow dataset and FS=6400.')
    print('Then rerun the complete crossed constituent-level evaluation with all preprocessing, model fitting, and threshold calibration performed within the corresponding training folds.')
    print('\nDo NOT apply the original 12.8-kHz trained classifiers directly to the 6.4-kHz features.')
if __name__ == '__main__':
    main()
