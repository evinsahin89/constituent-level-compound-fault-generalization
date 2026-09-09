"""Physics-guided feature extraction for the 6.4-kHz sampling-rate analysis.

The run-level physics representation is regenerated from anti-aliased 6.4-kHz
recordings while reusing the original manifests. Native window boundaries are
mapped to the reduced sampling rate, angle resampling remains at 256 samples
per revolution, and the same physical feature definitions are retained.

Feature selection is not repeated at 6.4 kHz; the native locked feature map is
applied by the downstream evaluator.
"""
from pathlib import Path
import gc
import json
import math
import re
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.signal import hilbert, periodogram
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
ROOT = Path('D:\\yay\u0131nlar\\Multi-mode Fault Diagnosis Datasets of Three-phase')
PROTOCOL_DIR = ROOT / 'EXPERIMENT_PROTOCOL'
SHADOW_RAW_ROOT = ROOT / 'SAMPLING_RATE_SENSITIVITY' / 'RAW_6P4KHZ'
RESULT_DIR = ROOT / 'SAMPLING_RATE_SENSITIVITY' / 'PHYSICS_6P4KHZ'
RESULT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_NAMES = ['train', 'validation', 'P1', 'P2', 'P3', 'P4']
RUN_MANIFESTS = {name: PROTOCOL_DIR / f'runs_{name}.csv' for name in SPLIT_NAMES}
WINDOW_MANIFESTS = {name: PROTOCOL_DIR / f'windows_{name}.csv' for name in SPLIT_NAMES}
NATIVE_FS = 12800
FS = 6400
DOWNSAMPLE_RATIO = FS / NATIVE_FS
WINDOW_SECONDS = 2.0
NATIVE_WINDOW_SAMPLES = int(NATIVE_FS * WINDOW_SECONDS)
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
EXPECTED_SAMPLES = FS * 90
SIGNAL_USECOLS = [3, 4, 5, 6, 7, 8]
EXPECTED_CSV_COLUMNS = 9
SAMPLES_PER_REV = 256
MIN_REVOLUTIONS_PER_WINDOW = 4.0
MAX_WINDOWS_PER_RUN = 12
MIN_KEYPHASE_EDGE_GAP_SAMPLES = 32
EPS = 1e-12
ORDER_BAND_HALF_WIDTH = 0.12
ELECTRICAL_ORDER_SEARCH_MIN = 0.5
ELECTRICAL_ORDER_SEARCH_MAX = 30.0
BPFO = 3.585
BPFI = 5.415
BSF = 2.357
VIB_ORDERS = {'shaft_1x': 1.0, 'shaft_2x': 2.0, 'shaft_3x': 3.0, 'BSF': BSF, 'BPFO': BPFO, 'BPFI': BPFI, '2BSF': 2.0 * BSF, '2BPFO': 2.0 * BPFO, '2BPFI': 2.0 * BPFI}
TOP_SIGNATURE_FEATURES = 5
MIN_SIGNATURE_AUC = 0.65
COMPOUND_OVERRIDES = {'bearing_outer_H_and_inner_H': ('bearing_outer_H', 'bearing_inner_H')}

def robust_log10(x):
    return np.log10(np.maximum(x, EPS))

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def sanitize_name(text):
    return re.sub('[^A-Za-z0-9_]+', '_', str(text))

def load_complete_inventory():
    run_frames = []
    window_frames = []
    print('\n' + '=' * 100)
    print('LOAD COMPLETE INVENTORY')
    print('=' * 100)
    for name in SPLIT_NAMES:
        rp = RUN_MANIFESTS[name]
        wp = WINDOW_MANIFESTS[name]
        if not rp.exists():
            raise FileNotFoundError(str(rp))
        if not wp.exists():
            raise FileNotFoundError(str(wp))
        rdf = pd.read_csv(rp)
        wdf = pd.read_csv(wp)
        run_frames.append(rdf)
        window_frames.append(wdf)
        print(f'{name:12s}: {len(rdf):4d} runs | {len(wdf):6d} windows')
    runs = pd.concat(run_frames, ignore_index=True)
    windows = pd.concat(window_frames, ignore_index=True)
    for col in ['run_id', 'fault_raw']:
        if col not in runs.columns:
            raise RuntimeError(f'runs manifest missing column: {col}')
    for col in ['run_id', 'window_id', 'start_sample', 'end_sample']:
        if col not in windows.columns:
            raise RuntimeError(f'windows manifest missing column: {col}')
    runs['run_id'] = runs['run_id'].astype(str)
    runs['fault_raw'] = runs['fault_raw'].astype(str)
    windows['run_id'] = windows['run_id'].astype(str)
    windows['window_id'] = windows['window_id'].astype(str)
    dup = runs.groupby('run_id')['fault_raw'].nunique()
    bad = dup[dup > 1]
    if len(bad) > 0:
        raise RuntimeError(f'Conflicting fault_raw for same run_id: {bad.index.tolist()[:10]}')
    runs = runs.drop_duplicates('run_id').reset_index(drop=True)
    windows = windows.drop_duplicates('window_id').reset_index(drop=True)
    if len(runs) != 288:
        raise RuntimeError(f'Expected 288 runs, found {len(runs)}')
    if set(runs['run_id']) != set(windows['run_id']):
        raise RuntimeError('Run/window inventory mismatch')
    lengths = windows['end_sample'] - windows['start_sample']
    if not (lengths == NATIVE_WINDOW_SAMPLES).all():
        raise RuntimeError(f'Native window-manifest length mismatch; expected {NATIVE_WINDOW_SAMPLES}.')
    filepath_col = None
    for candidate in ['filepath', 'file_path', 'csv_path', 'path', 'source_path']:
        if candidate in runs.columns:
            filepath_col = candidate
            break
    if filepath_col is None:
        for candidate in ['filepath', 'file_path', 'csv_path', 'path', 'source_path']:
            if candidate in windows.columns:
                filepath_col = candidate
                fp_map = windows[['run_id', candidate]].drop_duplicates('run_id').set_index('run_id')[candidate]
                runs[candidate] = runs['run_id'].map(fp_map)
                break
    if filepath_col is None:
        raise RuntimeError('No original CSV filepath column found in run/window manifests. Expected one of: filepath, file_path, csv_path, path, source_path')
    source_csv = runs[filepath_col].copy()

    def _missing_source(v):
        if v is None:
            return True
        try:
            if pd.isna(v):
                return True
        except Exception:
            pass
        s = str(v).strip()
        return s == '' or s.lower() in {'nan', 'none', 'null', '<na>'}
    window_path_col = next((candidate for candidate in ['filepath', 'file_path', 'csv_path', 'path', 'source_path'] if candidate in windows.columns), None)
    if window_path_col is not None:
        window_path_map = windows[['run_id', window_path_col]].dropna(subset=[window_path_col]).drop_duplicates('run_id').set_index('run_id')[window_path_col]
        missing_mask = source_csv.apply(_missing_source)
        if missing_mask.any():
            source_csv.loc[missing_mask] = runs.loc[missing_mask, 'run_id'].map(window_path_map)
    runs['source_csv'] = source_csv
    runs = add_condition_columns(runs)
    runs['fault_role'] = runs['fault_raw'].apply(classify_fault_role)
    runs.to_csv(RESULT_DIR / 'run_inventory.csv', index=False, encoding='utf-8-sig')
    print(f'\nRuns: {len(runs)}')
    print(f"Fault classes: {runs['fault_raw'].nunique()}")
    print('\nFault roles:')
    print(runs['fault_role'].value_counts().to_string())
    print('\nCondition count:', runs['condition_key'].nunique())
    return (runs, windows)

def classify_fault_role(fault_raw):
    low = str(fault_raw).strip().lower()
    if low in {'health', 'healthy', 'normal', 'nc', 'normal_condition'}:
        return 'health'
    if '_and_' in low:
        return 'compound'
    return 'single'

def add_condition_columns(runs):
    """
    Operating condition resolution priority:
      1) manifest condition_id / condition_key
      2) explicit manifest mode/torque/rpm columns
      3) original CSV path / filename

    No signal-based relabeling is performed here.
    """
    out = runs.copy()
    lower_to_actual = {str(c).lower(): c for c in out.columns}

    def first_existing(names):
        for name in names:
            actual = lower_to_actual.get(str(name).lower())
            if actual is not None:
                return actual
        return None
    condition_col = first_existing(['condition_id', 'condition_key', 'condition', 'operating_condition'])
    mode_col = first_existing(['mode', 'operation_mode', 'working_mode', 'mode_resolved'])
    torque_col = first_existing(['torque_nm', 'torque_Nm', 'torque', 'nominal_torque', 'torque_resolved'])
    rpm_col = first_existing(['rpm_nominal', 'rpm', 'speed_rpm', 'nominal_rpm', 'rpm_resolved'])
    condition_pattern = re.compile('(speed_circulation|torque_circulation)__T(\\d+(?:\\.\\d+)?)__R(\\d+(?:\\.\\d+)?)', flags=re.IGNORECASE)
    path_pattern = re.compile('(speed_circulation|torque_circulation)_(\\d+(?:\\.\\d+)?)Nm_(\\d+(?:\\.\\d+)?)rpm', flags=re.IGNORECASE)

    def missing(value):
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
        text_value = str(value).strip()
        return text_value == '' or text_value.lower() in {'nan', 'none', 'null', '<na>'}

    def numeric_value(value):
        if missing(value):
            return np.nan
        try:
            return float(value)
        except Exception:
            pass
        m = re.search('-?\\d+(?:\\.\\d+)?', str(value))
        if m:
            return float(m.group(0))
        return np.nan
    parsed_mode = []
    parsed_torque = []
    parsed_rpm = []
    resolution_source = []
    for _, row in out.iterrows():
        mode = None
        torque = np.nan
        rpm = np.nan
        source = None
        if condition_col is not None and (not missing(row[condition_col])):
            m = condition_pattern.search(str(row[condition_col]))
            if m:
                mode = m.group(1).lower()
                torque = float(m.group(2))
                rpm = float(m.group(3))
                source = f'manifest:{condition_col}'
        if mode_col is not None and (not missing(row[mode_col])):
            explicit_mode = str(row[mode_col]).strip().lower()
            if explicit_mode in {'speed_circulation', 'torque_circulation'}:
                mode = explicit_mode
        if torque_col is not None and (not missing(row[torque_col])):
            value = numeric_value(row[torque_col])
            if np.isfinite(value):
                torque = value
        if rpm_col is not None and (not missing(row[rpm_col])):
            value = numeric_value(row[rpm_col])
            if np.isfinite(value):
                rpm = value
        if source is None and mode is not None and np.isfinite(torque) and np.isfinite(rpm):
            source = 'explicit_manifest_columns'
        path_text = str(row.get('source_csv', ''))
        m = path_pattern.search(path_text)
        if m:
            if mode is None:
                mode = m.group(1).lower()
            if not np.isfinite(torque):
                torque = float(m.group(2))
            if not np.isfinite(rpm):
                rpm = float(m.group(3))
            if source is None:
                source = 'source_csv'
        parsed_mode.append(mode)
        parsed_torque.append(torque)
        parsed_rpm.append(rpm)
        resolution_source.append(source)
    out['mode_resolved'] = parsed_mode
    out['torque_resolved'] = parsed_torque
    out['rpm_resolved'] = parsed_rpm
    out['condition_resolution_source'] = resolution_source
    bad_mask = out[['mode_resolved', 'torque_resolved', 'rpm_resolved']].isna().any(axis=1)
    if bad_mask.any():
        debug_cols = ['run_id', 'fault_raw', 'source_csv']
        if condition_col is not None:
            debug_cols.append(condition_col)
        for col in [mode_col, torque_col, rpm_col]:
            if col is not None and col not in debug_cols:
                debug_cols.append(col)
        bad = out.loc[bad_mask, debug_cols]
        print('\nUNRESOLVED CONDITION ROWS:')
        print(bad.head(20).to_string(index=False))
        raise RuntimeError(f'Could not resolve operating condition for {int(bad_mask.sum())} run(s).')
    out['torque_resolved'] = out['torque_resolved'].astype(float)
    out['rpm_resolved'] = out['rpm_resolved'].astype(float)
    out['condition_key'] = out['mode_resolved'].astype(str) + '__T' + out['torque_resolved'].round().astype(int).astype(str) + '__R' + out['rpm_resolved'].round().astype(int).astype(str)
    return out

def resolve_compound_mapping(runs):
    singles = set(runs.loc[runs['fault_role'] == 'single', 'fault_raw'].astype(str))
    compound_labels = sorted(runs.loc[runs['fault_role'] == 'compound', 'fault_raw'].unique())
    rows = []
    mapping = {}
    for label in compound_labels:
        resolved = None
        method = None
        if label in COMPOUND_OVERRIDES:
            a, b = COMPOUND_OVERRIDES[label]
            if a in singles and b in singles:
                resolved = (a, b)
                method = 'explicit_override'
        if resolved is None:
            left, right = label.split('_and_', 1)
            if left in singles and right in singles:
                resolved = (left, right)
                method = 'exact_fault_raw_split'
            else:
                candidates = []
                if left.startswith('bearing_') and (not right.startswith('bearing_')):
                    candidates.append('bearing_' + right)
                if left.startswith('winding_') and (not right.startswith('winding_')):
                    candidates.append('winding_' + right)
                for cand in candidates:
                    if left in singles and cand in singles:
                        resolved = (left, cand)
                        method = 'verified_shorthand_repair'
                        break
        if resolved is not None:
            mapping[label] = resolved
            status = 'RESOLVED'
            a, b = resolved
        else:
            status = 'UNRESOLVED'
            a, b = (None, None)
        rows.append({'compound_fault_raw': label, 'constituent_A': a, 'constituent_B': b, 'status': status, 'resolution_method': method})
    table = pd.DataFrame(rows)
    table.to_csv(RESULT_DIR / 'compound_constituent_mapping.csv', index=False, encoding='utf-8-sig')
    unresolved = table[table['status'] != 'RESOLVED']
    print('\n' + '=' * 100)
    print('COMPOUND MAPPING')
    print('=' * 100)
    print(table.to_string(index=False))
    if len(unresolved) > 0:
        print('\nWARNING: Unresolved compound labels will be skipped in Test A:')
        print(unresolved['compound_fault_raw'].tolist())
    return (mapping, table)

def resolve_source_csv(path_text):
    """Resolve the ORIGINAL path stored by the experiment manifests."""
    p = Path(str(path_text))
    if p.exists():
        return p
    p2 = ROOT / p
    if p2.exists():
        return p2
    matches = list(ROOT.rglob(p.name))
    matches = [m for m in matches if SHADOW_RAW_ROOT not in m.parents]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f'Could not resolve original source CSV: {path_text}')

def resolve_shadow_csv(path_text):
    """Map a manifest's native CSV path to its 6.4-kHz shadow counterpart."""
    p = Path(str(path_text))
    parent_name = p.parent.name
    if parent_name in {'MCC5-THU Motor_speed_circulation', 'MCC5-THU Motor_torque_circulation'}:
        candidate = SHADOW_RAW_ROOT / parent_name / p.name
        if candidate.exists():
            return candidate
    parts = list(p.parts)
    for folder_name in ['MCC5-THU Motor_speed_circulation', 'MCC5-THU Motor_torque_circulation']:
        if folder_name in parts:
            candidate = SHADOW_RAW_ROOT / folder_name / p.name
            if candidate.exists():
                return candidate
    matches = list(SHADOW_RAW_ROOT.rglob(p.name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f'Could not resolve 6.4-kHz shadow CSV for: {path_text}')

def validate_inputs(runs):
    print('\n' + '=' * 100)
    print('6.4-kHz SHADOW INPUT VALIDATION')
    print('=' * 100)
    if not SHADOW_RAW_ROOT.exists():
        raise FileNotFoundError(f'Shadow raw root not found: {SHADOW_RAW_ROOT}')
    bad = []
    resolved = []
    for i, row in runs.iterrows():
        run_id = str(row['run_id'])
        try:
            p = resolve_shadow_csv(row['source_csv'])
            preview = pd.read_csv(p, header=None, nrows=4)
            if preview.shape[1] != EXPECTED_CSV_COLUMNS:
                bad.append((run_id, f'expected 9 columns, found {preview.shape[1]}'))
            else:
                resolved.append(str(p))
        except Exception as exc:
            bad.append((run_id, repr(exc)))
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(runs)} checked')
    if bad:
        raise RuntimeError(f'6.4-kHz input problems found. First examples: {bad[:10]}')
    if len(set(resolved)) != len(runs):
        raise RuntimeError('Shadow CSV mapping is not one-to-one across the 288 runs.')
    print('\nPASS: all 288 manifest runs map uniquely to 6.4-kHz shadow CSVs.')

def read_shadow_signals(csv_path):
    """Read six vibration/current channels from one 6.4-kHz shadow CSV."""
    df = pd.read_csv(csv_path, header=None, usecols=SIGNAL_USECOLS, dtype=np.float32, engine='c')
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if arr.shape != (EXPECTED_SAMPLES, 6):
        raise RuntimeError(f'Unexpected 6.4-kHz signal shape for {csv_path}: {arr.shape}; expected {(EXPECTED_SAMPLES, 6)}')
    return arr

def read_keyphase(csv_path):
    series = pd.read_csv(csv_path, header=None, usecols=[1], dtype=np.float32, engine='c').iloc[:, 0].to_numpy(dtype=np.float32, copy=False)
    return series

def _debounce_edges(edges, min_gap):
    if len(edges) == 0:
        return edges
    kept = [int(edges[0])]
    for e in edges[1:]:
        e = int(e)
        if e - kept[-1] >= min_gap:
            kept.append(e)
    return np.asarray(kept, dtype=np.int64)

def _edge_quality(edges):
    if len(edges) < 4:
        return -np.inf
    gaps = np.diff(edges).astype(float)
    gaps = gaps[gaps > 0]
    if len(gaps) < 3:
        return -np.inf
    med = np.median(gaps)
    if med < FS * 60 / 6000 or med > FS * 60 / 100:
        plausibility_penalty = 2.0
    else:
        plausibility_penalty = 0.0
    cv = np.std(gaps) / (np.mean(gaps) + EPS)
    return math.log(len(edges) + 1.0) - 0.25 * cv - plausibility_penalty

def detect_keyphase_edges(keyphase):
    x = np.asarray(keyphase, dtype=np.float32)
    q05, q95 = np.percentile(x, [5, 95])
    if not np.isfinite(q05) or not np.isfinite(q95) or abs(q95 - q05) < 1e-09:
        return (np.array([], dtype=np.int64), 'flat')
    threshold = 0.5 * (q05 + q95)
    high = x > threshold
    low = x < threshold
    rising_high = np.flatnonzero(~high[:-1] & high[1:]) + 1
    rising_low = np.flatnonzero(~low[:-1] & low[1:]) + 1
    rising_high = _debounce_edges(rising_high, MIN_KEYPHASE_EDGE_GAP_SAMPLES)
    rising_low = _debounce_edges(rising_low, MIN_KEYPHASE_EDGE_GAP_SAMPLES)
    score_high = _edge_quality(rising_high)
    score_low = _edge_quality(rising_low)
    if score_high >= score_low:
        return (rising_high, 'high_pulse')
    return (rising_low, 'low_pulse')

def angle_resample_window(signal_6ch, edges, start, end):
    """
    Use 1/rev keyphase edges to linearly map sample index -> revolution count,
    then resample the six signals at uniform shaft angle.
    """
    if len(edges) < 4:
        return None
    inside = edges[(edges >= start) & (edges < end)]
    if len(inside) < int(MIN_REVOLUTIONS_PER_WINDOW) + 1:
        return None
    first_edge = int(inside[0])
    last_edge = int(inside[-1])
    if last_edge <= first_edge:
        return None
    i0 = int(np.searchsorted(edges, first_edge))
    i1 = int(np.searchsorted(edges, last_edge))
    rev_start = float(i0)
    rev_stop = float(i1)
    n_revs = rev_stop - rev_start
    if n_revs < MIN_REVOLUTIONS_PER_WINDOW:
        return None
    target_rev = np.arange(rev_start, rev_stop, 1.0 / SAMPLES_PER_REV, dtype=np.float64)
    if len(target_rev) < SAMPLES_PER_REV * MIN_REVOLUTIONS_PER_WINDOW:
        return None
    edge_rev = np.arange(len(edges), dtype=np.float64)
    target_sample = np.interp(target_rev, edge_rev, edges.astype(np.float64))
    local_x = np.arange(start, end, dtype=np.float64)
    resampled = np.empty((len(target_sample), 6), dtype=np.float32)
    for c in range(6):
        resampled[:, c] = np.interp(target_sample, local_x, signal_6ch[:, c].astype(np.float64)).astype(np.float32)
    return (resampled, n_revs)

def order_power_spectrum(x, samples_per_rev=SAMPLES_PER_REV):
    """
    Angle-domain one-sided spectrum with `scaling="spectrum"`.
    Summed bins therefore retain an approximately mean-square power scale,
    which is preferable to raw FFT magnitudes for the descriptive additivity test.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n < 128:
        return (None, None)
    orders, power = periodogram(x, fs=float(samples_per_rev), window='hann', detrend='constant', scaling='spectrum', return_onesided=True, axis=0)
    return (orders, power)

def band_power_metrics(orders, power, center_order, half_width=ORDER_BAND_HALF_WIDTH):
    if orders is None or power is None:
        return (None, None)
    mask = (orders >= center_order - half_width) & (orders <= center_order + half_width)
    valid_total = (orders >= 0.25) & (orders <= min(orders[-1], 60.0))
    if not np.any(mask) or not np.any(valid_total):
        return (None, None)
    band_power = np.sum(power[mask], axis=0)
    total_power = np.sum(power[valid_total], axis=0) + EPS
    ratio = band_power / total_power
    return (band_power, ratio)

def dominant_electrical_order(orders, current_power):
    agg = np.sum(current_power, axis=1)
    mask = (orders >= ELECTRICAL_ORDER_SEARCH_MIN) & (orders <= ELECTRICAL_ORDER_SEARCH_MAX)
    if not np.any(mask):
        return np.nan
    idxs = np.flatnonzero(mask)
    best = idxs[np.argmax(agg[idxs])]
    return float(orders[best])

def complex_coefficient_angle_domain(x, order):
    """Complex Fourier coefficient at arbitrary shaft order."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rev = np.arange(n, dtype=np.float64) / SAMPLES_PER_REV
    basis = np.exp(-1j * 2.0 * np.pi * order * rev)
    return np.sum((x - np.mean(x)) * basis) / max(n, 1)

def current_sequence_ratio(current_3ch, electrical_order):
    if not np.isfinite(electrical_order):
        return np.nan
    ia = complex_coefficient_angle_domain(current_3ch[:, 0], electrical_order)
    ib = complex_coefficient_angle_domain(current_3ch[:, 1], electrical_order)
    ic = complex_coefficient_angle_domain(current_3ch[:, 2], electrical_order)
    a = np.exp(1j * 2.0 * np.pi / 3.0)
    seq1 = (ia + a * ib + a ** 2 * ic) / 3.0
    seq2 = (ia + a ** 2 * ib + a * ic) / 3.0
    mag1 = abs(seq1)
    mag2 = abs(seq2)
    pos = max(mag1, mag2)
    neg = min(mag1, mag2)
    return float(neg / (pos + EPS))

def extract_window_features(angle_signal):
    features = {}
    vib = angle_signal[:, 0:3].astype(np.float64)
    cur = angle_signal[:, 3:6].astype(np.float64)
    vib_rms = np.sqrt(np.mean(vib ** 2, axis=0))
    cur_rms = np.sqrt(np.mean(cur ** 2, axis=0))
    for i, name in enumerate(['vib_h', 'vib_a', 'vib_v']):
        features[f'{name}_rms'] = float(vib_rms[i])
    for i, name in enumerate(['cur_A', 'cur_B', 'cur_C']):
        features[f'{name}_rms'] = float(cur_rms[i])
    features['vib_rms_mean'] = float(np.mean(vib_rms))
    features['current_rms_mean'] = float(np.mean(cur_rms))
    features['current_rms_cv'] = float(np.std(cur_rms) / (np.mean(cur_rms) + EPS))
    vib_orders, vib_power = order_power_spectrum(vib)
    if vib_orders is None:
        return None
    for label, order in VIB_ORDERS.items():
        bandpowers, ratios = band_power_metrics(vib_orders, vib_power, order)
        if ratios is None or np.any(~np.isfinite(ratios)):
            for axis in ['h', 'a', 'v']:
                features[f'vib_raw_{axis}_{label}_bandpower'] = np.nan
                features[f'vib_raw_{axis}_{label}_logratio'] = np.nan
        else:
            for j, axis in enumerate(['h', 'a', 'v']):
                features[f'vib_raw_{axis}_{label}_bandpower'] = float(bandpowers[j])
                features[f'vib_raw_{axis}_{label}_logratio'] = float(robust_log10(ratios[j]))
            features[f'vib_raw_mean_{label}_bandpower'] = float(np.mean(bandpowers))
            features[f'vib_raw_mean_{label}_logratio'] = float(robust_log10(np.mean(ratios)))
    vib_env = np.abs(hilbert(vib - np.mean(vib, axis=0, keepdims=True), axis=0))
    env_orders, env_power = order_power_spectrum(vib_env)
    for label, order in VIB_ORDERS.items():
        bandpowers, ratios = band_power_metrics(env_orders, env_power, order)
        if ratios is None or np.any(~np.isfinite(ratios)):
            for axis in ['h', 'a', 'v']:
                features[f'vib_env_{axis}_{label}_bandpower'] = np.nan
                features[f'vib_env_{axis}_{label}_logratio'] = np.nan
        else:
            for j, axis in enumerate(['h', 'a', 'v']):
                features[f'vib_env_{axis}_{label}_bandpower'] = float(bandpowers[j])
                features[f'vib_env_{axis}_{label}_logratio'] = float(robust_log10(ratios[j]))
            features[f'vib_env_mean_{label}_bandpower'] = float(np.mean(bandpowers))
            features[f'vib_env_mean_{label}_logratio'] = float(robust_log10(np.mean(ratios)))
    cur_orders, cur_power = order_power_spectrum(cur)
    electrical_order = dominant_electrical_order(cur_orders, cur_power)
    features['electrical_order'] = electrical_order
    features['current_negative_sequence_ratio'] = current_sequence_ratio(cur, electrical_order)
    if np.isfinite(electrical_order):
        targets = {'carrier': electrical_order, 'carrier_minus_1x': max(0.05, electrical_order - 1.0), 'carrier_plus_1x': electrical_order + 1.0, 'carrier_minus_2x': max(0.05, electrical_order - 2.0), 'carrier_plus_2x': electrical_order + 2.0, 'second_harmonic': 2.0 * electrical_order}
        ia, ib, ic = (cur[:, 0], cur[:, 1], cur[:, 2])
        alpha = 2.0 / 3.0 * (ia - 0.5 * ib - 0.5 * ic)
        beta = 2.0 / 3.0 * (np.sqrt(3.0) / 2.0 * (ib - ic))
        space_mag = np.sqrt(alpha ** 2 + beta ** 2)
        sv_orders, sv_power = order_power_spectrum(space_mag)
        for label, center in targets.items():
            bandpowers, ratios = band_power_metrics(cur_orders, cur_power, center)
            if ratios is not None and np.all(np.isfinite(ratios)):
                features[f'current_phase_mean_{label}_bandpower'] = float(np.mean(bandpowers))
                features[f'current_phase_mean_{label}_logratio'] = float(robust_log10(np.mean(ratios)))
                for j, phase in enumerate(['A', 'B', 'C']):
                    features[f'current_{phase}_{label}_bandpower'] = float(bandpowers[j])
                    features[f'current_{phase}_{label}_logratio'] = float(robust_log10(ratios[j]))
            sv_bandpower, sv_ratio = band_power_metrics(sv_orders, sv_power, center)
            if sv_ratio is not None:
                sv_ratio_scalar = float(np.ravel(sv_ratio)[0])
                sv_bp_scalar = float(np.ravel(sv_bandpower)[0])
                features[f'current_spacevec_{label}_bandpower'] = sv_bp_scalar
                features[f'current_spacevec_{label}_logratio'] = float(robust_log10(sv_ratio_scalar))
    return features

def choose_windows_for_run(run_windows, max_windows):
    run_windows = run_windows.sort_values('start_sample').reset_index(drop=True)
    if len(run_windows) <= max_windows:
        return run_windows
    idx = np.linspace(0, len(run_windows) - 1, num=max_windows, dtype=int)
    return run_windows.iloc[idx].reset_index(drop=True)

def aggregate_window_feature_dicts(feature_dicts):
    if len(feature_dicts) == 0:
        return {}
    df = pd.DataFrame(feature_dicts)
    out = {}
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[col] = np.nan
            continue
        out[col] = float(np.median(vals))
        out[col + '__iqr'] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    return out

def extract_all_run_features(runs, windows):
    feature_path = RESULT_DIR / 'RUN_PHYSICS_FEATURES.csv'
    qc_path = RESULT_DIR / 'feature_extraction_qc.csv'
    if feature_path.exists() and qc_path.exists():
        cached = pd.read_csv(feature_path)
        cached['run_id'] = cached['run_id'].astype(str)
        if set(cached['run_id']) == set(runs['run_id']):
            print('\n6.4-kHz RUN_PHYSICS_FEATURES.csv already exists; reusing it.')
            return cached
    print('\n' + '=' * 100)
    print('EXTRACT 6.4-kHz RUN-LEVEL PHYSICS FEATURES')
    print('=' * 100)
    rows = []
    qc_rows = []
    windows_by_run = {run_id: group.copy() for run_id, group in windows.groupby('run_id')}
    for counter, (_, run_row) in enumerate(runs.iterrows(), start=1):
        run_id = str(run_row['run_id'])
        csv_path = resolve_shadow_csv(run_row['source_csv'])
        keyphase = read_keyphase(csv_path)
        if len(keyphase) != EXPECTED_SAMPLES:
            raise RuntimeError(f'Unexpected keyphase length for {run_id}: {len(keyphase)}; expected {EXPECTED_SAMPLES}')
        edges, edge_mode = detect_keyphase_edges(keyphase)
        signals = read_shadow_signals(csv_path)
        selected_windows = choose_windows_for_run(windows_by_run[run_id], MAX_WINDOWS_PER_RUN)
        window_features = []
        valid_revs = []
        rejected = 0
        for _, wrow in selected_windows.iterrows():
            native_start = int(wrow['start_sample'])
            native_end = int(wrow['end_sample'])
            start = int(round(native_start * DOWNSAMPLE_RATIO))
            end = int(round(native_end * DOWNSAMPLE_RATIO))
            if end - start != WINDOW_SAMPLES:
                raise RuntimeError(f'Scaled window length mismatch for {run_id}: native=({native_start},{native_end}), target=({start},{end})')
            if start < 0 or end > EXPECTED_SAMPLES:
                raise RuntimeError(f'Scaled window out of bounds for {run_id}: {start}:{end}')
            raw = np.asarray(signals[start:end, :], dtype=np.float32)
            angle_result = angle_resample_window(raw, edges, start, end)
            if angle_result is None:
                rejected += 1
                continue
            angle_signal, n_revs = angle_result
            feat = extract_window_features(angle_signal)
            if feat is None:
                rejected += 1
                continue
            window_features.append(feat)
            valid_revs.append(n_revs)
        agg = aggregate_window_feature_dicts(window_features)
        base = {'run_id': run_id, 'fault_raw': run_row['fault_raw'], 'fault_role': run_row['fault_role'], 'condition_key': run_row['condition_key'], 'mode': run_row['mode_resolved'], 'torque_nm': int(run_row['torque_resolved']), 'rpm_nominal': int(run_row['rpm_resolved'])}
        base.update(agg)
        rows.append(base)
        qc_rows.append({'run_id': run_id, 'fault_raw': run_row['fault_raw'], 'condition_key': run_row['condition_key'], 'shadow_csv': str(csv_path), 'sampling_rate_hz': FS, 'keyphase_edge_mode': edge_mode, 'keyphase_edges': len(edges), 'windows_requested': len(selected_windows), 'windows_valid': len(window_features), 'windows_rejected': rejected, 'median_revolutions_per_valid_window': float(np.median(valid_revs)) if valid_revs else np.nan})
        del signals
        del keyphase
        gc.collect()
        if counter % 10 == 0 or counter == len(runs):
            print(f'{counter}/{len(runs)} runs processed')
    feature_df = pd.DataFrame(rows)
    qc_df = pd.DataFrame(qc_rows)
    if feature_df['run_id'].nunique() != 288 or len(feature_df) != 288:
        raise RuntimeError(f'Expected exactly 288 run-level feature rows, found {len(feature_df)}.')
    feature_df.to_csv(feature_path, index=False, encoding='utf-8-sig')
    qc_df.to_csv(qc_path, index=False, encoding='utf-8-sig')
    bad_qc = qc_df[qc_df['windows_valid'] < 3]
    if len(bad_qc) > 0:
        print(f'\nWARNING: {len(bad_qc)} runs have <3 valid tach-referenced windows.')
        print(bad_qc[['run_id', 'fault_raw', 'windows_valid']].head(20).to_string(index=False))
    print('\nPASS: 6.4-kHz RUN_PHYSICS_FEATURES.csv created.')
    return feature_df

def get_numeric_feature_columns(feature_df):
    metadata = {'run_id', 'fault_raw', 'fault_role', 'condition_key', 'mode', 'torque_nm', 'rpm_nominal'}
    cols = []
    for col in feature_df.columns:
        if col in metadata:
            continue
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            cols.append(col)
    good = []
    for col in cols:
        finite_fraction = np.isfinite(pd.to_numeric(feature_df[col], errors='coerce').to_numpy(dtype=float)).mean()
        if finite_fraction >= 0.75:
            good.append(col)
    return good

def median_impute_from_reference(reference_df, target_df, feature_cols):
    ref = reference_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    tgt = target_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    medians = ref.median(axis=0, skipna=True)
    medians = medians.fillna(0.0)
    return (tgt.fillna(medians).to_numpy(dtype=np.float64), medians)

def compute_univariate_signature_auc(feature_df, feature_cols):
    """
    Signature discovery uses only health + the target SINGLE fault.
    Compound runs are never used to select signature features.
    """
    health_labels = feature_df.loc[feature_df['fault_role'] == 'health', 'fault_raw'].unique()
    if len(health_labels) != 1:
        raise RuntimeError(f'Expected exactly one health label, found {health_labels.tolist()}')
    health_label = health_labels[0]
    single_faults = sorted(feature_df.loc[feature_df['fault_role'] == 'single', 'fault_raw'].unique())
    rows = []
    for fault in single_faults:
        subset = feature_df[feature_df['fault_raw'].isin([health_label, fault])].copy()
        y = (subset['fault_raw'] == fault).astype(int).to_numpy()
        for feature in feature_cols:
            x = pd.to_numeric(subset[feature], errors='coerce').to_numpy(dtype=float)
            finite = np.isfinite(x)
            if finite.sum() < 8 or len(np.unique(y[finite])) < 2:
                continue
            try:
                auc = roc_auc_score(y[finite], x[finite])
            except Exception:
                continue
            discriminative_auc = max(auc, 1.0 - auc)
            direction = 1 if auc >= 0.5 else -1
            fault_vals = x[(y == 1) & finite]
            health_vals = x[(y == 0) & finite]
            rows.append({'fault_raw': fault, 'feature': feature, 'auc_raw': float(auc), 'auc_discriminative': float(discriminative_auc), 'direction': int(direction), 'fault_median': float(np.median(fault_vals)) if len(fault_vals) else np.nan, 'health_median': float(np.median(health_vals)) if len(health_vals) else np.nan, 'median_effect_signed': float(direction * (np.median(fault_vals) - np.median(health_vals))) if len(fault_vals) and len(health_vals) else np.nan, 'n_fault': int(np.sum((y == 1) & finite)), 'n_health': int(np.sum((y == 0) & finite))})
    auc_df = pd.DataFrame(rows)
    auc_df.to_csv(RESULT_DIR / 'TEST_B_UNIVARIATE_SIGNATURE_AUC.csv', index=False, encoding='utf-8-sig')
    sig_rows = []
    for fault, group in auc_df.groupby('fault_raw'):
        group = group.sort_values(['auc_discriminative', 'median_effect_signed'], ascending=[False, False])
        eligible = group[group['auc_discriminative'] >= MIN_SIGNATURE_AUC]
        chosen = eligible.head(TOP_SIGNATURE_FEATURES)
        if len(chosen) < TOP_SIGNATURE_FEATURES:
            chosen = group.head(TOP_SIGNATURE_FEATURES)
        for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
            sig_rows.append({'fault_raw': fault, 'rank': rank, 'feature': row['feature'], 'auc_discriminative': row['auc_discriminative'], 'auc_raw': row['auc_raw'], 'direction': int(row['direction']), 'median_effect_signed': row['median_effect_signed']})
    signatures = pd.DataFrame(sig_rows)
    signatures.to_csv(RESULT_DIR / 'SIGNATURE_FEATURES.csv', index=False, encoding='utf-8-sig')
    print('\n' + '=' * 100)
    print('TEST B1 — TOP SINGLE-FAULT SIGNATURE FEATURES')
    print('=' * 100)
    top_print = signatures[signatures['rank'] <= 3]
    print(top_print.to_string(index=False))
    return (auc_df, signatures, health_label)

def condition_heldout_probe_auc(feature_df, feature_cols, health_label):
    """
    Two run-level probes per single fault:
      A) target single fault vs health
      B) target single fault vs all other health/single runs

    LeaveOneGroupOut uses operating condition as the group.
    No compound run is used in these probes.
    """
    base = feature_df[feature_df['fault_role'].isin(['health', 'single'])].copy()
    single_faults = sorted(base.loc[base['fault_role'] == 'single', 'fault_raw'].unique())
    rows = []
    for fault in single_faults:
        for probe_name in ['vs_health', 'vs_other_singles']:
            if probe_name == 'vs_health':
                df = base[base['fault_raw'].isin([health_label, fault])].copy()
            else:
                df = base.copy()
            y = (df['fault_raw'] == fault).astype(int).to_numpy()
            groups = df['condition_key'].astype(str).to_numpy()
            oof_prob = np.full(len(df), np.nan, dtype=float)
            logo = LeaveOneGroupOut()
            valid_fold_count = 0
            for train_idx, test_idx in logo.split(np.zeros(len(df)), y, groups):
                y_train = y[train_idx]
                y_test = y[test_idx]
                if len(np.unique(y_train)) < 2:
                    continue
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                X_train_raw = train_df[feature_cols].apply(pd.to_numeric, errors='coerce')
                X_test_raw = test_df[feature_cols].apply(pd.to_numeric, errors='coerce')
                medians = X_train_raw.median(axis=0, skipna=True).fillna(0.0)
                X_train = X_train_raw.fillna(medians).to_numpy(dtype=np.float64)
                X_test = X_test_raw.fillna(medians).to_numpy(dtype=np.float64)
                model = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=5000, class_weight='balanced', solver='liblinear', random_state=42))])
                model.fit(X_train, y_train)
                oof_prob[test_idx] = model.predict_proba(X_test)[:, 1]
                valid_fold_count += 1
            finite = np.isfinite(oof_prob)
            if finite.sum() == 0 or len(np.unique(y[finite])) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y[finite], oof_prob[finite])
            rows.append({'fault_raw': fault, 'probe': probe_name, 'condition_heldout_auc': float(auc) if np.isfinite(auc) else np.nan, 'n_runs': int(finite.sum()), 'n_positive': int(np.sum(y[finite] == 1)), 'n_negative': int(np.sum(y[finite] == 0)), 'n_valid_condition_folds': int(valid_fold_count)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / 'TEST_B_CONDITION_HELDOUT_AUC.csv', index=False, encoding='utf-8-sig')
    print('\n' + '=' * 100)
    print('TEST B2 — CONDITION-HELD-OUT RUN-LEVEL AUC')
    print('=' * 100)
    print(out.to_string(index=False))
    return out

def matched_run_row(feature_df, fault_raw, condition_key):
    hit = feature_df[(feature_df['fault_raw'] == fault_raw) & (feature_df['condition_key'] == condition_key)]
    if len(hit) != 1:
        return None
    return hit.iloc[0]

def test_a_preservation_and_additivity(feature_df, signatures, compound_mapping, health_label):
    preservation_rows = []
    additivity_rows = []
    signature_by_fault = {fault: group.sort_values('rank') for fault, group in signatures.groupby('fault_raw')}
    compound_rows = feature_df[feature_df['fault_role'] == 'compound']
    for _, comp_row in compound_rows.iterrows():
        compound_label = comp_row['fault_raw']
        condition = comp_row['condition_key']
        if compound_label not in compound_mapping:
            continue
        fault_a, fault_b = compound_mapping[compound_label]
        row_h = matched_run_row(feature_df, health_label, condition)
        row_a = matched_run_row(feature_df, fault_a, condition)
        row_b = matched_run_row(feature_df, fault_b, condition)
        row_ab = comp_row
        if any((x is None for x in [row_h, row_a, row_b])):
            continue
        for constituent in [fault_a, fault_b]:
            sig = signature_by_fault.get(constituent)
            if sig is None or len(sig) == 0:
                continue
            single_row = row_a if constituent == fault_a else row_b
            constituent_feature_rows = []
            for _, srow in sig.iterrows():
                feature = srow['feature']
                direction = int(srow['direction'])
                try:
                    h = float(row_h[feature])
                    s = float(single_row[feature])
                    ab = float(row_ab[feature])
                except Exception:
                    continue
                if not np.all(np.isfinite([h, s, ab])):
                    continue
                single_evidence = direction * (s - h)
                compound_evidence = direction * (ab - h)
                if single_evidence > EPS:
                    preservation_ratio = compound_evidence / single_evidence
                else:
                    preservation_ratio = np.nan
                preserved_positive = int(compound_evidence > 0.0)
                preserved_half = int(single_evidence > EPS and compound_evidence >= 0.5 * single_evidence)
                constituent_feature_rows.append({'compound_fault_raw': compound_label, 'condition_key': condition, 'constituent': constituent, 'other_constituent': fault_b if constituent == fault_a else fault_a, 'feature': feature, 'signature_rank': int(srow['rank']), 'signature_auc': float(srow['auc_discriminative']), 'direction': direction, 'healthy_value': h, 'single_value': s, 'compound_value': ab, 'single_evidence': single_evidence, 'compound_evidence': compound_evidence, 'preservation_ratio': preservation_ratio, 'preserved_positive': preserved_positive, 'preserved_half': preserved_half})
            preservation_rows.extend(constituent_feature_rows)
        metadata = {'run_id', 'fault_raw', 'fault_role', 'condition_key', 'mode', 'torque_nm', 'rpm_nominal'}
        for feature in feature_df.columns:
            if feature in metadata:
                continue
            if not feature.endswith('_bandpower'):
                continue
            if not pd.api.types.is_numeric_dtype(feature_df[feature]):
                continue
            try:
                h = float(row_h[feature])
                a = float(row_a[feature])
                b = float(row_b[feature])
                ab = float(row_ab[feature])
            except Exception:
                continue
            if not np.all(np.isfinite([h, a, b, ab])):
                continue
            excess_a = a - h
            excess_b = b - h
            excess_ab = ab - h
            expected = excess_a + excess_b
            normalized_error = abs(excess_ab - expected) / (abs(excess_a) + abs(excess_b) + EPS)
            additivity_rows.append({'compound_fault_raw': compound_label, 'condition_key': condition, 'constituent_A': fault_a, 'constituent_B': fault_b, 'feature': feature, 'healthy_value': h, 'single_A_value': a, 'single_B_value': b, 'compound_value': ab, 'excess_A': excess_a, 'excess_B': excess_b, 'excess_compound': excess_ab, 'expected_additive_excess': expected, 'normalized_additivity_error': normalized_error})
    preservation = pd.DataFrame(preservation_rows)
    additivity = pd.DataFrame(additivity_rows)
    preservation.to_csv(RESULT_DIR / 'TEST_A_PRESERVATION_DETAIL.csv', index=False, encoding='utf-8-sig')
    additivity.to_csv(RESULT_DIR / 'TEST_A_ADDITIVITY_DETAIL.csv', index=False, encoding='utf-8-sig')
    if len(preservation) > 0:
        pres_summary = preservation.groupby(['compound_fault_raw', 'constituent']).agg(n_feature_condition=('preservation_ratio', 'size'), median_preservation_ratio=('preservation_ratio', 'median'), mean_preservation_ratio=('preservation_ratio', 'mean'), positive_evidence_rate=('preserved_positive', 'mean'), half_evidence_preserved_rate=('preserved_half', 'mean'), median_single_evidence=('single_evidence', 'median'), median_compound_evidence=('compound_evidence', 'median')).reset_index()
    else:
        pres_summary = pd.DataFrame()
    pres_summary.to_csv(RESULT_DIR / 'TEST_A_PRESERVATION_SUMMARY.csv', index=False, encoding='utf-8-sig')
    if len(additivity) > 0:
        add_summary = additivity.groupby('compound_fault_raw').agg(n_feature_condition=('normalized_additivity_error', 'size'), median_normalized_additivity_error=('normalized_additivity_error', 'median'), mean_normalized_additivity_error=('normalized_additivity_error', 'mean'), q25_normalized_additivity_error=('normalized_additivity_error', lambda x: np.percentile(x, 25)), q75_normalized_additivity_error=('normalized_additivity_error', lambda x: np.percentile(x, 75))).reset_index()
    else:
        add_summary = pd.DataFrame()
    add_summary.to_csv(RESULT_DIR / 'TEST_A_ADDITIVITY_SUMMARY.csv', index=False, encoding='utf-8-sig')
    print('\n' + '=' * 100)
    print('TEST A — CONSTITUENT EVIDENCE PRESERVATION')
    print('=' * 100)
    if len(pres_summary) > 0:
        print(pres_summary.to_string(index=False))
    else:
        print('No preservation rows produced.')
    print('\n' + '=' * 100)
    print('TEST A — APPROXIMATE ADDITIVITY')
    print('=' * 100)
    if len(add_summary) > 0:
        print(add_summary.to_string(index=False))
    else:
        print('No additivity rows produced.')
    return (preservation, pres_summary, additivity, add_summary)

def save_final_summary(runs, feature_df, mapping_table, auc_multivariate, pres_summary, add_summary):
    qc = pd.read_csv(RESULT_DIR / 'feature_extraction_qc.csv')
    summary = {'n_runs': int(len(runs)), 'n_fault_classes': int(runs['fault_raw'].nunique()), 'n_conditions': int(runs['condition_key'].nunique()), 'n_health_runs': int(np.sum(runs['fault_role'] == 'health')), 'n_single_runs': int(np.sum(runs['fault_role'] == 'single')), 'n_compound_runs': int(np.sum(runs['fault_role'] == 'compound')), 'n_compound_labels': int(np.sum(mapping_table['status'] == 'RESOLVED')), 'n_unresolved_compound_labels': int(np.sum(mapping_table['status'] != 'RESOLVED')), 'n_physics_features': int(len(get_numeric_feature_columns(feature_df))), 'feature_extraction': {'median_valid_windows_per_run': float(qc['windows_valid'].median()), 'min_valid_windows_per_run': int(qc['windows_valid'].min()), 'runs_with_fewer_than_3_valid_windows': int(np.sum(qc['windows_valid'] < 3))}}
    if len(auc_multivariate) > 0:
        for probe in auc_multivariate['probe'].unique():
            vals = auc_multivariate.loc[auc_multivariate['probe'] == probe, 'condition_heldout_auc'].dropna()
            summary[f'test_B_{probe}'] = {'median_auc': float(vals.median()) if len(vals) else np.nan, 'mean_auc': float(vals.mean()) if len(vals) else np.nan, 'min_auc': float(vals.min()) if len(vals) else np.nan, 'max_auc': float(vals.max()) if len(vals) else np.nan}
    if len(pres_summary) > 0:
        summary['test_A_preservation'] = {'median_constituent_preservation_ratio': float(pres_summary['median_preservation_ratio'].median()), 'mean_positive_evidence_rate': float(pres_summary['positive_evidence_rate'].mean()), 'mean_half_evidence_preserved_rate': float(pres_summary['half_evidence_preserved_rate'].mean())}
    if len(add_summary) > 0:
        summary['test_A_additivity'] = {'median_of_compound_median_normalized_error': float(add_summary['median_normalized_additivity_error'].median()), 'mean_of_compound_median_normalized_error': float(add_summary['median_normalized_additivity_error'].mean())}
    with open(RESULT_DIR / 'FINAL_DIAGNOSTIC_SUMMARY.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    return summary

def main():
    print('\n' + '=' * 100)
    print('6.4-kHz PHYSICS-GUIDED FEATURE EXTRACTION')
    print('=' * 100)
    print(f'Native sampling rate : {NATIVE_FS} Hz')
    print(f'Target sampling rate : {FS} Hz')
    print(f'Shadow raw root      : {SHADOW_RAW_ROOT}')
    print(f'Output               : {RESULT_DIR}')
    runs, windows = load_complete_inventory()
    validate_inputs(runs)
    feature_df = extract_all_run_features(runs, windows)
    usable = get_numeric_feature_columns(feature_df)
    config = {'analysis': 'sampling_rate_sensitivity_feature_extraction', 'native_sampling_rate_hz': NATIVE_FS, 'target_sampling_rate_hz': FS, 'window_seconds': WINDOW_SECONDS, 'native_window_samples': NATIVE_WINDOW_SAMPLES, 'target_window_samples': WINDOW_SAMPLES, 'window_stride_mapping': 'native manifest indices scaled by 0.5', 'samples_per_revolution_after_angle_resampling': SAMPLES_PER_REV, 'max_windows_per_run': MAX_WINDOWS_PER_RUN, 'order_band_half_width': ORDER_BAND_HALF_WIDTH, 'n_runs': int(len(feature_df)), 'n_numeric_features_available': int(len(usable)), 'native_signal_cache_used': False, 'shadow_raw_root': str(SHADOW_RAW_ROOT), 'output_feature_csv': str(RESULT_DIR / 'RUN_PHYSICS_FEATURES.csv'), 'note': 'The locked 12.8-kHz constituent feature map is applied later by the 6.4-kHz crossed evaluator; no 6.4-kHz feature reselection is used.'}
    with open(RESULT_DIR / 'sampling_feature_extraction_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print('\n' + '=' * 100)
    print('FEATURE EXTRACTION COMPLETED')
    print('=' * 100)
    print(f'Runs                    : {len(feature_df)}')
    print(f'Numeric feature columns : {len(usable)}')
    print(f"Output                   : {RESULT_DIR / 'RUN_PHYSICS_FEATURES.csv'}")
    print('\nNEXT: run FINAL_CROSSED_COMPOSITION_CONDITION_EVALUATION_6P4KHZ.py')
if __name__ == '__main__':
    main()
