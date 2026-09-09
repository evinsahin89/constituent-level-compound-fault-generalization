"""Dataset inventory and manifest preparation for the MCC5-THU motor dataset.

This script inventories the released recordings, encodes the nine constituent
fault mechanisms, performs run-level integrity checks, and creates run/window
manifests used by the downstream analyses. Splitting is performed at the
physical-run level before window generation so that windows from the same
recording cannot enter different partitions.

The legacy P1-P4 partitions are retained only to reproduce the manifests
consumed by later analyses; the primary crossed constituent-level evaluation
is implemented separately.
"""
from pathlib import Path
import re
import json
import hashlib
import random
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
ROOT = Path('D:\\Multi-mode Fault Diagnosis Datasets of Three-phase')
SPEED_DIR = ROOT / 'MCC5-THU Motor_speed_circulation'
TORQUE_DIR = ROOT / 'MCC5-THU Motor_torque_circulation'
OUTPUT_DIR = ROOT / 'EXPERIMENT_PROTOCOL'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FS = 12800
EXPECTED_DURATION_S = 90.0
EXPECTED_SAMPLES = int(FS * EXPECTED_DURATION_S)
N_CHANNELS = 9
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(FS * WINDOW_SECONDS)
WINDOW_OVERLAP = 0.5
WINDOW_STRIDE = int(WINDOW_SIZE * (1.0 - WINDOW_OVERLAP))
START_SECONDS = 0.0
END_SECONDS = 90.0
START_SAMPLE = int(START_SECONDS * FS)
END_SAMPLE = int(END_SECONDS * FS)
VALIDATION_RATIO = 0.15
P1_TEST_RATIO = 0.2
EXPECTED_MODES = ['speed_circulation', 'torque_circulation']
EXPECTED_TORQUES = [20.0, 40.0]
EXPECTED_SPEEDS = [1000.0, 2000.0, 3000.0]
HELD_OUT_CONDITION = {'mode': 'speed_circulation', 'torque_Nm': 40.0, 'speed_rpm': 3000.0}
HELD_OUT_COMPOSITIONS = ['winding_H_and_bearing_inner_H', 'winding_H_and_bearing_outer_H', 'dynamic_eccentricity_and_bearing_inner_H', 'dynamic_eccentricity_and_bearing_outer_H']
FAULT_COMPONENTS = ['bearing_ball', 'bearing_inner', 'bearing_outer', 'bend', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'voltage_unbalance', 'winding']
SIGNAL_COLUMNS = ['time', 'keyphase', 'torque', 'vibration_horizontal', 'vibration_axial', 'vibration_vertical', 'current_A', 'current_B', 'current_C']

def stable_hash(text):
    value = hashlib.sha256(str(text).encode('utf-8')).hexdigest()
    return int(value[:16], 16)

def parse_filename(path):
    path = Path(path)
    name = path.stem
    result = {'filename': path.name, 'filepath': str(path), 'folder': path.parent.name, 'fault_raw': None, 'mode': None, 'torque_Nm': np.nan, 'speed_rpm': np.nan, 'severity': None, 'is_compound': False, 'timestamp': None}
    if '_speed_circulation_' in name:
        result['mode'] = 'speed_circulation'
        fault_part = name.split('_speed_circulation_')[0]
    elif '_torque_circulation_' in name:
        result['mode'] = 'torque_circulation'
        fault_part = name.split('_torque_circulation_')[0]
    else:
        fault_part = name
    result['fault_raw'] = fault_part
    result['is_compound'] = '_and_' in fault_part.lower()
    severities = re.findall('(?:^|_)(L|H)(?:_|$)', fault_part, flags=re.IGNORECASE)
    if severities:
        result['severity'] = '+'.join((s.upper() for s in severities))
    match = re.search('(\\d+(?:\\.\\d+)?)Nm', name, flags=re.IGNORECASE)
    if match:
        result['torque_Nm'] = float(match.group(1))
    match = re.search('(\\d+(?:\\.\\d+)?)rpm', name, flags=re.IGNORECASE)
    if match:
        result['speed_rpm'] = float(match.group(1))
    match = re.search('_(\\d{12})[a-zA-Z]?$', name)
    if match:
        result['timestamp'] = match.group(1)
    return result

def build_inventory():
    files = []
    if SPEED_DIR.exists():
        files.extend(SPEED_DIR.rglob('*.csv'))
    if TORQUE_DIR.exists():
        files.extend(TORQUE_DIR.rglob('*.csv'))
    rows = []
    for path in sorted(files):
        info = parse_filename(path)
        stat = path.stat()
        info['size_bytes'] = stat.st_size
        info['size_MB'] = stat.st_size / 1024 ** 2
        rows.append(info)
    df = pd.DataFrame(rows)
    df['run_id'] = [f'RUN_{i:04d}' for i in range(1, len(df) + 1)]
    return df

def encode_fault_components(fault_raw):
    """Encode a released fault label into the nine constituent mechanisms.

The original dataset label is preserved. Healthy runs map to all-zero
constituent targets; single faults map to one active mechanism; compound labels
map to the corresponding multi-label constituent vector.
"""
    fault = str(fault_raw).lower()
    labels = {component: 0 for component in FAULT_COMPONENTS}
    if fault == 'health':
        return labels
    if 'bearing_ball' in fault:
        labels['bearing_ball'] = 1
    if 'bearing_inner' in fault or '_and_inner_' in fault:
        labels['bearing_inner'] = 1
    if 'bearing_outer' in fault or '_and_outer_' in fault:
        labels['bearing_outer'] = 1
    if fault == 'bend' or fault.startswith('bend_') or '_and_bend' in fault:
        labels['bend'] = 1
    if 'broken_bar' in fault:
        labels['broken_bar'] = 1
    if 'dynamic_eccentricity' in fault:
        labels['dynamic_eccentricity'] = 1
    if 'static_eccentricity' in fault:
        labels['static_eccentricity'] = 1
    if 'voltage_unbalance' in fault:
        labels['voltage_unbalance'] = 1
    if 'winding' in fault:
        labels['winding'] = 1
    return labels

def add_multilabel_columns(inventory):
    encoded_rows = []
    for _, row in inventory.iterrows():
        encoded = encode_fault_components(row['fault_raw'])
        encoded_rows.append(encoded)
    label_df = pd.DataFrame(encoded_rows)
    result = pd.concat([inventory.reset_index(drop=True), label_df.reset_index(drop=True)], axis=1)
    result['num_fault_components'] = result[FAULT_COMPONENTS].sum(axis=1)
    return result

def validate_fault_encoding(inventory):
    print('\n' + '=' * 100)
    print('FAULT ONTOLOGY VALIDATION')
    print('=' * 100)
    errors = []
    health = inventory[inventory['fault_raw'] == 'health']
    if not health.empty:
        bad_health = health[health[FAULT_COMPONENTS].sum(axis=1) != 0]
        if not bad_health.empty:
            errors.append('Healthy runs contain active fault components.')
    singles = inventory[~inventory['is_compound'] & (inventory['fault_raw'] != 'health')]
    bad_single = singles[singles['num_fault_components'] != 1]
    if not bad_single.empty:
        print('\nINVALID SINGLE-FAULT ENCODING:')
        print(bad_single[['fault_raw', 'num_fault_components']].drop_duplicates().to_string(index=False))
        errors.append('Some single-fault labels do not map to exactly one constituent.')
    compounds = inventory[inventory['is_compound']]
    bad_compound = compounds[compounds['num_fault_components'] < 2]
    if not bad_compound.empty:
        print('\nINVALID COMPOUND ENCODING:')
        print(bad_compound[['fault_raw', 'num_fault_components']].drop_duplicates().to_string(index=False))
        errors.append("Some compound-fault labels map to fewer than two constituents.")
    dual_bearing = inventory[inventory['fault_raw'] == 'bearing_outer_H_and_inner_H']
    if not dual_bearing.empty:
        correct = ((dual_bearing['bearing_outer'] == 1) & (dual_bearing['bearing_inner'] == 1)).all()
        if not correct:
            errors.append('bearing_outer_H_and_inner_H was not encoded as outer+inner.')
    if errors:
        print('\nONTOLOGY VALIDATION FAILED')
        for error in errors:
            print('-', error)
        raise RuntimeError('Fault ontology encoding failed.')
    print('\nPASS: fault ontology encoding validated.')

def add_condition_id(inventory):
    inventory = inventory.copy()
    inventory['condition_id'] = inventory['mode'].astype(str) + '__T' + inventory['torque_Nm'].astype(int).astype(str) + '__R' + inventory['speed_rpm'].astype(int).astype(str)
    return inventory

def is_held_out_condition(row):
    return row['mode'] == HELD_OUT_CONDITION['mode'] and float(row['torque_Nm']) == float(HELD_OUT_CONDITION['torque_Nm']) and (float(row['speed_rpm']) == float(HELD_OUT_CONDITION['speed_rpm']))

def is_held_out_composition(row):
    return row['fault_raw'] in HELD_OUT_COMPOSITIONS

def add_generalization_groups(inventory):
    df = inventory.copy()
    df['is_unseen_condition'] = df.apply(is_held_out_condition, axis=1)
    df['is_unseen_composition'] = df.apply(is_held_out_composition, axis=1)

    def assign_group(row):
        unseen_condition = row['is_unseen_condition']
        unseen_composition = row['is_unseen_composition']
        if not unseen_condition and (not unseen_composition):
            return 'P1_POOL'
        if unseen_condition and (not unseen_composition):
            return 'P2_TEST'
        if not unseen_condition and unseen_composition:
            return 'P3_TEST'
        return 'P4_TEST'
    df['generalization_group'] = df.apply(assign_group, axis=1)
    return df

def select_p1_test_runs(p1_pool):
    """Select legacy P1 test runs while retaining the same fault state and operating condition in the remaining training pool."""
    df = p1_pool.copy()
    candidates = list(df.index)
    rng = np.random.default_rng(SEED)
    rng.shuffle(candidates)
    target_n = max(1, int(round(len(df) * P1_TEST_RATIO)))
    selected = []
    for idx in candidates:
        if len(selected) >= target_n:
            break
        candidate = df.loc[idx]
        remaining = df.drop(index=selected + [idx])
        same_fault_remaining = (remaining['fault_raw'] == candidate['fault_raw']).any()
        same_condition_remaining = (remaining['condition_id'] == candidate['condition_id']).any()
        if same_fault_remaining and same_condition_remaining:
            selected.append(idx)
    return selected

def split_train_validation(train_pool):
    """Create a deterministic run-level train/validation split.

A hash-based split is used because many fault-by-condition cells contain only
one physical run. Positive support for every constituent is retained in the
training partition.
"""
    df = train_pool.copy()
    scores = []
    for _, row in df.iterrows():
        key = str(row['run_id']) + '__' + str(SEED)
        score = stable_hash(key) % 1000000 / 1000000.0
        scores.append(score)
    df['_split_score'] = scores
    validation_mask = df['_split_score'] < VALIDATION_RATIO
    validation = df[validation_mask].copy()
    train = df[~validation_mask].copy()
    for component in FAULT_COMPONENTS:
        positive_count = int(train[component].sum())
        if positive_count == 0:
            candidates = validation[validation[component] == 1]
            if not candidates.empty:
                move_idx = candidates.index[0]
                train = pd.concat([train, validation.loc[[move_idx]]])
                validation = validation.drop(index=move_idx)
    train = train.drop(columns=['_split_score'], errors='ignore')
    validation = validation.drop(columns=['_split_score'], errors='ignore')
    return (train, validation)

def build_split(inventory):
    df = inventory.copy()
    p2_test = df[df['generalization_group'] == 'P2_TEST'].copy()
    p3_test = df[df['generalization_group'] == 'P3_TEST'].copy()
    p4_test = df[df['generalization_group'] == 'P4_TEST'].copy()
    p1_pool = df[df['generalization_group'] == 'P1_POOL'].copy()
    p1_indices = select_p1_test_runs(p1_pool)
    p1_test = p1_pool.loc[p1_indices].copy()
    train_pool = p1_pool.drop(index=p1_indices).copy()
    train, validation = split_train_validation(train_pool)
    train['split'] = 'train'
    train['protocol'] = 'TRAIN'
    validation['split'] = 'validation'
    validation['protocol'] = 'VALIDATION'
    p1_test['split'] = 'test'
    p1_test['protocol'] = 'P1'
    p2_test['split'] = 'test'
    p2_test['protocol'] = 'P2'
    p3_test['split'] = 'test'
    p3_test['protocol'] = 'P3'
    p4_test['split'] = 'test'
    p4_test['protocol'] = 'P4'
    return {'train': train, 'validation': validation, 'P1': p1_test, 'P2': p2_test, 'P3': p3_test, 'P4': p4_test}

def leakage_check(splits):
    print('\n' + '=' * 100)
    print('RUN-LEVEL LEAKAGE CHECK')
    print('=' * 100)
    names = list(splits.keys())
    passed = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name = names[i]
            b_name = names[j]
            a = set(splits[a_name]['run_id'])
            b = set(splits[b_name]['run_id'])
            overlap = a.intersection(b)
            print(f'{a_name:10s} vs {b_name:10s}: {len(overlap)} overlap')
            if overlap:
                passed = False
    if not passed:
        raise RuntimeError('RUN-LEVEL DATA LEAKAGE DETECTED.')
    print('\nPASS: no run-level leakage detected.')

def train_coverage_check(train):
    print('\n' + '=' * 100)
    print('TRAIN MULTI-LABEL COVERAGE')
    print('=' * 100)
    for component in FAULT_COMPONENTS:
        count = int(train[component].sum())
        print(f'{component:25s}: {count}')
        if count == 0:
            raise RuntimeError(f'Training data contain no positive examples for {component}.')

def unseen_composition_check(splits):
    train_faults = set(splits['train']['fault_raw'])
    print('\n' + '=' * 100)
    print('UNSEEN COMPOSITION CHECK')
    print('=' * 100)
    for protocol in ['P3', 'P4']:
        test_faults = sorted(set(splits[protocol]['fault_raw']))
        print(f'\n{protocol}:')
        for fault in test_faults:
            status = 'SEEN' if fault in train_faults else 'UNSEEN'
            print(f'  {fault:55s} {status}')
            if fault in train_faults:
                raise RuntimeError(f'{protocol}: {fault} is present in training.')

def constituent_visibility_check(splits):
    """Verify that held-out compound identities are absent from training while their constituent mechanisms remain represented."""
    print('\n' + '=' * 100)
    print('CONSTITUENT FAULT VISIBILITY CHECK')
    print('=' * 100)
    train = splits['train']
    for protocol in ['P3', 'P4']:
        print(f'\n{protocol}:')
        df = splits[protocol]
        fault_rows = df[['fault_raw'] + FAULT_COMPONENTS].drop_duplicates(subset=['fault_raw'])
        for _, row in fault_rows.iterrows():
            active_components = [component for component in FAULT_COMPONENTS if int(row[component]) == 1]
            print(f"\n  {row['fault_raw']}")
            for component in active_components:
                count = int(train[component].sum())
                status = 'SEEN' if count > 0 else 'MISSING'
                print(f'      {component:25s} {status} (train positives={count})')
                if count == 0:
                    raise RuntimeError(f"{protocol}: {row['fault_raw']} contains {component} with no training support.")
    print("\nPASS: held-out compound constituents remain represented in training.")

def unseen_condition_check(splits):
    train_conditions = set(splits['train']['condition_id'])
    print('\n' + '=' * 100)
    print('UNSEEN CONDITION CHECK')
    print('=' * 100)
    for protocol in ['P2', 'P4']:
        conditions = sorted(set(splits[protocol]['condition_id']))
        print(f'\n{protocol}:')
        for condition in conditions:
            status = 'SEEN' if condition in train_conditions else 'UNSEEN'
            print(f'  {condition:45s} {status}')
            if condition in train_conditions:
                raise RuntimeError(f'{protocol}: {condition} is present in training.')

def p1_validity_check(splits):
    print('\n' + '=' * 100)
    print('P1 SEEN CONDITION / SEEN COMPOSITION CHECK')
    print('=' * 100)
    train = splits['train']
    p1 = splits['P1']
    train_faults = set(train['fault_raw'])
    train_conditions = set(train['condition_id'])
    unseen_faults = sorted(set(p1['fault_raw']) - train_faults)
    unseen_conditions = sorted(set(p1['condition_id']) - train_conditions)
    print(f'\nP1 fault composition not in train: {len(unseen_faults)}')
    if unseen_faults:
        print(unseen_faults)
    print(f'P1 conditions not in train: {len(unseen_conditions)}')
    if unseen_conditions:
        print(unseen_conditions)
    if unseen_faults or unseen_conditions:
        raise RuntimeError('P1 seen-condition/seen-composition requirement is not satisfied.')
    print('\nPASS: P1 composition and condition are represented in training.')

def create_window_manifest(runs):
    rows = []
    for _, run in runs.iterrows():
        start = START_SAMPLE
        window_index = 0
        while start + WINDOW_SIZE <= END_SAMPLE:
            end = start + WINDOW_SIZE
            window_id = run['run_id'] + f'_W{window_index:04d}'
            record = {'window_id': window_id, 'run_id': run['run_id'], 'filename': run['filename'], 'filepath': run['filepath'], 'fault_raw': run['fault_raw'], 'is_compound': run['is_compound'], 'severity': run['severity'], 'mode': run['mode'], 'torque_Nm': run['torque_Nm'], 'speed_rpm': run['speed_rpm'], 'condition_id': run['condition_id'], 'split': run['split'], 'protocol': run['protocol'], 'window_index': window_index, 'start_sample': start, 'end_sample': end, 'start_time_s': start / FS, 'end_time_s': end / FS, 'window_samples': WINDOW_SIZE, 'window_seconds': WINDOW_SECONDS}
            for component in FAULT_COMPONENTS:
                record[component] = int(run[component])
            rows.append(record)
            start += WINDOW_STRIDE
            window_index += 1
    return pd.DataFrame(rows)

def window_leakage_check(manifests):
    print('\n' + '=' * 100)
    print('WINDOW-LEVEL LEAKAGE CHECK')
    print('=' * 100)
    run_location = {}
    for split_name, df in manifests.items():
        for run_id in df['run_id'].unique():
            if run_id in run_location:
                raise RuntimeError(f'{run_id} appears in multiple window splits: {run_location[run_id]} and {split_name}')
            run_location[run_id] = split_name
    print('PASS: all windows from each run remain within one split.')

def print_split_summary(splits):
    print('\n' + '=' * 100)
    print('RUN SPLIT SUMMARY')
    print('=' * 100)
    total = 0
    for name, df in splits.items():
        total += len(df)
        print(f'{name:12s}: {len(df):4d} runs')
    print(f'\nTOTAL: {total} runs')
    print('\nTest protocol details:')
    for protocol in ['P1', 'P2', 'P3', 'P4']:
        df = splits[protocol]
        print(f'\n{protocol}')
        print(f'Runs: {len(df)}')
        if not df.empty:
            print('\nFault compositions:')
            print(df['fault_raw'].value_counts().to_string())
            print('\nConditions:')
            print(df['condition_id'].value_counts().to_string())

def print_window_summary(manifests):
    print('\n' + '=' * 100)
    print('WINDOW SUMMARY')
    print('=' * 100)
    total = 0
    for name, df in manifests.items():
        count = len(df)
        total += count
        print(f'{name:12s}: {count:7d} windows')
    print(f'\nTOTAL WINDOWS: {total}')

def save_run_splits(splits):
    for name, df in splits.items():
        path = OUTPUT_DIR / f'runs_{name}.csv'
        df.to_csv(path, index=False, encoding='utf-8-sig')

def save_window_manifests(manifests):
    for name, df in manifests.items():
        path = OUTPUT_DIR / f'windows_{name}.csv'
        df.to_csv(path, index=False, encoding='utf-8-sig')

def save_combined_runs(splits):
    combined = pd.concat(list(splits.values()), ignore_index=True)
    combined.to_csv(OUTPUT_DIR / 'ALL_RUN_SPLITS.csv', index=False, encoding='utf-8-sig')
    return combined

def save_combined_windows(manifests):
    combined = pd.concat(list(manifests.values()), ignore_index=True)
    combined.to_csv(OUTPUT_DIR / 'ALL_WINDOW_MANIFEST.csv', index=False, encoding='utf-8-sig')
    return combined

def save_config():
    config = {'seed': SEED, 'sampling_frequency_hz': FS, 'expected_duration_s': EXPECTED_DURATION_S, 'expected_samples': EXPECTED_SAMPLES, 'window_seconds': WINDOW_SECONDS, 'window_samples': WINDOW_SIZE, 'window_overlap': WINDOW_OVERLAP, 'window_stride_samples': WINDOW_STRIDE, 'validation_ratio': VALIDATION_RATIO, 'p1_test_ratio': P1_TEST_RATIO, 'held_out_condition': HELD_OUT_CONDITION, 'held_out_compositions': HELD_OUT_COMPOSITIONS, 'fault_components': FAULT_COMPONENTS, 'protocols': {'P1': 'seen condition + seen composition', 'P2': 'unseen condition + seen composition', 'P3': 'seen condition + unseen composition', 'P4': 'unseen condition + unseen composition'}, 'important_rules': ['Dataset original labels are preserved.', 'Run-level split is performed before windowing.', 'Windows from the same run cannot enter different splits.', 'Compound faults are represented using constituent multi-label targets.']}
    with open(OUTPUT_DIR / 'experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def print_fault_encoding(inventory):
    columns = ['fault_raw', 'is_compound', 'num_fault_components'] + FAULT_COMPONENTS
    encoding = inventory[columns].drop_duplicates().sort_values('fault_raw')
    print('\n' + '=' * 100)
    print('FAULT -> MULTI-LABEL ENCODING')
    print('=' * 100)
    print(encoding.to_string(index=False))
    encoding.to_csv(OUTPUT_DIR / 'FAULT_COMPONENT_ENCODING.csv', index=False, encoding='utf-8-sig')

def final_assertions(inventory, combined_runs, combined_windows):
    print('\n' + '=' * 100)
    print('FINAL ASSERTIONS')
    print('=' * 100)
    assert len(combined_runs) == len(inventory), 'Run count mismatch.'
    assert combined_runs['run_id'].is_unique, 'Duplicate run_id.'
    assert set(combined_windows['run_id']).issubset(set(combined_runs['run_id'])), 'Window manifest contains an unknown run_id.'
    assert combined_runs['split'].notna().all(), 'At least one run has no split assignment.'
    assert combined_windows[FAULT_COMPONENTS].notna().all().all(), 'At least one window is missing a target label.'
    print('PASS: every run is assigned to exactly one split.')
    print('PASS: every window references a valid run_id.')
    print('PASS: every window has a multi-label target.')

def main():
    print('\n' + '=' * 100)
    print('MCC5-THU LEAKAGE-FREE EXPERIMENT PROTOCOL')
    print('=' * 100)
    print('\nScanning dataset...')
    inventory = build_inventory()
    print(f'Total runs: {len(inventory)}')
    if len(inventory) != 288:
        print('\nWARNING: expected 288 released recordings.')
    inventory = add_multilabel_columns(inventory)
    validate_fault_encoding(inventory)
    inventory = add_condition_id(inventory)
    inventory = add_generalization_groups(inventory)
    inventory.to_csv(OUTPUT_DIR / 'DATASET_RUN_INVENTORY.csv', index=False, encoding='utf-8-sig')
    print_fault_encoding(inventory)
    splits = build_split(inventory)
    print_split_summary(splits)
    leakage_check(splits)
    train_coverage_check(splits['train'])
    p1_validity_check(splits)
    unseen_composition_check(splits)
    constituent_visibility_check(splits)
    unseen_condition_check(splits)
    save_run_splits(splits)
    combined_runs = save_combined_runs(splits)
    print('\nCreating window manifests...')
    manifests = {}
    for name, run_df in splits.items():
        manifests[name] = create_window_manifest(run_df)
    window_leakage_check(manifests)
    print_window_summary(manifests)
    save_window_manifests(manifests)
    combined_windows = save_combined_windows(manifests)
    save_config()
    final_assertions(inventory, combined_runs, combined_windows)
    print('\n' + '=' * 100)
    print('EXPERIMENT PROTOCOL COMPLETED')
    print('=' * 100)
    print(f'\nOutput folder:\n{OUTPUT_DIR}')
    print('\nKey output files:')
    print('1. DATASET_RUN_INVENTORY.csv')
    print('2. FAULT_COMPONENT_ENCODING.csv')
    print('3. ALL_RUN_SPLITS.csv')
    print('4. runs_train.csv')
    print('5. runs_validation.csv')
    print('6. runs_P1.csv')
    print('7. runs_P2.csv')
    print('8. runs_P3.csv')
    print('9. runs_P4.csv')
    print('10. ALL_WINDOW_MANIFEST.csv')
    print('11. windows_train.csv')
    print('12. windows_validation.csv')
    print('13. windows_P1.csv')
    print('14. windows_P2.csv')
    print('15. windows_P3.csv')
    print('16. windows_P4.csv')
    print('17. experiment_config.json')
if __name__ == '__main__':
    main()
