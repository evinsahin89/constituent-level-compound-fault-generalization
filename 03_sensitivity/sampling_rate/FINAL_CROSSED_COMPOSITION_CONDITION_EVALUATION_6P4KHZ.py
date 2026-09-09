"""Crossed constituent-level evaluation at 6.4 kHz.

The locked crossed composition-context-condition evaluation is rerun using the
independently extracted 6.4-kHz physics features. Split logic, logistic
regression, training-only preprocessing, grouped inner threshold calibration,
threshold grid, and evaluation metrics match the native analysis.

The exact mechanism-specific feature map locked at 12.8 kHz is enforced without
reselection.
"""
from pathlib import Path
from collections import defaultdict
import json
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
ROOT = Path('D:\\yay\u0131nlar\\Multi-mode Fault Diagnosis Datasets of Three-phase')
PROTOCOL_DIR = ROOT / 'EXPERIMENT_PROTOCOL'
PHYSICS_DIR = ROOT / 'SAMPLING_RATE_SENSITIVITY' / 'PHYSICS_6P4KHZ'
RESULT_DIR = ROOT / 'SAMPLING_RATE_SENSITIVITY' / 'FINAL_CROSSED_6P4KHZ'
RESULT_DIR.mkdir(parents=True, exist_ok=True)
PHYSICS_PATH = PHYSICS_DIR / 'RUN_PHYSICS_FEATURES.csv'
LOCKED_FEATURE_MAP_PATH = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS' / 'LOCKED_FEATURE_MAP.csv'
SAMPLING_RATE_HZ = 6400
NATIVE_SAMPLING_RATE_HZ = 12800
OLD_SPLITS = ['train', 'validation', 'P1', 'P2', 'P3', 'P4']
COMPONENTS = ['bearing_ball', 'bearing_inner', 'bearing_outer', 'bend', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'voltage_unbalance', 'winding']
LOGISTIC_C = 1.0
SEED = 42
INNER_GROUP_FOLDS = 4
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 42
CONTEXT_REGIMES = ['RECOMBINATION', 'ONE_SIDED_CONTEXT_ZS', 'TWO_SIDED_CONTEXT_ZS']
CONDITION_MODES = ['SEEN_CONDITION', 'UNSEEN_CONDITION']

def infer_fault_role(fault_raw):
    low = str(fault_raw).strip().lower()
    if low in {'health', 'healthy', 'normal', 'nc', 'normal_condition'}:
        return 'health'
    if '_and_' in low:
        return 'compound'
    return 'single'

def active_components_from_row(row):
    return tuple((component for component in COMPONENTS if int(row[component]) == 1))

def mechanism_composition_key_from_row(row):
    """
    Mechanism-level composition identity.
    Severity text in fault_raw is intentionally ignored.
    """
    active = active_components_from_row(row)
    if len(active) < 2:
        return ''
    return '+'.join(sorted(active))

def load_data():
    print('\n' + '=' * 130)
    print('LOAD FULL DATASET')
    print('=' * 130)
    if not PHYSICS_PATH.exists():
        raise FileNotFoundError(f'RUN_PHYSICS_FEATURES.csv not found:\n{PHYSICS_PATH}')
    if not LOCKED_FEATURE_MAP_PATH.exists():
        raise FileNotFoundError(f'Native 12.8-kHz LOCKED_FEATURE_MAP.csv not found:\n{LOCKED_FEATURE_MAP_PATH}')
    manifest_frames = []
    for split in OLD_SPLITS:
        path = PROTOCOL_DIR / f'runs_{split}.csv'
        if not path.exists():
            raise FileNotFoundError(f'Manifest not found:\n{path}')
        df = pd.read_csv(path)
        required = {'run_id', 'fault_raw', *COMPONENTS}
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f'{path.name}: missing columns: {missing}')
        df = df.copy()
        df['run_id'] = df['run_id'].astype(str)
        df['fault_raw'] = df['fault_raw'].astype(str)
        df['old_split'] = split
        manifest_frames.append(df)
    runs = pd.concat(manifest_frames, ignore_index=True)
    if runs['run_id'].duplicated().any():
        duplicate = runs.loc[runs['run_id'].duplicated(keep=False), ['run_id', 'old_split', 'fault_raw']]
        raise RuntimeError('Duplicate run_id found in old manifests:\n' + duplicate.head(20).to_string(index=False))
    physics = pd.read_csv(PHYSICS_PATH)
    physics['run_id'] = physics['run_id'].astype(str)
    if physics['run_id'].duplicated().any():
        raise RuntimeError('RUN_PHYSICS_FEATURES.csv contains duplicate run_id.')
    if 'condition_key' not in physics.columns:
        raise RuntimeError('RUN_PHYSICS_FEATURES.csv does not contain condition_key.')
    physics['condition_key'] = physics['condition_key'].astype(str)
    physics_for_merge = physics.drop(columns=['fault_raw', 'fault_role'], errors='ignore')
    data = runs.merge(physics_for_merge, on='run_id', how='left', validate='one_to_one')
    if data['condition_key'].isna().any():
        missing_runs = data.loc[data['condition_key'].isna(), 'run_id'].tolist()
        raise RuntimeError(f'Physics feature coverage missing for run IDs:\n{missing_runs[:20]}')
    data['fault_role'] = data['fault_raw'].apply(infer_fault_role)
    data['composition_key'] = data.apply(mechanism_composition_key_from_row, axis=1)
    compound = data[data['fault_role'] == 'compound'].copy()
    if len(compound) == 0:
        raise RuntimeError('No compound runs found.')
    compound_cardinality = compound[COMPONENTS].astype(int).sum(axis=1)
    if (compound_cardinality < 2).any():
        raise RuntimeError('A compound run has fewer than two active mechanisms.')
    definition = pd.read_csv(LOCKED_FEATURE_MAP_PATH)
    required_definition = {'component', 'feature'}
    missing = sorted(required_definition - set(definition.columns))
    if missing:
        raise RuntimeError(f'Native LOCKED_FEATURE_MAP.csv missing columns: {missing}')
    definition = definition.rename(columns={'component': 'mechanism_key'})[['mechanism_key', 'feature']].copy()
    if definition.duplicated(['mechanism_key', 'feature']).any():
        raise RuntimeError('Native LOCKED_FEATURE_MAP.csv contains duplicates.')
    print(f'Total runs        : {len(data)}')
    print(f"Health            : {(data['fault_role'] == 'health').sum()}")
    print(f"Single            : {(data['fault_role'] == 'single').sum()}")
    print(f"Compound          : {(data['fault_role'] == 'compound').sum()}")
    print(f"Conditions        : {data['condition_key'].nunique()}")
    print(f"Compound mechanisms compositions: {compound['composition_key'].nunique()}")
    return (data.reset_index(drop=True), definition)

def build_feature_map_and_raw_matrices(data, definition):
    print('\n' + '=' * 130)
    print('STRICT NATIVE 12.8-kHz LOCKED FEATURE MAP APPLIED AT 6.4 kHz')
    print('=' * 130)
    feature_map = {}
    raw_matrices = {}
    feature_rows = []
    expected_dims = {'bearing_ball': 16, 'bearing_inner': 16, 'bearing_outer': 16, 'bend': 16, 'broken_bar': 40, 'dynamic_eccentricity': 56, 'static_eccentricity': 56, 'voltage_unbalance': 2, 'winding': 12}
    for component in COMPONENTS:
        requested = definition.loc[definition['mechanism_key'].astype(str) == component, 'feature'].dropna().astype(str).tolist()
        if not requested:
            raise RuntimeError(f'{component}: no feature exists in native LOCKED_FEATURE_MAP.csv.')
        if len(requested) != expected_dims[component]:
            raise RuntimeError(f'{component}: native locked-map dimension is {len(requested)}, expected {expected_dims[component]}.')
        missing_features = [feature for feature in requested if feature not in data.columns]
        if missing_features:
            raise RuntimeError(f'{component}: 6.4-kHz feature table is missing locked features: {missing_features}')
        non_numeric = [feature for feature in requested if not pd.api.types.is_numeric_dtype(data[feature])]
        if non_numeric:
            raise RuntimeError(f'{component}: locked features are non-numeric at 6.4 kHz: {non_numeric}')
        features = sorted(requested)
        feature_map[component] = features
        raw_matrices[component] = data[features].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
        print(f'{component:24s}: {len(features):3d} locked features')
        for feature in features:
            feature_rows.append({'component': component, 'feature': feature})
    locked_out = pd.DataFrame(feature_rows)
    locked_out.to_csv(RESULT_DIR / 'LOCKED_FEATURE_MAP.csv', index=False, encoding='utf-8-sig')
    native_pairs = set(map(tuple, definition[['mechanism_key', 'feature']].astype(str).to_numpy().tolist()))
    output_pairs = set(((str(r['component']), str(r['feature'])) for r in feature_rows))
    if native_pairs != output_pairs:
        raise RuntimeError('6.4-kHz locked feature map differs from the native 12.8-kHz map.')
    return (feature_map, raw_matrices)

def fit_binary_detector(X_raw, y):
    X_raw = np.asarray(X_raw, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise RuntimeError('Binary training data contains only one class.')
    medians = np.nanmedian(X_raw, axis=0)
    usable = np.isfinite(medians)
    if usable.sum() == 0:
        raise RuntimeError('All detector features are non-finite in this training fold.')
    medians_used = medians[usable]
    X_used = X_raw[:, usable].copy()
    missing = ~np.isfinite(X_used)
    if missing.any():
        row_idx, col_idx = np.where(missing)
        X_used[row_idx, col_idx] = medians_used[col_idx]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_used)
    model = LogisticRegression(penalty='l2', C=LOGISTIC_C, class_weight='balanced', solver='liblinear', max_iter=5000, random_state=SEED)
    model.fit(X_scaled, y)
    return {'usable': usable, 'medians': medians_used, 'scaler': scaler, 'model': model}

def predict_binary_detector(fitted, X_raw):
    X_raw = np.asarray(X_raw, dtype=np.float64)
    X_used = X_raw[:, fitted['usable']].copy()
    missing = ~np.isfinite(X_used)
    if missing.any():
        row_idx, col_idx = np.where(missing)
        X_used[row_idx, col_idx] = fitted['medians'][col_idx]
    X_scaled = fitted['scaler'].transform(X_used)
    return fitted['model'].predict_proba(X_scaled)[:, 1]

def select_threshold_from_oof(y_true, probability):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    finite = np.isfinite(probability)
    y = y_true[finite]
    p = probability[finite]
    if len(y) < 4 or len(np.unique(y)) != 2:
        return {'threshold': 0.5, 'status': 'FALLBACK_0.50_INSUFFICIENT_OOF', 'oof_n': int(len(y)), 'oof_positive': int(y.sum()) if len(y) else 0, 'oof_negative': int(len(y) - y.sum()) if len(y) else 0, 'oof_f1': np.nan, 'oof_balanced_accuracy': np.nan}
    best = None
    for threshold in THRESHOLD_GRID:
        pred = (p >= threshold).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        balanced = balanced_accuracy_score(y, pred)
        distance = abs(float(threshold) - 0.5)
        candidate = {'threshold': float(threshold), 'oof_f1': float(f1), 'oof_balanced_accuracy': float(balanced), 'distance': float(distance)}
        if best is None:
            best = candidate
            continue
        if candidate['oof_f1'] > best['oof_f1'] + 1e-12:
            best = candidate
            continue
        if abs(candidate['oof_f1'] - best['oof_f1']) <= 1e-12 and candidate['oof_balanced_accuracy'] > best['oof_balanced_accuracy'] + 1e-12:
            best = candidate
            continue
        if abs(candidate['oof_f1'] - best['oof_f1']) <= 1e-12 and abs(candidate['oof_balanced_accuracy'] - best['oof_balanced_accuracy']) <= 1e-12 and (candidate['distance'] < best['distance']):
            best = candidate
    return {'threshold': best['threshold'], 'status': 'GROUPED_INNER_OOF', 'oof_n': int(len(y)), 'oof_positive': int(y.sum()), 'oof_negative': int(len(y) - y.sum()), 'oof_f1': best['oof_f1'], 'oof_balanced_accuracy': best['oof_balanced_accuracy']}

def inner_grouped_threshold(X_raw_outer, y_outer, groups_outer):
    """
    Threshold calibration entirely inside outer training data.
    """
    X_raw_outer = np.asarray(X_raw_outer, dtype=np.float64)
    y_outer = np.asarray(y_outer, dtype=int)
    groups_outer = np.asarray(groups_outer)
    if len(np.unique(y_outer)) != 2:
        return {'threshold': 0.5, 'status': 'FALLBACK_0.50_OUTER_TRAIN_SINGLE_CLASS', 'oof_n': 0, 'oof_positive': int(y_outer.sum()), 'oof_negative': int(len(y_outer) - y_outer.sum()), 'oof_f1': np.nan, 'oof_balanced_accuracy': np.nan, 'inner_folds_requested': 0, 'inner_folds_valid': 0}
    unique_groups = np.unique(groups_outer)
    n_splits = min(INNER_GROUP_FOLDS, len(unique_groups))
    if n_splits < 2:
        return {'threshold': 0.5, 'status': 'FALLBACK_0.50_TOO_FEW_CONDITION_GROUPS', 'oof_n': 0, 'oof_positive': int(y_outer.sum()), 'oof_negative': int(len(y_outer) - y_outer.sum()), 'oof_f1': np.nan, 'oof_balanced_accuracy': np.nan, 'inner_folds_requested': n_splits, 'inner_folds_valid': 0}
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y_outer), np.nan, dtype=float)
    valid_folds = 0
    dummy = np.zeros((len(y_outer), 1), dtype=np.float32)
    for train_idx, val_idx in splitter.split(dummy, y_outer, groups_outer):
        y_train = y_outer[train_idx]
        if len(np.unique(y_train)) != 2:
            continue
        fitted = fit_binary_detector(X_raw_outer[train_idx], y_train)
        oof[val_idx] = predict_binary_detector(fitted, X_raw_outer[val_idx])
        valid_folds += 1
    result = select_threshold_from_oof(y_outer, oof)
    result['inner_folds_requested'] = int(n_splits)
    result['inner_folds_valid'] = int(valid_folds)
    return result

def build_composition_catalog(data):
    compound = data[data['fault_role'] == 'compound'].copy()
    rows = []
    for composition_key, group in compound.groupby('composition_key', sort=True):
        first = group.iloc[0]
        active = active_components_from_row(first)
        raw_labels = sorted(group['fault_raw'].astype(str).unique().tolist())
        conditions = sorted(group['condition_key'].astype(str).unique().tolist())
        rows.append({'composition_key': composition_key, 'constituent_count': len(active), 'constituents': '|'.join(active), 'constituent_1': active[0] if len(active) >= 1 else '', 'constituent_2': active[1] if len(active) >= 2 else '', 'n_runs': len(group), 'n_conditions': len(conditions), 'fault_raw_labels': '|'.join(raw_labels), 'conditions': '|'.join(conditions)})
    catalog = pd.DataFrame(rows)
    catalog.to_csv(RESULT_DIR / 'COMPOSITION_CATALOG.csv', index=False, encoding='utf-8-sig')
    print('\n' + '=' * 130)
    print('COMPOSITION CATALOG')
    print('=' * 130)
    print(catalog.to_string(index=False))
    return catalog

def constituent_context_count(train_df, component):
    """
    Number of DISTINCT training compound compositions containing component.
    """
    subset = train_df[(train_df['fault_role'] == 'compound') & (train_df[component].astype(int) == 1)]
    return int(subset['composition_key'].nunique())

def build_context_specs_for_composition(data, composition_key):
    """
    Returns feasible context-regime specifications BEFORE optional condition
    holdout is applied.
    """
    target_rows = data[(data['fault_role'] == 'compound') & (data['composition_key'] == composition_key)]
    if len(target_rows) == 0:
        raise RuntimeError(f'No rows for composition {composition_key}')
    first = target_rows.iloc[0]
    constituents = active_components_from_row(first)
    if len(constituents) != 2:
        raise RuntimeError(f'This final protocol currently expects two-constituent compounds; {composition_key} has {len(constituents)}.')
    a, b = constituents
    specs = []
    exclude_context = ((data['fault_role'] == 'compound') & (data['composition_key'] == composition_key)).to_numpy()
    train_base = data.loc[~exclude_context]
    count_a = constituent_context_count(train_base, a)
    count_b = constituent_context_count(train_base, b)
    if count_a > 0 and count_b > 0:
        specs.append({'context_regime': 'RECOMBINATION', 'context_unseen_component': '', 'context_seen_component': '', 'excluded_context_mask': exclude_context, 'precondition_context_count_1': count_a, 'precondition_context_count_2': count_b})
    exclude_a = ((data['fault_role'] == 'compound') & (data[a].astype(int) == 1)).to_numpy()
    train_a_unseen = data.loc[~exclude_a]
    count_a_after = constituent_context_count(train_a_unseen, a)
    count_b_after = constituent_context_count(train_a_unseen, b)
    if count_a_after == 0 and count_b_after > 0:
        specs.append({'context_regime': 'ONE_SIDED_CONTEXT_ZS', 'context_unseen_component': a, 'context_seen_component': b, 'excluded_context_mask': exclude_a, 'precondition_context_count_1': count_a_after, 'precondition_context_count_2': count_b_after})
    exclude_b = ((data['fault_role'] == 'compound') & (data[b].astype(int) == 1)).to_numpy()
    train_b_unseen = data.loc[~exclude_b]
    count_a_after_b = constituent_context_count(train_b_unseen, a)
    count_b_after_b = constituent_context_count(train_b_unseen, b)
    if count_b_after_b == 0 and count_a_after_b > 0:
        specs.append({'context_regime': 'ONE_SIDED_CONTEXT_ZS', 'context_unseen_component': b, 'context_seen_component': a, 'excluded_context_mask': exclude_b, 'precondition_context_count_1': count_a_after_b, 'precondition_context_count_2': count_b_after_b})
    exclude_both = ((data['fault_role'] == 'compound') & ((data[a].astype(int) == 1) | (data[b].astype(int) == 1))).to_numpy()
    train_both_unseen = data.loc[~exclude_both]
    count_a_two = constituent_context_count(train_both_unseen, a)
    count_b_two = constituent_context_count(train_both_unseen, b)
    if count_a_two == 0 and count_b_two == 0:
        specs.append({'context_regime': 'TWO_SIDED_CONTEXT_ZS', 'context_unseen_component': f'{a}|{b}', 'context_seen_component': '', 'excluded_context_mask': exclude_both, 'precondition_context_count_1': count_a_two, 'precondition_context_count_2': count_b_two})
    return (constituents, specs)

def run_outer_fold(data, raw_matrices, train_mask, test_mask, fold_meta):
    """
    Fit 9 fixed independent detectors on OUTER TRAIN.
    Thresholds are calibrated by inner grouped CV on OUTER TRAIN only.
    """
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    if len(test_indices) == 0:
        raise RuntimeError(f"{fold_meta['fold_id']}: empty outer test.")
    if np.intersect1d(train_indices, test_indices).size > 0:
        raise RuntimeError(f"{fold_meta['fold_id']}: outer train/test overlap.")
    train_df = data.iloc[train_indices]
    test_df = data.iloc[test_indices]
    groups_outer = train_df['condition_key'].astype(str).to_numpy()
    prediction = test_df[['run_id', 'old_split', 'fault_raw', 'fault_role', 'condition_key', 'composition_key', *COMPONENTS]].copy()
    threshold_rows = []
    for component in COMPONENTS:
        X_all = raw_matrices[component]
        X_train = X_all[train_indices]
        X_test = X_all[test_indices]
        y_train = train_df[component].astype(int).to_numpy()
        threshold_info = inner_grouped_threshold(X_train, y_train, groups_outer)
        if len(np.unique(y_train)) != 2:
            probability = np.full(len(test_indices), float(y_train[0]) if len(y_train) else 0.0)
            final_fit_status = 'OUTER_TRAIN_SINGLE_CLASS'
        else:
            fitted = fit_binary_detector(X_train, y_train)
            probability = predict_binary_detector(fitted, X_test)
            final_fit_status = 'OK'
        threshold = float(threshold_info['threshold'])
        pred = (probability >= threshold).astype(int)
        prediction[f'prob_{component}'] = probability
        prediction[f'pred_{component}'] = pred
        prediction[f'threshold_{component}'] = threshold
        threshold_rows.append({**fold_meta, 'component': component, 'n_outer_train': len(train_indices), 'n_train_positive': int(y_train.sum()), 'n_train_negative': int(len(y_train) - y_train.sum()), 'threshold': threshold, 'threshold_status': threshold_info['status'], 'inner_oof_n': threshold_info['oof_n'], 'inner_oof_positive': threshold_info['oof_positive'], 'inner_oof_negative': threshold_info['oof_negative'], 'inner_oof_f1': threshold_info['oof_f1'], 'inner_oof_balanced_accuracy': threshold_info['oof_balanced_accuracy'], 'inner_folds_requested': threshold_info['inner_folds_requested'], 'inner_folds_valid': threshold_info['inner_folds_valid'], 'final_fit_status': final_fit_status})
    for key, value in fold_meta.items():
        prediction[key] = value
    return (prediction, pd.DataFrame(threshold_rows))

def execute_final_protocol(data, catalog, raw_matrices):
    all_predictions = []
    all_thresholds = []
    fold_definition_rows = []
    fold_counter = 0
    total_specs = 0
    for composition_key in catalog['composition_key'].tolist():
        _, specs = build_context_specs_for_composition(data, composition_key)
        total_specs += len(specs)
    n_conditions = data['condition_key'].nunique()
    estimated_folds = total_specs + total_specs * n_conditions
    print('\n' + '=' * 130)
    print('FINAL OUTER EVALUATION')
    print('=' * 130)
    print(f'Context specifications : {total_specs}')
    print(f'Conditions             : {n_conditions}')
    print(f'Estimated outer folds  : {estimated_folds}')
    print(f'Inner grouped folds    : up to {INNER_GROUP_FOLDS}')
    for _, catalog_row in catalog.iterrows():
        composition_key = str(catalog_row['composition_key'])
        constituents, specs = build_context_specs_for_composition(data, composition_key)
        target_composition_mask = ((data['fault_role'] == 'compound') & (data['composition_key'] == composition_key)).to_numpy()
        target_conditions = sorted(data.loc[target_composition_mask, 'condition_key'].astype(str).unique().tolist())
        for spec in specs:
            context_excluded = np.asarray(spec['excluded_context_mask'], dtype=bool)
            condition_mode = 'SEEN_CONDITION'
            train_mask = ~context_excluded
            test_mask = target_composition_mask.copy()
            if train_mask[test_mask].any():
                raise RuntimeError('Target composition leaked into seen-condition training.')
            train_conditions = set(data.loc[train_mask, 'condition_key'].astype(str))
            test_conditions = set(data.loc[test_mask, 'condition_key'].astype(str))
            condition_seen_all = test_conditions.issubset(train_conditions)
            if not condition_seen_all:
                raise RuntimeError(f'{composition_key}: a test condition is not represented in seen-condition outer training.')
            train_df = data.loc[train_mask]
            context_counts = {component: constituent_context_count(train_df, component) for component in constituents}
            fold_counter += 1
            fold_id = f'F{fold_counter:04d}'
            fold_meta = {'fold_id': fold_id, 'condition_mode': condition_mode, 'context_regime': spec['context_regime'], 'target_composition': composition_key, 'target_constituents': '|'.join(constituents), 'context_unseen_component': spec['context_unseen_component'], 'context_seen_component': spec['context_seen_component'], 'heldout_condition': '', 'n_target_test_runs': int(test_mask.sum()), 'n_outer_train': int(train_mask.sum()), 'n_context_excluded': int(context_excluded.sum()), 'n_condition_excluded': 0, 'constituent_1_contexts_in_outer_train': context_counts[constituents[0]], 'constituent_2_contexts_in_outer_train': context_counts[constituents[1]]}
            prediction, thresholds = run_outer_fold(data, raw_matrices, train_mask, test_mask, fold_meta)
            all_predictions.append(prediction)
            all_thresholds.append(thresholds)
            fold_definition_rows.append(fold_meta)
            print(f"[{fold_counter:04d}] {condition_mode:16s} | {spec['context_regime']:22s} | {composition_key:45s} | test={test_mask.sum():2d} train={train_mask.sum():3d}")
            for heldout_condition in target_conditions:
                condition_mask = (data['condition_key'].astype(str) == heldout_condition).to_numpy()
                train_mask = ~context_excluded & ~condition_mask
                test_mask = target_composition_mask & condition_mask
                if test_mask.sum() == 0:
                    continue
                if train_mask[test_mask].any():
                    raise RuntimeError('Crossed outer train/test overlap.')
                if data.loc[train_mask, 'condition_key'].astype(str).eq(heldout_condition).any():
                    raise RuntimeError('Held-out condition leaked into crossed outer training.')
                if data.loc[train_mask, 'composition_key'].astype(str).eq(composition_key).any():
                    raise RuntimeError('Held-out composition leaked into crossed outer training.')
                train_df = data.loc[train_mask]
                context_counts = {component: constituent_context_count(train_df, component) for component in constituents}
                fold_counter += 1
                fold_id = f'F{fold_counter:04d}'
                fold_meta = {'fold_id': fold_id, 'condition_mode': 'UNSEEN_CONDITION', 'context_regime': spec['context_regime'], 'target_composition': composition_key, 'target_constituents': '|'.join(constituents), 'context_unseen_component': spec['context_unseen_component'], 'context_seen_component': spec['context_seen_component'], 'heldout_condition': heldout_condition, 'n_target_test_runs': int(test_mask.sum()), 'n_outer_train': int(train_mask.sum()), 'n_context_excluded': int(context_excluded.sum()), 'n_condition_excluded': int(condition_mask.sum()), 'constituent_1_contexts_in_outer_train': context_counts[constituents[0]], 'constituent_2_contexts_in_outer_train': context_counts[constituents[1]]}
                prediction, thresholds = run_outer_fold(data, raw_matrices, train_mask, test_mask, fold_meta)
                all_predictions.append(prediction)
                all_thresholds.append(thresholds)
                fold_definition_rows.append(fold_meta)
                print(f"[{fold_counter:04d}] {'UNSEEN_CONDITION':16s} | {spec['context_regime']:22s} | {composition_key:45s} | {heldout_condition:35s} | test={test_mask.sum():2d} train={train_mask.sum():3d}")
    predictions = pd.concat(all_predictions, ignore_index=True)
    thresholds = pd.concat(all_thresholds, ignore_index=True)
    fold_definitions = pd.DataFrame(fold_definition_rows)
    predictions.to_csv(RESULT_DIR / 'ALL_OUTER_PREDICTIONS.csv', index=False, encoding='utf-8-sig')
    thresholds.to_csv(RESULT_DIR / 'FOLD_THRESHOLDS.csv', index=False, encoding='utf-8-sig')
    fold_definitions.to_csv(RESULT_DIR / 'FOLD_DEFINITIONS.csv', index=False, encoding='utf-8-sig')
    return (predictions, thresholds, fold_definitions)

def get_arrays(df):
    y_true = df[COMPONENTS].astype(int).to_numpy()
    y_pred = np.column_stack([df[f'pred_{component}'].astype(int).to_numpy() for component in COMPONENTS])
    return (y_true, y_pred)

def calculate_metrics(df):
    if len(df) == 0:
        return {}
    y_true, y_pred = get_arrays(df)
    all_f1 = []
    active_f1 = []
    for j in range(len(COMPONENTS)):
        truth = y_true[:, j]
        pred = y_pred[:, j]
        f1 = f1_score(truth, pred, zero_division=0)
        all_f1.append(f1)
        if truth.sum() > 0:
            active_f1.append(f1)
    micro_f1 = float(f1_score(y_true.reshape(-1), y_pred.reshape(-1), zero_division=0))
    exact = float(np.mean(np.all(y_true == y_pred, axis=1)))
    total_true = int(y_true.sum())
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    constituent_recall = true_positive / total_true if total_true > 0 else np.nan
    false_additions_each = ((y_true == 0) & (y_pred == 1)).sum(axis=1)
    true_recovered_each = ((y_true == 1) & (y_pred == 1)).sum(axis=1)
    true_count_each = y_true.sum(axis=1)
    return {'n_prediction_rows': len(df), 'n_unique_runs': int(df['run_id'].nunique()), 'mechanism_all9_macro_f1': float(np.mean(all_f1)), 'mechanism_active_macro_f1': float(np.mean(active_f1)) if len(active_f1) > 0 else np.nan, 'mechanism_micro_f1': micro_f1, 'exact_match': exact, 'constituent_recall': float(constituent_recall), 'false_additions_per_run': float(false_additions_each.mean()), 'any_constituent_recovered_rate': float((true_recovered_each > 0).mean()), 'all_constituents_recovered_rate': float((true_recovered_each == true_count_each).mean()), 'mean_predicted_component_count': float(y_pred.sum(axis=1).mean()), 'mean_true_component_count': float(y_true.sum(axis=1).mean())}

def summarize_groups(predictions, group_columns):
    rows = []
    grouped = predictions.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(calculate_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)

def build_summary_tables(predictions):
    summary_protocol = summarize_groups(predictions, ['condition_mode', 'context_regime'])
    summary_direction = summarize_groups(predictions, ['condition_mode', 'context_regime', 'context_unseen_component'])
    summary_composition = summarize_groups(predictions, ['condition_mode', 'context_regime', 'target_composition', 'context_unseen_component'])
    summary_condition = summarize_groups(predictions[predictions['condition_mode'] == 'UNSEEN_CONDITION'], ['context_regime', 'heldout_condition'])
    summary_protocol.to_csv(RESULT_DIR / 'SUMMARY_BY_PROTOCOL.csv', index=False, encoding='utf-8-sig')
    summary_direction.to_csv(RESULT_DIR / 'SUMMARY_BY_PROTOCOL_DIRECTION.csv', index=False, encoding='utf-8-sig')
    summary_composition.to_csv(RESULT_DIR / 'SUMMARY_BY_COMPOSITION.csv', index=False, encoding='utf-8-sig')
    summary_condition.to_csv(RESULT_DIR / 'SUMMARY_BY_CONDITION.csv', index=False, encoding='utf-8-sig')
    return (summary_protocol, summary_direction, summary_composition, summary_condition)

def build_per_component_summary(predictions):
    rows = []
    for condition_mode, context_regime, component in ((mode, regime, component) for mode in sorted(predictions['condition_mode'].unique()) for regime in sorted(predictions['context_regime'].unique()) for component in COMPONENTS):
        group = predictions[(predictions['condition_mode'] == condition_mode) & (predictions['context_regime'] == context_regime)]
        if len(group) == 0:
            continue
        truth = group[component].astype(int).to_numpy()
        pred = group[f'pred_{component}'].astype(int).to_numpy()
        positive_support = int(truth.sum())
        negative_support = int(len(truth) - positive_support)
        precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(truth, pred, labels=[1], zero_division=0)
        rows.append({'condition_mode': condition_mode, 'context_regime': context_regime, 'component': component, 'positive_support': positive_support, 'negative_support': negative_support, 'precision': float(precision_arr[0]), 'recall': float(recall_arr[0]), 'f1': float(f1_arr[0])})
    result = pd.DataFrame(rows)
    result.to_csv(RESULT_DIR / 'SUMMARY_BY_COMPONENT.csv', index=False, encoding='utf-8-sig')
    return result
BOOTSTRAP_METRICS = ['mechanism_micro_f1', 'mechanism_active_macro_f1', 'exact_match', 'constituent_recall', 'false_additions_per_run', 'all_constituents_recovered_rate']

def cluster_bootstrap_ci(group, rng, repeats=BOOTSTRAP_REPEATS):
    """
    Cluster bootstrap by run_id.

    ONE_SIDED_CONTEXT_ZS contains two directional predictions for the same
    physical run. Resampling run_id as a cluster avoids treating those two
    rows as independent experimental units.
    """
    unique_runs = group['run_id'].astype(str).unique()
    if len(unique_runs) < 2:
        return {metric: (np.nan, np.nan) for metric in BOOTSTRAP_METRICS}
    by_run = {run_id: group[group['run_id'].astype(str) == run_id].copy() for run_id in unique_runs}
    values = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(repeats):
        sampled_ids = rng.choice(unique_runs, size=len(unique_runs), replace=True)
        sampled_frames = []
        for bootstrap_instance, run_id in enumerate(sampled_ids):
            frame = by_run[run_id].copy()
            frame['_bootstrap_instance'] = bootstrap_instance
            sampled_frames.append(frame)
        sample = pd.concat(sampled_frames, ignore_index=True)
        metrics = calculate_metrics(sample)
        for metric in BOOTSTRAP_METRICS:
            value = metrics[metric]
            if np.isfinite(value):
                values[metric].append(float(value))
    intervals = {}
    for metric in BOOTSTRAP_METRICS:
        arr = np.asarray(values[metric], dtype=float)
        if len(arr) == 0:
            intervals[metric] = (np.nan, np.nan)
        else:
            intervals[metric] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return intervals

def build_bootstrap_ci(predictions):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (condition_mode, context_regime), group in predictions.groupby(['condition_mode', 'context_regime'], sort=True):
        point = calculate_metrics(group)
        intervals = cluster_bootstrap_ci(group, rng)
        for metric in BOOTSTRAP_METRICS:
            low, high = intervals[metric]
            rows.append({'condition_mode': condition_mode, 'context_regime': context_regime, 'metric': metric, 'point_estimate': point[metric], 'ci95_low': low, 'ci95_high': high, 'bootstrap_cluster': 'run_id', 'bootstrap_repeats': BOOTSTRAP_REPEATS})
    result = pd.DataFrame(rows)
    result.to_csv(RESULT_DIR / 'BOOTSTRAP_CI_BY_PROTOCOL.csv', index=False, encoding='utf-8-sig')
    return result

def final_integrity_checks(data, catalog, predictions, fold_definitions):
    print('\n' + '=' * 130)
    print('FINAL PROTOCOL INTEGRITY')
    print('=' * 130)
    compound_run_ids = set(data.loc[data['fault_role'] == 'compound', 'run_id'].astype(str))
    print(f'Compound physical runs: {len(compound_run_ids)}')
    for mode in CONDITION_MODES:
        for regime in CONTEXT_REGIMES:
            group = predictions[(predictions['condition_mode'] == mode) & (predictions['context_regime'] == regime)]
            if len(group) == 0:
                print(f'{mode:16s} | {regime:22s} : NO FEASIBLE FOLDS')
                continue
            run_counts = group['run_id'].astype(str).value_counts()
            expected_per_run = 2 if regime == 'ONE_SIDED_CONTEXT_ZS' else 1
            correct_multiplicity = bool((run_counts == expected_per_run).all())
            coverage = set(run_counts.index)
            full_coverage = coverage == compound_run_ids
            print(f'{mode:16s} | {regime:22s} | rows={len(group):4d} | unique_runs={len(coverage):3d} | expected multiplicity={expected_per_run} | multiplicity PASS={correct_multiplicity} | coverage PASS={full_coverage}')
    if (fold_definitions['n_outer_train'] <= 0).any():
        raise RuntimeError('An outer fold has no training runs.')
    print('\nIntegrity checks completed.')

def save_config(data, catalog, fold_definitions, feature_map):
    config = {'experiment': 'sampling_rate_sensitivity_crossed_evaluation_6p4khz', 'sampling_rate_hz': SAMPLING_RATE_HZ, 'native_reference_sampling_rate_hz': NATIVE_SAMPLING_RATE_HZ, 'locked_feature_map_source': str(LOCKED_FEATURE_MAP_PATH), 'locked_classifier': {'independent_detector_per_component': True, 'shared_trunk': False, 'feature_source': 'native 12.8-kHz LOCKED_FEATURE_MAP.csv', 'preprocessing': 'outer/inner-training median imputation + StandardScaler', 'classifier': 'L2 LogisticRegression', 'C': LOGISTIC_C, 'C_tuned': False, 'class_weight': 'balanced', 'solver': 'liblinear', 'random_state': SEED}, 'composition_identity': 'sorted active mechanism labels; fault severity text ignored', 'context_regimes': {'RECOMBINATION': 'target pair removed; both constituents remain seen in other compound contexts', 'ONE_SIDED_CONTEXT_ZS': 'all compounds containing one designated constituent removed; the other constituent must remain seen in another compound context', 'TWO_SIDED_CONTEXT_ZS': 'all compounds containing either target constituent removed; both remain available as single faults'}, 'condition_modes': {'SEEN_CONDITION': 'context exclusions only; target test conditions remain represented in outer training through other runs', 'UNSEEN_CONDITION': 'context exclusions plus removal of every run from the target operating condition'}, 'inner_threshold_calibration': {'method': 'GroupKFold by condition_key', 'n_splits_max': INNER_GROUP_FOLDS, 'threshold_grid': THRESHOLD_GRID.tolist(), 'primary': 'F1', 'secondary': 'balanced_accuracy', 'final_tie_break': 'closest_to_0.50', 'outer_test_used': False}, 'bootstrap': {'cluster': 'run_id', 'repeats': BOOTSTRAP_REPEATS, 'seed': BOOTSTRAP_SEED, 'reason': 'ONE_SIDED_CONTEXT_ZS contains two directional predictions for the same physical run'}, 'n_total_runs': len(data), 'n_compound_runs': int((data['fault_role'] == 'compound').sum()), 'n_conditions': int(data['condition_key'].nunique()), 'n_compositions': len(catalog), 'n_outer_folds': len(fold_definitions), 'n_features_per_component': {component: len(feature_map[component]) for component in COMPONENTS}, 'previous_P3_P4_partitions_used_for_selection': False, 'post_outer_result_policy': 'Do not change classifier, feature families, C, or threshold objective after inspecting these results.'}
    with open(RESULT_DIR / 'experiment_config.json', 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

def main():
    print('\n' + '=' * 145)
    print('6.4-kHz SAMPLING SENSITIVITY — CROSSED COMPOSITION × CONTEXT × CONDITION')
    print('=' * 145)
    print('\nLOCKED METHOD:')
    print('  9 independent constituent detectors')
    print('  curated mechanism-specific physics features')
    print('  L2 logistic regression')
    print('  C = 1.0 FIXED')
    print('  grouped inner-CV threshold calibration')
    print('  NO outer-test tuning')
    print('  EXACT native 12.8-kHz locked feature map')
    print(f'  sampling rate = {SAMPLING_RATE_HZ} Hz')
    data, definition = load_data()
    feature_map, raw_matrices = build_feature_map_and_raw_matrices(data, definition)
    catalog = build_composition_catalog(data)
    predictions, thresholds, fold_definitions = execute_final_protocol(data, catalog, raw_matrices)
    final_integrity_checks(data, catalog, predictions, fold_definitions)
    summary_protocol, summary_direction, summary_composition, summary_condition = build_summary_tables(predictions)
    component_summary = build_per_component_summary(predictions)
    bootstrap_ci = build_bootstrap_ci(predictions)
    save_config(data, catalog, fold_definitions, feature_map)
    print('\n' + '=' * 155)
    print('FINAL PRIMARY TABLE')
    print('=' * 155)
    display_columns = ['condition_mode', 'context_regime', 'n_prediction_rows', 'n_unique_runs', 'mechanism_active_macro_f1', 'mechanism_micro_f1', 'exact_match', 'constituent_recall', 'false_additions_per_run', 'any_constituent_recovered_rate', 'all_constituents_recovered_rate']
    print(summary_protocol[display_columns].to_string(index=False))
    print('\n' + '=' * 155)
    print('INTERPRETATION GUIDE')
    print('=' * 155)
    print('RECOMBINATION:')
    print('  Both constituents were seen in other compound contexts; only their pairing is new.')
    print('\nONE_SIDED_CONTEXT_ZS:')
    print('  One constituent was never seen in any compound context during that outer fold; it was available only as a single fault.')
    print('\nTWO_SIDED_CONTEXT_ZS:')
    print('  Neither constituent was seen in any compound context during that outer fold; both were available only as single faults.')
    print('\nSEEN_CONDITION vs UNSEEN_CONDITION:')
    print('  Directly quantifies the additional penalty from operating-condition shift after controlling compound-context exposure.')
    print('\nResults:')
    print(RESULT_DIR)
    print('\nMost important files:')
    for name in ['COMPOSITION_CATALOG.csv', 'FOLD_DEFINITIONS.csv', 'FOLD_THRESHOLDS.csv', 'ALL_OUTER_PREDICTIONS.csv', 'SUMMARY_BY_PROTOCOL.csv', 'SUMMARY_BY_PROTOCOL_DIRECTION.csv', 'SUMMARY_BY_COMPOSITION.csv', 'SUMMARY_BY_COMPONENT.csv', 'SUMMARY_BY_CONDITION.csv', 'BOOTSTRAP_CI_BY_PROTOCOL.csv', 'experiment_config.json']:
        print(' -', name)
    print('\nFINAL EVALUATION COMPLETED.')
    print('Do not modify the locked classifier from these outer-test results.')
if __name__ == '__main__':
    main()
