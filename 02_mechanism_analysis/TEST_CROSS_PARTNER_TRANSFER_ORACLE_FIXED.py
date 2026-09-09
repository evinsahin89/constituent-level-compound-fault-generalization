"""Cross-partner transfer analysis for constituent-specific fault evidence.

The analysis tests whether evidence learned for a target constituent transfers
to the same constituent when paired with a previously excluded partner.
Training uses isolated single-fault evidence, alternative compound partners,
or both. Seen- and unseen-operating-condition settings are evaluated
separately.

A within-pair leave-one-condition-out analysis is retained as a reference only.
All preprocessing is fitted on training data and classifier hyperparameters are
fixed.
"""
from pathlib import Path
import json
import random
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
ROOT = Path('D:\\Multi-mode Fault Diagnosis Datasets of Three-phase')
PROTOCOL_DIR = ROOT / 'EXPERIMENT_PROTOCOL'
PHYSICS_PATH = ROOT / 'PHYSICS_DIAGNOSTICS_RESULTS' / 'RUN_PHYSICS_FEATURES.csv'
RESULT_DIR = ROOT / 'CROSS_PARTNER_TRANSFER_ORACLE_RESULTS'
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OLD_SPLITS = ['train', 'validation', 'P1', 'P2', 'P3', 'P4']
COMPONENTS = ['bearing_ball', 'bearing_inner', 'bearing_outer', 'bend', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'voltage_unbalance', 'winding']
COMPOUND_COMPONENTS = ['bearing_inner', 'bearing_outer', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'winding']
SEED = 42
LOGISTIC_C = 1.0
BOOTSTRAP_REPEATS = 3000
PERMUTATION_REPEATS = 5000
MIN_TRAIN_PER_CLASS = 4
MIN_TEST_CONDITIONS = 4
FEATURE_METADATA_EXCLUDE = {'run_id', 'fault_raw', 'fault_role', 'condition_key', 'mode', 'torque_nm', 'rpm_nominal'}

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
set_seed()

def infer_fault_role(fault_raw):
    text = str(fault_raw).strip().lower()
    if text in {'health', 'healthy', 'normal', 'nc', 'normal_condition'}:
        return 'health'
    if '_and_' in text:
        return 'compound'
    return 'single'

def active_components_from_row(row):
    return tuple((component for component in COMPONENTS if int(row[component]) == 1))

def composition_key_from_row(row):
    active = active_components_from_row(row)
    if len(active) < 2:
        return ''
    return '+'.join(sorted(active))

def mechanism_key(fault_raw):
    fault_raw = str(fault_raw)
    if fault_raw.startswith('bearing_ball'):
        return 'bearing_ball'
    if fault_raw.startswith('bearing_inner'):
        return 'bearing_inner'
    if fault_raw.startswith('bearing_outer'):
        return 'bearing_outer'
    if fault_raw == 'bend':
        return 'bend'
    if fault_raw == 'broken_bar':
        return 'broken_bar'
    if fault_raw == 'dynamic_eccentricity':
        return 'dynamic_eccentricity'
    if fault_raw.startswith('static_eccentricity'):
        return 'static_eccentricity'
    if fault_raw.startswith('voltage_unbalance'):
        return 'voltage_unbalance'
    if fault_raw.startswith('winding'):
        return 'winding'
    return None

def load_full_data():
    print('\n' + '=' * 125)
    print('LOAD FULL DATASET')
    print('=' * 125)
    if not PHYSICS_PATH.exists():
        raise FileNotFoundError(f'RUN_PHYSICS_FEATURES.csv not found:\n{PHYSICS_PATH}')
    frames = []
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
        frames.append(df)
    runs = pd.concat(frames, ignore_index=True)
    if runs['run_id'].duplicated().any():
        raise RuntimeError('Protocol manifests contain duplicate run_id.')
    runs['fault_role'] = runs['fault_raw'].apply(infer_fault_role)
    runs['composition_key'] = runs.apply(composition_key_from_row, axis=1)
    physics = pd.read_csv(PHYSICS_PATH)
    physics['run_id'] = physics['run_id'].astype(str)
    if physics['run_id'].duplicated().any():
        raise RuntimeError('RUN_PHYSICS_FEATURES.csv contains duplicate run_id.')
    physics_labels_removed = physics.drop(columns=['fault_raw', 'fault_role'], errors='ignore')
    data = runs.merge(physics_labels_removed, on='run_id', how='left', validate='one_to_one')
    if data['condition_key'].isna().any():
        raise RuntimeError('Some runs have no physics condition_key.')
    data['condition_key'] = data['condition_key'].astype(str)
    print(f'Total runs        : {len(data)}')
    print(f"Health / single / compound : {(data['fault_role'] == 'health').sum()} / {(data['fault_role'] == 'single').sum()} / {(data['fault_role'] == 'compound').sum()}")
    print(f"Conditions        : {data['condition_key'].nunique()}")
    print(f"Compound pairs    : {data.loc[data['fault_role'] == 'compound', 'composition_key'].nunique()}")
    return data

def select_feature_columns(data):
    feature_columns = []
    for column in data.columns:
        if column in FEATURE_METADATA_EXCLUDE:
            continue
        if column in {'old_split', 'composition_key', *COMPONENTS}:
            continue
        if not pd.api.types.is_numeric_dtype(data[column]):
            continue
        values = pd.to_numeric(data[column], errors='coerce').to_numpy(dtype=float)
        finite_fraction = np.isfinite(values).mean()
        if finite_fraction >= 0.75:
            feature_columns.append(column)
    if len(feature_columns) < 300:
        raise RuntimeError(f'Expected broad physics feature space was not found. Feature count={len(feature_columns)}')
    pd.DataFrame({'feature': feature_columns}).to_csv(RESULT_DIR / 'FEATURE_LIST.csv', index=False, encoding='utf-8-sig')
    print(f'Usable numeric physics features: {len(feature_columns)}')
    return feature_columns

def build_aggregated_table(data, feature_columns):
    """
    Aggregate duplicate fault_raw + condition rows by median.

    Labels are not changed. This is only for obtaining one matched oracle
    reference vector for each named fault state and operating condition.
    """
    identity_columns = ['fault_raw', 'fault_role', 'composition_key', 'condition_key']
    grouped_features = data[identity_columns + feature_columns].groupby(identity_columns, dropna=False, as_index=False)[feature_columns].median(numeric_only=True)
    counts = data.groupby(identity_columns, dropna=False).size().reset_index(name='source_run_count')
    aggregated = grouped_features.merge(counts, on=identity_columns, how='left', validate='one_to_one')
    duplicate_refs = aggregated[aggregated['source_run_count'] > 1]
    print(f'Aggregated state-condition rows : {len(aggregated)}')
    print(f'Duplicate references aggregated : {len(duplicate_refs)}')
    return aggregated

def available_single_labels(aggregated, component):
    labels = aggregated.loc[(aggregated['fault_role'] == 'single') & (aggregated['fault_raw'].apply(mechanism_key) == component), 'fault_raw'].astype(str).unique().tolist()
    return sorted(labels)

def resolve_single_label(aggregated, component, compound_fault_raw):
    """
    Resolve the single-fault label corresponding to a constituent in a
    compound label WITHOUT changing the dataset label.

    Important dataset naming exception:
        bearing_outer_H_and_inner_H

    Here the inner-bearing constituent is written as "inner_H" rather than
    "bearing_inner_H". Therefore simple substring matching is insufficient.

    Resolution order:
        1) exact candidate substring, when unique
        2) infer H/L severity from component-specific aliases in compound text
        3) exact non-severity component label
        4) unique candidate fallback

    Examples:
        bearing_outer_H_and_inner_H + bearing_inner -> bearing_inner_H
        bearing_outer_H_and_inner_H + bearing_outer -> bearing_outer_H
        winding_H_and_bearing_inner_H + winding     -> winding_H
        broken_bar_and_bearing_inner_H + broken_bar -> broken_bar
    """
    candidates = available_single_labels(aggregated, component)
    if len(candidates) == 0:
        raise RuntimeError(f'No single label exists for {component}.')
    compound_fault_raw = str(compound_fault_raw)
    contained = [label for label in sorted(candidates, key=len, reverse=True) if label in compound_fault_raw]
    if len(contained) == 1:
        return contained[0]
    severity_aliases = {'bearing_ball': ['bearing_ball', 'ball'], 'bearing_inner': ['bearing_inner', 'inner'], 'bearing_outer': ['bearing_outer', 'outer'], 'static_eccentricity': ['static_eccentricity'], 'voltage_unbalance': ['voltage_unbalance'], 'winding': ['winding']}
    aliases = severity_aliases.get(component, [component])
    severities_found = []
    for alias in aliases:
        pattern = '(?:^|_)' + re.escape(alias) + '_(H|L)(?:_|$)'
        matches = re.findall(pattern, compound_fault_raw)
        severities_found.extend(matches)
    severities_found = sorted(set(severities_found))
    if len(severities_found) == 1:
        severity = severities_found[0]
        severity_candidates = [label for label in candidates if label.endswith(f'_{severity}')]
        if len(severity_candidates) == 1:
            return severity_candidates[0]
    exact_component = [label for label in candidates if label == component]
    if len(exact_component) == 1:
        return exact_component[0]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f'Could not uniquely resolve single label. component={component}, compound={compound_fault_raw}, candidates={candidates}, severities_found={severities_found}')

def get_health_label(aggregated):
    health_labels = aggregated.loc[aggregated['fault_role'] == 'health', 'fault_raw'].astype(str).unique().tolist()
    if len(health_labels) != 1:
        raise RuntimeError(f'Expected one health label, found {health_labels}')
    return health_labels[0]

def build_composition_catalog(aggregated):
    compound = aggregated[aggregated['fault_role'] == 'compound'].copy()
    rows = []
    for composition_key, group in compound.groupby('composition_key', sort=True):
        active = composition_key.split('+')
        raw_labels = sorted(group['fault_raw'].astype(str).unique().tolist())
        if len(raw_labels) != 1:
            raise RuntimeError(f'{composition_key}: multiple fault_raw labels: {raw_labels}')
        rows.append({'composition_key': composition_key, 'constituent_1': active[0], 'constituent_2': active[1], 'fault_raw': raw_labels[0], 'n_conditions': group['condition_key'].nunique(), 'n_aggregated_rows': len(group)})
    catalog = pd.DataFrame(rows)
    catalog.to_csv(RESULT_DIR / 'COMPOSITION_CATALOG.csv', index=False, encoding='utf-8-sig')
    return catalog

def state_rows(aggregated, fault_raw):
    return aggregated.loc[aggregated['fault_raw'].astype(str) == str(fault_raw)].copy().sort_values('condition_key').reset_index(drop=True)

def matched_binary_context(aggregated, positive_fault_raw, negative_fault_raw, feature_columns, context_name):
    """
    Return matched positive and negative rows on common condition_key.
    """
    pos = state_rows(aggregated, positive_fault_raw)
    neg = state_rows(aggregated, negative_fault_raw)
    common_conditions = sorted(set(pos['condition_key']) & set(neg['condition_key']))
    if len(common_conditions) == 0:
        return pd.DataFrame()
    pos = pos[pos['condition_key'].isin(common_conditions)].set_index('condition_key').loc[common_conditions].reset_index()
    neg = neg[neg['condition_key'].isin(common_conditions)].set_index('condition_key').loc[common_conditions].reset_index()
    pos_out = pos[['condition_key', *feature_columns]].copy()
    pos_out['y'] = 1
    pos_out['context_name'] = context_name
    pos_out['state_name'] = positive_fault_raw
    neg_out = neg[['condition_key', *feature_columns]].copy()
    neg_out['y'] = 0
    neg_out['context_name'] = context_name
    neg_out['state_name'] = negative_fault_raw
    return pd.concat([pos_out, neg_out], ignore_index=True)

def build_directed_test_catalog(aggregated, composition_catalog):
    rows = []
    for _, composition in composition_catalog.iterrows():
        target_composition = str(composition['composition_key'])
        target_fault_raw = str(composition['fault_raw'])
        constituents = [str(composition['constituent_1']), str(composition['constituent_2'])]
        for A in constituents:
            B = constituents[1] if A == constituents[0] else constituents[0]
            A_single = resolve_single_label(aggregated, A, target_fault_raw)
            B_single = resolve_single_label(aggregated, B, target_fault_raw)
            rows.append({'target_composition': target_composition, 'target_fault_raw': target_fault_raw, 'constituent_A': A, 'partner_B': B, 'A_single_label': A_single, 'B_single_label': B_single})
    directed = pd.DataFrame(rows)
    if len(directed) != 18:
        print(f'WARNING: expected 18 directed tests, found {len(directed)}')
    directed.to_csv(RESULT_DIR / 'DIRECTED_TEST_CATALOG.csv', index=False, encoding='utf-8-sig')
    return directed

def other_partner_contexts(aggregated, composition_catalog, directed_row, feature_columns):
    """
    Build A+C vs C(single) contexts, excluding target A+B.

    Only contexts whose A severity resolves to the SAME A single label as the
    target test are retained.
    """
    target_composition = str(directed_row['target_composition'])
    A = str(directed_row['constituent_A'])
    A_single_target = str(directed_row['A_single_label'])
    contexts = []
    audit_rows = []
    for _, candidate in composition_catalog.iterrows():
        candidate_key = str(candidate['composition_key'])
        if candidate_key == target_composition:
            continue
        components = candidate_key.split('+')
        if A not in components:
            continue
        C = components[1] if components[0] == A else components[0]
        candidate_fault_raw = str(candidate['fault_raw'])
        candidate_A_single = resolve_single_label(aggregated, A, candidate_fault_raw)
        if candidate_A_single != A_single_target:
            audit_rows.append({'candidate_composition': candidate_key, 'partner_C': C, 'included': 0, 'reason': 'A severity/single label does not match target', 'candidate_A_single': candidate_A_single})
            continue
        C_single = resolve_single_label(aggregated, C, candidate_fault_raw)
        context_name = f'{candidate_key}__VS__{C_single}'
        frame = matched_binary_context(aggregated, positive_fault_raw=candidate_fault_raw, negative_fault_raw=C_single, feature_columns=feature_columns, context_name=context_name)
        if len(frame) == 0:
            audit_rows.append({'candidate_composition': candidate_key, 'partner_C': C, 'included': 0, 'reason': 'no matched conditions', 'candidate_A_single': candidate_A_single})
            continue
        contexts.append(frame)
        audit_rows.append({'candidate_composition': candidate_key, 'partner_C': C, 'included': 1, 'reason': 'included', 'candidate_A_single': candidate_A_single, 'matched_conditions': int(frame['condition_key'].nunique())})
    return (contexts, audit_rows)

def build_training_protocols(aggregated, composition_catalog, directed_row, feature_columns, health_label):
    A_single = str(directed_row['A_single_label'])
    single_context = matched_binary_context(aggregated, positive_fault_raw=A_single, negative_fault_raw=health_label, feature_columns=feature_columns, context_name=f'{A_single}__VS__{health_label}')
    other_context_frames, audit_rows = other_partner_contexts(aggregated, composition_catalog, directed_row, feature_columns)
    if len(other_context_frames) > 0:
        other_partner_train = pd.concat(other_context_frames, ignore_index=True)
    else:
        other_partner_train = pd.DataFrame()
    if len(single_context) > 0 and len(other_partner_train) > 0:
        hybrid = pd.concat([single_context, other_partner_train], ignore_index=True)
    elif len(single_context) > 0:
        hybrid = single_context.copy()
    else:
        hybrid = other_partner_train.copy()
    return {'SINGLE_ONLY_TRANSFER': single_context, 'OTHER_PARTNERS_TRANSFER': other_partner_train, 'HYBRID_TRANSFER': hybrid, '_audit': audit_rows}

def build_target_test(aggregated, directed_row, feature_columns):
    target_fault_raw = str(directed_row['target_fault_raw'])
    B_single = str(directed_row['B_single_label'])
    return matched_binary_context(aggregated, positive_fault_raw=target_fault_raw, negative_fault_raw=B_single, feature_columns=feature_columns, context_name=f'TARGET__{target_fault_raw}__VS__{B_single}')

def fit_classifier(train_df, feature_columns):
    y = train_df['y'].astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise RuntimeError('Training data contains one class.')
    X_raw = train_df[feature_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    medians = np.nanmedian(X_raw, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    missing = ~np.isfinite(X_raw)
    if missing.any():
        row_idx, col_idx = np.where(missing)
        X_raw[row_idx, col_idx] = medians[col_idx]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    model = LogisticRegression(penalty='l2', C=LOGISTIC_C, class_weight='balanced', solver='liblinear', max_iter=5000, random_state=SEED)
    model.fit(X, y)
    return {'medians': medians, 'scaler': scaler, 'model': model}

def score_classifier(fitted, df, feature_columns):
    X = df[feature_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    missing = ~np.isfinite(X)
    if missing.any():
        row_idx, col_idx = np.where(missing)
        X[row_idx, col_idx] = fitted['medians'][col_idx]
    X = fitted['scaler'].transform(X)
    return fitted['model'].predict_proba(X)[:, 1]

def paired_scores_by_condition(scored_df):
    """
    Require one positive and one negative score per condition.
    """
    rows = []
    for condition, group in scored_df.groupby('condition_key', sort=True):
        pos = group.loc[group['y'] == 1, 'score'].to_numpy(dtype=float)
        neg = group.loc[group['y'] == 0, 'score'].to_numpy(dtype=float)
        if len(pos) != 1 or len(neg) != 1:
            continue
        rows.append({'condition_key': condition, 'positive_score': float(pos[0]), 'negative_score': float(neg[0])})
    return pd.DataFrame(rows)

def auc_from_pair_table(pair_table):
    if len(pair_table) < 2:
        return np.nan
    y = np.concatenate([np.ones(len(pair_table), dtype=int), np.zeros(len(pair_table), dtype=int)])
    score = np.concatenate([pair_table['positive_score'].to_numpy(dtype=float), pair_table['negative_score'].to_numpy(dtype=float)])
    if len(np.unique(y)) != 2:
        return np.nan
    return float(roc_auc_score(y, score))

def paired_bootstrap_ci(pair_table, repeats=BOOTSTRAP_REPEATS, seed=SEED):
    if len(pair_table) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(pair_table)
    values = []
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        sample = pair_table.iloc[idx]
        auc = auc_from_pair_table(sample)
        if np.isfinite(auc):
            values.append(auc)
    if len(values) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))

def paired_permutation_p(pair_table, observed_auc, repeats=PERMUTATION_REPEATS, seed=SEED):
    """
    Under H0, within each operating condition the positive/negative score
    labels are exchangeable. Randomly swap them.
    """
    if len(pair_table) < 2 or not np.isfinite(observed_auc):
        return np.nan
    rng = np.random.default_rng(seed)
    pos = pair_table['positive_score'].to_numpy(dtype=float)
    neg = pair_table['negative_score'].to_numpy(dtype=float)
    null_values = []
    for _ in range(repeats):
        swap = rng.random(len(pair_table)) < 0.5
        perm_pos = np.where(swap, neg, pos)
        perm_neg = np.where(swap, pos, neg)
        temp = pd.DataFrame({'positive_score': perm_pos, 'negative_score': perm_neg})
        null_values.append(auc_from_pair_table(temp))
    null_values = np.asarray(null_values, dtype=float)
    p = (1.0 + np.sum(null_values >= observed_auc - 1e-12)) / (1.0 + len(null_values))
    return float(p)

def transfer_band(auc):
    """
    Descriptive only — not an inferential claim.
    """
    if not np.isfinite(auc):
        return 'UNAVAILABLE'
    if auc >= 0.9:
        return 'VERY_HIGH'
    if auc >= 0.8:
        return 'HIGH'
    if auc >= 0.7:
        return 'MODERATE'
    if auc >= 0.6:
        return 'WEAK'
    return 'POOR'

def evaluate_seen_condition(train_df, test_df, feature_columns):
    if len(train_df) == 0:
        return None
    train_counts = train_df['y'].value_counts()
    if train_counts.get(1, 0) < MIN_TRAIN_PER_CLASS or train_counts.get(0, 0) < MIN_TRAIN_PER_CLASS:
        return None
    fitted = fit_classifier(train_df, feature_columns)
    scored = test_df.copy()
    scored['score'] = score_classifier(fitted, scored, feature_columns)
    pair_table = paired_scores_by_condition(scored)
    if len(pair_table) < MIN_TEST_CONDITIONS:
        return None
    auc = auc_from_pair_table(pair_table)
    ci_low, ci_high = paired_bootstrap_ci(pair_table)
    p = paired_permutation_p(pair_table, auc)
    return {'auc': auc, 'ci95_low': ci_low, 'ci95_high': ci_high, 'permutation_p': p, 'n_train_positive': int((train_df['y'] == 1).sum()), 'n_train_negative': int((train_df['y'] == 0).sum()), 'n_train_contexts': int(train_df['context_name'].nunique()), 'n_test_conditions': len(pair_table), 'pair_scores': pair_table}

def evaluate_unseen_condition(train_df, test_df, feature_columns):
    if len(train_df) == 0:
        return None
    score_rows = []
    for condition in sorted(test_df['condition_key'].unique()):
        train_fold = train_df[train_df['condition_key'] != condition].copy()
        test_fold = test_df[test_df['condition_key'] == condition].copy()
        if len(test_fold) != 2:
            continue
        if set(test_fold['y'].astype(int)) != {0, 1}:
            continue
        counts = train_fold['y'].value_counts()
        if counts.get(1, 0) < MIN_TRAIN_PER_CLASS or counts.get(0, 0) < MIN_TRAIN_PER_CLASS:
            continue
        fitted = fit_classifier(train_fold, feature_columns)
        scores = score_classifier(fitted, test_fold, feature_columns)
        for (_, row), score in zip(test_fold.iterrows(), scores):
            score_rows.append({'condition_key': condition, 'y': int(row['y']), 'score': float(score)})
    if len(score_rows) == 0:
        return None
    scored = pd.DataFrame(score_rows)
    pair_table = paired_scores_by_condition(scored)
    if len(pair_table) < MIN_TEST_CONDITIONS:
        return None
    auc = auc_from_pair_table(pair_table)
    ci_low, ci_high = paired_bootstrap_ci(pair_table)
    p = paired_permutation_p(pair_table, auc)
    return {'auc': auc, 'ci95_low': ci_low, 'ci95_high': ci_high, 'permutation_p': p, 'n_train_positive': int((train_df['y'] == 1).sum()), 'n_train_negative': int((train_df['y'] == 0).sum()), 'n_train_contexts': int(train_df['context_name'].nunique()), 'n_test_conditions': len(pair_table), 'pair_scores': pair_table}

def evaluate_within_pair_loco(test_df, feature_columns):
    """
    Target pair data ARE used in training here.
    This is only a condition-held-out within-pair reference.
    """
    score_rows = []
    conditions = sorted(test_df['condition_key'].unique())
    for condition in conditions:
        train_fold = test_df[test_df['condition_key'] != condition].copy()
        test_fold = test_df[test_df['condition_key'] == condition].copy()
        if len(test_fold) != 2:
            continue
        counts = train_fold['y'].value_counts()
        if counts.get(1, 0) < MIN_TRAIN_PER_CLASS or counts.get(0, 0) < MIN_TRAIN_PER_CLASS:
            continue
        fitted = fit_classifier(train_fold, feature_columns)
        scores = score_classifier(fitted, test_fold, feature_columns)
        for (_, row), score in zip(test_fold.iterrows(), scores):
            score_rows.append({'condition_key': condition, 'y': int(row['y']), 'score': float(score)})
    if len(score_rows) == 0:
        return None
    scored = pd.DataFrame(score_rows)
    pair_table = paired_scores_by_condition(scored)
    if len(pair_table) < MIN_TEST_CONDITIONS:
        return None
    auc = auc_from_pair_table(pair_table)
    ci_low, ci_high = paired_bootstrap_ci(pair_table)
    p = paired_permutation_p(pair_table, auc)
    return {'auc': auc, 'ci95_low': ci_low, 'ci95_high': ci_high, 'permutation_p': p, 'n_train_positive': max(len(pair_table) - 1, 0), 'n_train_negative': max(len(pair_table) - 1, 0), 'n_train_contexts': 1, 'n_test_conditions': len(pair_table), 'pair_scores': pair_table}

def run_all_tests(aggregated, composition_catalog, directed_catalog, feature_columns):
    health_label = get_health_label(aggregated)
    result_rows = []
    score_rows = []
    audit_rows_all = []
    print('\n' + '=' * 145)
    print('CROSS-PARTNER TRANSFER TESTS')
    print('=' * 145)
    for test_index, directed_row in directed_catalog.iterrows():
        target_composition = str(directed_row['target_composition'])
        A = str(directed_row['constituent_A'])
        B = str(directed_row['partner_B'])
        print(f'\n[{test_index + 1:02d}/{len(directed_catalog):02d}] A={A:22s} | B={B:22s} | target={target_composition}')
        training_protocols = build_training_protocols(aggregated, composition_catalog, directed_row, feature_columns, health_label)
        for audit in training_protocols['_audit']:
            audit_rows_all.append({'target_composition': target_composition, 'constituent_A': A, 'partner_B': B, **audit})
        target_test = build_target_test(aggregated, directed_row, feature_columns)
        if len(target_test) == 0:
            print('  Target matched test unavailable.')
            continue
        for protocol in ['SINGLE_ONLY_TRANSFER', 'OTHER_PARTNERS_TRANSFER', 'HYBRID_TRANSFER']:
            train_df = training_protocols[protocol]
            for condition_mode in ['SEEN_CONDITION', 'UNSEEN_CONDITION']:
                if condition_mode == 'SEEN_CONDITION':
                    result = evaluate_seen_condition(train_df, target_test, feature_columns)
                else:
                    result = evaluate_unseen_condition(train_df, target_test, feature_columns)
                if result is None:
                    result_rows.append({**directed_row.to_dict(), 'protocol': protocol, 'condition_mode': condition_mode, 'auc': np.nan, 'ci95_low': np.nan, 'ci95_high': np.nan, 'permutation_p': np.nan, 'transfer_band': 'UNAVAILABLE', 'n_train_positive': int((train_df['y'] == 1).sum()) if len(train_df) else 0, 'n_train_negative': int((train_df['y'] == 0).sum()) if len(train_df) else 0, 'n_train_contexts': int(train_df['context_name'].nunique()) if len(train_df) else 0, 'n_test_conditions': 0, 'status': 'UNAVAILABLE'})
                    print(f'  {protocol:28s} | {condition_mode:16s} | UNAVAILABLE')
                    continue
                result_rows.append({**directed_row.to_dict(), 'protocol': protocol, 'condition_mode': condition_mode, 'auc': result['auc'], 'ci95_low': result['ci95_low'], 'ci95_high': result['ci95_high'], 'permutation_p': result['permutation_p'], 'transfer_band': transfer_band(result['auc']), 'n_train_positive': result['n_train_positive'], 'n_train_negative': result['n_train_negative'], 'n_train_contexts': result['n_train_contexts'], 'n_test_conditions': result['n_test_conditions'], 'status': 'OK'})
                pair_scores = result['pair_scores'].copy()
                pair_scores['target_composition'] = target_composition
                pair_scores['constituent_A'] = A
                pair_scores['partner_B'] = B
                pair_scores['protocol'] = protocol
                pair_scores['condition_mode'] = condition_mode
                score_rows.append(pair_scores)
                print(f"  {protocol:28s} | {condition_mode:16s} | AUC={result['auc']:.3f} [{result['ci95_low']:.3f},{result['ci95_high']:.3f}] p={result['permutation_p']:.4f} train_ctx={result['n_train_contexts']}")
        within = evaluate_within_pair_loco(target_test, feature_columns)
        if within is not None:
            result_rows.append({**directed_row.to_dict(), 'protocol': 'WITHIN_PAIR_ORACLE_LOCO', 'condition_mode': 'LOCO_CONDITION_REFERENCE', 'auc': within['auc'], 'ci95_low': within['ci95_low'], 'ci95_high': within['ci95_high'], 'permutation_p': within['permutation_p'], 'transfer_band': transfer_band(within['auc']), 'n_train_positive': within['n_train_positive'], 'n_train_negative': within['n_train_negative'], 'n_train_contexts': within['n_train_contexts'], 'n_test_conditions': within['n_test_conditions'], 'status': 'REFERENCE_ONLY'})
            pair_scores = within['pair_scores'].copy()
            pair_scores['target_composition'] = target_composition
            pair_scores['constituent_A'] = A
            pair_scores['partner_B'] = B
            pair_scores['protocol'] = 'WITHIN_PAIR_ORACLE_LOCO'
            pair_scores['condition_mode'] = 'LOCO_CONDITION_REFERENCE'
            score_rows.append(pair_scores)
            print(f"  {'WITHIN_PAIR_ORACLE_LOCO':28s} | AUC={within['auc']:.3f} [{within['ci95_low']:.3f},{within['ci95_high']:.3f}] p={within['permutation_p']:.4f}")
    results = pd.DataFrame(result_rows)
    scores = pd.concat(score_rows, ignore_index=True) if len(score_rows) else pd.DataFrame()
    audit = pd.DataFrame(audit_rows_all)
    results.to_csv(RESULT_DIR / 'CROSS_PARTNER_RESULTS.csv', index=False, encoding='utf-8-sig')
    scores.to_csv(RESULT_DIR / 'SCORE_LEVEL_RESULTS.csv', index=False, encoding='utf-8-sig')
    audit.to_csv(RESULT_DIR / 'TRAIN_CONTEXT_AUDIT.csv', index=False, encoding='utf-8-sig')
    return results

def summarize_results(results):
    ok = results[results['auc'].notna()].copy()
    constituent_rows = []
    for (protocol, condition_mode, component), group in ok.groupby(['protocol', 'condition_mode', 'constituent_A'], sort=True):
        constituent_rows.append({'protocol': protocol, 'condition_mode': condition_mode, 'constituent_A': component, 'n_directed_tests': len(group), 'mean_auc': float(group['auc'].mean()), 'median_auc': float(group['auc'].median()), 'min_auc': float(group['auc'].min()), 'max_auc': float(group['auc'].max()), 'n_auc_ge_080': int((group['auc'] >= 0.8).sum()), 'n_auc_ge_090': int((group['auc'] >= 0.9).sum()), 'n_perm_p_lt_005': int((group['permutation_p'] < 0.05).sum())})
    by_constituent = pd.DataFrame(constituent_rows)
    by_constituent.to_csv(RESULT_DIR / 'CROSS_PARTNER_BY_CONSTITUENT.csv', index=False, encoding='utf-8-sig')
    overall_rows = []
    for (protocol, condition_mode), group in ok.groupby(['protocol', 'condition_mode'], sort=True):
        overall_rows.append({'protocol': protocol, 'condition_mode': condition_mode, 'n_directed_tests': len(group), 'mean_auc': float(group['auc'].mean()), 'median_auc': float(group['auc'].median()), 'min_auc': float(group['auc'].min()), 'max_auc': float(group['auc'].max()), 'n_auc_ge_070': int((group['auc'] >= 0.7).sum()), 'n_auc_ge_080': int((group['auc'] >= 0.8).sum()), 'n_auc_ge_090': int((group['auc'] >= 0.9).sum()), 'n_perm_p_lt_005': int((group['permutation_p'] < 0.05).sum())})
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(RESULT_DIR / 'CROSS_PARTNER_OVERALL_SUMMARY.csv', index=False, encoding='utf-8-sig')
    return (by_constituent, overall)

def save_config(feature_columns):
    config = {'experiment': 'cross_partner_transfer_oracle', 'primary_question': 'Can constituent A be learned from contexts other than A+B and transferred to held-out A+B against B(single)?', 'primary_protocol': 'OTHER_PARTNERS_TRANSFER', 'strongest_protocol': 'OTHER_PARTNERS_TRANSFER + UNSEEN_CONDITION', 'protocols': {'SINGLE_ONLY_TRANSFER': 'train A(single) vs health; test A+B vs B(single)', 'OTHER_PARTNERS_TRANSFER': 'train A+C vs C(single) for C != B; test A+B vs B(single)', 'HYBRID_TRANSFER': 'combine single-only and other-partner training contexts', 'WITHIN_PAIR_ORACLE_LOCO': 'reference only; target pair is trained on all but one condition'}, 'condition_modes': {'SEEN_CONDITION': 'training may contain target operating conditions', 'UNSEEN_CONDITION': 'for each target condition, remove that condition from training and pool held-out scores'}, 'classifier': {'type': 'LogisticRegression', 'penalty': 'L2', 'C': LOGISTIC_C, 'C_tuned': False, 'class_weight': 'balanced', 'solver': 'liblinear'}, 'features': {'source': str(PHYSICS_PATH), 'count': len(feature_columns), 'condition_metadata_excluded': True, 'train_only_median_imputation': True, 'train_only_standard_scaling': True}, 'duplicate_policy': 'median aggregate duplicate fault_raw+condition reference rows; original dataset labels are not changed', 'statistics': {'bootstrap': {'cluster': 'condition', 'repeats': BOOTSTRAP_REPEATS}, 'permutation': {'type': 'paired within-condition score-label swap', 'repeats': PERMUTATION_REPEATS}}, 'interpretation': 'High OTHER_PARTNERS_TRANSFER AUC supports partner-transferable constituent evidence. A large gap between WITHIN_PAIR_ORACLE and OTHER_PARTNERS_TRANSFER supports partner-dependent interaction/evidence geometry.'}
    with open(RESULT_DIR / 'experiment_config.json', 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

def print_primary_decision(results, overall):
    print('\n' + '=' * 150)
    print('PRIMARY CROSS-PARTNER TRANSFER SUMMARY')
    print('=' * 150)
    primary = results[(results['protocol'] == 'OTHER_PARTNERS_TRANSFER') & (results['condition_mode'] == 'SEEN_CONDITION')].copy()
    strongest = results[(results['protocol'] == 'OTHER_PARTNERS_TRANSFER') & (results['condition_mode'] == 'UNSEEN_CONDITION')].copy()
    reference = results[results['protocol'] == 'WITHIN_PAIR_ORACLE_LOCO'].copy()
    display_columns = ['target_composition', 'constituent_A', 'partner_B', 'auc', 'ci95_low', 'ci95_high', 'permutation_p', 'n_train_contexts']
    print('\nOTHER_PARTNERS_TRANSFER — SEEN CONDITION')
    print(primary[display_columns].to_string(index=False))
    print('\nOTHER_PARTNERS_TRANSFER — UNSEEN CONDITION')
    print(strongest[display_columns].to_string(index=False))
    print('\nOVERALL')
    print(overall.to_string(index=False))
    if len(primary) > 0:
        median_seen = float(primary['auc'].median())
        min_seen = float(primary['auc'].min())
        high_seen = int((primary['auc'] >= 0.8).sum())
        print(f'\nPrimary transfer median AUC : {median_seen:.3f}')
        print(f'Primary transfer min AUC    : {min_seen:.3f}')
        print(f'AUC >= 0.80 directed tests  : {high_seen}/{len(primary)}')
    if len(primary) > 0 and len(reference) > 0:
        merged = primary[['target_composition', 'constituent_A', 'auc']].merge(reference[['target_composition', 'constituent_A', 'auc']], on=['target_composition', 'constituent_A'], suffixes=('_cross_partner', '_within_pair'))
        merged['transfer_gap'] = merged['auc_within_pair'] - merged['auc_cross_partner']
        print('\nWITHIN-PAIR minus CROSS-PARTNER AUC GAP:')
        print(merged[['target_composition', 'constituent_A', 'auc_cross_partner', 'auc_within_pair', 'transfer_gap']].to_string(index=False))
        print(f"\nMedian transfer gap: {merged['transfer_gap'].median():.3f}")
    print('\nDECISION GUIDE:')
    print('  If OTHER_PARTNERS_TRANSFER stays high (especially under UNSEEN_CONDITION), partner-invariant constituent evidence is genuinely transferable and a better disentangling decoder is justified.')
    print('  If WITHIN_PAIR_ORACLE is high but OTHER_PARTNERS_TRANSFER collapses, evidence exists but is partner-dependent; the next method should model interaction / conditional residual / explain-away structure instead of forcing one invariant constituent manifold.')

def main():
    print('\n' + '=' * 155)
    print('CROSS-PARTNER TRANSFER ORACLE')
    print('Does A learned from other contexts transfer to held-out A+B?')
    print('=' * 155)
    print(f'\nClassifier C            : {LOGISTIC_C} FIXED')
    print(f'Bootstrap repeats       : {BOOTSTRAP_REPEATS}')
    print(f'Permutation repeats     : {PERMUTATION_REPEATS}')
    data = load_full_data()
    feature_columns = select_feature_columns(data)
    aggregated = build_aggregated_table(data, feature_columns)
    composition_catalog = build_composition_catalog(aggregated)
    directed_catalog = build_directed_test_catalog(aggregated, composition_catalog)
    print('\nDirected tests:')
    print(directed_catalog.to_string(index=False))
    results = run_all_tests(aggregated, composition_catalog, directed_catalog, feature_columns)
    by_constituent, overall = summarize_results(results)
    save_config(feature_columns)
    print_primary_decision(results, overall)
    print('\nResults:')
    print(RESULT_DIR)
    print('\nMost important files:')
    for filename in ['CROSS_PARTNER_RESULTS.csv', 'CROSS_PARTNER_BY_CONSTITUENT.csv', 'CROSS_PARTNER_OVERALL_SUMMARY.csv', 'SCORE_LEVEL_RESULTS.csv', 'TRAIN_CONTEXT_AUDIT.csv', 'experiment_config.json']:
        print(' -', filename)
    print('\nEXPERIMENT COMPLETED.')
if __name__ == '__main__':
    main()
