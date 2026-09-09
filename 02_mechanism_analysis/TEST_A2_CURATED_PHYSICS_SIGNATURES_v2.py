"""Curated physics-signature preservation analysis for the MCC5-THU motor dataset.

Predefined mechanism-specific feature families are used to quantify whether
physical evidence observed in isolated single faults is preserved when the
same constituent appears in a compound fault. Signature direction and scale
are estimated from healthy and single-fault data only.

The analysis uses previously extracted run-level physics features and does not
retrain the primary diagnostic models or modify labels or predictions.
"""
from pathlib import Path
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
ROOT = Path('D:\\Multi-mode Fault Diagnosis Datasets of Three-phase')
RESULT_DIR = ROOT / 'PHYSICS_DIAGNOSTICS_RESULTS'
FEATURE_PATH = RESULT_DIR / 'RUN_PHYSICS_FEATURES.csv'
MAPPING_PATH = RESULT_DIR / 'compound_constituent_mapping.csv'
OUT_DEFINITION = RESULT_DIR / 'CURATED_SIGNATURE_DEFINITION.csv'
OUT_SINGLE_VALIDATION = RESULT_DIR / 'CURATED_SINGLE_SIGNATURE_VALIDATION.csv'
OUT_FEATURE_DETAIL = RESULT_DIR / 'TEST_A2_CURATED_FEATURE_DETAIL.csv'
OUT_FEATURE_SUMMARY = RESULT_DIR / 'TEST_A2_CURATED_FEATURE_SUMMARY.csv'
OUT_SCORE_DETAIL = RESULT_DIR / 'TEST_A2_CURATED_SCORE_DETAIL.csv'
OUT_SCORE_SUMMARY = RESULT_DIR / 'TEST_A2_CURATED_SCORE_SUMMARY.csv'
OUT_FINAL = RESULT_DIR / 'TEST_A2_CURATED_FINAL_SUMMARY.json'
OUT_DUPLICATE_AUDIT = RESULT_DIR / 'DUPLICATE_FAULT_CONDITION_AUDIT.csv'
OUT_CONDITION_DETAIL = RESULT_DIR / 'TEST_A2_CONDITION_BY_CONDITION.csv'
OUT_CONDITION_FAILURES = RESULT_DIR / 'TEST_A2_CONDITION_FAILURES.csv'
OUT_CONDITION_SUMMARY = RESULT_DIR / 'TEST_A2_CONDITION_DEPENDENCE_SUMMARY.csv'
CONDITION_PLOT_DIR = RESULT_DIR / 'TEST_A2_CONDITION_PLOTS'
EPS = 1e-12
MIN_STANDARDIZED_SINGLE_EVIDENCE = 0.1
HALF_PRESERVATION_THRESHOLD = 0.5
CURATED_RULES = {'bearing_inner': {'family': 'BPFI envelope-order family', 'patterns': ['^vib_env_(?:h|a|v|mean)_(?:BPFI|2BPFI)_(?:bandpower|logratio)$']}, 'bearing_outer': {'family': 'BPFO envelope-order family', 'patterns': ['^vib_env_(?:h|a|v|mean)_(?:BPFO|2BPFO)_(?:bandpower|logratio)$']}, 'bearing_ball': {'family': 'BSF envelope-order family', 'patterns': ['^vib_env_(?:h|a|v|mean)_(?:BSF|2BSF)_(?:bandpower|logratio)$']}, 'bend': {'family': 'shaft 1x/2x raw-vibration family', 'patterns': ['^vib_raw_(?:h|a|v|mean)_shaft_(?:1x|2x)_(?:bandpower|logratio)$']}, 'broken_bar': {'family': 'electrical carrier sideband proxy family', 'patterns': ['^current_(?:phase_mean|spacevec|A|B|C)_carrier_(?:minus|plus)_(?:1x|2x)_(?:bandpower|logratio)$']}, 'dynamic_eccentricity': {'family': 'shaft-order + electrical carrier sideband family', 'patterns': ['^vib_raw_(?:h|a|v|mean)_shaft_(?:1x|2x)_(?:bandpower|logratio)$', '^current_(?:phase_mean|spacevec|A|B|C)_carrier_(?:minus|plus)_(?:1x|2x)_(?:bandpower|logratio)$']}, 'static_eccentricity': {'family': 'electrical carrier sideband + shaft-order family', 'patterns': ['^current_(?:phase_mean|spacevec|A|B|C)_carrier_(?:minus|plus)_(?:1x|2x)_(?:bandpower|logratio)$', '^vib_raw_(?:h|a|v|mean)_shaft_(?:1x|2x)_(?:bandpower|logratio)$']}, 'voltage_unbalance': {'family': 'negative-sequence + phase RMS imbalance family', 'patterns': ['^current_negative_sequence_ratio$', '^current_rms_cv$']}, 'winding': {'family': 'negative-sequence + electrical second-harmonic family', 'patterns': ['^current_negative_sequence_ratio$', '^current_rms_cv$', '^current_(?:phase_mean|spacevec|A|B|C)_second_harmonic_(?:bandpower|logratio)$']}}

def mechanism_key(fault_raw):
    fault_raw = str(fault_raw)
    if fault_raw.startswith('bearing_inner'):
        return 'bearing_inner'
    if fault_raw.startswith('bearing_outer'):
        return 'bearing_outer'
    if fault_raw.startswith('bearing_ball'):
        return 'bearing_ball'
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

def robust_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < EPS:
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale < EPS:
        scale = np.std(values)
    if not np.isfinite(scale) or scale < EPS:
        magnitude = np.median(np.abs(values))
        scale = max(magnitude * 0.001, 1e-09)
    return float(scale)

def matched_row(feature_df, fault_raw, condition_key):
    """
    Return one representative row for a fault-condition cell.

    If exactly one run exists, return it unchanged.

    If multiple runs exist for the SAME original dataset label and
    SAME operating condition, DO NOT discard the condition. Instead,
    aggregate all numeric physics features with the median.

    This fixes cells such as the duplicated bearing_outer_H condition
    without relabeling, renaming, or deleting any original run.

    Non-numeric metadata are taken from the first row; run_id is replaced
    by a joined audit string and n_aggregated_runs records multiplicity.
    """
    hit = feature_df[(feature_df['fault_raw'] == fault_raw) & (feature_df['condition_key'] == condition_key)].copy()
    if len(hit) == 0:
        return None
    if len(hit) == 1:
        row = hit.iloc[0].copy()
        row['n_aggregated_runs'] = 1
        return row
    representative = hit.iloc[0].copy()
    numeric_cols = hit.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        values = pd.to_numeric(hit[col], errors='coerce')
        if values.notna().any():
            representative[col] = float(values.median())
    representative['run_id'] = '|'.join(hit['run_id'].astype(str).tolist())
    representative['n_aggregated_runs'] = int(len(hit))
    return representative

def save_duplicate_fault_condition_audit(feature_df):
    """
    Audit duplicate original-label × condition cells.

    This is descriptive only. No run is removed and no label is changed.
    """
    grouped = feature_df.groupby(['fault_raw', 'condition_key'], dropna=False).agg(n_runs=('run_id', 'size'), run_ids=('run_id', lambda x: '|'.join(x.astype(str)))).reset_index()
    duplicate = grouped[grouped['n_runs'] > 1].copy().sort_values(['fault_raw', 'condition_key']).reset_index(drop=True)
    duplicate.to_csv(OUT_DUPLICATE_AUDIT, index=False, encoding='utf-8-sig')
    print('\n' + '=' * 100)
    print('DUPLICATE FAULT × CONDITION AUDIT')
    print('=' * 100)
    if len(duplicate) == 0:
        print('No duplicate fault-condition cells.')
    else:
        print(duplicate.to_string(index=False))
        print('\nNOTE: duplicated cells are median-aggregated for matched single/health reference rows.')
    return duplicate

def detect_health_label(feature_df):
    health = feature_df.loc[feature_df['fault_role'] == 'health', 'fault_raw'].dropna().unique().tolist()
    if len(health) != 1:
        raise RuntimeError(f'Expected exactly one health label, found: {health}')
    return str(health[0])

def select_curated_features(feature_df, fault_raw):
    key = mechanism_key(fault_raw)
    if key is None or key not in CURATED_RULES:
        return (None, [])
    rule = CURATED_RULES[key]
    selected = []
    for column in feature_df.columns:
        if column.endswith('__iqr'):
            continue
        if not pd.api.types.is_numeric_dtype(feature_df[column]):
            continue
        for pattern in rule['patterns']:
            if re.search(pattern, column):
                selected.append(column)
                break
    selected = sorted(set(selected))
    return (rule['family'], selected)

def load_inputs():
    print('\n' + '=' * 100)
    print('LOAD INPUTS')
    print('=' * 100)
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f'\nRUN_PHYSICS_FEATURES.csv not found.\nRun DIAGNOSE_COMPOSITIONAL_PHYSICS_v2.py first. \nExpected file:\n{FEATURE_PATH}')
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f'\ncompound_constituent_mapping.csv not found.\nRun DIAGNOSE_COMPOSITIONAL_PHYSICS_v2.py first. \nExpected file:\n{MAPPING_PATH}')
    feature_df = pd.read_csv(FEATURE_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    required_feature_cols = ['run_id', 'fault_raw', 'fault_role', 'condition_key']
    missing = [col for col in required_feature_cols if col not in feature_df.columns]
    if missing:
        raise RuntimeError(f'RUN_PHYSICS_FEATURES.csv missing columns: {missing}')
    required_mapping_cols = ['compound_fault_raw', 'constituent_A', 'constituent_B', 'status']
    missing = [col for col in required_mapping_cols if col not in mapping_df.columns]
    if missing:
        raise RuntimeError(f'Mapping CSV missing columns: {missing}')
    feature_df['run_id'] = feature_df['run_id'].astype(str)
    feature_df['fault_raw'] = feature_df['fault_raw'].astype(str)
    feature_df['condition_key'] = feature_df['condition_key'].astype(str)
    resolved = mapping_df[mapping_df['status'] == 'RESOLVED'].copy()
    if len(resolved) == 0:
        raise RuntimeError('No resolved compound mappings.')
    print(f'Runs             : {len(feature_df)}')
    print(f"Fault classes    : {feature_df['fault_raw'].nunique()}")
    print(f"Conditions       : {feature_df['condition_key'].nunique()}")
    print(f'Resolved compounds: {len(resolved)}')
    return (feature_df, resolved)

def build_signature_definition(feature_df, mapping_df):
    single_faults = sorted(feature_df.loc[feature_df['fault_role'] == 'single', 'fault_raw'].unique().tolist())
    compound_constituents = set(mapping_df['constituent_A'].dropna()) | set(mapping_df['constituent_B'].dropna())
    rows = []
    signature_map = {}
    for fault in single_faults:
        family, features = select_curated_features(feature_df, fault)
        signature_map[fault] = {'family': family, 'features': features}
        if len(features) == 0:
            rows.append({'fault_raw': fault, 'mechanism_key': mechanism_key(fault), 'signature_family': family, 'feature': None, 'used_in_compound_test': fault in compound_constituents})
        else:
            for feature in features:
                rows.append({'fault_raw': fault, 'mechanism_key': mechanism_key(fault), 'signature_family': family, 'feature': feature, 'used_in_compound_test': fault in compound_constituents})
    definition = pd.DataFrame(rows)
    definition.to_csv(OUT_DEFINITION, index=False, encoding='utf-8-sig')
    print('\n' + '=' * 100)
    print('CURATED SIGNATURE DEFINITION')
    print('=' * 100)
    summary = definition.groupby(['fault_raw', 'signature_family'], dropna=False).agg(n_features=('feature', lambda x: int(x.notna().sum())), used_in_compound_test=('used_in_compound_test', 'max')).reset_index()
    print(summary.to_string(index=False))
    empty = summary[summary['n_features'] == 0]
    if len(empty) > 0:
        raise RuntimeError(f"\nCurated rule matched zero features for: {empty['fault_raw'].tolist()}")
    return (signature_map, definition)

def fit_single_signature_parameters(feature_df, health_label, signature_map):
    """
    For every fault-specific curated feature:

    direction_f =
        sign(median(single_f - health_f))
        over matched operating conditions

    scale_f =
        robust scale of HEALTH feature across conditions

    This uses only HEALTH + SINGLE-FAULT data.
    No compound values are used to define direction or scale.
    """
    parameter_rows = []
    single_validation_rows = []
    parameters = {}
    health_df = feature_df[feature_df['fault_raw'] == health_label]
    for fault, spec in signature_map.items():
        features = spec['features']
        if len(features) == 0:
            continue
        fault_df = feature_df[feature_df['fault_raw'] == fault]
        matched_conditions = sorted(set(health_df['condition_key']) & set(fault_df['condition_key']))
        if len(matched_conditions) == 0:
            continue
        parameters[fault] = {}
        condition_scores = []
        for feature in features:
            diffs = []
            health_values = []
            valid_conditions = []
            for condition in matched_conditions:
                row_h = matched_row(feature_df, health_label, condition)
                row_s = matched_row(feature_df, fault, condition)
                if row_h is None or row_s is None:
                    continue
                try:
                    h = float(row_h[feature])
                    s = float(row_s[feature])
                except Exception:
                    continue
                if not np.all(np.isfinite([h, s])):
                    continue
                diffs.append(s - h)
                health_values.append(h)
                valid_conditions.append(condition)
            diffs = np.asarray(diffs, dtype=float)
            health_values = np.asarray(health_values, dtype=float)
            if len(diffs) == 0:
                continue
            median_diff = float(np.median(diffs))
            if median_diff > 0:
                direction = 1.0
            elif median_diff < 0:
                direction = -1.0
            else:
                direction = 1.0
            scale = robust_scale(health_values)
            if not np.isfinite(scale) or scale <= 0:
                continue
            directed_diffs = direction * diffs
            sign_consistency = float(np.mean(directed_diffs > 0))
            median_standardized_single = float(np.median(directed_diffs / scale))
            parameters[fault][feature] = {'direction': direction, 'scale': scale, 'median_single_minus_health': median_diff, 'sign_consistency': sign_consistency, 'median_standardized_single': median_standardized_single}
            parameter_rows.append({'fault_raw': fault, 'signature_family': spec['family'], 'feature': feature, 'n_matched_conditions': len(diffs), 'direction': int(direction), 'healthy_robust_scale': scale, 'median_single_minus_health': median_diff, 'single_sign_consistency': sign_consistency, 'median_standardized_single_evidence': median_standardized_single})
        for condition in matched_conditions:
            row_h = matched_row(feature_df, health_label, condition)
            row_s = matched_row(feature_df, fault, condition)
            if row_h is None or row_s is None:
                continue
            evidence = []
            for feature, param in parameters[fault].items():
                try:
                    h = float(row_h[feature])
                    s = float(row_s[feature])
                except Exception:
                    continue
                if not np.all(np.isfinite([h, s])):
                    continue
                e = param['direction'] * (s - h) / param['scale']
                if np.isfinite(e):
                    evidence.append(e)
            if len(evidence) == 0:
                continue
            condition_scores.append({'fault_raw': fault, 'condition_key': condition, 'signature_family': spec['family'], 'n_features_used': len(evidence), 'single_signature_score': float(np.median(evidence))})
        fault_scores = pd.DataFrame([x for x in condition_scores if x['fault_raw'] == fault])
        if len(fault_scores) > 0:
            scores = fault_scores['single_signature_score'].to_numpy(dtype=float)
            single_validation_rows.append({'fault_raw': fault, 'signature_family': spec['family'], 'n_curated_features': len(parameters[fault]), 'n_conditions': len(scores), 'median_single_signature_score': float(np.median(scores)), 'mean_single_signature_score': float(np.mean(scores)), 'positive_signature_condition_rate': float(np.mean(scores > 0)), 'strong_signature_condition_rate': float(np.mean(scores >= 0.5))})
    parameter_df = pd.DataFrame(parameter_rows)
    single_validation = pd.DataFrame(single_validation_rows)
    single_validation.to_csv(OUT_SINGLE_VALIDATION, index=False, encoding='utf-8-sig')
    definition = pd.read_csv(OUT_DEFINITION)
    definition = definition.merge(parameter_df, on=['fault_raw', 'signature_family', 'feature'], how='left')
    definition.to_csv(OUT_DEFINITION, index=False, encoding='utf-8-sig')
    print('\n' + '=' * 100)
    print('CURATED SINGLE-FAULT SIGNATURE VALIDATION')
    print('=' * 100)
    print(single_validation.to_string(index=False))
    return (parameters, single_validation)

def test_a2_feature_level(feature_df, mapping_df, health_label, signature_map, parameters):
    rows = []
    compound_rows = feature_df[feature_df['fault_role'] == 'compound']
    mapping = {}
    for _, row in mapping_df.iterrows():
        mapping[str(row['compound_fault_raw'])] = (str(row['constituent_A']), str(row['constituent_B']))
    for _, compound_row in compound_rows.iterrows():
        compound_label = str(compound_row['fault_raw'])
        condition = str(compound_row['condition_key'])
        if compound_label not in mapping:
            continue
        constituent_a, constituent_b = mapping[compound_label]
        row_h = matched_row(feature_df, health_label, condition)
        if row_h is None:
            continue
        for constituent in [constituent_a, constituent_b]:
            if constituent not in parameters:
                continue
            row_s = matched_row(feature_df, constituent, condition)
            if row_s is None:
                continue
            family = signature_map[constituent]['family']
            for feature, param in parameters[constituent].items():
                try:
                    h = float(row_h[feature])
                    single = float(row_s[feature])
                    compound = float(compound_row[feature])
                except Exception:
                    continue
                if not np.all(np.isfinite([h, single, compound])):
                    continue
                direction = float(param['direction'])
                scale = float(param['scale'])
                single_evidence = direction * (single - h) / scale
                compound_evidence = direction * (compound - h) / scale
                if abs(single_evidence) >= MIN_STANDARDIZED_SINGLE_EVIDENCE:
                    preservation_ratio = compound_evidence / single_evidence
                else:
                    preservation_ratio = np.nan
                rows.append({'compound_run_id': compound_row['run_id'], 'compound_fault_raw': compound_label, 'condition_key': condition, 'constituent': constituent, 'signature_family': family, 'feature': feature, 'direction': int(direction), 'healthy_value': h, 'single_value': single, 'compound_value': compound, 'single_evidence_z': single_evidence, 'compound_evidence_z': compound_evidence, 'preservation_ratio': preservation_ratio, 'positive_single_evidence': bool(single_evidence > 0), 'positive_compound_evidence': bool(compound_evidence > 0), 'half_evidence_preserved': bool(single_evidence > 0 and compound_evidence >= HALF_PRESERVATION_THRESHOLD * single_evidence)})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_FEATURE_DETAIL, index=False, encoding='utf-8-sig')
    if len(detail) == 0:
        raise RuntimeError('Test A2 feature-level detail is empty.')
    valid = detail[detail['positive_single_evidence']].copy()
    summary = valid.groupby(['compound_fault_raw', 'constituent', 'signature_family']).agg(n_feature_condition=('feature', 'size'), n_unique_features=('feature', 'nunique'), n_conditions=('condition_key', 'nunique'), median_single_evidence_z=('single_evidence_z', 'median'), median_compound_evidence_z=('compound_evidence_z', 'median'), median_preservation_ratio=('preservation_ratio', 'median'), q25_preservation_ratio=('preservation_ratio', lambda x: np.nanpercentile(x, 25)), q75_preservation_ratio=('preservation_ratio', lambda x: np.nanpercentile(x, 75)), positive_compound_evidence_rate=('positive_compound_evidence', 'mean'), half_evidence_preserved_rate=('half_evidence_preserved', 'mean')).reset_index()
    summary.to_csv(OUT_FEATURE_SUMMARY, index=False, encoding='utf-8-sig')
    return (detail, summary)

def test_a2_signature_score(feature_df, mapping_df, health_label, signature_map, parameters):
    """
    A fault-specific signature score is the MEDIAN standardized
    directed evidence across its curated features.

    For each matched condition:

        single_score =
            median_f[
                direction_f *
                (single_f - health_f) /
                scale_f
            ]

        compound_score =
            median_f[
                direction_f *
                (compound_f - health_f) /
                scale_f
            ]

    Then:
        preservation_ratio =
            compound_score / single_score
    """
    rows = []
    mapping = {str(row['compound_fault_raw']): (str(row['constituent_A']), str(row['constituent_B'])) for _, row in mapping_df.iterrows()}
    compound_rows = feature_df[feature_df['fault_role'] == 'compound']
    for _, compound_row in compound_rows.iterrows():
        compound_label = str(compound_row['fault_raw'])
        condition = str(compound_row['condition_key'])
        if compound_label not in mapping:
            continue
        row_h = matched_row(feature_df, health_label, condition)
        if row_h is None:
            continue
        constituent_a, constituent_b = mapping[compound_label]
        for constituent in [constituent_a, constituent_b]:
            if constituent not in parameters:
                continue
            row_s = matched_row(feature_df, constituent, condition)
            if row_s is None:
                continue
            single_feature_evidence = []
            compound_feature_evidence = []
            feature_names = []
            for feature, param in parameters[constituent].items():
                try:
                    h = float(row_h[feature])
                    s = float(row_s[feature])
                    ab = float(compound_row[feature])
                except Exception:
                    continue
                if not np.all(np.isfinite([h, s, ab])):
                    continue
                direction = float(param['direction'])
                scale = float(param['scale'])
                e_single = direction * (s - h) / scale
                e_compound = direction * (ab - h) / scale
                if not (np.isfinite(e_single) and np.isfinite(e_compound)):
                    continue
                single_feature_evidence.append(e_single)
                compound_feature_evidence.append(e_compound)
                feature_names.append(feature)
            if len(single_feature_evidence) == 0:
                continue
            single_score = float(np.median(single_feature_evidence))
            compound_score = float(np.median(compound_feature_evidence))
            if single_score > MIN_STANDARDIZED_SINGLE_EVIDENCE:
                ratio = compound_score / single_score
            else:
                ratio = np.nan
            rows.append({'compound_run_id': compound_row['run_id'], 'compound_fault_raw': compound_label, 'condition_key': condition, 'constituent': constituent, 'signature_family': signature_map[constituent]['family'], 'n_features_used': len(feature_names), 'single_signature_score': single_score, 'compound_signature_score': compound_score, 'preservation_ratio': ratio, 'positive_single_signature': bool(single_score > 0), 'positive_compound_signature': bool(compound_score > 0), 'half_signature_preserved': bool(single_score > 0 and compound_score >= HALF_PRESERVATION_THRESHOLD * single_score)})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_SCORE_DETAIL, index=False, encoding='utf-8-sig')
    if len(detail) == 0:
        raise RuntimeError('Test A2 signature-score detail is empty.')
    valid = detail[detail['positive_single_signature']].copy()
    summary = valid.groupby(['compound_fault_raw', 'constituent', 'signature_family']).agg(n_conditions=('condition_key', 'nunique'), median_n_features_used=('n_features_used', 'median'), median_single_signature_score=('single_signature_score', 'median'), median_compound_signature_score=('compound_signature_score', 'median'), median_preservation_ratio=('preservation_ratio', 'median'), q25_preservation_ratio=('preservation_ratio', lambda x: np.nanpercentile(x, 25)), q75_preservation_ratio=('preservation_ratio', lambda x: np.nanpercentile(x, 75)), positive_compound_signature_rate=('positive_compound_signature', 'mean'), half_signature_preserved_rate=('half_signature_preserved', 'mean')).reset_index()
    summary.to_csv(OUT_SCORE_SUMMARY, index=False, encoding='utf-8-sig')
    return (detail, summary)

def _condition_metadata_table(feature_df):
    """
    Build one metadata row per condition.
    """
    wanted = ['condition_key']
    for candidate in ['mode', 'torque_nm', 'rpm_nominal']:
        if candidate in feature_df.columns:
            wanted.append(candidate)
    meta = feature_df[wanted].drop_duplicates().copy()
    meta = meta.sort_values('condition_key').drop_duplicates('condition_key').reset_index(drop=True)
    pattern = re.compile('^(speed_circulation|torque_circulation)__T(\\d+(?:\\.\\d+)?)__R(\\d+(?:\\.\\d+)?)$', flags=re.IGNORECASE)
    if 'mode' not in meta.columns:
        meta['mode'] = np.nan
    if 'torque_nm' not in meta.columns:
        meta['torque_nm'] = np.nan
    if 'rpm_nominal' not in meta.columns:
        meta['rpm_nominal'] = np.nan
    for idx, row in meta.iterrows():
        key = str(row['condition_key'])
        match = pattern.match(key)
        if match is None:
            continue
        if pd.isna(row['mode']):
            meta.at[idx, 'mode'] = match.group(1).lower()
        if pd.isna(row['torque_nm']):
            meta.at[idx, 'torque_nm'] = float(match.group(2))
        if pd.isna(row['rpm_nominal']):
            meta.at[idx, 'rpm_nominal'] = float(match.group(3))
    return meta

def _safe_filename(text_value):
    text_value = str(text_value)
    text_value = re.sub('[^A-Za-z0-9._-]+', '_', text_value)
    return text_value.strip('_')

def analyze_condition_dependence(score_detail, feature_df):
    """
    Quantify whether constituent preservation depends on operating condition.

    Input:
        TEST_A2 signature-score detail

    Outputs:
        - one row per compound × constituent × condition
        - explicit low-preservation/failure cases
        - per compound-constituent condition-dependence summary
        - one line plot per compound
        - one global preservation heatmap

    IMPORTANT:
        This is descriptive. No threshold is tuned on P3/P4 and no model
        evaluation data are used for training.
    """
    condition_meta = _condition_metadata_table(feature_df)
    detail = score_detail.merge(condition_meta, on='condition_key', how='left').copy()
    detail['eligible_for_preservation'] = detail['positive_single_signature'] & np.isfinite(detail['preservation_ratio'])
    eligible = detail[detail['eligible_for_preservation']].copy()
    eligible['condition_label'] = eligible['mode'].astype(str).str.replace('_circulation', '', regex=False) + ' | T=' + eligible['torque_nm'].round().astype('Int64').astype(str) + ' | R=' + eligible['rpm_nominal'].round().astype('Int64').astype(str)
    eligible['preservation_state'] = np.select([eligible['compound_signature_score'] <= 0, eligible['preservation_ratio'] < 0.5, eligible['preservation_ratio'] < 1.0], ['SIGN_REVERSED_OR_LOST', 'SEVERE_SUPPRESSION', 'PARTIAL_SUPPRESSION'], default='PRESERVED_OR_ENHANCED')
    eligible = eligible.sort_values(['compound_fault_raw', 'constituent', 'mode', 'torque_nm', 'rpm_nominal']).reset_index(drop=True)
    eligible.to_csv(OUT_CONDITION_DETAIL, index=False, encoding='utf-8-sig')
    failures = eligible[(eligible['compound_signature_score'] <= 0) | (eligible['preservation_ratio'] < 0.5)].copy().sort_values(['preservation_ratio', 'compound_fault_raw', 'constituent']).reset_index(drop=True)
    failures.to_csv(OUT_CONDITION_FAILURES, index=False, encoding='utf-8-sig')
    summary_rows = []
    for (compound, constituent, family), group in eligible.groupby(['compound_fault_raw', 'constituent', 'signature_family']):
        group = group.sort_values(['mode', 'torque_nm', 'rpm_nominal'])
        ratios = group['preservation_ratio'].to_numpy(dtype=float)
        worst_idx = group['preservation_ratio'].idxmin()
        best_idx = group['preservation_ratio'].idxmax()
        worst_row = group.loc[worst_idx]
        best_row = group.loc[best_idx]
        q25 = float(np.nanpercentile(ratios, 25))
        q75 = float(np.nanpercentile(ratios, 75))
        summary_rows.append({'compound_fault_raw': compound, 'constituent': constituent, 'signature_family': family, 'n_conditions': int(len(group)), 'median_preservation_ratio': float(np.nanmedian(ratios)), 'mean_preservation_ratio': float(np.nanmean(ratios)), 'std_preservation_ratio': float(np.nanstd(ratios, ddof=1)) if len(ratios) > 1 else 0.0, 'q25_preservation_ratio': q25, 'q75_preservation_ratio': q75, 'iqr_preservation_ratio': q75 - q25, 'min_preservation_ratio': float(np.nanmin(ratios)), 'max_preservation_ratio': float(np.nanmax(ratios)), 'condition_span': float(np.nanmax(ratios) - np.nanmin(ratios)), 'positive_condition_rate': float(np.mean(group['compound_signature_score'] > 0)), 'half_preserved_condition_rate': float(np.mean(group['preservation_ratio'] >= 0.5)), 'fully_preserved_condition_rate': float(np.mean(group['preservation_ratio'] >= 1.0)), 'n_severe_suppression_conditions': int(np.sum(group['preservation_ratio'] < 0.5)), 'n_sign_reversed_or_lost_conditions': int(np.sum(group['compound_signature_score'] <= 0)), 'worst_condition': str(worst_row['condition_key']), 'worst_preservation_ratio': float(worst_row['preservation_ratio']), 'best_condition': str(best_row['condition_key']), 'best_preservation_ratio': float(best_row['preservation_ratio'])})
    summary = pd.DataFrame(summary_rows).sort_values(['median_preservation_ratio', 'condition_span'], ascending=[True, False]).reset_index(drop=True)
    summary.to_csv(OUT_CONDITION_SUMMARY, index=False, encoding='utf-8-sig')
    CONDITION_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for compound, group in eligible.groupby('compound_fault_raw'):
        condition_order = group[['condition_key', 'condition_label', 'mode', 'torque_nm', 'rpm_nominal']].drop_duplicates().sort_values(['mode', 'torque_nm', 'rpm_nominal']).reset_index(drop=True)
        ordered_keys = condition_order['condition_key'].tolist()
        labels = condition_order['condition_label'].tolist()
        x = np.arange(len(ordered_keys))
        plt.figure(figsize=(max(10, len(ordered_keys) * 0.9), 6))
        for constituent, sub in group.groupby('constituent'):
            value_map = sub.set_index('condition_key')['preservation_ratio'].to_dict()
            y = np.array([value_map.get(key, np.nan) for key in ordered_keys], dtype=float)
            plt.plot(x, y, marker='o', linewidth=2, label=str(constituent))
        plt.axhline(1.0, linestyle='--', linewidth=1)
        plt.axhline(0.5, linestyle=':', linewidth=1)
        plt.xticks(x, labels, rotation=60, ha='right')
        plt.ylabel('Preservation ratio')
        plt.xlabel('Operating condition')
        plt.title('Curated constituent evidence preservation\n' + str(compound))
        plt.legend()
        plt.tight_layout()
        figure_path = CONDITION_PLOT_DIR / (_safe_filename(compound) + '_condition_preservation.png')
        plt.savefig(figure_path, dpi=220, bbox_inches='tight')
        plt.close()
    heat = eligible.assign(row_key=eligible['compound_fault_raw'] + ' | ' + eligible['constituent']).pivot_table(index='row_key', columns='condition_key', values='preservation_ratio', aggfunc='median')
    condition_order = condition_meta.sort_values(['mode', 'torque_nm', 'rpm_nominal'])['condition_key'].tolist()
    heat = heat.reindex(columns=[col for col in condition_order if col in heat.columns])
    plt.figure(figsize=(max(12, 0.8 * len(heat.columns)), max(8, 0.42 * len(heat.index))))
    image = plt.imshow(heat.to_numpy(dtype=float), aspect='auto')
    plt.colorbar(image, label='Preservation ratio')
    plt.xticks(np.arange(len(heat.columns)), heat.columns, rotation=60, ha='right')
    plt.yticks(np.arange(len(heat.index)), heat.index)
    plt.xlabel('Operating condition')
    plt.ylabel('Compound | constituent')
    plt.title('Condition-dependent curated signature preservation')
    plt.tight_layout()
    plt.savefig(CONDITION_PLOT_DIR / 'GLOBAL_CONDITION_PRESERVATION_HEATMAP.png', dpi=220, bbox_inches='tight')
    plt.close()
    print('\n' + '=' * 120)
    print('CONDITION-DEPENDENT SIGNATURE PRESERVATION')
    print('=' * 120)
    display_cols = ['compound_fault_raw', 'constituent', 'n_conditions', 'median_preservation_ratio', 'condition_span', 'half_preserved_condition_rate', 'n_severe_suppression_conditions', 'n_sign_reversed_or_lost_conditions', 'worst_condition', 'worst_preservation_ratio']
    print(summary[display_cols].to_string(index=False))
    print(f'\nSevere suppression / sign-loss rows: {len(failures)}')
    return (eligible, failures, summary)

def save_final_summary(feature_df, mapping_df, single_validation, feature_summary, score_summary, condition_summary):
    summary = {'analysis': 'curated_physics_signature_preservation', 'n_runs': int(len(feature_df)), 'n_conditions': int(feature_df['condition_key'].nunique()), 'n_resolved_compound_labels': int(len(mapping_df)), 'n_curated_single_faults': int(len(single_validation)), 'single_signature_validation': {'median_positive_signature_condition_rate': float(single_validation['positive_signature_condition_rate'].median()), 'median_strong_signature_condition_rate': float(single_validation['strong_signature_condition_rate'].median()), 'median_single_signature_score': float(single_validation['median_single_signature_score'].median())}, 'feature_level_preservation': {'median_of_constituent_median_preservation_ratio': float(feature_summary['median_preservation_ratio'].median()), 'mean_positive_compound_evidence_rate': float(feature_summary['positive_compound_evidence_rate'].mean()), 'mean_half_evidence_preserved_rate': float(feature_summary['half_evidence_preserved_rate'].mean())}, 'signature_score_preservation': {'median_of_constituent_median_preservation_ratio': float(score_summary['median_preservation_ratio'].median()), 'mean_positive_compound_signature_rate': float(score_summary['positive_compound_signature_rate'].mean()), 'mean_half_signature_preserved_rate': float(score_summary['half_signature_preserved_rate'].mean())}, 'condition_dependence': {'median_condition_span': float(condition_summary['condition_span'].median()), 'max_condition_span': float(condition_summary['condition_span'].max()), 'mean_half_preserved_condition_rate': float(condition_summary['half_preserved_condition_rate'].mean()), 'total_severe_suppression_conditions': int(condition_summary['n_severe_suppression_conditions'].sum()), 'total_sign_reversed_or_lost_conditions': int(condition_summary['n_sign_reversed_or_lost_conditions'].sum())}}
    with open(OUT_FINAL, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    return summary

def main():
    print('\n' + '=' * 100)
    print('CURATED PHYSICS SIGNATURE PRESERVATION')
    print('=' * 100)
    feature_df, mapping_df = load_inputs()
    duplicate_audit = save_duplicate_fault_condition_audit(feature_df)
    health_label = detect_health_label(feature_df)
    print(f'\nHealth label: {health_label}')
    signature_map, definition = build_signature_definition(feature_df, mapping_df)
    parameters, single_validation = fit_single_signature_parameters(feature_df, health_label, signature_map)
    feature_detail, feature_summary = test_a2_feature_level(feature_df, mapping_df, health_label, signature_map, parameters)
    score_detail, score_summary = test_a2_signature_score(feature_df, mapping_df, health_label, signature_map, parameters)
    condition_detail, condition_failures, condition_summary = analyze_condition_dependence(score_detail, feature_df)
    final_summary = save_final_summary(feature_df, mapping_df, single_validation, feature_summary, score_summary, condition_summary)
    print('\n' + '=' * 120)
    print('CURATED FEATURE-LEVEL PRESERVATION')
    print('=' * 120)
    display_feature_cols = ['compound_fault_raw', 'constituent', 'signature_family', 'n_unique_features', 'n_conditions', 'median_preservation_ratio', 'positive_compound_evidence_rate', 'half_evidence_preserved_rate']
    print(feature_summary[display_feature_cols].to_string(index=False))
    print('\n' + '=' * 120)
    print('CURATED SIGNATURE-SCORE PRESERVATION')
    print('=' * 120)
    display_score_cols = ['compound_fault_raw', 'constituent', 'signature_family', 'n_conditions', 'median_single_signature_score', 'median_compound_signature_score', 'median_preservation_ratio', 'positive_compound_signature_rate', 'half_signature_preserved_rate']
    print(score_summary[display_score_cols].to_string(index=False))
    print('\n' + '=' * 100)
    print('FINAL SUMMARY')
    print('=' * 100)
    print(json.dumps(final_summary, indent=4, ensure_ascii=False))
    print('\nResults:')
    print(RESULT_DIR)
    print('\nMost important files:')
    print('1. CURATED_SIGNATURE_DEFINITION.csv')
    print('2. CURATED_SINGLE_SIGNATURE_VALIDATION.csv')
    print('3. TEST_A2_CURATED_FEATURE_SUMMARY.csv')
    print('4. TEST_A2_CURATED_SCORE_SUMMARY.csv')
    print('5. DUPLICATE_FAULT_CONDITION_AUDIT.csv')
    print('6. TEST_A2_CONDITION_BY_CONDITION.csv')
    print('7. TEST_A2_CONDITION_FAILURES.csv')
    print('8. TEST_A2_CONDITION_DEPENDENCE_SUMMARY.csv')
    print('9. TEST_A2_CONDITION_PLOTS/')
    print('10. TEST_A2_CURATED_FINAL_SUMMARY.json')
    print('\nCOMPLETED.')
if __name__ == '__main__':
    main()
