"""Dataset-consistency and severity-sensitivity analyses for the MCC5-THU study.

The script audits the released dataset structure and evaluates sensitivity to
severity-specific single-fault training support. Low-severity single-fault
examples are removed for selected mechanisms while all compound test runs and
the locked evaluation procedure remain unchanged.

The severity variant is a sensitivity analysis and does not replace the primary
mechanism-level formulation.
"""
from pathlib import Path
import importlib.util
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
ROOT = Path('D:\\Multi-mode Fault Diagnosis Datasets of Three-phase')
FINAL_SCRIPT = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_EVALUATION.py'
LOCKED_DIR = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS'
LOCKED_SUMMARY = LOCKED_DIR / 'SUMMARY_BY_PROTOCOL.csv'
LOCKED_PREDICTIONS = LOCKED_DIR / 'ALL_OUTER_PREDICTIONS.csv'
OUT_DIR = ROOT / 'DATASET_AND_SEVERITY_SENSITIVITY'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEVERITY_DIR = OUT_DIR / 'SEVERITY_ABLATION'
SEVERITY_DIR.mkdir(parents=True, exist_ok=True)
RUN_RELEASE_AUDIT = True
RUN_SEVERITY_ABLATION = True
SKIP_COMPLETED_SEVERITY_ABLATION = True
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 42
DOCUMENTED_TOTAL_RUNS = 282
LOW_SEVERITY_SINGLE_LABELS_TO_REMOVE = ['bearing_inner_L', 'bearing_outer_L', 'static_eccentricity_L', 'winding_L']
COMPOUND_COMPONENTS = ['bearing_inner', 'bearing_outer', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'winding']
ALL_COMPONENTS = ['bearing_ball', 'bearing_inner', 'bearing_outer', 'bend', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'voltage_unbalance', 'winding']
METRICS_FOR_ABLATION = ['mechanism_micro_f1', 'exact_match', 'constituent_recall', 'false_additions_per_run', 'all_constituents_recovered_rate']
EXPECTED_FULL_PREFIXES = ['bearing_ball', 'bearing_inner', 'bearing_outer', 'bend', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'voltage_unbalance', 'winding']

def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Required file not found:\n{path}')

def import_module_from_path(path, module_name):
    require_file(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not import module:\n{path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def metric_value(module, frame, metric):
    values = module.calculate_metrics(frame)
    value = values.get(metric, np.nan)
    try:
        return float(value)
    except Exception:
        return np.nan

def percentile_ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

def compound_support_table(data):
    compound = data[data['fault_role'] == 'compound'].copy()
    rows = []
    for component in ALL_COMPONENTS:
        n_runs = int(compound[component].astype(int).sum())
        n_compositions = int(compound.loc[compound[component].astype(int) == 1, 'composition_key'].nunique())
        rows.append({'component': component, 'compound_run_support': n_runs, 'compound_composition_degree': n_compositions, 'appears_in_compound': bool(n_runs > 0)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / 'COMPOUND_CONSTITUENT_SUPPORT.csv', index=False, encoding='utf-8-sig')
    return result

def single_fault_grid_audit(data):
    singles = data[data['fault_role'] == 'single'].copy()
    all_conditions = sorted(data['condition_key'].astype(str).unique().tolist())
    observed = singles.groupby(['fault_raw', 'condition_key'], dropna=False).size().rename('n_runs').reset_index()
    fault_labels = sorted(singles['fault_raw'].astype(str).unique().tolist())
    full_grid = pd.MultiIndex.from_product([fault_labels, all_conditions], names=['fault_raw', 'condition_key']).to_frame(index=False)
    grid = full_grid.merge(observed, on=['fault_raw', 'condition_key'], how='left')
    grid['n_runs'] = grid['n_runs'].fillna(0).astype(int)
    grid['status'] = np.select([grid['n_runs'] == 0, grid['n_runs'] > 1], ['MISSING', 'DUPLICATE'], default='OK')
    grid.to_csv(OUT_DIR / 'SINGLE_FAULT_GRID_AUDIT.csv', index=False, encoding='utf-8-sig')
    duplicates = grid[grid['status'] == 'DUPLICATE'].copy()
    missing = grid[grid['status'] == 'MISSING'].copy()
    duplicates.to_csv(OUT_DIR / 'DUPLICATE_SINGLE_FAULT_CELLS.csv', index=False, encoding='utf-8-sig')
    missing.to_csv(OUT_DIR / 'MISSING_SINGLE_FAULT_CELLS.csv', index=False, encoding='utf-8-sig')
    return (grid, duplicates, missing)

def token_has_full_fault_prefix(token):
    low = str(token).strip().lower()
    return any((low.startswith(prefix) for prefix in EXPECTED_FULL_PREFIXES))

def compound_label_abbreviation_audit(data):
    compound_labels = sorted(data.loc[data['fault_role'] == 'compound', 'fault_raw'].astype(str).unique().tolist())
    rows = []
    for label in compound_labels:
        parts = str(label).split('_and_')
        bad_parts = [part for part in parts if not token_has_full_fault_prefix(part)]
        subset = data[data['fault_raw'].astype(str) == label]
        first = subset.iloc[0]
        active = [component for component in ALL_COMPONENTS if int(first[component]) == 1]
        rows.append({'fault_raw': label, 'raw_parts': '|'.join(parts), 'abbreviated_or_unrecognized_parts': '|'.join(bad_parts), 'has_abbreviated_or_unrecognized_part': bool(len(bad_parts) > 0), 'authoritative_manifest_constituents': '|'.join(active), 'n_runs': int(len(subset))})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / 'COMPOUND_LABEL_ABBREVIATION_AUDIT.csv', index=False, encoding='utf-8-sig')
    return result

def feature_count_audit(module, definition):
    rows = []
    for component in ALL_COMPONENTS:
        requested = definition.loc[definition['mechanism_key'].astype(str) == component, 'feature'].dropna().astype(str).tolist()
        requested_non_iqr = sorted({feature for feature in requested if not feature.endswith('__iqr')})
        rows.append({'component': component, 'curated_non_iqr_feature_count': int(len(requested_non_iqr)), 'compound_run_support': np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / 'CURATED_FEATURE_COUNT_AUDIT.csv', index=False, encoding='utf-8-sig')
    return result

def run_release_audit(module):
    print('\n' + '=' * 150)
    print('A) DATASET / RELEASE AUDIT')
    print('=' * 150)
    data, definition = module.load_data()
    support = compound_support_table(data)
    grid, duplicates, missing = single_fault_grid_audit(data)
    label_audit = compound_label_abbreviation_audit(data)
    feature_counts = feature_count_audit(module, definition)
    n_compound_components = int(support['appears_in_compound'].sum())
    summary_rows = [{'item': 'released_run_count', 'value': int(len(data)), 'reference_or_expected': DOCUMENTED_TOTAL_RUNS, 'status': 'MISMATCH' if len(data) != DOCUMENTED_TOTAL_RUNS else 'MATCH'}, {'item': 'compound_run_count', 'value': int((data['fault_role'] == 'compound').sum()), 'reference_or_expected': '', 'status': 'INFO'}, {'item': 'constituents_in_full_vocabulary', 'value': len(ALL_COMPONENTS), 'reference_or_expected': '', 'status': 'INFO'}, {'item': 'constituents_represented_in_compounds', 'value': n_compound_components, 'reference_or_expected': len(COMPOUND_COMPONENTS), 'status': 'PASS' if n_compound_components == len(COMPOUND_COMPONENTS) else 'CHECK'}, {'item': 'duplicate_single_fault_condition_cells', 'value': int(len(duplicates)), 'reference_or_expected': 0, 'status': 'PASS' if len(duplicates) == 0 else 'RELEASE_INCONSISTENCY'}, {'item': 'missing_single_fault_condition_cells', 'value': int(len(missing)), 'reference_or_expected': 0, 'status': 'PASS' if len(missing) == 0 else 'RELEASE_INCONSISTENCY'}, {'item': 'compound_labels_with_abbreviated_or_unrecognized_part', 'value': int(label_audit['has_abbreviated_or_unrecognized_part'].sum()), 'reference_or_expected': 0, 'status': 'PASS' if int(label_audit['has_abbreviated_or_unrecognized_part'].sum()) == 0 else 'RELEASE_INCONSISTENCY'}]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / 'DATASET_RELEASE_AUDIT_SUMMARY.csv', index=False, encoding='utf-8-sig')
    print('\nDATASET RELEASE AUDIT SUMMARY')
    print(summary.to_string(index=False))
    print('\nCOMPOUND CONSTITUENT SUPPORT')
    print(support.to_string(index=False))
    if len(duplicates):
        print('\nDUPLICATE SINGLE-FAULT CELLS')
        print(duplicates.to_string(index=False))
    if len(missing):
        print('\nMISSING SINGLE-FAULT CELLS')
        print(missing.to_string(index=False))
    flagged_labels = label_audit[label_audit['has_abbreviated_or_unrecognized_part']]
    if len(flagged_labels):
        print('\nFLAGGED COMPOUND LABELS')
        print(flagged_labels.to_string(index=False))
    return (data, definition, summary, support, feature_counts)

def build_ablation_load_data(original_load_data):

    def load_data_ablation():
        data, definition = original_load_data()
        data = data.copy()
        remove_mask = (data['fault_role'] == 'single') & data['fault_raw'].astype(str).isin(LOW_SEVERITY_SINGLE_LABELS_TO_REMOVE)
        removed = data.loc[remove_mask, [column for column in ['run_id', 'fault_raw', 'condition_key', 'old_split'] if column in data.columns]].copy()
        removed.to_csv(OUT_DIR / 'SEVERITY_ABLATION_REMOVED_RUNS.csv', index=False, encoding='utf-8-sig')
        filtered = data.loc[~remove_mask].copy().reset_index(drop=True)
        n_compound_before = int((data['fault_role'] == 'compound').sum())
        n_compound_after = int((filtered['fault_role'] == 'compound').sum())
        if n_compound_before != n_compound_after:
            raise RuntimeError('Severity ablation accidentally changed compound test support.')
        print('\n' + '=' * 130)
        print('SEVERITY ABLATION DATA FILTER')
        print('=' * 130)
        print('Removed LOW-severity single-fault support only:')
        for label in LOW_SEVERITY_SINGLE_LABELS_TO_REMOVE:
            count = int((data['fault_raw'].astype(str) == label).sum())
            print(f'  {label:32s}: {count:3d} runs')
        print(f'\nRuns before  : {len(data)}')
        print(f'Runs after   : {len(filtered)}')
        print(f'Compound test: {n_compound_after} runs (UNCHANGED)')
        return (filtered, definition)
    return load_data_ablation

def build_ablation_save_config(module):

    def save_config(data, catalog, fold_definitions, feature_map):
        config = {'experiment': 'severity_support_sensitivity', 'variant': 'HIGH_ONLY_SINGLE_SUPPORT_FOR_COMPOUND_SEVERITY_MECHANISMS', 'scientific_question': 'Does merging low-severity single-fault examples into the mechanism-level positive class materially affect crossed compound constituent generalization?', 'locked_primary_reference': str(LOCKED_DIR), 'removed_low_severity_single_labels': LOW_SEVERITY_SINGLE_LABELS_TO_REMOVE, 'compound_test_runs_changed': False, 'classifier': 'same locked L2 LogisticRegression, C=1, balanced', 'features': 'same curated mechanism-specific physics features', 'threshold_calibration': 'same grouped inner-CV by condition, F1 primary, balanced-accuracy tie, nearest-0.5 tie', 'outer_protocol': 'same crossed recombination / one-sided / two-sided context x seen/unseen-condition protocol', 'sensitivity_analysis': True, 'primary_method_replaced': False, 'outer_test_used_for_variant_selection': False, 'n_runs_in_ablation': int(len(data)), 'n_compound_runs': int((data['fault_role'] == 'compound').sum()), 'n_outer_folds': int(len(fold_definitions)), 'n_features_per_component': {component: int(len(feature_map[component])) for component in module.COMPONENTS}, 'interpretation_rule': 'Use only as sensitivity analysis. Do not select the severity definition from final outer-test performance.'}
        with open(module.RESULT_DIR / 'experiment_config.json', 'w', encoding='utf-8') as file:
            json.dump(config, file, indent=4, ensure_ascii=False)
    return save_config

def run_severity_ablation(module):
    summary_path = SEVERITY_DIR / 'SUMMARY_BY_PROTOCOL.csv'
    if SKIP_COMPLETED_SEVERITY_ABLATION and summary_path.exists():
        print('\n' + '=' * 150)
        print('B) SEVERITY ABLATION ALREADY COMPLETED -> SKIP RERUN')
        print('=' * 150)
        return
    print('\n' + '=' * 150)
    print('B) SEVERITY-MERGING ABLATION')
    print('=' * 150)
    original_result_dir = module.RESULT_DIR
    original_load_data = module.load_data
    original_save_config = module.save_config
    try:
        module.RESULT_DIR = SEVERITY_DIR
        module.load_data = build_ablation_load_data(original_load_data)
        module.save_config = build_ablation_save_config(module)
        t0 = time.time()
        module.main()
        elapsed_minutes = (time.time() - t0) / 60.0
        (SEVERITY_DIR / 'runtime_minutes.txt').write_text(f'{elapsed_minutes:.6f}\n', encoding='utf-8')
        print(f'\nSeverity ablation completed in {elapsed_minutes:.2f} min.')
    finally:
        module.RESULT_DIR = original_result_dir
        module.load_data = original_load_data
        module.save_config = original_save_config

def build_severity_summary_comparison():
    require_file(LOCKED_SUMMARY)
    severity_summary_path = SEVERITY_DIR / 'SUMMARY_BY_PROTOCOL.csv'
    require_file(severity_summary_path)
    primary = pd.read_csv(LOCKED_SUMMARY)
    ablation = pd.read_csv(severity_summary_path)
    primary['severity_training_variant'] = 'MERGED_HL_PRIMARY'
    ablation['severity_training_variant'] = 'HIGH_ONLY_SINGLE_SUPPORT'
    combined = pd.concat([primary, ablation], ignore_index=True, sort=False)
    keep_columns = ['severity_training_variant', 'condition_mode', 'context_regime', 'n_prediction_rows', 'n_unique_runs', 'mechanism_active_macro_f1', 'mechanism_micro_f1', 'exact_match', 'constituent_recall', 'false_additions_per_run', 'any_constituent_recovered_rate', 'all_constituents_recovered_rate']
    combined[keep_columns].to_csv(OUT_DIR / 'SEVERITY_ABLATION_COMPARISON.csv', index=False, encoding='utf-8-sig')
    print('\nSEVERITY ABLATION COMPARISON')
    print(combined[keep_columns].to_string(index=False))
    return combined

def identify_alignment_keys(primary, ablation):
    candidates = ['run_id', 'condition_mode', 'context_regime', 'target_composition', 'context_unseen_component', 'context_seen_component', 'heldout_condition']
    keys = [column for column in candidates if column in primary.columns and column in ablation.columns]
    if 'run_id' not in keys:
        raise RuntimeError('run_id not available for paired severity bootstrap.')
    if 'condition_mode' not in keys:
        raise RuntimeError('condition_mode not available for paired severity bootstrap.')
    if 'context_regime' not in keys:
        raise RuntimeError('context_regime not available for paired severity bootstrap.')
    return keys

def verify_prediction_alignment(primary, ablation, keys):
    p_keys = primary[keys].astype(str).agg('||'.join, axis=1)
    a_keys = ablation[keys].astype(str).agg('||'.join, axis=1)
    if p_keys.duplicated().any():
        duplicated = primary.loc[p_keys.duplicated(keep=False), keys]
        raise RuntimeError('Primary prediction alignment key is not unique:\n' + duplicated.head(30).to_string(index=False))
    if a_keys.duplicated().any():
        duplicated = ablation.loc[a_keys.duplicated(keep=False), keys]
        raise RuntimeError('Ablation prediction alignment key is not unique:\n' + duplicated.head(30).to_string(index=False))
    set_p = set(p_keys.tolist())
    set_a = set(a_keys.tolist())
    if set_p != set_a:
        only_p = sorted(set_p - set_a)[:20]
        only_a = sorted(set_a - set_p)[:20]
        raise RuntimeError(f'Primary and severity-ablation prediction rows do not align.\nOnly primary examples: {only_p}\nOnly ablation examples: {only_a}')

def paired_bootstrap_severity(module):
    require_file(LOCKED_PREDICTIONS)
    ablation_path = SEVERITY_DIR / 'ALL_OUTER_PREDICTIONS.csv'
    require_file(ablation_path)
    primary = pd.read_csv(LOCKED_PREDICTIONS)
    ablation = pd.read_csv(ablation_path)
    keys = identify_alignment_keys(primary, ablation)
    verify_prediction_alignment(primary, ablation, keys)
    primary = primary.copy()
    ablation = ablation.copy()
    primary['__align__'] = primary[keys].astype(str).agg('||'.join, axis=1)
    ablation['__align__'] = ablation[keys].astype(str).agg('||'.join, axis=1)
    primary = primary.sort_values('__align__').reset_index(drop=True)
    ablation = ablation.sort_values('__align__').reset_index(drop=True)
    if not np.array_equal(primary['__align__'].to_numpy(), ablation['__align__'].to_numpy()):
        raise RuntimeError('Alignment failed after sorting.')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    condition_modes = sorted(primary['condition_mode'].astype(str).unique().tolist())
    context_regimes = sorted(primary['context_regime'].astype(str).unique().tolist())
    for condition_mode in condition_modes:
        for context_regime in context_regimes:
            mask = (primary['condition_mode'].astype(str) == condition_mode) & (primary['context_regime'].astype(str) == context_regime)
            p = primary.loc[mask].copy().reset_index(drop=True)
            a = ablation.loc[mask].copy().reset_index(drop=True)
            unique_runs = sorted(p['run_id'].astype(str).unique().tolist())
            n_runs = len(unique_runs)
            if n_runs == 0:
                continue
            run_to_indices = {run_id: np.flatnonzero(p['run_id'].astype(str).to_numpy() == run_id) for run_id in unique_runs}
            for metric in METRICS_FOR_ABLATION:
                primary_point = metric_value(module, p, metric)
                ablation_point = metric_value(module, a, metric)
                point_difference = ablation_point - primary_point
                bootstrap_differences = []
                for _ in range(BOOTSTRAP_REPEATS):
                    sampled_runs = rng.choice(unique_runs, size=n_runs, replace=True)
                    p_frames = []
                    a_frames = []
                    for sampled_run in sampled_runs:
                        idx = run_to_indices[str(sampled_run)]
                        p_frames.append(p.iloc[idx])
                        a_frames.append(a.iloc[idx])
                    p_boot = pd.concat(p_frames, ignore_index=True)
                    a_boot = pd.concat(a_frames, ignore_index=True)
                    p_metric = metric_value(module, p_boot, metric)
                    a_metric = metric_value(module, a_boot, metric)
                    if np.isfinite(p_metric) and np.isfinite(a_metric):
                        bootstrap_differences.append(a_metric - p_metric)
                ci_low, ci_high = percentile_ci(bootstrap_differences)
                rows.append({'condition_mode': condition_mode, 'context_regime': context_regime, 'metric': metric, 'n_unique_physical_runs': n_runs, 'merged_HL_primary': primary_point, 'high_only_single_support': ablation_point, 'difference_Honly_minus_merged': point_difference, 'ci95_low': ci_low, 'ci95_high': ci_high, 'ci_excludes_zero': bool(np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_low > 0 or ci_high < 0)), 'bootstrap_cluster': 'run_id', 'bootstrap_repeats': BOOTSTRAP_REPEATS})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / 'SEVERITY_ABLATION_PAIRED_BOOTSTRAP.csv', index=False, encoding='utf-8-sig')
    print('\nPAIRED SEVERITY-ABLATION BOOTSTRAP')
    display = result[result['metric'].isin(['mechanism_micro_f1', 'constituent_recall', 'false_additions_per_run', 'exact_match', 'all_constituents_recovered_rate'])]
    print(display.to_string(index=False))
    return result

def save_master_config():
    config = {'script': 'RUN_REMAINING_PAPER_ANALYSES.py', 'run_release_audit': RUN_RELEASE_AUDIT, 'run_severity_ablation': RUN_SEVERITY_ABLATION, 'severity_ablation': {'primary': 'mechanism-level H/L merged single-fault support', 'ablation': 'remove L single-fault runs only for severity-coded mechanisms participating in compounds', 'removed_fault_raw_labels': LOW_SEVERITY_SINGLE_LABELS_TO_REMOVE, 'compound_test_set_changed': False, 'sensitivity_analysis_only': True, 'do_not_replace_locked_primary': True}, 'paired_bootstrap': {'cluster': 'run_id', 'repeats': BOOTSTRAP_REPEATS, 'seed': BOOTSTRAP_SEED}, 'release_audit': {'documented_total_runs': DOCUMENTED_TOTAL_RUNS, 'compound_claim_scope': COMPOUND_COMPONENTS}}
    with open(OUT_DIR / 'analysis_config.json', 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

def main():
    print('\n' + '=' * 155)
    print('DATASET CONSISTENCY AND SEVERITY SENSITIVITY')
    print('=' * 155)
    module = import_module_from_path(FINAL_SCRIPT, 'final_crossed_for_remaining_analyses')
    if RUN_RELEASE_AUDIT:
        run_release_audit(module)
    if RUN_SEVERITY_ABLATION:
        run_severity_ablation(module)
        build_severity_summary_comparison()
        paired_bootstrap_severity(module)
    save_master_config()
    print('\n' + '=' * 155)
    print('COMPLETED')
    print('=' * 155)
    print(f'\nResults:\n{OUT_DIR}')
    print('\nMost important outputs:')
    for name in ['DATASET_RELEASE_AUDIT_SUMMARY.csv', 'COMPOUND_CONSTITUENT_SUPPORT.csv', 'DUPLICATE_SINGLE_FAULT_CELLS.csv', 'MISSING_SINGLE_FAULT_CELLS.csv', 'COMPOUND_LABEL_ABBREVIATION_AUDIT.csv', 'CURATED_FEATURE_COUNT_AUDIT.csv', 'SEVERITY_ABLATION_COMPARISON.csv', 'SEVERITY_ABLATION_PAIRED_BOOTSTRAP.csv', 'SEVERITY_ABLATION_REMOVED_RUNS.csv', 'analysis_config.json']:
        print(' -', name)
    print('\nInterpretation rule:')
    print('  The H-only result is a sensitivity analysis.')
    print('  The locked H/L-merged primary method remains the primary analysis.')
if __name__ == '__main__':
    main()
