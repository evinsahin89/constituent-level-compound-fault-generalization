"""Single-fault learnability analysis for constituent-level compound diagnosis.

The analysis evaluates whether constituents with weak compound recovery are
already difficult to recognize in isolation or whether degradation emerges in
compound context. Isolated single faults are evaluated against healthy runs
and other non-compound faults using leave-one-operating-condition-out
evaluation.

The locked mechanism-specific features, logistic-regression specification, and
grouped threshold-calibration procedure are reused. Compound runs are not used
for fitting or threshold calibration.
"""
from pathlib import Path
import importlib.util
import json
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
warnings.filterwarnings('ignore')
ROOT = Path('D:\\Multi-mode Fault Diagnosis Datasets of Three-phase')
FINAL_SCRIPT = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_EVALUATION.py'
FINAL_RESULT_DIR = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS'
COMPOUND_COMPONENT_SUMMARY = FINAL_RESULT_DIR / 'SUMMARY_BY_COMPONENT.csv'
OUT_DIR = ROOT / 'SINGLE_FAULT_LEARNABILITY_RESULTS'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONTRASTS = ['SINGLE_VS_HEALTH', 'SINGLE_VS_OTHER_NONCOMPOUND']
COMPONENTS = ['bearing_ball', 'bearing_inner', 'bearing_outer', 'bend', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'voltage_unbalance', 'winding']
DISPLAY_NAMES = {'bearing_ball': 'Bearing ball', 'bearing_inner': 'Bearing inner', 'bearing_outer': 'Bearing outer', 'bend': 'Shaft bending', 'broken_bar': 'Broken bar', 'dynamic_eccentricity': 'Dynamic eccentricity', 'static_eccentricity': 'Static eccentricity', 'voltage_unbalance': 'Voltage unbalance', 'winding': 'Winding'}

def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Required file not found:\n{path}')

def import_module_from_path(path, module_name):
    require_file(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not import:\n{path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def safe_auc(y_true, probability):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    finite = np.isfinite(p)
    y = y[finite]
    p = p[finite]
    if len(y) == 0 or len(np.unique(y)) != 2:
        return np.nan
    return float(roc_auc_score(y, p))

def safe_auprc(y_true, probability):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    finite = np.isfinite(p)
    y = y[finite]
    p = p[finite]
    if len(y) == 0 or y.sum() == 0:
        return np.nan
    return float(average_precision_score(y, p))

def percentile_ci(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))

def get_contrast_frame(data, component, contrast):
    """
    Return ONLY non-compound observations belonging to the requested
    single-fault diagnostic contrast.

    Target is stored in __target__.
    """
    noncompound = data[data['fault_role'].isin(['health', 'single'])].copy()
    positive = (noncompound['fault_role'] == 'single') & (noncompound[component].astype(int) == 1)
    if contrast == 'SINGLE_VS_HEALTH':
        negative = noncompound['fault_role'] == 'health'
    elif contrast == 'SINGLE_VS_OTHER_NONCOMPOUND':
        negative = ~positive
    else:
        raise ValueError(f'Unknown contrast: {contrast}')
    frame = noncompound.loc[positive | negative].copy().reset_index(drop=True)
    frame['__target__'] = frame[component].astype(int)
    if frame.loc[frame['__target__'] == 1, 'fault_role'].astype(str).ne('single').any():
        raise RuntimeError(f'{component}/{contrast}: positive set contains non-single rows.')
    if frame['fault_role'].eq('compound').any():
        raise RuntimeError(f'{component}/{contrast}: compound row leaked into single-fault analysis.')
    return frame

def run_single_fault_analysis(module, data, feature_map, raw_matrices):
    prediction_rows = []
    condition_rows = []
    all_conditions = sorted(data['condition_key'].astype(str).unique().tolist())
    row_lookup = {str(run_id): idx for idx, run_id in enumerate(data['run_id'].astype(str).tolist())}
    for contrast in CONTRASTS:
        print('\n' + '=' * 150)
        print(contrast)
        print('=' * 150)
        for component in COMPONENTS:
            frame = get_contrast_frame(data, component, contrast)
            print(f"\n{component:24s} rows={len(frame):3d} positive={int(frame['__target__'].sum()):3d}")
            for heldout_condition in all_conditions:
                test_df = frame[frame['condition_key'].astype(str) == heldout_condition].copy()
                train_df = frame[frame['condition_key'].astype(str) != heldout_condition].copy()
                y_test = test_df['__target__'].astype(int).to_numpy()
                y_train = train_df['__target__'].astype(int).to_numpy()
                if len(test_df) == 0:
                    continue
                if len(np.unique(y_test)) != 2:
                    print(f'  SKIP {heldout_condition}: test cell does not contain both classes')
                    continue
                if len(np.unique(y_train)) != 2:
                    print(f'  SKIP {heldout_condition}: outer train does not contain both classes')
                    continue
                train_indices = np.asarray([row_lookup[str(run_id)] for run_id in train_df['run_id'].astype(str)], dtype=int)
                test_indices = np.asarray([row_lookup[str(run_id)] for run_id in test_df['run_id'].astype(str)], dtype=int)
                X_train = raw_matrices[component][train_indices]
                X_test = raw_matrices[component][test_indices]
                train_groups = train_df['condition_key'].astype(str).to_numpy()
                threshold_info = module.inner_grouped_threshold(X_train, y_train, train_groups)
                threshold = float(threshold_info['threshold'])
                fitted = module.fit_binary_detector(X_train, y_train)
                probability = module.predict_binary_detector(fitted, X_test)
                prediction = (probability >= threshold).astype(int)
                condition_auc = safe_auc(y_test, probability)
                condition_auprc = safe_auprc(y_test, probability)
                precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(y_test, prediction, labels=[1], zero_division=0)
                condition_rows.append({'contrast': contrast, 'component': component, 'display_name': DISPLAY_NAMES[component], 'heldout_condition': heldout_condition, 'n_train': int(len(train_df)), 'n_train_positive': int(y_train.sum()), 'n_test': int(len(test_df)), 'n_test_positive': int(y_test.sum()), 'threshold': threshold, 'threshold_status': threshold_info.get('status', ''), 'inner_folds_requested': threshold_info.get('inner_folds_requested', np.nan), 'inner_folds_valid': threshold_info.get('inner_folds_valid', np.nan), 'precision': float(precision_arr[0]), 'recall': float(recall_arr[0]), 'f1': float(f1_arr[0]), 'balanced_accuracy': float(balanced_accuracy_score(y_test, prediction)), 'auroc': condition_auc, 'auprc': condition_auprc})
                for local_idx, (index, row) in enumerate(test_df.reset_index(drop=True).iterrows()):
                    prediction_rows.append({'contrast': contrast, 'component': component, 'display_name': DISPLAY_NAMES[component], 'heldout_condition': heldout_condition, 'run_id': str(row['run_id']), 'fault_raw': str(row['fault_raw']), 'fault_role': str(row['fault_role']), 'truth': int(y_test[local_idx]), 'probability': float(probability[local_idx]), 'prediction': int(prediction[local_idx]), 'threshold': threshold})
    predictions = pd.DataFrame(prediction_rows)
    conditions = pd.DataFrame(condition_rows)
    predictions.to_csv(OUT_DIR / 'SINGLE_FAULT_OUTER_PREDICTIONS.csv', index=False, encoding='utf-8-sig')
    conditions.to_csv(OUT_DIR / 'SINGLE_FAULT_BY_CONDITION.csv', index=False, encoding='utf-8-sig')
    return (predictions, conditions)

def summarize_single_fault(predictions, conditions):
    rows = []
    for (contrast, component), group in predictions.groupby(['contrast', 'component'], sort=True):
        y = group['truth'].astype(int).to_numpy()
        pred = group['prediction'].astype(int).to_numpy()
        prob = group['probability'].astype(float).to_numpy()
        precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
        pooled_auc = safe_auc(y, prob)
        pooled_auprc = safe_auprc(y, prob)
        condition_sub = conditions[(conditions['contrast'] == contrast) & (conditions['component'] == component)]
        condition_auc_values = pd.to_numeric(condition_sub['auroc'], errors='coerce').dropna().to_numpy(dtype=float)
        condition_auc_lo, condition_auc_hi = percentile_ci(condition_auc_values)
        condition_auprc_values = pd.to_numeric(condition_sub['auprc'], errors='coerce').dropna().to_numpy(dtype=float)
        prevalence = float(y.mean())
        always_positive_f1 = float(f1_score(y, np.ones(len(y), dtype=int), zero_division=0))
        rows.append({'contrast': contrast, 'component': component, 'display_name': DISPLAY_NAMES[component], 'n_outer_rows': int(len(group)), 'n_positive': int(y.sum()), 'n_negative': int(len(y) - y.sum()), 'n_conditions': int(group['heldout_condition'].nunique()), 'prevalence': prevalence, 'always_positive_f1': always_positive_f1, 'precision': float(precision_arr[0]), 'recall': float(recall_arr[0]), 'f1': float(f1_arr[0]), 'f1_minus_always_positive': float(f1_arr[0] - always_positive_f1), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'pooled_oof_auroc': pooled_auc, 'auc_sep': max(pooled_auc, 1.0 - pooled_auc) if np.isfinite(pooled_auc) else np.nan, 'pooled_oof_auprc': pooled_auprc, 'auprc_minus_prevalence': pooled_auprc - prevalence if np.isfinite(pooled_auprc) else np.nan, 'mean_condition_auc': float(np.mean(condition_auc_values)) if len(condition_auc_values) else np.nan, 'median_condition_auc': float(np.median(condition_auc_values)) if len(condition_auc_values) else np.nan, 'condition_auc_min': float(np.min(condition_auc_values)) if len(condition_auc_values) else np.nan, 'condition_auc_max': float(np.max(condition_auc_values)) if len(condition_auc_values) else np.nan, 'condition_auc_empirical_2p5': condition_auc_lo, 'condition_auc_empirical_97p5': condition_auc_hi, 'mean_condition_auprc': float(np.mean(condition_auprc_values)) if len(condition_auprc_values) else np.nan, 'mean_threshold': float(pd.to_numeric(group['threshold'], errors='coerce').mean())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / 'SINGLE_FAULT_BY_CONTRAST_COMPONENT.csv', index=False, encoding='utf-8-sig')
    return result

def load_compound_summary():
    require_file(COMPOUND_COMPONENT_SUMMARY)
    df = pd.read_csv(COMPOUND_COMPONENT_SUMMARY)
    required = {'component', 'precision', 'recall', 'f1'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f'SUMMARY_BY_COMPONENT.csv missing columns:\n{missing}')
    df = df[df['component'].isin(COMPONENTS)].copy()
    rows = []
    for component, group in df.groupby('component', sort=True):
        rows.append({'component': component, 'compound_n_scenarios': int(len(group)), 'compound_mean_precision': float(pd.to_numeric(group['precision'], errors='coerce').mean()), 'compound_mean_recall': float(pd.to_numeric(group['recall'], errors='coerce').mean()), 'compound_mean_f1': float(pd.to_numeric(group['f1'], errors='coerce').mean())})
    return pd.DataFrame(rows)

def build_single_to_compound_comparison(single_summary):
    compound = load_compound_summary()
    health = single_summary[single_summary['contrast'] == 'SINGLE_VS_HEALTH'].copy()
    specificity = single_summary[single_summary['contrast'] == 'SINGLE_VS_OTHER_NONCOMPOUND'].copy()
    keep = ['component', 'recall', 'f1', 'pooled_oof_auroc', 'auc_sep', 'pooled_oof_auprc', 'mean_condition_auc']
    health = health[keep].rename(columns={'recall': 'single_vs_health_recall', 'f1': 'single_vs_health_f1', 'pooled_oof_auroc': 'single_vs_health_auroc', 'auc_sep': 'single_vs_health_auc_sep', 'pooled_oof_auprc': 'single_vs_health_auprc', 'mean_condition_auc': 'single_vs_health_mean_condition_auc'})
    specificity = specificity[keep].rename(columns={'recall': 'single_specificity_recall', 'f1': 'single_specificity_f1', 'pooled_oof_auroc': 'single_specificity_auroc', 'auc_sep': 'single_specificity_auc_sep', 'pooled_oof_auprc': 'single_specificity_auprc', 'mean_condition_auc': 'single_specificity_mean_condition_auc'})
    result = health.merge(specificity, on='component', how='outer', validate='one_to_one').merge(compound, on='component', how='left', validate='one_to_one')
    result['display_name'] = result['component'].map(DISPLAY_NAMES)
    result['compound_recall_drop_from_single_specificity'] = result['compound_mean_recall'] - result['single_specificity_recall']
    result['compound_f1_drop_from_single_specificity'] = result['compound_mean_f1'] - result['single_specificity_f1']
    interpretations = []
    for _, row in result.iterrows():
        health_auc = row['single_vs_health_auroc']
        specificity_auc = row['single_specificity_auroc']
        compound_recall = row['compound_mean_recall']
        if pd.notna(health_auc) and health_auc >= 0.8 and pd.notna(specificity_auc) and (specificity_auc >= 0.8) and pd.notna(compound_recall) and (compound_recall < 0.4):
            label = 'strong_isolated_learning_but_poor_compound_recovery'
        elif pd.notna(health_auc) and health_auc >= 0.8 and pd.notna(specificity_auc) and (specificity_auc < 0.7):
            label = 'detectable_vs_health_but_low_single_fault_specificity'
        elif pd.notna(health_auc) and health_auc < 0.7:
            label = 'weak_isolated_detectability_with_curated_representation'
        else:
            label = 'mixed_or_intermediate_pattern'
        interpretations.append(label)
    result['descriptive_pattern'] = interpretations
    order = ['component', 'display_name', 'single_vs_health_recall', 'single_vs_health_f1', 'single_vs_health_auroc', 'single_vs_health_auprc', 'single_specificity_recall', 'single_specificity_f1', 'single_specificity_auroc', 'single_specificity_auprc', 'compound_mean_precision', 'compound_mean_recall', 'compound_mean_f1', 'compound_recall_drop_from_single_specificity', 'compound_f1_drop_from_single_specificity', 'descriptive_pattern']
    result = result[[c for c in order if c in result.columns]]
    result.to_csv(OUT_DIR / 'SINGLE_TO_COMPOUND_COMPARISON.csv', index=False, encoding='utf-8-sig')
    return result

def save_feature_audit(feature_map):
    rows = []
    for component in COMPONENTS:
        features = feature_map[component]
        for feature in features:
            rows.append({'component': component, 'display_name': DISPLAY_NAMES[component], 'n_features': len(features), 'feature': feature})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'FEATURE_AUDIT.csv', index=False, encoding='utf-8-sig')

def save_config(module):
    config = {'analysis': 'single_fault_learnability_vs_compound_recovery', 'status': 'descriptive_diagnostic', 'primary_model_changed': False, 'outer_protocol': 'leave_one_operating_condition_out', 'training_roles': 'health_and_single_only', 'compound_rows_used_for_single_fault_training': False, 'contrasts': {'SINGLE_VS_HEALTH': 'target constituent isolated single-fault positive versus healthy negative', 'SINGLE_VS_OTHER_NONCOMPOUND': 'target constituent isolated single-fault positive versus health and all other isolated single-fault negatives'}, 'feature_source': 'same CURATED_SIGNATURE_DEFINITION.csv as locked primary', 'classifier': {'type': 'L2 LogisticRegression', 'C': float(module.LOGISTIC_C), 'class_weight': 'balanced', 'solver': 'liblinear'}, 'inner_threshold_calibration': {'method': 'GroupKFold grouped by condition_key', 'max_folds': int(module.INNER_GROUP_FOLDS), 'primary': 'F1', 'secondary': 'balanced accuracy', 'final_tiebreak': 'distance to 0.5', 'outer_test_used': False}, 'interpretation_guardrail': 'This analysis diagnoses whether weak compound recovery originates in isolated detectability, single-fault specificity, or compound-context transfer. It is not used for model selection.'}
    with open(OUT_DIR / 'analysis_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def print_final_table(comparison):
    show_columns = ['display_name', 'single_vs_health_auroc', 'single_specificity_auroc', 'single_specificity_recall', 'single_specificity_f1', 'compound_mean_recall', 'compound_mean_f1', 'compound_recall_drop_from_single_specificity', 'descriptive_pattern']
    show = comparison[[c for c in show_columns if c in comparison.columns]].copy()
    for column in show.columns:
        if column in {'display_name', 'descriptive_pattern'}:
            continue
        show[column] = pd.to_numeric(show[column], errors='coerce').round(4)
    print('\n' + '=' * 180)
    print('SINGLE-FAULT LEARNABILITY -> COMPOUND RECOVERY')
    print('=' * 180)
    print(show.to_string(index=False))

def main():
    print('\n' + '=' * 150)
    print('SINGLE-FAULT LEARNABILITY VS COMPOUND RECOVERY')
    print('=' * 150)
    print('Loading exact locked final-pipeline implementation...')
    module = import_module_from_path(FINAL_SCRIPT, 'locked_crossed_module_single_fault_audit')
    module.RESULT_DIR = OUT_DIR
    data, definition = module.load_data()
    feature_map, raw_matrices = module.build_feature_map_and_raw_matrices(data, definition)
    save_feature_audit(feature_map)
    predictions, conditions = run_single_fault_analysis(module, data, feature_map, raw_matrices)
    summary = summarize_single_fault(predictions, conditions)
    comparison = build_single_to_compound_comparison(summary)
    save_config(module)
    print_final_table(comparison)
    print('\n' + '=' * 150)
    print('DONE')
    print('=' * 150)
    print('\nOutputs:')
    print(OUT_DIR)
    print('\nMost important file:')
    print(OUT_DIR / 'SINGLE_TO_COMPOUND_COMPARISON.csv')
    print('\nKey interpretation:')
    print('  High single_vs_health AUROC + high single_specificity AUROC + low compound recall')
    print('      -> isolated constituent is learnable, but its evidence does not transfer reliably into compound context.')
    print('\n  High single_vs_health AUROC + low single_specificity AUROC')
    print('      -> evidence exists versus health, but curated features are not sufficiently constituent-specific.')
    print('\n  Low single_vs_health AUROC')
    print('      -> weakness already exists at isolated-fault representation level.')
if __name__ == '__main__':
    main()
