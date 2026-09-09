"""Composition-level statistical analysis of the crossed compound-fault evaluation.

The 108 compound recordings arise from nine distinct fault compositions.
Uncertainty is therefore recomputed with fault composition as the bootstrap
resampling unit. Each sampled composition retains all associated operating
conditions and evaluation cells, and paired contrasts use the same sampled
composition multiplicities.

The analysis reads existing locked outer-test predictions and does not retrain
models, recalibrate thresholds, or modify predictions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path('D:\\yay\u0131nlar\\Multi-mode Fault Diagnosis Datasets of Three-phase')
PRED_PATH = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS' / 'ALL_OUTER_PREDICTIONS.csv'
SUMMARY_PATH = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS' / 'SUMMARY_BY_PROTOCOL.csv'
OLD_CI_PATH = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS' / 'BOOTSTRAP_CI_BY_PROTOCOL.csv'
OUT = ROOT / 'COMPOSITION_LEVEL_AUDIT_RESULTS'
OUT.mkdir(parents=True, exist_ok=True)
COMPONENTS = ['bearing_ball', 'bearing_inner', 'bearing_outer', 'bend', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'voltage_unbalance', 'winding']
ACTIVE_COMPONENTS = ['bearing_inner', 'bearing_outer', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'winding']
RECOMBINATION = 'RECOMBINATION'
ONE_SIDED = 'ONE_SIDED_CONTEXT_ZS'
TWO_SIDED = 'TWO_SIDED_CONTEXT_ZS'
SEEN = 'SEEN_CONDITION'
UNSEEN = 'UNSEEN_CONDITION'
REGIMES = [RECOMBINATION, ONE_SIDED, TWO_SIDED]
CONDITIONS = [SEEN, UNSEEN]
CONTRASTS = [('R_minus_O', RECOMBINATION, ONE_SIDED), ('R_minus_T', RECOMBINATION, TWO_SIDED), ('O_minus_T', ONE_SIDED, TWO_SIDED)]
METRICS = ['mechanism_micro_f1', 'mechanism_active_macro_f1', 'exact_match', 'constituent_recall', 'false_additions_per_run', 'all_constituents_recovered_rate']
N_BOOT = 10000
SEED = 20260814

def binary_f1(y_true, y_pred):
    """
    Binary F1 equivalent to sklearn f1_score(..., zero_division=0).

    Implemented directly here to make the 10,000-bootstrap audit faster.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)

def arrays(df):
    y_true = df[COMPONENTS].to_numpy(dtype=int)
    y_pred = np.column_stack([df[f'pred_{c}'].to_numpy(dtype=int) for c in COMPONENTS])
    return (y_true, y_pred)

def calculate_metrics(df):
    """
    Metric definitions match the locked primary evaluation.
    """
    if len(df) == 0:
        return {metric: np.nan for metric in METRICS}
    y_true, y_pred = arrays(df)
    mechanism_micro_f1 = binary_f1(y_true.reshape(-1), y_pred.reshape(-1))
    active_f1 = []
    for component in ACTIVE_COMPONENTS:
        j = COMPONENTS.index(component)
        active_f1.append(binary_f1(y_true[:, j], y_pred[:, j]))
    mechanism_active_macro_f1 = float(np.mean(active_f1))
    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total_true = int(y_true.sum())
    constituent_recall = tp / total_true if total_true > 0 else np.nan
    false_additions_per_run = float(np.mean(((y_true == 0) & (y_pred == 1)).sum(axis=1)))
    recovered_true = (y_true * y_pred).sum(axis=1)
    true_count = y_true.sum(axis=1)
    all_constituents_recovered_rate = float(np.mean(recovered_true == true_count))
    return {'mechanism_micro_f1': mechanism_micro_f1, 'mechanism_active_macro_f1': mechanism_active_macro_f1, 'exact_match': exact_match, 'constituent_recall': constituent_recall, 'false_additions_per_run': false_additions_per_run, 'all_constituents_recovered_rate': all_constituents_recovered_rate}

def load_predictions():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f'\nLocked prediction file not found:\n{PRED_PATH}\n')
    df = pd.read_csv(PRED_PATH)
    required = {'run_id', 'composition_key', 'condition_key', 'condition_mode', 'context_regime', 'context_unseen_component', *COMPONENTS, *[f'pred_{c}' for c in COMPONENTS]}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError('ALL_OUTER_PREDICTIONS.csv is missing columns:\n' + '\n'.join(missing))
    for c in ['run_id', 'composition_key', 'condition_key', 'condition_mode', 'context_regime']:
        df[c] = df[c].astype(str).str.strip()
    for c in COMPONENTS:
        df[c] = pd.to_numeric(df[c], errors='raise').astype(int)
        df[f'pred_{c}'] = pd.to_numeric(df[f'pred_{c}'], errors='raise').astype(int)
    keep = df['condition_mode'].isin(CONDITIONS) & df['context_regime'].isin(REGIMES)
    df = df.loc[keep].copy()
    if df.empty:
        raise RuntimeError('No crossed compound-evaluation predictions found.')
    return df

def get_cell(df, condition_mode, context_regime):
    return df[(df['condition_mode'] == condition_mode) & (df['context_regime'] == context_regime)]

def coverage_audit(df):
    rows = []
    grouped = df.groupby(['composition_key', 'condition_mode', 'context_regime'], sort=True)
    for (composition, condition_mode, context_regime), g in grouped:
        direction_count = g['context_unseen_component'].dropna().astype(str).nunique()
        rows.append({'composition_key': composition, 'condition_mode': condition_mode, 'context_regime': context_regime, 'n_prediction_rows': len(g), 'n_unique_runs': g['run_id'].nunique(), 'n_conditions': g['condition_key'].nunique(), 'n_one_sided_directions': direction_count})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / 'COMPOSITION_COVERAGE_AUDIT.csv', index=False, encoding='utf-8-sig')
    compositions = sorted(df['composition_key'].unique().tolist())
    print('\n' + '=' * 120)
    print('COMPOSITION COVERAGE AUDIT')
    print('=' * 120)
    print(f'Distinct compositions: {len(compositions)}')
    if len(compositions) != 9:
        raise RuntimeError(f'Expected 9 compositions, found {len(compositions)}.')
    expected_cells = 9 * len(CONDITIONS) * len(REGIMES)
    if len(audit) != expected_cells:
        raise RuntimeError(f'Incomplete composition x condition x regime grid.\nExpected {expected_cells} rows, found {len(audit)}.')
    bad_runs = audit[(audit['n_unique_runs'] != 12) | (audit['n_conditions'] != 12)]
    if not bad_runs.empty:
        raise RuntimeError('Unexpected run/condition coverage:\n' + bad_runs.to_string(index=False))
    bad_standard = audit[audit['context_regime'].isin([RECOMBINATION, TWO_SIDED]) & (audit['n_prediction_rows'] != 12)]
    if not bad_standard.empty:
        raise RuntimeError('Unexpected prediction-row count in Recombination/Two-sided cells:\n' + bad_standard.to_string(index=False))
    bad_one_sided = audit[(audit['context_regime'] == ONE_SIDED) & ((audit['n_prediction_rows'] != 24) | (audit['n_one_sided_directions'] != 2))]
    if not bad_one_sided.empty:
        raise RuntimeError('Unexpected one-sided directional coverage:\n' + bad_one_sided.to_string(index=False))
    print('PASS: 9 compositions x 12 operating conditions are represented consistently across all six cells.')
    print('PASS: one-sided evaluation contains exactly two directional predictions per physical run.')
    return compositions

def full_sample_point_estimates(df):
    rows = []
    point = {}
    for condition_mode in CONDITIONS:
        for context_regime in REGIMES:
            g = get_cell(df, condition_mode, context_regime)
            metrics = calculate_metrics(g)
            point[condition_mode, context_regime] = metrics
            for metric in METRICS:
                rows.append({'condition_mode': condition_mode, 'context_regime': context_regime, 'metric': metric, 'point_estimate': metrics[metric], 'n_prediction_rows': len(g), 'n_unique_runs': g['run_id'].nunique(), 'n_compositions': g['composition_key'].nunique()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'POINT_ESTIMATE_AUDIT.csv', index=False, encoding='utf-8-sig')
    return (point, out)

def verify_against_existing_summary(point_df):
    if not SUMMARY_PATH.exists():
        print('\nExisting SUMMARY_BY_PROTOCOL.csv not found; skipping point-estimate verification.')
        return
    old = pd.read_csv(SUMMARY_PATH)
    rows = []
    max_error = 0.0
    for _, r in point_df.iterrows():
        metric = r['metric']
        match = old[(old['condition_mode'] == r['condition_mode']) & (old['context_regime'] == r['context_regime'])]
        if len(match) != 1:
            raise RuntimeError(f"Could not uniquely match existing protocol summary for:\n{r['condition_mode']} / {r['context_regime']}")
        existing = float(match.iloc[0][metric])
        recomputed = float(r['point_estimate'])
        error = abs(existing - recomputed)
        max_error = max(max_error, error)
        rows.append({'condition_mode': r['condition_mode'], 'context_regime': r['context_regime'], 'metric': metric, 'existing_value': existing, 'recomputed_value': recomputed, 'absolute_difference': error})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / 'POINT_ESTIMATE_EXISTING_RESULT_CHECK.csv', index=False, encoding='utf-8-sig')
    print(f'\nMaximum point-estimate discrepancy vs existing SUMMARY_BY_PROTOCOL.csv: {max_error:.12g}')
    if max_error > 1e-10:
        raise RuntimeError('Recomputed point estimates do not match the locked protocol summary.')
    print('POINT ESTIMATE CHECK: PASS')

def per_composition_metrics(df, compositions):
    rows = []
    for composition in compositions:
        comp_df = df[df['composition_key'] == composition]
        for condition_mode in CONDITIONS:
            for context_regime in REGIMES:
                g = get_cell(comp_df, condition_mode, context_regime)
                metrics = calculate_metrics(g)
                row = {'composition_key': composition, 'condition_mode': condition_mode, 'context_regime': context_regime, 'n_prediction_rows': len(g), 'n_unique_runs': g['run_id'].nunique()}
                row.update(metrics)
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'PER_COMPOSITION_PROTOCOL_METRICS.csv', index=False, encoding='utf-8-sig')
    return out

def calculate_full_sample_contrasts(point):
    contrast_rows = []
    contrast_point = {}
    for condition_mode in CONDITIONS:
        for contrast_name, first_regime, second_regime in CONTRASTS:
            for metric in METRICS:
                value = point[condition_mode, first_regime][metric] - point[condition_mode, second_regime][metric]
                contrast_point[condition_mode, contrast_name, metric] = value
                contrast_rows.append({'condition_mode': condition_mode, 'contrast': contrast_name, 'first_regime': first_regime, 'second_regime': second_regime, 'metric': metric, 'point_estimate': value})
    interaction_rows = []
    interaction_point = {}
    for contrast_name, first_regime, second_regime in CONTRASTS:
        for metric in METRICS:
            seen_delta = contrast_point[SEEN, contrast_name, metric]
            unseen_delta = contrast_point[UNSEEN, contrast_name, metric]
            interaction = unseen_delta - seen_delta
            interaction_point[contrast_name, metric] = interaction
            interaction_rows.append({'contrast': contrast_name, 'first_regime': first_regime, 'second_regime': second_regime, 'metric': metric, 'delta_seen': seen_delta, 'delta_unseen': unseen_delta, 'interaction_unseen_minus_seen': interaction})
    return (contrast_point, interaction_point, pd.DataFrame(contrast_rows), pd.DataFrame(interaction_rows))

def composition_cluster_bootstrap(df, compositions, n_boot=N_BOOT, seed=SEED):
    """
    Nonparametric composition-level cluster bootstrap.

    Sampling unit:
        composition_key

    Each selected composition carries all of its operating-condition
    observations across ALL SIX evaluation cells.

    If a composition is sampled more than once, its entire block is
    deliberately duplicated.

    The same bootstrap composition sample is therefore shared by:
        all protocol cells,
        all paired protocol contrasts,
        all context x condition interactions.
    """
    rng = np.random.default_rng(seed)
    composition_blocks = {composition: df[df['composition_key'] == composition].copy() for composition in compositions}
    boot_rows = []
    print('\n' + '=' * 120)
    print('FULL COMPOSITION-LEVEL CLUSTER BOOTSTRAP')
    print('=' * 120)
    print(f'Composition clusters : {len(compositions)}')
    print('Runs / composition   : 12 physical runs')
    print('One-sided rows       : 24 / composition (two directions)')
    print(f'Bootstrap repeats    : {n_boot}')
    print(f'Seed                 : {seed}')
    for b in range(n_boot):
        sampled = rng.choice(compositions, size=len(compositions), replace=True)
        pieces = []
        for sample_position, composition in enumerate(sampled):
            block = composition_blocks[composition].copy()
            block['_bootstrap_cluster_instance'] = sample_position
            pieces.append(block)
        boot_df = pd.concat(pieces, ignore_index=True)
        row = {'bootstrap_id': b + 1}
        cell_metrics = {}
        for condition_mode in CONDITIONS:
            for context_regime in REGIMES:
                g = get_cell(boot_df, condition_mode, context_regime)
                metrics = calculate_metrics(g)
                cell_metrics[condition_mode, context_regime] = metrics
                for metric in METRICS:
                    col = f'cell__{condition_mode}__{context_regime}__{metric}'
                    row[col] = metrics[metric]
        contrast_values = {}
        for condition_mode in CONDITIONS:
            for contrast_name, first_regime, second_regime in CONTRASTS:
                for metric in METRICS:
                    delta = cell_metrics[condition_mode, first_regime][metric] - cell_metrics[condition_mode, second_regime][metric]
                    contrast_values[condition_mode, contrast_name, metric] = delta
                    col = f'contrast__{condition_mode}__{contrast_name}__{metric}'
                    row[col] = delta
        for contrast_name, first_regime, second_regime in CONTRASTS:
            for metric in METRICS:
                seen_delta = contrast_values[SEEN, contrast_name, metric]
                unseen_delta = contrast_values[UNSEEN, contrast_name, metric]
                interaction = unseen_delta - seen_delta
                col = f'interaction__{contrast_name}__{metric}'
                row[col] = interaction
        boot_rows.append(row)
        if (b + 1) % 1000 == 0:
            print(f'  completed {b + 1:5d} / {n_boot}')
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / 'COMPOSITION_CLUSTER_BOOTSTRAP_DISTRIBUTION.csv', index=False, encoding='utf-8-sig')
    return boot

def percentile_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return (float(lo), float(hi))

def bootstrap_sign_fractions(values):
    """
    Descriptive bootstrap sign fractions.

    These are NOT permutation-test p-values.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return (float(np.mean(values <= 0)), float(np.mean(values >= 0)))

def summarize_protocol_ci(point, boot):
    rows = []
    for condition_mode in CONDITIONS:
        for context_regime in REGIMES:
            for metric in METRICS:
                col = f'cell__{condition_mode}__{context_regime}__{metric}'
                values = boot[col].to_numpy(dtype=float)
                lo, hi = percentile_ci(values)
                rows.append({'condition_mode': condition_mode, 'context_regime': context_regime, 'metric': metric, 'point_estimate': point[condition_mode, context_regime][metric], 'composition_cluster_ci95_low': lo, 'composition_cluster_ci95_high': hi, 'n_composition_clusters': 9, 'bootstrap_repeats': len(values)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'COMPOSITION_CLUSTER_CI_BY_PROTOCOL.csv', index=False, encoding='utf-8-sig')
    return summary

def summarize_paired_protocol_differences(contrast_point, boot):
    rows = []
    for condition_mode in CONDITIONS:
        for contrast_name, first_regime, second_regime in CONTRASTS:
            for metric in METRICS:
                col = f'contrast__{condition_mode}__{contrast_name}__{metric}'
                values = boot[col].to_numpy(dtype=float)
                lo, hi = percentile_ci(values)
                frac_le_zero, frac_ge_zero = bootstrap_sign_fractions(values)
                rows.append({'condition_mode': condition_mode, 'contrast': contrast_name, 'first_regime': first_regime, 'second_regime': second_regime, 'metric': metric, 'point_estimate': contrast_point[condition_mode, contrast_name, metric], 'composition_cluster_ci95_low': lo, 'composition_cluster_ci95_high': hi, 'ci_includes_zero': bool(lo <= 0 <= hi), 'bootstrap_fraction_le_zero': frac_le_zero, 'bootstrap_fraction_ge_zero': frac_ge_zero, 'n_composition_clusters': 9, 'bootstrap_repeats': len(values)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'COMPOSITION_CLUSTER_PAIRED_PROTOCOL_DIFFERENCES.csv', index=False, encoding='utf-8-sig')
    return summary

def summarize_interactions(interaction_point, boot):
    rows = []
    for contrast_name, first_regime, second_regime in CONTRASTS:
        for metric in METRICS:
            col = f'interaction__{contrast_name}__{metric}'
            values = boot[col].to_numpy(dtype=float)
            lo, hi = percentile_ci(values)
            frac_le_zero, frac_ge_zero = bootstrap_sign_fractions(values)
            rows.append({'contrast': contrast_name, 'first_regime': first_regime, 'second_regime': second_regime, 'metric': metric, 'interaction_definition': '[A-B]_UNSEEN - [A-B]_SEEN', 'point_estimate': interaction_point[contrast_name, metric], 'composition_cluster_ci95_low': lo, 'composition_cluster_ci95_high': hi, 'ci_includes_zero': bool(lo <= 0 <= hi), 'bootstrap_fraction_le_zero': frac_le_zero, 'bootstrap_fraction_ge_zero': frac_ge_zero, 'n_composition_clusters': 9, 'bootstrap_repeats': len(values)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'COMPOSITION_CLUSTER_CONTEXT_CONDITION_INTERACTIONS.csv', index=False, encoding='utf-8-sig')
    return summary

def compare_old_run_ci(composition_ci):
    if not OLD_CI_PATH.exists():
        print('\nExisting BOOTSTRAP_CI_BY_PROTOCOL.csv not found; skipping run-vs-composition CI comparison.')
        return None
    old = pd.read_csv(OLD_CI_PATH)
    required = {'condition_mode', 'context_regime', 'metric', 'point_estimate', 'ci95_low', 'ci95_high'}
    if not required.issubset(set(old.columns)):
        print('\nExisting CI file does not have the expected columns; skipping comparison.')
        return None
    merged = old.merge(composition_ci, on=['condition_mode', 'context_regime', 'metric'], how='outer', suffixes=('_run_level', '_composition_level'))
    if 'point_estimate_run_level' in merged.columns and 'point_estimate_composition_level' in merged.columns:
        merged['point_estimate_difference'] = merged['point_estimate_composition_level'] - merged['point_estimate_run_level']
    if 'ci95_low' in merged.columns:
        merged = merged.rename(columns={'ci95_low': 'run_cluster_ci95_low', 'ci95_high': 'run_cluster_ci95_high', 'bootstrap_cluster': 'original_bootstrap_cluster', 'bootstrap_repeats_run_level': 'original_bootstrap_repeats'})
    merged.to_csv(OUT / 'RUN_VS_COMPOSITION_CI_AUDIT.csv', index=False, encoding='utf-8-sig')
    return merged

def print_headline_results(protocol_ci, paired, interactions):
    print('\n' + '=' * 120)
    print('COMPOSITION-LEVEL MICRO-F1 BY PROTOCOL')
    print('=' * 120)
    x = protocol_ci[protocol_ci['metric'] == 'mechanism_micro_f1']
    print(x[['condition_mode', 'context_regime', 'point_estimate', 'composition_cluster_ci95_low', 'composition_cluster_ci95_high']].to_string(index=False))
    print('\n' + '=' * 120)
    print('COMPOSITION-LEVEL PAIRED MICRO-F1 CONTRASTS')
    print('=' * 120)
    x = paired[paired['metric'] == 'mechanism_micro_f1']
    print(x[['condition_mode', 'contrast', 'point_estimate', 'composition_cluster_ci95_low', 'composition_cluster_ci95_high', 'ci_includes_zero']].to_string(index=False))
    print('\n' + '=' * 120)
    print('R-T CONTEXT x CONDITION INTERACTION')
    print('=' * 120)
    x = interactions[interactions['contrast'] == 'R_minus_T']
    print(x[['metric', 'point_estimate', 'composition_cluster_ci95_low', 'composition_cluster_ci95_high', 'ci_includes_zero']].to_string(index=False))

def main():
    print('\n' + '=' * 120)
    print('FULL COMPOSITION-LEVEL STATISTICAL ANALYSIS')
    print('=' * 120)
    print(f'Input : {PRED_PATH}')
    print(f'Output: {OUT}')
    df = load_predictions()
    print(f'\nLoaded prediction rows: {len(df)}')
    print(f"Unique physical runs  : {df['run_id'].nunique()}")
    print(f"Unique compositions   : {df['composition_key'].nunique()}")
    compositions = coverage_audit(df)
    point, point_df = full_sample_point_estimates(df)
    verify_against_existing_summary(point_df)
    per_composition_metrics(df, compositions)
    contrast_point, interaction_point, _, _ = calculate_full_sample_contrasts(point)
    boot = composition_cluster_bootstrap(df, compositions, n_boot=N_BOOT, seed=SEED)
    protocol_ci = summarize_protocol_ci(point, boot)
    paired = summarize_paired_protocol_differences(contrast_point, boot)
    interactions = summarize_interactions(interaction_point, boot)
    compare_old_run_ci(protocol_ci)
    print_headline_results(protocol_ci, paired, interactions)
    print('\n' + '#' * 120)
    print('AUDIT COMPLETE')
    print('#' * 120)
    print('\nGenerated files:')
    for p in sorted(OUT.glob('*.csv')):
        print(' -', p.name)
if __name__ == '__main__':
    main()
