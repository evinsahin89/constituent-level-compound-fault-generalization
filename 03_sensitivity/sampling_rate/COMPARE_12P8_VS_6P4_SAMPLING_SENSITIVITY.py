"""Paired comparison of native and reduced sampling-rate evaluations.

The native 12.8-kHz and independently rerun 6.4-kHz crossed evaluations are
compared after verifying identical outer-test rows, true constituent labels,
and locked feature maps. Performance differences are computed within each
crossed scenario.

Uncertainty is estimated with paired composition-level cluster bootstrap
resampling over the nine compound compositions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
ROOT = Path('D:\\yay\u0131nlar\\Multi-mode Fault Diagnosis Datasets of Three-phase')
NATIVE_DIR = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS'
REDUCED_DIR = ROOT / 'SAMPLING_RATE_SENSITIVITY' / 'FINAL_CROSSED_6P4KHZ'
OUT_DIR = ROOT / 'SAMPLING_RATE_SENSITIVITY' / 'COMPARISON_12P8_VS_6P4'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_12P8 = NATIVE_DIR / 'ALL_OUTER_PREDICTIONS.csv'
PRED_6P4 = REDUCED_DIR / 'ALL_OUTER_PREDICTIONS.csv'
MAP_12P8 = NATIVE_DIR / 'LOCKED_FEATURE_MAP.csv'
MAP_6P4 = REDUCED_DIR / 'LOCKED_FEATURE_MAP.csv'
COMPONENTS = ['bearing_ball', 'bearing_inner', 'bearing_outer', 'bend', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'voltage_unbalance', 'winding']
KEY_COLUMNS = ['condition_mode', 'context_regime', 'target_composition', 'context_unseen_component', 'heldout_condition', 'run_id']
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20260908
METRICS = ['mechanism_micro_f1', 'mechanism_active_macro_f1', 'exact_match', 'constituent_recall', 'false_additions_per_run', 'all_constituents_recovered_rate']

def normalize_keys(df):
    out = df.copy()
    for c in KEY_COLUMNS:
        if c not in out.columns:
            raise RuntimeError(f'Prediction table missing key column: {c}')
        out[c] = out[c].fillna('').astype(str)
    out['run_id'] = out['run_id'].astype(str)
    return out

def get_arrays(df):
    y_true = df[COMPONENTS].astype(int).to_numpy()
    y_pred = np.column_stack([df[f'pred_{c}'].astype(int).to_numpy() for c in COMPONENTS])
    return (y_true, y_pred)

def calculate_metrics(df):
    if len(df) == 0:
        return {m: np.nan for m in METRICS}
    y_true, y_pred = get_arrays(df)
    active_f1 = []
    for j in range(len(COMPONENTS)):
        truth = y_true[:, j]
        pred = y_pred[:, j]
        if truth.sum() > 0:
            active_f1.append(f1_score(truth, pred, zero_division=0))
    micro = f1_score(y_true.reshape(-1), y_pred.reshape(-1), zero_division=0)
    exact = np.mean(np.all(y_true == y_pred, axis=1))
    total_true = int(y_true.sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    recall = tp / total_true if total_true > 0 else np.nan
    false_each = ((y_true == 0) & (y_pred == 1)).sum(axis=1)
    recovered_each = ((y_true == 1) & (y_pred == 1)).sum(axis=1)
    true_each = y_true.sum(axis=1)
    return {'mechanism_micro_f1': float(micro), 'mechanism_active_macro_f1': float(np.mean(active_f1)) if active_f1 else np.nan, 'exact_match': float(exact), 'constituent_recall': float(recall), 'false_additions_per_run': float(false_each.mean()), 'all_constituents_recovered_rate': float((recovered_each == true_each).mean())}

def load_and_audit():
    for p in [PRED_12P8, PRED_6P4, MAP_12P8, MAP_6P4]:
        if not p.exists():
            raise FileNotFoundError(str(p))
    p12 = normalize_keys(pd.read_csv(PRED_12P8))
    p64 = normalize_keys(pd.read_csv(PRED_6P4))
    if p12.duplicated(KEY_COLUMNS).any():
        raise RuntimeError('12.8-kHz predictions contain duplicate comparison keys.')
    if p64.duplicated(KEY_COLUMNS).any():
        raise RuntimeError('6.4-kHz predictions contain duplicate comparison keys.')
    keys12 = set(map(tuple, p12[KEY_COLUMNS].to_numpy().tolist()))
    keys64 = set(map(tuple, p64[KEY_COLUMNS].to_numpy().tolist()))
    if keys12 != keys64:
        only12 = list(keys12 - keys64)[:10]
        only64 = list(keys64 - keys12)[:10]
        raise RuntimeError(f'Outer-test prediction rows differ between sampling rates.\nOnly 12.8 examples: {only12}\nOnly 6.4 examples: {only64}')
    p12 = p12.sort_values(KEY_COLUMNS).reset_index(drop=True)
    p64 = p64.sort_values(KEY_COLUMNS).reset_index(drop=True)
    for c in COMPONENTS:
        if not np.array_equal(p12[c].astype(int).to_numpy(), p64[c].astype(int).to_numpy()):
            raise RuntimeError(f'True labels differ between sampling rates for {c}.')
    m12 = pd.read_csv(MAP_12P8).astype(str).sort_values(['component', 'feature']).reset_index(drop=True)
    m64 = pd.read_csv(MAP_6P4).astype(str).sort_values(['component', 'feature']).reset_index(drop=True)
    map_equal = m12.equals(m64)
    map_audit = pd.DataFrame([{'native_feature_pairs': len(m12), 'reduced_feature_pairs': len(m64), 'exact_feature_map_match': bool(map_equal)}])
    map_audit.to_csv(OUT_DIR / 'SAMPLING_FEATURE_MAP_AUDIT.csv', index=False, encoding='utf-8-sig')
    if not map_equal:
        raise RuntimeError('6.4-kHz feature map is not identical to the native locked map.')
    return (p12, p64)

def point_estimates(p12, p64):
    rows = []
    for condition_mode in sorted(p12['condition_mode'].unique()):
        for context_regime in sorted(p12['context_regime'].unique()):
            mask12 = (p12['condition_mode'] == condition_mode) & (p12['context_regime'] == context_regime)
            mask64 = (p64['condition_mode'] == condition_mode) & (p64['context_regime'] == context_regime)
            g12 = p12.loc[mask12]
            g64 = p64.loc[mask64]
            if len(g12) == 0 or len(g64) == 0:
                continue
            m12 = calculate_metrics(g12)
            m64 = calculate_metrics(g64)
            for rate, m in [(12800, m12), (6400, m64)]:
                row = {'sampling_rate_hz': rate, 'condition_mode': condition_mode, 'context_regime': context_regime, 'n_prediction_rows': len(g12 if rate == 12800 else g64), 'n_unique_runs': int((g12 if rate == 12800 else g64)['run_id'].nunique()), 'n_compositions': int((g12 if rate == 12800 else g64)['target_composition'].nunique())}
                row.update(m)
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / 'SAMPLING_POINT_ESTIMATES_BY_PROTOCOL.csv', index=False, encoding='utf-8-sig')
    return out

def paired_composition_bootstrap(p12, p64):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for condition_mode in sorted(p12['condition_mode'].unique()):
        for context_regime in sorted(p12['context_regime'].unique()):
            g12 = p12[(p12['condition_mode'] == condition_mode) & (p12['context_regime'] == context_regime)].copy()
            g64 = p64[(p64['condition_mode'] == condition_mode) & (p64['context_regime'] == context_regime)].copy()
            if len(g12) == 0 or len(g64) == 0:
                continue
            compositions = sorted(g12['target_composition'].unique().tolist())
            if compositions != sorted(g64['target_composition'].unique().tolist()):
                raise RuntimeError('Composition coverage mismatch.')
            if len(compositions) != 9:
                raise RuntimeError(f'Expected 9 compositions in {condition_mode}/{context_regime}, found {len(compositions)}.')
            by12 = {c: g12[g12['target_composition'] == c].copy() for c in compositions}
            by64 = {c: g64[g64['target_composition'] == c].copy() for c in compositions}
            point12 = calculate_metrics(g12)
            point64 = calculate_metrics(g64)
            dist = {m: np.empty(BOOTSTRAP_REPEATS, dtype=float) for m in METRICS}
            for b in range(BOOTSTRAP_REPEATS):
                sampled = rng.choice(compositions, size=len(compositions), replace=True)
                s12 = pd.concat([by12[c] for c in sampled], ignore_index=True)
                s64 = pd.concat([by64[c] for c in sampled], ignore_index=True)
                m12 = calculate_metrics(s12)
                m64 = calculate_metrics(s64)
                for metric in METRICS:
                    dist[metric][b] = m64[metric] - m12[metric]
            for metric in METRICS:
                arr = dist[metric]
                rows.append({'condition_mode': condition_mode, 'context_regime': context_regime, 'metric': metric, 'native_12p8': point12[metric], 'reduced_6p4': point64[metric], 'delta_6p4_minus_12p8': point64[metric] - point12[metric], 'ci95_low': float(np.percentile(arr, 2.5)), 'ci95_high': float(np.percentile(arr, 97.5)), 'bootstrap_cluster': 'target_composition', 'n_compositions': len(compositions), 'bootstrap_repeats': BOOTSTRAP_REPEATS})
            print(f'BOOTSTRAP DONE: {condition_mode:16s} | {context_regime:22s}')
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / 'SAMPLING_PAIRED_COMPOSITION_BOOTSTRAP_DIFFERENCES.csv', index=False, encoding='utf-8-sig')
    return out

def write_summary(point, diff):
    lines = []
    lines.append('SAMPLING-RATE SENSITIVITY: 12.8 kHz vs 6.4 kHz')
    lines.append('=' * 88)
    lines.append('Native: 12.8 kHz')
    lines.append('Reduced: 6.4 kHz (anti-aliased downsampling)')
    lines.append('Feature map: exact native locked map')
    lines.append('Difference: 6.4 kHz - 12.8 kHz')
    lines.append('')
    for condition_mode in sorted(diff['condition_mode'].unique()):
        lines.append(condition_mode)
        lines.append('-' * len(condition_mode))
        sub = diff[diff['condition_mode'] == condition_mode]
        for context_regime in sorted(sub['context_regime'].unique()):
            lines.append(f'  {context_regime}')
            ss = sub[sub['context_regime'] == context_regime]
            for metric in METRICS:
                r = ss[ss['metric'] == metric].iloc[0]
                lines.append(f"    {metric:36s}: 12.8={r['native_12p8']:.4f}  6.4={r['reduced_6p4']:.4f}  Delta={r['delta_6p4_minus_12p8']:+.4f}  95% CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]")
            lines.append('')
    lines.append('Across-scenario descriptive maximum absolute changes')
    lines.append('-' * 56)
    for metric in METRICS:
        ss = diff[diff['metric'] == metric]
        lines.append(f"{metric:36s}: {ss['delta_6p4_minus_12p8'].abs().max():.4f}")
    path = OUT_DIR / 'SAMPLING_RATE_COMPARISON_SUMMARY.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print('\n' + '\n'.join(lines))
    return path

def main():
    print('\n' + '=' * 110)
    print('PAIRED SAMPLING-RATE SENSITIVITY: 12.8 kHz vs 6.4 kHz')
    print('=' * 110)
    p12, p64 = load_and_audit()
    print(f'Prediction rows at each rate: {len(p12)}')
    print(f"Unique physical compound runs: {p12['run_id'].nunique()}")
    print('Feature-map audit: PASS')
    print('Outer-test row/label audit: PASS')
    point = point_estimates(p12, p64)
    diff = paired_composition_bootstrap(p12, p64)
    summary_path = write_summary(point, diff)
    config = {'analysis': 'sampling_rate_sensitivity_comparison', 'native_sampling_rate_hz': 12800, 'reduced_sampling_rate_hz': 6400, 'difference_definition': '6.4_kHz_minus_12.8_kHz', 'paired_unit': 'same outer-test prediction row', 'bootstrap_cluster': 'target_composition', 'n_compositions': 9, 'bootstrap_repeats': BOOTSTRAP_REPEATS, 'bootstrap_seed': BOOTSTRAP_SEED, 'feature_map_identical': True, 'native_predictions': str(PRED_12P8), 'reduced_predictions': str(PRED_6P4)}
    with open(OUT_DIR / 'sampling_comparison_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print('\nOutputs:')
    print(OUT_DIR)
    print('\nMost important file:')
    print(summary_path)
if __name__ == '__main__':
    main()
