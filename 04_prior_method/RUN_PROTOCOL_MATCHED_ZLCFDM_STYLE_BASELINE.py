"""Protocol-matched semantic-prototype baseline for compound-fault diagnosis.

Healthy and single-fault runs are used to learn a training-only linear mapping
from constituent semantics to the standardized locked physics-feature space.
Compound classes are represented by two-constituent semantic combinations and
decoded by nearest semantic-generated prototype over the nine released
compound compositions.

Seen- and unseen-operating-condition evaluations use the same physical compound
test runs as the primary analysis. This is a protocol-matched adaptation of the
semantic-prototype principle and not an exact reproduction of the original
CNN/time-frequency architecture.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')
ROOT = Path('D:\\yay\u0131nlar\\Multi-mode Fault Diagnosis Datasets of Three-phase')
PROTOCOL_DIR = ROOT / 'EXPERIMENT_PROTOCOL'
PHYSICS_PATH = ROOT / 'PHYSICS_DIAGNOSTICS_RESULTS' / 'RUN_PHYSICS_FEATURES.csv'
PRIMARY_DIR = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS'
LOCKED_MAP_PATH = PRIMARY_DIR / 'LOCKED_FEATURE_MAP.csv'
PRIMARY_PRED_PATH = PRIMARY_DIR / 'ALL_OUTER_PREDICTIONS.csv'
OUT = ROOT / 'PRIOR_METHOD_COMPARISON'
OUT.mkdir(parents=True, exist_ok=True)
OLD_SPLITS = ['train', 'validation', 'P1', 'P2', 'P3', 'P4']
COMPONENTS = ['bearing_ball', 'bearing_inner', 'bearing_outer', 'bend', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'voltage_unbalance', 'winding']
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20260908

def infer_fault_role(fault_raw):
    low = str(fault_raw).strip().lower()
    if low in {'health', 'healthy', 'normal', 'nc', 'normal_condition'}:
        return 'health'
    if '_and_' in low:
        return 'compound'
    return 'single'

def composition_key_from_row(row):
    active = [c for c in COMPONENTS if int(row[c]) == 1]
    if len(active) < 2:
        return ''
    return '+'.join(sorted(active))

def semantic_matrix(df):
    return df[COMPONENTS].astype(float).to_numpy()

def compute_metrics(df, pred_prefix='pred_'):
    y = df[COMPONENTS].astype(int).to_numpy()
    p = np.column_stack([df[f'{pred_prefix}{c}'].astype(int).to_numpy() for c in COMPONENTS])
    tp = int(np.logical_and(y == 1, p == 1).sum())
    fp = int(np.logical_and(y == 0, p == 1).sum())
    fn = int(np.logical_and(y == 1, p == 0).sum())
    micro = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    active = np.where(y.sum(axis=0) > 0)[0]
    per_f1 = []
    for j in active:
        per_f1.append(f1_score(y[:, j], p[:, j], zero_division=0))
    macro = float(np.mean(per_f1)) if per_f1 else np.nan
    em = float(np.mean(np.all(y == p, axis=1)))
    true_pos = y.sum()
    recall = float(np.logical_and(y == 1, p == 1).sum() / true_pos) if true_pos else np.nan
    fa = float(np.logical_and(y == 0, p == 1).sum(axis=1).mean())
    any_rec = float(np.mean(np.logical_and(y == 1, p == 1).sum(axis=1) > 0))
    all_rec = float(np.mean(np.logical_and(y == 1, p == 1).sum(axis=1) == y.sum(axis=1)))
    return {'mechanism_active_macro_f1': macro, 'mechanism_micro_f1': float(micro), 'exact_match': em, 'constituent_recall': recall, 'false_additions_per_run': fa, 'any_constituent_recovered_rate': any_rec, 'all_constituents_recovered_rate': all_rec}

def fit_semantic_to_visual(train_df, X_train_raw):
    """
    Training-only preprocessing + semantic -> standardized visual feature mapping.
    OLS is deliberately used to avoid tuned hyperparameters.
    """
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_train_raw)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)
    S = semantic_matrix(train_df)
    mapper = LinearRegression(fit_intercept=True)
    mapper.fit(S, X_std)
    return (imputer, scaler, mapper)

def predict_compound_classes(test_df, X_test_raw, imputer, scaler, mapper, composition_semantics):
    X_std = scaler.transform(imputer.transform(X_test_raw))
    comp_names = list(composition_semantics.keys())
    S_comp = np.vstack([composition_semantics[k] for k in comp_names]).astype(float)
    prototypes = mapper.predict(S_comp)
    d2 = ((X_std[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    winner = np.argmin(d2, axis=1)
    pred_sem = S_comp[winner].astype(int)
    pred_comp = [comp_names[i] for i in winner]
    return (pred_sem, pred_comp, d2)
print('=' * 118)
print('PROTOCOL-MATCHED SEMANTIC-PROTOTYPE BASELINE')
print('=' * 118)
manifest_frames = []
for split in OLD_SPLITS:
    p = PROTOCOL_DIR / f'runs_{split}.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    need = {'run_id', 'fault_raw', *COMPONENTS}
    missing = sorted(need - set(d.columns))
    if missing:
        raise RuntimeError(f'{p.name}: missing {missing}')
    d = d.copy()
    d['run_id'] = d['run_id'].astype(str)
    d['old_split'] = split
    manifest_frames.append(d)
runs = pd.concat(manifest_frames, ignore_index=True)
if runs['run_id'].duplicated().any():
    raise RuntimeError('Duplicate run_id in manifests.')
physics = pd.read_csv(PHYSICS_PATH)
physics['run_id'] = physics['run_id'].astype(str)
if 'condition_key' not in physics.columns:
    raise RuntimeError('condition_key missing from RUN_PHYSICS_FEATURES.csv')
physics_for_merge = physics.drop(columns=['fault_raw', 'fault_role', *COMPONENTS], errors='ignore')
data = runs.merge(physics_for_merge, on='run_id', how='left', validate='one_to_one')
if data['condition_key'].isna().any():
    raise RuntimeError('Missing physics features for one or more manifest runs.')
data['condition_key'] = data['condition_key'].astype(str)
data['fault_role'] = data['fault_raw'].map(infer_fault_role)
data['composition_key'] = data.apply(composition_key_from_row, axis=1)
print(f'Runs       : {len(data)}')
print(f"Health     : {(data.fault_role == 'health').sum()}")
print(f"Single     : {(data.fault_role == 'single').sum()}")
print(f"Compound   : {(data.fault_role == 'compound').sum()}")
print(f'Conditions : {data.condition_key.nunique()}')
if not LOCKED_MAP_PATH.exists():
    raise FileNotFoundError(LOCKED_MAP_PATH)
locked_map = pd.read_csv(LOCKED_MAP_PATH)
if 'feature' not in locked_map.columns:
    raise RuntimeError('LOCKED_FEATURE_MAP.csv missing feature column.')
features = []
for f in locked_map['feature'].dropna().astype(str):
    if f in data.columns and f not in features:
        features.append(f)
if not features:
    raise RuntimeError('No locked features found in native physics table.')
non_numeric = [f for f in features if not pd.api.types.is_numeric_dtype(data[f])]
if non_numeric:
    raise RuntimeError(f'Non-numeric locked features: {non_numeric[:20]}')
X_all = data[features].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
print(f'Unique locked feature union: {len(features)}')
print('NOTE: same locked physics feature pool is used for the prior-method adaptation.')
compound = data[data['fault_role'] == 'compound'].copy()
composition_semantics = {}
for key, g in compound.groupby('composition_key', sort=True):
    s = g.iloc[0][COMPONENTS].astype(int).to_numpy()
    if int(s.sum()) != 2:
        raise RuntimeError(f'{key}: expected 2 constituents, got {int(s.sum())}')
    if not np.all(g[COMPONENTS].astype(int).to_numpy() == s[None, :]):
        raise RuntimeError(f'{key}: inconsistent constituent labels.')
    composition_semantics[key] = s
if len(composition_semantics) != 9:
    raise RuntimeError(f'Expected 9 compound compositions, found {len(composition_semantics)}')
print(f'Candidate unseen compound semantics: {len(composition_semantics)}')
prediction_frames = []
train_mask_seen = data['fault_role'].isin(['health', 'single']).to_numpy()
test_mask_seen = (data['fault_role'] == 'compound').to_numpy()
train_idx = np.flatnonzero(train_mask_seen)
test_idx = np.flatnonzero(test_mask_seen)
imp, scl, mapper = fit_semantic_to_visual(data.iloc[train_idx], X_all[train_idx])
pred_sem, pred_comp, d2 = predict_compound_classes(data.iloc[test_idx], X_all[test_idx], imp, scl, mapper, composition_semantics)
out_seen = data.iloc[test_idx][['run_id', 'fault_raw', 'condition_key', 'composition_key', *COMPONENTS]].copy()
out_seen['condition_mode'] = 'SEEN_CONDITION'
out_seen['baseline_method'] = 'ZLCFDM_STYLE_SEMANTIC_PROTOTYPE'
out_seen['predicted_composition'] = pred_comp
for j, c in enumerate(COMPONENTS):
    out_seen[f'pred_{c}'] = pred_sem[:, j]
prediction_frames.append(out_seen)
for condition in sorted(data['condition_key'].unique()):
    train_mask = (data['fault_role'].isin(['health', 'single']) & (data['condition_key'] != condition)).to_numpy()
    test_mask = ((data['fault_role'] == 'compound') & (data['condition_key'] == condition)).to_numpy()
    tr = np.flatnonzero(train_mask)
    te = np.flatnonzero(test_mask)
    if len(te) == 0:
        continue
    imp, scl, mapper = fit_semantic_to_visual(data.iloc[tr], X_all[tr])
    pred_sem, pred_comp, d2 = predict_compound_classes(data.iloc[te], X_all[te], imp, scl, mapper, composition_semantics)
    o = data.iloc[te][['run_id', 'fault_raw', 'condition_key', 'composition_key', *COMPONENTS]].copy()
    o['condition_mode'] = 'UNSEEN_CONDITION'
    o['baseline_method'] = 'ZLCFDM_STYLE_SEMANTIC_PROTOTYPE'
    o['predicted_composition'] = pred_comp
    for j, c in enumerate(COMPONENTS):
        o[f'pred_{c}'] = pred_sem[:, j]
    prediction_frames.append(o)
pred = pd.concat(prediction_frames, ignore_index=True)
for mode in ['SEEN_CONDITION', 'UNSEEN_CONDITION']:
    g = pred[pred['condition_mode'] == mode]
    if len(g) != 108 or g['run_id'].nunique() != 108:
        raise RuntimeError(f"{mode}: expected 108 rows / 108 unique runs; got {len(g)} / {g['run_id'].nunique()}")
pred.to_csv(OUT / 'ZLCFDM_STYLE_OUTER_PREDICTIONS.csv', index=False, encoding='utf-8-sig')
summary_rows = []
for mode, g in pred.groupby('condition_mode', sort=True):
    m = compute_metrics(g)
    summary_rows.append({'method': 'ZLCFDM_STYLE_SEMANTIC_PROTOTYPE', 'condition_mode': mode, 'n_runs': int(g['run_id'].nunique()), **m})
baseline_summary = pd.DataFrame(summary_rows)
baseline_summary.to_csv(OUT / 'ZLCFDM_STYLE_SUMMARY.csv', index=False, encoding='utf-8-sig')
print('\n' + '=' * 118)
print('ZLCFDM-STYLE BASELINE SUMMARY')
print('=' * 118)
print(baseline_summary.to_string(index=False))
if not PRIMARY_PRED_PATH.exists():
    raise FileNotFoundError(PRIMARY_PRED_PATH)
primary = pd.read_csv(PRIMARY_PRED_PATH)
primary['run_id'] = primary['run_id'].astype(str)
need_primary = {'run_id', 'condition_mode', 'context_regime', 'composition_key', *COMPONENTS, *[f'pred_{c}' for c in COMPONENTS]}
missing = sorted(need_primary - set(primary.columns))
if missing:
    raise RuntimeError(f'Primary ALL_OUTER_PREDICTIONS missing: {missing}')
primary_two = primary[primary['context_regime'] == 'TWO_SIDED_CONTEXT_ZS'].copy()
comparison_rows = []
for mode in ['SEEN_CONDITION', 'UNSEEN_CONDITION']:
    p = primary_two[primary_two['condition_mode'] == mode].copy()
    b = pred[pred['condition_mode'] == mode].copy()
    if len(p) != 108 or len(b) != 108:
        raise RuntimeError(f'{mode}: comparison row count mismatch.')
    if set(p['run_id']) != set(b['run_id']):
        raise RuntimeError(f'{mode}: test run IDs do not match.')
    mp = compute_metrics(p)
    mb = compute_metrics(b)
    comparison_rows.append({'condition_mode': mode, 'method': 'PROPOSED_TWO_SIDED_CONTEXT_ZS', 'n_runs': 108, **mp})
    comparison_rows.append({'condition_mode': mode, 'method': 'ZLCFDM_STYLE_SEMANTIC_PROTOTYPE', 'n_runs': 108, **mb})
comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(OUT / 'PRIMARY_TWO_SIDED_VS_ZLCFDM_STYLE.csv', index=False, encoding='utf-8-sig')
print('\n' + '=' * 118)
print('SAME-TEST COMPARISON')
print('=' * 118)
print(comparison.to_string(index=False))
rng = np.random.default_rng(BOOTSTRAP_SEED)
metric_names = ['mechanism_micro_f1', 'mechanism_active_macro_f1', 'exact_match', 'constituent_recall', 'false_additions_per_run', 'all_constituents_recovered_rate']
boot_rows = []
for mode in ['SEEN_CONDITION', 'UNSEEN_CONDITION']:
    p = primary_two[primary_two['condition_mode'] == mode].copy()
    b = pred[pred['condition_mode'] == mode].copy()
    p = p.sort_values('run_id').reset_index(drop=True)
    b = b.sort_values('run_id').reset_index(drop=True)
    if not np.array_equal(p['run_id'].to_numpy(), b['run_id'].to_numpy()):
        raise RuntimeError(f'{mode}: paired run alignment failed.')
    if not np.array_equal(p['composition_key'].astype(str).to_numpy(), b['composition_key'].astype(str).to_numpy()):
        raise RuntimeError(f'{mode}: composition alignment failed.')
    compositions = sorted(p['composition_key'].astype(str).unique())
    if len(compositions) != 9:
        raise RuntimeError(f'{mode}: expected 9 compositions.')
    point_p = compute_metrics(p)
    point_b = compute_metrics(b)
    dist = {m: [] for m in metric_names}
    for rep in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(compositions, size=len(compositions), replace=True)
        p_parts = []
        b_parts = []
        for new_cluster_id, comp in enumerate(sampled):
            pp = p[p['composition_key'].astype(str) == str(comp)].copy()
            bb = b[b['composition_key'].astype(str) == str(comp)].copy()
            pp['_boot_cluster'] = new_cluster_id
            bb['_boot_cluster'] = new_cluster_id
            p_parts.append(pp)
            b_parts.append(bb)
        ps = pd.concat(p_parts, ignore_index=True)
        bs = pd.concat(b_parts, ignore_index=True)
        mp = compute_metrics(ps)
        mb = compute_metrics(bs)
        for m in metric_names:
            dist[m].append(mp[m] - mb[m])
    for m in metric_names:
        vals = np.asarray(dist[m], dtype=float)
        boot_rows.append({'condition_mode': mode, 'metric': m, 'proposed_value': point_p[m], 'baseline_value': point_b[m], 'delta_proposed_minus_baseline': point_p[m] - point_b[m], 'ci_low': float(np.percentile(vals, 2.5)), 'ci_high': float(np.percentile(vals, 97.5)), 'bootstrap_repeats': BOOTSTRAP_REPEATS, 'cluster_unit': 'composition'})
boot = pd.DataFrame(boot_rows)
boot.to_csv(OUT / 'PAIRED_COMPOSITION_BOOTSTRAP_PRIMARY_MINUS_BASELINE.csv', index=False, encoding='utf-8-sig')
print('\n' + '=' * 118)
print('PAIRED COMPOSITION-CLUSTER DIFFERENCE: PROPOSED - BASELINE')
print('=' * 118)
print(boot.to_string(index=False))
config = {'baseline_name': 'ZLCFDM_STYLE_SEMANTIC_PROTOTYPE', 'claim_scope': 'Protocol-matched adaptation of the core semantic-prototype zero-shot principle; not an exact architectural reproduction.', 'training_roles': ['health', 'single'], 'compound_training_used': False, 'feature_source': str(PHYSICS_PATH), 'locked_feature_map': str(LOCKED_MAP_PATH), 'n_unique_locked_features': len(features), 'semantic_dimension': len(COMPONENTS), 'n_candidate_compound_compositions': len(composition_semantics), 'semantic_to_visual_mapper': 'LinearRegression ordinary least squares', 'distance': 'squared Euclidean in training-standardized feature space', 'primary_comparator': 'TWO_SIDED_CONTEXT_ZS', 'bootstrap_repeats': BOOTSTRAP_REPEATS, 'bootstrap_cluster': 'fault composition', 'bootstrap_seed': BOOTSTRAP_SEED}
with open(OUT / 'ZLCFDM_STYLE_CONFIG.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
summary_txt = OUT / 'PRIOR_METHOD_COMPARISON_SUMMARY.txt'
with open(summary_txt, 'w', encoding='utf-8') as f:
    f.write('PROTOCOL-MATCHED PRIOR-METHOD COMPARISON\n')
    f.write('=' * 78 + '\n\n')
    f.write('Baseline: protocol-matched ZLCFDM-style semantic-prototype adaptation.\nThis is NOT claimed as an exact CNN/time-frequency reproduction.\n\n')
    f.write('Training uses healthy + single-fault runs only. Compound semantics are multi-hot combinations of the two constituent semantics. A training-only semantic-to-standardized-feature mapping is learned by ordinary least squares, and test compounds are assigned to the nearest semantic-generated compound prototype by Euclidean distance.\n\n')
    f.write('BASELINE SUMMARY\n')
    f.write(baseline_summary.to_string(index=False))
    f.write('\n\nSAME-TEST COMPARISON\n')
    f.write(comparison.to_string(index=False))
    f.write('\n\nPAIRED COMPOSITION-CLUSTER DIFFERENCES (PROPOSED - BASELINE)\n')
    f.write(boot.to_string(index=False))
    f.write('\n')
print('\nOutputs:')
for name in ['ZLCFDM_STYLE_OUTER_PREDICTIONS.csv', 'ZLCFDM_STYLE_SUMMARY.csv', 'PRIMARY_TWO_SIDED_VS_ZLCFDM_STYLE.csv', 'PAIRED_COMPOSITION_BOOTSTRAP_PRIMARY_MINUS_BASELINE.csv', 'ZLCFDM_STYLE_CONFIG.json', 'PRIOR_METHOD_COMPARISON_SUMMARY.txt']:
    print(' -', OUT / name)
print('\nDONE.')
print('Do not describe this as an exact reproduction of the prior CNN architecture.')
