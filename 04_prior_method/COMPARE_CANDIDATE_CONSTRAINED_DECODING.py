"""Output-space-matched decoding analysis for prior-method comparison.

The locked two-sided constituent probabilities are decoded over the same nine
predefined compound candidates used by the semantic-prototype baseline.
Candidate selection uses independent-Bernoulli log likelihood and requires no
retraining, threshold calibration, or hyperparameter tuning.

This analysis isolates the effect of output-space constraints; the unrestricted
constituent decoder remains the primary formulation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path('D:\\yay\u0131nlar\\Multi-mode Fault Diagnosis Datasets of Three-phase')
PRIMARY_PATH = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS' / 'ALL_OUTER_PREDICTIONS.csv'
BASELINE_PATH = ROOT / 'PRIOR_METHOD_COMPARISON' / 'ZLCFDM_STYLE_OUTER_PREDICTIONS.csv'
OUT = ROOT / 'PRIOR_METHOD_COMPARISON'
OUT.mkdir(parents=True, exist_ok=True)
COMPONENTS = ['bearing_ball', 'bearing_inner', 'bearing_outer', 'bend', 'broken_bar', 'dynamic_eccentricity', 'static_eccentricity', 'voltage_unbalance', 'winding']
BOOTSTRAP_REPEATS = 10000
SEED = 20260908
EPS = 1e-12

def metrics(df):
    y = df[COMPONENTS].astype(int).to_numpy()
    p = np.column_stack([df[f'pred_{c}'].astype(int).to_numpy() for c in COMPONENTS])
    tp = np.logical_and(y == 1, p == 1).sum()
    fp = np.logical_and(y == 0, p == 1).sum()
    fn = np.logical_and(y == 1, p == 0).sum()
    micro = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    active = np.where(y.sum(axis=0) > 0)[0]
    f1s = []
    for j in active:
        yt = y[:, j]
        yp = p[:, j]
        tpj = np.logical_and(yt == 1, yp == 1).sum()
        fpj = np.logical_and(yt == 0, yp == 1).sum()
        fnj = np.logical_and(yt == 1, yp == 0).sum()
        den = 2 * tpj + fpj + fnj
        f1s.append(0.0 if den == 0 else 2 * tpj / den)
    exact = np.mean(np.all(y == p, axis=1))
    recall = np.logical_and(y == 1, p == 1).sum() / y.sum()
    fa = np.logical_and(y == 0, p == 1).sum(axis=1).mean()
    any_rec = np.mean(np.logical_and(y == 1, p == 1).sum(axis=1) > 0)
    all_rec = np.mean(np.logical_and(y == 1, p == 1).sum(axis=1) == y.sum(axis=1))
    return {'mechanism_active_macro_f1': float(np.mean(f1s)), 'mechanism_micro_f1': float(micro), 'exact_match': float(exact), 'constituent_recall': float(recall), 'false_additions_per_run': float(fa), 'any_constituent_recovered_rate': float(any_rec), 'all_constituents_recovered_rate': float(all_rec)}
print('=' * 110)
print('OUTPUT-SPACE-MATCHED CANDIDATE DECODING')
print('=' * 110)
primary = pd.read_csv(PRIMARY_PATH)
primary['run_id'] = primary['run_id'].astype(str)
need_prob = [f'prob_{c}' for c in COMPONENTS]
missing = [c for c in need_prob if c not in primary.columns]
if missing:
    raise RuntimeError(f'Missing probability columns in primary predictions: {missing}')
two = primary[primary['context_regime'] == 'TWO_SIDED_CONTEXT_ZS'].copy()
cand = two[['composition_key', *COMPONENTS]].drop_duplicates().sort_values('composition_key').reset_index(drop=True)
if len(cand) != 9:
    raise RuntimeError(f'Expected 9 candidate compositions, found {len(cand)}')
candidate_names = cand['composition_key'].astype(str).tolist()
S = cand[COMPONENTS].astype(int).to_numpy()
print(f'Candidate compound set: {len(candidate_names)} compositions')
decoded_frames = []
for mode in ['SEEN_CONDITION', 'UNSEEN_CONDITION']:
    g = two[two['condition_mode'] == mode].copy().reset_index(drop=True)
    if len(g) != 108:
        raise RuntimeError(f'{mode}: expected 108 rows, got {len(g)}')
    P = g[need_prob].astype(float).to_numpy()
    P = np.clip(P, EPS, 1.0 - EPS)
    ll = (S[None, :, :] * np.log(P[:, None, :]) + (1 - S[None, :, :]) * np.log(1.0 - P[:, None, :])).sum(axis=2)
    winner = np.argmax(ll, axis=1)
    pred_sem = S[winner]
    pred_names = [candidate_names[i] for i in winner]
    out = g[['run_id', 'fault_raw', 'condition_key', 'composition_key', *COMPONENTS]].copy()
    out['condition_mode'] = mode
    out['method'] = 'PROPOSED_TWO_SIDED_CANDIDATE_CONSTRAINED'
    out['predicted_composition'] = pred_names
    for j, c in enumerate(COMPONENTS):
        out[f'pred_{c}'] = pred_sem[:, j]
    decoded_frames.append(out)
decoded = pd.concat(decoded_frames, ignore_index=True)
decoded.to_csv(OUT / 'PROPOSED_TWO_SIDED_CANDIDATE_CONSTRAINED_PREDICTIONS.csv', index=False, encoding='utf-8-sig')
baseline = pd.read_csv(BASELINE_PATH)
baseline['run_id'] = baseline['run_id'].astype(str)
unrestricted = two.copy()
rows = []
for mode in ['SEEN_CONDITION', 'UNSEEN_CONDITION']:
    gu = unrestricted[unrestricted['condition_mode'] == mode].copy()
    gc = decoded[decoded['condition_mode'] == mode].copy()
    gb = baseline[baseline['condition_mode'] == mode].copy()
    for label, df in [('PROPOSED_TWO_SIDED_UNRESTRICTED', gu), ('PROPOSED_TWO_SIDED_CANDIDATE_CONSTRAINED', gc), ('ZLCFDM_STYLE_SEMANTIC_PROTOTYPE', gb)]:
        rows.append({'condition_mode': mode, 'method': label, 'n_runs': int(df['run_id'].nunique()), **metrics(df)})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'OUTPUT_SPACE_MATCHED_COMPARISON.csv', index=False, encoding='utf-8-sig')
print('\n' + '=' * 110)
print('OUTPUT-SPACE MATCHED SUMMARY')
print('=' * 110)
print(summary.to_string(index=False))
metric_names = ['mechanism_micro_f1', 'mechanism_active_macro_f1', 'exact_match', 'constituent_recall', 'false_additions_per_run', 'all_constituents_recovered_rate']
rng = np.random.default_rng(SEED)
boot_rows = []
for mode in ['SEEN_CONDITION', 'UNSEEN_CONDITION']:
    p = decoded[decoded['condition_mode'] == mode].sort_values('run_id').reset_index(drop=True)
    b = baseline[baseline['condition_mode'] == mode].sort_values('run_id').reset_index(drop=True)
    if not np.array_equal(p['run_id'].to_numpy(), b['run_id'].to_numpy()):
        raise RuntimeError(f'{mode}: paired run IDs do not align.')
    comps = sorted(p['composition_key'].astype(str).unique())
    if len(comps) != 9:
        raise RuntimeError(f'{mode}: expected 9 compositions.')
    pm = metrics(p)
    bm = metrics(b)
    dist = {m: [] for m in metric_names}
    for _ in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(comps, size=9, replace=True)
        pp = pd.concat([p[p['composition_key'].astype(str) == str(c)] for c in sampled], ignore_index=True)
        bb = pd.concat([b[b['composition_key'].astype(str) == str(c)] for c in sampled], ignore_index=True)
        x = metrics(pp)
        y = metrics(bb)
        for m in metric_names:
            dist[m].append(x[m] - y[m])
    for m in metric_names:
        vals = np.asarray(dist[m], dtype=float)
        boot_rows.append({'condition_mode': mode, 'metric': m, 'candidate_constrained_proposed': pm[m], 'zlcfmd_style_baseline': bm[m], 'delta_proposed_minus_baseline': pm[m] - bm[m], 'ci_low': float(np.percentile(vals, 2.5)), 'ci_high': float(np.percentile(vals, 97.5)), 'bootstrap_repeats': BOOTSTRAP_REPEATS, 'cluster_unit': 'composition'})
boot = pd.DataFrame(boot_rows)
boot.to_csv(OUT / 'PAIRED_COMPOSITION_BOOTSTRAP_CANDIDATE_CONSTRAINED_MINUS_BASELINE.csv', index=False, encoding='utf-8-sig')
print('\n' + '=' * 110)
print('CANDIDATE-CONSTRAINED PROPOSED - ZLCFDM-STYLE BASELINE')
print('=' * 110)
print(boot.to_string(index=False))
txt = OUT / 'OUTPUT_SPACE_MATCHED_COMPARISON_SUMMARY.txt'
with open(txt, 'w', encoding='utf-8') as f:
    f.write('OUTPUT-SPACE-MATCHED DECODING ANALYSIS\n')
    f.write('=' * 72 + '\n\n')
    f.write('The locked proposed method predicts nine constituents independently and can emit arbitrary subsets. The ZLCFDM-style comparator selects among the nine known unseen compound compositions. This sensitivity analysis applies the same nine-candidate restriction to the already-generated proposed-method probabilities without retraining or tuning.\n\n')
    f.write(summary.to_string(index=False))
    f.write('\n\nPAIRED COMPOSITION-CLUSTER DIFFERENCES\n')
    f.write(boot.to_string(index=False))
    f.write('\n')
print('\nOutputs:')
print(' -', OUT / 'PROPOSED_TWO_SIDED_CANDIDATE_CONSTRAINED_PREDICTIONS.csv')
print(' -', OUT / 'OUTPUT_SPACE_MATCHED_COMPARISON.csv')
print(' -', OUT / 'PAIRED_COMPOSITION_BOOTSTRAP_CANDIDATE_CONSTRAINED_MINUS_BASELINE.csv')
print(' -', txt)
print('\nDONE.')
print('This sensitivity analysis does not replace the locked primary decoder.')
