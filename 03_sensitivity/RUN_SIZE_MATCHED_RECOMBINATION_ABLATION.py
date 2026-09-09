"""Training-size-matched recombination sensitivity analysis.

For each crossed fold, the recombination training pool is downsampled within
operating condition to match the corresponding two-sided training size while
retaining both target constituents in alternative compound contexts.

The locked classifier, curated feature sets, grouped threshold calibration, and
test data remain unchanged. Multiple random realizations are summarized with
composition-level cluster bootstrap inference.
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
FINAL_DIR = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS'
LOCKED_PREDICTIONS = FINAL_DIR / 'ALL_OUTER_PREDICTIONS.csv'
OUT = ROOT / 'SIZE_MATCHED_RECOMBINATION_RESULTS'
OUT.mkdir(parents=True, exist_ok=True)
N_REPEATS = 30
N_BOOT = 5000
SEED = 20260814
SKIP_COMPLETED_REPEATS = True
RECOMBINATION = 'RECOMBINATION'
TWO_SIDED = 'TWO_SIDED_CONTEXT_ZS'
SIZE_MATCHED = 'SIZE_MATCHED_RECOMBINATION'
SEEN = 'SEEN_CONDITION'
UNSEEN = 'UNSEEN_CONDITION'
PRIMARY_METRICS = ['mechanism_micro_f1', 'constituent_recall', 'all_constituents_recovered_rate', 'exact_match', 'false_additions_per_run']

def require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Required file not found:\n{path}')

def import_module_from_path(path, module_name):
    require(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not import:\n{path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def percentile_ci(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))

def metric_value(module, df, metric):
    m = module.calculate_metrics(df)
    return float(m.get(metric, np.nan))

def make_size_matched_training_mask(module, data, recombination_train_mask, two_sided_train_mask, target_constituents, seed, max_attempts=500):
    """
    Downsample RECOMBINATION training data so that its number of training
    runs within EACH condition_key exactly equals TWO_SIDED.

    Sampling never uses outer-test performance.

    It also verifies that both target constituents remain represented in at
    least one compound context after subsampling.
    """
    recomb_df = data.loc[recombination_train_mask].copy()
    two_df = data.loc[two_sided_train_mask].copy()
    target_counts = two_df['condition_key'].astype(str).value_counts().sort_index()
    recomb_counts = recomb_df['condition_key'].astype(str).value_counts().sort_index()
    for condition, desired in target_counts.items():
        available = int(recomb_counts.get(condition, 0))
        if available < int(desired):
            raise RuntimeError(f'RECOMBINATION has fewer rows than TWO_SIDED for condition={condition}: available={available}, desired={desired}')
    a, b = target_constituents
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt * 1000003)
        selected_global_indices = []
        for condition, desired in target_counts.items():
            candidate_indices = np.flatnonzero(recombination_train_mask & (data['condition_key'].astype(str).to_numpy() == str(condition)))
            chosen = rng.choice(candidate_indices, size=int(desired), replace=False)
            selected_global_indices.extend(chosen.tolist())
        selected_global_indices = np.asarray(sorted(selected_global_indices), dtype=int)
        mask = np.zeros(len(data), dtype=bool)
        mask[selected_global_indices] = True
        if int(mask.sum()) != int(two_sided_train_mask.sum()):
            raise RuntimeError('Size matching failed.')
        sampled_df = data.loc[mask]
        context_a = module.constituent_context_count(sampled_df, a)
        context_b = module.constituent_context_count(sampled_df, b)
        if context_a > 0 and context_b > 0:
            return (mask, {'sampling_attempt': attempt + 1, 'context_count_A': int(context_a), 'context_count_B': int(context_b)})
    raise RuntimeError(f'Could not generate a valid size-matched RECOMBINATION sample after {max_attempts} attempts.')

def run_one_repeat(module, data, catalog, raw_matrices, repeat_id):
    repeat_predictions = []
    repeat_thresholds = []
    training_audit = []
    fold_counter = 0
    compositions = catalog['composition_key'].astype(str).tolist()
    print('\n' + '=' * 145)
    print(f'SIZE-MATCHED RECOMBINATION — REPEAT {repeat_id:02d}/{N_REPEATS}')
    print('=' * 145)
    for composition_index, composition_key in enumerate(compositions):
        constituents, specs = module.build_context_specs_for_composition(data, composition_key)
        if len(constituents) != 2:
            raise RuntimeError(f'{composition_key}: expected 2 constituents.')
        a, b = constituents
        recomb_spec = None
        two_spec = None
        for spec in specs:
            if spec['context_regime'] == RECOMBINATION:
                recomb_spec = spec
            elif spec['context_regime'] == TWO_SIDED:
                two_spec = spec
        if recomb_spec is None or two_spec is None:
            raise RuntimeError(f'{composition_key}: missing R or T context specification.')
        target_compound_mask = ((data['fault_role'] == 'compound') & (data['composition_key'].astype(str) == composition_key)).to_numpy()
        recomb_train_mask = ~recomb_spec['excluded_context_mask']
        two_train_mask = ~two_spec['excluded_context_mask']
        test_mask = target_compound_mask.copy()
        seed = SEED + repeat_id * 100000 + composition_index * 1000 + 1
        sampled_train_mask, sample_info = make_size_matched_training_mask(module=module, data=data, recombination_train_mask=recomb_train_mask, two_sided_train_mask=two_train_mask, target_constituents=constituents, seed=seed)
        fold_counter += 1
        fold_id = f'SMR_R{repeat_id:02d}_S_{composition_index:02d}'
        fold_meta = {'fold_id': fold_id, 'ablation_repeat': repeat_id, 'condition_mode': SEEN, 'context_regime': SIZE_MATCHED, 'target_composition': composition_key, 'target_constituents': f'{a}|{b}', 'context_unseen_component': '', 'context_seen_component': '', 'heldout_condition': '', 'sampling_seed': seed, 'matched_two_sided_train_n': int(two_train_mask.sum())}
        prediction, thresholds = module.run_outer_fold(data=data, raw_matrices=raw_matrices, train_mask=sampled_train_mask, test_mask=test_mask, fold_meta=fold_meta)
        repeat_predictions.append(prediction)
        repeat_thresholds.append(thresholds)
        sampled_df = data.loc[sampled_train_mask]
        training_audit.append({**fold_meta, 'n_recombination_original': int(recomb_train_mask.sum()), 'n_two_sided': int(two_train_mask.sum()), 'n_size_matched_recombination': int(sampled_train_mask.sum()), 'n_conditions': int(sampled_df['condition_key'].nunique()), 'target_A': a, 'target_B': b, 'positive_support_A': int(sampled_df[a].astype(int).sum()), 'positive_support_B': int(sampled_df[b].astype(int).sum()), 'compound_contexts_A': int(sample_info['context_count_A']), 'compound_contexts_B': int(sample_info['context_count_B']), 'sampling_attempt': int(sample_info['sampling_attempt']), 'n_health': int((sampled_df['fault_role'] == 'health').sum()), 'n_single': int((sampled_df['fault_role'] == 'single').sum()), 'n_compound': int((sampled_df['fault_role'] == 'compound').sum())})
        target_conditions = sorted(data.loc[target_compound_mask, 'condition_key'].astype(str).unique().tolist())
        for condition_index, heldout_condition in enumerate(target_conditions):
            condition_exclusion = data['condition_key'].astype(str).to_numpy() == heldout_condition
            recomb_train_mask = ~recomb_spec['excluded_context_mask'] & ~condition_exclusion
            two_train_mask = ~two_spec['excluded_context_mask'] & ~condition_exclusion
            test_mask = target_compound_mask & (data['condition_key'].astype(str).to_numpy() == heldout_condition)
            if int(test_mask.sum()) != 1:
                raise RuntimeError(f'{composition_key} / {heldout_condition}: expected exactly one target test run, found {int(test_mask.sum())}')
            seed = SEED + repeat_id * 100000 + composition_index * 1000 + condition_index * 10 + 7
            sampled_train_mask, sample_info = make_size_matched_training_mask(module=module, data=data, recombination_train_mask=recomb_train_mask, two_sided_train_mask=two_train_mask, target_constituents=constituents, seed=seed)
            fold_counter += 1
            fold_id = f'SMR_R{repeat_id:02d}_U_{composition_index:02d}_{condition_index:02d}'
            fold_meta = {'fold_id': fold_id, 'ablation_repeat': repeat_id, 'condition_mode': UNSEEN, 'context_regime': SIZE_MATCHED, 'target_composition': composition_key, 'target_constituents': f'{a}|{b}', 'context_unseen_component': '', 'context_seen_component': '', 'heldout_condition': heldout_condition, 'sampling_seed': seed, 'matched_two_sided_train_n': int(two_train_mask.sum())}
            prediction, thresholds = module.run_outer_fold(data=data, raw_matrices=raw_matrices, train_mask=sampled_train_mask, test_mask=test_mask, fold_meta=fold_meta)
            repeat_predictions.append(prediction)
            repeat_thresholds.append(thresholds)
            sampled_df = data.loc[sampled_train_mask]
            training_audit.append({**fold_meta, 'n_recombination_original': int(recomb_train_mask.sum()), 'n_two_sided': int(two_train_mask.sum()), 'n_size_matched_recombination': int(sampled_train_mask.sum()), 'n_conditions': int(sampled_df['condition_key'].nunique()), 'target_A': a, 'target_B': b, 'positive_support_A': int(sampled_df[a].astype(int).sum()), 'positive_support_B': int(sampled_df[b].astype(int).sum()), 'compound_contexts_A': int(sample_info['context_count_A']), 'compound_contexts_B': int(sample_info['context_count_B']), 'sampling_attempt': int(sample_info['sampling_attempt']), 'n_health': int((sampled_df['fault_role'] == 'health').sum()), 'n_single': int((sampled_df['fault_role'] == 'single').sum()), 'n_compound': int((sampled_df['fault_role'] == 'compound').sum())})
        print(f'  repeat {repeat_id:02d} | composition {composition_index + 1:02d}/{len(compositions):02d} | {composition_key}')
    predictions = pd.concat(repeat_predictions, ignore_index=True)
    thresholds = pd.concat(repeat_thresholds, ignore_index=True)
    audit = pd.DataFrame(training_audit)
    for condition_mode in [SEEN, UNSEEN]:
        sub = predictions[predictions['condition_mode'] == condition_mode]
        if len(sub) != 108:
            raise RuntimeError(f'Repeat {repeat_id}, {condition_mode}: expected 108 predictions, got {len(sub)}')
        if sub['run_id'].nunique() != 108:
            raise RuntimeError(f'Repeat {repeat_id}, {condition_mode}: physical-run coverage is not 108 unique runs.')
    if not (audit['n_size_matched_recombination'] == audit['n_two_sided']).all():
        raise RuntimeError('Size-match integrity failure.')
    if (audit['compound_contexts_A'] <= 0).any():
        raise RuntimeError('A target constituent lost all recombination context.')
    if (audit['compound_contexts_B'] <= 0).any():
        raise RuntimeError('B target constituent lost all recombination context.')
    predictions.to_csv(OUT / f'repeat_{repeat_id:02d}_predictions.csv', index=False, encoding='utf-8-sig')
    thresholds.to_csv(OUT / f'repeat_{repeat_id:02d}_thresholds.csv', index=False, encoding='utf-8-sig')
    audit.to_csv(OUT / f'repeat_{repeat_id:02d}_training_audit.csv', index=False, encoding='utf-8-sig')
    print(f'Repeat {repeat_id:02d} complete. Outer folds={fold_counter}')
    return (predictions, thresholds, audit)

def execute_repeats(module, data, catalog, raw_matrices):
    all_predictions = []
    all_audits = []
    for repeat_id in range(1, N_REPEATS + 1):
        pred_path = OUT / f'repeat_{repeat_id:02d}_predictions.csv'
        audit_path = OUT / f'repeat_{repeat_id:02d}_training_audit.csv'
        if SKIP_COMPLETED_REPEATS and pred_path.exists() and audit_path.exists():
            print(f'\nRepeat {repeat_id:02d}: existing result reused.')
            p = pd.read_csv(pred_path)
            a = pd.read_csv(audit_path)
        else:
            p, _thresholds, a = run_one_repeat(module=module, data=data, catalog=catalog, raw_matrices=raw_matrices, repeat_id=repeat_id)
        all_predictions.append(p)
        all_audits.append(a)
    predictions = pd.concat(all_predictions, ignore_index=True)
    audits = pd.concat(all_audits, ignore_index=True)
    predictions.to_csv(OUT / 'SIZE_MATCHED_ALL_PREDICTIONS.csv', index=False, encoding='utf-8-sig')
    audits.to_csv(OUT / 'SIZE_MATCHED_TRAINING_AUDIT.csv', index=False, encoding='utf-8-sig')
    return (predictions, audits)

def build_per_repeat_summary(module, size_predictions, locked_predictions):
    locked_two = locked_predictions[locked_predictions['context_regime'] == TWO_SIDED].copy()
    locked_recomb = locked_predictions[locked_predictions['context_regime'] == RECOMBINATION].copy()
    summary_rows = []
    effect_rows = []
    repeat_ids = sorted(pd.to_numeric(size_predictions['ablation_repeat'], errors='raise').astype(int).unique().tolist())
    for repeat_id in repeat_ids:
        rep = size_predictions[pd.to_numeric(size_predictions['ablation_repeat'], errors='raise').astype(int) == repeat_id]
        repeat_effects = {}
        for condition_mode in [SEEN, UNSEEN]:
            smr = rep[rep['condition_mode'] == condition_mode]
            two = locked_two[locked_two['condition_mode'] == condition_mode]
            original_r = locked_recomb[locked_recomb['condition_mode'] == condition_mode]
            smr_metrics = module.calculate_metrics(smr)
            two_metrics = module.calculate_metrics(two)
            original_r_metrics = module.calculate_metrics(original_r)
            summary_rows.append({'repeat': repeat_id, 'condition_mode': condition_mode, **{f'SMR_{metric}': float(smr_metrics[metric]) for metric in PRIMARY_METRICS}, **{f'TWO_{metric}': float(two_metrics[metric]) for metric in PRIMARY_METRICS}, **{f'ORIGINAL_R_{metric}': float(original_r_metrics[metric]) for metric in PRIMARY_METRICS}})
            for metric in PRIMARY_METRICS:
                delta_smr_two = float(smr_metrics[metric]) - float(two_metrics[metric])
                delta_original_two = float(original_r_metrics[metric]) - float(two_metrics[metric])
                delta_original_smr = float(original_r_metrics[metric]) - float(smr_metrics[metric])
                effect_rows.append({'repeat': repeat_id, 'condition_mode': condition_mode, 'metric': metric, 'SMR_minus_TWO': delta_smr_two, 'ORIGINAL_R_minus_TWO': delta_original_two, 'ORIGINAL_R_minus_SMR': delta_original_smr})
                repeat_effects[condition_mode, metric] = delta_smr_two
        for metric in PRIMARY_METRICS:
            seen_delta = repeat_effects[SEEN, metric]
            unseen_delta = repeat_effects[UNSEEN, metric]
            effect_rows.append({'repeat': repeat_id, 'condition_mode': 'INTERACTION', 'metric': metric, 'SMR_minus_TWO': unseen_delta - seen_delta, 'ORIGINAL_R_minus_TWO': np.nan, 'ORIGINAL_R_minus_SMR': np.nan})
    summary = pd.DataFrame(summary_rows)
    effects = pd.DataFrame(effect_rows)
    summary.to_csv(OUT / 'SIZE_MATCHED_PER_REPEAT_SUMMARY.csv', index=False, encoding='utf-8-sig')
    effects.to_csv(OUT / 'SIZE_MATCHED_EFFECTS_BY_REPEAT.csv', index=False, encoding='utf-8-sig')
    return (summary, effects)

def aggregate_effects(effects):
    rows = []
    for (condition_mode, metric), g in effects.groupby(['condition_mode', 'metric'], sort=True):
        values = pd.to_numeric(g['SMR_minus_TWO'], errors='coerce').dropna().to_numpy(dtype=float)
        lo, hi = percentile_ci(values)
        rows.append({'condition_mode': condition_mode, 'metric': metric, 'n_repeats': len(values), 'mean_SMR_minus_TWO': float(np.mean(values)), 'median_SMR_minus_TWO': float(np.median(values)), 'sd_across_repeats': float(np.std(values, ddof=1)) if len(values) > 1 else np.nan, 'repeat_percentile_2p5': lo, 'repeat_percentile_97p5': hi, 'fraction_positive': float(np.mean(values > 0)), 'min': float(np.min(values)), 'max': float(np.max(values))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'SIZE_MATCHED_EFFECTS_AGGREGATE.csv', index=False, encoding='utf-8-sig')
    return out

def resample_compositions(df, sampled_compositions):
    by_comp = {str(comp): g.copy() for comp, g in df.groupby('composition_key', sort=False)}
    pieces = []
    for bootstrap_instance, comp in enumerate(sampled_compositions):
        block = by_comp[str(comp)].copy()
        block['_bootstrap_composition_instance'] = bootstrap_instance
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)

def composition_cluster_bootstrap(module, size_predictions, locked_predictions):
    rng = np.random.default_rng(SEED + 777)
    locked_two = locked_predictions[locked_predictions['context_regime'] == TWO_SIDED].copy()
    repeat_ids = sorted(pd.to_numeric(size_predictions['ablation_repeat'], errors='raise').astype(int).unique().tolist())
    compositions = sorted(locked_two['composition_key'].astype(str).unique().tolist())
    if len(compositions) != 9:
        raise RuntimeError(f'Expected 9 compositions, found {len(compositions)}.')
    rows = []
    print('\n' + '=' * 145)
    print('SIZE-MATCHED COMPOSITION-LEVEL CLUSTER BOOTSTRAP')
    print('=' * 145)
    print(f'Random size-matching repeats : {len(repeat_ids)}')
    print(f'Composition clusters         : {len(compositions)}')
    print(f'Bootstrap repetitions        : {N_BOOT}')
    for b in range(N_BOOT):
        repeat_id = int(rng.choice(repeat_ids))
        smr_repeat = size_predictions[pd.to_numeric(size_predictions['ablation_repeat'], errors='raise').astype(int) == repeat_id]
        sampled_compositions = rng.choice(compositions, size=len(compositions), replace=True)
        cells = {}
        for condition_mode in [SEEN, UNSEEN]:
            smr = smr_repeat[smr_repeat['condition_mode'] == condition_mode]
            two = locked_two[locked_two['condition_mode'] == condition_mode]
            smr_boot = resample_compositions(smr, sampled_compositions)
            two_boot = resample_compositions(two, sampled_compositions)
            cells[condition_mode, 'SMR'] = module.calculate_metrics(smr_boot)
            cells[condition_mode, 'TWO'] = module.calculate_metrics(two_boot)
        row = {'bootstrap_id': b + 1, 'sampled_repeat': repeat_id}
        for metric in PRIMARY_METRICS:
            delta_seen = float(cells[SEEN, 'SMR'][metric]) - float(cells[SEEN, 'TWO'][metric])
            delta_unseen = float(cells[UNSEEN, 'SMR'][metric]) - float(cells[UNSEEN, 'TWO'][metric])
            row[f'{metric}__delta_seen'] = delta_seen
            row[f'{metric}__delta_unseen'] = delta_unseen
            row[f'{metric}__interaction'] = delta_unseen - delta_seen
        rows.append(row)
        if (b + 1) % 500 == 0:
            print(f'  completed {b + 1:5d} / {N_BOOT}')
    boot = pd.DataFrame(rows)
    boot.to_csv(OUT / 'SIZE_MATCHED_COMPOSITION_BOOTSTRAP.csv', index=False, encoding='utf-8-sig')
    return boot

def summarize_bootstrap(effects, boot):
    rows = []
    for metric in PRIMARY_METRICS:
        effect_metric = effects[effects['metric'] == metric]
        point_seen = float(effect_metric.loc[effect_metric['condition_mode'] == SEEN, 'SMR_minus_TWO'].mean())
        point_unseen = float(effect_metric.loc[effect_metric['condition_mode'] == UNSEEN, 'SMR_minus_TWO'].mean())
        point_interaction = float(effect_metric.loc[effect_metric['condition_mode'] == 'INTERACTION', 'SMR_minus_TWO'].mean())
        for effect_name, point in [('delta_seen', point_seen), ('delta_unseen', point_unseen), ('interaction', point_interaction)]:
            values = pd.to_numeric(boot[f'{metric}__{effect_name}'], errors='coerce').dropna().to_numpy(dtype=float)
            lo, hi = percentile_ci(values)
            rows.append({'metric': metric, 'effect': effect_name, 'point_estimate_mean_across_size_match_repeats': point, 'composition_cluster_ci95_low': lo, 'composition_cluster_ci95_high': hi, 'bootstrap_fraction_positive': float(np.mean(values > 0)), 'n_size_match_repeats': int(effects['repeat'].nunique()), 'n_composition_clusters': 9, 'n_bootstrap': len(values)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'SIZE_MATCHED_COMPOSITION_BOOTSTRAP_SUMMARY.csv', index=False, encoding='utf-8-sig')
    return summary

def print_training_audit(audit):
    print('\n' + '=' * 145)
    print('SIZE-MATCH TRAINING AUDIT')
    print('=' * 145)
    for condition_mode, g in audit.groupby('condition_mode', sort=True):
        print(f'\n{condition_mode}')
        print(f"  original recombination n : median={g['n_recombination_original'].median():.1f}, range=[{g['n_recombination_original'].min()}, {g['n_recombination_original'].max()}]")
        print(f"  locked two-sided n       : median={g['n_two_sided'].median():.1f}, range=[{g['n_two_sided'].min()}, {g['n_two_sided'].max()}]")
        print(f"  size-matched recomb n    : median={g['n_size_matched_recombination'].median():.1f}, range=[{g['n_size_matched_recombination'].min()}, {g['n_size_matched_recombination'].max()}]")
        print('  match exact              : ', bool((g['n_size_matched_recombination'] == g['n_two_sided']).all()))
        print(f"  median positive A/B      : {g['positive_support_A'].median():.1f} / {g['positive_support_B'].median():.1f}")
        print(f"  minimum contexts A/B     : {g['compound_contexts_A'].min()} / {g['compound_contexts_B'].min()}")

def print_key_results(aggregate, bootstrap_summary):
    print('\n' + '=' * 145)
    print('SIZE-MATCHED RECOMBINATION vs TWO-SIDED')
    print('=' * 145)
    key_metrics = ['mechanism_micro_f1', 'constituent_recall', 'all_constituents_recovered_rate']
    for metric in key_metrics:
        print(f'\n{metric}')
        for condition_mode in [SEEN, UNSEEN, 'INTERACTION']:
            row = aggregate[(aggregate['metric'] == metric) & (aggregate['condition_mode'] == condition_mode)]
            if len(row) == 1:
                row = row.iloc[0]
                print(f"  {condition_mode:18s} mean={row['mean_SMR_minus_TWO']:+.6f} median={row['median_SMR_minus_TWO']:+.6f} positive={100.0 * row['fraction_positive']:.1f}% range=[{row['min']:+.6f}, {row['max']:+.6f}]")
    print('\n' + '=' * 145)
    print('COMPOSITION-CLUSTER BOOTSTRAP')
    print('=' * 145)
    show = bootstrap_summary[bootstrap_summary['metric'].isin(key_metrics)].copy()
    print(show.to_string(index=False))
    print('\n' + '#' * 145)
    print('SIZE-MATCHED INTERPRETATION CHECK')
    print('#' * 145)
    for metric in ['mechanism_micro_f1', 'constituent_recall']:
        unseen = bootstrap_summary[(bootstrap_summary['metric'] == metric) & (bootstrap_summary['effect'] == 'delta_unseen')]
        if len(unseen) != 1:
            continue
        row = unseen.iloc[0]
        point = float(row['point_estimate_mean_across_size_match_repeats'])
        lo = float(row['composition_cluster_ci95_low'])
        hi = float(row['composition_cluster_ci95_high'])
        print(f'\n{metric}')
        print(f'  unseen SMR - TWO = {point:+.6f}')
        print(f'  composition 95% CI = [{lo:+.6f}, {hi:+.6f}]')
        if lo > 0:
            print('  => Advantage persists after total training-size matching.')
        elif hi < 0:
            print('  => Direction reverses after training-size matching.')
        else:
            print('  => Difference is not resolved after training-size matching.')
    print('#' * 145)

def save_config():
    config = {'experiment': 'size_matched_recombination_sensitivity', 'analysis_question': 'training-set cardinality sensitivity', 'locked_primary_changed': False, 'size_matching': {'source_pool': 'RECOMBINATION outer-training pool', 'target_size': 'corresponding TWO_SIDED_CONTEXT_ZS outer-training size', 'stratification': 'exact matching of training-run counts separately within every condition_key', 'performance_used_for_sampling': False, 'context_constraint': 'both target constituents must retain at least one non-target compound context after subsampling', 'n_random_repeats': N_REPEATS, 'seed': SEED}, 'model': 'same locked L2 LogisticRegression, C=1, class_weight=balanced', 'features': 'same curated mechanism-specific physics features', 'threshold_calibration': 'same grouped inner GroupKFold by condition_key, F1 objective, balanced-accuracy tie-break, nearest-0.5 tie', 'outer_test_changed': False, 'inference': {'cluster': 'compound composition', 'n_compositions': 9, 'bootstrap_repeats': N_BOOT, 'random_size_match_uncertainty': 'one size-matching repeat sampled per bootstrap replicate'}, 'interpretation_limit': 'This analysis controls total training-set cardinality and operating-condition distribution. It intentionally does not equalize target-positive support, because compound-context positive support is part of the exposure being tested.'}
    with open(OUT / 'experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def main():
    start = time.time()
    print('\n' + '=' * 145)
    print('training-cardinality — SIZE-MATCHED RECOMBINATION ABLATION')
    print('=' * 145)
    print(f'Random size-match repeats : {N_REPEATS}')
    print(f'Composition bootstrap     : {N_BOOT}')
    print(f'Output                    : {OUT}')
    require(FINAL_SCRIPT)
    require(LOCKED_PREDICTIONS)
    module = import_module_from_path(FINAL_SCRIPT, 'locked_crossed_protocol_for_size_match')
    data, definition = module.load_data()
    _feature_map, raw_matrices = module.build_feature_map_and_raw_matrices(data, definition)
    catalog = module.build_composition_catalog(data)
    locked_predictions = pd.read_csv(LOCKED_PREDICTIONS)
    locked_predictions['run_id'] = locked_predictions['run_id'].astype(str)
    size_predictions, audit = execute_repeats(module=module, data=data, catalog=catalog, raw_matrices=raw_matrices)
    print_training_audit(audit)
    per_repeat, effects = build_per_repeat_summary(module=module, size_predictions=size_predictions, locked_predictions=locked_predictions)
    aggregate = aggregate_effects(effects)
    boot = composition_cluster_bootstrap(module=module, size_predictions=size_predictions, locked_predictions=locked_predictions)
    bootstrap_summary = summarize_bootstrap(effects=effects, boot=boot)
    print_key_results(aggregate=aggregate, bootstrap_summary=bootstrap_summary)
    save_config()
    elapsed = time.time() - start
    print('\n' + '=' * 145)
    print('DONE')
    print('=' * 145)
    print(f'Elapsed: {elapsed / 60.0:.2f} minutes')
    print('\nGenerated key files:')
    for name in ['SIZE_MATCHED_TRAINING_AUDIT.csv', 'SIZE_MATCHED_PER_REPEAT_SUMMARY.csv', 'SIZE_MATCHED_EFFECTS_BY_REPEAT.csv', 'SIZE_MATCHED_EFFECTS_AGGREGATE.csv', 'SIZE_MATCHED_COMPOSITION_BOOTSTRAP_SUMMARY.csv', 'experiment_config.json']:
        print(' -', name)
if __name__ == '__main__':
    main()
