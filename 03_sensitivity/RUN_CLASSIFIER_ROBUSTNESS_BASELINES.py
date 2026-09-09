"""Classifier-family robustness analysis for the crossed constituent-level protocol.

Random forest, RBF-SVM, and multilayer perceptron models are evaluated with the
same mechanism-specific features, crossed outer folds, training-only
preprocessing, grouped threshold calibration, and evaluation metrics as the
locked logistic-regression analysis.

Alternative classifiers are used only to characterize robustness and are not
used to replace or select the primary model.
"""
from pathlib import Path
import importlib.util
import inspect
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings('ignore')
ROOT = Path('D:\\Multi-mode Fault Diagnosis Datasets of Three-phase')
FINAL_SCRIPT = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_EVALUATION.py'
LOCKED_LR_DIR = ROOT / 'FINAL_CROSSED_COMPOSITION_CONDITION_RESULTS'
OUT_ROOT = ROOT / 'MODEL_ROBUSTNESS_COMPARISON'
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SEED = 42
MODELS_TO_RUN = ['RANDOM_FOREST', 'RBF_SVM', 'DEEP_MLP']
SKIP_COMPLETED_MODELS = True
MODEL_CONFIGS = {'LOGISTIC_REGRESSION': {'role': 'locked primary reference', 'hyperparameters': {'penalty': 'L2', 'C': 1.0, 'class_weight': 'balanced', 'solver': 'liblinear'}}, 'RANDOM_FOREST': {'role': 'nonlinear tree-ensemble robustness baseline', 'hyperparameters': {'n_estimators': 100, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'class_weight': 'balanced_subsample', 'random_state': SEED, 'n_jobs': -1}}, 'RBF_SVM': {'role': 'nonlinear kernel robustness baseline', 'hyperparameters': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'class_weight': 'balanced', 'probability': True, 'random_state': SEED}}, 'DEEP_MLP': {'role': 'three-hidden-layer feed-forward neural-network robustness baseline', 'hyperparameters': {'hidden_layer_sizes': [64, 32, 16], 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001, 'learning_rate_init': 0.001, 'max_iter': 400, 'random_state': SEED}}}

def load_final_protocol_module():
    if not FINAL_SCRIPT.exists():
        raise FileNotFoundError(f'Final protocol script not found:\n{FINAL_SCRIPT}')
    spec = importlib.util.spec_from_file_location('final_crossed_protocol_module', FINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not import:\n{FINAL_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def prepare_features(X_raw):
    X_raw = np.asarray(X_raw, dtype=np.float64)
    medians = np.nanmedian(X_raw, axis=0)
    usable = np.isfinite(medians)
    if usable.sum() == 0:
        raise RuntimeError('All features are non-finite in this training fold.')
    medians_used = medians[usable]
    X_used = X_raw[:, usable].copy()
    missing = ~np.isfinite(X_used)
    if missing.any():
        r, c = np.where(missing)
        X_used[r, c] = medians_used[c]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_used)
    return (usable, medians_used, scaler, X_scaled)

def transform_features(fitted, X_raw):
    X_raw = np.asarray(X_raw, dtype=np.float64)
    X_used = X_raw[:, fitted['usable']].copy()
    missing = ~np.isfinite(X_used)
    if missing.any():
        r, c = np.where(missing)
        X_used[r, c] = fitted['medians'][c]
    return fitted['scaler'].transform(X_used)

def deterministic_balanced_oversample(X, y, seed=SEED):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return (X, y)
    target = max(len(pos), len(neg))
    rng = np.random.default_rng(seed)
    if len(pos) < target:
        pos = np.concatenate([pos, rng.choice(pos, target - len(pos), replace=True)])
    if len(neg) < target:
        neg = np.concatenate([neg, rng.choice(neg, target - len(neg), replace=True)])
    idx = rng.permutation(np.concatenate([pos, neg]))
    return (X[idx], y[idx])

def make_model(model_name):
    if model_name == 'RANDOM_FOREST':
        return RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_leaf=2, class_weight='balanced_subsample', random_state=SEED, n_jobs=-1)
    if model_name == 'RBF_SVM':
        return SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=SEED)
    if model_name == 'DEEP_MLP':
        return MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=400, random_state=SEED)
    raise ValueError(f'Unknown model: {model_name}')

def build_fit_function(model_name):

    def fit_binary_detector(X_raw, y):
        X_raw = np.asarray(X_raw, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) != 2:
            raise RuntimeError('Binary training data contains only one class.')
        usable, medians, scaler, X_scaled = prepare_features(X_raw)
        model = make_model(model_name)
        fit_note = ''
        if model_name == 'DEEP_MLP':
            weights = compute_sample_weight(class_weight='balanced', y=y)
            try:
                sig = inspect.signature(model.fit)
                if 'sample_weight' in sig.parameters:
                    model.fit(X_scaled, y, sample_weight=weights)
                    fit_note = 'balanced_sample_weight'
                else:
                    Xb, yb = deterministic_balanced_oversample(X_scaled, y)
                    model.fit(Xb, yb)
                    fit_note = 'deterministic_balanced_oversampling'
            except (TypeError, ValueError):
                Xb, yb = deterministic_balanced_oversample(X_scaled, y)
                model.fit(Xb, yb)
                fit_note = 'deterministic_balanced_oversampling'
        else:
            model.fit(X_scaled, y)
        return {'usable': usable, 'medians': medians, 'scaler': scaler, 'model': model, 'model_name': model_name, 'fit_note': fit_note}
    return fit_binary_detector

def common_predict_binary_detector(fitted, X_raw):
    X_scaled = transform_features(fitted, X_raw)
    model = fitted['model']
    if not hasattr(model, 'predict_proba'):
        raise RuntimeError(f'{type(model).__name__} has no predict_proba.')
    return np.asarray(model.predict_proba(X_scaled)[:, 1], dtype=float)

def make_save_config_function(module, model_name):

    def save_config(*args, **kwargs):
        config = {'experiment': 'classifier_robustness_crossed_protocol', 'model': model_name, 'scientific_role': MODEL_CONFIGS[model_name]['role'], 'hyperparameters': MODEL_CONFIGS[model_name]['hyperparameters'], 'feature_source': 'CURATED_SIGNATURE_DEFINITION.csv', 'preprocessing': 'outer/inner-training median imputation + StandardScaler', 'outer_protocol': 'identical crossed composition-context-condition protocol as locked LR', 'inner_threshold_calibration': {'method': 'GroupKFold by condition_key', 'n_splits_max': int(module.INNER_GROUP_FOLDS), 'threshold_grid': module.THRESHOLD_GRID.tolist(), 'primary': 'F1', 'secondary': 'balanced_accuracy', 'final_tie_break': 'closest_to_0.50', 'outer_test_used': False}, 'sensitivity_analysis': True, 'model_selection_from_outer_results': False, 'hyperparameter_tuning_from_outer_results': False, 'interpretation': 'Robustness analysis only. Alternative models are not used to replace or select the locked primary logistic classifier.'}
        with open(module.RESULT_DIR / 'experiment_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    return save_config

def run_one_model(module, model_name):
    result_dir = OUT_ROOT / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / 'SUMMARY_BY_PROTOCOL.csv'
    if SKIP_COMPLETED_MODELS and summary_path.exists():
        print('\n' + '=' * 145)
        print(f'{model_name}: completed result already exists -> SKIP')
        print('=' * 145)
        return
    print('\n' + '=' * 145)
    print(f'MODEL ROBUSTNESS RUN: {model_name}')
    print('=' * 145)
    print(json.dumps(MODEL_CONFIGS[model_name]['hyperparameters'], indent=2))
    module.RESULT_DIR = result_dir
    module.fit_binary_detector = build_fit_function(model_name)
    module.predict_binary_detector = common_predict_binary_detector
    module.save_config = make_save_config_function(module, model_name)
    t0 = time.time()
    module.main()
    elapsed = (time.time() - t0) / 60.0
    (result_dir / 'runtime_minutes.txt').write_text(f'{elapsed:.6f}\n', encoding='utf-8')
    print(f'\n{model_name} completed in {elapsed:.2f} min.')

def read_model_summary(model_name, directory):
    path = directory / 'SUMMARY_BY_PROTOCOL.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing summary:\n{path}')
    df = pd.read_csv(path)
    df.insert(0, 'model', model_name)
    return df

def build_combined_summary():
    frames = [read_model_summary('LOGISTIC_REGRESSION', LOCKED_LR_DIR)]
    for model_name in MODELS_TO_RUN:
        frames.append(read_model_summary(model_name, OUT_ROOT / model_name))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(OUT_ROOT / 'COMBINED_SUMMARY_BY_PROTOCOL.csv', index=False, encoding='utf-8-sig')
    return combined
PUBLICATION_METRICS = ['mechanism_active_macro_f1', 'mechanism_micro_f1', 'exact_match', 'constituent_recall', 'false_additions_per_run', 'any_constituent_recovered_rate', 'all_constituents_recovered_rate']

def build_publication_table(combined):
    table = combined[['model', 'condition_mode', 'context_regime', *PUBLICATION_METRICS]].copy()
    table = table.rename(columns={'mechanism_active_macro_f1': 'active_macro_f1', 'mechanism_micro_f1': 'micro_f1', 'exact_match': 'EM', 'constituent_recall': 'R_const', 'false_additions_per_run': 'FA_per_run', 'any_constituent_recovered_rate': 'R_any', 'all_constituents_recovered_rate': 'R_all'})
    table.to_csv(OUT_ROOT / 'PUBLICATION_MODEL_COMPARISON.csv', index=False, encoding='utf-8-sig')
    return table

def get_protocol_row(sub, context_regime):
    row = sub[sub['context_regime'] == context_regime]
    return row.iloc[0] if len(row) == 1 else None

def build_context_effects(combined):
    rows = []
    for model_name, model_df in combined.groupby('model', sort=False):
        for condition_mode, sub in model_df.groupby('condition_mode', sort=False):
            recomb = get_protocol_row(sub, 'RECOMBINATION')
            one = get_protocol_row(sub, 'ONE_SIDED_CONTEXT_ZS')
            two = get_protocol_row(sub, 'TWO_SIDED_CONTEXT_ZS')
            if recomb is None:
                continue
            for metric in ['mechanism_micro_f1', 'constituent_recall', 'exact_match', 'false_additions_per_run']:
                rec = float(recomb[metric])
                if one is not None:
                    rows.append({'model': model_name, 'condition_mode': condition_mode, 'metric': metric, 'contrast': 'RECOMBINATION - ONE_SIDED_CONTEXT_ZS', 'difference': rec - float(one[metric])})
                if two is not None:
                    rows.append({'model': model_name, 'condition_mode': condition_mode, 'metric': metric, 'contrast': 'RECOMBINATION - TWO_SIDED_CONTEXT_ZS', 'difference': rec - float(two[metric])})
    effects = pd.DataFrame(rows)
    effects.to_csv(OUT_ROOT / 'CONTEXT_EFFECTS_BY_MODEL.csv', index=False, encoding='utf-8-sig')
    return effects

def build_model_ranking(combined):
    ranking = combined.groupby('model', as_index=False).agg(mean_active_macro_f1=('mechanism_active_macro_f1', 'mean'), mean_micro_f1=('mechanism_micro_f1', 'mean'), mean_exact_match=('exact_match', 'mean'), mean_constituent_recall=('constituent_recall', 'mean'), mean_FA_per_run=('false_additions_per_run', 'mean'), mean_R_any=('any_constituent_recovered_rate', 'mean'), mean_R_all=('all_constituents_recovered_rate', 'mean'))
    ranking.to_csv(OUT_ROOT / 'MODEL_RANKING_DESCRIPTIVE.csv', index=False, encoding='utf-8-sig')
    return ranking

def save_global_config():
    config = {'experiment': 'classifier_robustness_of_crossed_constituent_generalization', 'status': 'classifier robustness sensitivity analysis', 'primary_locked_model': 'LOGISTIC_REGRESSION', 'alternative_models': MODELS_TO_RUN, 'model_configs': MODEL_CONFIGS, 'same_across_all_models': ['curated mechanism-specific feature map', 'physical runs and constituent labels', 'outer crossed context/condition split definitions', 'training-only median imputation', 'training-only StandardScaler', 'grouped inner-CV threshold calibration', 'threshold grid/objective', 'evaluation metrics'], 'forbidden_interpretation': 'Do not select a new primary model or retune hyperparameters from these final outer results.', 'intended_interpretation': 'Assess whether qualitative generalization limitations observed with the locked LR persist under different classifier families.'}
    with open(OUT_ROOT / 'experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def print_key_tables(publication, effects, ranking):
    print('\n' + '=' * 165)
    print('PUBLICATION MODEL COMPARISON')
    print('=' * 165)
    cols = ['model', 'condition_mode', 'context_regime', 'micro_f1', 'EM', 'R_const', 'FA_per_run', 'R_any', 'R_all']
    print(publication[cols].to_string(index=False))
    print('\n' + '=' * 165)
    print('DESCRIPTIVE CONTEXT EFFECTS')
    print('=' * 165)
    show = effects[effects['metric'].isin(['mechanism_micro_f1', 'constituent_recall'])]
    print(show.to_string(index=False))
    print('\n' + '=' * 165)
    print('DESCRIPTIVE MODEL AVERAGES ACROSS SIX SCENARIOS')
    print('=' * 165)
    print(ranking.to_string(index=False))

def main():
    print('\n' + '=' * 145)
    print('CLASSIFIER ROBUSTNESS BASELINES')
    print('=' * 145)
    print('Sensitivity analysis; primary model remains fixed.')
    print(f'Locked LR results reused from:\n  {LOCKED_LR_DIR}')
    if not (LOCKED_LR_DIR / 'SUMMARY_BY_PROTOCOL.csv').exists():
        raise FileNotFoundError('Locked LR SUMMARY_BY_PROTOCOL.csv not found.')
    print('\nAlternative models frozen before running:')
    for name in MODELS_TO_RUN:
        print(f'\n{name}')
        print(json.dumps(MODEL_CONFIGS[name]['hyperparameters'], indent=2))
    module = load_final_protocol_module()
    for model_name in MODELS_TO_RUN:
        run_one_model(module, model_name)
    combined = build_combined_summary()
    publication = build_publication_table(combined)
    effects = build_context_effects(combined)
    ranking = build_model_ranking(combined)
    save_global_config()
    print_key_tables(publication, effects, ranking)
    print('\n' + '=' * 145)
    print('DONE')
    print('=' * 145)
    print(f'Outputs:\n{OUT_ROOT}')
    print('\n')
if __name__ == '__main__':
    main()
