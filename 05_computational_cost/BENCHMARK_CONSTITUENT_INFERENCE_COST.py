"""Computational-cost benchmark for constituent-level inference.

The benchmark measures the run-level feature-to-decision cost of the locked
nine-detector logistic-regression architecture, including median imputation,
standardization, linear scoring, sigmoid evaluation, and constituent-specific
thresholding.

Synthetic numeric inputs are used only to measure arithmetic and runtime cost.
Raw-signal loading, window extraction, keyphase processing, angle resampling,
spectral analysis, and physics-feature extraction are excluded.
"""
from pathlib import Path
import json
import math
import os
import platform
import statistics
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path('D:\\yay\u0131nlar\\Multi-mode Fault Diagnosis Datasets of Three-phase')
OUT = ROOT / 'COMPUTATIONAL_COST'
OUT.mkdir(parents=True, exist_ok=True)
DETECTOR_DIMS = {'bearing_ball': 16, 'bearing_inner': 16, 'bearing_outer': 16, 'bend': 16, 'broken_bar': 40, 'dynamic_eccentricity': 56, 'static_eccentricity': 56, 'voltage_unbalance': 2, 'winding': 12}
SEED = 20260908
N_WARMUP = 5000
N_LATENCY = 50000
BATCH_SIZE = 10000
N_BATCH_REPEATS = 20
DTYPE = np.float64

def stable_sigmoid(x):
    """
    Numerically stable scalar sigmoid.
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def create_detector_state(seed=SEED):
    """
    Create deterministic synthetic detector state.

    Values are synthetic because the benchmark measures arithmetic/runtime
    cost, not diagnostic accuracy.

    For each constituent:
        median  : training-only imputation state
        mean    : standardization mean
        scale   : standardization scale
        weight  : LR coefficient vector
        bias    : LR intercept
        threshold : constituent decision threshold
    """
    rng = np.random.default_rng(seed)
    state = {}
    for name, dim in DETECTOR_DIMS.items():
        median = rng.normal(loc=0.0, scale=1.0, size=dim).astype(DTYPE)
        mean = rng.normal(loc=0.0, scale=1.0, size=dim).astype(DTYPE)
        scale = rng.uniform(low=0.5, high=2.0, size=dim).astype(DTYPE)
        weight = rng.normal(loc=0.0, scale=0.25, size=dim).astype(DTYPE)
        bias = float(rng.normal(loc=0.0, scale=0.1))
        threshold = float(rng.uniform(low=0.2, high=0.8))
        state[name] = {'median': median, 'mean': mean, 'scale': scale, 'weight': weight, 'bias': bias, 'threshold': threshold}
    return state

def create_one_run(seed=SEED + 1):
    """
    Generate one synthetic run-level mechanism-specific input set.

    A small fraction of values are NaN so the actual imputation operation
    is included in the timing.
    """
    rng = np.random.default_rng(seed)
    run = {}
    for name, dim in DETECTOR_DIMS.items():
        x = rng.normal(loc=0.0, scale=1.0, size=dim).astype(DTYPE)
        mask = rng.random(dim) < 0.02
        x[mask] = np.nan
        run[name] = x
    return run

def predict_one_run(run, state):
    """
    Full feature-to-decision stage for nine constituent detectors.

    Includes:
        median imputation
        standardization
        LR score
        sigmoid
        thresholding
    """
    predictions = {}
    for name in DETECTOR_DIMS:
        x = run[name]
        s = state[name]
        x_clean = np.where(np.isnan(x), s['median'], x)
        z = (x_clean - s['mean']) / s['scale']
        score = float(np.dot(z, s['weight'])) + s['bias']
        probability = stable_sigmoid(score)
        prediction = int(probability >= s['threshold'])
        predictions[name] = prediction
    return predictions

def create_batch(batch_size, seed=SEED + 2):
    """
    Synthetic batch for throughput measurement.
    """
    rng = np.random.default_rng(seed)
    batch = {}
    for name, dim in DETECTOR_DIMS.items():
        x = rng.normal(loc=0.0, scale=1.0, size=(batch_size, dim)).astype(DTYPE)
        mask = rng.random(size=x.shape) < 0.02
        x[mask] = np.nan
        batch[name] = x
    return batch

def predict_batch(batch, state):
    """
    Vectorized batch feature-to-decision inference.
    """
    predictions = {}
    for name in DETECTOR_DIMS:
        x = batch[name]
        s = state[name]
        x_clean = np.where(np.isnan(x), s['median'][None, :], x)
        z = (x_clean - s['mean'][None, :]) / s['scale'][None, :]
        scores = z @ s['weight'] + s['bias']
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        predictions[name] = (probabilities >= s['threshold']).astype(np.uint8)
    return predictions

def benchmark_single_run_latency(run, state):
    """
    Measure latency for ONE run-level feature representation.

    Each timed call produces all nine constituent decisions.
    """
    print('\nWarming up single-run inference...')
    for _ in range(N_WARMUP):
        predict_one_run(run, state)
    print(f'Warm-up complete: {N_WARMUP:,} runs')
    print(f'\nMeasuring {N_LATENCY:,} single-run inference calls...')
    times_ns = np.empty(N_LATENCY, dtype=np.int64)
    for i in range(N_LATENCY):
        start = time.perf_counter_ns()
        predict_one_run(run, state)
        end = time.perf_counter_ns()
        times_ns[i] = end - start
    times_ms = times_ns.astype(np.float64) / 1000000.0
    result = {'benchmark': 'single_run_feature_to_decision', 'n_measurements': N_LATENCY, 'median_ms': float(np.median(times_ms)), 'mean_ms': float(np.mean(times_ms)), 'std_ms': float(np.std(times_ms)), 'p05_ms': float(np.percentile(times_ms, 5)), 'p95_ms': float(np.percentile(times_ms, 95)), 'p99_ms': float(np.percentile(times_ms, 99)), 'min_ms': float(np.min(times_ms)), 'max_ms': float(np.max(times_ms))}
    if result['median_ms'] > 0:
        result['median_runs_per_second'] = 1000.0 / result['median_ms']
    else:
        result['median_runs_per_second'] = np.nan
    return result

def benchmark_batch_throughput(batch, state):
    print('\nWarming up vectorized batch inference...')
    for _ in range(5):
        predict_batch(batch, state)
    elapsed = []
    print('\nBatch throughput benchmark:')
    print(f'  batch size : {BATCH_SIZE:,}')
    print(f'  repetitions: {N_BATCH_REPEATS}')
    for i in range(N_BATCH_REPEATS):
        start = time.perf_counter()
        predict_batch(batch, state)
        end = time.perf_counter()
        elapsed.append(end - start)
    elapsed = np.asarray(elapsed, dtype=np.float64)
    runs_per_second = BATCH_SIZE / elapsed
    per_run_ms = elapsed / BATCH_SIZE * 1000.0
    return {'benchmark': 'vectorized_batch_feature_to_decision', 'batch_size': BATCH_SIZE, 'n_measurements': N_BATCH_REPEATS, 'median_batch_seconds': float(np.median(elapsed)), 'median_per_run_ms': float(np.median(per_run_ms)), 'median_runs_per_second': float(np.median(runs_per_second)), 'p05_runs_per_second': float(np.percentile(runs_per_second, 5)), 'p95_runs_per_second': float(np.percentile(runs_per_second, 95))}

def theoretical_cost():
    """
    Compute transparent operation/state counts from the locked input sizes.
    """
    total_features = int(sum(DETECTOR_DIMS.values()))
    n_detectors = len(DETECTOR_DIMS)
    lr_weights = total_features
    lr_intercepts = n_detectors
    thresholds = n_detectors
    preprocessing_values = 3 * total_features
    total_stored_scalars = lr_weights + lr_intercepts + thresholds + preprocessing_values
    float64_bytes = total_stored_scalars * 8
    float32_bytes = total_stored_scalars * 4
    rows = []
    for name, dim in DETECTOR_DIMS.items():
        rows.append({'item': name, 'input_dimension': dim, 'lr_weights': dim, 'lr_intercepts': 1, 'thresholds': 1, 'preprocessing_scalars': 3 * dim})
    rows.append({'item': 'TOTAL', 'input_dimension': total_features, 'lr_weights': lr_weights, 'lr_intercepts': lr_intercepts, 'thresholds': thresholds, 'preprocessing_scalars': preprocessing_values})
    df = pd.DataFrame(rows)
    summary = {'n_binary_detectors': n_detectors, 'total_detector_specific_feature_inputs': total_features, 'lr_weight_coefficients': lr_weights, 'lr_intercepts': lr_intercepts, 'decision_thresholds': thresholds, 'preprocessing_scalars': preprocessing_values, 'total_stored_numeric_scalars': total_stored_scalars, 'approx_state_bytes_float64': float64_bytes, 'approx_state_kib_float64': float64_bytes / 1024.0, 'approx_state_bytes_float32': float32_bytes, 'approx_state_kib_float32': float32_bytes / 1024.0, 'dot_product_multiplications_per_run': total_features, 'dot_product_and_bias_additions_per_run': total_features, 'sigmoid_evaluations_per_run': n_detectors, 'threshold_comparisons_per_run': n_detectors}
    return (df, summary)

def get_environment_info():
    info = {'python_version': sys.version.replace('\n', ' '), 'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(), 'cpu_count_logical': os.cpu_count(), 'numpy_version': np.__version__, 'pandas_version': pd.__version__, 'dtype': str(np.dtype(DTYPE))}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info['ram_total_gib'] = vm.total / 1024 ** 3
    except Exception:
        info['ram_total_gib'] = np.nan
    return info

def print_results(single_result, batch_result, theory_summary, environment):
    print('\n' + '=' * 120)
    print('COMPUTATIONAL-COST BENCHMARK RESULTS')
    print('=' * 120)
    print('\nHardware / software:')
    print(f"  Processor        : {environment['processor']}")
    print(f"  Logical CPUs     : {environment['cpu_count_logical']}")
    print(f'  Python           : {platform.python_version()}')
    print(f'  NumPy            : {np.__version__}')
    print(f'  dtype            : {np.dtype(DTYPE)}')
    print('\nLocked constituent architecture:')
    print(f"  Binary detectors : {theory_summary['n_binary_detectors']}")
    print(f"  Feature inputs   : {theory_summary['total_detector_specific_feature_inputs']}")
    print(f"  LR coefficients : {theory_summary['lr_weight_coefficients']}")
    print(f"  Total state      : {theory_summary['approx_state_kib_float64']:.3f} KiB (float64, including preprocessing state and thresholds)")
    print('\nSingle-run feature-to-decision latency:')
    print(f"  Median : {single_result['median_ms']:.6f} ms")
    print(f"  Mean   : {single_result['mean_ms']:.6f} ms")
    print(f"  p95    : {single_result['p95_ms']:.6f} ms")
    print(f"  p99    : {single_result['p99_ms']:.6f} ms")
    print(f"  Median equivalent throughput: {single_result['median_runs_per_second']:,.1f} runs/s")
    print('\nVectorized throughput:')
    print(f"  Median per-run equivalent: {batch_result['median_per_run_ms']:.6f} ms")
    print(f"  Median throughput         : {batch_result['median_runs_per_second']:,.1f} runs/s")
    print('\nBENCHMARK SCOPE:')
    print('These timings measure ONLY the run-level feature-to-decision classifier stage.')
    print('They do NOT include raw-signal loading, angle resampling, spectral analysis, or physics-feature extraction.')

def main():
    print('\n' + '=' * 120)
    print('CONSTITUENT-LEVEL COMPUTATIONAL-COST BENCHMARK')
    print('=' * 120)
    print('\nFor stable timing measurements, run this benchmark on an otherwise idle system.')
    print('\nOutput:')
    print(OUT)
    state = create_detector_state()
    one_run = create_one_run()
    batch = create_batch(BATCH_SIZE)
    prediction = predict_one_run(one_run, state)
    if len(prediction) != len(DETECTOR_DIMS):
        raise RuntimeError('Inference sanity check failed.')
    theory_df, theory_summary = theoretical_cost()
    single_result = benchmark_single_run_latency(one_run, state)
    batch_result = benchmark_batch_throughput(batch, state)
    environment = get_environment_info()
    summary_df = pd.DataFrame([single_result, batch_result])
    summary_path = OUT / 'COMPUTATIONAL_COST_SUMMARY.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    theoretical_path = OUT / 'COMPUTATIONAL_COST_THEORETICAL.csv'
    theory_df.to_csv(theoretical_path, index=False, encoding='utf-8-sig')
    environment_df = pd.DataFrame([environment])
    environment_path = OUT / 'COMPUTATIONAL_COST_ENVIRONMENT.csv'
    environment_df.to_csv(environment_path, index=False, encoding='utf-8-sig')
    config = {'benchmark_scope': 'run-level feature-to-decision classifier stage only', 'included_operations': ['median imputation', 'standardization', 'nine logistic-regression linear scores', 'nine sigmoid evaluations', 'nine constituent-specific threshold comparisons'], 'excluded_operations': ['raw-signal file loading', 'window extraction', 'key-phase processing', 'angle resampling', 'periodogram calculation', 'Hilbert-envelope calculation', 'physics-feature extraction'], 'detector_dimensions': DETECTOR_DIMS, 'total_detector_specific_features': sum(DETECTOR_DIMS.values()), 'warmup_runs': N_WARMUP, 'single_run_latency_repeats': N_LATENCY, 'batch_size': BATCH_SIZE, 'batch_repeats': N_BATCH_REPEATS, 'random_seed': SEED, 'single_run_result': single_result, 'batch_result': batch_result, 'theoretical_summary': theory_summary, 'environment': environment}
    config_path = OUT / 'computational_cost_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print_results(single_result, batch_result, theory_summary, environment)
    print('\nGenerated files:')
    for p in sorted(OUT.glob('*')):
        print(' -', p.name)
    print('\nDone.')
if __name__ == '__main__':
    main()
