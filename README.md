# Constituent-Level Generalization in Compound Motor Fault Diagnosis

Code accompanying the manuscript:

**Beyond Closed-Set Accuracy: Constituent-Level Generalization in Compound Motor Fault Diagnosis**

This repository contains a curated set of analysis scripts for the principal constituent-level evaluation, mechanism-specific diagnostics, robustness analyses, prior-method comparison, and computational-cost characterization reported in the study.

## Dataset

The raw MCC5-THU motor dataset is not redistributed here.

Dataset DOI:

`https://doi.org/10.17632/6s3dggj9mw.1`

Download the dataset separately and define the local project root in the scripts before execution.

## Repository structure

```text
01_core/
02_mechanism_analysis/
03_sensitivity/
    sampling_rate/
04_prior_method/
05_computational_cost/
docs/
```

The repository focuses on the analysis scripts required for the reported results and excludes temporary development utilities and publication-figure generation code.

## 1. Core analysis

Recommended order:

1. `01_core/EXPERIMENT_PROTOCOL.py`
2. `01_core/DIAGNOSE_COMPOSITIONAL_PHYSICS_v2.py`
3. `01_core/FINAL_CROSSED_COMPOSITION_CONDITION_EVALUATION.py`
4. `01_core/TRAIN_ALL_CLASS_MULTIMODAL.py`

The crossed evaluator implements the recombination, one-sided context zero-shot, and two-sided context zero-shot regimes under seen and unseen operating conditions. Preprocessing and threshold calibration are restricted to training data within each evaluation fold.

## 2. Mechanism-specific analyses

- `TEST_A2_CURATED_PHYSICS_SIGNATURES_v2.py` — physical-signature preservation in compounds.
- `ANALYZE_SINGLE_FAULT_LEARNABILITY_VS_COMPOUND.py` — isolated-fault learnability versus compound recovery.
- `TEST_CROSS_PARTNER_TRANSFER_ORACLE_FIXED.py` — cross-partner transfer analysis.
- `COMPOSITION_LEVEL_FULL_AUDIT.py` — composition-cluster statistical analysis for the nine compound compositions.

## 3. Sensitivity analyses

- `RUN_CLASSIFIER_ROBUSTNESS_BASELINES.py` — alternative classifier families under the same crossed protocol.
- `RUN_SIZE_MATCHED_RECOMBINATION_ABLATION.py` — training-size-matched recombination analysis.
- `RUN_REMAINING_PAPER_ANALYSES.py` — severity-support sensitivity and dataset consistency checks.

### Sampling-rate sensitivity

Run in this order:

1. `BUILD_6P4KHZ_SENSITIVITY_DATASET.py`
2. `DIAGNOSE_COMPOSITIONAL_PHYSICS_6P4KHZ.py`
3. `FINAL_CROSSED_COMPOSITION_CONDITION_EVALUATION_6P4KHZ.py`
4. `COMPARE_12P8_VS_6P4_SAMPLING_SENSITIVITY.py`

The 6.4-kHz analysis uses anti-aliased downsampling and independently reruns feature extraction, fitting, and threshold calibration while enforcing the feature map locked by the native 12.8-kHz analysis.

## 4. Prior-method comparison

Run:

1. `RUN_PROTOCOL_MATCHED_ZLCFDM_STYLE_BASELINE.py`
2. `COMPARE_CANDIDATE_CONSTRAINED_DECODING.py`

The baseline is a protocol-matched adaptation of the semantic-prototype zero-shot principle used in prior compound-fault work. It is not an exact reproduction of the original CNN/time-frequency architecture. The candidate-constrained decoder is an output-space comparison only and does not replace the unrestricted primary constituent decoder.

## 5. Computational cost

`BENCHMARK_CONSTITUENT_INFERENCE_COST.py` characterizes the run-level feature-to-decision classifier stage. It does not include raw-signal loading, angle resampling, spectral analysis, or physics-feature extraction.

## Local paths

The research scripts retain an explicit local `ROOT = Path(...)` setting. Replace it with your own project root before execution, for example:

```python
from pathlib import Path
ROOT = Path(r"D:\path\to\your\MCC5-THU-project")
```

The paths were intentionally not refactored in this release so that the scientifically verified analysis logic remains unchanged.

## Reproducibility principles

The released workflow preserves the main safeguards used in the manuscript:

- physical-run-level separation;
- training-only imputation and standardization;
- grouped inner-CV threshold calibration;
- target-condition exclusion before unseen-condition fitting;
- no outer-test tuning of the locked primary analysis;
- composition-level resampling where inference must account for the nine underlying compound compositions.

## Requirements

Install the packages listed in `requirements.txt`.

## License

The GitHub repository uses the MIT License. The MCC5-THU data remain subject to the dataset authors' own license and distribution terms.

## Citation

The final article citation will be added after publication.
