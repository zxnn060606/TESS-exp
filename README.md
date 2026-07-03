# TESS RC2

## Overview

TESS-RC2 contains the current primitive inference cache flow plus RC2 training and evaluation for numeric and primitive-enhanced time-series forecasting. The main local workflow is: prepare raw dataset splits, build or load grouped primitive gate caches, inspect the joined dataset, train numeric or primitive-aware models, and compare run `summary.json` files.

## Data Layout

Expected dataset and primitive cache paths:

```text
dataset/{dataset}/{split}.json
data_cache/gate_grouped/{dataset}/{split}.json
```

`primitive_inference_rc2/tess_dataset.py` reads raw `x`/`y` values from `dataset/{dataset}/{split}.json` and reads primitive labels, margins, masks, and gate targets from `data_cache/gate_grouped/{dataset}/{split}.json`.

## Environment

Known local environment style:

```bash
source /home/lyc/anaconda3/etc/profile.d/conda.sh
conda activate tslib
cd /home/lyc/workspace/TESS-RC2
```

## Inspect Dataset

```bash
python -m experiments.inspect_tess_dataset \
  --root /home/lyc/workspace/TESS-RC2 \
  --dataset fnspid \
  --output-json outputs/tess_basic/real_fnspid_legacy_standard/inspect_report.json
```

## Metric Scale

Use `--scale legacy_standard` for legacy-compatible metrics. This fits one global mean/std on flattened train `historical_data` only, then standardizes both `x` and `y` for train/vali/test with that train-history scaler. Main reported MSE/MAE are normalized-space metrics. Raw-scale add-on metrics may also be saved as `raw_mse` and `raw_mae`.

Use `--scale raw` only when you intentionally want raw-value training and metrics. Do not compare raw-scale metrics directly with legacy normalized metrics.

## Basic Training Command

```bash
python -m experiments.train_tess_basic \
  --root /home/lyc/workspace/TESS-RC2 \
  --dataset fnspid \
  --model legacy_multimodal_primitive \
  --primitive-source text \
  --epochs 25 \
  --batch-size 32 \
  --lr 0.0005 \
  --device cuda \
  --scale legacy_standard \
  --seed 42 \
  --output-dir outputs/tess_basic/real_fnspid_legacy_standard/example_run \
  --overwrite
```

Implemented RC2 model names:

```text
numeric_mlp
tess_nogate
tiny_temporal
tiny_temporal_tess
legacy_timesnet
legacy_multimodal_primitive
legacy_multimodal_primitive_gate
legacy_multimodal_primitive_delta_gate
```

## Primitive Inference Caches

Unified primitive pipeline entry point:

```bash
python -m primitive_inference_rc2.run_primitive_pipeline \
  --root /home/lyc/workspace/TESS-RC2 \
  --dataset fnspid \
  --primitives distribution_shift volatility shape temporal_influence \
  --splits train vali test \
  --backend mock \
  --prompt-source auto \
  --num-samples 8 \
  --seed 42 \
  --stages sampled gate grouped audit \
  --overwrite
```

Use `--backend mock` for local validation. For real LLM inference, use `--backend openai-compatible` and provide `--base-url`, `--api-key`, and `--model` for a vLLM/OpenAI-compatible server. Start with a small `--limit` smoke test before full inference.

`scripts/run_other_datasets_sampling4.sh` runs sampling-4 primitive inference for other datasets with an OpenAI-compatible backend by default. Set `DATASETS`, `ROOT`, `BASE_URL`, `API_KEY`, and `MODEL` for the target server.

## Common Scripts

- `scripts/run_tess_real_basic.sh`: runs dataset inspection and the basic real-data RC2 suite.
- `scripts/run_rc2_timesnet_fnspid_numeric.sh`: RC2 pure-numeric TimesNet on FNSPID.
- `scripts/run_legacy_timesnet_fnspid_numeric.sh`: legacy pure-numeric TimesNet on FNSPID and comparison helper.
- `scripts/run_rc2_timesnet_fnspid_full.sh`: full RC2 TimesNet numeric run with full legacy hyperparameters.
- `scripts/run_legacy_timesnet_fnspid_full.sh`: full legacy TimesNet numeric run with full legacy hyperparameters.
- `scripts/run_legacy_multimodal_primitive_fnspid.sh`: no-gate primitive model with `text`, `gt`, or both.
- `scripts/run_legacy_multimodal_primitive_gate_fnspid.sh`: direct margin-gate primitive model using text primitives.
- `scripts/run_legacy_multimodal_primitive_delta_gate_fnspid.sh`: conservative delta-gate primitive model using text primitives.
- `scripts/run_other_datasets_sampling4.sh`: real primitive inference cache generation for other datasets using sampling=4.

## Model / Primitive Source Matrix

| Model family | Models | `--primitive-source` |
| --- | --- | --- |
| Numeric only | `numeric_mlp`, `tiny_temporal`, `legacy_timesnet` | `none` |
| Primitive no-gate | `tess_nogate`, `tiny_temporal_tess`, `legacy_multimodal_primitive` | `text` or `gt` |
| Primitive gate | `legacy_multimodal_primitive_gate`, `legacy_multimodal_primitive_delta_gate` | `text` |

`gt` primitive source is oracle-only and should be used for analysis, not deployment-style evaluation.

## Recommended Workflow

1. Generate or check grouped primitive caches under `data_cache/gate_grouped/{dataset}/`.
2. Inspect the joined dataset with `experiments.inspect_tess_dataset`.
3. Run a numeric baseline with `--primitive-source none`.
4. Run primitive no-gate variants with `--primitive-source text`, then optional oracle `gt`.
5. Run direct gate and delta-gate variants.
6. Compare `summary.json` files, especially validation-selected `best_reloaded_test_mse`.

## Outputs

Each training run writes:

```text
config.json
metrics.json
summary.json
best.pt
last.pt
```

The main result is `best_reloaded_test_mse`, selected by validation MSE and recomputed after loading `best.pt`.

TimesNet comparison utility:

```bash
python -m experiments.compare_timesnet_legacy_vs_rc2 \
  --legacy-root outputs/legacy_timesnet_fnspid_full \
  --rc2-summary outputs/tess_basic/real_fnspid_legacy_standard/legacy_timesnet_full/summary.json \
  --output-json outputs/tess_basic/real_fnspid_legacy_standard/legacy_timesnet_full/legacy_vs_rc2_comparison.json
```

## Notes / Caveats

- Do not compare raw-scale metrics with legacy normalized metrics.
- `gt` primitive source is oracle-only.
- Gate targets are supervision only and should not be used as model input.
- `dataset/` and `legacy/` should generally not be modified.
- Some real vLLM scripts contain server-specific defaults; check `ROOT`, `BASE_URL`, `MODEL`, and output paths before running them.
