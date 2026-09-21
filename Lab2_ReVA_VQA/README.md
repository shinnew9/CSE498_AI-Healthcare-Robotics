# Lab 2 — ReVA VQA: Qwen3-VL Fine-Tuning and VILA Comparison

CSE 498 · AI Healthcare Robotics
Assignment repository: https://github.com/likaiw2/VLM_evaluating_finetuing

## Results

| Model | Accuracy | Questions | Completion |
|---|---|---|---|
| Qwen3-VL-4B-Instruct (LoRA fine-tuned) | **70.94%** | 144/203 | 100% |
| Qwen3-VL-4B-Instruct (base) | **70.44%** | 143/203 | 100% |
| VILA1.5-3B (base) | **55.17%** | 112/203 | 100% |

Evaluation set: a 50-video / 203-question subset of the official ReVA `test_set.json`.
All three models were evaluated on identical questions with `MAX_FRAMES=4`.

## Contents

| Path | Description |
|---|---|
| `lab2.ipynb` | Full report — commands executed, results, qualitative analysis, environment issues |
| `code/` | The four files containing the completed `TODO(student)` functions |
| `results/` | Evaluation outputs for all three models |
| `infra_fixes.diff` | Seven fixes to the provided infrastructure code (library-version and GPU-architecture compatibility) |
| `zero3_offload.json` | DeepSpeed ZeRO-3 config with CPU parameter offload, added to fit training on an 11 GB GPU |

### Completed `TODO(student)` functions

| File | Functions |
|---|---|
| `code/build_qwen_train_data.py` | `format_options`, `format_question`, `format_answer`, `iter_reva_qas`, `convert_item` |
| `code/prepare_reva_v2_test_set.py` | `to_abs_path`, `to_rel_stem`, `get_qa_id`, `iter_flat_items` |
| `code/score_reva_predictions.py` | `extract_answer`, `extract_letter`, `load_predictions`, `score_predictions` |
| `code/vila_eval_reva_v2.py` | `load_instances`, `resolve_video_path`, `build_prompt`, `parse_choice`, `summarize` |

Unit tests (`pytest -q`): 7 passed

## Environment notes

Training and evaluation ran on shared RTX 2080 Ti GPUs (11 GB, Turing). This required:

- FlashAttention 2 → SDPA (FlashAttention 2 requires Ampere or newer)
- API updates for transformers 5.17.0 (`AutoModelForVision2Seq`, `warmup_ratio`, `apply_multimodal_rotary_pos_emb` were removed)
- Evaluation batch size reduced from the hard-coded 8 to 2 / 1
- DeepSpeed ZeRO-3 with CPU parameter offload for training

Full details, including every deviation from the recommended settings, are in §5 and §6 of `lab2.ipynb`.

## Not included

- ReVA dataset annotations and videos (see the ReVA dataset page for licensing)
- The LoRA training checkpoint (94 MB); it is not a required deliverable
