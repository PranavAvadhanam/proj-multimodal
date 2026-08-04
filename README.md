# AVUT Multimodal QA — PragCoT + Vanilla Baselines

Modality-weighted/modality-separated prompting strategies to avoid modality bias (ex. bias towards using the text content of an input and ignoring its audio/visual content). Evaluation with Gemini-based multiple-choice video QA on the [AVUT benchmark](https://huggingface.co/datasets/tsinghua-ee/AVUTBenchmark). See Citations section. Made with Cursor and Claude Code.

Two pipelines:

- **Idea 2 (Modality Separated PragCoT)**: perception → separated modality descriptions (ie. each description only based on one modality's content fed as input) → cross-modal reasoning → answer
- **Vanilla**: single direct video+question → answer

One key variation for each pipeline:

- Modality Importance Scoring: Generate a relative weight of each modality's importance and use this to weight the contribution of each modality's description.

Each pipeline has a **no-CoT** and a **CoT** variant (Gemini thinking tokens enabled/disabled).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=hf_your_huggingface_token
GOOGLE_CLOUD_PROJECT=your_gcp_project_id   # for Speech-to-Text transcription
```

## Scripts

### Shared CLI flags

All evaluation scripts accept:


| Flag                   | Description                                                   |
| ---------------------- | ------------------------------------------------------------- |
| `--max-samples N`      | Cap QA rows per pass (task-balanced across 6 AVUT task types) |
| `--run-sample ID`      | Run one sample end-to-end (smoke test)                        |
| `--prefetch-videos N`  | Cap distinct videos fetched from HF                           |
| `--no-prefetch-videos` | Skip HF video download entirely                               |
| `--split-max-samples`  | Run both AV-Human and AV-Gemini passes (splits N across both) |
| `--input FILE`         | Override QA JSONL (single-pass debug mode)                    |
| `--output-dir DIR`     | Override output directory                                     |


### Evaluation scripts


| Script                              | Pipeline | Thinking tokens | Output directory       | Extra flags                        |
| ----------------------------------- | -------- | --------------- | ---------------------- | ---------------------------------- |
| `python scripts/run_vanilla.py`     | Vanilla  | 0               | `outputs/vanilla/`     | —                                  |
| `python scripts/run_vanilla_cot.py` | Vanilla  | 768             | `outputs/vanilla_cot/` | —                                  |
| `python scripts/run_idea2.py`       | Idea 2   | 0               | `outputs/idea2/`       | `--mis` → `outputs/idea2_mis/`     |
| `python scripts/run_idea2_cot.py`   | Idea 2   | 768             | `outputs/idea2_cot/`   | `--mis` → `outputs/idea2_mis_cot/` |


### The `--mis` flag (Idea 2 only)

Controls how text descriptions are generated and how audio/visual token budgets are set:


| Mode                     | Text description                                           | Audio/visual budgets          |
| ------------------------ | ---------------------------------------------------------- | ----------------------------- |
| **Default** (no `--mis`) | Raw transcript passed directly (no Gemini call)            | Uniform 512 tokens each       |
| `**--mis`**              | Gemini summarizes transcript to MIS-allocated token budget | Set by MIS calibration output |


Use `--mis` only after running MIS calibration (see below). Vanilla scripts are fully independent of MIS.

## Quick start

```bash
# Smoke test — one sample
python scripts/run_vanilla.py --run-sample 1
python scripts/run_idea2.py --run-sample 1

# 20-sample eval
python scripts/run_vanilla.py --max-samples 20
python scripts/run_idea2.py --max-samples 20

# CoT variants
python scripts/run_vanilla_cot.py --max-samples 20
python scripts/run_idea2_cot.py --max-samples 20

# Idea 2 with MIS modality weighting
python scripts/run_idea2.py --max-samples 20 --mis
python scripts/run_idea2_cot.py --max-samples 20 --mis

# Full AVUT benchmark (all 1734 AV-Human samples)
python scripts/run_vanilla.py
python scripts/run_idea2.py
```

## MIS calibration (one-time setup for `--mis`)

MIS (Modality Importance Score) determines how to allocate description tokens across text, audio, and visual modalities. This is a **one-time calibration step** — run it once, and all subsequent `--mis` runs read the saved weights.

### What it does

1. Selects N held-out calibration samples (separated from evaluation)
2. Generates text/audio/visual descriptions for each sample
3. Evaluates all 7 non-empty modality subsets per sample
4. Computes per-modality importance via subset ablation
5. Converts scores to softmax-weighted token allocations

### Scripts

No-CoT and CoT calibrations are stored in **separate directories** so they don't overwrite each other:


| Script                                | Thinking tokens | Output directory   |
| ------------------------------------- | --------------- | ------------------ |
| `python scripts/run_misprompt.py`     | 0               | `outputs/mis/`     |
| `python scripts/run_misprompt_cot.py` | 768             | `outputs/mis_cot/` |


Each `run_idea2` variant automatically reads from its matching MIS directory when `--mis` is passed:


| Eval script              | Reads MIS from     |
| ------------------------ | ------------------ |
| `run_idea2.py --mis`     | `outputs/mis/`     |
| `run_idea2_cot.py --mis` | `outputs/mis_cot/` |


### How many calibration prompts?

Use `**--mis-samples N`** to set how many AV-Human QA rows are used for MIS (default **30**). Selection is task-balanced across the six AVUT task types, after excluding any IDs already seen in existing `idea2_predictions_*.jsonl` files and IDs listed in prior `mis_excluded_sample_ids.json` for that MIS directory.

```bash
# Example: MIS on 50 calibration questions (no-CoT)
python scripts/run_misprompt.py --mis-samples 50

# Example: MIS on 10 questions (cheap smoke test; CoT variant)
python scripts/run_misprompt_cot.py --mis-samples 10
```

**API cost**: each calibration sample runs **7** subset ablations in Phase 2, plus modality-description calls in Phase 1 — scale `N` to your quota.

### Usage

```bash
# Calibrate no-CoT (writes to outputs/mis/)
python scripts/run_misprompt.py --mis-samples 30

# Calibrate CoT (writes to outputs/mis_cot/)
python scripts/run_misprompt_cot.py --mis-samples 30

# Check results
cat outputs/mis/token_allocation.json
cat outputs/mis_cot/token_allocation.json

# Run Idea 2 with matching MIS weighting
python scripts/run_idea2.py --max-samples 50 --mis         # reads outputs/mis/
python scripts/run_idea2_cot.py --max-samples 50 --mis     # reads outputs/mis_cot/
```

MIS-specific flags: `--mis-samples N`, `--seed S`, `--total-token-budget B`, `--prefetch-videos N`, `--no-prefetch-videos`, `--output-dir DIR`.

### MIS outputs (per directory)


| File                           | Description                                              |
| ------------------------------ | -------------------------------------------------------- |
| `token_allocation.json`        | Softmax-weighted token budgets (text / audio / visual)   |
| `mis_results.json`             | Raw MIS scores and per-subset accuracies                 |
| `mis_detail.jsonl`             | Per-(sample, subset) correctness traces                  |
| `mis_excluded_sample_ids.json` | Sample IDs reserved for calibration (excluded from eval) |


## Outputs

Each pipeline writes to its own subdirectory under `outputs/`:

```
outputs/
├── vanilla/              # run_vanilla.py
├── vanilla_cot/          # run_vanilla_cot.py
├── idea2/                # run_idea2.py
├── idea2_cot/            # run_idea2_cot.py
├── idea2_mis/            # run_idea2.py --mis
├── idea2_mis_cot/        # run_idea2_cot.py --mis
├── mis/                  # run_misprompt.py (calibration)
├── mis_cot/              # run_misprompt_cot.py (calibration)
├── gemini_media_cache/   # shared preprocessed video cache
└── gemini_call_metrics_detailed.jsonl  # shared per-call Gemini log
```

Each subdirectory contains: `*_predictions_av_human.jsonl`, `*_metrics_av_human.json`, `audio_cache/`.

### Results summary

Generate a comparison table of the most recent run from each pipeline:

```bash
python scripts/summarize_results.py              # prints to terminal
python scripts/summarize_results.py --png        # also saves outputs/results_summary.png
```

## Configuration

All inference parameters live in `src/config.py` and are overridable via `.env`:


| Env var                                   | Default    | Description                                              |
| ----------------------------------------- | ---------- | -------------------------------------------------------- |
| `GEMINI_API_KEY`                          | (required) | Gemini API key                                           |
| `GEMINI_TEMPERATURE`                      | `0.1`      | Generation temperature                                   |
| `GEMINI_MAX_OUTPUT_TOKENS`                | `256`      | Global output token cap                                  |
| `GEMINI_MAX_OUTPUT_TOKENS_DESCRIBE`       | `512`      | Per-modality description budget (no MIS)                 |
| `GEMINI_MAX_OUTPUT_TOKENS_IDEA2_ANSWER`   | `256`      | Idea 2 final answer cap                                  |
| `GEMINI_MAX_OUTPUT_TOKENS_VANILLA_ANSWER` | `256`      | Vanilla answer cap                                       |
| `GEMINI_THINKING_BUDGET`                  | `0`        | Global thinking tokens (overridden by scripts)           |
| `GEMINI_THINKING_BUDGET_IDEA2`            | `768`      | Idea 2 reasoning thinking tokens (overridden by scripts) |
| `VIDEO_SAMPLE_FPS`                        | `2`        | ffmpeg downsampling fps                                  |
| `VIDEO_SAMPLE_FPS_ALIGNMENT`              | `3`        | fps for alignment tasks (AVOM, AVSM, AEL)                |
| `VIDEO_MAX_WIDTH`                         | `640`      | Max frame width                                          |
| `VIDEO_CRF`                               | `20`       | H.264 quality (lower = better)                           |
| `FORMAT_RETRY_ATTEMPTS`                   | `3`        | Answer-format parse retries                              |
| `MAX_REPAIR_ATTEMPTS`                     | `3`        | Repair-prompt retries                                    |
| `MAX_AUDIO_DURATION_SECONDS`              | `60`       | Max audio length for STT                                 |
| `HF_TOKEN`                                | —          | Hugging Face Hub token                                   |
| `GOOGLE_CLOUD_PROJECT`                    | —          | GCP project for Speech-to-Text v2                        |

Citations
Bhosale, S., Chaudhuri, A., Williams, A.L.R., Tiwari, D., Dutta, A., Zhu, X., Bhattacharyya, P., Kanojia, D. (2023). Sarcasm in Sight and Sound: Benchmarking and Expansion to Improve Multimodal Sarcasm Detection. arXiv preprint arXiv:2310.01430.

Castro, S., Hazarika, D., Pérez-Rosas, V., Zimmermann, R., Mihalcea, R., Poria, S. (2019). Towards Multimodal Sarcasm Detection (An _Obviously_ Perfect Paper). arXiv preprint arXiv:1906.01815.

Chaubey, A., Pang, J., Siniukov, M., Soleymani, M. (2026). AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization. arXiv preprint arXiv:2602.07054.

Chen, J., Guo, Z., Chun, J., Wang, P., Perrault, A., Elsner, M. (2025). Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance. arXiv preprint arXiv:2510.10444.

Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C.E., Hernandez, D., Li, D., Durmus, E., Hubinger, E., Kernion, J., Lukovsiut.e, K., Nguyen, K., Cheng, N., Joseph, N., Schiefer, N., Rausch, O., Larson, R., McCandlish, S., Kundu, S., Kadavath, S., Yang, S., Henighan, T., Maxwell, T.D., Telleen-Lawton, T., Hume, T., Hatfield-Dodds, Z., Kaplan, J., Brauner, J., Bowman, S., & Perez, E. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning. ArXiv, abs/2307.13702.

Li, Z., Chen, Y., Lai, H., Gao, X., Nayak, S., Coler, M. (2026). SarcasmMiner: A Dual-Track Post-Training Framework for Robust Audio-Visual Sarcasm Reasoning. arXiv preprint arXiv:2603.05275.

Park, J., Jang, K.J., Alasaly, B., Mopidevi, S., Zolensky, A., Eaton, E., Lee, I., & Johnson, K. (2024). Assessing Modality Bias in Video Question Answering Benchmarks with Multimodal Large Language Models. ArXiv, abs/2408.12763.
Saha, A., Suresh, V., Hospedales, T., Demberg, V. (2025). MUStReason: A Benchmark for Diagnosing Pragmatic Reasoning in Video-LMs for Multimodal Sarcasm Detection. arXiv preprint arXiv:2510.23727.

Turpin, M., Michael, J., Perez, E., & Bowman, S. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. ArXiv, abs/2305.04388.

Wang, D., Liu, S., Zhang, T., Chen, Y., Li, J., Meng, H. (2026). EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning. arXiv preprint arXiv:2601.15668.


