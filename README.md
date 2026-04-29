# AVUT PragCoT Minimal Scaffold

Minimal Gemini-based scaffold for AVUT Idea 2:
- Perception (modality-separated descriptions)
- Decoding (cross-modal signal aggregation)
- Reasoning (final MCQ answer)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `.env` and replace `GEMINI_API_KEY`.
Default model is `gemini-2.5-flash-lite` (override via `GEMINI_MODEL`).

### Gemini generation controls

Generation behavior is configured in `src/config.py` and can be overridden via `.env`:

- `GEMINI_TEMPERATURE` (default: `0.1`) - lower = less verbose, less random.
- `GEMINI_MAX_OUTPUT_TOKENS` (default: `256`) - hard cap for response length.
- `GEMINI_TIMEOUT_MS` (default: `90000`) - per-request timeout to avoid indefinite network hangs.
- `GEMINI_SYSTEM_INSTRUCTION` (default: concise MCQ-focused instruction) - global style/verbosity guidance.

These settings are applied to all Gemini calls (transcription, modality descriptions, decoding, and final reasoning).

### Hugging Face (dataset + video prefetch)

Set a read token so Hub requests are authenticated (higher rate limits, fewer stalls):

- In `.env`: `HF_TOKEN=hf_...` (create at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
- Alias: `HUGGING_FACE_HUB_TOKEN` also works.

`run_idea2.py` loads `.env` **before** importing the pipeline so the token is visible on first `load_dataset` / download.

Optional faster downloads: install **`hf-transfer`** (already in `requirements.txt`); the script sets `HF_HUB_ENABLE_HF_TRANSFER=1` when that package is importable.

**Train split layout (video-only streaming):** the Hub stream has only a `video` column. The code maps **AV-Human** rows first (`sample_id` = row index + 1 for the first `HF_AVUT_HUMAN_ROW_COUNT` rows, default **1734**), then **AV-Gemini** (`sample_id` = row index − that offset). Override with `HF_AVUT_HUMAN_ROW_COUNT` if your revision differs.

To **skip** Hub prefetch entirely (fastest for iterating on Gemini / prompts): `--no-prefetch-videos`.

## Run (default: two passes, AVUT Table 3)

By default the script runs **two independent passes** (no mixing), aligned with AVUT Table 3:
**AV-Human** (left of `/`) and **AV-Gemini** (right of `/`). Each pass uses only its filtered QA JSONL and its filtered metadata JSON (`src/config.py`).

`--max-samples N` means **up to N QA rows per pass** (task-balanced within that pass), so up to `2×N` rows total.

`--prefetch-videos M` caps how many distinct **sample_id** values are prefetched **per pass** (Human list and Gemini list are capped separately). Omit to prefetch every id needed on each side.

`--run-sample <sample_id>` runs a **basic smoke mode** for one AV-Human row: one matching HF video + full describe/decode/reason inference for that sample.

Progress: Hugging Face per-file download bars are disabled; you get one prefetch bar, then a **green** tqdm per pass over Gemini work (one step per QA row).

```bash
python scripts/run_idea2.py --max-samples 50 --prefetch-videos 40
python scripts/run_idea2.py --max-samples 5 --no-prefetch-videos
python scripts/run_idea2.py --run-sample 1
```

Outputs:
- `outputs/idea2_predictions_av_human.jsonl` + `outputs/idea2_metrics_av_human.json`
- `outputs/idea2_predictions_av_gemini.jsonl` + `outputs/idea2_metrics_av_gemini.json`
- `outputs/gemini_call_metrics_detailed.jsonl` (stream of per-call Gemini lifecycle events: start, dispatch, success/error, elapsed_ms, stage, model, timeout, sample/task context, media kind)

## Vanilla baseline (single-pass direct video QA)

Use `run_vanilla.py` for a direct baseline: one Gemini call per sample using only
video + MCQ prompt (no describe/decode/reason pipeline). CLI flags match
`run_idea2.py` (`--input`, `--output-dir`, `--max-samples`, `--run-sample`,
`--prefetch-videos`, `--no-prefetch-videos`).

```bash
python scripts/run_vanilla.py --max-samples 50 --prefetch-videos 40
python scripts/run_vanilla.py --run-sample 1
```

Outputs:
- `outputs/vanilla_predictions_av_human.jsonl` + `outputs/vanilla_metrics_av_human.json`
- `outputs/vanilla_predictions_av_gemini.jsonl` + `outputs/vanilla_metrics_av_gemini.json`

## Single-pass override (debug)

```bash
python scripts/run_idea2.py --input data/avut/sample.jsonl
```
