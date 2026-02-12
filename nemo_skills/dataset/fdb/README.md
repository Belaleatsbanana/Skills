# Full-Duplex-Bench (FDB) for nemo-skills

This package prepares the [Full-Duplex-Bench](https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3) dataset for speech-to-speech evaluation in nemo-skills.

## Dataset versions

- **v1.0** — writes to `fdb_v1/` (subtests: pause_candor, pause_synthetic, backchannel, turn_taking, interruption).
- **v1.5** — writes to `fdb_v1_5/` (subtests: background_speech, talking_to_other, backchannel, interruption).

Source layout (v1.0): `candor_pause_handling/{id}/input.wav`, `synthetic_pause_handling/{id}/input.wav`, `icc_backchannel/`, `candor_turn_taking/`, `synthetic_user_interruption/`.

## Prepare data

From the repo root (or with `PYTHONPATH` set so `nemo_skills` is importable):

```bash
# Download v1.0 to default path and prepare
python -m nemo_skills.dataset.fdb.prepare

# Use existing Full-Duplex-Bench data (no download)
python -m nemo_skills.dataset.fdb.prepare --fdb_data_path /path/to/Full-Duplex-Bench-data

# Prepare v1.5 only
python -m nemo_skills.dataset.fdb.prepare --version v1.5 --fdb_data_path /path/to/Full-Duplex-Bench-data

# Skip copying audio (faster, for testing)
python -m nemo_skills.dataset.fdb.prepare --fdb_data_path /path/to/data --no-audio
```

Audio files are copied into `fdb_v1/data/` (or `fdb_v1_5/data/`) with names like `pause_candor_18.wav`, `pause_synthetic_18.wav`, `backchannel_0.wav`, etc., so each subtest has distinct filenames.

## Output layout

After prepare:

- `fdb_v1/<subtest>/test.jsonl` — one JSONL per subtest (e.g. pause_candor, pause_synthetic).
- `fdb_v1/data/*.wav` — shared audio directory; entries in JSONL reference `fdb_v1/data/<audio_id>.wav`.

Each entry includes `messages`, `messages_text_audio`, and `messages_text` variants.

## Evaluation

Set `data_dir` in your eval config to the **fdb package directory** (the parent of `fdb_v1`/`fdb_v1_5`), e.g.:

```yaml
data_dir: /path/to/nemo_skills/dataset/fdb
```

Example configs and runner:

- `scripts/fdb_s2s_offline_v1.0_config.yaml`, `scripts/fdb_s2s_offline_v1.5_config.yaml`
- `scripts/run_eval.py` — run eval for a given config.
- `scripts/run_prepare_and_eval_both.sh` — prepare v1.0 and v1.5 then run offline eval for both (intended for cluster use with lustre paths).

## Source data

Download from Google Drive:  
https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3

If you pass `--fdb_data_path` and the requested version (v1.0 or v1.5) is missing, the script can attempt to download and extract it (requires `gdown`).

## output.wav pipeline (2-channel)

FDB expects **2-channel (stereo)** `output.wav`. Scoring uses the [Full-Duplex-Bench-NV](https://github.com/kevinhu-nv/Full-Duplex-Bench-NV) repo (or a clone) for ASR and evaluation; see `STEREO_REFERENCE_EVIDENCE.md` for where 2-channel is created vs consumed in that repo and in this codebase.

1. **Server** (`s2s_voicechat` backend in `recipes/multimodal/server/backends/s2s_voicechat_infer_backend.py`) runs `offline_inference(..., decode_audio=True)`. The backend **outputs 2-channel WAV**: it duplicates the model’s mono output to both channels so the bytes sent in the API response are already stereo.
2. **Client** (`vllm_multimodal._process_audio_response`) decodes and writes those bytes to `output_dir/audio/<id>.wav` as-is (no channel change).
3. **Prepare** (`prepare_fdb_eval_dir`) copies that file to `fdb_prepared/<id>/output.wav` and **verifies** it is 2-channel; if mono, the script fails with a clear error.

So when using the **s2s_voicechat** backend with `--decode_audio`, the server already returns stereo; the prepare step only verifies and copies.

### Scoring consistency (stereo + backchannel)

The pipeline is **prepare (stereo output.wav) → ASR (--stereo, ch1) → evaluate**. For consistency:

- **fdb_repo_path** in your config must point to a Full-Duplex-Bench clone that has:
  1. **get_transcript/asr.py** with `--stereo` (uses channel 1 for transcription).
  2. **evaluation/eval_backchannel.py** that converts stereo to the model channel (ch1) before calling Silero VAD (Silero expects 1D audio).

The config `fdb_s2s_offline_v1.0_config.yaml` uses **our** Full-Duplex-Bench clone (`.../Full-Duplex-Bench`) for scoring so backchannel runs with the stereo-safe eval. On the cluster, ensure that clone has the updated `eval_backchannel.py` (and `asr.py` with `--stereo`) deployed.


### Reference inference vs s2s_voicechat backend

The reference script `examples/speechlm2/nemotron_voicechat_infer.py` uses **Lightning** `trainer.validate(model, datamodule)` with `DuplexS2SDataset`. It does **not** call `offline_inference()` or write WAVs; audio shape and channel layout come from the model’s validation step and the datamodule. Our **s2s_voicechat** backend instead calls `model.offline_inference()` directly (same `NemotronVoiceChat`, different entry point), then encodes `outputs["audio"]` to WAV and forces 2-channel (duplicate mono to both channels or keep stereo) so FDB receives stereo. So multi-channel behavior is: reference = defined inside NeMo validation/dataset; ours = explicit 2-channel WAV in the backend.

**By default the model output is single-channel (mono).** NemotronVoiceChat’s `offline_inference` returns agent speech only, one channel. The s2s_voicechat backend therefore duplicates that mono to both channels when writing WAV so FDB gets 2-channel. If the model ever returns stereo `(samples, 2)`, the backend keeps it as-is.

### How to check: NeMo validation vs nemo_skills (s2s_voicechat)

You are **using the nemo_skills path (s2s_voicechat backend)**, not the NeMo validation pipeline, if:

1. **Config:** Your FDB config has `server_entrypoint: "-m nemo_skills.inference.server.serve_unified"` and `server_args` with `--backend s2s_voicechat`. Then run_eval.py starts that server and the pipeline sends API requests to it; inference is `model.offline_inference()` inside the backend, and we write 2-channel by duplicating the mono output.
2. **How you run:** You run `python run_eval.py --config fdb_s2s_offline_v1.0_config.yaml` (or similar). You do **not** run the NeMo SLURM script `infer_nano9b_s2s.sh` and then `eval_all_s2s_baseline.sh` / `run_full_duplex_bench_eval.sh` with reorganize_candor_outputs.
3. **Where output.wav comes from:** After generation, audio lives under `output_dir/eval-results/<benchmark>/audio/*.wav` (saved by the client from the API response). Prepare then copies those to `fdb_prepared/<id>/output.wav`. You do **not** have a `validation_logs/pred_wavs` directory from NeMo validation.

You are **using the NeMo validation path** if you run the NeMo inference script (e.g. `infer_nano9b_s2s.sh`), which runs `nemotron_voicechat_infer.py` with `trainer.validate()`; predictions are written to something like `.../validation_logs/pred_wavs`, and you then run `run_full_duplex_bench_eval.sh` with reorganize_candor_outputs. In that case, 2-channel output.wav (if any) comes from whatever writes into pred_wavs (NeMo validation code), not from our backend.

**Summary:** If your config uses `serve_unified` and `--backend s2s_voicechat`, you are on the nemo_skills path; NeMo code is only used via `--code_path` so the backend can import and run `NemotronVoiceChat.offline_inference()`.

## Troubleshooting

### Model output truncated (cut off early)

Generation uses a **max token limit**; the default is **512** (client and server). For S2S, that can truncate longer replies.

- **Where it’s set:** The client sends `max_tokens` from Hydra `inference.tokens_to_generate` (default 512). The server `serve_unified` also has `--max_new_tokens` (default 512).
- **Fix:** In your FDB config (e.g. `scripts/fdb_s2s_offline_v1.0_config.yaml`), set:
  ```yaml
  inference_overrides: "++inference.tokens_to_generate=2048"
  ```
  Increase (e.g. 2048, 4096) as needed so responses aren’t cut off.

### "FDB requires 2-channel (stereo) output"

The prepare step does not convert mono to stereo. Configure the **model/server** (S2S-Duplex inference or serve_unified s2s_voicechat backend) to output 2-channel audio so the WAV written by the client is already stereo.

### ASR step: HuggingFace read timeout

If scoring fails with `ReadTimeout` when loading the ASR model, or you see `ZeroDivisionError` in `eval_smooth_turn_taking`, the ASR step produced no `output.json` files. Pre-download the ASR model once (e.g. `python -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b')"`) in an environment with good HuggingFace access, or set `HF_HUB_DOWNLOAD_TIMEOUT=300`.

### FileNotFoundError: ... interruption_0.wav (or other fdb_v1/data/*.wav)

Generation reads `test.jsonl` and resolves audio paths as `data_dir + path` (e.g. `data_dir`/fdb_v1/data/interruption_0.wav). The JSONL must reference the same filenames as the files on disk. Audio filenames are `{subtest}_{sample_id}.wav` (e.g. `interruption_1.wav`, not `interruption_0.wav` for sample_id 1). If your `test.jsonl` still references 0-based names (e.g. `interruption_0.wav`) while the data dir has `interruption_1.wav`, the paths are out of sync.

**Fix on the cluster:** Re-run the FDB prepare so that `test.jsonl` and `data/` are regenerated together. Use the same `data_dir` as in your eval config so output goes to Lustre:

```bash
# On the cluster (with PYTHONPATH set so nemo_skills is importable)
python -m nemo_skills.dataset.fdb.prepare \
  --fdb_data_path /lustre/fsw/portfolios/convai/users/mmkrtchyan/projects/speechLM/s2s/Full-Duplex-Bench-data \
  --data_dir /lustre/fsw/portfolios/convai/users/mmkrtchyan/projects/speechLM/s2s/Skills/nemo_skills/dataset/fdb
```

Then re-run your eval; generation will find e.g. `interruption_1.wav` at the path in the new `test.jsonl`.
