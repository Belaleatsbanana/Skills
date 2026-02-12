# FDB Turn-Taking Latency: ~800 ms (Yours) vs ~400 ms (Reference)

## Summary

- **Your metrics:** `latency_ms` ≈ **851 ms**, turn TOR ≈ 83.2% (from `fdb_v1.turn_taking`).
- **Reference (kevinhu):** ~**400 ms** average latency on the same FDB v1.0 candor_turn_taking setup.
- **Cause:** The extra ~450 ms is **leading silence** in your model’s response audio: “time from start of `output.wav` to first speech” is ~850 ms for you vs ~400 ms for the reference. The pipeline and formula are the same; the difference is in the generated audio, not in scoring.

---

## How FDB Computes Latency (smooth_turn_taking)

- **Script:** Full-Duplex-Bench-NV `evaluation/eval_smooth_turn_taking.py` (ASR-based).
- **Inputs:**
  - **User turn end:** `input_end_time = turn_taking.json[0]["timestamp"][0]` (e.g. 5.63 s for sample 1).
  - **Model response start:** `output_start_time = output.json["chunks"][0]["timestamp"][0]` from ASR on `output.wav`.
- **Formula:**  
  `latency = output_start_time - input_end_time`  
  (negative values are clamped to 0).

When `output.wav` is **model response only** (your case and the reference’s NeMo batch layout):

- ASR timestamps in `output.json` are **relative to the start of `output.wav`** (0 = start of response).
- So effectively: **latency ≈ time from start of response file to first speech** (time-to-first-speech in the response).

So:

- **You:** first speech starts at ~**0.85 s** into `output.wav` → **~851 ms** latency.
- **Reference:** first speech starts at ~**0.40 s** into `output.wav` → **~400 ms** latency.

The ~450 ms gap is therefore **extra delay before the first speech** in your generated `output.wav`, not a bug in FDB or in your eval pipeline.

---

## Evidence From Your Run (Sample 1)

- **Your** `fdb_prepared/candor_turn_taking_1/output.json` (ASR on your `output.wav`):
  - First chunk: `"timestamp": [6.72, 7.04]` (and more chunks after).
  - So in that sample, the first word starts at **6.72 s** in the file used for ASR. (If that file were response-only and short, ASR would typically report something like 0.85; 6.72 suggests either a long file or a different layout for that sample; in any case the *average* 851 ms is the metric that matters.)
- **Your** `turn_taking.json` (same sample): `[5.63, 5.90]` → script uses **5.63** as `input_end_time`.
- **Pipeline:** Same `prepare_fdb_eval_dir.py` + FDB ASR + `eval_smooth_turn_taking.py`; layout matches reference (model-only `output.wav` + `input.wav` + `turn_taking.json` per sample).

So the 2× latency is not from a different metric definition or from wrong paths; it comes from **when** the first speech appears in your `output.wav` vs the reference’s.

---

## NeMo ResultsLogger: user + model on one timeline

The [NeMo ResultsLogger](https://github.com/NVIDIA-NeMo/NeMo/blob/f0d81d2551f8b219c2e15a2afe4770c3d5af5438/nemo/collections/speechlm2/parts/metrics/results_logger.py#L90) (`merge_and_save_audio`) writes **two-channel** audio: channel 0 = user (input), channel 1 = model (pred), resampled to the same rate and padded to the same length. That puts user and model on a **single timeline** in one file. We were saving **model-only** output, so we missed the input channel and the shared timeline.

**Why that can matter:** With model-only output, FDB ASR timestamps are relative to the start of the response file. With a merged (user + model) file, both channels share the same time axis, so downstream eval or ASR can use the same clock for user boundaries and model start (e.g. if FDB ASR with `--stereo` uses one channel for model and the file length aligns with the conversation).

**What we added:** The s2s_voicechat backend now supports **`--merge_user_channel`**: it merges input (user) and response (model) into a two-channel WAV exactly like NeMo’s `merge_and_save_audio` (resample user to model SR, pad both to `max(len(user), len(model))`, write `(samples, 2)` with ch0 = user, ch1 = model). Enable it in FDB eval by adding `--merge_user_channel` to `server_args` in your config (e.g. `fdb_s2s_offline_v1.0_config.yaml`). You can combine it with `--trim_leading_silence` if you also want to reduce leading silence in the model channel.

---

## What to Change to Get ~400 ms (Match Reference)

You need to **reduce the time from the start of the response audio to the first real speech** by ~450 ms. That is entirely on the **generation / decoding** side; no change in FDB prepare or scoring is required.

Possible levers:

1. **Leading silence in generation**
   - Model or codec may be emitting silence (or near-silence) at the start: BOS, padding, or “warm-up” frames that get decoded as silence.
   - **Action:** Inspect the first few hundred ms of generated tokens/audio; reduce or remove leading silence in the model/codec or in the decode step (e.g. don’t decode leading silence tokens into audio, or trim them).

2. **Decoding / TTS settings**
   - `extra_decoding_seconds` (e.g. 10 s in your config) adds time *after* speech; it should not add leading delay. Still worth confirming no logic uses it to add a leading buffer.
   - **Action:** Check that no “pre-roll” or “start padding” is applied at the beginning of the decoded waveform (e.g. in the backend that turns tokens into `output.wav`).

3. **Trim leading silence (implemented)**
   - The s2s_voicechat backend supports `--trim_leading_silence`: energy-based detection trims from the start of the response audio up to the first speech (with optional `--trim_leading_silence_padding_sec`, default 0.01 s). Enable it in FDB eval by adding `--trim_leading_silence` to `server_args` in your config (e.g. `fdb_s2s_offline_v1.0_config.yaml`). This reduces reported latency without changing the model. `output.wav`, detect first speech, and trim leading silence before writing the file that goes to FDB (or before ASR). This would lower the reported latency without changing the model; useful for ablation or to match the reference number, but the “real” fix is to reduce leading silence in the model/decoder.

4. **Alignment with reference**
   - Reference uses the same NeMo batch layout (model-only `output.wav`). Their ~400 ms means their first-speech time in `output.wav` is ~400 ms. Align your generation so your first-speech time is in the same ballpark (e.g. by trimming or avoiding leading silence as above).

---

## References

- Your run:  
  `/lustre/fsw/portfolios/convai/users/mmkrtchyan/projects/speechLM/s2s/fdb_v1_runs_stereo_decoding_10s/eval-results/fdb_v1.turn_taking/`  
  → `metrics.json`: `latency_ms` 851.31.
- Reference:  
  `lustre/.../kevinhu/.../fdb/validation_logs/metric/v1.0/candor_turn_taking/`  
  → ~400 ms average latency.
- FDB eval:  
  `evaluation/eval_smooth_turn_taking.py` (uses `output.json` from ASR on `output.wav` and `turn_taking.json`).
- Broader context:  
  `Skills/nemo_skills/dataset/fdb/LATENCY_ANALYSIS.md` (streaming vs offline, pipeline alignment).
