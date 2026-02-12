# Full-Duplex-Bench Latency: Your Implementation vs Reference

This document explains why your reported latency is higher than the [Full-Duplex-Bench-NV](https://github.com/kevinhu-nv/Full-Duplex-Bench-NV/tree/nv) reference and what affects it.

## Summary: Streaming vs Offline

| Aspect | Full-Duplex-Bench-NV (reference) | Your implementation (nemo_skills) |
|--------|----------------------------------|-----------------------------------|
| **Inference mode** | **Streaming** (real-time) | **Offline** (batch) |
| **Input** | Audio sent in **30 ms frames** over Socket.IO | **Entire** `input.wav` in one HTTP request (base64) |
| **Output** | **Time-aligned**: one output frame per input frame, written as stream progresses | Single response audio; written as one `output.wav` after request completes |
| **When model “starts”** | As soon as it has enough context (can be early in the turn) | Only **after** full input is received and processed |
| **Reported latency** | Low (response start close to user turn end on the timeline) | High (includes full encode + full forward + decode) |

So the difference is not a bug in your code; it comes from **how** inference is run (streaming vs offline).

**Actual inference used for FDB (NeMo batch):** The run you shared uses **NeMo batch validation**, not the Freeze-Omni streaming client: `nemotron_voicechat_infer.py` with `trainer.validate(model, datamodule)` (S2S-Duplex codebase, `NemotronVoiceChat` + `DuplexS2SDataset`). So for that setup, inference is **offline** as well: full input per sample, one response per sample. In that case, raw outputs are also "model response only" per sample; to get correct FDB latency for turn_taking/interruption they would need a **prepare step** that builds one timeline (e.g. output.wav = input + response, or equivalent). The table above compares to the **FDB repo’s example** (streaming client); when the reference uses NeMo batch like you, the main difference is how the eval input is prepared (one timeline for latency), not streaming vs batch.

**Reference scoring pipeline (after NeMo inference):** Their eval entrypoint is `eval_all_s2s_baseline.sh`. For FDB it sets `pred_wav_dir=${ckpt_dir}/fdb/pad0_bos0_eos0/validation_logs/pred_wavs` and `output_dir=${ckpt_dir}/fdb/.../validation_logs/metric`, then runs `run_full_duplex_bench_eval.sh ${pred_wav_dir} ${output_dir}`. So FDB metrics come from that script (which presumably copies/layouts pred_wavs into FDB’s expected dir structure and runs the FDB repo’s evaluation). The full FDB flow is in `run_full_duplex_bench_eval.sh` (see next section). Voicebench uses a separate path: `eval_intel.sh` on `validation_logs/metadatas`, calling `get_scores.sh` for commoneval, openbookqa, etc. (text/QA scoring with LLM judge).

---

## Reference FDB evaluation pipeline (full)

From `run_full_duplex_bench_eval.sh` (S2S-Duplex `scripts/speech_eval/`). **Inputs:** `INFERENCE_OUT_DIR` (pred_wavs from NeMo validation), `FULL_DUPLEX_BENCH_RESULT_DIR` (reorganized data + metrics), `ORIGINAL_DATA_BASE_PATH` (FDB v1.0 original data). **Datasets:** `candor_turn_taking`, `candor_pause_handling`, `synthetic_pause_handling`, `synthetic_user_interruption`.

For each dataset they run three steps:

| Step | Their script | Purpose |
|------|----------------|--------|
| **[1/3] Reorganize** | `reorganize_candor_outputs.py` | Copies each pred WAV to `.../v1.0/<dataset_name>/<id>/output.wav` (model prediction only). Then for each ID copies **all contents** from `original_data_path/<id>/` into that dir (input.wav, turn_taking.json, interrupt.json, etc.) **without overwriting** output.wav. So each sample dir has: **output.wav** = model only, **input.wav** + metadata from original. No concatenation. |
| **[2/3] ASR** | Full-Duplex-Bench-NV `get_transcript/asr.py` | `--root_dir` = reorganized dir, `--task` = `full` or `user_interruption`, **`--stereo`**. Produces `output.json` per sample. |
| **[3/3] Evaluate** | Full-Duplex-Bench-NV `evaluation/evaluate.py` | `--task` smooth_turn_taking \| pause_handling \| user_interruption, `--root_dir` same dir; for user_interruption also `--client_key_jsonl`. |

**Mapping to our pipeline (now aligned):**

| Step | Ours | Theirs |
|------|------|--------|
| **1. Prepare** | `prepare_fdb_eval_dir.py`: output.wav = **model response only**; for turn_taking/interruption copies **input.wav** and metadata (turn_taking.json, interrupt.json) from `--fdb_data_path`. Writes stereo (2ch) when `--stereo`. | Same idea: output.wav = model only; copy original dir contents (input.wav + metadata) into each sample dir. |
| **2. ASR** | FDB `asr.py` with **`--stereo`** when prepare was run with `--stereo`. | Same script with **`--stereo`**. |
| **3. Evaluate** | FDB `evaluate.py` on `fdb_prepared`. | Same. |

**Match:** We now mirror their behavior: model-only output.wav, input.wav + metadata in each sample dir for turn_taking/interruption, and **stereo** (prepare writes 2-channel output.wav and run_fdb_scoring passes `--stereo` to prepare and ASR).

---

## How the FDB repo example does inference (streaming)

From `model_inference/freeze-omni/inference.py`:

1. **Frame size**: `FRAME_MS = 30`, so input is sent in **30 ms chunks** (480 samples at 16 kHz).
2. **Loop**: For each input frame:
   - Emit the frame to the server via Socket.IO (`audio` event).
   - Wait `frame_dur = 0.03` s.
   - Read whatever the server has sent back (one 30 ms output chunk at 24 kHz).
   - Write that chunk to `output.wav`.

So `output.wav` is **time-synchronous** with `input.wav`: at any time index, the output sample is what the model produced when it had seen input up to (roughly) that time. The model can start speaking as soon as it decides to (e.g. after user turn end), so “time to first response” can be small.

Evaluation then measures latency on this **aligned** timeline (e.g. time from end of user turn to start of model speech). That gives the low latency numbers in the reference.

---

## How your implementation works (offline)

1. **Single request**: You load the full `input.wav`, base64-encode it, and send it in **one** chat completion request (`VLLMModel` / `VLLMMultimodalModel`).
2. **Server**: Sees the **entire** utterance, runs the full forward pass, then returns one response (e.g. one audio blob).
3. **Output**: That blob is saved (e.g. via `VLLMMultimodalModel._process_audio_response`) and later copied to `fdb_prepared/<id>/output.wav` by `prepare_fdb_eval_dir.py`.

So:

- The model cannot produce any output until the **whole** input has been received and processed.
- Your `output.wav` is “the model’s reply” only; it is **not** frame-by-frame aligned to the input timeline like in the reference.

**How FDB does alignment (reference vs ASR):** [Full-Duplex-Bench-NV](https://github.com/kevinhu-nv/Full-Duplex-Bench-NV/tree/nv) uses **reference alignment for the user side** and **ASR only on the generated audio** for the agent side:

- **User side (turn boundaries):** From dataset metadata — no ASR on input. For smooth turn-taking they read `turn_taking.json` (e.g. `[TURN-TAKING]` with `timestamp: [5.63, 5.90]`) to get the turn end time. For user interruption they use `interrupt.json` and optionally crop audio before ASR.
- **Agent side (when the model spoke):** They run **ASR on `output.wav`** via `get_transcript/asr.py` (NeMo parakeet-tdt-0.6b-v2) to produce `output.json` with word-level timestamps. Latency and TOR use the first chunk’s start in `output.json` as “model response start”.

So they do **not** run ASR on the input; they use **reference alignment for the user side** and **build alignment for the agent side** by running ASR on the generated `output.wav`. For v1.5 timing, `evaluation/get_timing.py` uses **VAD only** (Silero) on both `input.wav` and `output.wav` (no ASR, no reference JSON) to compute overlap and response gaps; it expects both files in the same folder on the same timeline.

**What the FDB reference actually does (not concatenation):** In their repo, `model_inference/freeze-omni/inference.py` does **streaming**: they send input in 30 ms frames and write one 30 ms output chunk per frame. So **output.wav has the same duration as input.wav** and is **time-synchronous** — at each time index, the sample is the model’s output (or silence). They do **not** write “input.wav + model response”; they write **model output only**, aligned to the input timeline. Their **output.wav is mono** (`channels=1`, 24 kHz in the script). Input is converted to mono before sending (`_mono(wav)`).

**NeMo batch (reorganize_candor_outputs):** Uses model-only output.wav and copies input.wav + metadata; we match that (no concatenation).

**How the reference (reorganize_candor_outputs) handles output.wav:** They write **output.wav = model prediction only** and copy **input.wav** and metadata from the original dataset into each sample dir. So each dir has both input.wav and output.wav; FDB eval uses turn_taking.json/interrupt.json (from original) and ASR on output.wav. We match this: output.wav = model only, and we copy input.wav + metadata for turn_taking/interruption.

**FDB v1.0 subtasks and output.wav:**

| Subtest | FDB eval uses | output.wav | Pipeline behavior |
|---------|----------------|------------|-------------------|
| **pause_candor** / **pause_synthetic** | `output.json` only (TOR from chunks) | Model only | `output.wav` = model response only; no input.wav copy. |
| **backchannel** | `output.wav` + `output.json` (VAD + chunks) | Model only | `output.wav` = model response only. |
| **turn_taking** | `turn_taking.json` + `output.json`; latency = model_start − turn_end | Model only | `output.wav` = model only; copy **input.wav** + turn_taking.json into sample dir (match reference). |
| **interruption** | `interrupt.json` + `output.json` (ASR crops post-interrupt); latency = model_start − interrupt_end | Model only | `output.wav` = model only; copy **input.wav** + interrupt.json into sample dir (match reference). |

The prepare script (`prepare_fdb_eval_dir.py`) now matches **reorganize_candor_outputs**: output.wav = model response only; for turn_taking/interruption it copies input.wav and metadata from `--fdb_data_path`. With `--stereo` (default in run_fdb_scoring), output.wav is written as 2-channel and FDB ASR is run with `--stereo`.

**How input_audio and output_audio are used during scoring**

- **Upstream:** In our generation pipeline, **input_audio** is the audio sent to the model (user/conversation); **output_audio** is the model’s response, saved under the eval run’s audio dir and then referenced in `output.jsonl`.
- **Prepare:** That **output_audio** becomes `fdb_prepared/<id>/output.wav`. For turn_taking and interruption we also copy **input.wav** from FDB original data into the same sample dir (so the dir has both files, like the reference).
- **Scoring (FDB v1.0 tasks):** The scripts invoked by `evaluate.py` for v1.0 (**smooth_turn_taking**, **pause_handling**, **user_interruption**, **backchannel**) do **not** load `input.wav`. They use only: (1) **output.wav** — ASR produces `output.json`; backchannel also runs VAD on output.wav; (2) **metadata** — `turn_taking.json`, `interrupt.json` for turn boundaries. So for the reported v1.0 metrics (TOR, latency, JSD, etc.), **input.wav is not used**. We still copy input.wav into each sample dir so the layout matches the reference and so v1.5 or other tools can use it.
- **Scoring (FDB v1.5 timing):** `get_timing.py` **does** use input audio: it runs VAD on **both** `input.wav` and `output.wav` in each sample dir (and only processes folders where both exist) to compute overlap and response-gap intervals.

When FDB computes "Average latency", it typically uses time-aligned transcripts (e.g. from ASR) on both input and output. For the reference, “first model speech” happens early on the timeline. For you, “first model speech” in the output effectively starts only after:

- Upload of full audio  
- Server encoding of full audio  
- Full model forward pass  
- Decoding and return  

So the reported latency includes this **full offline delay**, which is why your numbers are higher.

---

## What you didn’t do wrong

- Your pipeline (prepare → generation → scoring) and use of `output.wav` for FDB are consistent.
- The higher latency is expected when comparing **offline** to **streaming** on a benchmark that is designed and measured in a **streaming** setup.

---

## Options to reduce reported latency (or interpret it)

### 1. Move to streaming inference (match reference)

To get latency comparable to the reference you would need to:

- Send audio in **small chunks** (e.g. 30 ms) to the model server.
- Receive **streaming** audio back (or chunks).
- Build **time-aligned** `output.wav`: for each time step (or frame), write the corresponding output chunk (or silence) so that `output.wav` has the same length and alignment as `input.wav`.

That requires:

- A server that supports streaming audio in/out (e.g. Socket.IO or similar), not just one-shot chat completion.
- Client code similar to `inference.py`: frame loop, emit input frames, collect output frames, write them in order.

Your current VLLM chat completion path is one-shot; it doesn’t produce a time-aligned stream by itself.

### 2. Keep offline and document it

- Keep your current design and report latency as “offline latency” (full request latency).
- In papers or reports, state that you run **offline** (full audio in, full audio out) and that the benchmark reference uses **streaming**; then the latency gap is expected and not an implementation bug.

### 3. Check FDB evaluation details

- In the FDB repo, `evaluation/get_timing.py` and task-specific eval scripts (e.g. `eval_smooth_turn_taking_text.py`, `eval_user_interruption_text.py`) define exactly how “Average latency” is computed (e.g. which timestamps from ASR or metadata are used).
- If you ever add a streaming path, you’d want your `output.wav` to follow the same time alignment and format as the reference so that this metric is comparable.

---

## References

- Full-Duplex-Bench-NV (nv branch): https://github.com/kevinhu-nv/Full-Duplex-Bench-NV/tree/nv  
- Model inference (streaming): `model_inference/README.md`, `model_inference/freeze-omni/inference.py`  
- Freeze-Omni streaming: 30 ms frames, Socket.IO, time-aligned `output.wav`  
- Your flow: `nemo_skills.inference.model.vllm` / `vllm_multimodal`, full-audio request → single response → `prepare_fdb_eval_dir` → `output.wav`
