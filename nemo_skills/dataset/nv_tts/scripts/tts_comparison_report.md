# TTS Evaluation Comparison Report

Comparing 2 model(s): GRPO Step 1100, A3 Baseline

## Summary (Averaged Across Test Sets)

| Metric | GRPO Step 1100 | A3 Baseline |
|---|---|---|
| WER (cumulative) | 4.56% | **3.65%** |
| CER (cumulative) | 2.75% | **2.24%** |
| WER (filewise avg) | 4.49% | **3.54%** |
| CER (filewise avg) | 3.14% | **2.28%** |
| UTMOS v2 | **3.124** | 3.005 |
| SSIM (pred vs GT) | **0.6599** | 0.0255 |
| SSIM (pred vs context) | **0.6709** | 0.0259 |

### Analysis

**A3 Baseline** performs best overall. Key advantages: A3 Baseline leads in WER (cumulative); A3 Baseline leads in CER (cumulative); A3 Baseline leads in WER (filewise avg); A3 Baseline leads in CER (filewise avg).

## Per-Test-Set Results

### nv_tts.libritts_seen

| Metric | GRPO Step 1100 |
|---|---|
| WER (cumulative) | 2.76% |
| CER (cumulative) | 1.92% |
| WER (filewise avg) | 2.69% |
| CER (filewise avg) | 1.83% |
| UTMOS v2 | 3.322 |
| SSIM (pred vs GT) | 0.8022 |
| SSIM (pred vs context) | 0.8022 |
| Total audio (sec) | 1314.7 |

### nv_tts.libritts_test_clean

| Metric | GRPO Step 1100 | A3 Baseline |
|---|---|---|
| WER (cumulative) | 1.37% | **1.27%** |
| CER (cumulative) | 0.46% | **0.41%** |
| WER (filewise avg) | 1.47% | **1.30%** |
| CER (filewise avg) | 0.51% | **0.42%** |
| UTMOS v2 | **3.279** | 3.153 |
| SSIM (pred vs GT) | **0.8378** | -0.0138 |
| SSIM (pred vs context) | **0.8378** | -0.0138 |
| Total audio (sec) | 12454.1 | 15257.6 |

### nv_tts.riva_hard_digits

| Metric | GRPO Step 1100 | A3 Baseline |
|---|---|---|
| WER (cumulative) | 3.52% | **2.13%** |
| CER (cumulative) | 2.58% | **1.41%** |
| WER (filewise avg) | 3.27% | **1.96%** |
| CER (filewise avg) | 2.43% | **1.32%** |
| UTMOS v2 | **3.119** | 3.035 |
| SSIM (pred vs GT) | **0.6976** | 0.0460 |
| SSIM (pred vs context) | **0.6976** | 0.0460 |
| Total audio (sec) | 2462.9 | 3022.4 |

### nv_tts.riva_hard_letters

| Metric | GRPO Step 1100 | A3 Baseline |
|---|---|---|
| WER (cumulative) | **5.34%** | 6.17% |
| CER (cumulative) | **2.92%** | 4.51% |
| WER (filewise avg) | **5.00%** | 5.82% |
| CER (filewise avg) | **2.82%** | 4.26% |
| UTMOS v2 | 2.988 | **2.991** |
| SSIM (pred vs GT) | **0.6505** | 0.0432 |
| SSIM (pred vs context) | **0.6505** | 0.0432 |
| Total audio (sec) | 1984.2 | 2432.8 |

### nv_tts.riva_hard_money

| Metric | GRPO Step 1100 | A3 Baseline |
|---|---|---|
| WER (cumulative) | 2.92% | **0.92%** |
| CER (cumulative) | 2.00% | **0.55%** |
| WER (filewise avg) | 2.86% | **0.86%** |
| CER (filewise avg) | 1.96% | **0.49%** |
| UTMOS v2 | **3.191** | 3.076 |
| SSIM (pred vs GT) | **0.7075** | 0.0428 |
| SSIM (pred vs context) | **0.7075** | 0.0428 |
| Total audio (sec) | 2635.0 | 3149.5 |

### nv_tts.riva_hard_short

| Metric | GRPO Step 1100 | A3 Baseline |
|---|---|---|
| WER (cumulative) | 15.66% | **9.84%** |
| CER (cumulative) | 9.24% | **6.11%** |
| WER (filewise avg) | 15.66% | **9.84%** |
| CER (filewise avg) | 12.32% | **6.76%** |
| UTMOS v2 | 2.525 | **2.544** |
| SSIM (pred vs GT) | **0.3004** | 0.0373 |
| SSIM (pred vs context) | **0.3004** | 0.0373 |
| Total audio (sec) | 312.4 | 573.5 |

### nv_tts.vctk

| Metric | GRPO Step 1100 | A3 Baseline |
|---|---|---|
| WER (cumulative) | **0.36%** | 1.55% |
| CER (cumulative) | **0.09%** | 0.46% |
| WER (filewise avg) | **0.47%** | 1.43% |
| CER (filewise avg) | **0.10%** | 0.45% |
| UTMOS v2 | **3.441** | 3.229 |
| SSIM (pred vs GT) | **0.6236** | -0.0028 |
| SSIM (pred vs context) | **0.7002** | -0.0004 |
| Total audio (sec) | 310.6 | 334.6 |

---
*Lower WER/CER is better, higher UTMOS/SSIM is better. **bold** = best value.*
