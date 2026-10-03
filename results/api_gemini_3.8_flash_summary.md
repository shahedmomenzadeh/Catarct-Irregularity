# Cataract Surgery Irregularity Benchmark (CSI-Bench) Evaluation Report

- **Model ID:** `ag/gemini-3.8-flash`
- **Tag:** `api_gemini_3.8_flash`
- **Total Cases Evaluated:** 41
- **Overall Accuracy:** 51.22%
- **Macro-Balanced Accuracy:** 47.85%
- **Strict JSON Contract Adherence:** 100.00%

---

## 1. Per-Class Diagnostic Performance

| Option | Category | Cases | Accuracy | Sensitivity (Recall) | Specificity | Precision (PPV) | F1-Score |
|:------:|:---------|:-----:|:--------:|:--------------------:|:-----------:|:---------------:|:--------:|
| **A** | Normal | 10 | 63.4% | 40.0% | 71.0% | 30.8% | 0.3478 |
| **B** | Lens Irregularity | 9 | 73.2% | 44.4% | 81.2% | 40.0% | 0.4211 |
| **C** | Pupil Contraction | 22 | 65.8% | 59.1% | 73.7% | 72.2% | 0.6500 |

---

## 2. 3x3 Confusion Matrix

| Ground Truth \ Predicted | Normal (A) | Lens Irreg. (B) | Pupil Cont. (C) | Invalid / None |
|:-------------------------|:----------:|:---------------:|:---------------:|:--------------:|
| **Normal (A)** | 4 | 3 | 3 | 0 |
| **Lens Irregularity (B)** | 3 | 4 | 2 | 0 |
| **Pupil Contraction (C)** | 6 | 3 | 13 | 0 |

