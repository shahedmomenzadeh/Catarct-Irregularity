# Cataract Surgery Irregularity Benchmark (CSI-Bench)

A standardized, self-contained academic evaluation framework for Vision-Language Models (VLMs) on intraoperative ophthalmic irregularity detection and 3-class taxonomy classification.
Derived from the clinical **Cataract-1K** surgical video dataset.

---

## 1. Key Architectural Principles

1. **100% Deterministic Evaluation (No LLM Judge):**
   * Structured as a 3-choice closed Multiple-Choice Question (MCQ) with ground-truth clinical labels.
   * Scored via strict JSON parsing, tolerant JSON block extraction, and regex letter extractors (`[A-C]`).
   * Eliminates the variance, non-reproducibility, latency, and cost of external LLM judges.

2. **3-Class Clinical Irregularity Taxonomy:**
   * **Option A — Normal:** Routine cataract extraction with normal capsular anatomy and stable pupil dilation.
   * **Option B — Lens Irregularity:** Capsular tear, zonular laxity/dehiscence, crystalline lens subluxation, or abnormal lens dynamics.
   * **Option C — Pupil Contraction:** Intraoperative miosis, iris prolapse, Intraoperative Floppy Iris Syndrome (IFIS), or irregular pupil margin dynamics.

3. **VLM-Ready Dual-Output Prompt Design:**
   * Each prompt asks the model to first thoroughly analyze the surgical video, explain the visible intraoperative events and anatomical structures, and then output its chosen classification option.
   * Formatted as a JSON object with `"explanation"` and `"answer"`.

4. **Pure Visual Video Evaluation:**
   * All audio tracks are stripped via `ffmpeg -an`.
   * Zero transcripts or acoustic cues in prompts, guaranteeing that model predictions reflect true computer vision understanding.

5. **Per-Model Isolated Environments:**
   * Automated dependency management via `uv` isolating environment stacks:
     * `.venv-api` for API endpoints (`openai`, `httpx`, `pydantic`).
     * `.venv-qwen3vl` for Qwen3-VL, Qwen2.5-VL, and Lingshu-7B (`transformers>=4.50`, `qwen-vl-utils`).
     * `.venv-hulumed` for Hulu-Med-7B / 4B (`transformers==4.51.2`, `decord`).
     * `.venv-magevl` for Mage-VL (`causal-conv1d`, `mamba-ssm`).

---

## 2. Directory Structure

```
E:\Catarct-Irregularity\
├── data/                               # Original organized video folders
│   ├── Normal/                         # 10 routine cases (.mp4 + .jsonl)
│   ├── Lens_irregularity/              # 9 lens irregularity cases (.mp4 + .jsonl)
│   └── Pupil_Contraction/              # 23 pupil contraction cases (.mp4 + .jsonl)
├── Irregularity_dataset/               # Standardized flat benchmark for Hugging Face Hub
│   ├── case_2000.mp4
│   ├── case_2000.jsonl
│   ├── case_3459.mp4
│   ├── case_3459.jsonl
│   └── README.md                       # Hugging Face Dataset Card
├── metadata/
│   ├── case_manifest.csv               # Complete video metadata (durations, resolutions, codecs)
│   └── dataset_summary.json            # Class distribution and taxonomy definitions
├── requirements/
│   ├── requirements-api.txt
│   ├── requirements-qwen3vl.txt
│   ├── requirements-hulumed.txt
│   └── requirements-magevl.txt
├── prompts.py                          # Taxonomy definitions and VLM prompt templates
├── dataset_loader.py                   # Unified loader supporting data/ and Irregularity_dataset/
├── evaluator.py                        # Deterministic 3-class evaluator & confusion matrix generator
├── eval_common.py                      # GPU VRAM management, retry logic, and resume loop
├── create_irregularity_dataset.py      # Automated dataset generator and probe script
├── api_inference.py                    # OpenAI / 9router Gemini API runner
├── qwen3VL_inference.py                # Qwen3-VL model series runner
├── lingshu_inference.py                # Lingshu-7B & Qwen2.5-VL runner
├── hulumed_inference.py                # Hulu-Med-7B & 4B runner
├── mage_vl_inference.py                # Mage-VL runner
├── main.py                             # Central CLI entry point
├── upload_to_hf.py                     # Hugging Face Hub upload utility
├── hf_loader.py                        # Hugging Face snapshot downloader
├── eval_all.bat                        # Windows batch runner
└── eval_all.sh                         # Linux / Git-Bash runner
```

---

## 3. Dataset Specification

| Class | Folder | Cases | Total Duration | Correct Answer |
|:------|:-------|:-----:|:--------------:|:--------------:|
| **Normal** | `data/Normal/` | 10 | 80.3 min | **A** |
| **Lens Irregularity** | `data/Lens_irregularity/` | 9 | 53.4 min | **B** |
| **Pupil Contraction** | `data/Pupil_Contraction/` | 23 | 232.9 min | **C** |
| **Total** | | **42** | **366.5 min (~6.1 hrs)** | |

---

## 4. Prompt Template

```
You are given a cataract procedure video. Analyze the surgical video carefully, explain the visible intraoperative events, anatomical structures, and any observed irregularities, and then classify the case into one of the following 3 options:

Options:
A) Normal
B) Lens Irregularity
C) Pupil Contraction

Respond ONLY with a JSON object formatted exactly as:
{
  "explanation": "<3-10 sentences describing the visible intraoperative evidence, anatomical findings, and rationale>",
  "answer": "<single uppercase letter corresponding to your chosen option: A, B, or C>"
}
Do not include any text outside the JSON object.
```

---

## 5. Usage & Execution

### A. Dry Run Verification
Inspect dataset loading and prompt verification without loading model weights:
```bash
python main.py --dry-run
```

### B. Evaluate API Vision Models (Gemini 3.8 Flash / OpenAI)
Run against local 9router (port 20128) or any OpenAI-compatible vision endpoint. Built-in network tolerance includes 600s request timeouts and progressive retry delays for poor internet connections:
```bash
python main.py \
    --model-family api \
    --model-id "ag/gemini-3.8-flash" \
    --api-base-url "http://localhost:20128/v1" \
    --api-timeout 600 \
    --api-retries 5
```

### C. Evaluate Local Open-Source Models

#### Qwen3-VL Series (2B, 4B, 8B):
```bash
python main.py --model-family qwen3vl --model-id "Qwen/Qwen3-VL-2B-Instruct" --load-in-4bit
```

#### Hulu-Med Series:
```bash
python main.py --model-family hulumed --model-id "ZJU-AI4H/Hulu-Med-7B" --frame-size 224
```

#### Lingshu-7B:
```bash
python main.py --model-family lingshu --model-id "lingshu-medical-mllm/Lingshu-7B"
```

#### Mage-VL:
```bash
python main.py --model-family mage_vl --model-id "microsoft/Mage-VL"
```

### D. Automated Multi-Model Runner
Run all benchmarks or a single family automatically:
```bash
# On Windows:
eval_all.bat api
eval_all.bat qwen3vl

# On Linux / Git-Bash:
./eval_all.sh api
./eval_all.sh qwen3vl
```

---

## 6. Output Metrics & Reports

For each run, CSI-Bench produces:
1. `results/<tag>_responses.jsonl` — Raw model responses and extracted reasoning.
2. `results/<tag>_scores.jsonl` — Per-case deterministic scores, extraction method, format validity.
3. `results/<tag>_summary.json` — Machine-readable summary statistics.
4. `results/<tag>_report.md` — Complete academic markdown report featuring:
   - **Overall Accuracy**
   - **Macro-Balanced Accuracy** ($\frac{1}{3} \sum \text{Recall}_k$)
   - **JSON Format Adherence Rate**
   - **Per-Class Metrics:** Sensitivity, Specificity, Precision, F1-Score
   - **3x3 Confusion Matrix**

---

## 7. Citation

```bibtex
@inproceedings{Cataract-1K,
    author    = {Negin Ghamsarian and
                Yosuf El-Shabrawi and
                Sahar Nasirihaghighi and
                Doris Putzgruber-Adamitsch and
                Martin Zinkernagel and
                Sebastian Wolf and
                Klaus Schoeffmann and
                Raphael Sznitman},
    title     = {Cataract-1K: Cataract Surgery Dataset for Scene Segmentation, Phase Recognition, and Irregularity Detection (to appear)},
    
}
```

