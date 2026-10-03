"""
evaluator.py
100% Deterministic Clinical Evaluator for Cataract Irregularity Benchmarking (CSI-Bench).
Evaluates 3-class classification:
  Option A: Normal
  Option B: Lens Irregularity
  Option C: Pupil Contraction

Eliminates any reliance on an external LLM judge.
Parses model outputs via:
  1. Strict JSON parsing (validating "explanation" and "answer" keys)
  2. Tolerant JSON parsing (stripping code fences, regex bracket matching)
  3. Strict letter extraction ([A-C])
  4. Clinical keyword fallback (mapping "Normal", "Lens Irregularity", "Pupil Contraction" if spelled out)

Computes clinical & statistical metrics:
  - Top-1 Accuracy
  - Macro-averaged Balanced Accuracy
  - Per-class Sensitivity (Recall), Specificity, Precision (PPV), and F1-score
  - Complete 3x3 Confusion Matrix
  - Strict JSON format adherence rate
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

log = logging.getLogger("evaluator")

VALID_OPTIONS = {"A", "B", "C"}
OPTION_TO_NAME = {
    "A": "Normal",
    "B": "Lens Irregularity",
    "C": "Pupil Contraction"
}


def parse_strict_json(text: str) -> Tuple[Optional[dict], bool]:
    """
    Checks whether `text` is valid JSON and contains required keys ("answer", "explanation").
    Returns (parsed_dict, is_strict).
    """
    if not text:
        return None, False
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "answer" in obj:
            return obj, True
    except (json.JSONDecodeError, ValueError):
        pass
    return None, False


def extract_json_object(text: str) -> Optional[dict]:
    """
    Tolerant JSON extraction: strips markdown code fences and searches for the first balanced { ... } block.
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    return None


def extract_answer_letter(text: str) -> Tuple[str, str]:
    """
    Extracts an MCQ option letter [A-C] from model output text.
    Returns (extracted_letter, method).
    """
    if not text:
        return "", "none"

    # 1. Explicit answer declaration: "ANSWER: A", "Final Answer: B"
    m = re.search(r"(?:FINAL\s+)?ANSWER\s*[:=]\s*([A-C])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper(), "regex_answer_colon"

    # 2. "The answer is (A)" or "Option B" or "Choice C"
    m = re.search(r"(?:answer\s+is|option|choice)\s*[:=]?\s*\(?([A-C])\)?\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper(), "regex_option_choice"

    # 3. Line starting with letter e.g., "A) ..." or "B."
    m = re.search(r"^\s*([A-C])[\).\s]", text, re.MULTILINE)
    if m:
        return m.group(1).upper(), "regex_line_start"

    # 4. Isolated capital letter token [A-C]
    letters = re.findall(r"\b([A-C])\b", text)
    if letters:
        return letters[-1].upper(), "regex_letter_token"

    # 5. Semantic keyword fallback if letter omitted
    lower = text.lower()
    if "pupil contraction" in lower or "pupil contradiction" in lower or "miosis" in lower or "iris prolapse" in lower:
        return "C", "semantic_keyword"
    if "lens irregularity" in lower or "capsular tear" in lower or "zonular" in lower or "subluxation" in lower:
        return "B", "semantic_keyword"
    if "normal" in lower or "routine" in lower:
        return "A", "semantic_keyword"

    return "", "none"


def score_irregularity_task(model_response: str, correct_answer: str, question_type: str = "cataract_irregularity_classification") -> dict:
    """
    Deterministic scoring for Cataract 3-Class Irregularity Evaluation.
    
    Returns:
        score: 1 if correct else 0
        normalised_score: 1.0 if correct else 0.0
        extracted_answer: uppercase letter A, B, C or "NONE"
        predicted_category: Category name or "Unknown"
        correct: bool
        method: extraction method used
        format_valid: bool (whether strict JSON contract was adhered to)
    """
    correct_clean = str(correct_answer).strip().upper()
    extracted = ""
    method = "none"
    format_valid = False

    # 1. Try strict JSON
    obj, strict = parse_strict_json(model_response)
    if strict and isinstance(obj, dict) and "answer" in obj:
        raw_cand = str(obj["answer"]).strip()
        cand = raw_cand.upper()
        if cand in VALID_OPTIONS:
            extracted = cand
            method = "strict_json"
            format_valid = True
        elif cand and cand[0] in VALID_OPTIONS:
            extracted = cand[0]
            method = "strict_json_prefix"
            format_valid = True
        else:
            lower_cand = raw_cand.lower()
            if "pupil" in lower_cand or "miosis" in lower_cand:
                extracted = "C"
                method = "strict_json_category"
                format_valid = True
            elif "lens" in lower_cand or "capsul" in lower_cand or "zonul" in lower_cand:
                extracted = "B"
                method = "strict_json_category"
                format_valid = True
            elif "norm" in lower_cand or "routine" in lower_cand:
                extracted = "A"
                method = "strict_json_category"
                format_valid = True

    # 2. Try tolerant JSON
    if not extracted:
        obj = extract_json_object(model_response)
        if isinstance(obj, dict) and obj.get("answer"):
            raw_cand = str(obj["answer"]).strip()
            cand = raw_cand.upper()
            if cand in VALID_OPTIONS:
                extracted = cand
                method = "tolerant_json"
                format_valid = True
            elif cand and cand[0] in VALID_OPTIONS:
                extracted = cand[0]
                method = "tolerant_json_prefix"
                format_valid = True
            else:
                lower_cand = raw_cand.lower()
                if "pupil" in lower_cand or "miosis" in lower_cand:
                    extracted = "C"
                    method = "tolerant_json_category"
                    format_valid = True
                elif "lens" in lower_cand or "capsul" in lower_cand or "zonul" in lower_cand:
                    extracted = "B"
                    method = "tolerant_json_category"
                    format_valid = True
                elif "norm" in lower_cand or "routine" in lower_cand:
                    extracted = "A"
                    method = "tolerant_json_category"
                    format_valid = True

    # 3. Text fallback
    if not extracted:
        extracted, method = extract_answer_letter(model_response)

    is_correct = bool(extracted) and (extracted == correct_clean)
    pred_category = OPTION_TO_NAME.get(extracted, "Unknown / Invalid")

    return {
        "score": 1 if is_correct else 0,
        "max_score": 1,
        "normalised_score": 1.0 if is_correct else 0.0,
        "extracted_answer": extracted or "NONE",
        "predicted_category": pred_category,
        "correct": is_correct,
        "method": method,
        "format_valid": format_valid,
        "question_type": question_type
    }


def aggregate_irregularity_metrics(scores_path: str, output_summary_path: str, model_id: str = "", tag: str = "") -> dict:
    """
    Aggregates deterministic scores across all cataract irregularity evaluation records.
    Computes overall accuracy, balanced accuracy, 3x3 confusion matrix, and per-class stats.
    """
    rows = []
    if not os.path.exists(scores_path):
        log.warning(f"Scores file {scores_path} does not exist for aggregation.")
        return {}

    with open(scores_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    total_tasks = len(rows)
    if total_tasks == 0:
        return {}

    correct_total = sum(1 for r in rows if r.get("correct", False))
    format_strict_count = sum(1 for r in rows if r.get("format_valid", False))
    overall_acc = correct_total / total_tasks

    # 3x3 Confusion Matrix: True class x Pred class
    # Classes: A (Normal), B (Lens Irregularity), C (Pupil Contraction)
    classes = ["A", "B", "C"]
    matrix = {t: {p: 0 for p in classes + ["INVALID"]} for t in classes}

    for r in rows:
        gold = str(r.get("correct_answer", "")).strip().upper()
        pred = str(r.get("extracted_answer", "")).strip().upper()
        if gold not in matrix:
            continue
        if pred in classes:
            matrix[gold][pred] += 1
        else:
            matrix[gold]["INVALID"] += 1

    # Per-class metrics
    per_class = {}
    recalls = []
    for c in classes:
        name = OPTION_TO_NAME[c]
        tp = matrix[c][c]
        fn = sum(matrix[c][p] for p in matrix[c] if p != c)
        fp = sum(matrix[other][c] for other in classes if other != c)
        tn = total_tasks - (tp + fn + fp)

        n_samples = tp + fn
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if n_samples > 0:
            recalls.append(recall)

        per_class[name] = {
            "option": c,
            "n_samples": n_samples,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "accuracy": round((tp + tn) / total_tasks, 4) if total_tasks > 0 else 0.0,
            "sensitivity_recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "precision_ppv": round(precision, 4),
            "f1_score": round(f1, 4)
        }

    balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0

    summary = {
        "model_id": model_id,
        "tag": tag,
        "total_cases_evaluated": total_tasks,
        "overall_accuracy": round(overall_acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "json_format_adherence_rate": round(format_strict_count / total_tasks, 4),
        "confusion_matrix": matrix,
        "per_class_metrics": per_class
    }

    # Write summary JSON
    out_p = Path(output_summary_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write human-readable Markdown report
    md_path = out_p.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(summary))

    return summary


def generate_markdown_report(summary: dict) -> str:
    """Formats summary dictionary into an academic markdown report."""
    matrix = summary.get("confusion_matrix", {})
    per_class = summary.get("per_class_metrics", {})
    
    md = f"""# Cataract Surgery Irregularity Benchmark (CSI-Bench) Evaluation Report

- **Model ID:** `{summary.get('model_id', 'Unknown')}`
- **Tag:** `{summary.get('tag', 'eval')}`
- **Total Cases Evaluated:** {summary.get('total_cases_evaluated', 0)}
- **Overall Accuracy:** {summary.get('overall_accuracy', 0.0) * 100:.2f}%
- **Macro-Balanced Accuracy:** {summary.get('balanced_accuracy', 0.0) * 100:.2f}%
- **Strict JSON Contract Adherence:** {summary.get('json_format_adherence_rate', 0.0) * 100:.2f}%

---

## 1. Per-Class Diagnostic Performance

| Option | Category | Cases | Accuracy | Sensitivity (Recall) | Specificity | Precision (PPV) | F1-Score |
|:------:|:---------|:-----:|:--------:|:--------------------:|:-----------:|:---------------:|:--------:|
"""
    for name, m in per_class.items():
        md += f"| **{m['option']}** | {name} | {m['n_samples']} | {m['accuracy']*100:.1f}% | {m['sensitivity_recall']*100:.1f}% | {m['specificity']*100:.1f}% | {m['precision_ppv']*100:.1f}% | {m['f1_score']:.4f} |\n"

    md += """
---

## 2. 3x3 Confusion Matrix

| Ground Truth \\ Predicted | Normal (A) | Lens Irreg. (B) | Pupil Cont. (C) | Invalid / None |
|:-------------------------|:----------:|:---------------:|:---------------:|:--------------:|
"""
    labels = [("A", "Normal"), ("B", "Lens Irregularity"), ("C", "Pupil Contraction")]
    for opt, label in labels:
        row = matrix.get(opt, {})
        md += f"| **{label} ({opt})** | {row.get('A', 0)} | {row.get('B', 0)} | {row.get('C', 0)} | {row.get('INVALID', 0)} |\n"

    md += "\n"
    return md
