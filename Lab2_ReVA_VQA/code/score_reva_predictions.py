#!/usr/bin/env python3
"""Score Qwen-style ReVA prediction JSONL files.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_answer(text: str) -> str:
    """Extract text inside <answer>...</answer>; fall back to full text."""
    # TODO(student): use regex with DOTALL, strip whitespace, and return a string.
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def extract_letter(text: str) -> str:
    """Extract one answer letter A-H from model output."""
    # TODO(student): normalize case and robustly parse A-H from short or verbose output.
    answer_text = extract_answer(text)
    match = re.search(r"\b([A-Da-d])\b", answer_text)
    if match:
        return match.group(1).upper()
    return ""


def load_predictions(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL files from output_dir, ignoring result.json."""
    # TODO(student): read every prediction *.json line-by-line and index by item["id"].
    preds = {}
    for filename in sorted(os.listdir(output_dir)):
        if filename == "result.json":
            continue
        if not filename.endswith(".json"):
            continue

        path = os.path.join(output_dir, filename)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                preds[item["id"]] = item
    return preds


def score_predictions(preds: dict[str, dict[str, Any]], gt_list: list[dict[str, Any]]) -> tuple[dict, str]:
    """Score every GT item and report completion separately from accuracy."""
    # TODO(student): compare predictions against every GT item, count missing predictions as incorrect, and report Completed plus Total/subcategory accuracy statistics.
    results = {}
    subcat_totals = {}

    num_completed = 0
    num_correct = 0

    for gt in gt_list:
        qa_id = gt["id"]
        question_type = gt.get("question_type", "unknown")
        gt_letter = gt["answer"].upper()
        pred = preds.get(qa_id)

        if pred is not None:
            num_completed += 1
            pred_letter = extract_letter(pred.get("pred", ""))
        else:
            pred_letter = ""

        is_correct = int(pred_letter == gt_letter)
        num_correct += is_correct

        results[qa_id] = {
            "pred_letter": pred_letter,
            "gt_letter": gt_letter,
            "acc": is_correct, 
            "question_type": question_type,
        }

        totals = subcat_totals.setdefault(question_type, [0, 0])
        totals[1] += 1
        totals[0] += is_correct

    num_total = len(gt_list)
    completed_pct = 100*num_completed / num_total if num_total else 0.0
    total_pct = 100*num_correct / num_total if num_total else 0.0

    lines = [
        f"Completed: {num_completed}/{num_total} = {completed_pct:.2f}%",
        f"Total: {num_correct}/{num_total} = {total_pct:.2f}%",
        "",
        "Subcategory Accuracy:",
    ]

    for subcat, (correct, total) in sorted(subcat_totals.items()):
        pct = 100* correct / total if total else 0.0
        lines.append(f"{subcat}:{correct}/{total} = {pct:.2f}%")

    return results, "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = load_predictions(args.output_dir)
    gt_list = json.loads(args.gt_file.read_text(encoding="utf-8"))
    results, csv_text = score_predictions(preds, gt_list)

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Results ===")
    print(csv_text)

    csv_path = args.output_dir / "result.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
