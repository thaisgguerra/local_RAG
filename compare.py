#!/usr/bin/env python3
"""
Compare two evaluation runs (for example before and after an improvement).

    python evaluate.py --label before    # with MAX_DIST=0 and AUTO_WARNING=0
    python evaluate.py --label after     # with the improvements on
    python compare.py                    # compares results/results_before.csv and results_after.csv
    python compare.py before after       # or other labels

Output: a table in the terminal and results/comparison.md
"""

import sys

import evaluate


def load(label: str) -> dict:
    evaluate.RESULTS_FILE = f"results/results_{label}.csv"
    rows = evaluate.read_results()
    if not rows:
        sys.exit(f"No results found for '{label}' (results/results_{label}.csv).")
    return evaluate.metrics(rows)


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, tuple):
        a, b = v
        return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"
    return str(v)


def main():
    before = sys.argv[1] if len(sys.argv) > 1 else "before"
    after = sys.argv[2] if len(sys.argv) > 2 else "after"
    a, b = load(before), load(after)
    items = [
        ("Expected section retrieved", "retrieval"),
        ("Answer has expected keywords", "answer_correct"),
        ("Refused with 'Not in the knowledge base' (unanswerable)", "refusal_strict"),
        ("Refused in any way (unanswerable)", "refusal_loose"),
        ("Wrong refusals (KB has the answer)", "wrong_refusal"),
        ("Refusals by the distance guardrail", "guardrail_refusals"),
        ("Answers with automatic warning", "auto_warnings"),
    ]
    lines = [f"# Comparison: {before} vs {after}", "", f"| Metric | {before} | {after} |", "|---|---|---|"]
    for name, key in items:
        lines.append(f"| {name} | {fmt(a.get(key))} | {fmt(b.get(key))} |")
    if a.get("seconds") is not None and b.get("seconds") is not None:
        lines.append(f"| Average time per question | {a['seconds']:.0f} s | {b['seconds']:.0f} s |")
    text = "\n".join(lines) + "\n"
    with open("results/comparison.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("File written: results/comparison.md")


if __name__ == "__main__":
    main()
