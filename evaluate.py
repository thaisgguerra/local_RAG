#!/usr/bin/env python3
"""
Automatic evaluation of the local RAG.

For every question in a CSV it measures:
  - Retrieval: was the expected section among the retrieved chunks?
  - Answer: does the generated text contain the expected keywords?
  - Refusal: for unanswerable questions, did the assistant say "Not in the knowledge base"?
  - Distance: which distance limit best separates answerable from unanswerable questions?

Each result is written to disk immediately, so nothing is lost if the run is interrupted.

Usage (same environment variables as rag.py):
    python evaluate.py                       # full evaluation with the default question set
    python evaluate.py --resume              # continue where it stopped
    python evaluate.py --summarize           # only rebuild the summary from saved results
    python evaluate.py --no-llm              # retrieval and distances only (fast)
    python evaluate.py --questions my.csv    # another question set
    python evaluate.py --label before        # writes results/results_before.csv and summary_before.md

CSV columns: id,type,question,expected_file,expected_section,keywords
  type = answerable | unanswerable. Answerable questions without expected_file/expected_section
  or keywords are still run, and their answers can be reviewed by hand in the results CSV.
  keywords are separated by "|".
"""

import argparse
import csv
import os
import statistics
import time
from datetime import date

import rag

RESULTS_FILE = "results/results.csv"
SUMMARY_FILE = "results/summary.md"
FIELDS = [
    "id", "type", "question", "expected", "retrieval_hit", "best_distance",
    "keywords_found", "answer_correct", "refusal_strict", "refusal_loose",
    "guardrail_refusal", "auto_warning", "seconds", "retrieved_sections", "answer", "error",
]
BOOLS = ("retrieval_hit", "answer_correct", "refusal_strict", "refusal_loose", "guardrail_refusal", "auto_warning")
LOOSE_REFUSAL = (
    "not in the knowledge base", "does not say", "does not specify", "not specified",
    "no information", "not mentioned", "does not include",
    "nao consta", "nao foi especificad", "nao ha informacao", "nao encontrei", "nao esta na base",
)


def refusal_strict(text: str) -> bool:
    return rag.is_refusal_text(text)


def refusal_loose(text: str) -> bool:
    n = rag.strip_accents(text)
    return any(p in n for p in LOOSE_REFUSAL)


def pct(a, b) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def _bool(v):
    return {"True": True, "False": False}.get(v)


def read_results() -> list[dict]:
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in BOOLS:
            r[c] = _bool(r.get(c, ""))
        r["best_distance"] = float(r["best_distance"]) if r.get("best_distance") else None
        r["seconds"] = float(r["seconds"]) if r.get("seconds") else 0.0
        r["error"] = r.get("error", "")
    return rows


def append_row(row: dict, new_file: bool):
    os.makedirs(os.path.dirname(RESULTS_FILE) or ".", exist_ok=True)
    with open(RESULTS_FILE, "w" if new_file else "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in FIELDS})
        f.flush()
        os.fsync(f.fileno())


def sweep_thresholds(ok_distances, unanswerable_distances):
    """Accept if best distance <= limit, refuse if above. Returns the most accurate limit."""
    total = len(ok_distances) + len(unanswerable_distances)
    if total == 0:
        return None, 0.0
    best, best_acc = None, -1.0
    t = 0.30
    while t <= 0.80:
        right = sum(d <= t for d in ok_distances) + sum(d > t for d in unanswerable_distances)
        acc = right / total
        if acc > best_acc + 1e-9:
            best, best_acc = round(t, 2), acc
        t += 0.01
    return best, best_acc


def metrics(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r["error"] and r["best_distance"] is not None]
    ok = [r for r in valid if r["type"] == "answerable"]
    no = [r for r in valid if r["type"] != "answerable"]
    ok_ret = [r for r in ok if r["retrieval_hit"] is not None]
    ok_kw = [r for r in ok if r["answer_correct"] is not None]
    with_llm = any(r["refusal_strict"] is not None for r in valid)
    m = {
        "n_ok": len(ok), "n_no": len(no), "with_llm": with_llm,
        "retrieval": (sum(1 for r in ok_ret if r["retrieval_hit"]), len(ok_ret)),
        "median_ok": statistics.median([r["best_distance"] for r in ok]) if ok else None,
        "median_no": statistics.median([r["best_distance"] for r in no]) if no else None,
    }
    if with_llm:
        m["answer_correct"] = (sum(1 for r in ok_kw if r["answer_correct"]), len(ok_kw))
        m["refusal_strict"] = (sum(1 for r in no if r["refusal_strict"]), len(no))
        m["refusal_loose"] = (sum(1 for r in no if r["refusal_loose"]), len(no))
        m["wrong_refusal"] = (sum(1 for r in ok if r["refusal_strict"]), len(ok))
        m["guardrail_refusals"] = sum(1 for r in valid if r["guardrail_refusal"])
        m["auto_warnings"] = sum(1 for r in valid if r["auto_warning"])
        m["seconds"] = statistics.mean(r["seconds"] for r in valid) if valid else 0.0
    return m


def build_summary(rows: list[dict], total_questions: int) -> str:
    valid = [r for r in rows if not r["error"] and r["best_distance"] is not None]
    errors = [r for r in rows if r["error"]]
    ok = [r for r in valid if r["type"] == "answerable"]
    no = [r for r in valid if r["type"] != "answerable"]
    md = ["# Evaluation result", ""]
    md.append(
        f"Date: {date.today().isoformat()} | Model: {rag.LLM_MODEL} | Embeddings: {rag.EMBED_MODEL} | TOP_K: {rag.TOP_K}"
        f" | Distance guardrail: {rag.MAX_DIST if rag.MAX_DIST > 0 else 'off'}"
        f" | Automatic warning: {'on' if rag.AUTO_WARNING else 'off'}"
    )
    md.append(f"Questions evaluated: {len(valid)} of {total_questions} ({len(ok)} answerable, {len(no)} unanswerable)")
    if len(valid) < total_questions:
        md.append("WARNING: partial evaluation. Use `python evaluate.py --resume` to finish.")
    if not valid:
        return "\n".join(md) + "\n\nNo question was evaluated yet.\n"

    m = metrics(rows)
    d_ok = [r["best_distance"] for r in ok]
    d_no = [r["best_distance"] for r in no]
    ok_ret = [r for r in ok if r["retrieval_hit"] is not None]

    md += ["", "## Retrieval", f"- Expected section among the retrieved chunks: {pct(*m['retrieval'])}"]
    if d_ok:
        line = f"- Median best distance: {m['median_ok']:.2f} (answerable)"
        if d_no:
            line += f" vs {m['median_no']:.2f} (unanswerable)"
        md.append(line)
    misses = [r for r in ok_ret if not r["retrieval_hit"]]
    if misses:
        md.append("- Retrieval misses: " + ", ".join(f"#{r['id']}" for r in misses))

    if m["with_llm"]:
        ok_kw = [r for r in ok if r["answer_correct"] is not None]
        md += ["", "## Generation"]
        md.append(f"- Answers with all expected keywords: {pct(*m['answer_correct'])}")
        md.append(f"- Average time per question (retrieval + answer): {m['seconds']:.0f} s")
        md += ["", "## Refusal (unanswerable questions)"]
        md.append(f"- Refused with 'Not in the knowledge base': {pct(*m['refusal_strict'])}")
        md.append(f"- Refused in any way (loose criterion): {pct(*m['refusal_loose'])}")
        md.append(f"- Wrong refusals (refused something the KB answers): {pct(*m['wrong_refusal'])}")
        md.append(f"- Refusals by the distance guardrail: {m['guardrail_refusals']} | Answers with automatic warning: {m['auto_warnings']}")
        missed = [r for r in no if not r["refusal_loose"]]
        if missed:
            md.append("- Did NOT refuse (hallucination risk): " + ", ".join(f"#{r['id']}" for r in missed))
        incomplete = [r for r in ok_kw if not r["answer_correct"]]
        if incomplete:
            md.append("- Incomplete answers: " + ", ".join(f"#{r['id']}" for r in incomplete))

    limit, acc = sweep_thresholds(d_ok, d_no)
    if limit is not None and d_ok and d_no:
        md += ["", "## Distance guardrail (calibration)"]
        md.append(f"- Best separating limit on this set: {limit} (right {acc * 100:.0f}% of the time)")
        md.append("- Validate on a fresh question set before adopting it. A limit tuned on the same data can overfit.")
    if errors:
        md += ["", "## Errors during the run"]
        for r in errors:
            md.append(f"- #{r['id']}: {r['error'][:120]}")
    return "\n".join(md) + "\n"


def save_summary(total_questions: int):
    text = build_summary(read_results(), total_questions)
    os.makedirs(os.path.dirname(SUMMARY_FILE) or ".", exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + "=" * 70)
    print(text)
    print(f"Files: {SUMMARY_FILE} and {RESULTS_FILE}")


def evaluate_question(cli, prompt, q, no_llm: bool) -> dict:
    t0 = time.time()
    answerable = q["type"] == "answerable"
    exp_file, exp_section = q.get("expected_file", ""), q.get("expected_section", "")
    has_key = answerable and bool(exp_file) and bool(exp_section)
    results = rag.retrieve(cli, q["question"])
    row = {
        "id": q["id"], "type": q["type"], "question": q["question"],
        "expected": f"{exp_file} > {exp_section}" if has_key else "",
        "best_distance": round(rag.best_distance(results), 3),
        "retrieved_sections": " | ".join(f"{m['file']} > {m['section']}" for _, m, _ in results),
        "error": "",
    }
    if has_key:
        row["retrieval_hit"] = any(m["file"] == exp_file and m["section"] == exp_section for _, m, _ in results)
    if not no_llm:
        text, by_guardrail = rag.full_answer(cli, q["question"], prompt, results)
        row["answer"] = text.replace("\n", " ").strip()
        row["refusal_strict"] = refusal_strict(text)
        row["refusal_loose"] = refusal_loose(text)
        row["guardrail_refusal"] = by_guardrail
        row["auto_warning"] = text.startswith("Warning:") or "\n\nWarning: this answer uses" in text
        keywords = [k.strip() for k in q.get("keywords", "").split("|") if k.strip()]
        if answerable and keywords:
            found = sum(rag.strip_accents(k) in rag.strip_accents(text) for k in keywords)
            row["keywords_found"] = f"{found}/{len(keywords)}"
            row["answer_correct"] = found == len(keywords)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    global RESULTS_FILE, SUMMARY_FILE
    ap = argparse.ArgumentParser(description="Evaluate the local RAG")
    ap.add_argument("--questions", default="eval/questions_sample.csv")
    ap.add_argument("--no-llm", action="store_true", help="evaluate retrieval only")
    ap.add_argument("--resume", action="store_true", help="continue where the last run stopped")
    ap.add_argument("--summarize", action="store_true", help="only rebuild the summary from saved results")
    ap.add_argument("--label", default="", help="suffix of the output files (e.g. before, after)")
    args = ap.parse_args()
    if args.label:
        RESULTS_FILE = f"results/results_{args.label}.csv"
        SUMMARY_FILE = f"results/summary_{args.label}.md"

    with open(args.questions, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    if args.summarize:
        save_summary(len(questions))
        return

    done, new_file = set(), True
    if args.resume and os.path.exists(RESULTS_FILE):
        done = {r["id"] for r in read_results() if not r["error"]}
        new_file = False
        print(f"Resuming: {len(done)} questions already evaluated.")

    cli = rag.ollama_client()
    prompt = rag.load_system_prompt()
    print(f"Evaluating {len(questions)} questions | model: {rag.LLM_MODEL} | TOP_K: {rag.TOP_K}"
          f" | guardrail: {rag.MAX_DIST if rag.MAX_DIST > 0 else 'off'}"
          f" | warning: {'on' if rag.AUTO_WARNING else 'off'}")
    print("Each result is saved immediately. Ctrl+C stops the run and writes a partial summary.\n")
    try:
        for i, q in enumerate(questions, 1):
            if q["id"] in done:
                continue
            try:
                row = evaluate_question(cli, prompt, q, args.no_llm)
                mark = ""
                if q["type"] == "answerable":
                    if "retrieval_hit" in row:
                        mark = f"retrieval={'OK' if row['retrieval_hit'] else 'MISS'}"
                    if "answer_correct" in row:
                        mark += f" answer={'OK' if row['answer_correct'] else 'INCOMPLETE'}"
                    elif "answer" in row:
                        mark += " (review answer in the CSV)"
                elif "refusal_strict" in row:
                    mark = f"refusal={'OK' if row['refusal_strict'] else 'DID NOT REFUSE'}"
                print(f"[{i}/{len(questions)}] {q['question'][:55]:55} dist={row['best_distance']:.2f} {mark} ({row['seconds']:.0f}s)", flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # one failing question does not stop the whole run
                row = {"id": q["id"], "type": q["type"], "question": q["question"], "error": f"{type(e).__name__}: {e}"}
                print(f"[{i}/{len(questions)}] ERROR on #{q['id']}: {row['error'][:100]}", flush=True)
            append_row(row, new_file)
            new_file = False
    except KeyboardInterrupt:
        print("\nInterrupted. Writing partial summary...")
    save_summary(len(questions))


if __name__ == "__main__":
    main()
