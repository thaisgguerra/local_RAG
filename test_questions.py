"""The sample question set must point to sections that really exist in the sample KB."""
import csv
from pathlib import Path

import rag

ROOT = Path(__file__).resolve().parent.parent


def test_expected_sections_exist():
    chunks = rag.load_chunks(ROOT / "kb_sample")
    existing = {(c["file"], c["section"]) for c in chunks}
    with open(ROOT / "eval/questions_sample.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    answerable = [r for r in rows if r["type"] == "answerable"]
    assert len(answerable) >= 15 and len(rows) > len(answerable)
    for r in answerable:
        assert (r["expected_file"], r["expected_section"]) in existing, r["id"]


def test_keywords_appear_in_expected_section():
    chunks = rag.load_chunks(ROOT / "kb_sample")
    by_section = {}
    for c in chunks:
        by_section.setdefault((c["file"], c["section"]), "")
        by_section[(c["file"], c["section"])] += rag.strip_accents(c["text"])
    with open(ROOT / "eval/questions_sample.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["type"] != "answerable":
                continue
            text = by_section[(r["expected_file"], r["expected_section"])]
            kws = [k for k in r["keywords"].split("|") if k]
            assert any(rag.strip_accents(k) in text for k in kws), (r["id"], kws)
