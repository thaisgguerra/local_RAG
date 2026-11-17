"""Tests that run without Ollama or Chroma (pure logic: chunking and guardrails)."""
from pathlib import Path

import rag

KB = Path(__file__).resolve().parent.parent / "kb_sample"


def fake_results(*items):
    """items: (distance, unverified)"""
    return [
        (f"text {i}", {"file": f"doc{i}.md", "section": f"Section {i}", "unverified": unv}, d)
        for i, (d, unv) in enumerate(items)
    ]


def test_every_kb_file_produces_chunks():
    chunks = rag.load_chunks(KB)
    files = {c["file"] for c in chunks}
    assert len(files) == 5
    assert all(c["text"].strip() for c in chunks)


def test_chunk_metadata_and_unverified_flag():
    chunks = rag.load_chunks(KB)
    assert any(c["unverified"] for c in chunks)
    assert all({"file", "section", "unverified", "id"} <= c.keys() for c in chunks)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_long_section_is_split_without_losing_text():
    body = "\n".join(f"- line number {i} with some words" for i in range(200))
    parts = rag.split_long_section(body, max_chars=300)
    assert len(parts) > 1
    assert all(len(p) <= 400 for p in parts)
    assert "\n".join(parts).count("line number") == 200


def test_files_with_ignored_prefix_are_not_indexed(tmp_path):
    (tmp_path / "_template.md").write_text("# T\n\n## A\nbody", encoding="utf-8")
    (tmp_path / "doc.md").write_text("# D\n\n## A\nbody", encoding="utf-8")
    assert {c["file"] for c in rag.load_chunks(tmp_path)} == {"doc.md"}


def test_guardrail_off_by_default_never_refuses(monkeypatch):
    monkeypatch.setattr(rag, "MAX_DIST", 0)
    assert rag.should_refuse(fake_results((0.99, False))) is False


def test_guardrail_refuses_only_above_limit(monkeypatch):
    monkeypatch.setattr(rag, "MAX_DIST", 0.45)
    assert rag.should_refuse(fake_results((0.50, False), (0.60, False))) is True
    assert rag.should_refuse(fake_results((0.40, False), (0.60, False))) is False


def test_refusal_text_detection_ignores_accents():
    assert rag.is_refusal_text("Not in the knowledge base.")
    assert rag.is_refusal_text("Não consta na base.")
    assert not rag.is_refusal_text("The database is PostgreSQL.")


def test_warning_lists_only_close_unverified_chunks(monkeypatch):
    monkeypatch.setattr(rag, "AUTO_WARNING", True)
    close = fake_results((0.30, False), (0.32, True), (0.70, True))
    msg = rag.unverified_warning(close)
    assert "doc1.md > Section 1" in msg
    assert "doc2.md" not in msg


def test_no_warning_when_nothing_unverified_or_disabled(monkeypatch):
    monkeypatch.setattr(rag, "AUTO_WARNING", True)
    assert rag.unverified_warning(fake_results((0.3, False))) == ""
    monkeypatch.setattr(rag, "AUTO_WARNING", False)
    assert rag.unverified_warning(fake_results((0.3, True))) == ""


def test_full_answer_appends_warning_and_skips_it_on_refusal(monkeypatch):
    monkeypatch.setattr(rag, "MAX_DIST", 0)
    monkeypatch.setattr(rag, "AUTO_WARNING", True)

    class FakeClient:
        def __init__(self, text):
            self.text = text

        def chat(self, **_kw):
            return {"message": {"content": self.text}}

    results = fake_results((0.30, True))
    text, by_guard = rag.full_answer(FakeClient("It is 24 hours."), "q", "sys", results)
    assert "Warning:" in text and by_guard is False
    text, _ = rag.full_answer(FakeClient("Not in the knowledge base."), "q", "sys", results)
    assert "Warning:" not in text


def test_full_answer_guardrail_skips_the_model(monkeypatch):
    monkeypatch.setattr(rag, "MAX_DIST", 0.45)

    class Boom:
        def chat(self, **_kw):
            raise AssertionError("model must not be called")

    text, by_guard = rag.full_answer(Boom(), "q", "sys", fake_results((0.9, False)))
    assert by_guard is True and rag.is_refusal_text(text)
