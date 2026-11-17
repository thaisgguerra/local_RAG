#!/usr/bin/env python3
"""
Local RAG over a Markdown knowledge base.

Stack: Ollama (bge-m3 embeddings + a Llama model) and Chroma (local vector database).

Commands:
    python rag.py ingest                # (re)index the .md files in KB_DIR
    python rag.py ask "your question"   # one-off question
    python rag.py chat                  # interactive mode

Configuration (environment variables, all optional):
    KB_DIR        folder with the .md files             (default: ./kb_sample)
    DB_DIR        folder of the vector database         (default: ./chroma_db)
    PROMPT_FILE   system prompt file                    (default: ./prompts/system_prompt.txt)
    OLLAMA_HOST   Ollama address                        (default: http://localhost:11434)
    EMBED_MODEL   embedding model                       (default: bge-m3)
    LLM_MODEL     generation model                      (default: llama3.2:3b)
    TOP_K         chunks retrieved per question         (default: 5)
    MAX_DIST      distance guardrail: if the best chunk is farther than this value, refuse
                  without calling the model. 0 turns it off (default: 0, see docs/evaluation.md)
    AUTO_WARNING  1 adds an automatic warning when a relevant chunk is marked as unverified
                  (default: 1; use 0 to turn off)
"""

import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path

KB_DIR = Path(os.getenv("KB_DIR", "./kb_sample"))
DB_DIR = os.getenv("DB_DIR", "./chroma_db")
PROMPT_FILE = Path(os.getenv("PROMPT_FILE", "./prompts/system_prompt.txt"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_DIST = float(os.getenv("MAX_DIST", "0"))
AUTO_WARNING = os.getenv("AUTO_WARNING", "1") != "0"
WARNING_MARGIN = 0.06  # only warn about unverified chunks almost as relevant as the best one

COLLECTION = "kb_chunks"
MAX_CHARS = 1200  # approximate maximum size of a chunk
EMBED_BATCH = 16
IGNORE_PREFIXES = ("_",)  # files starting with these prefixes are not indexed

# A chunk containing any of these markers is treated as "not yet verified by the owner".
# The Portuguese marker is kept so the same code works with a private Portuguese knowledge base.
UNVERIFIED_MARKERS = ("[TO VERIFY]", "[A VALIDAR]")

# Phrases that count as a refusal in the answer text (compared without accents, lowercase).
REFUSAL_PHRASES = ("not in the knowledge base", "nao consta na base")

REFUSAL_MESSAGE = (
    "Not in the knowledge base. Consider adding this question to the open-questions "
    "file so the owner of the docs can answer it."
)

DEFAULT_PROMPT = (
    "You are a technical assistant. Answer only from the passages provided.\n"
    "1. Give complete, well-structured answers: start with a one-sentence summary, then the details.\n"
    "2. Cite the file and section of each piece of information.\n"
    '3. If the passages do not contain the answer, reply "Not in the knowledge base."\n'
    "4. If a passage carries an unverified marker, say it needs confirmation from the owner.\n"
    "5. Never invent numbers, models, components or features.\n"
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_long_section(body: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split a long section into groups of lines without breaking a bullet in the middle."""
    groups, current, size = [], [], 0
    for line in body.split("\n"):
        if current and size + len(line) > max_chars:
            groups.append("\n".join(current).strip())
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        groups.append("\n".join(current).strip())
    return [g for g in groups if g]


def split_document(path: Path) -> list[dict]:
    """Read a .md file and return one chunk per section (##), with metadata."""
    text = path.read_text(encoding="utf-8")
    doc_title = path.stem
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if m:
        doc_title = m.group(1).strip()

    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)
    sections = []
    header = re.sub(r"^#\s+.+$", "", parts[0], flags=re.MULTILINE).strip()
    if header:
        sections.append(("Source and context", header))
    for part in parts[1:]:
        first, _, rest = part.partition("\n")
        sections.append((first.strip(), rest.strip()))

    chunks = []
    for section_title, body in sections:
        if not body:
            continue
        for i, piece in enumerate(split_long_section(body)):
            chunks.append(
                {
                    "id": f"{path.name}:{section_title}:{i}",
                    "text": f"{doc_title} > {section_title}\n{piece}",
                    "file": path.name,
                    "section": section_title,
                    "unverified": any(mk in piece for mk in UNVERIFIED_MARKERS),
                }
            )
    return chunks


def load_chunks(kb_dir: Path) -> list[dict]:
    files = sorted(p for p in kb_dir.glob("*.md") if not p.name.startswith(IGNORE_PREFIXES))
    if not files:
        sys.exit(f"No .md files found in {kb_dir.resolve()}")
    chunks = []
    for f in files:
        chunks.extend(split_document(f))
    return chunks


def load_system_prompt() -> str:
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# Deterministic guardrails (they do not depend on the model)
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFKD", text)
    return s.encode("ascii", "ignore").decode("ascii").lower()


def best_distance(results) -> float:
    return min(d for _t, _m, d in results)


def should_refuse(results) -> bool:
    """Distance guardrail: the best chunk is too far from the question."""
    return MAX_DIST > 0 and best_distance(results) > MAX_DIST


def is_refusal_text(text: str) -> bool:
    n = strip_accents(text)
    return any(p in n for p in REFUSAL_PHRASES)


def unverified_warning(results) -> str:
    """Automatic warning, written by code and not by the model, for unverified chunks."""
    if not AUTO_WARNING:
        return ""
    limit = best_distance(results) + WARNING_MARGIN
    used = []
    for _t, meta, dist in results:
        if meta.get("unverified") and dist <= limit:
            ref = f"{meta['file']} > {meta['section']}"
            if ref not in used:
                used.append(ref)
    if not used:
        return ""
    return (
        "Warning: this answer uses information marked as unverified ("
        + "; ".join(used)
        + "). Confirm with the owner of the docs before relying on it."
    )


# ---------------------------------------------------------------------------
# Ollama and Chroma
# ---------------------------------------------------------------------------

def ollama_client():
    try:
        import ollama
    except ImportError:
        sys.exit("Install the dependencies first: pip install -r requirements.txt")
    return ollama.Client(host=OLLAMA_HOST)


def get_collection(reset: bool = False):
    try:
        import chromadb
    except ImportError:
        sys.exit("Install the dependencies first: pip install -r requirements.txt")
    db = chromadb.PersistentClient(path=DB_DIR)
    if reset:
        try:
            db.delete_collection(COLLECTION)
        except Exception:
            pass
    return db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def embed(cli, texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = cli.embed(model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        out.extend(resp["embeddings"])
    return out


def retrieve(cli, question: str):
    col = get_collection()
    if col.count() == 0:
        sys.exit("The vector database is empty. Run first: python rag.py ingest")
    emb = embed(cli, [question])[0]
    res = col.query(query_embeddings=[emb], n_results=min(TOP_K, col.count()))
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def build_context(results) -> str:
    blocks = []
    for i, (text, meta, _dist) in enumerate(results, 1):
        mark = " (contains an unverified marker)" if meta.get("unverified") else ""
        blocks.append(f"### Passage {i}: {meta['file']} > {meta['section']}{mark}\n{text}")
    return "\n\n".join(blocks)


def build_messages(question: str, system_prompt: str, results) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Passages from the knowledge base:\n\n{build_context(results)}\n\nQuestion: {question}",
        },
    ]


def full_answer(cli, question: str, system_prompt: str, results):
    """Non-streaming answer, used by the evaluation. Returns (text, refused_by_guardrail)."""
    if should_refuse(results):
        return REFUSAL_MESSAGE, True
    resp = cli.chat(
        model=LLM_MODEL,
        messages=build_messages(question, system_prompt, results),
        options={"temperature": 0.1},
    )
    text = resp["message"]["content"]
    if not is_refusal_text(text):
        warning = unverified_warning(results)
        if warning:
            text = f"{text}\n\n{warning}"
    return text, False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_ingest(_args):
    chunks = load_chunks(KB_DIR)
    print(f"{len(chunks)} chunks found in {KB_DIR.resolve()}")
    cli = ollama_client()
    embeddings = embed(cli, [c["text"] for c in chunks])
    col = get_collection(reset=True)  # small KB: reindex everything so deleted text disappears
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"file": c["file"], "section": c["section"], "unverified": c["unverified"]} for c in chunks],
    )
    n_unverified = sum(c["unverified"] for c in chunks)
    print(f"Indexing finished: {len(chunks)} chunks ({n_unverified} unverified).")


def print_sources(results):
    print("\n\nSources consulted:")
    seen = set()
    for _text, meta, dist in results:
        key = (meta["file"], meta["section"])
        if key in seen:
            continue
        seen.add(key)
        flag = "  [unverified]" if meta.get("unverified") else ""
        print(f"  - {meta['file']} > {meta['section']} (distance {dist:.2f}){flag}")


def answer(cli, question: str, system_prompt: str):
    results = retrieve(cli, question)
    print()
    if should_refuse(results):
        print(REFUSAL_MESSAGE)
        print(f"(Automatic refusal: best distance {best_distance(results):.2f} "
              f"is above the limit {MAX_DIST:.2f})", end="")
        print_sources(results)
        return
    text = ""
    for part in cli.chat(
        model=LLM_MODEL,
        messages=build_messages(question, system_prompt, results),
        stream=True,
        options={"temperature": 0.1},
    ):
        piece = part["message"]["content"]
        text += piece
        print(piece, end="", flush=True)
    if not is_refusal_text(text):
        warning = unverified_warning(results)
        if warning:
            print(f"\n\n{warning}", end="")
    print_sources(results)


def cmd_ask(args):
    answer(ollama_client(), args.question, load_system_prompt())


def cmd_chat(_args):
    cli = ollama_client()
    prompt = load_system_prompt()
    print("Local RAG. Type your question (or 'exit').")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "sair"}:
            break
        if question:
            answer(cli, question, prompt)


def main():
    parser = argparse.ArgumentParser(description="Local RAG over a Markdown knowledge base")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="index the .md files").set_defaults(func=cmd_ingest)
    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)
    sub.add_parser("chat", help="interactive mode").set_defaults(func=cmd_chat)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
