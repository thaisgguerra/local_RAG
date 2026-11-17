# Architecture

```
 Markdown docs (kb_sample/*.md)
        |  split by "## " sections, max ~1200 chars
        v
   chunks + metadata (file, section, unverified)
        |  embeddings: bge-m3 via Ollama
        v
   Chroma (persistent, cosine distance)
        ^
        |  top-k (default 5) with distances
 question --- embed ---+
        |
        v
  [optional] distance guardrail: best distance > MAX_DIST ? refuse without calling the model
        |
        v
  grounded prompt: rules + retrieved passages + question
        |  Llama via Ollama (temperature 0.1)
        v
  answer with citations (file > section)
        |
        v
  [code, not model] append warning if a close chunk is marked [TO VERIFY]
```

## Design decisions

| Decision | Why |
|---|---|
| Fully local (Ollama, Chroma) | Internal documents never leave the machine and there is no cost per query. Trade-off: speed and quality of small models. |
| Chunk by document section | Each `##` section is one idea, so citations are precise (`file > section`). Long sections are split by lines, never in the middle of a bullet. |
| Metadata `unverified` | Documents mark uncertain facts with `[TO VERIFY]`. The flag travels with the chunk and drives the warning. |
| Warning written by code | A 3B model does not follow "add a warning" reliably. Code does, every time. |
| Guardrail off by default | It did not generalize to informal questions. See `evaluation.md`. |
| Refusal phrase is fixed | "Not in the knowledge base." can be detected by code and counted in the evaluation. |
| Evaluation writes each result immediately | A run of 50 questions on a small model takes minutes. An interruption must not lose data (`--resume`). |

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `KB_DIR` | `./kb_sample` | Folder with the `.md` documents |
| `DB_DIR` | `./chroma_db` | Where the vector database is stored |
| `EMBED_MODEL` | `bge-m3` | Embedding model in Ollama |
| `LLM_MODEL` | `llama3.2:3b` | Chat model in Ollama |
| `TOP_K` | `5` | Chunks sent to the model |
| `MAX_DIST` | `0` | Distance guardrail limit (0 turns it off) |
| `AUTO_WARNING` | `1` | Append the unverified warning (`0` turns it off) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama address |
| `PROMPT_FILE` | `./prompts/system_prompt.txt` | System prompt |

## Using your own documents

1. Put `.md` files in a new folder, for example `kb_private/` (already in `.gitignore`).
2. Start each fact section with `## Title`. Mark uncertain facts with `[TO VERIFY]`.
3. Run `KB_DIR=kb_private python rag.py ingest` (on Windows PowerShell: `$env:KB_DIR="kb_private"`).
4. Write 30 or more questions, including some the KB cannot answer, and some in informal language.
