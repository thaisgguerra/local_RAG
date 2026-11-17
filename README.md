# rag-local-eval

A fully local RAG assistant (Ollama + Chroma) over a Markdown knowledge base, built together with the part most
RAG demos skip: **an evaluation harness** that measures retrieval, answer quality and refusal, and shows where a
"safety" feature quietly breaks.

> Built as a product exercise by a Product Manager: define the problem, define what "reliable" means, measure it,
> and decide from the data.

## The problem

Technical knowledge lived in scattered documents and in the heads of the engineers. Prototyping and validating ideas
meant interrupting the team, even for questions already answered somewhere. The goal: an assistant that answers from
the documents, cites the source, says "I don't know" when the documents do not know, and warns when a fact is not
validated yet. And a way to know whether it can be trusted.

## What it does

- Splits Markdown by section, embeds with `bge-m3`, stores in Chroma (cosine).
- Answers with a grounded prompt, citing `file > section`.
- Refuses with a fixed phrase, `Not in the knowledge base.`, when the passages do not contain the answer.
- Appends a warning, written by code and not by the model, when a relevant passage is marked `[TO VERIFY]`.
- Optional distance guardrail (off by default, see the findings below).
- `evaluate.py` and `compare.py` for before/after experiments.

Everything runs on your machine. No document leaves it.

## Quickstart

Requirements: Python 3.10+, [Ollama](https://ollama.com).

```bash
ollama pull bge-m3
ollama pull llama3.2:3b

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python rag.py ingest               # indexes kb_sample/
python rag.py ask "How long is the pickup QR code valid?"
python rag.py chat                 # interactive
```

The sample knowledge base in `kb_sample/` is fictitious (a smart parcel locker company).

## Evaluate

```bash
python evaluate.py                 # 28 questions: retrieval, answer keywords, refusal
python evaluate.py --no-llm        # retrieval only, fast
python evaluate.py --resume        # continue after an interruption

# before/after experiment
AUTO_WARNING=0 python evaluate.py --label before
MAX_DIST=0.45  python evaluate.py --label after
python compare.py before after
```

Results are written per question as the run goes, so nothing is lost if you stop it.

## Key findings (from the original study on a private KB)

| Test set (34 questions) | Correct answer | Strict refusal | Wrong refusal |
|---|---|---|---|
| Baseline (3B, no guardrail) | 96% | 60% | 4% |
| 3B + guardrail 0.45 + warning | 92% | 90% | 8% |

| Informal questions (18) | Wrong refusal |
|---|---|
| 3B + guardrail 0.45 + warning | 38% (5 of 13 answerable refused) |

1. The distance guardrail improved refusal on my test set, but **blocked 38% of answerable informal questions**.
   A threshold tuned on your own data does not generalize to the way real users ask.
2. The 8B model was 2 to 3 times slower and did not change the aggregate numbers. The bottleneck was the threshold.
3. The 3B model varies between identical runs, so one question of difference is noise.

Full tables, method and limitations: [docs/evaluation.md](docs/evaluation.md).
Design decisions: [docs/architecture.md](docs/architecture.md).

## Honest limitations

- Small question sets (one question moves a metric by 3 to 8 points). Each configuration ran once.
- The answer metric is keyword-based. It does not detect invented details.
- The sample KB is small and fictitious, so numbers you get here are not comparable to the study above.
- The proposed fix (two-band guardrail) is documented, not implemented.

## Roadmap

- [ ] Two-band guardrail (answer, answer with caution, refuse) validated on unseen questions
- [ ] LLM judge or human review for answer quality
- [ ] Query rewriting for informal language
- [ ] Repeat runs and report ranges

## Project layout

```
rag.py                 ingest, ask, chat
evaluate.py            evaluation harness
compare.py             before/after comparison
prompts/               system prompt
kb_sample/             fictitious knowledge base
eval/                  sample question set
docs/                  architecture and evaluation report
tests/                 unit tests (no Ollama needed)
```

## Tests

```bash
pip install pytest
pytest -q
```

## License

MIT
