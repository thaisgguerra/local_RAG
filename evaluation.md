# Evaluation report

This report summarizes the evaluation of the first version of this assistant, run on a private technical
knowledge base (7 curated documents, 58 chunks, 9 of them marked as unverified). The private documents are not
published. Only aggregate numbers are shown here. The repository ships a fictitious sample KB so anyone can
reproduce the method, not the exact numbers.

## Method

- **Test set (34 questions):** 24 answerable and 10 unanswerable, written from the documents.
- **Informal set (18 questions):** 13 answerable and 5 unanswerable, written in the way people actually ask
  (short, with slang and missing context). These were written with AI assistance.
- **Metrics:**
  - Retrieval hit: the expected section is among the top 5 chunks.
  - Answer correct: the answer contains all expected keywords (a rough proxy, see limitations).
  - Strict refusal: for unanswerable questions, the answer contains "Not in the knowledge base".
  - Wrong refusal: the assistant refused something the KB can answer.
- **Configurations:** Llama 3.2 3B and Llama 3.1 8B, with and without the distance guardrail (limits 0.45 and 0.48),
  with the automatic unverified warning on.

## Results on the test set (34 questions)

| Configuration | Correct answer | Strict refusal | Wrong refusal | Time per question |
|---|---|---|---|---|
| Baseline: 3B, no guardrail, no warning | 23/24 (96%) | 6/10 (60%) | 1/24 (4%) | 18 s |
| 3B, guardrail 0.45, warning | 22/24 (92%) | 9/10 (90%) | 2/24 (8%) | 13 s |
| 3B, guardrail 0.48, warning | 23/24 (96%) | 8/10 (80%) | 1/24 (4%) | 14 s |
| 8B, guardrail 0.45, warning | 21/24 (88%) | 8/10 (80%) | 2/24 (8%) | 37 s |

Retrieval found the expected section in 24 of 24 answerable questions.

## Results on informal questions (18 questions)

| Configuration | Retrieval | Correct answer | Wrong refusal | Time per question |
|---|---|---|---|---|
| 3B, guardrail 0.45, warning | 11/12 (92%) | 8/12 (67%) | 5/13 (38%) | 9 s |
| 3B, guardrail 0.48, warning | 11/12 (92%) | 7/12 (58%) | 5/13 (38%) | 12 s |
| 8B, guardrail 0.45, warning | 11/12 (92%) | 8/12 (67%) | 5/13 (38%) | 24 s |

## Main finding

The guardrail improved strict refusal on the test set (6 to 9 out of 10), but it refused 5 of 13 answerable
informal questions. In 3 of them the right chunk was ranked first and the model was never called.

The cause is distribution shift. The limit was calibrated on questions written from the documents, and informal
questions sit farther from the chunks (median best distance 0.44 against 0.38). A threshold tuned on your own test
set can look good and still fail on real questions. For that reason the guardrail is **off by default**
(`MAX_DIST=0`) in this repository.

Other observations:

- The 8B model gets content right more often when it receives the right chunk, but it is 2 to 3 times slower
  and did not change the aggregate numbers. The bottleneck was the threshold, not the model.
- The 3B model varies between identical runs, so a difference of one question is noise.

## Limitations

- Small sets (34 and 18 questions): one question moves a metric by 3 to 8 percentage points.
- Each configuration ran once. Report ranges from 2 or 3 runs before drawing conclusions.
- The answer metric only checks keywords. It does not catch invented details (the 3B model cited a registry that
  the KB never mentioned) and it can score a false hit (the text of the automatic warning contains words that
  match one answer key).
- The test set was written from the documents, so the 100% retrieval on it is optimistic.
- The baseline already used the improved prompt. The comparison measures only the guardrail and the warning.

## Next steps (proposed, not implemented)

1. Replace the single threshold with two bands: below about 0.45 answer normally, between 0.45 and 0.50 answer
   with an extra caution instruction, above 0.50 refuse.
2. Validate any new threshold on a third batch of unseen questions before adopting it.
3. Improve retrieval for informal language (query rewriting or synonyms).
4. Improve metrics (an LLM judge or human review for a sample) and run each configuration several times.
5. Re-test the 8B model with the two-band rule before choosing a model on cost and benefit.

## How to reproduce the method with the sample KB

```bash
python rag.py ingest
python evaluate.py --label before            # MAX_DIST=0 and AUTO_WARNING=0
python evaluate.py --label after             # the configuration you want to test
python compare.py before after
```

Numbers on the sample KB will differ from the table above, because it is a different, smaller knowledge base.
