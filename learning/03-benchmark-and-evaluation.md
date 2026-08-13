# 03 — Benchmark & Evaluation: Proving the Truth

**In one line:** We don't guess if the system is accurate — we run an automated harness across a curated set of policy questions spanning multiple query types and languages, and grade the pipeline on both hard facts and semantic accuracy.

---

## The Problem with "Vibes-Based" Testing

Building a RAG system is straightforward. Knowing if a change to the embedding model, alpha weight, or chunk size actually *improved* things is hard. Asking a few questions and feeling good about the answers doesn't catch regressions.

Mimir includes a structured benchmark harness (`benchmark/benchmark.json`) — a curated dataset of test cases built directly from the actual policy documents in the corpus.

---

## Benchmark Query Categories

The dataset deliberately covers a full range of query types an officer might actually send:

| Category | Description | Example |
|---|---|---|
| `simple_english` | Direct factual questions in English | "How many temporary posts were extended in the Libraries directorate?" |
| `marathi_query` | Same questions phrased naturally in Marathi | "ग्रंथालय संचालनालयात किती तात्पुरती पदे होती?" |
| `hindi_query` | Colloquial Hindi queries | "Library directorate mein kitne temporary posts the?" |
| `complex_english` | Multi-fact synthesis requiring reading across clauses | "What are the phased staffing numbers for the Kolhapur college over 4 years?" |
| `complex_marathi` | Complex queries in Marathi | "कोविड-19 मुळे 2020-21 च्या नियमावलीत काय बदल झाले?" |
| `gr_number_lookup` | Explicit GR number cited in query | "Tell me about GR No. MUWAD-2016/(38/16)/MASHI-1" |
| `not_found` | Topics deliberately absent from the corpus | "What is the pension amount for retired professors?" |

Each case includes:
- **`query`** — the question as an officer would phrase it
- **`expected_answer`** — human-verified ground truth
- **`expected_terms`** — specific terms (GR numbers, counts, dates) that *must* appear in the answer
- **`category`** — query type for disaggregated scoring
- **`source_doc`** — the exact document the answer should come from (`null` for not-found cases)

---

## The Dual-Evaluator System

When you run `python main.py` with `RUN_BENCHMARK = True`, the system simulates all queries through the full pipeline and grades each answer with two independent evaluators:

### 1. Term-Match Scorer (Deterministic)

Checks whether each `expected_term` appears in the generated answer. If the answer says "142 posts" but `expected_terms` includes `"142"` and `"libraries"`, both must be present. Missing critical terms (statute numbers, dates, counts) is a hard failure.

This catches the most common RAG failure: the LLM writing a fluent, confident-sounding answer that omits the specific fact the officer needed.

### 2. LLM Judge (Semantic)

Term-matching is rigid — the system can state the correct fact with slightly different phrasing and still fail. The Judge LLM receives the expected answer and the pipeline's answer, then grades from 0–5 on:
- Factual accuracy vs. ground truth
- Absence of hallucination
- Completeness

### Pass/Fail Rule

A case counts as a **pass** if `judge_score >= 3.0`, OR if `judge_score >= 2.0` AND `term_score >= 0.5`. This lets a well-reasoned, mostly-correct answer pass on judge score alone, while catching cases where the judge is lenient on phrasing but the answer is actually missing the hard facts (GR numbers, dates, counts) the term scorer checks for.

The overall run is graded A–D from `average_judge_score` and `average_term_score` (see `print_benchmark_report` in `benchmark/runner.py`), separate from the per-case pass/fail rate.

### Results (100-case benchmark, 533-document corpus)

> **These figures were measured against a 533-document corpus and are not a current score.** The deployed corpus is now roughly **104,000 documents and 790,000 indexed sections across 33 departments** — around two hundred times larger, not the order of magnitude this note previously claimed. Scale changes retrieval difficulty in both directions: more documents that could answer a question, and far more near-duplicates to confuse it. Note that the dominant failure mode below is *exactly* the one a larger corpus makes worse. The numbers record what the two fixes bought on the corpus of the day. They do not transfer, and re-running is the only way to know.

**88/100 cases passing** (`average_judge_score = 4.13`, `average_term_score = 0.653`), up from an 83/100 baseline. The improvement came from two fixes, not from touching the benchmark or ground-truth data:
- Root-caused and fixed a deadlocked translation microservice (undersized RAM on that node caused silent hangs on Marathi/Hindi queries).
- Wired in previously-dead retrieval helpers in `retrieval/search.py` — deterministic BM25 alias/keyword expansion (`build_fast_search_query`), compact document-anchored reranker input (`build_rerank_text`), and per-document result diversification (`diversify_results`) — none of which were actually being called before.

The remaining ~12% of failures share one dominant pattern: the correct document is retrieved, but the model states a wrong specific fact (a date or GR number) belonging to a near-duplicate document about a different person or case. This is a generation-side entity-attribution problem, not a retrieval failure — worth targeted prompt work in a future pass, but out of scope for the current fix cycle.

---

## Generating a Full-Corpus Benchmark Dataset

A document-level LLM generator builds the evaluation set:

```bash
python scratch/generate_benchmark_full.py \
  --out benchmark/benchmark_100.json \
  --max-doc-chars 15000 \
  --resume
```

**Key design decisions in the generator:**

- **Document-level, not chunk-level.** The full document text is fed to Gemini, not random chunks. This avoids the "retrieval-trivial" problem where a question is a paraphrase of a chunk and BM25 always finds it — inflating scores without testing real retrieval.
- **Explicit diversity prompt.** Gemini is instructed to produce exactly: 1 simple English, 1 Marathi, 1 complex, and 1 GR number lookup question — all from an officer persona, not an academic one.
- **Size filter.** Documents larger than `--max-doc-chars` characters are skipped rather than truncated mid-annexure. Large acts with boilerplate appendices would generate misleading questions from partial context.
- **Incremental.** `--resume` continues from a partial run without re-querying already-processed documents.

---

## What "Not Found" Cases Test

The `not_found` category is specifically designed to catch hallucination. The expected answer is always a clean refusal:

> *"The retrieved documents do not contain information about pension amounts for retired professors."*

A system that hallucinates a pension figure is worse than one that admits it doesn't know. The LLM Judge grades hallucinated answers as 0/5 regardless of how plausible they sound.

---

## Running the Benchmark

```bash
# Via main.py flags
RUN_BENCHMARK = True
python main.py

# Or standalone
python benchmark/runner.py
```

Results are written to `benchmark/benchmark_results.json`. Each result includes the query, pipeline answer, term scores, judge score, and a pass/fail verdict.

---

*For the raw datasets:*
- **[Benchmark Dataset](../benchmark/benchmark.json)** — current ground-truth cases
- **[Benchmark Results](../benchmark/benchmark_results.json)** — latest run output
