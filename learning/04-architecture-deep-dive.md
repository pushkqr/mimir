# 04 — Architecture Deep Dive

**In one line:** Mimir is a production RAG engine for government policy documents, built on hybrid retrieval, self-hosted open-weight models, and a measured evaluation harness.

Every figure and code reference here is taken from the source. Where something is unmeasured, it says so.

---

## System at a Glance

| | |
|---|---|
| Corpus | ~104,000 documents, 790,053 indexed sections, 33 departments. The measured figures below were taken on an earlier 533-document corpus (10,194 chunks) |
| Accuracy | 88/100 pass rate, judge average 4.13/5, term average 0.653, **on that 533-document corpus** |
| Latency | 3 to 8 seconds end to end on hosted generation; ~16s observed in sovereign mode on a self-hosted 26B model |
| Infrastructure | One instance per service; the reference AWS topology runs 7 in a single VPC |
| Languages | English, Marathi, Hindi |

The accuracy figure is deliberately qualified. It demonstrates the harness works and reflects the retrieval design; it is not a current score for a corpus roughly two hundred times larger, and quoting it as one would be dishonest. Re-running the benchmark after a corpus change is the only way to know.

---

## 1. Hybrid Retrieval with Dynamic Alpha

Two retrieval paradigms fail in opposite directions.

**Dense vector search** matches meaning. It finds a document about "continuation of temporary posts" when the officer asked about "rules for extending temporary staff," which share no keywords. It is weak on exact identifiers, because a GR number is a character string, not a concept.

**BM25 keyword search** matches literally. Perfect for `MUWAD-2016/(38/16)/MASHI-1`. Useless when the officer's phrasing differs from the document's.

Weaviate runs both and fuses them, weighted by alpha. The key decision is that **alpha is not fixed** ([search.py:220-227](../retrieval/search.py#L220-L227)):

| Query shape | Alpha | Reasoning |
|---|---|---|
| Contains a GR-code pattern | 0.25 | Lean keyword: the identifier is an exact string |
| Everything else | 0.50 | Balanced fusion |

Detection is a regex for GR-code patterns against the standalone query. This single adjustment fixed an entire class of failures where dense similarity drowned out an exact identifier match.

### Deterministic Query Expansion

Rather than spend an LLM call expanding every query, fast mode uses `build_fast_search_query` ([search.py:21](../retrieval/search.py#L21)), which appends bilingual aliases when it detects known policy vocabulary. Triggers include GR references (adding `Government Resolution No`, `शासन निर्णय`, `क्रमांक`), appointments, professorial designations, probation, transfers, and dates.

It is deterministic and costs no tokens. The tradeoff is coverage: it helps only for vocabulary that was anticipated. Learned expansion would generalize better at the cost of latency.

### Known bug: alias dilution

The GR trigger fires on the *phrase* "government resolution" or "GR", not on an actual GR code. So a prose question about a specific named entity gets generic vocabulary appended:

```
"What department issued the Government Resolution for the Maharashtra State
 Loksahitya Samiti's term extension?"
  + "Government Resolution No GR number Government Decision शासन निर्णय क्रमांक"
```

Those added terms match essentially every document in a corpus of government resolutions, so they swamp the one distinctive token, `Loksahitya`. Measured directly against BM25:

| Query sent to BM25 | Correct document in top 5 |
|---|---|
| Raw user question | **Yes**, ranks 1 through 4 |
| After alias expansion | **No**, returns unrelated scholarship GRs |

The expansion demotes the correct document from rank 1 to off the list, and the pipeline then answers "no information found" for a document it holds and can trivially find.

The fix is to gate the alias block on a GR *code* pattern rather than the phrase, which is what the alpha-tuning regex already detects. It is deliberately not applied here: some queries that mention "Government Decision" in prose currently succeed via this path, so the change needs a full dual-grader benchmark run to confirm it is a net gain rather than a trade of one failure class for another.

---

## 2. Self-Hosted Embeddings and Reranking

The embedding model and the cross-encoder reranker share one node behind [Infinity](https://github.com/michaelfeil/infinity), which exposes an OpenAI-compatible API.

- **BGE-M3** (BAAI), multilingual, 1024-dimensional. Every ingestion and query embedding routes here.
- **BGE-reranker-v2-m3**, a cross-encoder that scores query and candidate jointly.

Self-hosting was not premature optimization. Vertex AI embedding quotas were actively throttling ingestion, which is what forced the move. The result is zero per-query embedding cost and no quota ceiling.

### Why Reranking Needs a Second Pass

The first search compares query and document *separately*, since both were embedded independently. A cross-encoder reads them *together*, which is more accurate and much slower. So retrieval is a funnel: `FAST_MODE_CANDIDATE_LIMIT=20` candidates from hybrid search, narrowed to `FAST_MODE_RERANK_LIMIT=12` after reranking.

**Measured cost.** Reranking accounts for roughly 3.4s of a 5.3s end-to-end response, about 60 to 65 percent of total latency. Because the funnel is narrow (20 in, 12 out), it discards only 8 candidates; most of what it buys is *ordering* the 12 that survive.

**Measured benefit, and a lesson about measuring it.** An A/B over 10 benchmark cases, scored by term overlap alone, showed *no* difference in accuracy and a 4s saving, which looked like a clear case for switching it off (`RERANK_ENABLED=false`).

That conclusion was wrong. Spot-checking real queries showed a large qualitative gap the metric could not see. On a Marathi query about temporary-post approvals:

- **With reranking:** the specific GR plus its exact validity dates.
- **Without:** a hedged list of three different GRs, none identified as the answer.

Both answers contain many of the same terms, so term overlap scored them identically. The difference is precision, which only the LLM judge half of the harness detects.

The lesson generalizes past this system: **a cheap proxy metric can report "no regression" for a change that noticeably degrades output.** The dual-grader design exists for exactly this, and using half of it produced a confidently wrong answer. Reranking stays enabled; the toggle remains for deployments that would rather have 2s responses than maximum precision.

### Batch Size Is a Timeout Budget

Ingestion embeds in batches of `EMBED_BATCH_SIZE` (default 64) against `LOCAL_EMBED_BATCH_TIMEOUT_S` (default 60s). At department scale that batching is what turns roughly 90,000 sequential round trips into about 1,400 calls.

Those two defaults are not independently safe. Measured on a CPU-only 2-vCPU embedding node, one ~500-token passage takes **3.2 seconds** to embed, and Infinity's own startup benchmark reports 0.19 embeddings/sec at 513 tokens against 25/sec at 2 tokens. Throughput is dominated by passage length, and it varies by two orders of magnitude across a real corpus.

So a 64-item batch of long passages needs roughly 200 seconds against a 60-second timeout. It cannot succeed. Short-passage documents complete comfortably, which is what makes this hard to see: the failure selects for the largest documents, exactly the ones most worth indexing. In one run, nine documents holding 1,594 chunks failed this way while a thousand smaller ones passed.

**The general lesson:** when a batch endpoint's cost scales with item size rather than item count, batch size and timeout are a single coupled decision. Sizing them separately produces a configuration that works on your test data and fails on your real data.

### A Memory Leak Looks Like a Slowdown

The same deployment showed a second, independent failure. Over seven hours of sustained ingestion, the embedding container grew from roughly 3.1 GB to 5.8 GB while available host memory fell from 4.3 GB to 2.1 GB. Throughput degraded with it, until even single-item requests exceeded a 10-second timeout. Restarting the container returned it to 1.9 GB and restored latency immediately.

Nothing crashed and nothing logged an error. The only visible symptom was work getting slower, and progress had already fallen from hundreds of documents per interval to roughly one. **A steadily rising memory floor under constant load is the signal; the timeouts are a lagging indicator.** Long ingestion runs against a self-hosted inference service want either periodic restarts or a memory ceiling that fails loudly instead of degrading quietly.

### The Latency Regression

An early version of `build_rerank_text` fed roughly 2,200 characters per candidate into the reranker, including full parent context. Search latency rose to 13 to 15 seconds, breaking the sub-10-second requirement.

The cause is structural: a cross-encoder scores `(query, document)` as a pair, so per-candidate length multiplies directly into total latency. The fix capped each candidate at roughly 900 characters, keeping document anchors (title, document number, section) and truncating the body to 600. Latency returned to 3 to 8 seconds with no measurable accuracy loss.

**The lesson worth carrying:** with cross-encoders, the input budget *is* a latency budget.

### Result Diversification

Hybrid search frequently returns many chunks from one document. `diversify_results` ([search.py:89](../retrieval/search.py#L89)) caps chunks per document at `FAST_MODE_MAX_CHUNKS_PER_DOC` (default 4), keyed on `source_filename` with fallback to `doc_number` then `document_title`, holding overflow back to backfill if the limit isn't reached.

This prevents evidence pile-on: five chunks from one document repeating a detail read to the model as strong corroboration, even when that document is the wrong one.

---

## 3. Multilingual Support

Officers query in Marathi, Hindi, and English. The corpus is indexed in English.

**At query time:** Devanagari input is detected and sent to a self-hosted IndicTrans2 service (AI4Bharat / IIT Madras) on its own node, returning English before retrieval. This one runs the distilled 200M model, where a query is short and latency is what the officer feels.

**At ingestion time:** a second instance of the same service runs the full 1B model in `float16`, which halves its weights to roughly 2GB so it fits on a CPU node. A document chunk is longer than a query and is translated once and stored, so accuracy on legal phrasing is worth more than speed. In `hybrid` mode this step can instead use GCP Cloud Translation v3.

Splitting one service into two instances with different models, sized to opposite priorities, costs nothing architecturally: they speak the same contract, and `INGEST_TRANSLATION_SERVICE_URL` decides which one ingestion talks to.

Translating at query time rather than searching Marathi directly is a deliberate choice about hybrid search. BM25 is purely lexical, so a Devanagari query against English text shares no tokens and that half of the search returns nothing. BGE-M3 is multilingual and would partly cope, but translating keeps **both** halves working in every language, against one index rather than parallel ones.

**Operational note:** this was the system's most fragile component. It deadlocked silently under RAM pressure on an undersized node, hanging on every Indic query with no error output. `docker stats` showing near-zero CPU during the hang is what proved deadlock rather than slowness.

---

## 4. Generation: Open-Weight Models with Failover

Two open-weight models alternate round-robin per request ([pipeline.py:59](../retrieval/pipeline.py#L59)):

```python
_CEREBRAS_MODELS = ["gpt-oss-120b", "gemma-4-31b"]
```

Both are served through Cerebras for inference speed, at temperature 0. On any Cerebras failure the request falls back to Gemini 2.5 Flash via Vertex AI ([pipeline.py:193-204](../retrieval/pipeline.py#L193-L204)).

**The architecturally significant point:** Cerebras is an inference host, not a model vendor. `gpt-oss-120b` and `gemma-4-31b` are openly published, as are BGE-M3, BGE-reranker-v2-m3, and IndicTrans2. **No proprietary model is load-bearing.** Moving generation on-premise means serving those same weights locally and repointing configuration, not re-architecting. Gemini remains only as a fallback path and for ingestion-time batch translation.

---

## 5. Ingestion

### Three-Tier Parser Fallback

Government PDFs arrive in inconsistent condition, so extraction degrades through three tiers:

1. **PyMuPDF4LLM** — fast, local, handles well-formed PDFs
2. **Google Document AI OCR** — scanned or image-only circulars
3. **Gemini Vision** — last resort for what defeats both

Orgpedia GRs arrive as pre-translated `.en.txt` plaintext and skip parsing entirely. That difference matters when assessing corpus difficulty: a corpus of only `.en.txt` files never exercises tiers 2 or 3.

### Table-Aware Chunking

Naive fixed-size chunking splits tables mid-body, orphaning rows from their headers. A row reading `| Kolhapur | 30 |` is meaningless once separated from the header saying what 30 counts.

`chunk_and_embed_circular` ([chunking.py:65](../ingestion/chunking.py#L65)) detects table boundaries and prepends the nearest preceding header rows to isolated row chunks before embedding.

### Parent-Child Hierarchy

Child chunks are the embedded, searchable unit. Parent sections supply surrounding context at generation time. This keeps the search index tight while giving the model enough context to interpret what it retrieved.

### Idempotence

Ingestion tracks processed files by hash in `scratch/ingestion_state.json`. Re-running is safe and never duplicates. This is what makes scaling the corpus a scheduling problem rather than an engineering one.

---

## 6. Security

### Zero-Trust Network Gating

Middleware validates the client IP against an allowlist **before** authentication is checked ([app.py:74-92](../app.py#L74-L92)). The list is environment-configured via `MIMIR_ALLOWED_SUBNETS`, defaulting to loopback and RFC1918 private ranges. Requests from outside are refused with 403.

Deploying inside a department means setting that variable to the department's range. Public exposure requires an explicit configuration change, so the secure posture is the default.

### Token Identity

No passwords. Officers hold generated tokens; only SHA-256 hashes are stored ([db.py](../db.py)). Comparison uses `hmac.compare_digest` to avoid timing leaks. An admin API and console handle provisioning, renaming, and revocation.

### The Access Log

Token identity is only half of accountability; the other half is knowing what was done with it. `record_audit` appends an entry for logins, denied attempts, token issuance and revocation, uploads, promotions, and feedback, each carrying actor, client address, and timestamp. The admin console reads it directly.

Two details matter more than the feature itself. Auditing never raises: a failure to record must not break the request that triggered it. And the actor is stored as a truncated hash of the token rather than the token, so the log identifies who acted without becoming a second place the credential lives.

### Upload Quarantine

An administrator uploading a document does not write to the live corpus. Uploads land in a separate quarantine collection and enter the corpus only on explicit promotion. This is the answer to "what stops someone adding a fabricated circular": nothing enters retrieval without a second, recorded decision.

### Known Gaps

Stated plainly, because pretending otherwise is worse:

- **No per-document authorization.** Every authenticated officer sees the whole corpus. Acceptable while the corpus is published material; a prerequisite before anything confidential is indexed.
- **The officer token is bearer-only.** Anyone holding it is that officer. There is no second factor and no binding to a device.
- **Quarantine checks provenance, not content.** Promotion records who promoted what. It does not verify that the document is authentic.

Two gaps listed in earlier revisions have since been closed. Admin routes no longer bypass the subnet gate: the network check now runs for `/api/admin/*` as well, since exempting them left a hole in a posture that claimed to be zero-trust. And audit logging is no longer informal, as described above.

---

## 7. Evaluation

Knowing whether a change helped is the hard part of RAG. Mimir has a harness rather than intuition.

**Dataset.** Candidate questions were generated document by document, then hand-filtered to 100 verified cases after removing ambiguous phrasing and near-duplicate document collisions. Coverage spans simple English, Marathi, Hindi, multi-document synthesis, direct GR lookups, and deliberately out-of-corpus questions.

**Dual grading.** A deterministic scorer checks that required terms (GR numbers, dates, counts) actually appear. An LLM judge scores factual accuracy 0 to 5 against a human-verified expected answer. Term matching alone is too rigid; a judge alone is too lenient on missing specifics.

**Pass rule.** `judge >= 3.0`, or `judge >= 2.0 AND term >= 0.5`.

**Result: 88/100.** Judge average 4.13, term average 0.653, up from an 83/100 baseline. The improvement came from two fixes, not from touching the test set:

1. Resolving the translation deadlock, which had been silently failing every Indic query.
2. Wiring three retrieval helpers that existed but were never called: `build_fast_search_query`, `build_rerank_text`, and `diversify_results`.

**The remaining 12%** share one pattern: the correct document is retrieved, but the model states a date or GR number belonging to a near-identical order about a different person or case. This is generation-side entity attribution, not retrieval failure. Naming the failure mode precisely is what makes it addressable.

An attempted prompt-level fix scored 87/100, worse than baseline, and was reverted. Recorded because the negative result is part of the finding.

---

## 8. Deployment

Every service is its own directory under `microservices/`, each carrying a `deploy.py` that uses only the standard library. That constraint is the point: a directory can be copied to a machine that has never seen this repository and still deploy itself. The root `deploy.py` orchestrates the full stack but never reimplements any service's startup logic, shelling out to those same scripts instead. Two entry points, one code path. If the orchestrator carried its own copy, the two would drift, and the remote-node path is the one that breaks silently, precisely when a department has just handed over a server.

The reference AWS deployment runs one service per instance in a single VPC:

| Instance | Runs |
|---|---|
| Application | FastAPI, Caddy. The only instance with a public address |
| Vector store | Weaviate |
| Inference | BGE-M3 and the cross-encoder reranker via Infinity |
| Generation | Ollama, model tier chosen from detected hardware |
| Translation (query) | IndicTrans2 200M, sized for latency |
| Translation (ingest) | IndicTrans2 1B in float16, sized for accuracy |
| Document parsing | Docling OCR |

Security groups admit traffic to each service from the application's group alone, so the six service instances have no route from the internet. Moving to an intranet deployment means removing the Elastic IP and the public ingress rule, which changes no application code.

### Deployment Modes

`DEPLOYMENT_MODE` sets defaults for four independent switches: generation, tier-2 PDF parsing, ingest structuring, and ingest translation. `sovereign` points all four at self-hosted services; `hybrid` allows third-party inference. Each remains individually overridable, and the admin console reports the configuration actually resolved at runtime rather than the one intended.

One finding from building this is worth recording. `.env` carried `GEN_PROVIDER=cerebras` as a literal line. Because several entry points call `load_dotenv()` independently, that line would silently reassert itself and defeat sovereign mode, with no error and no symptom beyond queries quietly leaving the network. The fix was to comment out the four hybrid defaults, since the code already falls back to the same values when they are absent. **A configuration file that restates a default is a trap when more than one process reads it.**

### Secrets and Docker

A production incident worth recording. `gcp-key.json` was being copied into the image by `COPY . .`. When the file did not exist at build time, Docker created an empty **directory** at that path and baked it into the layer. Once the real key appeared on the host, bind-mounting a file onto a path the image believed was a directory failed with a type mismatch, and every embedding call died with `Is a directory`.

Two things fixed it: adding `gcp-key.json` to `.dockerignore`, and rebuilding with `--no-cache` to drop the poisoned layer. Secrets now arrive only as runtime bind mounts, which is also the correct security posture.

The same failure mode recurred on first AWS deployment, from the opposite direction. `mimir_portal.db` is bind-mounted but matched by a `*.db` gitignore rule, so a fresh clone never contains it; Docker then created a directory at the host path, which could not mount onto the file the image did contain. **A bind mount whose host path does not exist yet is a directory waiting to happen.** The fix is to create the file before the container starts.

---

## Design Tradeoffs

| Decision | Bought | Cost |
|---|---|---|
| Hybrid over dense-only | Exact identifier lookups work | Extra latency per query |
| Dynamic alpha | Fixed a failure class | Regex heuristic, not learned |
| Self-hosted embeddings | No quota ceiling, no per-query cost | A node to operate, and its throughput becomes yours to size |
| Cross-encoder reranking | Better ordering into the model | Input length is a latency budget |
| Deterministic expansion | No token cost, no added latency | Covers only anticipated vocabulary |
| Two generation models | No single point of failure | Two integrations to maintain |
| Open-weight throughout | On-prem is configuration, not rewrite | Rules out frontier-only capabilities |
| Translate-then-retrieve | Both halves of hybrid search work | Translation is a hard dependency |

---

## Known Limitations

- **Chat history is unbounded.** It concatenates without a window and will eventually exceed the context limit.
- **Entity attribution on near-duplicate documents**, the dominant remaining failure mode.
- **Ingestion is manually triggered.** The pipeline is idempotent and ready for scheduling; the scheduler is not built.
- **No per-document access control**, as noted above.
- **A document that loses chunks is still recorded as complete.** If a passage fails to embed it is dropped rather than written with a bad vector, which is the right call for index integrity. But the file is still marked done in the state file, so a resume will not revisit it and the gap is invisible. Recovering means clearing those entries and deleting the document's chunks before re-ingesting, because objects are inserted with a random UUID rather than a deterministic one, so a re-run duplicates whatever succeeded the first time.

An earlier revision listed "retrieval does not see conversation history" here. That is now handled: `contextualize_query` rewrites a follow-up against recent history before the search step, so "what is the number of this GR" carries its antecedent into retrieval. It costs one extra inference call against a rate-limited budget, which `CONTEXTUALIZE_FOLLOWUPS=false` trades back.
