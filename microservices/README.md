# Mimir, on-premise deployment

Every component Mimir depends on, packaged to run inside a department's own network with no
outbound internet access at runtime.

The design goal was that going on-premise is a configuration change, not a rewrite. Retrieval
already talks to its embedding, reranking and translation services over HTTP, so those move
by changing a base URL. Generation was the one component bound to a third-party API; it now
sits behind the same kind of interface.

## What runs where

| Component | Directory | Image | Self-hosted | Purpose |
|---|---|---|---|---|
| Vector store | `weaviate/` | `semitechnologies/weaviate` | yes | Chunk storage, hybrid dense + BM25 search |
| Embeddings | `embeddings/` | `michaelf34/infinity` | yes | BGE-M3, 1024-dimensional vectors (pinned) |
| Reranker | `embeddings/` | `michaelf34/infinity` | yes | BGE cross-encoder, hardware-tiered |
| Translation | `translation/` | built from this directory | yes | IndicTrans2, Marathi to English, hardware-tiered |
| Generation | `generation/` | `ollama/ollama` | yes | Answer synthesis, hardware-tiered |
| Document parsing | `docling/` | built from `../../docling_ingestion` | yes | Optional, for scanned or table-heavy PDFs |

No component in this stack calls out to a third party once the images and model weights are
present. Weights are pulled once at install time and cached in named volumes.

## Two ways to run this

**Each service directory is independently deployable.** `scp -r microservices/weaviate/
user@node:` and it runs there with no dependency on the rest of this repository (except
`docling/`, whose build context reaches back into `../../docling_ingestion` — everything
else is fully self-contained). This is the point: if a department offers a machine on their
own network, one service — or the whole stack — can go straight onto it.

**`deploy.py` at the repository root** orchestrates all of them from a full checkout. It does
not reimplement any service's startup logic; it shells out to the exact same `deploy.py` each
service directory carries on its own, so the orchestrated path and the "just this one service
on a strange machine" path can never silently diverge.

```bash
python deploy.py init      # write every .env: derived URLs, generated secrets
python deploy.py check     # prerequisites, hardware report, and per-service readiness
python deploy.py up        # bring every service up, in order, then build and start the app
python deploy.py status    # live reachability of all six components, same probes /admin uses
python deploy.py config    # required values, values that must agree, what the services report
python deploy.py down      # tear everything down, reverse order
```

`up` runs `init` when `.env` files are missing, so a clean checkout deploys in one command.
See [`../DEPLOYMENT.md`](../DEPLOYMENT.md) for the full walkthrough.

Or one service directly, which is exactly what `deploy.py up` calls internally:

```bash
cd microservices/weaviate
cp .env.example .env   # or let the root `deploy.py init` write it, with a generated key
python deploy.py up
```

Then point the application's `.env` at wherever each service landed. `GEN_PROVIDER=local` is
the switch that moves generation off Cerebras; see the root `.env.example`'s commented block
for the full set of variables.

`docling/` is optional and not brought up by `deploy.py up` without asking: `python deploy.py
up --only docling`.

## Hardware-adaptive model tiers

Three of five services pick a model tier from detected hardware (RAM, GPU VRAM via
`nvidia-smi`) unless the operator sets one explicitly in `.env`. Run `python deploy.py tier`
inside any of `embeddings/`, `translation/` or `generation/` to see what it would pick without
starting anything.

| Service | Fixed | Adapts |
|---|---|---|
| Embeddings | **BAAI/bge-m3, always** | — |
| Reranker (in `embeddings/`) | | CPU: `bge-reranker-base` &middot; GPU 6GB+: `bge-reranker-v2-m3` |
| Translation | | CPU: `indictrans2-dist-200M` &middot; GPU 6GB+: `indictrans2-1B` |
| Generation | | picks an engine as well as a model, see below |

Generation selects the serving engine from the hardware, because the right engine on a GPU is
not the right engine on a CPU:

| Engine | Selected when | Models |
|---|---|---|
| SGLang | CUDA GPU, compute capability >= 7.5, Docker nvidia runtime present | `Qwen/Qwen3-1.7B` through `32B` by VRAM |
| Ollama + GPU | CUDA GPU older than 7.5 (K80, P100, V100) | `qwen3:1.7b` through `30b` by VRAM |
| Ollama | no usable GPU | `qwen3:1.7b`, or `4b` given 16+ cores |

All three bind `127.0.0.1:11500` on the host (not Ollama's own default of 11434, which
collides with any bare `ollama serve` a teammate runs on a shared machine) and speak
`/v1/chat/completions`, so `LOCAL_GEN_URL` does not change with the engine and switching is
not an application change. Compute capability 7.5 is
SGLang's floor, which excludes several cards still common in university clusters; those get
Ollama with the device attached rather than a failure.

Whether a GPU exists and whether Docker can hand it to a container are checked separately,
because they fail differently. A host can pass `nvidia-smi` and still have no nvidia container
runtime, in which case the container silently receives no device and runs a large tiered model
on the CPU, which is slower than the CPU tier it declined. That case falls back with a warning
rather than starting.

The CPU tier requires core count and not merely enough RAM to hold the weights. A 16GB
four-vCPU host runs `qwen3:4b` at 3.0 tokens per second, which is not usable interactively;
the binding constraint on CPU is arithmetic throughput, not capacity.

Where RAM is still consulted the threshold is 15GB rather than 16 on purpose. A nominally-16GB
cloud instance reports slightly less than that to the operating system once the hypervisor and
kernel have taken their share, so a strict `>= 16` test silently drops such a machine to the
smallest tier. That failure is invisible: the service starts, serves, and answers with a
weaker model than intended.

The translation tier above describes what hardware detection picks on its own. It is not the
ceiling: the 1B model runs on a CPU node in `float16`, which halves its weights to roughly
2GB, and the reference deployment runs exactly that for ingestion. Set `INDICTRANS_MODEL`,
`TRANSLATE_TORCH_DTYPE=float16` and a `TRANSLATE_MEM_LIMIT` large enough to hold it. Query
translation stays on the 200M model, where latency matters more than accuracy; ingestion uses
the 1B, where the reverse is true.

Both IndicTrans2 checkpoints are **gated on HuggingFace**. The service needs `HF_TOKEN` set to
an account that has been granted access, or it fails at model download.

**The embedding model is pinned and never selected by hardware.** Every vector already in the
corpus is 1024-dimensional. Swapping the model does not degrade quality — it makes the entire
index unreadable and forces a full re-ingest. This is enforced in code, not just documented:
`embeddings/deploy.py` hardcodes `BAAI/bge-m3` and has no tier logic that could touch it.

## Hardware

The stack runs on CPU as written, which is enough to demonstrate the full pipeline. It is not
enough to be pleasant.

- **CPU only, 16GB RAM.** Works. Reranking dominates query latency, and generation is bounded
  by prompt processing rather than by the model: measured on 4 vCPUs with a 1.7B model, 1024
  prompt tokens take 19s and 4096 take 100s, the rate decaying as the window grows. Suitable
  for evaluation, and honest about what a department without a GPU would experience.
- **One NVIDIA GPU with 16GB or more.** The intended configuration. `generation/` attaches the
  device itself now, choosing SGLang or Ollama from the card's compute capability; there is no
  longer a block to uncomment. For `embeddings/`, uncomment the `deploy` block in its compose
  file. Reranking drops by roughly an order of magnitude.

Disk: about 12GB for images, plus roughly 5GB of model weights, plus the corpus index.

## Verification status

Being explicit about what has and has not been run, because a deployment artifact that
quietly does not work is worse than none.

**Verified end to end.** The application's local generation path: retrieval, reranking,
citation and the conflict-detection logic were all exercised against a self-hosted
OpenAI-compatible server, with the correct model name reported through to the UI, streaming
intact, and a verified fallback when the server is unreachable.

**Verified without Docker actually running.** This machine has Docker installed but not
running, which turned out to exercise exactly the failure paths that matter most: every
`deploy.py check`/`up`/`down`/`status` across all five services and the root orchestrator was
run against that state, and each one fails with a specific, actionable message rather than a
raw traceback or a hang. `docker compose config` validates all five compose files (env-var
interpolation, YAML structure) without needing the daemon at all. The hardware-tier logic —
RAM via `ctypes`/`os.sysconf`, GPU VRAM via `nvidia-smi` — was verified against this machine's
real hardware (13.9GB RAM, no GPU, correctly resolves to the smallest CPU tier everywhere) and
against all four branches of the generation tier table with mocked values, including the
boundary conditions. `python deploy.py status` (and `/api/admin/topology`, which now shares
the same `core/health.py` probes) was verified against the live remote services this
deployment currently uses in hybrid mode — all six reachable, correct latencies, correct
self-hosted/third-party split.

**Now run with real containers, on seven cloud instances.** The whole stack has since been
deployed one service per machine, reached only over private addressing. Every service builds,
starts and serves. Both flagged risks are resolved, and one of them was real:

- **Infinity's CLI flags** were correct as written. No change needed.
- **`IndicTransToolkit`** failed exactly as predicted, and the cause is worth recording.
  `transformers` was unpinned, so a fresh build pulled 5.x, which removed
  `transformers.tokenization_utils.PreTrainedTokenizerBase`. The toolkit's collator imports
  that symbol at class-definition time, so the service crash-looped on startup with an
  `ImportError` before serving a single request. Pinned to `transformers==4.46.3`, verified by
  importing the toolkit in a throwaway container before trusting it.

Three further findings from that first real deployment, none of them design problems:

- **Both IndicTrans2 checkpoints are gated on HuggingFace.** They now require an account that
  has been granted access, so the service needs `HF_TOKEN` set or it fails at model download
  with a 401. Accepting a gated model's terms is a manual step that cannot be scripted.
- **Infinity's health check needs a generous timeout on CPU.** Startup runs a warmup benchmark
  that can take several minutes on a 2-vCPU node, during which the service is up but not yet
  answering. `deploy.py up` reporting a timeout there does not mean the service failed.
- **Batch size and timeout are one coupled decision, not two.** See the note below.

## Embedding throughput on CPU

Worth knowing before sizing an ingestion run. Measured on a 2-vCPU node with BGE-M3:

| Work | Time |
|---|---|
| One ~500-token passage | 3.2s |
| A 24-passage batch of the same | 81s |

Throughput scales with total token count, and Infinity's own startup benchmark spans two
orders of magnitude across passage lengths (25 embeddings/sec at 2 tokens, 0.19/sec at 513).
So the application's default `EMBED_BATCH_SIZE=64` against `LOCAL_EMBED_BATCH_TIMEOUT_S=60`
cannot succeed for long passages: that batch needs roughly 200 seconds.

The failure is selective, which is what makes it hard to spot. Documents with short passages
sail through; the ones that time out are the largest documents, exactly the ones most worth
indexing. On CPU, prefer a smaller batch with a much longer timeout (16 and 300s is a
reasonable starting point) and size both against your own corpus rather than the defaults.

A sustained run also wants watching for memory growth in the embedding container. One run saw
it climb from roughly 3.1GB to 5.8GB over seven hours with throughput collapsing alongside it,
and nothing logged an error: the only symptom was work getting slower. A restart cleared it
immediately.

## Why generation was the last piece

Embeddings, reranking and translation were self-hosted from early on, because those models
are small enough to run economically on commodity hardware and because sending every query's
text to a third-party embedding API was not acceptable for government documents.

Generation stayed on a hosted API for one reason: speed. Cerebras returns a full answer in
about two seconds, which is what keeps end-to-end query time under six. A 4B model on a CPU
node will not match that, and a 30B model on one GPU will not either.

So the tradeoff is explicit rather than hidden. If a deployment requires that no query text
leaves the building, `GEN_PROVIDER=local` satisfies that today and costs response time. If it
does not, the hosted path is faster. Both use identical retrieval, identical grounding, and
identical citations, because generation is the last step and it only ever sees text that
retrieval already selected.
