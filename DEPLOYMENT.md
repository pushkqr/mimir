# Deploying Mimir

A walkthrough for standing this up on hardware you control, from a bare machine to a
serving system, plus the failure modes that have actually happened rather than the ones
that seem likely.

`README.md` describes what Mimir is. `microservices/README.md` describes what each service
is and why it was chosen. This document is the operational path between them.

---

## Contents

1. [What you are deploying](#1-what-you-are-deploying)
2. [Prerequisites](#2-prerequisites)
3. [The short version](#3-the-short-version)
4. [The configuration model](#4-the-configuration-model)
5. [Walkthrough: one machine](#5-walkthrough-one-machine)
6. [Walkthrough: services on separate machines](#6-walkthrough-services-on-separate-machines)
7. [Hardware tiers and engine selection](#7-hardware-tiers-and-engine-selection)
8. [Verifying a deployment](#8-verifying-a-deployment)
9. [Loading a corpus](#9-loading-a-corpus)
10. [Switching between hosted and self-hosted generation](#10-switching-between-hosted-and-self-hosted-generation)
11. [Operational notes](#11-operational-notes)
12. [Troubleshooting](#12-troubleshooting)
13. [Tearing down](#13-tearing-down)

---

## 1. What you are deploying

Six components. Five are services with their own container; the sixth is the application.

| Component | Port | What it does | Needed |
|---|---|---|---|
| Weaviate | 8080, 50051 | Chunk storage, hybrid dense + BM25 search | always |
| Embeddings + reranker | 7997 | BGE-M3 vectors and cross-encoder reranking, one Infinity server for both | always |
| Translation | 8001 | IndicTrans2, Marathi and Hindi to English | always |
| Generation | 11500 | Answer synthesis | only in sovereign mode |
| Docling | 8002 | PDF OCR for scanned or table-heavy documents | optional |
| Application | 80, 443 | FastAPI, auth gate, admin console, the UI | always |

Generation is listed as conditional because it is the one component that can be a third
party. In `hybrid` mode it is Cerebras and the local generation service is not started. In
`sovereign` mode nothing leaves the network at query time. Everything else is self-hosted in
both modes.

Three deployment shapes are supported, and they are the same code path:

- **One machine.** Everything on localhost. Simplest, and what `deploy.py up` assumes.
- **Several machines.** One service per host, or any grouping. Each `microservices/*`
  directory is standalone and copyable.
- **The reference AWS topology.** `mimir-aws-stack.yaml` provisions one instance per service
  in a single VPC, with only the application holding a public address.

---

## 2. Prerequisites

| Requirement | Notes |
|---|---|
| Docker with Compose v2 | `docker compose version` must work, not `docker-compose` |
| Python 3.10+ | Only for `deploy.py`. The services themselves run in containers. |
| Disk | ~12GB images, ~5GB model weights, plus your corpus index |
| RAM | 16GB is a working minimum for the whole stack on one machine |
| `HF_TOKEN` | Required. Both IndicTrans2 checkpoints are gated on HuggingFace. |

`deploy.py` uses the standard library only, so it runs before you have installed anything
from `requirements.txt`. That is deliberate: the first thing you do on a new machine should
not require a working Python environment.

`python deploy.py check` reports all of this and prints the Docker install command for your
distribution if it is missing.

### The HuggingFace token is not optional

Both `ai4bharat/indictrans2-indic-en-dist-200M` and the 1B variant are gated. You need an
account that has accepted their terms, which is a manual step on the HuggingFace website and
cannot be scripted. Without it the translation service fails at model download with a 401 and
crash-loops. Accept the terms first, then generate a token.

---

## 3. The short version

On a machine with Docker and nothing else configured:

```bash
git clone https://github.com/pushkqr/mimir.git
cd mimir
python deploy.py up
```

`up` notices there are no `.env` files, generates them, and then starts every service in
order followed by the application. It will stop and tell you if a credential it cannot invent
is missing.

Everything below is that same sequence, explained.

---

## 4. The configuration model

Configuration is spread across one `.env` per service plus one for the application. Values
fall into three kinds, and only the third needs you.

**Derived.** Service URLs follow from which host the services run on and which port each
compose file publishes. There is nothing to decide, so `init` writes them.

Note that the published port is not always the port inside the container. Translation and
Docling both listen on 8000 internally and are published on 8001 and 8002. Writing these by
hand from the container port is a mistake that produces connection refused at query time.

**Generated.** Some values exist only to match on both sides. `WEAVIATE_API_KEY` in the
application's `.env` has to equal `WEAVIATE_API_KEY` in `microservices/weaviate/.env`; the
Infinity key has to appear in three places. Their content is irrelevant as long as they
agree. `init` generates each one once and writes it to every place it belongs, so they
cannot disagree.

This matters more than it sounds. When these drift, the service starts, passes its health
check, and rejects every real request. `core/config_check.py` exists to catch that class of
mistake after the fact; generating the values removes the opportunity to make it.

**Yours.** Credentials for third parties cannot be invented:

| Variable | Needed when |
|---|---|
| `HF_TOKEN` | always, for the gated IndicTrans2 checkpoints |
| `CEREBRAS_API_KEY` | `hybrid` mode only |
| `GOOGLE_CLOUD_PROJECT` and credentials | Gemini fallback, Document AI OCR, GCP translation |

`init` lists whichever of these are still unset rather than filling them with something
plausible.

### Re-running init is safe

`init` only replaces the placeholder values shipped in `.env.example`. A value you chose
survives. The one exception is a `localhost` URL when you pass `--host` naming a different
machine, because that is a leftover rather than a preference.

---

## 5. Walkthrough: one machine

### Step 1: check the machine

```bash
python deploy.py check
```

```
Prerequisites
-------------
  docker      : OK
  compose     : OK
  env files   : missing for weaviate, embeddings, translation, generation, application
                run: python deploy.py init

Hardware
--------
  CPUs        : 8
  RAM         : 31.3 GB
  GPU VRAM    : no GPU detected
  Disk free   : 88.1 GB (images ~12GB, models ~5GB, plus your corpus index)
```

This changes nothing. If Docker is missing it prints the install command for your
distribution instead of a link to a documentation page.

### Step 2: write the configuration

```bash
python deploy.py init
```

```
Shared secrets
--------------
  weaviate     generated, written to weaviate/.env and WEAVIATE_API_KEY
  infinity     generated, written to embeddings/.env and LOCAL_EMBED_API_KEY, LOCAL_RERANK_API_KEY
  admin token  generated

Derived service URLs
--------------------
  set   WEAVIATE_URL = http://localhost:8080
  set   LOCAL_EMBED_URL = http://localhost:7997/embeddings
  set   LOCAL_RERANK_URL = http://localhost:7997/rerank
  set   TRANSLATION_SERVICE_URL = http://localhost:8001/translate
  set   INGEST_TRANSLATION_SERVICE_URL = http://localhost:8001/translate
  set   LOCAL_GEN_MODEL = qwen3:1.7b  (from detected hardware)

Still needs you
---------------
  HF_TOKEN               IndicTrans2 checkpoints are gated on HuggingFace
```

### Step 3: fill in what only you can

Edit `.env` for whatever `init` listed as outstanding. At minimum `HF_TOKEN`.

Decide the mode while you are in there:

```bash
DEPLOYMENT_MODE=sovereign   # nothing leaves the network at query time
DEPLOYMENT_MODE=hybrid      # generation via Cerebras, everything else self-hosted
```

Also narrow the geofence. `MIMIR_ALLOWED_SUBNETS` ships as `0.0.0.0/0`, which accepts officer
logins from anywhere. `init` warns about this but deliberately does not change it, because
guessing a subnet can lock you out of the machine you are standing up.

```bash
MIMIR_ALLOWED_SUBNETS=10.0.0.0/8,192.168.0.0/16
```

### Step 4: bring it up

```bash
python deploy.py up
```

Services start in order, then the application builds and starts. Expect this to take a while
on first run: model weights are downloading.

Two of those waits look like failures and are not. Infinity runs a warmup benchmark at
startup that can take several minutes on a small CPU node, during which it is up but not
answering. And in sovereign mode the generation service pulls a model after the container is
already healthy.

### Step 5: verify

```bash
python deploy.py status   # is each component reachable
python deploy.py config   # is what they were told coherent
```

See [section 8](#8-verifying-a-deployment) for what each of these actually proves, because
they answer different questions and a stack can pass one while failing another.

---

## 6. Walkthrough: services on separate machines

Every directory under `microservices/` is standalone. It has no imports from this repository
and uses the standard library only, so it runs on a machine that has never seen the rest of
the code.

```bash
scp -r microservices/embeddings/ user@node:~/mimir-embeddings/
ssh user@node 'cd ~/mimir-embeddings && python3 deploy.py up'
```

The exception is `docling/`, whose build context reaches back into `../../docling_ingestion`.
Copy both, or build the image elsewhere and reference it.

### Two things to change that are easy to miss

**The compose files publish on `127.0.0.1` only.** That is correct for a single machine and
wrong for a distributed one: the application on another host cannot reach a loopback-bound
port. Change the binding in that service's `docker-compose.yml`:

```yaml
ports:
  - "7997:7997"          # instead of "127.0.0.1:7997:7997"
```

Do this only where the network itself is trusted, such as a private VPC with security groups.
Binding to all interfaces on a machine with a public address exposes the service.

**Point the application at the right hosts.** From the application machine:

```bash
python deploy.py init --host 10.0.1.20
```

That rewrites every service URL to that host. If the services are spread across several
different hosts rather than one, edit the URLs in `.env` directly; `--host` assumes they share
an address.

### The reference AWS topology

`mimir-aws-stack.yaml` provisions one instance per service in a single VPC. Security groups
admit traffic only from the application instance, and only the application holds an Elastic
IP.

```bash
aws cloudformation create-stack \
  --stack-name mimir-rag-stack \
  --template-body file://mimir-aws-stack.yaml \
  --parameters file://params.json \
  --region ap-south-1
```

Every secret is a `NoEcho` parameter with no default, so the template itself carries nothing
sensitive and is safe to keep in version control. `params.json` is the opposite: it holds the
admin and officer tokens, both service keys, the HuggingFace token and a GCP service account
key in plaintext. It is gitignored, along with `*.pem`, and must stay that way.

`AdminCidrIp` has no default on purpose. It opens a shell on every instance in the stack, so
CloudFormation refuses to deploy until you name a range rather than quietly accepting
`0.0.0.0/0`.

Reaching a private instance goes through the application host:

```bash
ssh -i mimir-key.pem \
    -o ProxyCommand="ssh -i mimir-key.pem -W %h:%p ubuntu@<APP_PUBLIC_IP>" \
    ubuntu@<PRIVATE_IP>
```

---

## 7. Hardware tiers and engine selection

Three services pick a model from detected hardware unless you set one explicitly. Ask any of
them what they would choose without starting anything:

```bash
python microservices/generation/deploy.py tier
```

```
engine     : ollama
model      : qwen3:1.7b
hardware   : CPU only, 4 logical cores, 15.4 GB RAM
WARNING    : Host has memory for a larger model but not the cores to run it at
             interactive speed; staying on the small tier.
```

### Generation picks an engine as well as a model

| Engine | Selected when | Models |
|---|---|---|
| SGLang | CUDA GPU, compute capability >= 7.5, Docker nvidia runtime present | Qwen3-1.7B through 32B by VRAM |
| Ollama + GPU | CUDA GPU older than 7.5 (K80, P100, V100) | `qwen3:1.7b` through `30b` by VRAM |
| Ollama | no usable GPU | `qwen3:1.7b`, or `4b` given 16+ cores |

All three bind `127.0.0.1:11500` on the host (not Ollama's own default of 11434, which
collides with any bare `ollama serve` a teammate runs on a shared machine) and speak
`/v1/chat/completions`, so `LOCAL_GEN_URL` is identical whichever is chosen. Changing engine
is not an application change.

Compute capability 7.5 is SGLang's floor, which excludes several cards still common in
university clusters. Those fall back to Ollama with the GPU attached rather than failing.

### The CPU tier needs cores, not just memory

A 16GB four-vCPU host holds `qwen3:4b` comfortably and generates at 3.0 tokens per second,
which is not usable for interactive answers. The binding constraint on CPU is arithmetic
throughput, not capacity, so the larger CPU tier requires real core count.

### Other services

| Service | Fixed | Adapts |
|---|---|---|
| Embeddings | **BAAI/bge-m3, always** | nothing |
| Reranker | | CPU: `bge-reranker-base`, GPU 6GB+: `bge-reranker-v2-m3` |
| Translation | | CPU: `indictrans2-dist-200M`, GPU 6GB+: `indictrans2-1B` |

**The embedding model is pinned and never selected by hardware.** Every vector in the corpus
is 1024-dimensional. Changing the model does not degrade quality, it makes the index
unreadable and forces a full re-ingest. This is enforced in `embeddings/deploy.py`, which has
no tier logic that could reach it.

The translation tier describes what detection picks on its own, not a ceiling. The 1B model
runs on CPU in `float16`, roughly 2GB of weights, and the reference deployment uses exactly
that for ingestion while keeping the 200M model for queries. Latency matters more for queries;
accuracy matters more for ingestion.

---

## 8. Verifying a deployment

Three commands answering three different questions. A stack can pass two and still be broken.

| Command | Asks | Catches |
|---|---|---|
| `check` | can the services start | missing Docker, missing `.env`, not enough disk |
| `status` | are they reachable right now | firewall rules, wrong host, a container that died |
| `config` | is what they were told coherent | values that disagree, values that never arrived |

`config` is the one worth running even when everything looks healthy:

```bash
python deploy.py config --env-file .env
```

It checks that required values are present, that values which must agree do agree, that no
key is defined twice, and that every key in the file actually reached the process. That last
one is a real failure this project hit: a variable present in `.env` never reached the
container because the compose file did not list it. The service had the value, the container
did not, and nothing reported it.

For an end-to-end assertion on answer content rather than reachability:

```bash
sudo docker compose exec -T mimir python -m scratch.regress
```

This runs four queries covering English retrieval, Marathi, contradiction detection across
superseding documents, and correct refusal, and asserts on what the answers say. `tests/` is
outdated and should not be treated as a gate.

---

## 9. Loading a corpus

Ingestion runs inside the application container:

```bash
sudo docker compose exec -d mimir python -m scratch.ingest_department
```

### Size the batch against your own corpus

`EMBED_BATCH_SIZE` and `LOCAL_EMBED_BATCH_TIMEOUT_S` are one coupled decision, not two.
Embedding throughput scales with total token count. Measured on a 2-vCPU node with BGE-M3:

| Work | Time |
|---|---|
| One ~500-token passage | 3.2s |
| A 24-passage batch of the same | 81s |

So a default of 64 passages against a 60s timeout cannot succeed for long passages: that batch
needs roughly 200 seconds. The failure is selective, which is what makes it hard to spot.
Documents with short passages sail through, and the ones that time out are the largest
documents, which are exactly the ones most worth indexing.

On CPU, prefer a smaller batch with a much longer timeout. 16 and 300s is a reasonable
starting point:

```bash
EMBED_BATCH_SIZE=16 LOCAL_EMBED_BATCH_TIMEOUT_S=300 LOCAL_EMBED_TIMEOUT_S=60
```

### Watch memory during long runs

One sustained run saw the embedding container climb from roughly 3.1GB to 5.8GB over seven
hours, with throughput collapsing alongside it and nothing logged as an error. The only symptom
was work getting slower. A restart cleared it immediately. The reference deployment runs a
memory-triggered guard on the embeddings host for this reason.

### Re-ingesting is idempotent

Chunk IDs are `uuid5` derived from the document, so re-ingesting a document updates its chunks
in place rather than writing a second copy. The namespace constant must never change.

If a document was ingested before that change, its chunks have random IDs and re-ingesting
leaves the old copies behind. Delete by `source_filename` first.

---

## 10. Switching between hosted and self-hosted generation

```bash
GEN_PROVIDER=cerebras   # hosted
GEN_PROVIDER=local      # self-hosted
```

`.env` is read when a container is **created**, not when it starts, so this needs a recreate:

```bash
sudo docker compose up -d
```

### What to expect from the self-hosted path

Retrieval, reranking, translation and the vector store are identical in both modes. Only the
final synthesis step differs, and on CPU it is dominated by prompt processing.

Measured on 4 vCPUs with a 1.7B model:

| Prompt tokens | Prefill |
|---|---|
| 1024 | 19.1s |
| 2048 | 42.1s |
| 3072 | 69.2s |
| 4096 | 100.3s |

The rate decays as the window grows, so context length is the dominant term and halving it
more than halves prefill. `SOVEREIGN_CONTEXT_BLOCK_CHARS` caps each evidence block for this
reason; it applies only when generation is local and does nothing to the hosted path.

Two settings interact and need setting together:

- `LOCAL_GEN_MAX_TOKENS` must cover **reasoning plus the answer**, not just the answer.
  Qwen3 emits its chain of thought into a separate field that the stream does not read, but it
  is still charged against the budget. Too small a budget produces an empty answer, which reads
  as a broken system rather than a slow one.
- `LOCAL_GEN_REASONING_EFFORT` trades speed against contradiction detection. With reasoning
  off the model answers fluently from a superseded document and never raises the discrepancy.
  With it on, it finds the conflict and costs decode time.

Neither `/no_think` nor `chat_template_kwargs` suppresses reasoning on Ollama. Only
`reasoning_effort` does.

---

## 11. Operational notes

Things that cost real time to discover.

**`.env` is read at container creation.** Editing it and restarting changes nothing. Recreate
with `docker compose up -d`.

**Application source is baked into the image, not bind-mounted.** A `git pull` on the host has
no effect until `docker compose up -d --build`. `docs/`, `scratch/`, `temp/` and the SQLite
database *are* bind-mounted and do write through.

**Restarting the application container kills a running ingestion**, because ingestion runs as
a process inside that container.

**Do not run ingestion during a latency test or a demonstration.** Query embedding and
reranking share the instance that ingestion saturates. One measurement saw a 0.5s query
embedding inflate to 119s while reranking timed out.

**Disable unattended upgrades on latency-sensitive hosts.** Ubuntu's `apt-daily` timers run
package upgrades on a schedule. One was measured taking 41% of a 2-vCPU host, which added
roughly 1.5s to every query while it ran.

```bash
sudo systemctl mask apt-daily.timer apt-daily-upgrade.timer
```

**`transformers` is pinned to 4.46.3.** Version 5 removed a symbol that `IndicTransToolkit`
imports at class-definition time, so the translation service crash-loops before serving
anything.

**Ollama's `/v1` endpoint discards the `options` block.** `num_ctx` and `num_thread` sent that
way are ignored, which silently truncates long prompts to the model's default window. Bake
them into a Modelfile instead:

```
FROM qwen3:1.7b
PARAMETER num_ctx 6144
PARAMETER num_thread 4
```

```bash
ollama create qwen3-mimir -f Modelfile
```

Then set `LOCAL_GEN_MODEL=qwen3-mimir`. `OLLAMA_NUM_THREADS` is not a real Ollama variable and
does nothing.

---

## 12. Troubleshooting

**A service starts, passes health checks, and rejects every request.**
A shared secret disagrees between the application and that service. Run
`python deploy.py config`. This is what `init` generating both sides at once is designed to
prevent.

**A variable is in `.env` but the application behaves as if it is unset.**
It never reached the container. `python deploy.py config --env-file .env` reports keys present
in the file but absent from the process. Usually the compose file does not pass it through.

**Marathi queries return answers to the wrong question.**
Query translation timed out and fell back to returning the original text, so a Devanagari query
searched an English corpus. The failure is silent by design. Raise `TRANSLATION_TIMEOUT_S`; the
service answers a short query in about 2s, so a tight ceiling leaves almost no margin.

`TRANSLATION_TIMEOUT_S` governs both the query path and ingestion, which sends much larger
chunks through the larger 1B model. Size it against the slower of the two, not the query.

**Translation crash-loops with a 401.**
`HF_TOKEN` is missing, or the account has not accepted the IndicTrans2 terms. Both checkpoints
are gated.

**Generation is far slower than the tier table implies, on a GPU machine.**
Docker has no nvidia runtime, so the container never received the device and is running the
model on CPU. `python microservices/generation/deploy.py check` reports this separately from
whether a GPU exists. Install `nvidia-container-toolkit`.

**The self-hosted path returns an empty answer.**
`LOCAL_GEN_MAX_TOKENS` is too small to hold reasoning plus the answer. Raise it, or set
`LOCAL_GEN_REASONING_EFFORT=none`.

**Retrieval finds nothing after changing the embedding model.**
It is not recoverable by configuration. Every stored vector is 1024-dimensional and a
different model makes the index unreadable. Restore `BAAI/bge-m3` or re-ingest the corpus.

**Ingestion appears to complete but documents are missing.**
Check whether the run was interrupted by a container restart. Ingestion runs inside the
application container.

**Infinity times out during `deploy.py up`.**
Its startup warmup benchmark can take several minutes on a small CPU node. The service is up
but not yet answering. This is not a failure.

---

## 13. Tearing down

```bash
python deploy.py down          # application, then every service in reverse order
python deploy.py down --only weaviate
```

Named volumes survive, so model weights and the corpus index are not lost. Remove them
deliberately:

```bash
docker volume ls | grep mimir
docker volume rm <name>
```

For the AWS reference topology, delete the CloudFormation stack and then confirm nothing was
left behind, since a stack delete can leave volumes or addresses if a resource was modified
outside the template:

```bash
aws cloudformation delete-stack --stack-name mimir-rag-stack --region ap-south-1
```
