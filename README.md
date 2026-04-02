<p align="center">
  <img src="assets/architecture.svg" alt="Kombinat Architecture" width="680"/>
</p>

<h1 align="center">kombinat</h1>

<p align="center">
  <strong>Distributed annotation coordination server for <a href="https://embedcollective.com">Embed Collective</a></strong>
</p>

<p align="center">
  <a href="https://github.com/embedcollective/kombinat/actions"><img src="https://img.shields.io/github/actions/workflow/status/embedcollective/kombinat/ci.yml?branch=main&style=flat-square" alt="CI"></a>
  <a href="https://github.com/embedcollective/kombinat"><img src="https://img.shields.io/badge/python-3.12+-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.12+"></a>
  <a href="https://github.com/embedcollective/kombinat/blob/main/LICENSE"><img src="https://img.shields.io/github/license/embedcollective/kombinat?style=flat-square" alt="License"></a>
  <a href="https://embedcollective.com"><img src="https://img.shields.io/badge/Embed_Collective-landing_page-black?style=flat-square" alt="Website"></a>
</p>

---

Kombinat is the backend that powers [Embed Collective](https://embedcollective.com) — an open, community-driven effort to build high-quality embedding models through distributed human+LLM annotation.

It coordinates batches of query-document pairs across anonymous contributors running the [**annotator**](https://github.com/embedcollective/annotator) CLI on their own hardware, validates results with honeypot quality checks, and aggregates labels at scale.

## How it works

1. The **ingest pipeline** loads source datasets, mines hard negatives via BM25 + dense retrieval + RRF fusion, and writes candidate pairs to PostgreSQL.
2. Contributors run the [annotator](https://github.com/embedcollective/annotator) CLI, which claims a batch of unlabeled pairs, scores them locally with a quantized LLM (Qwen 3B-7B), and streams labels back.
3. Kombinat **validates** annotations against embedded honeypots (~5% of each batch), updates contributor reputation, and promotes pairs to `verified` or `rejected` via majority vote.

## Quickstart

```bash
# clone
git clone https://github.com/embedcollective/kombinat.git
cd kombinat

# install
pip install -e ".[dev]"

# start postgres + server
docker compose up -d
uvicorn kombinat.main:app --reload
```

## Ingest pipeline

Mine hard-negative pairs from a HuggingFace dataset split and load them into the database:

```bash
pip install -e ".[ingest]"
python -m kombinat.tools.ingest --split squad --device cpu
```

## API

| Endpoint | Description |
|---|---|
| `POST /v1/auth/github` | GitHub OAuth token exchange |
| `GET  /v1/contributors/me` | Current contributor profile |
| `POST /v1/batches/claim` | Claim a batch of unlabeled pairs |
| `DELETE /v1/batches/{id}` | Release an unfinished batch |
| `POST /v1/annotations` | Submit annotations for a batch |
| `GET  /v1/stats` | Public progress stats |

---

## Annotation ledger

Live count of labeled query-document pairs across all datasets.

| Metric | Count |
|---|---|
| Total pairs ingested | — |
| Total annotations | — |
| Verified pairs | — |
| Rejected pairs | — |
| Active contributors | — |

> Ledger will be updated as annotation campaigns progress. See live stats at `GET /v1/stats`.

## Datasets

Source data comes from [nomic-ai/nomic-embed-unsupervised-data](https://huggingface.co/datasets/nomic-ai/nomic-embed-unsupervised-data) (239M rows, 29 splits).

### Active

| Dataset split | Status |
|---|---|
| `squad` | Ingesting |

### Planned

| Dataset split | Rows (approx) |
|---|---|
| `paq` | 65M |
| `reddit_title_body` | 43M |
| `s2orc_title_abstract` | 41M |
| `amazon_reviews` | 23M |
| `s2orc_citation_pairs` | 13M |
| `wikipedia` | 11M |
| `gooaq` | 3M |
| `codesearchnet` | 2M |
| `stackexchange_titlebody_bestanswer` | 1M |

---

## Contributing

Want to contribute compute? Install the [annotator](https://github.com/embedcollective/annotator) and start labeling — works on NVIDIA GPUs, Apple Silicon, or CPU.

## Links

- [Embed Collective](https://embedcollective.com) — project homepage
- [annotator](https://github.com/embedcollective/annotator) — distributed annotation CLI
- [nomic-embed-unsupervised-data](https://huggingface.co/datasets/nomic-ai/nomic-embed-unsupervised-data) — source dataset
