# rag-uk

RAG search over Ukrainian documents. Answers carry verbatim citations, and quality is measured on a hand-labeled question set.

The system splits documents into chunks, retrieves them in one of four ways, and asks an LLM to answer only from what was retrieved. Every claim in the answer has to be backed by a quote in «...» followed by a chunk id such as `[contract:0003]`. An id that was not in the retrieved context counts as an error.

The goal of the repository is to compare retrieval modes on your own corpus and see where they fail, not only to get a working chat.

## Retrieval modes

| Mode | What it does |
|---|---|
| `bm25` | Lexical Okapi BM25 search, no models and no network |
| `vector` | Cosine similarity against chunk embeddings |
| `hybrid` | BM25 and vector rankings merged with reciprocal rank fusion |
| `hybrid_rerank` | Candidates from `hybrid`, rescored by a cross-encoder |

Embeddings: `intfloat/multilingual-e5-base` and `paraphrase-multilingual-MiniLM-L12-v2`, both run locally through `sentence-transformers`. Reranker: `BAAI/bge-reranker-v2-m3`. Models are set in `config.yaml`; the `query:` and `passage:` prefixes that e5 needs are stored with each embedding profile.

## Quick start without models or keys

Requires Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python tools/build_index.py --embeddings none
python ask.py "What is the payment term for public customers?" --mode bm25 --no-generation
pytest -q
```

`ask.py` prints the retrieved chunks with their section and paragraph numbers. The corpus in `data/corpus/` is a single example file, so replace it with your own `.md` or `.txt` files. `#` headings define sections.

## Full run

```bash
pip install -e ".[vector,dev]"
cp .env.example .env          # set LLM_API_BASE, LLM_API_KEY, LLM_MODEL
set -a; source .env; set +a   # .env is not loaded automatically

python tools/build_index.py --embeddings all
python tools/label_cli.py     # label questions by hand
python eval.py
```

The client works with any OpenAI-compatible `/chat/completions` endpoint, including a local server. There is no default model name: without `LLM_MODEL` the system only shows the retrieved chunks.

## Evaluation

A person labels the question set. `tools/label_cli.py` shows a paragraph, asks for a question, and appends a row to `data/eval.jsonl`:

```json
{"id": "q-0001", "question": "...", "gold_chunk_ids": ["contract:0004"]}
```

`tools/draft_questions.py` can propose candidates through an LLM, but it writes them to a separate file marked "needs manual review" and never touches `eval.jsonl`. On an empty set, `eval.py` refuses to compute metrics.

What is measured:

- retrieval: `recall@1`, `recall@3`, `recall@5` and `MRR` for each mode and each embedding model;
- generation: the share of answers in which every citation points to a chunk that was in the context; answers without citations are counted separately;
- cost and latency: every model call (chat, embeddings, reranker) is logged to `results/calls.jsonl` with tokens and time; cost is computed only if prices are set in the config.

Results go to `results/metrics.md`, `results/eval_<time>.json` and `results/details_<time>.jsonl`.

**There are no numbers yet.** The system has not been run on a real corpus, so any metrics for this repository have to be produced by you.

## Configuration

Parameters live in `config.yaml`: chunk sizes, `k1` and `b` for BM25, `top_k`, `candidate_k`, `rrf_k`, LLM timeouts and retries. The values are starting points and were not tuned on a corpus. Environment variables override the file:

| Variable | Purpose |
|---|---|
| `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL` | LLM provider |
| `RAG_CONFIG` | Path to another config file |
| `RAG_RETRIEVAL_MODE` | Retrieval mode |
| `RAG_EMBEDDING_MODEL` | Embedding profile |

## Layout

```
ask.py             one question: retrieval and an answer with citations
eval.py            metrics across all modes
rag_uk/            library: chunking, indexes, fusion, reranker, LLM, citations, metrics
tools/             index building, manual labeling, question drafts
tests/             136 tests, no models or network required
docs/DESIGN.md     decisions, rejected alternatives, known failure modes (in Ukrainian)
```

## Known limitations

- BM25 has no lemmatization: the query "оплата" will not find "оплати". Ukrainian morphology is not covered yet.
- When a rule and its exception end up in different chunks, the model may answer from the rule alone. Expanding the context with neighboring chunks of the same section is planned; details are in `docs/DESIGN.md`.
- Chunk overlap increases the number of valid gold chunks for a paragraph, so `recall@k` partly depends on the chunking settings.
- The reranker has not been checked on Ukrainian.
- A citation is verified by chunk id, not by an exact text match.
