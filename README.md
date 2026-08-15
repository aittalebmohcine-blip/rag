*This project has been created as part of the 42 curriculum by mait-tal.*

# Description

This repository implements a small Retrieval-Augmented Generation (RAG) pipeline in Python. It is designed to ingest a local repository of documents and source files, break them into chunks, index them with BM25, retrieve relevant spans for a question, and optionally generate an answer using an LLM prompt.

The goal is to demonstrate how a RAG system can combine classical retrieval with a generative model to answer queries based on local knowledge.

# System architecture

The pipeline is composed of several stages:

- `data/raw`: source repository documents and supported files are collected from this folder.
- Chunking: files are segmented into overlapping chunks using `src/Chunkers.py` and `src/Chunker.py`.
- Indexing: `src/indexer.py` builds a BM25 index from chunk text.
- Retrieval: `src/searcher.py` queries the BM25 index and returns ranked source spans.
- Answering: `src/answerer.py` builds a prompt from retrieved spans and `src/Generator.py` runs a causal LLM to generate an answer.
- CLI: `src/CLI.py` exposes commands for indexing, search, dataset search, answering, and evaluation.

Components interact as follows:

1. `CLI.index()` collects files, chunks them, saves chunk metadata, and builds the BM25 index.
2. `CLI.search()` loads the index and chunk list, retrieves top-k chunks for a query, and prints the source spans.
3. `CLI.answer()` retrieves sources and passes them to `Generator` through a prompt built by `answerer.build_prompt()`.
4. `CLI.search_dataset()` runs retrieval over a dataset of questions and saves the ranked sources.

# Chunking strategy

Document segmentation uses two chunkers:

- `TextChunker`: splits plain text into fixed-size, overlapping windows. The overlap is computed as 15% of the chunk size. This preserves context across chunk boundaries and avoids missing answers that span chunks.
- `PythonChunker`: parses Python files with `ast` and extracts function/class node spans when possible. If AST parsing fails, it falls back to `TextChunker`.

Both chunkers preserve absolute character offsets so retrieved spans can be mapped back to the original files.

# Retrieval method

Retrieval uses the `bm25s` library:

- The indexer tokenizes the chunk texts with `bm25s.tokenize()`.
- `bm25s.BM25` is used to index the tokenized corpus.
- At query time, the question is tokenized and BM25 retrieves the top-k matching document IDs.
- Retrieved IDs are converted back into chunk source spans and returned in ranked order.

Ranking is based on BM25 relevance scores.

# Performance analysis

### Retrieval Quality

The retrieval system was evaluated using the provided Recall@k metric.

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------:|---------:|----------:|-----------:|
| Documentation | 64.0% | 83.0% | 87.0% | 89.0% |
| Code | 36.4% | 48.5% | 53.5% | 61.6% |

The implementation satisfies the project requirements:

| Requirement | Required | Achieved |
|------------|---------:|---------:|
| Docs Recall@5 | ≥ 80% | 87.0% |
| Code Recall@5 | ≥ 50% | 53.5% |

> The project includes an evaluate CLI command that reproduces the Recall@k metric for local testing. Final performance figures reported below were obtained using the official moulinette.

### Chunk Size Evaluation

The project uses a maximum chunk size of **2000** characters.

| Chunk Size | Docs Recall@5 | Code Recall@5 | Notes                                                                                                   |
| ---------- | ------------: | ------------: | ------------------------------------------------------------------------------------------------------- |
| 500        |         81.0% |         48.5% | Small chunks improve localization but often split relevant context, reducing documentation recall.      |
| 1000       |         87.0% |         55.6% | Best overall retrieval performance; balances context size and precision.                                |
| 1500       |         88.0% |         56.6% | Highest documentation recall, but larger chunks begin to reduce code retrieval accuracy.                |
| 2000       |         87.0% |         53.5% | Selected configuration to match the project's default maximum chunk size while maintaining high recall. |

>Documentation retrieval consistently benefited from larger chunks, as related information often spans multiple paragraphs. In contrast, code retrieval peaked with 1000-character chunks, suggesting that excessively large chunks introduce irrelevant code and reduce ranking precision. Although a 1500-character chunk size achieved the highest documentation recall, the 2000-character configuration was selected because it matches the project's default maximum chunk size and still provides strong performance on both documentation and code datasets.

**Possible Improvements**:
- Hybrid retrieval (BM25 + embeddings).
- Better Python chunking.
- Query expansion or reranking.
- Incremental indexing.
### Runtime Performance
- Indexing time: a few seconds for the whole corpus( < 10s).
- Retrieval throughput: a few seconds for 200 questions ( < 12s).

    >The system comfortably satisfies the project's performance requirements, completing both indexing and batch retrieval well within the required limits. The short execution times are primarily due to the use of a lightweight lexical BM25 index, efficient preprocessing performed only once during indexing, and direct in-memory retrieval without expensive embedding generation or neural reranking.

# Design decisions

Key implementation choices:

- BM25 for retrieval: efficient and easy to use for local text corpora.
- Overlapping chunks: reduces context gaps between consecutive segments.
- Structured Python chunking: extracts semantically meaningful code blocks from `.py` files.
- Pydantic models: provide validation and clear schema definitions for chunks, sources, and datasets.
- Hugging Face Transformers model wrapper: isolates LLM inference behind `src/Generator.py`.

# Challenges faced

- Aligning chunk offsets with file spans while preserving readability.
- Handling Python source files with AST parsing and falling back gracefully when parsing fails.
- Keeping the retrieval pipeline simple while supporting both text and code files.

# Instructions

0. you should have uv package manager

1. Install dependencies:

```bash
uv sync
```

2. Prepare input data:

- Place supported files under `data/raw/`.
- Supported extensions: `.md`, `.txt`, `.py`.

3. Build the index:

```bash
uv run python -m src index --max_chunk_size=2000
```

4. Search for a query:

```bash
uv run python -m src search "What is retrieval-augmented generation?"
```

5. Generate an answer:

```bash
uv run python -m src answer "How does the BM25 retriever work?"
```

# Example usage

- Index data:

```bash
uv run python -m src index --max_chunk_size=1500
```

- Search a single query:

```bash
uv run python -m src search "Find the best matching source for code chunking"
```

- Answer a question with generated text:

```bash
uv run python -m src answer "Explain the retrieval pipeline."
```

# Resources

- BM25 algorithm: https://en.wikipedia.org/wiki/Okapi_BM25
- Retrieval-augmented generation: https://arxiv.org/abs/2005.11401
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index
- Python AST documentation: https://docs.python.org/3/library/ast.html

## AI usage description

AI was used as a productivity tool throughout the development of this project, primarily for:

- Assisting with documentation and improving the clarity of the README.
- Explaining concepts and clarifying project requirements.
- Helping with repetitive programming tasks, such as generating boilerplate code, suggesting refactorings, and reviewing code for readability.
- Answering language- and syntax-related questions during development.

All architectural decisions, algorithms, implementation choices, and final code were designed, implemented, tested, and validated manually. AI-generated suggestions were reviewed and adapted before being incorporated into the project.