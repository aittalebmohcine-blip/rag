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

- recall@k: [empty]
- System latency: [empty]
- Index build time: [empty]

> Technical output placeholders are intentionally left empty here. Actual recall@k and performance values should be measured on the target dataset and environment.

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

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Prepare input data:

- Place supported files under `data/raw/`.
- Supported extensions: `.md`, `.txt`, `.py`.

3. Build the index:

```bash
python -m src index --max_chunk_size=2000
```

4. Search for a query:

```bash
python -m src search "What is retrieval-augmented generation?"
```

5. Generate an answer:

```bash
python -m src answer "How does the BM25 retriever work?"
```

# Example usage

- Index data:

```bash
python -m src index --max_chunk_size=1500
```

- Search a single query:

```bash
python -m src search "Find the best matching source for code chunking"
```

- Answer a question with generated text:

```bash
python -m src answer "Explain the retrieval pipeline."
```

# Resources

- BM25 algorithm: https://en.wikipedia.org/wiki/Okapi_BM25
- Retrieval-augmented generation: https://arxiv.org/abs/2005.11401
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index
- Python AST documentation: https://docs.python.org/3/library/ast.html

## AI usage description

AI is used in this project for the answer generation step only. The system uses a Hugging Face causal language model (`Qwen/Qwen3-0.6B`) in `src/Generator.py` to generate answers from a prompt assembled with retrieved source content. All retrieval and chunking logic is implemented with explicit Python code and classical indexing.
