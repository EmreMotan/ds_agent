# Memory Architecture (Updated)

## 1. Storage Layout

The memory backbone uses a structured directory layout:

```
/memory/
   raw/                 # Original files (verbatim, by hash)
   chroma/              # ChromaDB SQLite + FAISS index (vector store)
   tmp/                 # Temporary scratch space during ingestion
```

- **raw/**: Stores each ingested file as `{hash}{ext}` (e.g., `beef1234.txt`). Hash is SHA-256 of file content.
- **chroma/**: Persistent vector store (ChromaDB) for fast semantic search, with a file lock for concurrency.
- **tmp/**: Used for intermediate processing (future use).

## 2. Supported File Types

- Only files with extensions `.md`, `.txt`, `.py`, `.csv`, `.pdf`, `.yaml` are accepted for ingestion.
- Directories are recursively ingested, skipping unsupported files.

## 3. Chunking Algorithm

- **Chunk size:** 1000 characters
- **Overlap:** 500 characters (half the chunk size)
- If the file is smaller than the chunk size, it is a single chunk.
- Chunks are created by sliding a window with overlap, ensuring no data is missed and context is preserved.

## 4. Ingestion Flow

1. **File/Directory Check**
   - If a directory, recursively ingest all supported files.
   - If a file, check extension and compute SHA-256 hash.

2. **Deduplication**
   - If a file with the same hash is already present in the vector store, skip ingestion (idempotent).

3. **Copy to Raw**
   - File is copied to `raw/` as `{hash}{ext}` if not already present.

4. **Chunking & Embedding**
   - File is read as UTF-8 text and split into overlapping chunks.
   - Each chunk is embedded using the selected model (default: `sentence-transformers/all-MiniLM-L6-v2`).

5. **Storage in ChromaDB**
   - Chunks, embeddings, and metadata (including `source_uri`, `chunk_index`, and optional `episode_id`) are stored in ChromaDB.
   - All ChromaDB writes are protected by a file lock (`chroma/LOCK`) for safe parallel access.

## 5. Query Flow

- Query text is embedded using the same model.
- ChromaDB is searched for the top-k most similar chunks (default: k=5).
- Results include content, similarity score, source URI, and episode ID (if present).
- All ChromaDB queries are protected by a file lock for concurrency.

## 6. Embedding Model Selection

- The embedding model is selected via the `EMBEDDING_MODEL` environment variable.
- If unset, defaults to `sentence-transformers/all-MiniLM-L6-v2` (local, no API calls).
- Example:
  ```bash
  export EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
  ```
- The model is loaded at `Memory` initialization and logged.

## 7. Concurrency & Robustness

- All ChromaDB operations (ingest/query) use a file lock (`chroma/LOCK`) with a 30-second timeout to ensure safe concurrent access.
- If the lock cannot be acquired, a clear error is raised.

## 8. API Example

```python
from ds_agent.memory import Memory
from pathlib import Path

memory = Memory(Path("./memory"))
doc_id = memory.ingest(Path("docs/design.md"), episode_id="DS-25-002")
results = memory.query("signup success metric definition", k=3)
for chunk in results:
    print(chunk.content, chunk.score, chunk.source_uri)
```

## 9. Error Handling

- Ingestion and query methods raise clear exceptions for:
  - File not found
  - Unsupported file type
  - Embedding/model errors
  - ChromaDB errors (including lock timeouts)
- Duplicate files are skipped with a log message.

## 10. Extensibility

- The system is designed to support additional file types and chunking strategies in the future.
- Embedding model can be swapped via environment variable.
- Metadata can be extended (e.g., for episode linkage, chunk provenance).

---

This documentation now accurately reflects the current implementation and robustness of the DS-Agent memory subsystem. 