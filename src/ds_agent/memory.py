"""Memory Backbone for DS-Agent."""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Default embedding model
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configure logging
logger = logging.getLogger(__name__)

# Set of supported file types
SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".csv", ".pdf", ".yaml"}


class MemoryChunk(BaseModel):
    """A chunk of memory with metadata."""

    content: str
    score: float
    source_uri: str
    episode_id: Optional[str] = None


class Memory:
    """Memory Backbone for storing and retrieving documents."""

    def __init__(self, memory_dir: Path) -> None:
        """Initialize Memory Backbone.

        Args:
            memory_dir: Path to memory directory (contains raw/, chroma/, tmp/)
        """
        self.memory_dir = memory_dir
        self.raw_dir = memory_dir / "raw"
        self.chroma_dir = memory_dir / "chroma"
        self.tmp_dir = memory_dir / "tmp"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Check for model override in environment variable
        import os

        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
        logger.info(f"Using embedding model: {model_name}")

        # Initialize ChromaDB with custom settings
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
            ),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {e}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into overlapping chunks."""
        # If text is smaller than chunk_size, return as a single chunk
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        overlap = chunk_size // 2

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])

            # Ensure we always advance by at least 1 character
            next_start = end - overlap
            if next_start <= start:
                next_start = start + 1

            start = next_start

            # Check if we've reached the end
            if start >= len(text):
                break

        return chunks

    def ingest(self, path: Path, *, episode_id: Optional[str] = None) -> str:
        """Add a file or folder to memory.

        Args:
            path: Path to file or directory to ingest
            episode_id: Optional episode ID to associate with the document

        Returns:
            Document ID (hash of file content)

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If file type is not supported
            RuntimeError: If embedding or storage fails
        """
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Ensure raw_dir exists
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Handle directory
        if path.is_dir():
            doc_ids = []
            for file_path in path.glob("**/*"):
                if file_path.is_file():
                    try:
                        doc_id = self.ingest(file_path, episode_id=episode_id)
                        doc_ids.append(doc_id)
                    except ValueError as e:
                        logger.info(f"SKIPPED unsupported file: {file_path} - {e}")
                        continue  # Skip unsupported file types
            return ",".join(doc_ids)

        # Handle single file
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            msg = f"Unsupported file type: {path.suffix}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            logger.info(f"SKIPPED unsupported file: {path} - {msg}")
            raise ValueError(msg)

        # Compute file hash
        file_hash = self._compute_file_hash(path)

        # Check if already ingested
        try:
            result = self.collection.get(ids=[file_hash])
            if result["ids"]:
                logger.info(f"SKIPPED duplicate: {path} (hash: {file_hash})")
                return file_hash
        except Exception as e:
            logger.error(f"Failed to check existing document: {e}")
            raise RuntimeError(f"Failed to check existing document: {e}")

        # Copy to raw storage
        target_path = self.raw_dir / f"{file_hash}{path.suffix}"
        try:
            print(
                f"[DEBUG] Memory.ingest: checking if {target_path} exists: {target_path.exists()}"
            )
            # Only copy if the target file doesn't exist
            if not target_path.exists():
                print(f"[DEBUG] Memory.ingest: copying {path} to {target_path}")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target_path)
        except Exception as e:
            logger.error(f"Failed to copy file to raw storage: {e}")
            raise RuntimeError(f"Failed to copy file to raw storage: {e}")

        # Read and chunk content
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file content: {e}")
            raise RuntimeError(f"Failed to read file content: {e}")

        chunks = self._chunk_text(content)

        # Generate embeddings
        try:
            embeddings = self.embedding_model.encode(chunks)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")

        # Store in ChromaDB
        try:
            # Use filelock to ensure concurrent safety
            from filelock import FileLock, Timeout

            lock_path = self.chroma_dir / "LOCK"

            try:
                with FileLock(lock_path, timeout=30):  # Add 30-second timeout
                    # Handle both numpy arrays and list embeddings
                    emb_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

                    # Prepare metadata, ensuring no None values (ChromaDB doesn't accept None)
                    metadatas = []
                    for i in range(len(chunks)):
                        metadata = {
                            "source_uri": str(target_path),
                            "chunk_index": i,
                        }
                        # Only add episode_id if it's not None
                        if episode_id is not None:
                            metadata["episode_id"] = episode_id
                        metadatas.append(metadata)

                    self.collection.add(
                        ids=[f"{file_hash}_{i}" for i in range(len(chunks))],
                        embeddings=emb_list,
                        documents=chunks,
                        metadatas=metadatas,
                    )
                    logger.info(f"Ingested: {path} (hash: {file_hash}, chunks: {len(chunks)})")
            except Timeout:
                logger.error("Timed out waiting for file lock after 30 seconds")
                raise RuntimeError(
                    f"Timed out waiting for file lock after 30 seconds. Another process may be holding the lock."
                )
        except Exception as e:
            logger.error(f"Failed to store in ChromaDB: {e}")
            raise RuntimeError(f"Failed to store in ChromaDB: {e}")

        return file_hash

    def query(self, text: str, k: int = 5) -> List[MemoryChunk]:
        """Return top-k chunks matching the query.

        Args:
            text: Query text
            k: Number of chunks to return

        Returns:
            List of MemoryChunk objects

        Raises:
            RuntimeError: If embedding or query fails
        """
        # Generate query embedding
        try:
            query_embedding = self.embedding_model.encode(text)
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {e}")

        # Query ChromaDB
        try:
            # Use filelock to ensure concurrent safety
            from filelock import FileLock, Timeout

            lock_path = self.chroma_dir / "LOCK"

            try:
                with FileLock(lock_path, timeout=30):  # Add 30-second timeout
                    # Handle both numpy arrays and list embeddings
                    query_emb_list = (
                        query_embedding.tolist()
                        if hasattr(query_embedding, "tolist")
                        else query_embedding
                    )

                    results = self.collection.query(
                        query_embeddings=[query_emb_list],
                        n_results=k,
                    )
            except Timeout:
                raise RuntimeError(f"Timed out waiting for file lock during query after 30 seconds")
        except Exception as e:
            raise RuntimeError(f"Failed to query ChromaDB: {e}")

        # Convert to MemoryChunk objects
        chunks = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            chunks.append(
                MemoryChunk(
                    content=results["documents"][0][i],
                    score=results["distances"][0][i],
                    source_uri=metadata["source_uri"],
                    episode_id=metadata.get(
                        "episode_id"
                    ),  # Use .get() to safely handle missing keys
                )
            )

        return chunks
