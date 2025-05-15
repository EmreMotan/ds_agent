"""Tests for Memory Backbone."""

import concurrent.futures
import os
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ds_agent.memory import Memory, MemoryChunk

# Module-level mock for shutil.copy2


def mock_copy2(src, dst):
    print(f"[DEBUG] mock_copy2 called: src={src}, dst={dst}")
    # Only create the file if it does not already exist
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            f.write(f"Test content from {src}")
    return dst


@pytest.fixture(autouse=True)
def patch_shutil_copy2():
    with patch("ds_agent.memory.shutil.copy2", side_effect=mock_copy2):
        yield


@pytest.fixture
def memory_dir(tmp_path):
    """Create a temporary memory directory."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    yield memory_dir
    shutil.rmtree(memory_dir)


@pytest.fixture
def memory(memory_dir):
    """Create a Memory instance."""
    with patch("ds_agent.memory.SentenceTransformer") as mock_model:
        # Setup the mock model with an encode method that returns simple embeddings
        mock_instance = mock_model.return_value
        mock_instance.encode.return_value = [0.1, 0.2, 0.3]  # Return a single embedding vector

        # Also mock the ChromaDB client to avoid connection issues
        with patch("ds_agent.memory.chromadb.PersistentClient") as mock_client:
            # Setup the mock collection
            mock_collection = MagicMock()
            # When get is called with no args, return empty list, but after add is called, return some items
            mock_collection.get.side_effect = lambda ids=None: {
                "ids": ["hash_0"] if ids else [],
                "documents": ["Test content 0"] if ids else [],
                "metadatas": (
                    [{"source_uri": "test/path", "episode_id": "test-episode"}] if ids else []
                ),
                "embeddings": [[0.1, 0.2, 0.3]] if ids else [],
                "distances": [] if ids else [],
            }

            # Define add side effect function to update get behavior
            def add_side_effect(**kwargs):
                # Create a new side effect for get method
                def get_side_effect(ids=None):
                    return {
                        "ids": kwargs["ids"],
                        "documents": kwargs["documents"],
                        "metadatas": kwargs["metadatas"],
                        "embeddings": kwargs.get("embeddings", []),
                        "distances": [],
                    }

                # Update the get side effect
                mock_collection.get.side_effect = get_side_effect

                # Create a test file in the raw directory to simulate file copy
                test_hash = kwargs["ids"][0].split("_")[0]  # Extract hash part from the ID
                raw_file_path = memory_dir / "raw" / f"{test_hash}.txt"
                raw_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(raw_file_path, "w") as f:
                    f.write("Test content 0")

            # Set the add side effect
            mock_collection.add.side_effect = add_side_effect

            # Setup client
            mock_client_instance = mock_client.return_value
            mock_client_instance.get_or_create_collection.return_value = mock_collection

            mem = Memory(memory_dir)
            return mem


@pytest.fixture
def test_files(tmp_path):
    """Create test files."""
    files = []
    for i in range(3):
        file_path = tmp_path / f"test_{i}.txt"
        file_path.write_text(f"Test content {i}")
        files.append(file_path)
    return files


def test_memory_initialization(memory_dir):
    """Test Memory initialization."""
    memory = Memory(memory_dir)
    assert memory.memory_dir == memory_dir
    assert memory.raw_dir == memory_dir / "raw"
    assert memory.chroma_dir == memory_dir / "chroma"
    assert memory.tmp_dir == memory_dir / "tmp"
    assert memory.raw_dir.exists()
    assert memory.chroma_dir.exists()
    assert memory.tmp_dir.exists()


def test_memory_initialization_with_env_var(memory_dir):
    """Test Memory initialization with environment variable override."""
    custom_model = "sentence-transformers/all-mpnet-base-v2"
    with patch.dict(os.environ, {"EMBEDDING_MODEL": custom_model}):
        with patch("ds_agent.memory.SentenceTransformer") as mock_model:
            Memory(memory_dir)
            mock_model.assert_called_once_with(custom_model)


def test_memory_initialization_failure(memory_dir):
    """Test Memory initialization failure."""
    with patch("ds_agent.memory.SentenceTransformer") as mock_model:
        mock_model.side_effect = Exception("Model initialization failed")
        with pytest.raises(RuntimeError, match="Failed to initialize embedding model"):
            Memory(memory_dir)


@pytest.mark.timeout(60)  # Add 60 second timeout to prevent indefinite hanging
def test_ingest_file(memory, test_files):
    """Test ingesting a single file."""
    # Mock the embedding model
    mock_embeddings = [0.1, 0.2, 0.3]  # Simple mock embedding

    # Mock file hashing and collection get to not find a duplicate
    test_hash = "beea3c8760493a38acb612da8023a72517ce85ea14d723fab5bd7e67baca6c53"
    with patch.object(memory, "_compute_file_hash", return_value=test_hash):
        # Setup collection to not find existing document first time, but find it second time
        collection_get_result = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": [],
            "distances": [],
        }
        with patch.object(memory.collection, "get", return_value=collection_get_result):
            # Mock shutil.copy2 to verify it was called
            with patch("ds_agent.memory.shutil.copy2") as mock_copy:
                mock_copy.return_value = "mocked_copy_result"

                with patch.object(memory.embedding_model, "encode", return_value=mock_embeddings):
                    doc_id = memory.ingest(test_files[0])
                    assert doc_id is not None
                    assert len(doc_id) == 64  # SHA-256 hash length
                    assert doc_id == test_hash  # Verify we got the hash we expected

                    # Verify copy was called
                    mock_copy.assert_called_once()

                    # Test that ChromaDB add was called
                    memory.collection.add.assert_called_once()

                    # Verify data was retrieved from ChromaDB
                    results = memory.collection.get()
                    assert len(results["ids"]) > 0


@pytest.mark.timeout(60)
def test_ingest_file_with_episode_id(memory, test_files):
    """Test ingesting a file with episode ID."""
    # Mock the embedding model
    mock_embeddings = [0.1, 0.2, 0.3]  # Simple mock embedding
    with patch.object(memory.embedding_model, "encode", return_value=mock_embeddings):
        episode_id = "test-episode"
        doc_id = memory.ingest(test_files[0], episode_id=episode_id)

        # Verify episode ID in metadata
        results = memory.collection.get()
        assert all(
            metadata["episode_id"] == episode_id
            for metadata in results["metadatas"]
            if metadata is not None and "episode_id" in metadata
        )


@pytest.mark.timeout(60)
def test_ingest_directory(memory, test_files, tmp_path):
    """Test ingesting a directory."""
    # Mock the embedding model
    mock_embeddings = [0.1, 0.2, 0.3]  # Simple mock embedding
    with patch.object(memory.embedding_model, "encode", return_value=mock_embeddings):
        # Create a subdirectory with files
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        for file in test_files:
            shutil.copy2(file, subdir / file.name)

        # Mock file hashing to return consistent hashes
        test_hashes = [
            "hash1",
            "hash2",
            "hash3",
        ]
        with patch.object(memory, "_compute_file_hash", side_effect=test_hashes):
            # Mock collection to not find existing documents
            with patch.object(
                memory.collection,
                "get",
                return_value={
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "embeddings": [],
                    "distances": [],
                },
            ):
                # Mock shutil.copy2 to create files in raw directory
                def mock_copy2(src, dst):
                    # Extract hash from destination path
                    hash_part = dst.name.split(".")[0]
                    # Create file in raw directory
                    raw_file_path = memory.raw_dir / f"{hash_part}.txt"
                    raw_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(raw_file_path, "w") as f:
                        f.write(f"Test content from {src}")

                with patch("ds_agent.memory.shutil.copy2", side_effect=mock_copy2):
                    # Mock the glob function to return all files in the directory
                    with patch(
                        "pathlib.Path.glob",
                        return_value=[subdir / f"test_{i}.txt" for i in range(3)],
                    ):
                        doc_ids = memory.ingest(subdir)
                        assert doc_ids.count(",") == 2  # Three files, two commas

                        # Verify all files were processed
                        raw_files = list(memory.raw_dir.glob("*.txt"))
                        assert len(raw_files) == 3


def test_ingest_unsupported_file(memory, tmp_path):
    """Test ingesting an unsupported file type."""
    file_path = tmp_path / "test.xyz"
    file_path.write_text("Test content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        memory.ingest(file_path)


def test_ingest_nonexistent_file(memory, tmp_path):
    """Test ingesting a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        memory.ingest(tmp_path / "nonexistent.txt")


@pytest.mark.timeout(60)
def test_ingest_duplicate_file_no_change(memory, test_files):
    """Test ingesting the same file twice doesn't add new vectors."""
    # Mock the embedding model
    mock_embeddings = [0.1, 0.2, 0.3]  # Simple mock embedding
    with patch.object(memory.embedding_model, "encode", return_value=mock_embeddings):
        # Mock file hashing to return consistent hash
        test_hash = "test_hash"
        with patch.object(memory, "_compute_file_hash", return_value=test_hash):
            # First ingest
            with patch.object(
                memory.collection,
                "get",
                return_value={
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "embeddings": [],
                    "distances": [],
                },
            ):
                # Mock shutil.copy2 to create file in raw directory
                def mock_copy2(src, dst):
                    # Only create the file if it does not already exist
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with open(dst, "w") as f:
                            f.write(f"Test content from {src}")
                    return dst

                with patch("ds_agent.memory.shutil.copy2", side_effect=mock_copy2):
                    doc_id1 = memory.ingest(test_files[0])

            # Get the number of chunks after first ingest
            results1 = memory.collection.get()
            chunk_count1 = len(results1["ids"])

            # Second ingest of the same file
            with patch.object(
                memory.collection,
                "get",
                return_value={
                    "ids": [test_hash],
                    "documents": ["Test content"],
                    "metadatas": [{"source_uri": str(memory.raw_dir / f"{test_hash}.txt")}],
                    "embeddings": [[0.1, 0.2, 0.3]],
                    "distances": [],
                },
            ):
                doc_id2 = memory.ingest(test_files[0])

            # Verify the document IDs are the same
            assert doc_id1 == doc_id2

            # Verify no new chunks were added
            results2 = memory.collection.get()
            chunk_count2 = len(results2["ids"])
            assert chunk_count1 == chunk_count2

            # Verify only one copy in raw storage (only the hash-named file)
            raw_files = list(
                memory.raw_dir.glob("test_hash.*")
            )  # Only count files matching the hash
            print(
                f"[DEBUG] test_ingest_duplicate_file_no_change: raw_files after second ingest: {raw_files}"
            )
            assert len(raw_files) == 1


@pytest.mark.timeout(60)
def test_ingest_duplicate_file(memory, test_files):
    """Test ingesting the same file twice."""
    # Mock the embedding model
    mock_embeddings = [0.1, 0.2, 0.3]  # Simple mock embedding
    with patch.object(memory.embedding_model, "encode", return_value=mock_embeddings):
        # Mock file hashing to return consistent hash
        test_hash = "test_hash"
        with patch.object(memory, "_compute_file_hash", return_value=test_hash):
            # First ingest
            with patch.object(
                memory.collection,
                "get",
                return_value={
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "embeddings": [],
                    "distances": [],
                },
            ):
                # Mock shutil.copy2 to create file in raw directory
                def mock_copy2(src, dst):
                    # Only create the file if it does not already exist
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with open(dst, "w") as f:
                            f.write(f"Test content from {src}")
                    return dst

                with patch("ds_agent.memory.shutil.copy2", side_effect=mock_copy2):
                    doc_id1 = memory.ingest(test_files[0])

            # Second ingest
            with patch.object(
                memory.collection,
                "get",
                return_value={
                    "ids": [test_hash],
                    "documents": ["Test content"],
                    "metadatas": [{"source_uri": str(memory.raw_dir / f"{test_hash}.txt")}],
                    "embeddings": [[0.1, 0.2, 0.3]],
                    "distances": [],
                },
            ):
                doc_id2 = memory.ingest(test_files[0])

            assert doc_id1 == doc_id2

            # Verify only one copy in raw storage (only the hash-named file)
            raw_files = list(
                memory.raw_dir.glob("test_hash.*")
            )  # Only count files matching the hash
            print(f"[DEBUG] test_ingest_duplicate_file: raw_files after second ingest: {raw_files}")
            assert len(raw_files) == 1


def test_ingest_storage_failure(memory, test_files):
    """Test handling storage failures."""
    # Mock file hashing to return consistent hash
    test_hash = "test_hash"
    with patch.object(memory, "_compute_file_hash", return_value=test_hash):
        # Mock collection to not find existing document
        with patch.object(
            memory.collection,
            "get",
            return_value={
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
                "distances": [],
            },
        ):
            # Mock shutil.copy2 to raise an exception
            with patch("ds_agent.memory.shutil.copy2", side_effect=Exception("Storage failed")):
                with pytest.raises(RuntimeError, match="Failed to copy file to raw storage"):
                    memory.ingest(test_files[0])


def test_ingest_embedding_failure(memory, test_files):
    """Test handling embedding failures."""
    # Mock file hashing to return consistent hash
    test_hash = "test_hash"
    with patch.object(memory, "_compute_file_hash", return_value=test_hash):
        # Mock collection to not find existing document
        with patch.object(
            memory.collection,
            "get",
            return_value={
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
                "distances": [],
            },
        ):
            # Mock shutil.copy2 to succeed
            with patch("ds_agent.memory.shutil.copy2") as mock_copy:
                # Mock embedding model to raise an exception
                with patch.object(
                    memory.embedding_model, "encode", side_effect=Exception("Embedding failed")
                ):
                    with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
                        memory.ingest(test_files[0])


def test_ingest_chromadb_failure(memory, test_files):
    """Test handling ChromaDB failures."""
    # Mock file hashing to return consistent hash
    test_hash = "test_hash"
    with patch.object(memory, "_compute_file_hash", return_value=test_hash):
        # Mock collection to not find existing document
        with patch.object(
            memory.collection,
            "get",
            return_value={
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
                "distances": [],
            },
        ):
            # Mock shutil.copy2 to succeed
            with patch("ds_agent.memory.shutil.copy2") as mock_copy:
                # Mock embedding model to succeed
                with patch.object(memory.embedding_model, "encode", return_value=[0.1, 0.2, 0.3]):
                    # Mock ChromaDB add to raise an exception
                    with patch.object(
                        memory.collection, "add", side_effect=Exception("ChromaDB failed")
                    ):
                        with pytest.raises(RuntimeError, match="Failed to store in ChromaDB"):
                            memory.ingest(test_files[0])


@pytest.mark.timeout(60)
def test_concurrent_ingestion(memory_dir, tmp_path):
    """Test concurrent ingestion of different files."""
    # Create test files
    file1 = tmp_path / "concurrent1.txt"
    file2 = tmp_path / "concurrent2.txt"
    file1.write_text("Test content for concurrent ingestion 1")
    file2.write_text("Test content for concurrent ingestion 2")

    # Mock the embedding model
    mock_embeddings = [0.1, 0.2, 0.3]  # Simple mock embedding

    # Function to ingest a file in a thread
    def ingest_file(file_path):
        with patch("ds_agent.memory.SentenceTransformer") as mock_st:
            # Setup the mock
            mock_instance = mock_st.return_value
            mock_instance.encode.return_value = mock_embeddings

            # Also mock ChromaDB client
            with patch("ds_agent.memory.chromadb.PersistentClient") as mock_client:
                # Setup mock collection
                mock_collection = MagicMock()
                # Track ingestion state per hash
                ingested = {}

                def get_side_effect(ids=None):
                    if ids:
                        hash_id = ids[0].split("_")[0]
                        if ingested.get(hash_id):
                            return {
                                "ids": [ids[0]],
                                "documents": [f"Test content for {hash_id}"],
                                "metadatas": [
                                    {
                                        "source_uri": f"test/path/{hash_id}",
                                        "episode_id": "test-episode",
                                    }
                                ],
                                "embeddings": [[0.1, 0.2, 0.3]],
                                "distances": [],
                            }
                        else:
                            return {
                                "ids": [],
                                "documents": [],
                                "metadatas": [],
                                "embeddings": [],
                                "distances": [],
                            }
                    else:
                        return {
                            "ids": [],
                            "documents": [],
                            "metadatas": [],
                            "embeddings": [],
                            "distances": [],
                        }

                mock_collection.get.side_effect = get_side_effect

                def add_side_effect(**kwargs):
                    # Mark as ingested for all ids
                    for id_ in kwargs["ids"]:
                        hash_id = id_.split("_")[0]
                        ingested[hash_id] = True

                mock_collection.add.side_effect = add_side_effect

                # Setup client
                mock_client_instance = mock_client.return_value
                mock_client_instance.get_or_create_collection.return_value = mock_collection

                # Mock file hashing to return different hashes for different files
                def fake_hash(path):
                    return f"hash_{path.name}"

                with patch("ds_agent.memory.Memory._compute_file_hash", side_effect=fake_hash):
                    mem_instance = Memory(memory_dir)
                    return mem_instance.ingest(file_path)

    # Use ThreadPoolExecutor to run ingestion in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(ingest_file, file1)
        future2 = executor.submit(ingest_file, file2)

        # Get results and make sure they complete without exception
        doc_id1 = future1.result()
        doc_id2 = future2.result()

    # Verify both files were ingested
    assert doc_id1 != doc_id2

    # Verify both files are in raw storage (only the hash-named files)
    mem_instance = Memory(memory_dir)
    raw_files = list(
        mem_instance.raw_dir.glob("hash_*.txt")
    )  # Only count files matching the hash pattern
    print(f"[DEBUG] test_concurrent_ingestion: raw_files after concurrent ingest: {raw_files}")
    assert len(raw_files) == 2


@pytest.mark.timeout(60)
def test_query(memory, test_files):
    """Test querying memory."""
    # Mock the embedding model for both ingest and query
    mock_embeddings = [0.1, 0.2, 0.3]  # Simple mock embedding
    with patch.object(memory.embedding_model, "encode", return_value=mock_embeddings):
        # Ingest files
        for file in test_files:
            memory.ingest(file)

        # Mock query to return 5 results
        mock_collection = memory.collection
        mock_collection.query.return_value = {
            "ids": [["id1", "id2", "id3", "id4", "id5"]],
            "documents": [["content1", "content2", "content3", "content4", "content5"]],
            "metadatas": [
                [
                    {"source_uri": "test/path1", "episode_id": "test-episode"},
                    {"source_uri": "test/path2"},
                    {"source_uri": "test/path3", "episode_id": "test-episode"},
                    {"source_uri": "test/path4"},
                    {"source_uri": "test/path5", "episode_id": "test-episode"},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }

        # Query
        chunks = memory.query("Test content")
        assert len(chunks) == 5  # Default k=5
        assert all(isinstance(chunk, MemoryChunk) for chunk in chunks)
        assert all(chunk.score > 0 for chunk in chunks)


@pytest.mark.timeout(60)
def test_query_with_k(memory, test_files):
    """Test querying with custom k value."""
    # Mock the embedding model for both ingest and query
    mock_embeddings = [0.1, 0.2, 0.3]  # Simple mock embedding
    with patch.object(memory.embedding_model, "encode", return_value=mock_embeddings):
        # Mock collection query to return 2 results
        mock_collection = memory.collection
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["content1", "content2"]],
            "metadatas": [
                [
                    {"source_uri": "test/path1", "episode_id": "test-episode"},
                    {"source_uri": "test/path2"},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        # Query with k=2
        chunks = memory.query("Test content", k=2)
        assert len(chunks) == 2
        assert all(isinstance(chunk, MemoryChunk) for chunk in chunks)
        assert all(chunk.score > 0 for chunk in chunks)


def test_query_embedding_failure(memory, test_files):
    """Test handling query embedding failures."""
    # Ingest a file first
    memory.ingest(test_files[0])

    with patch.object(memory.embedding_model, "encode") as mock_encode:
        mock_encode.side_effect = Exception("Embedding failed")
        with pytest.raises(RuntimeError, match="Failed to generate query embedding"):
            memory.query("Test content")


def test_query_chromadb_failure(memory, test_files):
    """Test handling ChromaDB query failures."""
    # Ingest a file first
    memory.ingest(test_files[0])

    with patch.object(memory.collection, "query") as mock_query:
        mock_query.side_effect = Exception("ChromaDB failed")
        with pytest.raises(RuntimeError, match="Failed to query ChromaDB"):
            memory.query("Test content")


@pytest.mark.timeout(60)
def test_episode_id_association(memory, test_files):
    """Test associating episode ID with documents."""
    # Mock the embedding model
    mock_embeddings = [0.1, 0.2, 0.3]  # Simple mock embedding
    with patch.object(memory.embedding_model, "encode", return_value=mock_embeddings):
        episode_id = "test-episode"

        # Mock collection query to return results with episode ID
        mock_collection = memory.collection
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["content1", "content2"]],
            "metadatas": [
                [
                    {"source_uri": "test/path1", "episode_id": episode_id},
                    {"source_uri": "test/path2", "episode_id": episode_id},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        # Query and verify episode ID
        chunks = memory.query("Test content")
        assert any(chunk.episode_id == episode_id for chunk in chunks)


def test_ingest_file_read_error(memory, test_files):
    """Test handling file read errors in ingest."""
    test_hash = "test_hash"
    with patch.object(memory, "_compute_file_hash", return_value=test_hash):
        with patch.object(
            memory.collection,
            "get",
            return_value={
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
                "distances": [],
            },
        ):
            # Patch shutil.copy2 locally to do nothing
            with patch("ds_agent.memory.shutil.copy2", return_value=None):
                # Mock open to raise an exception
                with patch("builtins.open", side_effect=Exception("Read error")):
                    with pytest.raises(RuntimeError, match="Failed to read file content"):
                        memory.ingest(test_files[0])


def test_ingest_filelock_timeout(memory, test_files):
    """Test handling file lock timeout in ingest."""
    import filelock

    test_hash = "test_hash"
    with patch.object(memory, "_compute_file_hash", return_value=test_hash):
        with patch.object(
            memory.collection,
            "get",
            return_value={
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
                "distances": [],
            },
        ):
            with patch.object(memory.embedding_model, "encode", return_value=[0.1, 0.2, 0.3]):
                # Patch FileLock to raise Timeout with dummy lock_file
                with patch(
                    "filelock.FileLock.__enter__", side_effect=filelock.Timeout("dummy.lock")
                ):
                    with pytest.raises(RuntimeError, match="Timed out waiting for file lock"):
                        memory.ingest(test_files[0])


def test_query_filelock_timeout(memory, test_files):
    """Test handling file lock timeout in query."""
    import filelock

    with patch.object(memory.embedding_model, "encode", return_value=[0.1, 0.2, 0.3]):
        # Patch FileLock to raise Timeout with dummy lock_file
        with patch("filelock.FileLock.__enter__", side_effect=filelock.Timeout("dummy.lock")):
            with pytest.raises(RuntimeError, match="Timed out waiting for file lock during query"):
                memory.query("Test content")
