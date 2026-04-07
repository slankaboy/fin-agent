"""
RAG (Retrieval-Augmented Generation) engine for Fin-Agent.

Uses sentence-transformers for local embeddings and Milvus Lite (via pymilvus)
as the vector store. No external server needed — data persists in a local .db file.

DB file: ~/.config/fin-agent/rag.db
"""

import os
import hashlib
from typing import List, Dict, Tuple

from fin_agent.config import Config

_CHUNK_SIZE = 500       # characters per chunk
_CHUNK_OVERLAP = 100    # overlap between consecutive chunks
_TOP_K = 5              # default number of chunks to retrieve
_COLLECTION = "fin_agent_rag"
_DIM = 384              # paraphrase-multilingual-MiniLM-L12-v2 output dim
_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# _MODEL_NAME = "BAAI/bge-m3"

# Lazy-loaded singletons
_model = None
_client = None


def _get_db_path() -> str:
    return os.path.join(Config.get_config_dir(), "rag.db")


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for RAG. "
                "Install it with: pip install sentence-transformers"
            )
        import os
        # Use hf-mirror for first-time download in environments without direct HuggingFace access.
        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # Once the model is cached locally, force offline mode to skip network checks.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _get_client():
    global _client
    if _client is None:
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError(
                "pymilvus is required for RAG. "
                "Install it with: pip install pymilvus"
            )
        db_path = _get_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _client = MilvusClient(db_path)
        _ensure_collection(_client)
    return _client


def _ensure_collection(client):
    """Create the collection if it doesn't exist."""
    from pymilvus import MilvusClient, DataType

    if client.has_collection(_COLLECTION):
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=_DIM)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=2048)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="FLAT",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=_COLLECTION,
        schema=schema,
        index_params=index_params,
    )


def _chunk_text(text: str, source: str) -> List[Dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_SIZE
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append({
                "source": source,
                "text": chunk_text[:2048],  # Milvus VARCHAR limit
                "chunk_id": hashlib.md5(f"{source}:{start}".encode()).hexdigest(),
            })
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────

def index_document(text: str, source: str) -> Tuple[int, int]:
    """
    Index a document into Milvus.

    :param text: Full document text
    :param source: Source label (e.g. filename)
    :returns: (new_chunks_added, total_chunks_in_collection)
    """
    client = _get_client()
    model = _get_model()

    # Fetch existing chunk_ids for this source to avoid duplicates
    existing = client.query(
        collection_name=_COLLECTION,
        filter=f'source == "{source}"',
        output_fields=["chunk_id"],
    )
    existing_ids = {r["chunk_id"] for r in existing}

    chunks = _chunk_text(text, source)
    to_add = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not to_add:
        total = client.get_collection_stats(_COLLECTION)["row_count"]
        return 0, total

    texts = [c["text"] for c in to_add]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    rows = [
        {
            "embedding": embeddings[i],
            "source": to_add[i]["source"],
            "chunk_id": to_add[i]["chunk_id"],
            "text": to_add[i]["text"],
        }
        for i in range(len(to_add))
    ]
    client.insert(collection_name=_COLLECTION, data=rows)

    total = client.get_collection_stats(_COLLECTION)["row_count"]
    return len(to_add), total


def search(query: str, top_k: int = _TOP_K) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a query.

    :param query: Natural language query
    :param top_k: Number of top results to return
    :returns: List of dicts with keys: source, text, score
    """
    client = _get_client()
    model = _get_model()

    stats = client.get_collection_stats(_COLLECTION)
    if stats["row_count"] == 0:
        return []

    query_emb = model.encode([query], show_progress_bar=False)[0].tolist()

    results = client.search(
        collection_name=_COLLECTION,
        data=[query_emb],
        limit=top_k,
        output_fields=["source", "text"],
        search_params={"metric_type": "COSINE"},
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "source": hit["entity"]["source"],
            "text": hit["entity"]["text"],
            "score": round(hit["distance"], 4),
        })
    return hits


def list_indexed_sources() -> List[Dict]:
    """Return a summary of all indexed sources with chunk counts."""
    client = _get_client()

    all_rows = client.query(
        collection_name=_COLLECTION,
        filter="",
        output_fields=["source"],
    )
    counts: Dict[str, int] = {}
    for row in all_rows:
        s = row["source"]
        counts[s] = counts.get(s, 0) + 1
    return [{"source": s, "chunks": n} for s, n in counts.items()]


def delete_source(source: str) -> int:
    """Remove all chunks belonging to a source. Returns number of chunks removed."""
    client = _get_client()

    existing = client.query(
        collection_name=_COLLECTION,
        filter=f'source == "{source}"',
        output_fields=["id"],
    )
    if not existing:
        return 0

    ids = [r["id"] for r in existing]
    client.delete(collection_name=_COLLECTION, ids=ids)
    return len(ids)
