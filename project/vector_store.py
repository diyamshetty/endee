"""
vector_store.py — Endee vector database integration for ArXiv-RAG.
Manages index creation, upserting paper embeddings, and similarity search.
"""
import time
from typing import Any

from endee import Endee, Precision
from endee.schema import VectorItem

# MONKEYPATCH: The Endee SDK (v0.1.19) has a bug in `index.upsert` where it calls 
# `v_item.get("filter", None)` on a Pydantic BaseModel, which crashes.
if not hasattr(VectorItem, "get"):
    def _vector_item_get(self, key, default=None):
        return getattr(self, key, default)
    VectorItem.get = _vector_item_get

ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "arxiv_papers"
DIMENSION = 384        # all-MiniLM-L6-v2 output dimension
SPACE_TYPE = "cosine"
UPSERT_BATCH = 100     # upsert in batches of 100


def init_client(base_url: str = ENDEE_URL) -> Endee:
    """Connect to the local Endee server."""
    client = Endee()
    client.set_base_url(f"{base_url}/api/v1")
    print(f"[vector_store] Connected to Endee at {base_url}")
    return client


def create_or_get_index(client: Endee, index_name: str = INDEX_NAME):
    """
    Return the Endee index handle, creating the index if it does not exist.
    """
    # list_indexes() returns a list of objects, usually VectorItem with a .name attribute
    existing = client.list_indexes() or []
    existing_names = []
    
    for idx in existing:
        if isinstance(idx, str):
            existing_names.append(idx)
        elif isinstance(idx, dict) and "name" in idx:
            existing_names.append(idx["name"])
        elif hasattr(idx, "name"):
            existing_names.append(getattr(idx, "name"))
        elif hasattr(idx, "get"):
            existing_names.append(idx.get("name", ""))

    if index_name not in existing_names:
        print(f"[vector_store] Creating index '{index_name}' (dim={DIMENSION}, space={SPACE_TYPE}) ...")
        try:
            client.create_index(
                name=index_name,
                dimension=DIMENSION,
                space_type=SPACE_TYPE,
                precision=Precision.INT8,
                M=16,
                ef_con=200,
            )
            print(f"[vector_store] Index '{index_name}' created successfully.")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"[vector_store] Index '{index_name}' already exists (caught exception).")
            else:
                raise e
    else:
        print(f"[vector_store] Index '{index_name}' already exists, reusing.")

    return client.get_index(name=index_name)


def upsert_papers(
    index,
    papers: list[dict],
    embeddings: list[list[float]],
    batch_size: int = UPSERT_BATCH,
    progress_cb=None,
) -> None:
    """
    Upsert paper embeddings into the Endee index.

    Args:
        index      : Endee index handle (from create_or_get_index).
        papers     : List of paper dicts (id, title, authors, abstract, url, published).
        embeddings : Corresponding list of 384-dim vectors.
        batch_size : Number of vectors per upsert call.
        progress_cb: Optional callable(done, total) for progress reporting.
    """
    assert len(papers) == len(embeddings), "Papers and embeddings must have the same length."

    total = len(papers)
    batches = range(0, total, batch_size)

    for batch_start in batches:
        batch_end = min(batch_start + batch_size, total)
        records = []
        for i in range(batch_start, batch_end):
            p = papers[i]
            records.append(
                {
                    "id": p["id"],
                    "vector": embeddings[i],
                    "meta": {
                        "title": p["title"],
                        "abstract": p["abstract"],
                        "authors": ", ".join(p.get("authors", [])),
                        "url": p["url"],
                        "published": p.get("published", ""),
                    },
                    "filter": {
                        "published": p.get("published", ""),
                    },
                }
            )
        index.upsert(records)

        if progress_cb:
            progress_cb(batch_end, total)
        else:
            print(f"[vector_store]   Upserted {batch_end}/{total} vectors ...")

    print(f"[vector_store] All {total} vectors upserted successfully.")


def search(
    index,
    query_vector: list[float],
    top_k: int = 3,
) -> list[dict]:
    """
    Search the Endee index for the most similar papers.

    Args:
        index        : Endee index handle.
        query_vector : 384-dim query embedding.
        top_k        : Number of results to return.

    Returns:
        List of result dicts: {id, score, title, abstract, authors, url, published}
    """
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        ef=128,
        include_vectors=False,
    )

    output = []
    for r in results:
        # Endee SDK v0.1.x query() returns a dictionary
        meta = r.get("meta", {}) or {}
        output.append(
            {
                "id": r.get("id", ""),
                "score": round(float(r.get("similarity", 0.0)), 4),
                "title": meta.get("title", "Unknown Title"),
                "abstract": meta.get("abstract", ""),
                "authors": meta.get("authors", ""),
                "url": meta.get("url", ""),
                "published": meta.get("published", ""),
            }
        )
    return output


if __name__ == "__main__":
    print("=== Vector Store self-test ===")
    print("Connecting to Endee ...")
    client = init_client()

    print("Creating / getting index ...")
    index = create_or_get_index(client)

    # Insert a dummy vector
    dummy_vec = [0.0] * DIMENSION
    dummy_vec[0] = 1.0
    index.upsert(
        [
            {
                "id": "test-paper-001",
                "vector": dummy_vec,
                "meta": {
                    "title": "Test Paper",
                    "abstract": "This is a test abstract.",
                    "authors": "Test Author",
                    "url": "https://arxiv.org/abs/0000.00000",
                    "published": "2024-01-01",
                },
                "filter": {"published": "2024-01-01"},
            }
        ]
    )
    print("Upsert: OK")

    # Query
    results = search(index, dummy_vec, top_k=1)
    assert len(results) > 0, "Expected at least 1 result"
    print(f"Search returned: {results[0]['title']}")
    print("\nSelf-test passed ✓")
