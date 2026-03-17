"""
embedder.py — Generate vector embeddings for text using sentence-transformers.
Model: all-MiniLM-L6-v2 (384-dimensional vectors).
"""
import time
from typing import Generator
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazily load and cache the sentence-transformer model."""
    global _model
    if _model is None:
        print(f"[embedder] Loading model '{MODEL_NAME}' ...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[embedder] Model loaded. Embedding dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of strings to embed.
        batch_size: How many texts to process per batch.

    Returns:
        List of 384-dimensional float vectors (one per input text).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > batch_size,
        normalize_embeddings=True,   # cosine similarity: pre-normalize
    )
    return embeddings.tolist()


def embed_texts_batched(
    texts: list[str],
    batch_size: int = BATCH_SIZE,
) -> Generator[list[list[float]], None, None]:
    """
    Yield batches of embeddings (useful for streaming progress updates to a UI).

    Args:
        texts: List of strings to embed.
        batch_size: Batch size.

    Yields:
        Lists of embedding vectors, one batch at a time.
    """
    model = get_model()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        yield batch_embeddings.tolist()


if __name__ == "__main__":
    print("=== Embedder self-test ===")
    samples = [
        "Large language models have revolutionized natural language processing.",
        "Retrieval-Augmented Generation combines search with text generation.",
        "Vector databases store high-dimensional embeddings for similarity search.",
    ]
    start = time.time()
    vectors = embed_texts(samples)
    elapsed = time.time() - start

    print(f"\nEmbedded {len(samples)} texts in {elapsed:.2f}s")
    print(f"Embedding shape  : {len(vectors)} x {len(vectors[0])}")
    print(f"First vector (first 8 dims): {[round(v, 4) for v in vectors[0][:8]]}")
    print("\nSelf-test passed ✓")
