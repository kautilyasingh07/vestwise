"""sentence-transformers wrapper (spec FR-4, §6 tech stack).

The model is loaded once per process (lru_cache acts as a module-level
singleton) because loading takes about a second and holds ~90 MB of weights.
Vectors are L2-normalised, so cosine similarity equals the dot product.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

BATCH_SIZE = 32


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model named in settings (first call only)."""
    return SentenceTransformer(settings.embedding_model)


def count_tokens(text: str) -> int:
    """Number of model tokens in text, excluding the [CLS]/[SEP] markers the model adds."""
    return len(get_model().tokenizer.encode(text, add_special_tokens=False))


def max_input_tokens() -> int:
    """Longest text (in tokens, excluding special tokens) the model embeds without truncating."""
    return get_model().max_seq_length - 2  # room for [CLS] and [SEP]


def embedding_dim() -> int:
    """Length of each embedding vector (384 for all-MiniLM-L6-v2)."""
    return get_model().get_sentence_embedding_dimension()


def embed(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """Embed texts in batches; return one unit-length vector (list of floats) per text."""
    if not texts:
        return []
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
