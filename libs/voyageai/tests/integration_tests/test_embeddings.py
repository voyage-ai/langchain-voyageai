"""Test VoyageAI embeddings."""

import pytest

from langchain_voyageai import VoyageAIEmbeddings

# Please set VOYAGE_API_KEY in the environment variables
MODEL = "voyage-2"
CONTEXT_MODEL = "voyage-context-3"


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_langchain_voyageai_embedding_documents(model: str) -> None:
    """Test voyage embeddings."""
    documents = ["foo bar"]
    embedding = VoyageAIEmbeddings(model=model)  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_langchain_voyageai_embedding_documents_multiple(model: str) -> None:
    """Test voyage embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = VoyageAIEmbeddings(model=model, batch_size=2)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_langchain_voyageai_embedding_query(model: str) -> None:
    """Test voyage embeddings."""
    document = "foo bar"
    embedding = VoyageAIEmbeddings(model=model)  # type: ignore[call-arg]
    output = embedding.embed_query(document)
    assert len(output) == 1024


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
async def test_langchain_voyageai_async_embedding_documents_multiple(
    model: str,
) -> None:
    """Test voyage embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = VoyageAIEmbeddings(model=model, batch_size=2)
    output = await embedding.aembed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
async def test_langchain_voyageai_async_embedding_query(model: str) -> None:
    """Test voyage embeddings."""
    document = "foo bar"
    embedding = VoyageAIEmbeddings(model=model)  # type: ignore[call-arg]
    output = await embedding.aembed_query(document)
    assert len(output) == 1024


def test_langchain_voyageai_embedding_documents_with_output_dimension() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar"]
    embedding = VoyageAIEmbeddings(model="voyage-3-large", output_dimension=256)  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 256


def test_langchain_voyageai_contextual_embedding_documents() -> None:
    """Test contextual voyage embeddings for documents."""
    documents = ["foo bar", "baz qux"]
    embedding = VoyageAIEmbeddings(model="voyage-context-3")  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024  # Default embedding dimension
    assert len(output[1]) == 1024


def test_langchain_voyageai_contextual_embedding_query() -> None:
    """Test contextual voyage embeddings for query."""
    query = "foo bar"
    embedding = VoyageAIEmbeddings(model="voyage-context-3")  # type: ignore[call-arg]
    output = embedding.embed_query(query)
    assert len(output) == 1024


def test_langchain_voyageai_contextual_embedding_with_output_dimension() -> None:
    """Test contextual voyage embeddings with custom output dimension."""
    documents = ["foo bar"]
    embedding = VoyageAIEmbeddings(model="voyage-context-3", output_dimension=512)  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 512


async def test_langchain_voyageai_async_contextual_embedding_documents() -> None:
    """Test async contextual voyage embeddings for documents."""
    documents = ["foo bar", "baz qux", "hello world"]
    embedding = VoyageAIEmbeddings(model="voyage-context-3")  # type: ignore[call-arg]
    output = await embedding.aembed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


async def test_langchain_voyageai_async_contextual_embedding_query() -> None:
    """Test async contextual voyage embeddings for query."""
    query = "foo bar"
    embedding = VoyageAIEmbeddings(model="voyage-context-3")  # type: ignore[call-arg]
    output = await embedding.aembed_query(query)
    assert len(output) == 1024


def test_langchain_voyageai_contextual_embedding_realistic_documents() -> None:
    """Test contextual voyage embeddings with realistic document context."""
    documents = [
        "The Mediterranean diet emphasizes fish, olive oil, "
        "and vegetables, believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into "
        "glucose and produces essential oxygen.",
        "Apple's conference call to discuss fourth fiscal quarter "
        "results is scheduled for Thursday, November 2, 2023.",
    ]
    embedding = VoyageAIEmbeddings(model="voyage-context-3")  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert all(len(emb) == 1024 for emb in output)
    # Verify embeddings are different (not all zeros or identical)
    assert output[0] != output[1]
    assert output[1] != output[2]


def test_langchain_voyageai_contextual_embedding_query_with_context() -> None:
    """Test contextual voyage embeddings for query with realistic context."""
    query = "When is Apple's conference call scheduled?"
    embedding = VoyageAIEmbeddings(model="voyage-context-3")  # type: ignore[call-arg]
    output = embedding.embed_query(query)
    assert len(output) == 1024
    # Verify embedding is not all zeros
    assert any(val != 0.0 for val in output)


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_langchain_voyageai_embedding_with_small_batch_size(model: str) -> None:
    """Test embedding with small batch size to verify batching works."""
    documents = ["text1", "text2", "text3", "text4", "text5"]
    embedding = VoyageAIEmbeddings(model=model, batch_size=2)
    output = embedding.embed_documents(documents)

    # Should successfully embed all documents despite small batch size
    assert len(output) == 5
    assert all(len(emb) == 1024 for emb in output)
    # Verify embeddings are unique
    assert output[0] != output[1]


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
async def test_langchain_voyageai_async_embedding_with_small_batch_size(
    model: str,
) -> None:
    """Test async embedding with small batch size."""
    documents = ["text1", "text2", "text3", "text4", "text5"]
    embedding = VoyageAIEmbeddings(model=model, batch_size=2)
    output = await embedding.aembed_documents(documents)

    assert len(output) == 5
    assert all(len(emb) == 1024 for emb in output)


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_langchain_voyageai_embedding_large_texts(model: str) -> None:
    """Test embedding with large texts that might hit token limits."""
    # Create texts with varying sizes
    documents = [
        "Short text.",
        "This is a much longer text that contains many more words and should "
        "consume significantly more tokens than the previous short text. " * 10,
        "Another short one.",
        "Yet another long text with lots of repeated content to increase token count. "
        * 10,
    ]
    embedding = VoyageAIEmbeddings(model=model, batch_size=10)
    output = embedding.embed_documents(documents)

    assert len(output) == 4
    assert all(len(emb) == 1024 for emb in output)


def test_langchain_voyageai_contextual_embedding_with_batching() -> None:
    """Test contextual embeddings handle batching correctly."""
    documents = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
    embedding = VoyageAIEmbeddings(model="voyage-context-3", batch_size=3)
    output = embedding.embed_documents(documents)

    assert len(output) == 6
    assert all(len(emb) == 1024 for emb in output)
    # Verify each embedding is unique
    unique_embeddings = {tuple(emb) for emb in output}
    assert len(unique_embeddings) == 6


async def test_langchain_voyageai_async_contextual_embedding_with_batching() -> None:
    """Test async contextual embeddings handle batching correctly."""
    documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    embedding = VoyageAIEmbeddings(model="voyage-context-3", batch_size=2)
    output = await embedding.aembed_documents(documents)

    assert len(output) == 5
    assert all(len(emb) == 1024 for emb in output)


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_langchain_voyageai_embedding_empty_list(model: str) -> None:
    """Test embedding with empty list."""
    documents: list[str] = []
    embedding = VoyageAIEmbeddings(model=model, batch_size=72)
    output = embedding.embed_documents(documents)

    assert len(output) == 0
    assert isinstance(output, list)


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_langchain_voyageai_embedding_single_document(model: str) -> None:
    """Test embedding single document."""
    documents = ["single document"]
    embedding = VoyageAIEmbeddings(model=model, batch_size=72)
    output = embedding.embed_documents(documents)

    assert len(output) == 1
    assert len(output[0]) == 1024


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_langchain_voyageai_embedding_consistency(model: str) -> None:
    """Test that same text produces same embedding."""
    text = "consistency test text"
    embedding = VoyageAIEmbeddings(model=model, batch_size=72)

    output1 = embedding.embed_query(text)
    output2 = embedding.embed_query(text)

    # Same text should produce identical embeddings
    assert output1 == output2


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_langchain_voyageai_automatic_batching_with_long_texts(model: str) -> None:
    """Test automatic batching with long texts exceeding token limits."""
    # Context models have smaller context window (32k tokens)
    # Regular models have larger limits (320k tokens for voyage-2)
    # Create texts sized appropriately for each model type
    if "context" in model:
        # For context models: use shorter texts that fit in 32k window
        # but still trigger batching due to total token limits
        long_text = "This is a document with content for context model testing. " * 200
    else:
        # For regular models: use much longer texts
        long_text = (
            "This is a very long document with lots of content that will consume "
            "many tokens when processed by the embedding model. " * 1000
        )

    nr_of_texts = 100

    # Create multiple long documents
    # With default batch_size=1000 but token limits, this should create multiple batches
    documents = [f"Document {i}: {long_text}" for i in range(nr_of_texts)]

    # Use default batch_size (1000) - batching should happen due to token limits
    embedding = VoyageAIEmbeddings(model=model)

    # This should make at least 2 API calls due to token limit batching
    output = embedding.embed_documents(documents)

    # Verify all documents were embedded
    assert len(output) == nr_of_texts
    assert all(len(emb) == 1024 for emb in output)

    # Verify embeddings are unique (not all the same)
    unique_embeddings = {tuple(emb) for emb in output}
    assert len(unique_embeddings) == nr_of_texts

    # Each document should have different embeddings due to different content
    assert output[0] != output[1]
    assert output[1] != output[2]


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
def test_langchain_voyageai_batching_multiple_batches_created(model: str) -> None:
    """Test that multiple batches are created when token limits are exceeded."""
    if "context" in model:
        long_text = "This is a document with content for context model testing. " * 200
    else:
        long_text = (
            "This is a very long document with lots of content that will consume "
            "many tokens when processed by the embedding model. " * 1000
        )

    nr_of_texts = 100
    documents = [f"Document {i}: {long_text}" for i in range(nr_of_texts)]
    embedding = VoyageAIEmbeddings(model=model)

    # Count the number of batches created by _build_batches
    batches = list(embedding._build_batches(documents))
    batch_count = len(batches)

    # Verify multiple batches were created
    assert batch_count >= 2, f"Expected at least 2 batches, got {batch_count}"
    print(f"✓ Created {batch_count} batches for {nr_of_texts} documents ({model})")

    # Verify total texts across all batches equals input
    total_texts = sum(batch_size for _, batch_size in batches)
    assert (
        total_texts == nr_of_texts
    ), f"Expected {nr_of_texts} total texts, got {total_texts}"

    # Verify each batch has texts
    for i, (batch_texts, batch_size) in enumerate(batches):
        assert len(batch_texts) == batch_size, f"Batch {i} size mismatch"
        assert batch_size > 0, f"Batch {i} is empty"


@pytest.mark.parametrize("model", [MODEL, CONTEXT_MODEL])
async def test_langchain_voyageai_async_automatic_batching_with_long_texts(
    model: str,
) -> None:
    """Test async automatic batching with long texts exceeding limits."""
    # Context models have smaller context window (32k tokens)
    # Regular models have larger limits (320k tokens for voyage-2)
    if "context" in model:
        # For context models: use shorter texts that fit in 32k window
        long_text = "This is a document with content for context model testing. " * 200
    else:
        # For regular models: use much longer texts
        long_text = (
            "This is a very long document with lots of content that will consume "
            "many tokens when processed by the embedding model. " * 1000
        )

    documents = [f"Document {i}: {long_text}" for i in range(5)]

    # Use default batch_size - batching should happen due to token limits
    embedding = VoyageAIEmbeddings(model=model)

    # This should make at least 2 API calls due to token limit batching
    output = await embedding.aembed_documents(documents)

    # Verify all documents were embedded
    assert len(output) == 5
    assert all(len(emb) == 1024 for emb in output)

    # Verify embeddings are unique
    unique_embeddings = {tuple(emb) for emb in output}
    assert len(unique_embeddings) == 5
