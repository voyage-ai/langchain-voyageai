"""Test VoyageAI multimodal embeddings integration tests."""

import pytest

from langchain_voyageai import VoyageAIMultimodalEmbeddings

# Please set VOYAGE_API_KEY in the environment variables
MULTIMODAL_MODEL = "voyage-multimodal-3"
MULTIMODAL_MODEL_WITH_VIDEO = "voyage-multimodal-3.5"


def test_langchain_voyageai_multimodal_embedding_documents() -> None:
    """Test multimodal embeddings with text documents."""
    documents = ["foo bar"]
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


def test_langchain_voyageai_multimodal_embedding_documents_multiple() -> None:
    """Test multimodal embeddings with multiple text documents."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


def test_langchain_voyageai_multimodal_embedding_query() -> None:
    """Test multimodal embeddings for query."""
    document = "foo bar"
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    output = embedding.embed_query(document)
    assert len(output) == 1024


async def test_langchain_voyageai_async_multimodal_embedding_documents() -> None:
    """Test async multimodal embeddings with text documents."""
    documents = ["foo bar", "bar foo"]
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024


async def test_langchain_voyageai_async_multimodal_embedding_query() -> None:
    """Test async multimodal embeddings for query."""
    document = "foo bar"
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    output = await embedding.aembed_query(document)
    assert len(output) == 1024


def test_langchain_voyageai_multimodal_embed_multimodal_text_only() -> None:
    """Test embed_multimodal with text-only inputs."""
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    inputs = [
        ["This is a text about cats"],
        ["Another text about dogs"],
    ]
    output = embedding.embed_multimodal(inputs)
    assert len(output) == 2
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024


async def test_langchain_voyageai_async_multimodal_embed_multimodal_text_only() -> None:
    """Test async embed_multimodal with text-only inputs."""
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    inputs = [
        ["This is a text about cats"],
        ["Another text about dogs"],
    ]
    output = await embedding.aembed_multimodal(inputs)
    assert len(output) == 2
    assert len(output[0]) == 1024


def test_langchain_voyageai_multimodal_embedding_consistency() -> None:
    """Test that same text produces same embedding."""
    text = "consistency test text"
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)

    output1 = embedding.embed_query(text)
    output2 = embedding.embed_query(text)

    # Same text should produce identical embeddings
    assert output1 == output2


def test_langchain_voyageai_multimodal_3_5_text_embedding() -> None:
    """Test voyage-multimodal-3.5 model with text (this model also supports video)."""
    documents = ["foo bar"]
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL_WITH_VIDEO)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


def test_langchain_voyageai_multimodal_embed_from_dicts_text() -> None:
    """Test embed_multimodal_from_dicts with text content."""
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    inputs = [
        {
            "content": [
                {"type": "text", "text": "A description of something"},
            ]
        },
        {
            "content": [
                {"type": "text", "text": "Another description"},
            ]
        },
    ]
    output = embedding.embed_multimodal_from_dicts(inputs)
    assert len(output) == 2
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024


async def test_langchain_voyageai_async_multimodal_embed_from_dicts_text() -> None:
    """Test async embed_multimodal_from_dicts with text content."""
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    inputs = [
        {
            "content": [
                {"type": "text", "text": "A description of something"},
            ]
        },
    ]
    output = await embedding.aembed_multimodal_from_dicts(inputs)
    assert len(output) == 1
    assert len(output[0]) == 1024


def test_langchain_voyageai_multimodal_empty_list() -> None:
    """Test multimodal embedding with empty list."""
    documents: list[str] = []
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    output = embedding.embed_documents(documents)

    assert len(output) == 0
    assert isinstance(output, list)


def test_langchain_voyageai_multimodal_single_document() -> None:
    """Test multimodal embedding single document."""
    documents = ["single document"]
    embedding = VoyageAIMultimodalEmbeddings(model=MULTIMODAL_MODEL)
    output = embedding.embed_documents(documents)

    assert len(output) == 1
    assert len(output[0]) == 1024


@pytest.mark.parametrize("model", [MULTIMODAL_MODEL, MULTIMODAL_MODEL_WITH_VIDEO])
def test_langchain_voyageai_multimodal_different_models(model: str) -> None:
    """Test different multimodal models."""
    documents = ["test text"]
    embedding = VoyageAIMultimodalEmbeddings(model=model)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


@pytest.mark.parametrize("model", [MULTIMODAL_MODEL, MULTIMODAL_MODEL_WITH_VIDEO])
async def test_langchain_voyageai_async_multimodal_different_models(
    model: str,
) -> None:
    """Test async different multimodal models."""
    documents = ["test text"]
    embedding = VoyageAIMultimodalEmbeddings(model=model)
    output = await embedding.aembed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024
