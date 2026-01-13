"""Test multimodal embedding model integration."""

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from langchain_voyageai import VoyageAIMultimodalEmbeddings


def test_initialization_multimodal_3() -> None:
    """Test multimodal embedding model initialization."""
    emb = VoyageAIMultimodalEmbeddings(
        api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-multimodal-3",
    )
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-multimodal-3"
    assert emb._client is not None


def test_initialization_multimodal_3_5() -> None:
    """Test multimodal 3.5 embedding model initialization (supports video)."""
    emb = VoyageAIMultimodalEmbeddings(
        api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-multimodal-3.5",
    )
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-multimodal-3.5"
    assert emb._client is not None


def test_initialization_with_api_key_alias() -> None:
    """Test initialization using api_key alias."""
    emb = VoyageAIMultimodalEmbeddings(
        api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-multimodal-3",
    )
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-multimodal-3"


def test_initialization_with_truncation_disabled() -> None:
    """Test initialization with truncation disabled."""
    emb = VoyageAIMultimodalEmbeddings(
        api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-multimodal-3",
        truncation=False,
    )
    assert emb.truncation is False


def test_default_model() -> None:
    """Test default model is voyage-multimodal-3."""
    emb = VoyageAIMultimodalEmbeddings(
        api_key=SecretStr("NOT_A_VALID_KEY"),
    )
    assert emb.model == "voyage-multimodal-3"


def test_default_truncation() -> None:
    """Test default truncation is True."""
    emb = VoyageAIMultimodalEmbeddings(
        api_key=SecretStr("NOT_A_VALID_KEY"),
    )
    assert emb.truncation is True


def test_embeddings_interface_implementation() -> None:
    """Test that VoyageAIMultimodalEmbeddings implements Embeddings interface."""
    emb = VoyageAIMultimodalEmbeddings(
        api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-multimodal-3",
    )
    # Check that required Embeddings methods exist
    assert hasattr(emb, "embed_documents")
    assert hasattr(emb, "embed_query")
    assert hasattr(emb, "aembed_documents")
    assert hasattr(emb, "aembed_query")
    assert callable(emb.embed_documents)
    assert callable(emb.embed_query)
    assert callable(emb.aembed_documents)
    assert callable(emb.aembed_query)


def test_multimodal_methods_exist() -> None:
    """Test that multimodal-specific methods exist."""
    emb = VoyageAIMultimodalEmbeddings(
        api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-multimodal-3.5",
    )
    # Check multimodal-specific methods
    assert hasattr(emb, "embed_multimodal")
    assert hasattr(emb, "aembed_multimodal")
    assert hasattr(emb, "embed_multimodal_from_dicts")
    assert hasattr(emb, "aembed_multimodal_from_dicts")
    assert callable(emb.embed_multimodal)
    assert callable(emb.aembed_multimodal)
    assert callable(emb.embed_multimodal_from_dicts)
    assert callable(emb.aembed_multimodal_from_dicts)
