"""Test embedding model integration."""

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from langchain_voyageai import VoyageAIEmbeddings

MODEL = "voyage-2"


def test_initialization_voyage_2() -> None:
    """Test embedding model initialization."""
    emb = VoyageAIEmbeddings(voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model=MODEL)  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == 1000
    assert emb.model == MODEL
    assert emb._client is not None


def test_initialization_voyage_2_with_full_api_key_name() -> None:
    """Test embedding model initialization."""
    # Testing that we can initialize the model using `voyage_api_key`
    # instead of `api_key`
    emb = VoyageAIEmbeddings(voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model=MODEL)  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == 1000
    assert emb.model == MODEL
    assert emb._client is not None


def test_initialization_voyage_1() -> None:
    """Test embedding model initialisation."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model="voyage-01"
    )  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == 1000
    assert emb.model == "voyage-01"
    assert emb._client is not None


def test_initialization_voyage_1_batch_size() -> None:
    """Test embedding model initialization."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-01",
        batch_size=15,
    )
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == 15
    assert emb.model == "voyage-01"
    assert emb._client is not None


def test_initialization_with_output_dimension() -> None:
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-3-large",
        output_dimension=256,
        batch_size=10,
    )
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-3-large"
    assert emb.output_dimension == 256


def test_initialization_contextual_model() -> None:
    """Test initialization with contextual embedding model."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model="voyage-context-3"
    )  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-context-3"
    assert emb.batch_size == 1000  # Default batch size
    assert emb._client is not None


def test_initialization_contextual_model_with_custom_batch_size() -> None:
    """Test initialization of contextual model with custom batch size."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-context-3",
        batch_size=5,
    )
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-context-3"
    assert emb.batch_size == 5
    assert emb._client is not None


def test_initialization_contextual_model_with_output_dimension() -> None:
    """Test initialization of contextual model with output dimension."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model="voyage-context-3",
        output_dimension=512,
    )
    assert isinstance(emb, Embeddings)
    assert emb.model == "voyage-context-3"
    assert emb.output_dimension == 512
    assert emb._client is not None


def test_is_context_model_detection() -> None:
    """Test contextual model detection."""
    # Contextual model
    emb_context = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model="voyage-context-3"
    )  # type: ignore
    assert emb_context._is_context_model() is True

    # Regular model
    emb_regular = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model="voyage-3"
    )  # type: ignore
    assert emb_regular._is_context_model() is False

    # Another regular model
    emb_regular2 = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model="voyage-2"
    )  # type: ignore
    assert emb_regular2._is_context_model() is False


def test_contextual_model_variants() -> None:
    """Test different contextual model variants."""
    context_models = [
        "voyage-context-3",
        "voyage-context-lite",
        "custom-context-model",
    ]

    for model in context_models:
        emb = VoyageAIEmbeddings(
            voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model=model
        )  # type: ignore
        assert emb._is_context_model() is True, (
            f"Model {model} should be detected as contextual"
        )


def test_initialization_with_base_url() -> None:
    """Test embedding model initialization with custom base_url."""
    custom_url = "https://custom.example.com/v1"
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        base_url=custom_url,
    )  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.base_url == custom_url
    assert emb._client is not None


def test_initialization_without_base_url() -> None:
    """Test embedding model initialization without base_url (default behavior)."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
    )  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.base_url is None
    assert emb._client is not None


def test_initialization_with_none_base_url() -> None:
    """Test embedding model initialization with explicit None base_url."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        base_url=None,
    )  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.base_url is None
    assert emb._client is not None


def test_build_batches_basic() -> None:
    """Test basic batch building."""
    from unittest.mock import Mock

    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        batch_size=2,
    )  # type: ignore

    # Mock tokenize to return predictable token counts
    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    texts = ["text1", "text2", "text3", "text4"]
    batches = list(emb._build_batches(texts))

    # Should create 2 batches of 2 texts each
    assert len(batches) == 2
    assert batches[0] == (["text1", "text2"], 2)
    assert batches[1] == (["text3", "text4"], 2)


def test_build_batches_token_limit() -> None:
    """Test batch building respects token limits."""
    from unittest.mock import Mock

    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        batch_size=10,  # Small batch size to ensure batch_size limit
    )  # type: ignore

    # Mock tokenize to return large token counts that exceed limits
    # Each text has 200k tokens, so with 320k limit, only 1 text per batch
    mock_tokenize = Mock(return_value=[[1] * 200000])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    texts = ["text1", "text2", "text3"]
    batches = list(emb._build_batches(texts))

    # Token limit for voyage-2 is 320,000
    # Each text has 200k tokens, so should create 3 separate batches
    assert len(batches) == 3
    assert batches[0] == (["text1"], 1)
    assert batches[1] == (["text2"], 1)
    assert batches[2] == (["text3"], 1)


def test_build_batches_single_text() -> None:
    """Test batch building with single text."""
    from unittest.mock import Mock

    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        batch_size=10,
    )  # type: ignore

    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    texts = ["single text"]
    batches = list(emb._build_batches(texts))

    assert len(batches) == 1
    assert batches[0] == (["single text"], 1)


def test_build_batches_empty_list() -> None:
    """Test batch building with empty list."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
    )  # type: ignore

    texts: list[str] = []
    batches = list(emb._build_batches(texts))

    assert len(batches) == 0


def test_init_progress_bar_disabled() -> None:
    """Test progress bar initialization when disabled."""
    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        show_progress_bar=False,
    )  # type: ignore

    pbar = emb.init_progress_bar(100)
    assert pbar is None


def test_init_progress_bar_enabled() -> None:
    """Test progress bar initialization when enabled."""
    try:
        import tqdm  # type: ignore[import-untyped]  # noqa: F401

        emb = VoyageAIEmbeddings(
            voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
            model="voyage-2",
            show_progress_bar=True,
        )  # type: ignore

        pbar = emb.init_progress_bar(100)
        assert pbar is not None
        pbar.close()
    except ImportError:
        # Skip test if tqdm not installed
        pass


def test_voyage_4_family_initialization() -> None:
    """Test voyage-4 family model initialization."""
    voyage_4_models = ["voyage-4", "voyage-4-lite", "voyage-4-large"]

    for model in voyage_4_models:
        emb = VoyageAIEmbeddings(
            voyage_api_key=SecretStr("NOT_A_VALID_KEY"), model=model
        )  # type: ignore
        assert isinstance(emb, Embeddings)
        assert emb.model == model
        assert emb.batch_size == 1000
        assert emb._client is not None
        # Verify voyage-4 models are NOT detected as contextual
        assert emb._is_context_model() is False, (
            f"Model {model} should NOT be detected as contextual"
        )


def test_voyage_4_token_limits_in_registry() -> None:
    """Test voyage-4 family models have correct token limits in registry."""
    from langchain_voyageai.embeddings import VOYAGE_TOTAL_TOKEN_LIMITS

    # Verify all voyage-4 models are in the registry with correct limits
    assert "voyage-4" in VOYAGE_TOTAL_TOKEN_LIMITS
    assert VOYAGE_TOTAL_TOKEN_LIMITS["voyage-4"] == 320_000

    assert "voyage-4-lite" in VOYAGE_TOTAL_TOKEN_LIMITS
    assert VOYAGE_TOTAL_TOKEN_LIMITS["voyage-4-lite"] == 1_000_000

    assert "voyage-4-large" in VOYAGE_TOTAL_TOKEN_LIMITS
    assert VOYAGE_TOTAL_TOKEN_LIMITS["voyage-4-large"] == 120_000


def test_init_progress_bar_missing_tqdm() -> None:
    """Test progress bar raises error when tqdm missing."""
    import sys
    from unittest.mock import patch

    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        show_progress_bar=True,
    )  # type: ignore

    # Mock tqdm import failure
    with patch.dict(sys.modules, {"tqdm": None, "tqdm.auto": None}):
        try:
            emb.init_progress_bar(100)
            assert False, "Should have raised ImportError"
        except ImportError as e:
            assert "tqdm" in str(e)


def test_batch_embed_basic() -> None:
    """Test _batch_embed with mocked embed function."""
    from typing import List
    from unittest.mock import Mock

    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        batch_size=2,
    )  # type: ignore

    # Mock tokenize and embed function
    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    def mock_embed_fn(batch: List[str], input_type: str) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in batch]

    texts = ["text1", "text2", "text3"]
    result = emb._batch_embed(texts, "document", mock_embed_fn)

    assert len(result) == 3
    assert result[0] == [0.1, 0.2, 0.3]


def test_batch_embed_calls_embed_fn_with_correct_args() -> None:
    """Test _batch_embed passes correct arguments to embed_fn."""
    from unittest.mock import Mock

    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        batch_size=2,
    )  # type: ignore

    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]
    mock_embed_fn = Mock(return_value=[[0.1, 0.2]])

    texts = ["text1", "text2"]
    emb._batch_embed(texts, "query", mock_embed_fn)

    # Verify embed_fn was called with correct input_type
    mock_embed_fn.assert_called_once()
    args = mock_embed_fn.call_args
    assert args[0][0] == ["text1", "text2"]  # batch
    assert args[0][1] == "query"  # input_type


async def test_abatch_embed_basic() -> None:
    """Test _abatch_embed with mocked async embed function."""
    from typing import List
    from unittest.mock import Mock

    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        batch_size=2,
    )  # type: ignore

    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    async def mock_embed_fn(batch: List[str], input_type: str) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in batch]

    texts = ["text1", "text2", "text3"]
    result = await emb._abatch_embed(texts, "document", mock_embed_fn)

    assert len(result) == 3
    assert result[0] == [0.1, 0.2, 0.3]


def test_embed_context_vs_regular_routing() -> None:
    """Test that context models route to correct embed method."""
    from unittest.mock import Mock

    # Test context model routing
    emb_context = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-context-3",
        batch_size=1,
    )  # type: ignore

    mock_tokenize_context = Mock(return_value=[[1, 2, 3]])
    emb_context._client.tokenize = mock_tokenize_context  # type: ignore[method-assign]
    mock_contextualized_embed = Mock(
        return_value=Mock(results=[Mock(embeddings=[[0.1, 0.2, 0.3]])])
    )
    emb_context._client.contextualized_embed = mock_contextualized_embed  # type: ignore[method-assign]

    # Should use contextualized_embed for context models
    result = emb_context._embed_context(["test"], "document")
    assert mock_contextualized_embed.called
    assert len(result) == 1

    # Test regular model routing
    emb_regular = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        batch_size=1,
    )  # type: ignore

    mock_tokenize_regular = Mock(return_value=[[1, 2, 3]])
    emb_regular._client.tokenize = mock_tokenize_regular  # type: ignore[method-assign]
    mock_embed = Mock(return_value=Mock(embeddings=[[0.1, 0.2, 0.3]]))
    emb_regular._client.embed = mock_embed  # type: ignore[method-assign]

    # Should use regular embed for non-context models
    result = emb_regular._embed_regular(["test"], "document")
    assert mock_embed.called
    assert len(result) == 1


async def test_aembed_context_vs_regular_routing() -> None:
    """Test that async context models route to correct embed method."""
    from unittest.mock import AsyncMock, Mock

    # Test async context model routing
    emb_context = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-context-3",
        batch_size=1,
    )  # type: ignore

    mock_tokenize = Mock(return_value=[[1, 2, 3]])
    emb_context._client.tokenize = mock_tokenize  # type: ignore[method-assign]
    mock_async_contextualized_embed = AsyncMock(
        return_value=Mock(results=[Mock(embeddings=[[0.1, 0.2, 0.3]])])
    )
    emb_context._aclient.contextualized_embed = mock_async_contextualized_embed  # type: ignore[method-assign]

    result = await emb_context._aembed_context(["test"], "document")
    assert mock_async_contextualized_embed.called
    assert len(result) == 1


def test_automatic_batching_due_to_token_limits() -> None:
    """Test that batching happens automatically when token limits are exceeded."""
    from typing import List
    from unittest.mock import Mock

    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
        # Use default batch_size (1000) - batching should happen due to token limits
    )  # type: ignore

    # Mock tokenize to return large token counts
    # Each text has 100k tokens, so with 320k limit for voyage-2,
    # we can fit max 3 texts per batch, but we have 5 texts
    mock_tokenize = Mock(return_value=[[1] * 100000])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    call_count = 0

    def mock_embed_fn(batch: List[str], input_type: str) -> List[List[float]]:
        nonlocal call_count
        call_count += 1
        # Return embeddings for each text in the batch
        return [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(len(batch))]

    texts = ["text1", "text2", "text3", "text4", "text5"]
    result = emb._batch_embed(texts, "document", mock_embed_fn)

    # Verify all texts were embedded
    assert len(result) == 5

    # Verify multiple batches were created due to token limits
    # With 100k tokens per text and 320k limit:
    # Batch 1: text1, text2, text3 (300k tokens)
    # Batch 2: text4, text5 (200k tokens)
    assert call_count >= 2, f"Expected at least 2 API calls, got {call_count}"


async def test_async_automatic_batching_due_to_token_limits() -> None:
    """Test async batching happens automatically when token limits are exceeded."""
    from typing import List
    from unittest.mock import Mock

    emb = VoyageAIEmbeddings(
        voyage_api_key=SecretStr("NOT_A_VALID_KEY"),
        model="voyage-2",
    )  # type: ignore

    # Mock tokenize to return large token counts
    mock_tokenize = Mock(return_value=[[1] * 100000])
    emb._client.tokenize = mock_tokenize  # type: ignore[method-assign]

    call_count = 0

    async def mock_embed_fn(batch: List[str], input_type: str) -> List[List[float]]:
        nonlocal call_count
        call_count += 1
        return [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(len(batch))]

    texts = ["text1", "text2", "text3", "text4", "text5"]
    result = await emb._abatch_embed(texts, "document", mock_embed_fn)

    # Verify all texts were embedded
    assert len(result) == 5

    # Verify multiple batches were created due to token limits
    assert call_count >= 2, f"Expected at least 2 API calls, got {call_count}"
