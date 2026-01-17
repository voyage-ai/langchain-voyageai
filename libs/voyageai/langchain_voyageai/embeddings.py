import logging
from typing import Any, Generator, Iterable, List, Literal, Optional, Tuple, cast

import voyageai  # type: ignore
from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

logger = logging.getLogger(__name__)

VOYAGE_TOTAL_TOKEN_LIMITS = {
    "voyage-context-3": 32_000,
    "voyage-4-lite": 1_000_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-4": 320_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
    "voyage-4-large": 120_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-large-2": 120_000,
    "voyage-3": 120_000,
    "voyage-3-lite": 120_000,
    "voyage-code-2": 120_000,
    "voyage-3-m-exp": 120_000,
}


class VoyageAIEmbeddings(BaseModel, Embeddings):
    """VoyageAIEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_voyageai import VoyageAIEmbeddings

            model = VoyageAIEmbeddings()
    """

    _client: voyageai.Client = PrivateAttr()
    _aclient: voyageai.client_async.AsyncClient = PrivateAttr()
    model: str
    batch_size: Optional[int] = 1000

    output_dimension: Optional[Literal[256, 512, 1024, 2048]] = None
    show_progress_bar: bool = False
    truncation: bool = True
    voyage_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "VOYAGE_API_KEY",
            error_message="Must set `VOYAGE_API_KEY` environment variable or "
            "pass `api_key` to VoyageAIEmbeddings constructor.",
        ),
    )
    base_url: Optional[str] = None
    """Custom API endpoint URL. If not provided, the VoyageAI SDK determines
    the default based on the API key."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that VoyageAI credentials exist in environment."""
        api_key_str = self.voyage_api_key.get_secret_value()
        self._client = voyageai.Client(api_key=api_key_str, base_url=self.base_url)
        self._aclient = voyageai.client_async.AsyncClient(
            api_key=api_key_str, base_url=self.base_url
        )
        return self

    # Public API - Sync
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        if self._is_context_model():
            return self._embed_context(texts, "document")
        return self._embed_regular(texts, "document")

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        if self._is_context_model():
            result = self._embed_context([text], "query")
        else:
            result = self._embed_regular([text], "query")
        return result[0]

    # Public API - Async
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed search docs."""
        if self._is_context_model():
            return await self._aembed_context(texts, "document")
        return await self._aembed_regular(texts, "document")

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query text."""
        if self._is_context_model():
            result = await self._aembed_context([text], "query")
        else:
            result = await self._aembed_regular([text], "query")
        return result[0]

    # Helpers
    def _is_context_model(self) -> bool:
        """Check if the model is a contextualized embedding model."""
        return "context" in self.model

    def init_progress_bar(self, texts_len: int) -> Any:
        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore

                return tqdm(total=texts_len, desc="Encoding sentences")
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e
        else:
            return None

    def _build_batches(
        self, texts: List[str]
    ) -> Generator[Tuple[List[str], int], None, None]:
        """Generate batches of texts based on token limits."""
        max_tokens_per_batch = VOYAGE_TOTAL_TOKEN_LIMITS.get(self.model, 120_000)
        index = 0

        while index < len(texts):
            batch: List[str] = []
            batch_tokens = 0
            while (
                index < len(texts)
                and len(batch) < (self.batch_size or 1000)
                and batch_tokens < max_tokens_per_batch
            ):
                n_tokens = len(
                    self._client.tokenize([texts[index]], model=self.model)[0]
                )
                if batch_tokens + n_tokens > max_tokens_per_batch and len(batch) > 0:
                    break
                batch_tokens += n_tokens
                batch.append(texts[index])
                index += 1

            yield batch, len(batch)

    # Sync embedding implementation
    def _batch_embed(
        self, texts: List[str], input_type: str, embed_fn: Any
    ) -> List[List[float]]:
        """Common batching logic for embedding."""
        embeddings: List[List[float]] = []
        pbar = self.init_progress_bar(len(texts))

        for batch, batch_size in self._build_batches(texts):
            batch_embeddings = embed_fn(batch, input_type)
            embeddings.extend(cast(Iterable[List[float]], batch_embeddings))

            if pbar:
                pbar.update(batch_size)

        if pbar:
            pbar.close()

        return embeddings

    def _embed_context(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Embed using contextualized embedding API."""

        def embed_fn(batch: List[str], inp_type: str) -> List[List[float]]:
            r = self._client.contextualized_embed(
                inputs=[batch],
                model=self.model,
                input_type=inp_type,
                output_dimension=self.output_dimension,
            ).results
            return cast(List[List[float]], r[0].embeddings)

        return self._batch_embed(texts, input_type, embed_fn)

    def _embed_regular(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Embed using regular embedding API."""

        def embed_fn(batch: List[str], inp_type: str) -> List[List[float]]:
            return cast(
                List[List[float]],
                self._client.embed(
                    batch,
                    model=self.model,
                    input_type=inp_type,
                    truncation=self.truncation,
                    output_dimension=self.output_dimension,
                ).embeddings,
            )

        return self._batch_embed(texts, input_type, embed_fn)

    # Async embedding implementation
    async def _abatch_embed(
        self, texts: List[str], input_type: str, embed_fn: Any
    ) -> List[List[float]]:
        """Common async batching logic for embedding."""
        embeddings: List[List[float]] = []
        pbar = self.init_progress_bar(len(texts))

        for batch, batch_size in self._build_batches(texts):
            batch_embeddings = await embed_fn(batch, input_type)
            embeddings.extend(cast(Iterable[List[float]], batch_embeddings))

            if pbar:
                pbar.update(batch_size)

        if pbar:
            pbar.close()

        return embeddings

    async def _aembed_context(
        self, texts: List[str], input_type: str
    ) -> List[List[float]]:
        """Async embed using contextualized embedding API."""

        async def embed_fn(batch: List[str], inp_type: str) -> List[List[float]]:
            r = await self._aclient.contextualized_embed(
                inputs=[batch],
                model=self.model,
                input_type=inp_type,
                output_dimension=self.output_dimension,
            )
            return cast(List[List[float]], r.results[0].embeddings)

        return await self._abatch_embed(texts, input_type, embed_fn)

    async def _aembed_regular(
        self, texts: List[str], input_type: str
    ) -> List[List[float]]:
        """Async embed using regular embedding API."""

        async def embed_fn(batch: List[str], inp_type: str) -> List[List[float]]:
            r = await self._aclient.embed(
                batch,
                model=self.model,
                input_type=inp_type,
                truncation=self.truncation,
                output_dimension=self.output_dimension,
            )
            return cast(List[List[float]], r.embeddings)

        return await self._abatch_embed(texts, input_type, embed_fn)
