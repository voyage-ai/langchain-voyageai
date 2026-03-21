from __future__ import annotations

from copy import deepcopy
from typing import Optional, Sequence, Union

import voyageai  # type: ignore
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self
from voyageai.object import RerankingObject  # type: ignore


class VoyageAIRerank(BaseDocumentCompressor):
    """Document compressor that uses `VoyageAI Rerank API`."""

    client: voyageai.Client = None  # type: ignore
    aclient: voyageai.AsyncClient = None  # type: ignore
    """VoyageAI clients to use for compressing documents."""
    voyage_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "VOYAGE_API_KEY",
            error_message="Must set `VOYAGE_API_KEY` environment variable or "
            "pass `api_key` to VoyageAIRerank constructor.",
        ),
    )
    """VoyageAI API key. Must be specified directly or via environment variable
        VOYAGE_API_KEY."""
    base_url: Optional[str] = None
    """Custom API endpoint URL. If not provided, the VoyageAI SDK determines
    the default based on the API key."""
    model: str
    """Model to use for reranking."""
    top_k: Optional[int] = None
    """Number of documents to return."""
    truncation: bool = True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that VoyageAI credentials exist in environment."""
        api_key_str = self.voyage_api_key.get_secret_value()
        self.client = voyageai.Client(api_key=api_key_str, base_url=self.base_url)
        self.aclient = voyageai.AsyncClient(api_key=api_key_str, base_url=self.base_url)
        return self

    def _rerank(
        self,
        documents: Sequence[Union[str, Document]],
        query: str,
    ) -> RerankingObject:
        """Returns an ordered list of documents ordered by their relevance
        to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
        """
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        return self.client.rerank(
            query=query,
            documents=docs,
            model=self.model,
            top_k=self.top_k,
            truncation=self.truncation,
        )

    async def _arerank(
        self,
        documents: Sequence[Union[str, Document]],
        query: str,
    ) -> RerankingObject:
        """Returns an ordered list of documents ordered by their relevance
        to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
        """
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        return await self.aclient.rerank(
            query=query,
            documents=docs,
            model=self.model,
            top_k=self.top_k,
            truncation=self.truncation,
        )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using VoyageAI's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents in relevance_score order.
            Each document's metadata includes 'relevance_score' and 'total_tokens'.
        """
        if len(documents) == 0:
            return []

        rerank_result = self._rerank(documents, query)
        compressed = []
        for res in rerank_result.results:
            doc = documents[res.index]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res.relevance_score
            doc_copy.metadata["total_tokens"] = rerank_result.total_tokens
            compressed.append(doc_copy)
        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using VoyageAI's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents in relevance_score order.
            Each document's metadata includes 'relevance_score' and 'total_tokens'.
        """
        if len(documents) == 0:
            return []

        rerank_result = await self._arerank(documents, query)
        compressed = []
        for res in rerank_result.results:
            doc = documents[res.index]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res.relevance_score
            doc_copy.metadata["total_tokens"] = rerank_result.total_tokens
            compressed.append(doc_copy)
        return compressed
