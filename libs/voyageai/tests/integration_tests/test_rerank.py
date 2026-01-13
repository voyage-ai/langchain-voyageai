"""Test the voyageai reranker."""

import os

from langchain_core.documents import Document

from langchain_voyageai.rerank import VoyageAIRerank


def test_voyageai_reranker_init() -> None:
    """Test the voyageai reranker initializes correctly."""
    VoyageAIRerank(voyage_api_key="foo", model="foo")  # type: ignore[arg-type]


def test_voyageai_reranker_init_with_base_url() -> None:
    """Test the voyageai reranker initializes correctly with base_url."""
    VoyageAIRerank(
        voyage_api_key="foo",  # type: ignore[arg-type]
        model="foo",
        base_url="https://api.voyageai.com/v1",
    )


def test_sync() -> None:
    rerank = VoyageAIRerank(
        voyage_api_key=os.environ["VOYAGE_API_KEY"],  # type: ignore[arg-type]
        model="rerank-lite-1",
    )
    doc_list = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables"
        ", believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into glucose and "
        "produces essential oxygen.",
        "20th-century innovations, from radios to smartphones, centered "
        "on electronic advancements.",
        "Rivers provide water, irrigation, and habitat for aquatic species, "
        "vital for ecosystems.",
        "Apple’s conference call to discuss fourth fiscal quarter results and "
        "business updates is scheduled for Thursday, November 2, 2023 at 2:00 "
        "p.m. PT / 5:00 p.m. ET.",
        "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' "
        "endure in literature.",
    ]
    documents = [Document(page_content=x) for x in doc_list]

    result = rerank.compress_documents(
        query="When is the Apple's conference call scheduled?", documents=documents
    )
    assert len(doc_list) == len(result)
    for doc in result:
        assert "total_tokens" in doc.metadata
        assert isinstance(doc.metadata["total_tokens"], int)
        assert doc.metadata["total_tokens"] > 0


async def test_async() -> None:
    rerank = VoyageAIRerank(
        voyage_api_key=os.environ["VOYAGE_API_KEY"],  # type: ignore[arg-type]
        model="rerank-lite-1",
    )
    doc_list = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables"
        ", believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into glucose and "
        "produces essential oxygen.",
        "20th-century innovations, from radios to smartphones, centered "
        "on electronic advancements.",
        "Rivers provide water, irrigation, and habitat for aquatic species, "
        "vital for ecosystems.",
        "Apple’s conference call to discuss fourth fiscal quarter results and "
        "business updates is scheduled for Thursday, November 2, 2023 at 2:00 "
        "p.m. PT / 5:00 p.m. ET.",
        "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' "
        "endure in literature.",
    ]
    documents = [Document(page_content=x) for x in doc_list]

    result = await rerank.acompress_documents(
        query="When is the Apple's conference call scheduled?", documents=documents
    )
    assert len(doc_list) == len(result)
    for doc in result:
        assert "total_tokens" in doc.metadata
        assert isinstance(doc.metadata["total_tokens"], int)
        assert doc.metadata["total_tokens"] > 0
