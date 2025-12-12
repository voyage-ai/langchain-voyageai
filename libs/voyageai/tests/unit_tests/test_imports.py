from langchain_voyageai import __all__

EXPECTED_ALL = [
    "VoyageAIEmbeddings",
    "VoyageAIMultimodalEmbeddings",
    "VoyageAIRerank",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
