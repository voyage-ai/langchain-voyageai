# langchain-voyageai

This package contains the LangChain integrations for VoyageAI by MongoDB through their `voyageai` client package.

## Installation and Setup

- Install the LangChain partner package
```bash
pip install langchain-voyageai
```
- Get a VoyageAI by MongoDB api key and set it as an environment variable (`VOYAGE_API_KEY`) or use the API key as a parameter in the Client.



## Text Embedding Model

See a [usage example](https://python.langchain.com/docs/integrations/text_embedding/voyageai)

```python
from langchain_voyageai import VoyageAIEmbeddings
```

### Voyage-4 Family Models

The latest generation of VoyageAI by MongoDB embedding models with improved quality and flexibility:

```python
# voyage-4: Balanced model for general-purpose and multilingual retrieval
embeddings = VoyageAIEmbeddings(model="voyage-4")

# voyage-4-lite: Optimized for latency and cost, highest batch throughput (1M tokens/batch)
embeddings = VoyageAIEmbeddings(model="voyage-4-lite")

# voyage-4-large: Best retrieval quality for demanding applications
embeddings = VoyageAIEmbeddings(model="voyage-4-large")

# voyage-4-nano: Compact, efficiency-focused model for high-throughput workloads
embeddings = VoyageAIEmbeddings(model="voyage-4-nano")
```

All voyage-4 family models support flexible output dimensions (256, 512, 1024, 2048):

```python
embeddings = VoyageAIEmbeddings(
    model="voyage-4",
    output_dimension=512  # Choose from: 256, 512, 1024, 2048
)
```

### Code Embeddings

`voyage-code-4` is optimized for code retrieval and supports flexible output
dimensions (256, 512, 1024, 2048):

```python
# voyage-code-4: optimized for code retrieval
embeddings = VoyageAIEmbeddings(model="voyage-code-4")

embeddings = VoyageAIEmbeddings(
    model="voyage-code-4",
    output_dimension=512  # Choose from: 256, 512, 1024, 2048
)
```

### Contextualized Chunk Embeddings

`voyage-context-4` produces contextualized chunk embeddings, where each chunk is
embedded in the context of the other chunks from the same document:

```python
# voyage-context-4: contextualized chunk embeddings for document-aware retrieval
embeddings = VoyageAIEmbeddings(model="voyage-context-4")
```

It also supports flexible output dimensions (256, 512, 1024, 2048):

```python
embeddings = VoyageAIEmbeddings(
    model="voyage-context-4",
    output_dimension=512  # Choose from: 256, 512, 1024, 2048
)
```

### Custom API Endpoint

You can specify a custom API endpoint using the `base_url` parameter:

```python
embeddings = VoyageAIEmbeddings(
    model="voyage-3.5",
    base_url="https://ai.mongodb.com/v1"
)
```

This is useful for MongoDB Atlas users or custom deployments.
