# langchain-voyageai

This package contains the LangChain integrations for VoyageAI through their `voyageai` client package.

## Installation and Setup

- Install the LangChain partner package
```bash
pip install langchain-voyageai
```
- Get an VoyageAI api key and set it as an environment variable (`VOYAGE_API_KEY`) or use the API key as a parameter in the Client.



## Text Embedding Model

See a [usage example](https://python.langchain.com/docs/integrations/text_embedding/voyageai)

```python
from langchain_voyageai import VoyageAIEmbeddings
```

### Voyage-4 Family Models

The latest generation of VoyageAI embedding models with improved quality and flexibility:

```python
# voyage-4: Balanced model for general-purpose and multilingual retrieval
embeddings = VoyageAIEmbeddings(model="voyage-4")

# voyage-4-lite: Optimized for latency and cost, highest batch throughput (1M tokens/batch)
embeddings = VoyageAIEmbeddings(model="voyage-4-lite")

# voyage-4-large: Best retrieval quality for demanding applications
embeddings = VoyageAIEmbeddings(model="voyage-4-large")
```

All voyage-4 family models support flexible output dimensions (256, 512, 1024, 2048):

```python
embeddings = VoyageAIEmbeddings(
    model="voyage-4",
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
