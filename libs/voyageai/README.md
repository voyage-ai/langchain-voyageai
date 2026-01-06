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

### Custom API Endpoint

You can specify a custom API endpoint using the `base_url` parameter:

```python
embeddings = VoyageAIEmbeddings(
    model="voyage-3.5",
    base_url="https://ai.mongodb.com/v1"
)
```

This is useful for MongoDB Atlas users or custom deployments.
