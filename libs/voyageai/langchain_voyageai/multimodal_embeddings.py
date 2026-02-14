"""VoyageAI Multimodal Embeddings with support for text, images, and videos."""

import logging
from typing import Any, Dict, List, Optional

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
from voyageai.video_utils import Video  # type: ignore[import-not-found] # noqa: F401

logger = logging.getLogger(__name__)

# Type alias for multimodal input: each input is a list of segments
# where each segment can be text (str), image (PIL.Image.Image), or video (Video)
MultimodalInputType = List[List[Any]]


class VoyageAIMultimodalEmbeddings(BaseModel, Embeddings):
    """VoyageAI Multimodal Embeddings model with support for text, images, and videos.

    This class supports the voyage-multimodal-3 and voyage-multimodal-3.5 models,
    which can embed text, images, and videos (voyage-multimodal-3.5 only for video).

    Example:
        .. code-block:: python

            from langchain_voyageai import VoyageAIMultimodalEmbeddings
            from PIL import Image
            from voyageai.video_utils import Video

            model = VoyageAIMultimodalEmbeddings(model="voyage-multimodal-3.5")

            # Embed text only
            text_embeddings = model.embed_documents(["hello world"])

            # Embed multimodal content (text + image)
            image = Image.open("image.jpg")
            embeddings = model.embed_multimodal([
                ["This is a description", image],
                ["Another text"]
            ])

            # Embed multimodal content with video (voyage-multimodal-3.5 only)
            video = Video.from_path("video.mp4", model="voyage-multimodal-3.5")
            embeddings = model.embed_multimodal([
                ["Video description", video],
            ])
    """

    _client: voyageai.Client = PrivateAttr()
    _aclient: voyageai.client_async.AsyncClient = PrivateAttr()
    model: str = "voyage-multimodal-3"
    truncation: bool = True
    voyage_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "VOYAGE_API_KEY",
            error_message="Must set `VOYAGE_API_KEY` environment variable or "
            "pass `api_key` to VoyageAIMultimodalEmbeddings constructor.",
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that VoyageAI credentials exist in environment."""
        api_key_str = self.voyage_api_key.get_secret_value()
        self._client = voyageai.Client(api_key=api_key_str)
        self._aclient = voyageai.client_async.AsyncClient(api_key=api_key_str)
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs (text only).

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embeddings, one for each text.
        """
        inputs = [[text] for text in texts]
        return self._embed_multimodal(inputs, input_type="document")

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding for the query.
        """
        result = self._embed_multimodal([[text]], input_type="query")
        return result[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed search docs (text only).

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embeddings, one for each text.
        """
        inputs = [[text] for text in texts]
        return await self._aembed_multimodal(inputs, input_type="document")

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding for the query.
        """
        result = await self._aembed_multimodal([[text]], input_type="query")
        return result[0]

    def embed_multimodal(
        self,
        inputs: MultimodalInputType,
        input_type: Optional[str] = "document",
    ) -> List[List[float]]:
        """Embed multimodal content (text, images, and/or videos).

        Each input is a list of segments where each segment can be:
        - str: text content
        - PIL.Image.Image: an image
        - voyageai.video_utils.Video: a video (voyage-multimodal-3.5 only)

        Args:
            inputs: List of multimodal inputs. Each input is a list of segments.
            input_type: Type of input, either "document" or "query".

        Returns:
            List of embeddings, one for each input.

        Example:
            .. code-block:: python

                from PIL import Image
                from voyageai.video_utils import Video

                model = VoyageAIMultimodalEmbeddings(model="voyage-multimodal-3.5")

                # Text + image
                image = Image.open("photo.jpg")
                embeddings = model.embed_multimodal([
                    ["A photo of a cat", image],
                    ["Just text"],
                ])

                # Text + video (voyage-multimodal-3.5 only)
                video = Video.from_path("clip.mp4", model="voyage-multimodal-3.5")
                embeddings = model.embed_multimodal([
                    ["A video of a dog playing", video],
                ])
        """
        return self._embed_multimodal(inputs, input_type=input_type)

    async def aembed_multimodal(
        self,
        inputs: MultimodalInputType,
        input_type: Optional[str] = "document",
    ) -> List[List[float]]:
        """Async embed multimodal content (text, images, and/or videos).

        Each input is a list of segments where each segment can be:
        - str: text content
        - PIL.Image.Image: an image
        - voyageai.video_utils.Video: a video (voyage-multimodal-3.5 only)

        Args:
            inputs: List of multimodal inputs. Each input is a list of segments.
            input_type: Type of input, either "document" or "query".

        Returns:
            List of embeddings, one for each input.
        """
        return await self._aembed_multimodal(inputs, input_type=input_type)

    def embed_multimodal_from_dicts(
        self,
        inputs: List[Dict[str, Any]],
        input_type: Optional[str] = "document",
    ) -> List[List[float]]:
        """Embed multimodal content using dictionary format.

        This method accepts inputs in the dictionary format expected by the
        VoyageAI API directly.

        Args:
            inputs: List of dicts, each with a 'content' key containing a list of
                segment dicts with 'type' and corresponding content keys.
            input_type: Type of input, either "document" or "query".

        Returns:
            List of embeddings, one for each input.

        Example:
            .. code-block:: python

                model = VoyageAIMultimodalEmbeddings(model="voyage-multimodal-3.5")
                embeddings = model.embed_multimodal_from_dicts([
                    {
                        "content": [
                            {"type": "text", "text": "A description"},
                            {"type": "image_url", "image_url": "https://..."},
                        ]
                    },
                    {
                        "content": [
                            {"type": "text", "text": "Video description"},
                            {"type": "video_base64", "video_base64": "data:..."},
                        ]
                    },
                ])
        """
        result = self._client.multimodal_embed(
            inputs=inputs,
            model=self.model,
            input_type=input_type,
            truncation=self.truncation,
        )
        return result.embeddings

    async def aembed_multimodal_from_dicts(
        self,
        inputs: List[Dict[str, Any]],
        input_type: Optional[str] = "document",
    ) -> List[List[float]]:
        """Async embed multimodal content using dictionary format.

        Args:
            inputs: List of dicts, each with a 'content' key containing a list of
                segment dicts with 'type' and corresponding content keys.
            input_type: Type of input, either "document" or "query".

        Returns:
            List of embeddings, one for each input.
        """
        result = await self._aclient.multimodal_embed(
            inputs=inputs,
            model=self.model,
            input_type=input_type,
            truncation=self.truncation,
        )
        return result.embeddings

    def _embed_multimodal(
        self,
        inputs: MultimodalInputType,
        input_type: Optional[str] = None,
    ) -> List[List[float]]:
        """Internal method to embed multimodal content."""
        result = self._client.multimodal_embed(
            inputs=inputs,
            model=self.model,
            input_type=input_type,
            truncation=self.truncation,
        )
        return result.embeddings

    async def _aembed_multimodal(
        self,
        inputs: MultimodalInputType,
        input_type: Optional[str] = None,
    ) -> List[List[float]]:
        """Internal async method to embed multimodal content."""
        result = await self._aclient.multimodal_embed(
            inputs=inputs,
            model=self.model,
            input_type=input_type,
            truncation=self.truncation,
        )
        return result.embeddings
