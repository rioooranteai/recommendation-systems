import asyncio
import logging
from typing import List, Dict, Optional, Any, Union

from config import Config
from pinecone import Pinecone, PineconeAsyncio
from pinecone.grpc import PineconeGRPC

logger = logging.getLogger(__name__)


class PineconeService:
    """Synchronous Pinecone service using gRPC for vector operations.

    Uses two separate clients:
        - ``PineconeGRPC`` for high-throughput upsert/query operations.
        - ``Pinecone`` (REST) for the Inference (rerank) API.

    Manages two indexes for the two-stage retrieval strategy:
        - Image index : SigLIP embeddings (768-dim).
        - Text index  : BGE-M3 embeddings (1024-dim).
    """

    def __init__(self) -> None:
        """Initialize dual Pinecone clients and both vector indexes."""

        self.pc = PineconeGRPC(api_key=Config.PINECONE_API_KEY)
        self.pc_inference = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.image_index = self.pc.Index(Config.PINECONE_IMAGE_INDEX_NAME)
        self.text_index = self.pc.Index(Config.PINECONE_TEXT_INDEX_NAME)

        logger.info("PineconeService (Sync) initialized with Dual Clients (GRPC + REST)")

    def upsert_images(
            self,
            vectors: List[tuple],
            namespace: str = Config.PINECONE_NAMESPACE,
    ):
        """Upsert image vectors into the image index (768-dim).

        Args:
            vectors: List of (id, values, metadata) tuples to upsert.
            namespace: Pinecone namespace to write into.

        Returns:
            Pinecone upsert response object.

        Raises:
            Exception: Re-raises any Pinecone client exception after logging.
        """
        try:
            response = self.image_index.upsert(
                vectors=vectors,
                namespace=namespace,
            )
            return response
        except Exception as e:
            logger.error("Image upsert failed: %s", e)
            raise

    def query_images(
            self,
            vector: List[float],
            top_k: int = Config.TOP_K,
            filter: Optional[Dict] = None,
            namespace: str = Config.PINECONE_NAMESPACE,
    ):
        """Query the image index (768-dim) for nearest neighbours.

        Args:
            vector: Query embedding as a flat list of floats.
            top_k: Number of top results to retrieve.
            filter: Optional Pinecone metadata filter dict.
            namespace: Pinecone namespace to query.

        Returns:
            Pinecone query response object containing ``.matches``.

        Raises:
            Exception: Re-raises any Pinecone client exception after logging.
        """
        try:
            result = self.image_index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True,
            )
            return result
        except Exception as e:
            logger.error("Image query failed: %s", e)
            raise

    def upsert_text(
            self,
            vectors: List[tuple],
            namespace: str = Config.PINECONE_NAMESPACE,
    ):
        """Upsert text vectors into the text index (1024-dim).

        Args:
            vectors: List of (id, values, metadata) tuples to upsert.
            namespace: Pinecone namespace to write into.

        Returns:
            Pinecone upsert response object.

        Raises:
            Exception: Re-raises any Pinecone client exception after logging.
        """
        try:
            response = self.text_index.upsert(
                vectors=vectors,
                namespace=namespace,
            )
            return response
        except Exception as e:
            logger.error("Text upsert failed: %s", e)
            raise

    def query_text(
            self,
            vector: List[float],
            top_k: int = Config.TOP_K,
            filter: Optional[Dict] = None,
            namespace: str = Config.PINECONE_NAMESPACE,
    ):
        """Query the text index (1024-dim) for nearest neighbours.

        Args:
            vector: Query embedding as a flat list of floats.
            top_k: Number of top results to retrieve.
            filter: Optional Pinecone metadata filter dict.
            namespace: Pinecone namespace to query.

        Returns:
            Pinecone query response object containing ``.matches``.

        Raises:
            Exception: Re-raises any Pinecone client exception after logging.
        """
        try:
            result = self.text_index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True,
            )
            return result
        except Exception as e:
            logger.error("Text query failed: %s", e)
            raise

    def rerank(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            top_n: int = 10,
    ):
        """Rerank documents against a query using the Pinecone Inference API.

        Args:
            query: The search query string used for reranking.
            documents: List of dicts, each containing a ``"text"`` field.
            top_n: Number of top reranked results to return.

        Returns:
            Pinecone rerank response object containing ``.data``.

        Raises:
            Exception: Re-raises any Pinecone client exception after logging.
        """
        try:
            results = self.pc_inference.inference.rerank(
                model=Config.PINECONE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_n,
                rank_fields=["text"],
                return_documents=True,
                parameters={"truncate": "END"},
            )
            return results
        except Exception as e:
            logger.error("Rerank failed: %s", e)
            raise

    def delete_all_images(self, namespace: str = Config.PINECONE_NAMESPACE) -> None:
        """Delete all vectors from the image index in the given namespace.

        Args:
            namespace: Pinecone namespace to clear.
        """
        self.image_index.delete(delete_all=True, namespace=namespace)

    def delete_all_text(self, namespace: str = Config.PINECONE_NAMESPACE) -> None:
        """Delete all vectors from the text index in the given namespace.

        Args:
            namespace: Pinecone namespace to clear.
        """
        self.text_index.delete(delete_all=True, namespace=namespace)

    def delete_all(self, namespace: str = Config.PINECONE_NAMESPACE) -> None:
        """Delete all vectors from both image and text indexes.

        Args:
            namespace: Pinecone namespace to clear across both indexes.
        """
        self.delete_all_images(namespace)
        self.delete_all_text(namespace)
        logger.info("Deleted all vectors from both indexes")


class PineconeAsyncService:
    """Asynchronous Pinecone service using ``PineconeAsyncio`` for vector operations.

    Must be initialized via ``await service.initialize()`` before use,
    and cleaned up via ``await service.close()`` when done.

    The Pinecone Inference (rerank) API is synchronous in the current SDK;
    rerank calls are offloaded to a thread pool via ``asyncio.to_thread``
    to prevent blocking the event loop.
    """

    def __init__(self) -> None:
        """Declare instance attributes; call ``initialize()`` to connect."""
        self.pc: Optional[PineconeAsyncio] = None
        self.pc_inference: Optional[Pinecone] = None
        self.image_index = None
        self.text_index = None

    async def initialize(self) -> None:
        """Connect to Pinecone and bind both vector indexes.

        Must be awaited before calling any other method on this service.
        """
        # Async client for non-blocking vector upsert/query operations.
        self.pc = PineconeAsyncio(api_key=Config.PINECONE_API_KEY)

        # Sync REST client for Inference (rerank) — Pinecone SDK is synchronous here.
        self.pc_inference = Pinecone(api_key=Config.PINECONE_API_KEY)

        self.image_index = self.pc.Index(Config.PINECONE_IMAGE_INDEX_NAME)
        self.text_index = self.pc.Index(Config.PINECONE_TEXT_INDEX_NAME)

        logger.info("PineconeService (Async) initialized")

    async def close(self) -> None:
        """Close all open async index connections and the async client.

        ``pc_inference`` does not require explicit closing as it is a
        stateless REST client.
        """
        if self.image_index:
            await self.image_index.close()
        if self.text_index:
            await self.text_index.close()
        if self.pc:
            await self.pc.close()

    async def upsert_images(
            self,
            vectors: List[tuple],
            namespace: str = Config.PINECONE_NAMESPACE,
    ):
        """Asynchronously upsert image vectors into the image index (768-dim).

        Args:
            vectors: List of (id, values, metadata) tuples to upsert.
            namespace: Pinecone namespace to write into.

        Returns:
            Pinecone upsert response object.

        Raises:
            Exception: Re-raises any Pinecone client exception after logging.
        """
        try:
            response = await self.image_index.upsert(
                vectors=vectors,
                namespace=namespace,
            )
            return response
        except Exception as e:
            logger.error("Async image upsert failed: %s", e)
            raise

    async def query_images(
            self,
            vector: List[float],
            top_k: int = Config.TOP_K,
            filter: Optional[Dict] = None,
            namespace: str = Config.PINECONE_NAMESPACE,
    ):
        """Asynchronously query the image index (768-dim) for nearest neighbours.

        Args:
            vector: Query embedding as a flat list of floats.
            top_k: Number of top results to retrieve.
            filter: Optional Pinecone metadata filter dict.
            namespace: Pinecone namespace to query.

        Returns:
            Pinecone query response object containing ``.matches``.

        Raises:
            Exception: Re-raises any Pinecone client exception after logging.
        """
        try:
            results = await self.image_index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True,
            )
            return results
        except Exception as e:
            logger.error("Async image query failed: %s", e)
            raise

    async def upsert_text(
            self,
            vectors: List[tuple],
            namespace: str = Config.PINECONE_NAMESPACE,
    ):
        """Asynchronously upsert text vectors into the text index (1024-dim).

        Args:
            vectors: List of (id, values, metadata) tuples to upsert.
            namespace: Pinecone namespace to write into.

        Returns:
            Pinecone upsert response object.

        Raises:
            Exception: Re-raises any Pinecone client exception after logging.
        """
        try:
            response = await self.text_index.upsert(
                vectors=vectors,
                namespace=namespace,
            )
            return response
        except Exception as e:
            logger.error("Async text upsert failed: %s", e)
            raise

    async def query_text(
            self,
            vector: List[float],
            top_k: int = Config.TOP_K,
            filter: Optional[Dict] = None,
            namespace: str = Config.PINECONE_NAMESPACE,
    ):
        """Asynchronously query the text index (1024-dim) for nearest neighbours.

        Args:
            vector: Query embedding as a flat list of floats.
            top_k: Number of top results to retrieve.
            filter: Optional Pinecone metadata filter dict.
            namespace: Pinecone namespace to query.

        Returns:
            Pinecone query response object containing ``.matches``.

        Raises:
            Exception: Re-raises any Pinecone client exception after logging.
        """
        try:
            results = await self.text_index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True,
            )
            return results
        except Exception as e:
            logger.error("Async text query failed: %s", e)
            raise

    async def rerank(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            top_n: int = 10,
    ):
        """Rerank documents asynchronously using the Pinecone Inference API.

        The underlying SDK call is synchronous; it is offloaded to a thread
        pool via ``asyncio.to_thread`` to avoid blocking the event loop.

        Args:
            query: The search query string used for reranking.
            documents: List of dicts, each containing a ``"text"`` field.
            top_n: Number of top reranked results to return.

        Returns:
            Pinecone rerank response object containing ``.data``.

        Raises:
            Exception: Re-raises any Pinecone client exception after logging.
        """

        def _do_rerank_sync():
            return self.pc_inference.inference.rerank(
                model=Config.PINECONE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=False,
            )

        try:
            results = await asyncio.to_thread(_do_rerank_sync)
            return results
        except Exception as e:
            logger.error("Async rerank failed: %s", e)
            raise


def get_pinecone_service(
        async_mode: bool = Config.USE_ASYNC,
) -> Union[PineconeService, PineconeAsyncService]:
    """Factory function to instantiate the appropriate Pinecone service.

    Args:
        async_mode: If ``True``, returns a ``PineconeAsyncService`` instance.
            If ``False``, returns a ``PineconeService`` (sync/gRPC) instance.
            Defaults to ``Config.USE_ASYNC``.

    Returns:
        A ``PineconeAsyncService`` if ``async_mode`` is ``True``,
        otherwise a ``PineconeService``.
    """
    if async_mode:
        return PineconeAsyncService()
    return PineconeService()
