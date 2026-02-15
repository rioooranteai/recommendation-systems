import logging
from typing import List, Dict, Optional

from config import Config
from pinecone import PineconeAsyncio
from pinecone.grpc import PineconeGRPC

logger = logging.getLogger(__name__)


class PineconeService:

    def __init__(self):
        self.pc = PineconeGRPC(api_key=Config.PINECONE_API_KEY)

        # Two indexes for two-stage retrieval
        self.image_index = self.pc.Index(Config.PINECONE_IMAGE_INDEX_NAME)
        self.text_index = self.pc.Index(Config.PINECONE_TEXT_INDEX_NAME)

    def upsert_images(self, vectors: List[tuple], namespace: str = Config.PINECONE_NAMESPACE):
        """Upsert image vectors to image index (768-dim)"""
        try:
            response = self.image_index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            return response
        except Exception as e:
            logger.error(f"Image upsert failed: {e}")
            raise

    def query_images(
            self,
            vector: List[float],
            top_k: int = Config.TOP_K,
            filter: Optional[Dict] = None,
            namespace: str = Config.PINECONE_NAMESPACE
    ):
        """Query image index (768-dim)"""
        try:
            result = self.image_index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True
            )
            return result
        except Exception as e:
            logger.error(f"Image query failed: {e}")
            raise

    def upsert_text(self, vectors: List[tuple], namespace: str = Config.PINECONE_NAMESPACE):
        """Upsert text vectors to text index (1024-dim)"""
        try:
            response = self.text_index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            return response
        except Exception as e:
            logger.error(f"Text upsert failed: {e}")
            raise

    def query_text(
            self,
            vector: List[float],
            top_k: int = Config.TOP_K,
            filter: Optional[Dict] = None,
            namespace: str = Config.PINECONE_NAMESPACE
    ):
        """Query text index (1024-dim)"""
        try:
            result = self.text_index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True
            )
            return result
        except Exception as e:
            logger.error(f"Text query failed: {e}")
            raise

    def delete_all_images(self, namespace: str = Config.PINECONE_NAMESPACE):
        """Delete all vectors from image index"""
        self.image_index.delete(delete_all=True, namespace=namespace)

    def delete_all_text(self, namespace: str = Config.PINECONE_NAMESPACE):
        """Delete all vectors from text index"""
        self.text_index.delete(delete_all=True, namespace=namespace)

    def delete_all(self, namespace: str = Config.PINECONE_NAMESPACE):
        """Delete all vectors from both indexes"""
        self.delete_all_images(namespace)
        self.delete_all_text(namespace)
        logger.info("Deleted all vectors from both indexes")


class PineconeAsyncService:

    def __init__(self):
        self.pc = None
        self.image_index = None
        self.text_index = None

    async def initialize(self):
        self.pc = PineconeAsyncio(api_key=Config.PINECONE_API_KEY)
        self.image_index = self.pc.Index(Config.PINECONE_IMAGE_INDEX_NAME)
        self.text_index = self.pc.Index(Config.PINECONE_TEXT_INDEX_NAME)

    async def close(self):
        if self.image_index:
            await self.image_index.close()
        if self.text_index:
            await self.text_index.close()
        if self.pc:
            await self.pc.close()

    async def upsert_images(self, vectors: List[tuple], namespace: str = Config.PINECONE_NAMESPACE):
        try:
            response = await self.image_index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            return response
        except Exception as e:
            logger.error(f"Async image upsert failed: {e}")
            raise

    async def query_images(
            self,
            vector: List[float],
            top_k: int = Config.TOP_K,
            filter: Optional[Dict] = None,
            namespace: str = Config.PINECONE_NAMESPACE
    ):
        try:
            results = await self.image_index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True
            )
            return results
        except Exception as e:
            logger.error(f"Async image query failed: {e}")
            raise

    async def upsert_text(self, vectors: List[tuple], namespace: str = Config.PINECONE_NAMESPACE):
        try:
            response = await self.text_index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            return response
        except Exception as e:
            logger.error(f"Async text upsert failed: {e}")
            raise

    async def query_text(
            self,
            vector: List[float],
            top_k: int = Config.TOP_K,
            filter: Optional[Dict] = None,
            namespace: str = Config.PINECONE_NAMESPACE
    ):
        try:
            results = await self.text_index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True
            )
            return results
        except Exception as e:
            logger.error(f"Async text query failed: {e}")
            raise


# Factory function
def get_pinecone_service(async_mode: bool = Config.USE_ASYNC):
    if async_mode:
        return PineconeAsyncService()
    else:
        return PineconeService()
