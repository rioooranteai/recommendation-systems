import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from pinecone.grpc import PineconeGRPC
from pinecone import PineconeAsyncio
from config import Config

logger = logging.getLogger(__name__)
class PineconeService:

    def __init__(self):
        self.pc = PineconeGRPC(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)

    def upsert(self, vectors: List[tuple], namespace: str = ""):
        try:
            response = self.index.upsert(
                vectors=vectors,
                namespace=namespace
            )

            return response

        except Exception as e:
            logging.error(f"Upsert Failed: {e}")


    def query(
        self,
        vector: List[tuple],
        top_k: int = Config.TOP_K,
        filter: Optional[Dict] = None,
        namespace: str = ""):

        try:

            result = self.index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata = True
            )

            return  result

        except Exception as e:
            logging.error(f"Query Failed: {e}")


    def delete_all(self, namespace: str = ""):
        self.index.delete(delete_all=True, namespace=namespace)


class PineconeAsyncService:
    def __init__(self):
        self.pc = None
        self.index = None

    async def initialize(self):
        self.pc = PineconeAsyncio(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)

    async def close(self):
        if self.index:
            await self.index.close()
        if self.pc:
            await self.pc.close()

    async def query(
            self,
            vector: List[float],
            top_k: int = Config.TOP_K,
            filter: Optional[Dict] = None,
            namespace: str = ""
    ):
        try:
            results = await self.index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True
            )
            return results
        except Exception as e:
            logger.error(f"Async query failed: {e}")
            raise

    async def upsert(self, vectors: List[tuple], namespace: str = ""):
        try:
            response = await self.index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            return response
        except Exception as e:
            logger.error(f"Async upsert failed: {e}")
            raise


# Factory function
def get_pinecone_service(async_mode: bool = Config.USE_ASYNC):
    if async_mode:
        return PineconeAsyncService()
    else:
        return PineconeService()