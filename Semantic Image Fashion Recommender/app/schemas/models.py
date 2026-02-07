from typing import Optional, List
from pydantic import BaseModel, Field
class SearchResult(BaseModel):
    """Single search result item"""
    product_id: str = Field(..., description="Unique product identifier")
    score: float = Field(..., description="Similarity score (0-1)")
    category: str = Field(..., description="Product category")
    filename: str = Field(..., description="Image filename")
    image_score: Optional[float] = Field(None, description="Image similarity score")
    text_score: Optional[float] = Field(None, description="Text similarity score")


class SearchResponse(BaseModel):
    """Search API response"""
    success: bool = Field(..., description="Whether search was successful")
    query_type: str = Field(..., description="Type of search performed")
    total_results: int = Field(..., description="Number of results returned")
    results: List[SearchResult] = Field(..., description="List of search results")


class TextSearchRequest(BaseModel):
    """Request body for text-only search"""
    text_query: str = Field(..., min_length=1, description="Text query for search")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    category: Optional[str] = Field(None, description="Filter by category")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    device: str = Field(..., description="Compute device (CPU/CUDA)")
    model: str = Field(..., description="Embedding model name")
    embedding_dim: int = Field(..., description="Embedding dimension")
    pinecone_namespace: str = Field(..., description="Pinecone namespace")
    pinecone_index: str = Field(..., description="Pinecone index name")


class StatsResponse(BaseModel):
    """Index statistics response"""
    success: bool
    total_vectors: int
    dimension: int
    namespaces: dict