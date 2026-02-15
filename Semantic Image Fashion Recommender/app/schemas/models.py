from pydantic import BaseModel
from typing import List, Dict, Optional

class SearchResult(BaseModel):
    product_id: str
    score: float
    image_score: float
    text_score: float
    category: str
    filename: str
    sources: List[str]
class SearchResponse(BaseModel):
    success: bool
    query_type: str
    total_results: int
    results: List[SearchResult]

class HealthResponse(BaseModel):
    status: str
    device: str
    image_model: str
    text_model: str
    image_embedding_dim: int
    text_embedding_dim: int
    pinecone_namespace: str
    image_index: str
    text_index: str

class IndexStats(BaseModel):
    name: str
    total_vectors: int
    dimension: int
    namespaces: Dict[str, Dict]

class StatsResponse(BaseModel):
    success: bool
    image_index: IndexStats
    text_index: IndexStats