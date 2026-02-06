import logging
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
from config import Config

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, embedding_service, pinecone_service):
        self.embedding_service = embedding_service
        self.pinecone_service = pinecone_service
        self.namespace = Config.PINECONE_NAMESPACE

    def search(
            self,
            image: Optional[Image.Image] = None,
            text_query: Optional[str] = None,
            top_k: int = 10,
            filters: Optional[Dict] = None,
            image_weight: float = 0.7,
            text_weight: float = 0.3,
            use_rerank: bool = True
    ) -> List[Dict]:

        # Validation
        if image is None and text_query is None:
            raise ValueError("Must provide at least one of: image or text_query")

        # Normalize weights
        total_weight = image_weight + text_weight
        image_weight = image_weight / total_weight
        text_weight = text_weight / total_weight

        # Stage 1: Retrieve candidates
        if image:
            candidates = self._retrieve_candidates_by_image(
                image=image,
                top_k=top_k * 2 if (text_query and use_rerank) else top_k,
                filters=filters
            )
        else:
            # Text-only search (direct query to text vectors)
            candidates = self._retrieve_candidates_by_text(
                text_query=text_query,
                top_k=top_k,
                filters=filters
            )
            return self._format_results(candidates)

        # Stage 2: Rerank with text preference (if provided)
        if text_query and use_rerank:
            logger.info(f"Reranking {len(candidates)} candidates with text preference")
            results = self._rerank_with_text(
                candidates=candidates,
                text_query=text_query,
                image_weight=image_weight,
                text_weight=text_weight
            )
        else:
            results = self._format_results(candidates)

        # Return top K
        return results[:top_k]

    def _retrieve_candidates_by_image(
            self,
            image: Image.Image,
            top_k: int,
            filters: Optional[Dict] = None
    ) -> List:

        # Encode image
        img_embedding = self.embedding_service.encode_images(image)
        img_vector = self._to_list(img_embedding)

        # Build filter for image vectors only
        img_filter = {"kind": {"$eq": "img"}}
        if filters:
            img_filter.update(filters)

        # Query Pinecone
        results = self.pinecone_service.query(
            vector=img_vector,
            top_k=top_k,
            filter=img_filter,
            namespace=self.namespace
        )

        return results.matches

    def _retrieve_candidates_by_text(
            self,
            text_query: str,
            top_k: int,
            filters: Optional[Dict] = None
    ) -> List:

        # Encode text
        txt_embedding = self.embedding_service.encode_text(text_query)
        txt_vector = self._to_list(txt_embedding)

        # Build filter for text vectors only
        txt_filter = {"kind": {"$eq": "txt"}}
        if filters:
            txt_filter.update(filters)

        # Query Pinecone
        results = self.pinecone_service.query(
            vector=txt_vector,
            top_k=top_k,
            filter=txt_filter,
            namespace=self.namespace
        )

        logger.info(f"Retrieved {len(results.matches)} text candidates")
        return results.matches

    def _rerank_with_text(
            self,
            candidates: List,
            text_query: str,
            image_weight: float,
            text_weight: float
    ) -> List[Dict]:

        # Encode text query
        txt_embedding = self.embedding_service.encode_text(text_query)
        txt_query_vector = self._to_list(txt_embedding)
        txt_query_np = np.array(txt_query_vector)

        # Extract product IDs from candidates
        product_ids = [match.metadata['product_id'] for match in candidates]

        # Build text vector IDs
        txt_ids = [f"{pid}#txt" for pid in product_ids]

        # Fetch text vectors from Pinecone
        try:
            txt_vectors = self.pinecone_service.index.fetch(
                ids=txt_ids,
                namespace=self.namespace
            )
        except Exception as e:
            logger.error(f"Failed to fetch text vectors: {e}")
            # Fallback: return original candidates without rerank
            return self._format_results(candidates)

        # Compute combined scores
        reranked = []
        for match in candidates:
            product_id = match.metadata['product_id']
            image_score = match.score

            # Get text vector for this product
            txt_id = f"{product_id}#txt"
            txt_vec_data = txt_vectors.vectors.get(txt_id)

            if txt_vec_data and txt_vec_data.values:
                # Compute text similarity (cosine)
                txt_vec_np = np.array(txt_vec_data.values)
                text_score = float(np.dot(txt_query_np, txt_vec_np))
            else:
                # No text vector found, use 0
                text_score = 0.0
                logger.warning(f"Text vector not found for product {product_id}")

            # Combined score (weighted fusion)
            final_score = (image_weight * image_score) + (text_weight * text_score)

            reranked.append({
                'product_id': product_id,
                'score': final_score,
                'image_score': image_score,
                'text_score': text_score,
                'category': match.metadata.get('category', 'unknown'),
                'filename': match.metadata.get('filename', '')
            })

        # Sort by final score (descending)
        reranked.sort(key=lambda x: x['score'], reverse=True)

        return reranked

    def _format_results(self, matches: List) -> List[Dict]:
        results = []
        for match in matches:
            results.append({
                'product_id': match.metadata['product_id'],
                'score': match.score,
                'category': match.metadata.get('category', 'unknown'),
                'filename': match.metadata.get('filename', '')
            })

        return results

    def _to_list(self, arr) -> List[float]:
        if hasattr(arr, 'detach'):
            arr = arr.detach().cpu().numpy()

        if hasattr(arr, 'ndim') and arr.ndim == 2:
            arr = arr.flatten()

        return [float(x) for x in arr.tolist()]
