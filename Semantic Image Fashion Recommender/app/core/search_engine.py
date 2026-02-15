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
            alpha: float = 0.5  # 0=text only, 1=image only, 0.5=balanced
    ) -> List[Dict]:
        """
        Two-stage hybrid search:
        - Query both image index (768-dim) and text index (1024-dim)
        - Fuse results using weighted scoring
        """

        # Validation
        if image is None and text_query is None:
            raise ValueError("Must provide at least one of: image or text_query")

        # Collect candidates from both indexes
        image_candidates = []
        text_candidates = []

        # Stage 1: Retrieve from Image Index
        if image:
            image_candidates = self._retrieve_from_image_index(
                image=image,
                top_k=50,  # Retrieve more for better fusion
                filters=filters
            )
            logger.info(f"Retrieved {len(image_candidates)} image candidates")

        # Stage 2: Retrieve from Text Index
        if text_query:
            text_candidates = self._retrieve_from_text_index(
                text_query=text_query,
                top_k=50,
                filters=filters
            )
            logger.info(f"Retrieved {len(text_candidates)} text candidates")

        # Stage 3: Fuse results
        if image_candidates and text_candidates:
            # Hybrid search: fuse both
            results = self._fuse_results(
                image_candidates=image_candidates,
                text_candidates=text_candidates,
                alpha=alpha
            )
        elif image_candidates:
            # Image-only search
            results = self._format_results(image_candidates, 'image')
        else:
            # Text-only search
            results = self._format_results(text_candidates, 'text')

        # Return top K
        return results[:top_k]

    def _retrieve_from_image_index(
            self,
            image: Image.Image,
            top_k: int,
            filters: Optional[Dict] = None
    ) -> List:
        """Query image index (768-dim SigLIP vectors)"""

        # Encode image with SigLIP
        img_embedding = self.embedding_service.encode_images(image)
        img_vector = self._to_list(img_embedding)

        # Query image index
        results = self.pinecone_service.query_images(
            vector=img_vector,
            top_k=top_k,
            filter=filters,
            namespace=self.namespace
        )

        return results.matches

    def _retrieve_from_text_index(
            self,
            text_query: str,
            top_k: int,
            filters: Optional[Dict] = None
    ) -> List:
        """Query text index (1024-dim BGE-M3 vectors)"""

        # Encode text with BGE-M3
        txt_embedding = self.embedding_service.encode_text(text_query)
        txt_vector = self._to_list(txt_embedding)

        # Query text index
        results = self.pinecone_service.query_text(
            vector=txt_vector,
            top_k=top_k,
            filter=filters,
            namespace=self.namespace
        )

        return results.matches

    def _fuse_results(
            self,
            image_candidates: List,
            text_candidates: List,
            alpha: float
    ) -> List[Dict]:
        """
        Fuse results from image and text indexes using RRF (Reciprocal Rank Fusion)
        alpha: 0=text only, 1=image only, 0.5=balanced
        """

        product_scores = {}
        k = 60  # RRF constant

        # Process image results
        for rank, match in enumerate(image_candidates):
            product_id = match.metadata['product_id']

            # RRF score: weight / (k + rank)
            rrf_score = alpha / (k + rank)

            if product_id not in product_scores:
                product_scores[product_id] = {
                    'product_id': product_id,
                    'score': 0,
                    'image_score': match.score,
                    'text_score': 0,
                    'category': match.metadata.get('category', 'unknown'),
                    'filename': match.metadata.get('filename', ''),
                    'sources': []
                }

            product_scores[product_id]['score'] += rrf_score
            product_scores[product_id]['image_score'] = match.score
            product_scores[product_id]['sources'].append('image')

        # Process text results
        for rank, match in enumerate(text_candidates):
            product_id = match.metadata['product_id']

            # RRF score: weight / (k + rank)
            rrf_score = (1 - alpha) / (k + rank)

            if product_id not in product_scores:
                product_scores[product_id] = {
                    'product_id': product_id,
                    'score': 0,
                    'image_score': 0,
                    'text_score': match.score,
                    'category': match.metadata.get('category', 'unknown'),
                    'filename': match.metadata.get('filename', ''),
                    'sources': []
                }

            product_scores[product_id]['score'] += rrf_score
            product_scores[product_id]['text_score'] = match.score
            product_scores[product_id]['sources'].append('text')

        # Sort by fused score
        results = sorted(
            product_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        return results

    def _format_results(self, matches: List, source: str) -> List[Dict]:
        """Format results from single index"""
        results = []
        for match in matches:
            results.append({
                'product_id': match.metadata['product_id'],
                'score': match.score,
                'image_score': match.score if source == 'image' else 0,
                'text_score': match.score if source == 'text' else 0,
                'category': match.metadata.get('category', 'unknown'),
                'filename': match.metadata.get('filename', ''),
                'sources': [source]
            })

        return results

    def _to_list(self, arr) -> List[float]:
        """Convert array to list"""
        if hasattr(arr, 'detach'):
            arr = arr.detach().cpu().numpy()

        if hasattr(arr, 'ndim') and arr.ndim == 2:
            arr = arr.flatten()

        return [float(x) for x in arr.tolist()]