import time
import logging
import concurrent.futures
from typing import List, Dict, Optional, Tuple

from PIL import Image
from config import Config

logger = logging.getLogger(__name__)

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class SearchEngine:
    """Hybrid search engine combining image and text vector retrieval.

    Supports three search modes:
        - Image-only  : queries the image index (SigLIP, 768-dim).
        - Text-only   : queries the text index (BGE-M3, 1024-dim) with reranking.
        - Hybrid      : runs both in parallel, fuses results via RRF.
    """

    def __init__(self, embedding_service, pinecone_service) -> None:
        """Initialize SearchEngine with required services.

        Args:
            embedding_service: Service responsible for encoding images and text.
            pinecone_service: Service responsible for Pinecone queries and reranking.
        """
        self.embedding_service = embedding_service
        self.pinecone_service = pinecone_service
        self.namespace = Config.PINECONE_NAMESPACE

    def search(
        self,
        image: Optional[Image.Image] = None,
        text_query: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        alpha: float = 0.5,
    ) -> List[Dict]:
        """Run a two-stage hybrid search with optional parallel execution.

        Modes:
            - Image-only : queries image index (SigLIP, 768-dim).
            - Text-only  : queries text index (BGE-M3, 1024-dim) with reranking.
            - Hybrid     : both run in parallel, fused via RRF.

        Args:
            image: Optional PIL image for image-based retrieval.
            text_query: Optional text string for text-based retrieval.
            top_k: Number of final results to return.
            filters: Optional Pinecone metadata filters.
            alpha: RRF fusion weight. 0 = text only, 1 = image only, 0.5 = balanced.

        Returns:
            List of result dicts, each containing product_id, score,
            image_score, text_score, category, filename, and sources.

        Raises:
            ValueError: If neither image nor text_query is provided.
        """
        if image is None and text_query is None:
            raise ValueError("Must provide at least one of: image or text_query")

        total_start = time.perf_counter()
        image_candidates: List = []
        text_candidates: List = []
        search_type: str = "Unknown"

        if image is not None and text_query is not None:
            search_type = "Hybrid"
            results = self._run_hybrid_search(image, text_query, filters, alpha)

        elif image is not None:
            search_type = "Image-only"
            image_candidates, img_latency = self._timed_image_retrieval(
                image, Config._RERANK_TOP_N, filters
            )
            logger.info(
                "[Latency] Image Search: "
                "Retrieved %d candidates in %.4fs",
                len(image_candidates), img_latency,
            )
            results = self._format_results(image_candidates, "image")

        else:
            search_type = "Text-only"
            text_candidates, txt_latency = self._timed_text_retrieval(
                text_query, Config._RERANK_TOP_N, filters
            )
            logger.info(
                "[Latency] Text Search + Reranker: "
                "Retrieved %d candidates in %.4fs",
                len(text_candidates), txt_latency,
            )
            results = self._format_results(text_candidates, "text")

        total_latency = time.perf_counter() - total_start
        logger.info(
            "[Latency] TOTAL %s Search: Completed in %.4fs",
            search_type, total_latency,
        )

        return results[:top_k]

    def _run_hybrid_search(
        self,
        image: Image.Image,
        text_query: str,
        filters: Optional[Dict],
        alpha: float,
    ) -> List[Dict]:
        """Execute image and text retrieval in parallel, then fuse via RRF.

        Args:
            image: PIL image for image-based retrieval.
            text_query: Text string for text-based retrieval.
            filters: Optional Pinecone metadata filters.
            alpha: RRF fusion weight passed to ``_fuse_results``.

        Returns:
            Fused and sorted list of result dicts.
        """
        logger.info("[Parallel] Submitting Image + Text tasks to thread pool...")
        submit_time = time.perf_counter()

        future_img = _executor.submit(
            self._timed_image_retrieval, image, Config._RERANK_TOP_N, filters
        )
        future_txt = _executor.submit(
            self._timed_text_retrieval, text_query, Config._RERANK_TOP_N, filters
        )

        image_candidates, img_latency = future_img.result()
        text_candidates, txt_latency = future_txt.result()

        logger.info(
            "[Latency] Image Search (parallel): Retrieved %d candidates in %.4fs",
            len(image_candidates), img_latency,
        )
        logger.info(
            "[Latency] Text Search + Reranker (parallel): Retrieved %d candidates in %.4fs",
            len(text_candidates), txt_latency,
        )
        logger.info(
            "[Parallel] Wall-clock wait: %.4fs (expected ~max(%.2fs, %.2fs))",
            time.perf_counter() - submit_time, img_latency, txt_latency,
        )

        fusion_start = time.perf_counter()
        results = self._fuse_results(image_candidates, text_candidates, alpha)
        logger.info(
            "[Latency] RRF Fusion: Completed in %.4fs",
            time.perf_counter() - fusion_start,
        )

        return results

    def _timed_image_retrieval(
        self,
        image: Image.Image,
        top_k: int,
        filters: Optional[Dict],
    ) -> Tuple[List, float]:
        """Run image retrieval and return results with elapsed time.

        Args:
            image: PIL image to encode and query.
            top_k: Maximum number of candidates to retrieve.
            filters: Optional Pinecone metadata filters.

        Returns:
            Tuple of (matches list, elapsed seconds as float).
        """
        start = time.perf_counter()
        results = self._retrieve_from_image_index(image, top_k, filters)
        elapsed = time.perf_counter() - start
        logger.info("[Thread] Image retrieval done in %.4fs", elapsed)
        return results, elapsed

    def _timed_text_retrieval(
        self,
        text_query: str,
        top_k: int,
        filters: Optional[Dict],
    ) -> Tuple[List, float]:
        """Run text retrieval with reranking and return results with elapsed time.

        Args:
            text_query: Text string to encode and query.
            top_k: Maximum number of candidates to retrieve.
            filters: Optional Pinecone metadata filters.

        Returns:
            Tuple of (reranked matches list, elapsed seconds as float).
        """
        start = time.perf_counter()
        results = self._retrieve_from_text_index(text_query, top_k, filters)
        elapsed = time.perf_counter() - start
        logger.info("[Thread] Text+Rerank retrieval done in %.4fs", elapsed)
        return results, elapsed

    def _retrieve_from_image_index(
        self,
        image: Image.Image,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List:
        """Query the image index using SigLIP (768-dim) embeddings.

        Args:
            image: PIL image to encode.
            top_k: Maximum number of results to retrieve.
            filters: Optional Pinecone metadata filters.

        Returns:
            List of Pinecone match objects.
        """
        img_embedding = self.embedding_service.encode_images(image)
        img_vector = self._to_list(img_embedding)

        results = self.pinecone_service.query_images(
            vector=img_vector,
            top_k=top_k,
            filter=filters,
            namespace=self.namespace,
        )
        return results.matches

    def _retrieve_from_text_index(
        self,
        text_query: str,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List:
        """Query the text index (BGE-M3, 1024-dim) and rerank results.

        Profiled latency breakdown:
            BGE-M3 encoding : ~0.77s
            Pinecone query  : ~0.60s  (reduced from top_k=100 to _RERANK_CANDIDATES)
            Pinecone rerank : ~0.60s  (reduced from 100 docs to _RERANK_CANDIDATES)

        Args:
            text_query: Text string to encode and query.
            top_k: Maximum number of results to return after reranking.
            filters: Optional Pinecone metadata filters.

        Returns:
            List of reranked Pinecone match objects, or vector-score-sorted
            matches if reranking fails.
        """
        t0 = time.perf_counter()
        txt_embedding = self.embedding_service.encode_text(text_query)
        txt_vector = self._to_list(txt_embedding)
        logger.info("[Thread][Text] 1. BGE-M3 encoding:   %.4fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        initial_results = self.pinecone_service.query_text(
            vector=txt_vector,
            top_k=Config._RERANK_CANDIDATES,
            filter=filters,
            namespace=self.namespace,
        )
        matches = initial_results.matches
        logger.info(
            "[Thread][Text] 2. Pinecone query:     %.4fs  (%d matches, top_k=%d)",
            time.perf_counter() - t0, len(matches), Config._RERANK_CANDIDATES,
        )

        if not matches:
            return []

        docs_for_reranker = [
            {"text": match.metadata.get("text", "")} for match in matches
        ]

        t0 = time.perf_counter()
        try:
            rerank_response = self.pinecone_service.rerank(
                query=text_query,
                documents=docs_for_reranker,
                top_n=min(top_k, len(matches)),
            )
            logger.info(
                "[Thread][Text] 3. Pinecone rerank:   %.4fs",
                time.perf_counter() - t0,
            )

            reranked_matches = []
            for item in rerank_response.data:
                original_match = matches[item.index]
                original_match.score = item.score
                reranked_matches.append(original_match)

            return reranked_matches

        except Exception as e:
            logger.error("Reranking failed, falling back to vector scores: %s", e)
            return matches[:top_k]

    def _fuse_results(
        self,
        image_candidates: List,
        text_candidates: List,
        alpha: float,
    ) -> List[Dict]:
        """Fuse image and text candidates using Reciprocal Rank Fusion (RRF).

        Args:
            image_candidates: Ranked list of Pinecone matches from image retrieval.
            text_candidates: Ranked list of Pinecone matches from text retrieval.
            alpha: Image weight. 0 = text only, 1 = image only, 0.5 = balanced.

        Returns:
            List of fused result dicts sorted by descending RRF score.
        """
        product_scores: Dict[str, Dict] = {}

        for rank, match in enumerate(image_candidates):
            product_id = match.metadata["product_id"]
            rrf_score = alpha / (Config._RRF_K + rank)

            if product_id not in product_scores:
                product_scores[product_id] = {
                    "product_id": product_id,
                    "score": 0.0,
                    "image_score": 0.0,
                    "text_score": 0.0,
                    "category": match.metadata.get("category", "unknown"),
                    "filename": match.metadata.get("filename", ""),
                    "sources": [],
                }

            entry = product_scores[product_id]
            entry["score"] += rrf_score
            entry["image_score"] = match.score
            entry["sources"].append("image")

        for rank, match in enumerate(text_candidates):
            product_id = match.metadata["product_id"]
            rrf_score = (1 - alpha) / (Config._RRF_K + rank)

            if product_id not in product_scores:
                product_scores[product_id] = {
                    "product_id": product_id,
                    "score": 0.0,
                    "image_score": 0.0,
                    "text_score": 0.0,
                    "category": match.metadata.get("category", "unknown"),
                    "filename": match.metadata.get("filename", ""),
                    "sources": [],
                }

            entry = product_scores[product_id]
            entry["score"] += rrf_score
            entry["text_score"] = match.score
            entry["sources"].append("text")

        return sorted(
            product_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

    def _format_results(self, matches: List, source: str) -> List[Dict]:
        """Format raw Pinecone matches from a single index into result dicts.

        Args:
            matches: List of Pinecone match objects.
            source: Source label, either ``"image"`` or ``"text"``.

        Returns:
            List of result dicts with product_id, score, image_score,
            text_score, category, filename, and sources.
        """
        return [
            {
                "product_id": match.metadata["product_id"],
                "score": match.score,
                "image_score": match.score if source == "image" else 0.0,
                "text_score": match.score if source == "text" else 0.0,
                "category": match.metadata.get("category", "unknown"),
                "filename": match.metadata.get("filename", ""),
                "sources": [source],
            }
            for match in matches
        ]

    @staticmethod
    def _to_list(arr) -> List[float]:
        """Convert any array-like object to a flat Python list of floats.

        Handles PyTorch tensors (detach + cpu), 2-D numpy arrays (flatten),
        and any other iterable with a ``.tolist()`` method.

        Args:
            arr: Array-like object — torch.Tensor, np.ndarray, or similar.

        Returns:
            Flat list of Python floats.
        """
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        if hasattr(arr, "ndim") and arr.ndim == 2:
            arr = arr.flatten()
        return [float(x) for x in arr.tolist()]
