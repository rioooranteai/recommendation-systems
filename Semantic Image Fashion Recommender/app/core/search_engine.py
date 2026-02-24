import time
import logging
import concurrent.futures
from typing import List, Dict, Optional, Tuple

from PIL import Image
from config import Config

logger = logging.getLogger(__name__)

# Shared thread pool — created once at module load, reused across all requests
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Tuning constants — adjust here to balance latency vs recall
_RERANK_CANDIDATES = 30   # was 100 — fewer candidates = faster query + rerank
_RERANK_TOP_N = 50        # final candidates returned after rerank


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
        Two-stage hybrid search with parallel execution for hybrid mode.
        - Image-only : query image index (768-dim SigLIP)
        - Text-only  : query text index (1024-dim BGE-M3) + rerank
        - Hybrid     : BOTH run in parallel, then fuse via RRF
        """

        if image is None and text_query is None:
            raise ValueError("Must provide at least one of: image or text_query")

        total_start = time.perf_counter()
        image_candidates: List = []
        text_candidates: List = []

        # ------------------------------------------------------------------ #
        # HYBRID: run both retrievals in parallel                             #
        # ------------------------------------------------------------------ #
        if image is not None and text_query is not None:
            search_type = "Hybrid"

            logger.info("[Parallel] Submitting Image + Text tasks to thread pool...")
            submit_time = time.perf_counter()

            future_img = _executor.submit(
                self._timed_image_retrieval, image, 50, filters
            )
            future_txt = _executor.submit(
                self._timed_text_retrieval, text_query, 50, filters
            )

            image_candidates, img_latency = future_img.result()
            text_candidates, txt_latency = future_txt.result()

            logger.info(
                f"[Latency] Image Search (parallel): "
                f"Retrieved {len(image_candidates)} candidates in {img_latency:.4f}s"
            )
            logger.info(
                f"[Latency] Text Search + Reranker (parallel): "
                f"Retrieved {len(text_candidates)} candidates in {txt_latency:.4f}s"
            )
            logger.info(
                f"[Parallel] Wall-clock wait: {time.perf_counter() - submit_time:.4f}s "
                f"(expected ~max({img_latency:.2f}s, {txt_latency:.2f}s))"
            )

            fusion_start = time.perf_counter()
            results = self._fuse_results(image_candidates, text_candidates, alpha)
            logger.info(
                f"[Latency] RRF Fusion: Completed in "
                f"{time.perf_counter() - fusion_start:.4f}s"
            )

        # ------------------------------------------------------------------ #
        # IMAGE-ONLY                                                          #
        # ------------------------------------------------------------------ #
        elif image is not None:
            search_type = "Image-only"
            image_candidates, img_latency = self._timed_image_retrieval(
                image, 50, filters
            )
            logger.info(
                f"[Latency] Image Search: "
                f"Retrieved {len(image_candidates)} candidates in {img_latency:.4f}s"
            )
            results = self._format_results(image_candidates, "image")

        # ------------------------------------------------------------------ #
        # TEXT-ONLY                                                           #
        # ------------------------------------------------------------------ #
        else:
            search_type = "Text-only"
            text_candidates, txt_latency = self._timed_text_retrieval(
                text_query, 50, filters
            )
            logger.info(
                f"[Latency] Text Search + Reranker: "
                f"Retrieved {len(text_candidates)} candidates in {txt_latency:.4f}s"
            )
            results = self._format_results(text_candidates, "text")

        total_latency = time.perf_counter() - total_start
        logger.info(
            f"[Latency] TOTAL {search_type} Search: Completed in {total_latency:.4f}s"
        )

        return results[:top_k]

    # ---------------------------------------------------------------------- #
    # Timed wrappers                                                          #
    # ---------------------------------------------------------------------- #

    def _timed_image_retrieval(
            self,
            image: Image.Image,
            top_k: int,
            filters: Optional[Dict]
    ) -> Tuple[List, float]:
        start = time.perf_counter()
        results = self._retrieve_from_image_index(image, top_k, filters)
        elapsed = time.perf_counter() - start
        logger.info(f"[Thread] Image retrieval done in {elapsed:.4f}s")
        return results, elapsed

    def _timed_text_retrieval(
            self,
            text_query: str,
            top_k: int,
            filters: Optional[Dict]
    ) -> Tuple[List, float]:
        start = time.perf_counter()
        results = self._retrieve_from_text_index(text_query, top_k, filters)
        elapsed = time.perf_counter() - start
        logger.info(f"[Thread] Text+Rerank retrieval done in {elapsed:.4f}s")
        return results, elapsed

    # ---------------------------------------------------------------------- #
    # Core retrieval methods                                                  #
    # ---------------------------------------------------------------------- #

    def _retrieve_from_image_index(
            self,
            image: Image.Image,
            top_k: int,
            filters: Optional[Dict] = None
    ) -> List:
        """Query image index (768-dim SigLIP vectors)."""
        img_embedding = self.embedding_service.encode_images(image)
        img_vector = self._to_list(img_embedding)

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
        """
        Query text index (1024-dim BGE-M3) then rerank.

        Latency breakdown from profiling:
          BGE-M3 encoding : ~0.77s
          Pinecone query  : ~2.03s  (was top_k=100, now 30 → expect ~0.6s)
          Pinecone rerank : ~1.84s  (was 100 docs, now 30 → expect ~0.6s)
        """

        # Step 1: BGE-M3 encoding
        t0 = time.perf_counter()
        txt_embedding = self.embedding_service.encode_text(text_query)
        txt_vector = self._to_list(txt_embedding)
        logger.info(f"[Thread][Text] 1. BGE-M3 encoding:   {time.perf_counter() - t0:.4f}s")

        # Step 2: Pinecone vector query
        # Reduced from 100 → _RERANK_CANDIDATES to cut query + rerank payload
        t0 = time.perf_counter()
        initial_results = self.pinecone_service.query_text(
            vector=txt_vector,
            top_k=_RERANK_CANDIDATES,
            filter=filters,
            namespace=self.namespace
        )
        matches = initial_results.matches
        logger.info(
            f"[Thread][Text] 2. Pinecone query:     {time.perf_counter() - t0:.4f}s"
            f"  ({len(matches)} matches, top_k={_RERANK_CANDIDATES})"
        )

        if not matches:
            return []

        # Step 3: Rerank
        docs_for_reranker = [
            {"text": match.metadata.get("text", "")} for match in matches
        ]
        t0 = time.perf_counter()
        try:
            rerank_response = self.pinecone_service.rerank(
                query=text_query,
                documents=docs_for_reranker,
                top_n=min(top_k, len(matches))
            )
            logger.info(f"[Thread][Text] 3. Pinecone rerank:   {time.perf_counter() - t0:.4f}s")

            reranked_matches = []
            for item in rerank_response.data:
                original_match = matches[item.index]
                original_match.score = item.score
                reranked_matches.append(original_match)

            return reranked_matches

        except Exception as e:
            logger.error(f"Reranking failed, falling back to vector scores: {e}")
            return matches[:top_k]

    # ---------------------------------------------------------------------- #
    # Fusion & formatting                                                     #
    # ---------------------------------------------------------------------- #

    def _fuse_results(
            self,
            image_candidates: List,
            text_candidates: List,
            alpha: float
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF).
        alpha: 0 = text only, 1 = image only, 0.5 = balanced.
        """
        product_scores: Dict[str, Dict] = {}
        k = 60  # RRF constant

        for rank, match in enumerate(image_candidates):
            product_id = match.metadata["product_id"]
            rrf_score = alpha / (k + rank)

            if product_id not in product_scores:
                product_scores[product_id] = {
                    "product_id": product_id,
                    "score": 0.0,
                    "image_score": 0.0,
                    "text_score": 0.0,
                    "category": match.metadata.get("category", "unknown"),
                    "filename": match.metadata.get("filename", ""),
                    "sources": []
                }

            entry = product_scores[product_id]
            entry["score"] += rrf_score
            entry["image_score"] = match.score
            entry["sources"].append("image")

        for rank, match in enumerate(text_candidates):
            product_id = match.metadata["product_id"]
            rrf_score = (1 - alpha) / (k + rank)

            if product_id not in product_scores:
                product_scores[product_id] = {
                    "product_id": product_id,
                    "score": 0.0,
                    "image_score": 0.0,
                    "text_score": 0.0,
                    "category": match.metadata.get("category", "unknown"),
                    "filename": match.metadata.get("filename", ""),
                    "sources": []
                }

            entry = product_scores[product_id]
            entry["score"] += rrf_score
            entry["text_score"] = match.score
            entry["sources"].append("text")

        return sorted(
            product_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

    def _format_results(self, matches: List, source: str) -> List[Dict]:
        """Format results from a single index."""
        return [
            {
                "product_id": match.metadata["product_id"],
                "score": match.score,
                "image_score": match.score if source == "image" else 0.0,
                "text_score": match.score if source == "text" else 0.0,
                "category": match.metadata.get("category", "unknown"),
                "filename": match.metadata.get("filename", ""),
                "sources": [source]
            }
            for match in matches
        ]

    # ---------------------------------------------------------------------- #
    # Utilities                                                               #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _to_list(arr) -> List[float]:
        """Safely convert any array-like to a flat Python list of floats."""
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        if hasattr(arr, "ndim") and arr.ndim == 2:
            arr = arr.flatten()
        return [float(x) for x in arr.tolist()]