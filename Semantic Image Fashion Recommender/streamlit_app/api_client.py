import requests
from typing import Optional, Dict, List
from PIL import Image
import io
import logging

from streamlit_app.streamlit_config import StreamlitConfig

logger = logging.getLogger(__name__)


class APIClient:
    """Client for FastAPI backend"""

    def __init__(self, base_url: str = StreamlitConfig.API_BASE_URL):
        self.base_url = base_url
        self.timeout = 30  # seconds

    def check_health(self) -> Dict:
        """
        Check API health status

        Returns:
            Health status dict
        """
        try:
            response = requests.get(
                StreamlitConfig.HEALTH_ENDPOINT,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def search_by_image(
            self,
            image: Image.Image,
            text_query: Optional[str] = None,
            top_k: int = 10,
            image_weight: float = 0.7,
            text_weight: float = 0.3,
            use_rerank: bool = True,
            category: Optional[str] = None
    ) -> Dict:
        """
        Search by image

        Args:
            image: PIL Image
            text_query: Optional text preference
            top_k: Number of results
            image_weight: Weight for image similarity
            text_weight: Weight for text similarity
            use_rerank: Enable text reranking
            category: Category filter

        Returns:
            Search results dict
        """
        try:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)

            # Prepare files and data
            files = {
                'file': ('image.jpg', img_byte_arr, 'image/jpeg')
            }

            data = {
                'top_k': top_k,
                'image_weight': image_weight,
                'text_weight': text_weight,
                'use_rerank': use_rerank
            }

            if text_query:
                data['text_query'] = text_query

            if category:
                data['category'] = category

            # Make request
            response = requests.post(
                StreamlitConfig.IMAGE_SEARCH_ENDPOINT,
                files=files,
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Image search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def search_by_text(
            self,
            text_query: str,
            top_k: int = 10,
            category: Optional[str] = None
    ) -> Dict:
        """
        Search by text only

        Args:
            text_query: Text query
            top_k: Number of results
            category: Category filter

        Returns:
            Search results dict
        """
        try:
            data = {
                'text_query': text_query,
                'top_k': top_k
            }

            if category:
                data['category'] = category

            response = requests.post(
                StreamlitConfig.TEXT_SEARCH_ENDPOINT,
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Text search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def get_categories(self) -> List[str]:
        """
        Get available categories

        Returns:
            List of category names
        """
        try:
            response = requests.get(
                StreamlitConfig.CATEGORIES_ENDPOINT,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get('categories', [])

        except Exception as e:
            logger.error(f"Get categories failed: {e}")
            return []

    def get_stats(self) -> Dict:
        """
        Get index statistics

        Returns:
            Stats dict
        """
        try:
            response = requests.get(
                StreamlitConfig.STATS_ENDPOINT,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Get stats failed: {e}")
            return {"success": False, "error": str(e)}