import os


class StreamlitConfig:
    # FastAPI backend URL
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

    # API endpoints
    HEALTH_ENDPOINT = f"{API_BASE_URL}/api/health"
    IMAGE_SEARCH_ENDPOINT = f"{API_BASE_URL}/api/search/image"
    TEXT_SEARCH_ENDPOINT = f"{API_BASE_URL}/api/search/text"
    CATEGORIES_ENDPOINT = f"{API_BASE_URL}/api/categories"
    STATS_ENDPOINT = f"{API_BASE_URL}/api/stats"

    # Upload settings
    MAX_FILE_SIZE_MB = 10
    ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

    # Display settings
    DEFAULT_TOP_K = 10
    MAX_TOP_K = 50
    RESULTS_PER_ROW = 4

    # Image display size
    THUMBNAIL_SIZE = (200, 200)