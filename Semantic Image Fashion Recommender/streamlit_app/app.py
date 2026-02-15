import sys
import time
from pathlib import Path
import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api_client import APIClient
from streamlit_config import StreamlitConfig
from utils import (
    resize_image,
    validate_image,
    format_score
)

IMAGE_BASE_PATH = project_root / "data" / "fashion-mini" / "data"

if not IMAGE_BASE_PATH.exists():
    alt_path = Path("Semantic Image Fashion Recommender/data/fashion-mini/data")
    if alt_path.exists():
        IMAGE_BASE_PATH = alt_path
    else:
        st.warning(f"Image directory not found: {IMAGE_BASE_PATH}")

# Page configuration
st.set_page_config(
    page_title="Fashion Finder",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CUSTOM CSS - PROFESSIONAL ECOMMERCE STYLE ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ============================================
       GLOBAL RESET & BASE
    ============================================ */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Force clean white background */
    .stApp,
    .main,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* ============================================
       HERO HEADER SECTION
    ============================================ */
    .hero-header {
        background: linear-gradient(135deg, #f6f6f6 0%, #ffffff 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin-bottom: 2.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .hero-title {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1d1d1f !important;
        margin: 0 0 0.5rem 0 !important;
        line-height: 1.2 !important;
    }

    .hero-subtitle {
        font-size: 1.125rem;
        color: #6b7280;
        margin: 0;
        font-weight: 400;
    }

    /* ============================================
       SEARCH CONTAINER CARD
    ============================================ */
    .search-container {
        background: #ffffff;
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #1d1d1f;
    }

    /* ============================================
       TEXT INPUT - Rounded & Clean
    ============================================ */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
        padding: 0.875rem 1rem !important;
        font-size: 0.9375rem !important;
        background: #ffffff !important;
        color: #1d1d1f !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #1d1d1f !important;
        box-shadow: 0 0 0 4px rgba(29, 29, 31, 0.1) !important;
        outline: none !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #9ca3af !important;
    }

    /* Hide labels */
    .stTextInput label,
    .stFileUploader label,
    .stSelectbox label,
    .stSlider label {
        display: none;
    }

    /* ============================================
       FILE UPLOADER - Dark Premium Style
    ============================================ */
    [data-testid="stFileUploader"] {
        background: #2d3748 !important;
        border-radius: 16px !important;
        border: none !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        background: #1d2432 !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15) !important;
    }

    [data-testid="stFileUploader"] section {
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
        background: transparent !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }

    [data-testid="stFileUploader"] section button {
        background: #ffffff !important;
        color: #1d1d1f !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.625rem 1.25rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploader"] section button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }

    [data-testid="stFileUploader"] section small {
        color: #cbd5e1 !important;
        font-size: 0.875rem !important;
    }

    /* Uploaded image preview */
    [data-testid="stFileUploader"] + div [data-testid="stImage"] {
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid #e5e7eb !important;
    }

    /* ============================================
       BUTTONS - Dark & Rounded
    ============================================ */
    .stButton > button {
        background: #1d1d1f !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9375rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
        box-shadow: 0 2px 8px rgba(29, 29, 31, 0.15) !important;
    }

    .stButton > button:hover {
        background: #2d2d2f !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(29, 29, 31, 0.25) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ============================================
       SETTINGS CARD
    ============================================ */
    .settings-card {
        background: #f9fafb;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
    }

    .settings-label {
        font-size: 1rem;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 1rem;
        display: block;
    }

    /* ============================================
       SLIDER - Custom Style
    ============================================ */
    .stSlider {
        padding: 0.5rem 0;
    }

    .stSlider > div > div > div > div {
        background: #e5e7eb !important;
    }

    .stSlider [data-testid="stTickBar"] {
        background: #e5e7eb !important;
        height: 6px !important;
        border-radius: 3px !important;
    }

    .stSlider [data-testid="stThumbValue"] {
        background: #1d1d1f !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        font-size: 0.875rem !important;
        padding: 0.25rem 0.5rem !important;
    }

    /* Slider thumb */
    .stSlider input[type="range"]::-webkit-slider-thumb {
        background: #1d1d1f !important;
        width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }

    /* ============================================
       SELECTBOX - Rounded
    ============================================ */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
        background: #ffffff !important;
        transition: all 0.2s ease !important;
    }

    .stSelectbox > div > div:hover {
        border-color: #cbd5e1 !important;
    }

    .stSelectbox > div > div:focus-within {
        border-color: #1d1d1f !important;
        box-shadow: 0 0 0 3px rgba(29, 29, 31, 0.1) !important;
    }

    /* ============================================
       EXPANDER - Clean Style
    ============================================ */
    [data-testid="stExpander"] {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #1d1d1f !important;
        font-size: 0.9375rem !important;
        padding: 0.75rem 1rem !important;
    }

    .streamlit-expanderHeader:hover {
        background: #f3f4f6 !important;
    }

    [data-testid="stExpander"] > div > div {
        padding: 1rem !important;
    }

    /* ============================================
       DIVIDER - OR Text
    ============================================ */
    .divider-text {
        text-align: center;
        color: #9ca3af;
        margin: 1.5rem 0;
        font-size: 0.875rem;
        font-weight: 500;
        position: relative;
    }

    .divider-text::before,
    .divider-text::after {
        content: '';
        position: absolute;
        top: 50%;
        width: 45%;
        height: 1px;
        background: #e5e7eb;
    }

    .divider-text::before {
        left: 0;
    }

    .divider-text::after {
        right: 0;
    }

    /* ============================================
       PRODUCT CARDS - Premium Style
    ============================================ */
    .product-card {
        background: #ffffff;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }

    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
        border-color: #d1d5db;
    }

    /* Product image container */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        position: relative;
    }

    /* Product info styling */
    .product-id {
        font-size: 0.9375rem;
        font-weight: 600;
        color: #1d1d1f;
        margin: 0.75rem 0 0.25rem 0;
    }

    .product-category {
        font-size: 0.8125rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.025em;
        font-weight: 500;
    }

    .product-match {
        display: inline-block;
        background: #f9fafb;
        color: #1d1d1f;
        padding: 0.375rem 0.75rem;
        border-radius: 8px;
        font-size: 0.8125rem;
        font-weight: 600;
        margin-top: 0.5rem;
        border: 1px solid #e5e7eb;
    }

    .product-match.high {
        background: #dcfce7;
        color: #15803d;
        border-color: #bbf7d0;
    }

    .product-match.medium {
        background: #fef3c7;
        color: #92400e;
        border-color: #fde68a;
    }

    /* ============================================
       RESULTS HEADER
    ============================================ */
    .results-header {
        background: #f9fafb;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .results-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1d1d1f;
        margin: 0;
    }

    .results-count {
        font-size: 0.9375rem;
        color: #6b7280;
        font-weight: 500;
    }

    /* ============================================
       METRICS - Custom Style
    ============================================ */
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        color: #1d1d1f;
        font-weight: 700;
    }

    /* ============================================
       ALERT MESSAGES - Rounded
    ============================================ */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        padding: 1rem 1.25rem !important;
    }

    /* Success alert */
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.9375rem !important;
        font-weight: 500 !important;
    }

    /* ============================================
       EMPTY STATE
    ============================================ */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
        color: #6b7280;
        background: #f9fafb;
        border-radius: 24px;
        border: 2px dashed #e5e7eb;
    }

    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        opacity: 0.3;
        display: block;
    }

    .empty-state-title {
        font-size: 1.5rem;
        color: #1d1d1f;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .empty-state-text {
        font-size: 1rem;
        color: #6b7280;
        margin: 0;
    }

    /* ============================================
       DIVIDER
    ============================================ */
    hr {
        margin: 2.5rem 0 !important;
        border: none !important;
        border-top: 1px solid #e5e7eb !important;
    }

    /* ============================================
       LOADING SPINNER - Custom
    ============================================ */
    .stSpinner > div {
        border-top-color: #1d1d1f !important;
    }

    /* ============================================
       COLUMN SPACING
    ============================================ */
    [data-testid="column"] {
        padding: 0 0.75rem !important;
    }

    [data-testid="column"]:first-child {
        padding-left: 0 !important;
    }

    [data-testid="column"]:last-child {
        padding-right: 0 !important;
    }

    /* ============================================
       RESPONSIVE
    ============================================ */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero-header {
            padding: 2rem 1.5rem;
        }

        .hero-title {
            font-size: 2rem !important;
        }

        .hero-subtitle {
            font-size: 1rem;
        }

        .search-container {
            padding: 1.5rem;
        }

        .results-header {
            padding: 1rem 1.5rem;
            flex-direction: column;
            align-items: flex-start;
            gap: 1rem;
        }

        [data-testid="column"] {
            padding: 0 0.5rem !important;
        }
    }

    /* ============================================
       SMOOTH SCROLLING
    ============================================ */
    html {
        scroll-behavior: smooth;
    }

    /* ============================================
       SELECTION COLOR
    ============================================ */
    ::selection {
        background: rgba(29, 29, 31, 0.1);
        color: #1d1d1f;
    }
</style>
""", unsafe_allow_html=True)


# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient()


api_client = get_api_client()

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'search_history' not in st.session_state:
    st.session_state.search_history = []


def detect_search_mode(uploaded_image, text_query):
    """Automatically detect search mode based on inputs"""
    has_image = uploaded_image is not None
    has_text = text_query is not None and text_query.strip() != ""

    if has_image and has_text:
        return "hybrid"
    elif has_image:
        return "image"
    elif has_text:
        return "text"
    else:
        return None


def get_match_class(score):
    """Return CSS class based on match score"""
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    else:
        return ""


def main():
    """Main application"""

    # ========== HERO HEADER ==========
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🛍️ Fashion Finder</h1>
        <p class="hero-subtitle">Discover your perfect style with AI-powered visual search</p>
    </div>
    """, unsafe_allow_html=True)

    # ========== SEARCH SECTION ==========
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Search Products</div>', unsafe_allow_html=True)

    # Layout: 2 columns (search inputs | settings)
    col_search, col_settings = st.columns([2.5, 1.5], gap="large")

    with col_search:
        # Text search
        text_query = st.text_input(
            "text_search",
            placeholder="Describe what you're looking for... (e.g., red striped summer dress)",
            label_visibility="collapsed",
            key="text_input"
        )

        st.markdown('<p class="divider-text">or upload an image</p>', unsafe_allow_html=True)

        # Image upload
        uploaded_file = st.file_uploader(
            "Upload image",
            type=StreamlitConfig.ALLOWED_EXTENSIONS,
            key="image_upload",
            label_visibility="collapsed"
        )

        uploaded_image = None
        if uploaded_file:
            uploaded_image = validate_image(uploaded_file)
            if uploaded_image:
                display_img = resize_image(uploaded_image.copy(), (500, 500))
                st.image(display_img, use_container_width=True)
            else:
                st.error("❌ Invalid image file. Please upload a valid image.")

    with col_settings:
        st.markdown('<span class="settings-label">⚙️ Settings</span>', unsafe_allow_html=True)

        top_k = st.slider(
            "Number of results",
            min_value=5,
            max_value=50,
            value=12,
            step=1,
            help="How many similar products to show"
        )

        categories = api_client.get_categories()
        category = st.selectbox(
            "Category filter",
            ["All Categories"] + categories,
            help="Filter by product category"
        )
        category = None if category == "All Categories" else category

        # Advanced settings for hybrid
        search_mode = detect_search_mode(uploaded_image, text_query)

        image_weight = 0.7
        text_weight = 0.3
        use_rerank = True

        if search_mode == "hybrid":
            with st.expander("⚙️ Advanced Settings"):
                st.markdown("**Hybrid Search Weights**")

                image_weight = st.slider(
                    "Image importance",
                    0.0, 1.0, 0.7, 0.1,
                    help="How much to prioritize visual similarity"
                )

                text_weight = st.slider(
                    "Text importance",
                    0.0, 1.0, 0.3, 0.1,
                    help="How much to prioritize text description"
                )

                use_rerank = st.checkbox(
                    "Enable AI Reranking",
                    value=True,
                    help="Use advanced reranking for better results"
                )

        st.markdown("")
        search_clicked = st.button("🔍 Search Products", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== PERFORM SEARCH ==========
    if search_clicked:
        search_mode = detect_search_mode(uploaded_image, text_query)

        if search_mode is None:
            st.error("⚠️ Please provide either an image or text description to search")
        else:
            with st.spinner("🔎 Searching for similar products..."):
                start_time = time.time()

                try:
                    if search_mode == "image":
                        results = api_client.search_by_image(
                            image=uploaded_image,
                            top_k=top_k,
                            category=category
                        )
                        query_display = "Visual Search"

                    elif search_mode == "text":
                        results = api_client.search_by_text(
                            text_query=text_query,
                            top_k=top_k,
                            category=category
                        )
                        query_display = f'"{text_query}"'

                    else:  # hybrid
                        results = api_client.search_by_image(
                            image=uploaded_image,
                            text_query=text_query,
                            top_k=top_k,
                            image_weight=image_weight,
                            text_weight=text_weight,
                            use_rerank=use_rerank,
                            category=category
                        )
                        query_display = f'Hybrid: "{text_query}"'

                    search_time = time.time() - start_time

                    if results.get("success"):
                        st.session_state.search_results = results
                        st.session_state.search_time = search_time
                        st.session_state.search_mode = search_mode
                        st.session_state.query_display = query_display

                        st.success(f"✅ Found {len(results.get('results', []))} products in {search_time:.2f} seconds")
                    else:
                        st.error(f"❌ Search failed: {results.get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"❌ Error occurred: {str(e)}")

    # ========== RESULTS SECTION ==========
    if st.session_state.search_results:
        st.divider()

        items = st.session_state.search_results.get('results', [])

        if items:
            # Results header
            col_h1, col_h2, col_h3 = st.columns([2, 1, 1])

            with col_h1:
                st.markdown(f'<h2 class="results-title">{len(items)} Products Found</h2>',
                            unsafe_allow_html=True)
                st.markdown(f'<p class="results-count">Query: {st.session_state.get("query_display", "N/A")}</p>',
                            unsafe_allow_html=True)

            with col_h2:
                search_mode_display = st.session_state.get('search_mode', 'N/A').capitalize()
                st.metric("Search Mode", search_mode_display)

            with col_h3:
                st.metric("Response Time", f"{st.session_state.search_time:.2f}s")

            st.markdown("")
            st.markdown("")

            # Product grid (4 columns)
            num_cols = 4

            for i in range(0, len(items), num_cols):
                cols = st.columns(num_cols, gap="medium")

                for idx, col in enumerate(cols):
                    if i + idx < len(items):
                        item = items[i + idx]

                        with col:
                            # Container for product card
                            product_id = item.get('product_id', 'N/A')
                            image_path = IMAGE_BASE_PATH / f"{product_id}.jpg"

                            # Try to load image
                            if image_path.exists():
                                try:
                                    from PIL import Image
                                    img = Image.open(image_path)
                                    st.image(img, use_container_width=True)
                                except:
                                    st.markdown("""
                                    <div style='background: #f3f4f6; padding: 3rem 1rem; 
                                                text-align: center; border-radius: 16px;
                                                color: #9ca3af;'>
                                        <p style='font-size: 2rem; margin: 0;'>📷</p>
                                        <p style='font-size: 0.875rem; margin: 0.5rem 0 0 0;'>Image unavailable</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style='background: #f3f4f6; padding: 3rem 1rem; 
                                            text-align: center; border-radius: 16px;
                                            color: #9ca3af;'>
                                    <p style='font-size: 2rem; margin: 0;'>📷</p>
                                    <p style='font-size: 0.875rem; margin: 0.5rem 0 0 0;'>Image unavailable</p>
                                </div>
                                """, unsafe_allow_html=True)

                            # Product info
                            st.markdown(f'<p class="product-id">{product_id}</p>', unsafe_allow_html=True)
                            st.markdown(f'<p class="product-category">{item.get("category", "Unknown")}</p>',
                                        unsafe_allow_html=True)

                            # Match score with color coding
                            score = item.get('score', 0)
                            score_pct = int(score * 100)
                            match_class = get_match_class(score)

                            st.markdown(f'<span class="product-match {match_class}">✓ {score_pct}% Match</span>',
                                        unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-state-icon">🔍</span>
                <h3 class="empty-state-title">No Results Found</h3>
                <p class="empty-state-text">Try adjusting your search criteria or filters</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Initial empty state
        st.divider()
        st.markdown("""
        <div class="empty-state">
            <span class="empty-state-icon">🛍️</span>
            <h3 class="empty-state-title">Start Your Fashion Journey</h3>
            <p class="empty-state-text">Upload an image or describe what you're looking for to discover similar products</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()