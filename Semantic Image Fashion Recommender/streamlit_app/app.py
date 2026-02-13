import sys
import time
from pathlib import Path
import os
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
    page_title="Fashion Search 🛍️",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== IMPROVED CUSTOM CSS ==========
st.markdown("""
<style>
    /* ============================================
       GLOBAL & RESET
    ============================================ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }

    /* ============================================
       HEADER SECTION - Modern Gradient
    ============================================ */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #1d1d1f 0%, #4a4a4f 50%, #1d1d1f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        line-height: 1.2;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.15rem;
        margin-bottom: 3rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* ============================================
       SEARCH CONTAINER - Apple Style Card
    ============================================ */
    .search-container {
        background: #ffffff;
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 2.5rem;
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
    }

    .search-container:hover {
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }

    /* ============================================
       TABS - Rounded & Modern
    ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f6f6f6;
        border-radius: 16px;
        padding: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 12px;
        padding: 0 24px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #6b7280;
        background-color: transparent;
        border: none;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e5e7eb;
        color: #1d1d1f;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1d1d1f !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(29, 29, 31, 0.15);
    }

    /* ============================================
       FILTERS - Compact & Clean
    ============================================ */
    .filters-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 2.5rem;
        border: 1px solid #e5e7eb;
    }

    .filter-label {
        font-size: 1rem;
        font-weight: 700;
        color: #1d1d1f;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ============================================
       PRODUCT GRID - Enhanced Cards
    ============================================ */
    .product-card {
        background: white;
        border: 1px solid #f0f0f0;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .product-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(29, 29, 31, 0.02) 0%, rgba(29, 29, 31, 0.05) 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
        border-radius: 16px;
    }

    .product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
        border-color: #1d1d1f;
    }

    .product-card:hover::before {
        opacity: 1;
    }

    .product-image-container {
        position: relative;
        width: 100%;
        padding-bottom: 100%;
        background: linear-gradient(135deg, #fafafa 0%, #f0f0f0 100%);
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .product-image-placeholder {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 3.5rem;
        opacity: 0.3;
        filter: grayscale(20%);
    }

    /* ============================================
       SCORE BADGES - Premium Look
    ============================================ */
    .score-badge {
        position: absolute;
        top: 12px;
        right: 12px;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        z-index: 10;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        letter-spacing: 0.5px;
    }

    .score-high {
        background: linear-gradient(135deg, #1d1d1f 0%, #2d2d2f 100%);
        color: white;
    }

    .score-medium {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
    }

    .score-low {
        background: linear-gradient(135deg, #9ca3af 0%, #6b7280 100%);
        color: white;
    }

    .product-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #1d1d1f;
        margin-bottom: 0.4rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        letter-spacing: -0.3px;
    }

    .product-category {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
        font-weight: 500;
    }

    .product-meta {
        font-size: 0.85rem;
        color: #9ca3af;
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
    }

    .product-meta span {
        background: #f6f6f6;
        padding: 0.3rem 0.75rem;
        border-radius: 16px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .product-meta span:hover {
        background: #e5e7eb;
        transform: scale(1.05);
    }

    /* ============================================
       PRODUCT IMAGES - Polished
    ============================================ */
    .product-card img {
        border-radius: 12px;
        width: 100%;
        height: auto;
        object-fit: cover;
        aspect-ratio: 1 / 1;
        transition: transform 0.3s ease;
    }

    .product-card:hover img {
        transform: scale(1.05);
    }

    .product-card > div > img {
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }

    /* ============================================
       RESULTS HEADER - Stats Bar
    ============================================ */
    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding: 1.5rem 0;
        border-bottom: 2px solid #f0f0f0;
    }

    .results-count {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1d1d1f;
        letter-spacing: -0.5px;
    }

    .results-meta {
        color: #6b7280;
        font-size: 0.95rem;
        display: flex;
        gap: 2rem;
        align-items: center;
        font-weight: 500;
    }

    /* ============================================
       EMPTY STATE - Elegant
    ============================================ */
    .empty-state {
        text-align: center;
        padding: 6rem 2rem;
        color: #6b7280;
        background: linear-gradient(135deg, #fafafa 0%, #f0f0f0 100%);
        border-radius: 24px;
        margin: 2rem 0;
    }

    .empty-state-icon {
        font-size: 6rem;
        margin-bottom: 2rem;
        opacity: 0.4;
        filter: grayscale(30%);
    }

    .empty-state h3 {
        font-size: 1.8rem;
        color: #1d1d1f;
        margin-bottom: 0.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .empty-state p {
        font-size: 1.05rem;
        color: #6b7280;
        line-height: 1.6;
    }

    /* ============================================
       BADGES & PILLS
    ============================================ */
    .stats-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1d1d1f 0%, #2d2d2f 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-left: 0.75rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(29, 29, 31, 0.2);
    }

    /* ============================================
       BUTTONS - Premium Dark Style
    ============================================ */
    .stButton > button {
        background: #1d1d1f !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.3px !important;
    }

    .stButton > button:hover {
        background: #2d2d2f !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(29, 29, 31, 0.25) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Secondary Buttons */
    .stButton > button[data-testid="baseButton-secondary"] {
        background: white !important;
        border: 1.5px solid #e5e7eb !important;
        color: #1d1d1f !important;
    }

    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: #f6f6f6 !important;
        border-color: #1d1d1f !important;
    }

    /* ============================================
       INPUTS - Rounded & Clean
    ============================================ */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {
        border-radius: 12px !important;
        border: 1.5px solid #e5e7eb !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #1d1d1f !important;
        box-shadow: 0 0 0 3px rgba(29, 29, 31, 0.1) !important;
    }

    /* ============================================
       FILE UPLOADER - Enhanced
    ============================================ */
    .stFileUploader {
        border-radius: 16px;
        border: 2px dashed #e5e7eb;
        padding: 2rem;
        background: #fafafa;
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: #1d1d1f;
        background: white;
    }

    /* ============================================
       METRICS - Card Style
    ============================================ */
    [data-testid="stMetric"] {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #f0f0f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #1d1d1f;
        font-weight: 800;
    }

    /* ============================================
       ALERTS - Rounded
    ============================================ */
    .stAlert {
        border-radius: 16px !important;
        border: none !important;
        padding: 1.2rem 1.5rem !important;
    }

    /* ============================================
       SPINNERS - Modern
    ============================================ */
    .stSpinner > div {
        border-color: #1d1d1f !important;
        border-right-color: transparent !important;
    }

    /* ============================================
       SLIDER - Custom Style
    ============================================ */
    .stSlider > div > div > div {
        background: #1d1d1f !important;
    }

    /* ============================================
       EXPANDER - Cleaner
    ============================================ */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #1d1d1f !important;
        border-radius: 12px !important;
    }

    /* ============================================
       FOOTER - Minimal
    ============================================ */
    .footer-section {
        text-align: center;
        color: #9ca3af;
        font-size: 0.9rem;
        padding: 3rem 0;
        border-top: 1px solid #f0f0f0;
        margin-top: 4rem;
    }

    .footer-section strong {
        color: #1d1d1f;
        font-weight: 700;
    }

    /* ============================================
       RESPONSIVE
    ============================================ */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }

        .subtitle {
            font-size: 1rem;
        }

        .search-container {
            padding: 1.5rem;
            border-radius: 20px;
        }

        .results-count {
            font-size: 1.4rem;
        }

        .empty-state {
            padding: 4rem 1.5rem;
        }

        .empty-state-icon {
            font-size: 4rem;
        }
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

if 'filtered_results' not in st.session_state:
    st.session_state.filtered_results = None

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

if 'active_filters' not in st.session_state:
    st.session_state.active_filters = {
        'score_threshold': None,
        'category': None,
        'sort_by': 'score'
    }


def apply_filters(results):
    """Apply active filters to results"""
    if not results:
        return []

    items = results.get('results', [])
    filtered = items.copy()

    # Filter by score threshold
    if st.session_state.active_filters['score_threshold']:
        threshold = st.session_state.active_filters['score_threshold']
        filtered = [item for item in filtered if item.get('score', 0) >= threshold]

    # Filter by category
    if st.session_state.active_filters['category']:
        category = st.session_state.active_filters['category']
        filtered = [item for item in filtered if item.get('category') == category]

    # Sort results
    sort_by = st.session_state.active_filters['sort_by']
    if sort_by == 'score':
        filtered.sort(key=lambda x: x.get('score', 0), reverse=True)
    elif sort_by == 'category':
        filtered.sort(key=lambda x: x.get('category', ''))

    return filtered


def render_product_card(item, col):
    """Render a single product card with actual image"""
    with col:
        score = item.get('score', 0)
        score_class = "score-high" if score >= 0.8 else "score-medium" if score >= 0.6 else "score-low"

        category = item.get('category', 'Unknown')
        product_id = item.get('product_id', 'N/A')

        # ========== TRY TO LOAD IMAGE ==========
        image_filename = f"{product_id}.jpg"
        image_path = IMAGE_BASE_PATH / image_filename

        # Start card HTML
        st.markdown('<div class="product-card">', unsafe_allow_html=True)

        # ========== IMAGE SECTION ==========
        if image_path.exists():
            try:
                from PIL import Image
                img = Image.open(image_path)

                # Create container with relative positioning
                st.markdown(
                    '<div style="position: relative; margin-bottom: 1rem;">',
                    unsafe_allow_html=True
                )

                # Display image
                st.image(img, use_container_width=True)

                # Score badge overlay
                st.markdown(
                    f'<div class="score-badge {score_class}" '
                    f'style="position: absolute; top: 10px; right: 10px; z-index: 10;">'
                    f'{format_score(score)}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                # Fallback to emoji
                category_emoji = {
                    'Tshirts': '👕', 'Shirts': '👔', 'Jeans': '👖',
                    'Trousers': '👔', 'Shoes': '👟', 'Watches': '⌚',
                    'Bags': '👜', 'Sunglasses': '🕶️',
                }
                emoji = category_emoji.get(category, '👗')

                st.markdown(
                    f'<div class="product-image-container">'
                    f'<div class="product-image-placeholder">{emoji}</div>'
                    f'<span class="score-badge {score_class}">{format_score(score)}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            # Image not found - use emoji
            category_emoji = {
                'Tshirts': '👕', 'Shirts': '👔', 'Jeans': '👖',
                'Trousers': '👔', 'Shoes': '👟', 'Watches': '⌚',
                'Bags': '👜', 'Sunglasses': '🕶️',
            }
            emoji = category_emoji.get(category, '👗')

            st.markdown(
                f'<div class="product-image-container">'
                f'<div class="product-image-placeholder">{emoji}</div>'
                f'<span class="score-badge {score_class}">{format_score(score)}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ========== PRODUCT INFO ==========
        st.markdown(
            f'<div class="product-title" title="{product_id}">{product_id}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="product-category">📦 {category}</div>',
            unsafe_allow_html=True
        )

        # Meta badges
        meta_html = '<div class="product-meta">'

        if item.get('image_score') is not None:
            meta_html += f'<span>🖼️ {format_score(item["image_score"])}</span>'

        if item.get('text_score') is not None:
            meta_html += f'<span>📝 {format_score(item["text_score"])}</span>'

        meta_html += '</div></div>'  # Close product-meta and product-card

        st.markdown(meta_html, unsafe_allow_html=True)


def render_quick_filters(results):
    """Render quick filter buttons"""
    st.markdown('<div class="filters-container">', unsafe_allow_html=True)
    st.markdown('<div class="filter-label">🎯 Quick Filters</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    items = results.get('results', [])

    # Get unique categories from results
    categories = list(set([item.get('category', 'Unknown') for item in items]))
    categories.sort()

    # Count items by score range
    high_score_count = len([i for i in items if i.get('score', 0) >= 0.8])
    medium_score_count = len([i for i in items if 0.6 <= i.get('score', 0) < 0.8])

    with col1:
        if st.button(
                f"⭐ Top ({high_score_count})",
                use_container_width=True,
                key="filter_top"
        ):
            if st.session_state.active_filters['score_threshold'] == 0.8:
                st.session_state.active_filters['score_threshold'] = None
            else:
                st.session_state.active_filters['score_threshold'] = 0.8
            st.rerun()

    with col2:
        if st.button(
                f"🎯 Good ({medium_score_count})",
                use_container_width=True,
                key="filter_medium"
        ):
            if st.session_state.active_filters['score_threshold'] == 0.6:
                st.session_state.active_filters['score_threshold'] = None
            else:
                st.session_state.active_filters['score_threshold'] = 0.6
            st.rerun()

    with col3:
        # Category filter dropdown
        selected_category = st.selectbox(
            "Category",
            ["All Categories"] + categories,
            key="filter_category_select",
            label_visibility="collapsed"
        )

        if selected_category != "All Categories":
            st.session_state.active_filters['category'] = selected_category
        else:
            st.session_state.active_filters['category'] = None

    with col4:
        # Sort options
        sort_option = st.selectbox(
            "Sort",
            ["Score ↓", "Category A-Z"],
            key="sort_select",
            label_visibility="collapsed"
        )

        if sort_option == "Score ↓":
            st.session_state.active_filters['sort_by'] = 'score'
        else:
            st.session_state.active_filters['sort_by'] = 'category'

    with col5:
        if st.button("🔄 Reset", use_container_width=True, key="reset_filters"):
            st.session_state.active_filters = {
                'score_threshold': None,
                'category': None,
                'sort_by': 'score'
            }
            st.rerun()

    # Show active filters
    active_filter_tags = []

    if st.session_state.active_filters['score_threshold']:
        threshold = st.session_state.active_filters['score_threshold']
        active_filter_tags.append(f"Score ≥ {int(threshold * 100)}%")

    if st.session_state.active_filters['category']:
        active_filter_tags.append(f"{st.session_state.active_filters['category']}")

    if active_filter_tags:
        st.markdown(
            f"<div style='margin-top: 1rem; color: #6b7280; font-size: 0.9rem; font-weight: 600;'>"
            f"🔖 Active: {' • '.join(active_filter_tags)}</div>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application"""

    # Header
    st.markdown('<h1 class="main-header">🛍️ Fashion Finder</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI-powered visual search for fashion items</p>',
        unsafe_allow_html=True
    )

    # ========== SEARCH SECTION (TOP) ==========
    st.markdown('<div class="search-container">', unsafe_allow_html=True)

    # Search Mode Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Image Search",
        "📝 Text Search",
        "🔄 Hybrid Search",
        "📊 Statistics"
    ])

    uploaded_image = None
    text_query = None
    search_triggered = False

    # Tab 1: Image Search
    with tab1:
        st.markdown("### Upload Fashion Image")

        col_upload, col_settings = st.columns([2, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Drop your image here",
                type=StreamlitConfig.ALLOWED_EXTENSIONS,
                key="image_search_uploader",
                help="Drag and drop or click to upload (JPG, PNG)"
            )

            if uploaded_file:
                uploaded_image = validate_image(uploaded_file)
                if uploaded_image:
                    # Resize for display
                    display_img = resize_image(uploaded_image.copy(), (400, 400))
                    st.image(display_img, caption="Query Image", use_container_width=True)
                else:
                    st.error("❌ Invalid image file")

        with col_settings:
            st.markdown("#### ⚙️ Settings")

            top_k_img = st.slider(
                "Results",
                min_value=1,
                max_value=50,
                value=12,
                key="topk_img"
            )

            categories = api_client.get_categories()
            category_img = st.selectbox(
                "Category",
                ["All"] + categories,
                key="cat_img"
            )
            category_img = None if category_img == "All" else category_img

            st.markdown("")

            if st.button(
                    "🔍 Search",
                    key="search_img",
                    type="primary",
                    use_container_width=True
            ):
                if uploaded_image:
                    search_triggered = True
                    search_mode = "image"
                    top_k = top_k_img
                    category = category_img
                else:
                    st.error("Please upload an image first!")

    # Tab 2: Text Search
    with tab2:
        st.markdown("### Describe What You're Looking For")

        col_text, col_settings2 = st.columns([2, 1])

        with col_text:
            text_query = st.text_area(
                "Search query",
                placeholder="e.g., red striped t-shirt, blue denim jeans, leather jacket...",
                key="text_query_input",
                height=120,
                label_visibility="collapsed"
            )

            # Example queries
            st.markdown("**💡 Try these:**")
            example_col1, example_col2, example_col3 = st.columns(3)

            with example_col1:
                if st.button("Red striped shirt", key="ex1", use_container_width=True):
                    st.session_state.text_query_input = "red striped shirt"
                    st.rerun()

            with example_col2:
                if st.button("Blue denim jeans", key="ex2", use_container_width=True):
                    st.session_state.text_query_input = "blue denim jeans"
                    st.rerun()

            with example_col3:
                if st.button("Leather shoes", key="ex3", use_container_width=True):
                    st.session_state.text_query_input = "black leather shoes"
                    st.rerun()

        with col_settings2:
            st.markdown("#### ⚙️ Settings")

            top_k_txt = st.slider(
                "Results",
                min_value=1,
                max_value=50,
                value=12,
                key="topk_txt"
            )

            categories = api_client.get_categories()
            category_txt = st.selectbox(
                "Category",
                ["All"] + categories,
                key="cat_txt"
            )
            category_txt = None if category_txt == "All" else category_txt

            st.markdown("")

            if st.button(
                    "🔍 Search",
                    key="search_txt",
                    type="primary",
                    use_container_width=True
            ):
                if text_query:
                    search_triggered = True
                    search_mode = "text"
                    top_k = top_k_txt
                    category = category_txt
                else:
                    st.error("Please enter a search query!")

    # Tab 3: Hybrid Search
    with tab3:
        st.markdown("### Combine Image + Text Search")

        col_hybrid1, col_hybrid2 = st.columns([2, 1])

        with col_hybrid1:
            st.markdown("##### 1️⃣ Upload Image")
            uploaded_file_hybrid = st.file_uploader(
                "Image",
                type=StreamlitConfig.ALLOWED_EXTENSIONS,
                key="hybrid_uploader",
                label_visibility="collapsed"
            )

            if uploaded_file_hybrid:
                uploaded_image_hybrid = validate_image(uploaded_file_hybrid)
                if uploaded_image_hybrid:
                    display_img = resize_image(uploaded_image_hybrid.copy(), (300, 300))
                    st.image(display_img, width=300)

            st.markdown("##### 2️⃣ Add Text Preference")
            text_query_hybrid = st.text_input(
                "Text",
                placeholder="e.g., red color, striped pattern, casual style...",
                key="hybrid_text",
                label_visibility="collapsed"
            )

        with col_hybrid2:
            st.markdown("#### ⚙️ Settings")

            top_k_hyb = st.slider(
                "Results",
                min_value=1,
                max_value=50,
                value=12,
                key="topk_hyb"
            )

            categories = api_client.get_categories()
            category_hyb = st.selectbox(
                "Category",
                ["All"] + categories,
                key="cat_hyb"
            )
            category_hyb = None if category_hyb == "All" else category_hyb

            with st.expander("⚙️ Advanced"):
                image_weight = st.slider(
                    "Image Weight",
                    0.0, 1.0, 0.7, 0.1,
                    help="Visual similarity"
                )

                text_weight = st.slider(
                    "Text Weight",
                    0.0, 1.0, 0.3, 0.1,
                    help="Text preference"
                )

                use_rerank = st.checkbox(
                    "Enable Reranking",
                    value=True
                )

            st.markdown("")

            if st.button(
                    "🔍 Search",
                    key="search_hyb",
                    type="primary",
                    use_container_width=True
            ):
                if uploaded_file_hybrid and text_query_hybrid:
                    uploaded_image = uploaded_image_hybrid
                    text_query = text_query_hybrid
                    search_triggered = True
                    search_mode = "hybrid"
                    top_k = top_k_hyb
                    category = category_hyb
                else:
                    st.error("Need both image and text!")

    # Tab 4: Stats
    with tab4:
        st.markdown("### 📊 System Status")

        col_stat1, col_stat2 = st.columns(2)

        with col_stat1:
            if st.button("🔄 Refresh", key="refresh_stats"):
                st.cache_resource.clear()

        # API Health
        health = api_client.check_health()

        if health.get("status") == "healthy":
            st.success("✅ System Online")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Device", health.get("device", "N/A"))

            with col2:
                st.metric("Model", health.get("model", "N/A").split("/")[-1])

            with col3:
                st.metric("Embedding Dim", health.get("embedding_dim", "N/A"))

            st.markdown("---")
            st.markdown("#### Vector Index")

            stats = api_client.get_stats()

            if stats.get("success"):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Vectors", f"{stats.get('total_vectors', 0):,}")

                with col2:
                    st.metric("Dimension", stats.get("dimension", 0))

                if stats.get("namespaces"):
                    st.markdown("**Namespaces:**")
                    for ns, info in stats["namespaces"].items():
                        st.write(f"• `{ns}`: {info.get('vector_count', 0):,} vectors")

            # Search History
            if st.session_state.search_history:
                st.markdown("---")
                st.markdown("#### 📜 Recent Searches")

                for idx, hist in enumerate(reversed(st.session_state.search_history[-5:])):
                    st.write(
                        f"{idx + 1}. **{hist['mode']}** - "
                        f"*{hist['query']}* - "
                        f"{hist['results_count']} results "
                        f"({hist['time']:.2f}s)"
                    )
        else:
            st.error("❌ API Unavailable")
            if "error" in health:
                st.error(health["error"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== PERFORM SEARCH ==========
    if search_triggered:
        with st.spinner("🔍 Searching..."):
            start_time = time.time()

            try:
                if search_mode == "image":
                    results = api_client.search_by_image(
                        image=uploaded_image,
                        top_k=top_k,
                        category=category
                    )

                elif search_mode == "text":
                    results = api_client.search_by_text(
                        text_query=text_query,
                        top_k=top_k,
                        category=category
                    )

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

                search_time = time.time() - start_time

                if results.get("success"):
                    st.session_state.search_results = results
                    st.session_state.search_time = search_time

                    # Reset filters
                    st.session_state.active_filters = {
                        'score_threshold': None,
                        'category': None,
                        'sort_by': 'score'
                    }

                    # Add to history
                    st.session_state.search_history.append({
                        'mode': search_mode.capitalize(),
                        'query': text_query if text_query else "Image search",
                        'results_count': len(results.get('results', [])),
                        'time': search_time
                    })

                    st.success(
                        f"✅ Found {len(results.get('results', []))} items in {search_time:.2f}s"
                    )
                else:
                    st.error(f"❌ {results.get('error', 'Search failed')}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # ========== QUICK FILTERS ==========
    if st.session_state.search_results:
        st.markdown("---")
        render_quick_filters(st.session_state.search_results)

    # ========== RESULTS SECTION ==========
    st.markdown("---")

    if st.session_state.search_results:
        # Apply filters
        filtered_items = apply_filters(st.session_state.search_results)

        if filtered_items:
            # Results Header
            col_count, col_meta = st.columns([2, 1])

            with col_count:
                total = len(st.session_state.search_results.get('results', []))
                filtered = len(filtered_items)

                if filtered < total:
                    st.markdown(
                        f'<div class="results-count">'
                        f'🛍️ {filtered} / {total} Products'
                        f'<span class="stats-badge">Filtered</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="results-count">🛍️ {filtered} Products</div>',
                        unsafe_allow_html=True
                    )

            with col_meta:
                st.markdown(
                    f'<div class="results-meta">'
                    f'<span>⏱️ {st.session_state.search_time:.2f}s</span>'
                    f'<span>🔍 {st.session_state.search_results.get("query_type", "N/A")}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("")

            # Product Grid (4 columns)
            num_cols = 4

            for i in range(0, len(filtered_items), num_cols):
                cols = st.columns(num_cols)

                for idx, col in enumerate(cols):
                    if i + idx < len(filtered_items):
                        render_product_card(filtered_items[i + idx], col)

        else:
            # No results after filtering
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">🔍</div>
                    <h3>No Matches Found</h3>
                    <p>Try adjusting your filters or search criteria</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        # Initial empty state
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">👆</div>
                <h3>Start Searching</h3>
                <p>Upload an image or enter a description to find similar fashion items</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Footer
    st.markdown(
        """
        <div class="footer-section">
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                Powered by <strong>SigLIP</strong> + <strong>Pinecone</strong>
            </p>
            <p style="font-size: 0.85rem; color: #9ca3af;">
                🚀 Semantic Search • 🎯 AI Recommendations • ⚡ Real-time Results
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()