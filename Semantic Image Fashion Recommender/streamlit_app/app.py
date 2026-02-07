"""
Streamlit UI for Semantic Fashion Search - E-commerce Style
Complete with quick filters, sorting, and modern design
"""
import streamlit as st
from PIL import Image
import time
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import dengan path relatif (tanpa streamlit_app prefix)
from api_client import APIClient
from streamlit_config import StreamlitConfig
from utils import (
    resize_image,
    validate_image,
    format_score,
    get_result_color
)

# Page configuration
st.set_page_config(
    page_title="Fashion Search 🛍️",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - E-commerce Style
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Search Container */
    .search-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    }

    /* Quick Filters */
    .filters-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 2rem;
    }

    .filter-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
    }

    /* Product Grid */
    .product-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        height: 100%;
        cursor: pointer;
    }

    .product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.25);
        border-color: #667eea;
    }

    .product-image-container {
        position: relative;
        width: 100%;
        padding-bottom: 100%;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
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
        font-size: 3rem;
        color: #adb5bd;
    }

    .score-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        z-index: 10;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }

    .score-high {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .score-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }

    .score-low {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
        color: white;
    }

    .product-title {
        font-weight: 600;
        font-size: 1rem;
        color: #212529;
        margin-bottom: 0.3rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .product-category {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    .product-meta {
        font-size: 0.85rem;
        color: #868e96;
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .product-meta span {
        background: #f8f9fa;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
    }

    /* Results Header */
    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding: 1rem 0;
        border-bottom: 2px solid #e9ecef;
    }

    .results-count {
        font-size: 1.4rem;
        font-weight: 700;
        color: #212529;
    }

    .results-meta {
        color: #6c757d;
        font-size: 0.95rem;
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }

    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
        color: #6c757d;
    }

    .empty-state-icon {
        font-size: 5rem;
        margin-bottom: 1.5rem;
        opacity: 0.5;
    }

    .empty-state h3 {
        font-size: 1.5rem;
        color: #495057;
        margin-bottom: 0.5rem;
    }

    .empty-state p {
        font-size: 1rem;
        color: #868e96;
    }

    /* Stats Badge */
    .stats-badge {
        display: inline-block;
        background: #e7f3ff;
        color: #0066cc;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }

    /* Filter Active State */
    .stButton > button[data-testid="baseButton-secondary"] {
        background: white !important;
        border: 1px solid #dee2e6 !important;
        color: #495057 !important;
    }

    .stButton > button[data-testid="baseButton-secondary"]:hover {
        border-color: #667eea !important;
        color: #667eea !important;
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
    """Render a single product card"""
    with col:
        score = item.get('score', 0)
        score_class = "score-high" if score >= 0.8 else "score-medium" if score >= 0.6 else "score-low"

        # Determine emoji based on category
        category_emoji = {
            'Tshirts': '👕',
            'Shirts': '👔',
            'Jeans': '👖',
            'Trousers': '👔',
            'Shoes': '👟',
            'Casual Shoes': '👟',
            'Formal Shoes': '👞',
            'Sports Shoes': '👟',
            'Watches': '⌚',
            'Bags': '👜',
            'Sunglasses': '🕶️',
        }

        category = item.get('category', 'Unknown')
        emoji = category_emoji.get(category, '👗')

        # Product card HTML
        card_html = f"""
        <div class="product-card">
            <div class="product-image-container">
                <div class="product-image-placeholder">{emoji}</div>
                <span class="score-badge {score_class}">{format_score(score)}</span>
            </div>
            <div class="product-title">{item.get('product_id', 'N/A')}</div>
            <div class="product-category">📦 {category}</div>
            <div class="product-meta">
        """

        if item.get('image_score') is not None:
            card_html += f"<span>🖼️ {format_score(item['image_score'])}</span>"

        if item.get('text_score') is not None:
            card_html += f"<span>📝 {format_score(item['text_score'])}</span>"

        card_html += """
            </div>
        </div>
        """

        st.markdown(card_html, unsafe_allow_html=True)


def render_quick_filters(results):
    """Render quick filter buttons"""
    st.markdown('<div class="filters-container">', unsafe_allow_html=True)
    st.markdown('<div class="filter-label">🔍 Quick Filters</div>', unsafe_allow_html=True)

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
                f"⭐ Top Matches ({high_score_count})",
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
                f"🎯 Good Matches ({medium_score_count})",
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
            "Filter by Category",
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
            "Sort by",
            ["Score (High to Low)", "Category (A-Z)"],
            key="sort_select",
            label_visibility="collapsed"
        )

        if sort_option == "Score (High to Low)":
            st.session_state.active_filters['sort_by'] = 'score'
        else:
            st.session_state.active_filters['sort_by'] = 'category'

    with col5:
        if st.button("🔄 Reset Filters", use_container_width=True, key="reset_filters"):
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
        active_filter_tags.append(f"Category: {st.session_state.active_filters['category']}")

    if active_filter_tags:
        st.markdown(
            f"<div style='margin-top: 1rem; color: #6c757d; font-size: 0.9rem;'>"
            f"Active filters: {' • '.join(active_filter_tags)}</div>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application"""

    # Header
    st.markdown('<h1 class="main-header">🛍️ Fashion Finder</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Discover similar fashion items with AI-powered visual search</p>',
        unsafe_allow_html=True
    )

    # ========== SEARCH SECTION (TOP) ==========
    st.markdown('<div class="search-container">', unsafe_allow_html=True)

    # Search Mode Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Image Search",
        "📝 Text Search",
        "🔄 Hybrid Search",
        "📊 Stats"
    ])

    uploaded_image = None
    text_query = None
    search_triggered = False

    # Tab 1: Image Search
    with tab1:
        col_upload, col_settings = st.columns([2, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload a fashion item image",
                type=StreamlitConfig.ALLOWED_EXTENSIONS,
                key="image_search_uploader",
                help="Drag and drop or click to upload (JPG, PNG)"
            )

            if uploaded_file:
                uploaded_image = validate_image(uploaded_file)
                if uploaded_image:
                    # Resize for display
                    display_img = resize_image(uploaded_image.copy(), (400, 400))
                    st.image(display_img, caption="Query Image")
                else:
                    st.error("❌ Invalid image file")

        with col_settings:
            st.markdown("##### Settings")

            top_k_img = st.slider(
                "Number of results",
                min_value=1,
                max_value=50,
                value=12,
                key="topk_img"
            )

            categories = api_client.get_categories()
            category_img = st.selectbox(
                "Category filter",
                ["All"] + categories,
                key="cat_img"
            )
            category_img = None if category_img == "All" else category_img

            st.markdown("")  # Spacing

            if st.button(
                    "🔍 Search Similar Items",
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
        col_text, col_settings2 = st.columns([2, 1])

        with col_text:
            text_query = st.text_area(
                "What are you looking for?",
                placeholder="e.g., red striped t-shirt, blue denim jeans, leather jacket...",
                key="text_query_input",
                height=100
            )

            # Example queries
            st.markdown("**💡 Example queries:**")
            example_col1, example_col2, example_col3 = st.columns(3)

            with example_col1:
                if st.button("Red striped shirt", key="ex1"):
                    st.session_state.text_query_input = "red striped shirt"
                    st.rerun()

            with example_col2:
                if st.button("Blue denim jeans", key="ex2"):
                    st.session_state.text_query_input = "blue denim jeans"
                    st.rerun()

            with example_col3:
                if st.button("Black leather shoes", key="ex3"):
                    st.session_state.text_query_input = "black leather shoes"
                    st.rerun()

        with col_settings2:
            st.markdown("##### Settings")

            top_k_txt = st.slider(
                "Number of results",
                min_value=1,
                max_value=50,
                value=12,
                key="topk_txt"
            )

            categories = api_client.get_categories()
            category_txt = st.selectbox(
                "Category filter",
                ["All"] + categories,
                key="cat_txt"
            )
            category_txt = None if category_txt == "All" else category_txt

            st.markdown("")  # Spacing

            if st.button(
                    "🔍 Search Products",
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
        col_hybrid1, col_hybrid2 = st.columns([2, 1])

        with col_hybrid1:
            st.markdown("##### 1️⃣ Upload Image")
            uploaded_file_hybrid = st.file_uploader(
                "Choose an image",
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
                "Describe preferences",
                placeholder="e.g., red color, striped pattern, casual style...",
                key="hybrid_text",
                label_visibility="collapsed"
            )

        with col_hybrid2:
            st.markdown("##### Settings")

            top_k_hyb = st.slider(
                "Number of results",
                min_value=1,
                max_value=50,
                value=12,
                key="topk_hyb"
            )

            categories = api_client.get_categories()
            category_hyb = st.selectbox(
                "Category filter",
                ["All"] + categories,
                key="cat_hyb"
            )
            category_hyb = None if category_hyb == "All" else category_hyb

            with st.expander("⚙️ Advanced Settings"):
                image_weight = st.slider(
                    "Image Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Higher = more visual similarity"
                )

                text_weight = st.slider(
                    "Text Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Higher = more text preference"
                )

                use_rerank = st.checkbox(
                    "Enable Reranking",
                    value=True,
                    help="Use text to refine results"
                )

            st.markdown("")  # Spacing

            if st.button(
                    "🔍 Hybrid Search",
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
                    st.error("Please upload image AND enter text!")

    # Tab 4: Stats
    with tab4:
        st.markdown("### 📊 System Statistics")

        col_stat1, col_stat2 = st.columns(2)

        with col_stat1:
            if st.button("🔄 Refresh Stats", key="refresh_stats"):
                st.cache_resource.clear()

        # API Health
        health = api_client.check_health()

        if health.get("status") == "healthy":
            st.success("✅ API is healthy and running")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Device", health.get("device", "N/A"))

            with col2:
                st.metric("Model", health.get("model", "N/A").split("/")[-1])

            with col3:
                st.metric("Embedding Dim", health.get("embedding_dim", "N/A"))

            # Index Stats
            st.markdown("---")
            st.markdown("#### Vector Index Statistics")

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
                        st.write(f"- `{ns}`: {info.get('vector_count', 0):,} vectors")

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
            st.error("❌ API is unavailable")
            if "error" in health:
                st.error(health["error"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== PERFORM SEARCH ==========
    if search_triggered:
        with st.spinner("🔍 Searching for similar items..."):
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
                        f"✅ Found {len(results.get('results', []))} items "
                        f"in {search_time:.2f}s"
                    )
                else:
                    st.error(f"❌ {results.get('error', 'Search failed')}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # ========== QUICK FILTERS ==========
    if st.session_state.search_results:
        st.markdown("---")
        render_quick_filters(st.session_state.search_results)

    # ========== RESULTS SECTION (BOTTOM) ==========
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
                        f'<div class="results-count">🛍️ {filtered} Products Found</div>',
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

            st.markdown("")  # Spacing

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
                    <h3>No items match your filters</h3>
                    <p>Try adjusting or resetting the filters above</p>
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
                <h3>Start Your Search</h3>
                <p>Upload an image or enter a description to find similar fashion items</p>
                <p style="font-size: 0.9rem; margin-top: 1rem;">
                    Choose a search mode from the tabs above
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #adb5bd; font-size: 0.9rem; padding: 2rem 0;">
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                Powered by <strong>SigLIP</strong> embeddings + <strong>Pinecone</strong> vector database
            </p>
            <p style="font-size: 0.85rem; color: #ced4da;">
                🚀 Semantic Image Search | 🎯 AI-Powered Recommendations | ⚡ Lightning Fast
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()