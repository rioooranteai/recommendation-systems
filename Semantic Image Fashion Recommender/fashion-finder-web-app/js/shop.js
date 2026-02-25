const USE_MOCK_API = false; // ✅ SET TO FALSE untuk gunakan real backend
const TOP_K_DEFAULT = 12;

// Initialize API Client
const apiClient = new APIClient();

// Initialize Filter Manager
const filterManager = new FilterManager();

// ============================================
// APPLICATION STATE
// ============================================
const appState = {
    uploadedImage: null,
    searchResults: null,
    searchMode: null,
    searchQuery: null,
    currentSort: 'relevance'
};

// ============================================
// DOM ELEMENTS
// ============================================
const elements = {
    // Search inputs
    textSearch: document.getElementById('textSearch'),
    imageUpload: document.getElementById('imageUpload'),
    uploadArea: document.getElementById('uploadArea'),
    imagePreview: document.getElementById('imagePreview'),
    previewImg: document.getElementById('previewImg'),
    removeImage: document.getElementById('removeImage'),

    // Hybrid settings
    hybridSettings: document.getElementById('hybridSettings'),
    imageWeight: document.getElementById('imageWeight'),
    imageWeightValue: document.getElementById('imageWeightValue'),
    textWeight: document.getElementById('textWeight'),
    textWeightValue: document.getElementById('textWeightValue'),
    useRerank: document.getElementById('useRerank'),

    // Search button
    searchBtn: document.getElementById('searchBtn'),

    // Results
    resultsHeader: document.getElementById('resultsHeader'),
    resultsCount: document.getElementById('resultsCount'),
    searchModeBadge: document.getElementById('searchModeBadge'),
    searchQuery: document.getElementById('searchQuery'),
    productsGrid: document.getElementById('productsGrid'),
    emptyState: document.getElementById('emptyState'),
    noResults: document.getElementById('noResults'),

    // Sort
    sortSelect: document.getElementById('sortSelect'),

    // Loading & Toast
    loadingOverlay: document.getElementById('loadingOverlay'),
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toastMessage')
};

// ============================================
// INITIALIZATION
// ============================================
function init() {
    console.log('🚀 Initializing Fashion Empire Shop...');
    setupEventListeners();
    checkAPIHealth();
    loadCategories();

    // Open search section by default
    const searchHeader = document.querySelector('[data-section="search"]');
    const searchSection = document.getElementById('searchSection');
    if (searchHeader && searchSection) {
        searchHeader.classList.add('active');
        searchSection.classList.add('show');
    }

    console.log('✅ Shop initialized');
}

// ============================================
// EVENT LISTENERS
// ============================================
function setupEventListeners() {
    // Image upload
    elements.uploadArea.addEventListener('click', () => {
        elements.imageUpload.click();
    });

    elements.imageUpload.addEventListener('change', handleImageUpload);

    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('drop', handleDrop);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);

    // Remove image
    elements.removeImage.addEventListener('click', removeUploadedImage);

    // Text input - detect hybrid mode
    elements.textSearch.addEventListener('input', updateSearchMode);

    // Slider updates
    elements.imageWeight.addEventListener('input', (e) => {
        elements.imageWeightValue.textContent = e.target.value;
    });

    elements.textWeight.addEventListener('input', (e) => {
        elements.textWeightValue.textContent = e.target.value;
    });

    // Search button
    elements.searchBtn.addEventListener('click', handleSearch);

    // Enter key to search
    elements.textSearch.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });

    // Sort
    elements.sortSelect.addEventListener('change', handleSort);
}

// ============================================
// IMAGE UPLOAD HANDLING
// ============================================
function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Validate
    if (!validateImage(file)) {
        elements.imageUpload.value = '';
        return;
    }

    // Store file
    appState.uploadedImage = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImg.src = e.target.result;
        elements.uploadArea.style.display = 'none';
        elements.imagePreview.style.display = 'block';
        updateSearchMode();
    };
    reader.readAsDataURL(file);

    showToast('Image uploaded successfully', 'success');
}

function validateImage(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!validTypes.includes(file.type)) {
        showToast('Invalid file type. Please upload JPG, PNG, or WEBP', 'error');
        return false;
    }

    if (file.size > maxSize) {
        showToast('File too large. Maximum size is 10MB', 'error');
        return false;
    }

    return true;
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadArea.style.borderColor = 'var(--color-primary)';
    elements.uploadArea.style.background = 'var(--color-bg)';
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadArea.style.borderColor = '';
    elements.uploadArea.style.background = '';
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();

    elements.uploadArea.style.borderColor = '';
    elements.uploadArea.style.background = '';

    const file = e.dataTransfer.files[0];
    if (file) {
        elements.imageUpload.files = e.dataTransfer.files;
        handleImageUpload({ target: elements.imageUpload });
    }
}

function removeUploadedImage() {
    appState.uploadedImage = null;
    elements.imageUpload.value = '';
    elements.previewImg.src = '';
    elements.uploadArea.style.display = 'block';
    elements.imagePreview.style.display = 'none';
    updateSearchMode();
}

// ============================================
// SEARCH MODE DETECTION
// ============================================
function updateSearchMode() {
    const hasImage = appState.uploadedImage !== null;
    const hasText = elements.textSearch.value.trim() !== '';

    if (hasImage && hasText) {
        elements.hybridSettings.style.display = 'block';
        appState.searchMode = 'hybrid';
    } else {
        elements.hybridSettings.style.display = 'none';
        if (hasImage) {
            appState.searchMode = 'image';
        } else if (hasText) {
            appState.searchMode = 'text';
        } else {
            appState.searchMode = null;
        }
    }
}

// ============================================
// MAIN SEARCH HANDLER
// ============================================
async function handleSearch() {
    const textQuery = elements.textSearch.value.trim();
    const hasImage = appState.uploadedImage !== null;
    const hasText = textQuery !== '';

    // Validation
    if (!hasImage && !hasText) {
        showToast('Please provide an image or text description', 'error');
        return;
    }

    // Disable button
    elements.searchBtn.disabled = true;
    elements.searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';

    // Show loading
    showLoading();

    const startTime = performance.now();

    try {
        let results;
        let queryDisplay;

        const filters = filterManager.getActiveFilters();

        if (hasImage && hasText) {
            // ========== HYBRID SEARCH ==========
            const formData = new FormData();
            formData.append('file', appState.uploadedImage); // ✅ Backend expects 'file'
            formData.append('text_query', textQuery);
            formData.append('top_k', TOP_K_DEFAULT);

            // ✅ Calculate alpha from weights
            const imageWeight = parseFloat(elements.imageWeight.value);
            const textWeight = parseFloat(elements.textWeight.value);
            const totalWeight = imageWeight + textWeight;
            const alpha = totalWeight > 0 ? imageWeight / totalWeight : 0.7;

            formData.append('alpha', alpha);

            console.log(`🔀 Hybrid search: alpha=${alpha.toFixed(2)} (image=${imageWeight}, text=${textWeight})`);

            if (filters.category) {
                formData.append('category', filters.category);
            }

            results = await apiClient.hybridSearch(formData);
            queryDisplay = `"${textQuery}" + Image`;

        } else if (hasImage) {
            // ========== IMAGE SEARCH ==========
            const formData = new FormData();
            formData.append('file', appState.uploadedImage); // ✅ Backend expects 'file'
            formData.append('top_k', TOP_K_DEFAULT);
            formData.append('alpha', 1.0); // Pure image search

            console.log('🖼️ Image-only search');

            if (filters.category) {
                formData.append('category', filters.category);
            }

            results = await apiClient.searchByImage(formData);
            queryDisplay = 'Visual Search';

        } else {
            // ========== TEXT SEARCH ==========
            const params = {
                text_query: textQuery,
                top_k: TOP_K_DEFAULT,
                category: filters.category || null
            };

            console.log('📝 Text-only search:', params);

            results = await apiClient.searchByText(params);
            queryDisplay = `"${textQuery}"`;
        }

        const endTime = performance.now();
        const responseTime = ((endTime - startTime) / 1000).toFixed(2);

        console.log(`✅ Search completed in ${responseTime}s:`, results);

        // Handle results
        if (results.success) {
            appState.searchResults = results;
            appState.searchQuery = queryDisplay;

            // Apply additional client-side filters
            let filteredResults = filterManager.applyFilters(results.results);

            displayResults(filteredResults, queryDisplay, results.search_mode);
            showToast(`Found ${filteredResults.length} products in ${responseTime}s`, 'success');

        } else {
            throw new Error(results.error || 'Search failed');
        }

    } catch (error) {
        console.error('❌ Search error:', error);
        showToast(error.message || 'Search failed. Please try again.', 'error');
        showEmptyState();
    } finally {
        hideLoading();
        elements.searchBtn.disabled = false;
        elements.searchBtn.innerHTML = '<i class="fas fa-search"></i> Search Products';
    }
}

// ============================================
// DISPLAY RESULTS
// ============================================
function displayResults(products, queryDisplay, searchMode) {
    // Update header
    elements.resultsCount.textContent = `${products.length} product(s)`;
    elements.searchModeBadge.textContent = searchMode.toUpperCase();
    elements.searchQuery.textContent = queryDisplay;

    // Show/hide sections
    elements.resultsHeader.style.display = 'flex';
    elements.emptyState.style.display = 'none';

    if (products.length === 0) {
        elements.noResults.style.display = 'block';
        elements.productsGrid.innerHTML = '';
        return;
    }

    elements.noResults.style.display = 'none';

    // Render products
    renderProducts(products);

    // Scroll to results
    setTimeout(() => {
        elements.resultsHeader.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

function renderProducts(products) {
    elements.productsGrid.innerHTML = products.map(product => createProductCard(product)).join('');
}

function createProductCard(product) {
    const matchClass = getMatchClass(product.score);
    const scorePercent = Math.round(product.score * 100);

    const badges = [];
    if (product.isPreorder) badges.push('<span class="product-badge preorder">PRE-ORDER</span>');
    if (product.onSale) badges.push('<span class="product-badge">SALE</span>');

    return `
        <div class="product-card" data-product-id="${product.id}">
            <div class="product-image-container">
                ${product.image_path ?
                    `<img src="${product.image_path}"
                          alt="${product.title}"
                          class="product-image"
                          loading="lazy"
                          onerror="this.parentElement.innerHTML='<div class=\\'product-image-placeholder\\'><i class=\\'fas fa-image\\'></i><span>Image unavailable</span></div>'">` :
                    `<div class="product-image-placeholder">
                        <i class="fas fa-image"></i>
                        <span>Image unavailable</span>
                    </div>`
                }
                ${badges.length > 0 ? `<div class="product-badges">${badges.join('')}</div>` : ''}
            </div>
            <div class="product-info">
                <h3 class="product-title">${product.title}</h3>
                <div class="product-price">$${product.price}</div>
            </div>
        </div>
    `;
}

function getMatchClass(score) {
    if (score >= 0.8) return 'high';
    if (score >= 0.5) return 'medium';
    return '';
}

// ============================================
// SORTING
// ============================================
function handleSort() {
    if (!appState.searchResults) return;

    const sortValue = elements.sortSelect.value;
    appState.currentSort = sortValue;

    console.log('🔀 Sorting by:', sortValue);

    let products = [...appState.searchResults.results];

    // Apply filters first
    products = filterManager.applyFilters(products);

    // Sort
    switch (sortValue) {
        case 'relevance':
            products.sort((a, b) => b.score - a.score);
            break;
        case 'price-asc':
            products.sort((a, b) => parseFloat(a.price) - parseFloat(b.price));
            break;
        case 'price-desc':
            products.sort((a, b) => parseFloat(b.price) - parseFloat(a.price));
            break;
        case 'newest':
            products.sort((a, b) => b.id.localeCompare(a.id));
            break;
    }

    renderProducts(products);
}

// ============================================
// UI STATE FUNCTIONS
// ============================================
function showEmptyState() {
    elements.resultsHeader.style.display = 'none';
    elements.emptyState.style.display = 'block';
    elements.noResults.style.display = 'none';
    elements.productsGrid.innerHTML = '';
}

function showLoading() {
    elements.loadingOverlay.classList.add('show');
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('show');
}

function showToast(message, type = 'success') {
    elements.toastMessage.textContent = message;
    elements.toast.className = `toast show ${type}`;

    const icon = elements.toast.querySelector('.toast-icon');
    switch(type) {
        case 'success':
            icon.className = 'toast-icon fas fa-check-circle';
            break;
        case 'error':
            icon.className = 'toast-icon fas fa-exclamation-circle';
            break;
        default:
            icon.className = 'toast-icon fas fa-info-circle';
    }

    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

// ============================================
// API UTILITIES
// ============================================
async function checkAPIHealth() {
    try {
        const health = await apiClient.healthCheck();

        if (health.status === 'healthy' || health.status === 'ok') {
            console.log('✅ Backend is healthy:', health);

            if (health.mode === 'mock') {
                showToast('Running in DEMO mode (mock data)', 'warning');
            }
        } else {
            console.warn('⚠️ Backend health check returned:', health);
        }
    } catch (error) {
        console.error('❌ Backend connection failed:', error);
        showToast('Backend not available - using mock data', 'warning');
    }
}

async function loadCategories() {
    try {
        const data = await apiClient.getCategories();

        if (data.categories && data.categories.length > 0) {
            console.log('✅ Categories loaded:', data.categories.length);
            // You can dynamically populate category filter here if needed
        }
    } catch (error) {
        console.error('⚠️ Failed to load categories:', error);
    }
}

// ============================================
// INITIALIZE APP
// ============================================
document.addEventListener('DOMContentLoaded', init);