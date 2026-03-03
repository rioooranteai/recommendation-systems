const TOP_K_DEFAULT = 12;

const apiClient = new APIClient();
const filterManager = new FilterManager();

const appState = {
    uploadedImage: null,
    searchResults: null,
    searchMode: null,
    searchQuery: null,
    currentSort: 'relevance'
};

const elements = {
    textSearch: document.getElementById('textSearch'),
    imageUpload: document.getElementById('imageUpload'),
    uploadArea: document.getElementById('uploadArea'),
    imagePreview: document.getElementById('imagePreview'),
    previewImg: document.getElementById('previewImg'),
    removeImage: document.getElementById('removeImage'),
    hybridSettings: document.getElementById('hybridSettings'),
    imageWeight: document.getElementById('imageWeight'),
    imageWeightValue: document.getElementById('imageWeightValue'),
    textWeight: document.getElementById('textWeight'),
    textWeightValue: document.getElementById('textWeightValue'),
    useRerank: document.getElementById('useRerank'),
    searchBtn: document.getElementById('searchBtn'),
    resultsHeader: document.getElementById('resultsHeader'),
    resultsCount: document.getElementById('resultsCount'),
    searchModeBadge: document.getElementById('searchModeBadge'),
    searchQuery: document.getElementById('searchQuery'),
    productsGrid: document.getElementById('productsGrid'),
    emptyState: document.getElementById('emptyState'),
    noResults: document.getElementById('noResults'),
    sortSelect: document.getElementById('sortSelect'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toastMessage')
};

function init() {
    setupEventListeners();
    checkAPIHealth();
    loadCategories();

    const searchHeader = document.querySelector('[data-section="search"]');
    const searchSection = document.getElementById('searchSection');
    if (searchHeader && searchSection) {
        searchHeader.classList.add('active');
        searchSection.classList.add('show');
    }
}

function setupEventListeners() {
    elements.uploadArea.addEventListener('click', () => elements.imageUpload.click());
    elements.imageUpload.addEventListener('change', handleImageUpload);
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('drop', handleDrop);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.removeImage.addEventListener('click', removeUploadedImage);
    elements.textSearch.addEventListener('input', updateSearchMode);
    elements.imageWeight.addEventListener('input', (e) => {
        elements.imageWeightValue.textContent = e.target.value;
    });
    elements.textWeight.addEventListener('input', (e) => {
        elements.textWeightValue.textContent = e.target.value;
    });
    elements.searchBtn.addEventListener('click', handleSearch);
    elements.textSearch.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch();
    });
    elements.sortSelect.addEventListener('change', handleSort);
}

// Image upload handling

function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!validateImage(file)) {
        elements.imageUpload.value = '';
        return;
    }

    appState.uploadedImage = file;

    const reader = new FileReader();
    reader.onload = (ev) => {
        elements.previewImg.src = ev.target.result;
        elements.uploadArea.style.display = 'none';
        elements.imagePreview.style.display = 'block';
        updateSearchMode();
    };
    reader.readAsDataURL(file);
}


function validateImage(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    const maxSize = 10 * 1024 * 1024;

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

// Search mode detection

function updateSearchMode() {
    const hasImage = appState.uploadedImage !== null;
    const hasText = elements.textSearch.value.trim() !== '';

    if (hasImage && hasText) {
        elements.hybridSettings.style.display = 'block';
        appState.searchMode = 'hybrid';
    } else {
        elements.hybridSettings.style.display = 'none';
        if (hasImage) appState.searchMode = 'image';
        else if (hasText) appState.searchMode = 'text';
        else appState.searchMode = null;
    }
}

// Main search handler

async function handleSearch() {
    const textQuery = elements.textSearch.value.trim();
    const hasImage = appState.uploadedImage !== null;
    const hasText = textQuery !== '';

    if (!hasImage && !hasText) {
        showToast('Please provide an image or text description', 'error');
        return;
    }

    elements.searchBtn.disabled = true;
    elements.searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
    showLoading();

    const startTime = performance.now();

    try {
        let results;
        let queryDisplay;

        const filters = filterManager.getActiveFilters();

        if (hasImage && hasText) {
            const formData = new FormData();
            formData.append('file', appState.uploadedImage);
            formData.append('text_query', textQuery);
            formData.append('top_k', TOP_K_DEFAULT);

            const imageWeight = parseFloat(elements.imageWeight.value);
            const textWeight = parseFloat(elements.textWeight.value);
            const totalWeight = imageWeight + textWeight;
            const alpha = totalWeight > 0 ? imageWeight / totalWeight : 0.7;

            formData.append('alpha', alpha);

            if (filters.category) formData.append('category', filters.category);

            results = await apiClient.hybridSearch(formData);
            queryDisplay = `"${textQuery}" + Image`;

        } else if (hasImage) {
            const formData = new FormData();
            formData.append('file', appState.uploadedImage);
            formData.append('top_k', TOP_K_DEFAULT);
            formData.append('alpha', 1.0);

            if (filters.category) formData.append('category', filters.category);

            results = await apiClient.searchByImage(formData);
            queryDisplay = 'Visual Search';

        } else {
            const params = {
                text_query: textQuery,
                top_k: TOP_K_DEFAULT,
                category: filters.category || null
            };

            results = await apiClient.searchByText(params);
            queryDisplay = `"${textQuery}"`;
        }

        const responseTime = ((performance.now() - startTime) / 1000).toFixed(2);

        if (results.success) {
            appState.searchResults = results;
            appState.searchQuery = queryDisplay;

            const filteredResults = filterManager.applyFilters(results.results);
            displayResults(filteredResults, queryDisplay, results.search_mode);
            showToast(`Found ${filteredResults.length} products in ${responseTime}s`, 'success');
        } else {
            throw new Error(results.error || 'Search failed');
        }

    } catch (error) {
        console.error('Search error:', error);
        showToast(error.message || 'Search failed. Please try again.', 'error');
        showEmptyState();
    } finally {
        hideLoading();
        elements.searchBtn.disabled = false;
        elements.searchBtn.innerHTML = '<i class="fas fa-search"></i> Search Products';
    }
}

// Results rendering

function displayResults(products, queryDisplay, searchMode) {
    elements.resultsCount.textContent = `${products.length} product(s)`;
    elements.searchModeBadge.textContent = searchMode.toUpperCase();
    elements.searchQuery.textContent = queryDisplay;
    elements.resultsHeader.style.display = 'flex';
    elements.emptyState.style.display = 'none';

    if (products.length === 0) {
        elements.noResults.style.display = 'block';
        elements.productsGrid.innerHTML = '';
        return;
    }

    elements.noResults.style.display = 'none';
    renderProducts(products);

    setTimeout(() => {
        elements.resultsHeader.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

function renderProducts(products) {
    elements.productsGrid.innerHTML = products.map(product => createProductCard(product)).join('');
}

function createProductCard(product) {
    const badges = [];
    if (product.isPreorder) badges.push('<span class="product-badge preorder">PRE-ORDER</span>');
    if (product.onSale) badges.push('<span class="product-badge">SALE</span>');

    return `
        <div class="product-card" data-product-id="${product.id}" onclick="navigateToProduct('${product.id}')">
            <div class="product-image-container">
                ${product.image_path
                    ? `<img src="${product.image_path}"
                             alt="${product.title}"
                             class="product-image"
                             loading="lazy"
                             onerror="this.parentElement.innerHTML='<div class=\'product-image-placeholder\'><i class=\'fas fa-image\'></i><span>Image unavailable</span></div>'">`
                    : `<div class="product-image-placeholder">
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

function navigateToProduct(productId) {
    const product = appState.searchResults.results.find(p => p.id === productId);
    if (product) {
        localStorage.setItem('selectedProduct', JSON.stringify(product));
    }
    window.location.href = `product.html?id=${productId}`;
}

// Sorting

function handleSort() {
    if (!appState.searchResults) return;

    const sortValue = elements.sortSelect.value;
    appState.currentSort = sortValue;

    let products = filterManager.applyFilters([...appState.searchResults.results]);

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

// UI state

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
    if (type === 'success') icon.className = 'toast-icon fas fa-check-circle';
    else if (type === 'error') icon.className = 'toast-icon fas fa-exclamation-circle';
    else icon.className = 'toast-icon fas fa-info-circle';

    setTimeout(() => elements.toast.classList.remove('show'), 3000);
}

// API utilities

async function checkAPIHealth() {
    try {
        const health = await apiClient.healthCheck();
        if (health.mode === 'mock') {
            showToast('Running in DEMO mode (mock data)', 'warning');
        }
    } catch (error) {
        console.error('Backend connection failed:', error);
        showToast('Backend not available', 'warning');
    }
}

async function loadCategories() {
    try {
        const data = await apiClient.getCategories();
        console.log('Categories loaded:', data.categories.length);
    } catch (error) {
        console.error('Failed to load categories:', error);
    }
}

document.addEventListener('DOMContentLoaded', init);
