const apiClient = new APIClient();

const state = {
    product: null
};

const el = {
    skeleton: document.getElementById('productSkeleton'),
    detail: document.getElementById('productDetail'),
    mainImage: document.getElementById('mainImage'),
    imagePlaceholder: document.getElementById('imagePlaceholder'),
    breadcrumbTitle: document.getElementById('breadcrumbTitle'),
    detailCategory: document.getElementById('detailCategory'),
    detailBadges: document.getElementById('detailBadges'),
    detailTitle: document.getElementById('detailTitle'),
    detailStars: document.getElementById('detailStars'),
    detailRating: document.getElementById('detailRating'),
    detailReviewCount: document.getElementById('detailReviewCount'),
    detailPrice: document.getElementById('detailPrice'),
    detailDescription: document.getElementById('detailDescription'),
    metaCategory: document.getElementById('metaCategory'),
    metaType: document.getElementById('metaType'),
    metaStock: document.getElementById('metaStock'),
    btnAddCart: document.getElementById('btnAddCart'),
    btnWishlist: document.getElementById('btnWishlist'),
    recommendationsSection: document.getElementById('recommendationsSection'),
    recSkeleton: document.getElementById('recSkeleton'),
    recImageGrid: document.getElementById('recImageGrid'),
    recTextGrid: document.getElementById('recTextGrid'),
    recImageBlock: document.getElementById('recImageBlock'),
    recTextBlock: document.getElementById('recTextBlock'),
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toastMessage')
};

function init() {
    const params = new URLSearchParams(window.location.search);
    const productId = params.get('id');

    if (!productId) {
        window.location.href = 'index.html';
        return;
    }

    const cached = localStorage.getItem('selectedProduct');
    if (cached) {
        try {
            const product = JSON.parse(cached);
            if (product.id === productId) {
                loadProductDetail(product);
                return;
            }
        } catch (e) {
            // cache corrupt
        }
    }

    showToast('Product data not found. Please search again.', 'error');
    setTimeout(() => { window.location.href = 'index.html'; }, 2000);
}

function loadProductDetail(product) {
    state.product = product;

    el.skeleton.style.display = 'none';
    el.detail.style.display = 'grid';

    document.title = `${product.title} - Vale Mart`;
    el.breadcrumbTitle.textContent = product.title;

    if (product.image_path) {
        el.mainImage.src = product.image_path;
        el.mainImage.alt = product.title;
        el.mainImage.onerror = () => {
            el.mainImage.style.display = 'none';
            el.imagePlaceholder.style.display = 'flex';
        };
    } else {
        el.mainImage.style.display = 'none';
        el.imagePlaceholder.style.display = 'flex';
    }

    el.detailCategory.textContent = product.category;

    const badges = [];
    if (product.isPreorder) badges.push('<span class="product-badge preorder">PRE-ORDER</span>');
    if (product.onSale) badges.push('<span class="product-badge">SALE</span>');
    el.detailBadges.innerHTML = badges.join('');

    el.detailTitle.textContent = product.title;

    const rating = parseFloat(product.rating) || 0;
    el.detailStars.innerHTML = renderStars(rating);
    el.detailRating.textContent = rating.toFixed(1);
    el.detailReviewCount.textContent = `(${product.reviewCount || 0} reviews)`;

    el.detailPrice.textContent = `$${product.price}`;
    el.detailDescription.textContent = product.description || 'No description available.';

    el.metaCategory.textContent = product.category;
    el.metaType.textContent = product.type || '-';
    el.metaStock.textContent = product.stock > 0 ? `${product.stock} units` : 'Out of stock';

    el.btnAddCart.addEventListener('click', () => {
        showToast(`${product.title} added to cart`, 'success');
    });

    el.btnWishlist.addEventListener('click', () => {
        el.btnWishlist.classList.toggle('active');
        const isActive = el.btnWishlist.classList.contains('active');
        el.btnWishlist.querySelector('i').className = isActive ? 'fas fa-heart' : 'far fa-heart';
        showToast(isActive ? 'Added to wishlist' : 'Removed from wishlist', 'success');
    });

    loadRecommendations(product);
}

function renderStars(rating) {
    const stars = [];
    for (let i = 1; i <= 5; i++) {
        if (i <= Math.floor(rating)) {
            stars.push('<i class="star filled fas fa-star"></i>');
        } else if (i - rating < 1 && i - rating > 0) {
            stars.push('<i class="star half fas fa-star"></i>');
        } else {
            stars.push('<i class="star far fa-star"></i>');
        }
    }
    return stars.join('');
}

async function loadRecommendations(product) {
    el.recSkeleton.style.display = 'block';

    const filename = product.image_path ? product.image_path.split('/').pop() : null;

    try {
        const { byImage, byText } = await apiClient.getRecommendations(
            product.id,
            filename,
            product.description,
            product.title,
            product.category
        );

        el.recSkeleton.style.display = 'none';
        el.recommendationsSection.style.display = 'flex';

        if (byImage.length > 0) {
            el.recImageGrid.innerHTML = byImage.map(p => createRecCard(p)).join('');
            attachRecCardListeners(el.recImageGrid, byImage);
        } else {
            el.recImageBlock.style.display = 'none';
        }

        if (byText.length > 0) {
            el.recTextGrid.innerHTML = byText.map(p => createRecCard(p)).join('');
            attachRecCardListeners(el.recTextGrid, byText);
        } else {
            el.recTextBlock.style.display = 'none';
        }

    } catch (error) {
        el.recSkeleton.style.display = 'none';
        console.error('Failed to load recommendations:', error);
    }
}

function createRecCard(product) {
    const badges = [];
    if (product.isPreorder) badges.push('<span class="product-badge preorder">PRE-ORDER</span>');
    if (product.onSale) badges.push('<span class="product-badge">SALE</span>');

    return `
        <div class="product-card" data-product-id="${product.id}">
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

function attachRecCardListeners(container, products) {
    container.querySelectorAll('.product-card').forEach((card, index) => {
        card.addEventListener('click', () => {
            const product = products[index];
            localStorage.setItem('selectedProduct', JSON.stringify(product));
            window.location.href = `product.html?id=${product.id}`;
        });
    });
}

function showToast(message, type = 'success') {
    el.toastMessage.textContent = message;
    el.toast.className = `toast show ${type}`;

    const icon = el.toast.querySelector('.toast-icon');
    if (type === 'success') icon.className = 'toast-icon fas fa-check-circle';
    else if (type === 'error') icon.className = 'toast-icon fas fa-exclamation-circle';
    else icon.className = 'toast-icon fas fa-info-circle';

    setTimeout(() => el.toast.classList.remove('show'), 3000);
}

document.addEventListener('DOMContentLoaded', init);
