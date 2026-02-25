class APIClient {
    constructor(baseURL = 'http://127.0.0.1:8000/api') {
        this.baseURL = baseURL;
        this.imageBaseURL = 'http://127.0.0.1:8000/static/images';
    }

    /**
     * Search by image only
     * Endpoint: POST /api/search/image
     */
    async searchByImage(formData) {
        try {
            console.log('🔍 Sending image search request to backend...');

            const response = await fetch(`${this.baseURL}/search/image`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('✅ Raw backend response:', data);

            // ✅ Use REAL backend results
            return {
                success: data.success,
                results: data.results.map(item => this.transformBackendResult(item)),
                count: data.total_results,
                search_mode: data.query_type === 'hybrid' ? 'hybrid' : 'image'
            };
        } catch (error) {
            console.error('❌ Search by image error:', error);
            throw error;
        }
    }

    /**
     * Search by text only
     * Endpoint: POST /api/search/text
     */
    async searchByText(params) {
        try {
            console.log('🔍 Sending text search request:', params);

            const formData = new FormData();
            formData.append('text_query', params.text_query);
            formData.append('top_k', params.top_k || 12);

            if (params.category) {
                formData.append('category', params.category);
            }

            const response = await fetch(`${this.baseURL}/search/text`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('✅ Raw backend response:', data);

            // ✅ Use REAL backend results
            return {
                success: data.success,
                results: data.results.map(item => this.transformBackendResult(item)),
                count: data.total_results,
                search_mode: 'text'
            };
        } catch (error) {
            console.error('❌ Search by text error:', error);
            throw error;
        }
    }

    /**
     * Hybrid search (image + text)
     * Endpoint: POST /api/search/image (with text_query)
     */
    async hybridSearch(formData) {
        try {
            console.log('🔍 Sending hybrid search request to backend...');

            const response = await fetch(`${this.baseURL}/search/image`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('✅ Raw backend response:', data);

            // ✅ Use REAL backend results
            return {
                success: data.success,
                results: data.results.map(item => this.transformBackendResult(item)),
                count: data.total_results,
                search_mode: 'hybrid'
            };
        } catch (error) {
            console.error('❌ Hybrid search error:', error);
            throw error;
        }
    }

    /**
     * Get categories from backend
     * Endpoint: GET /api/categories
     */
    async getCategories() {
        try {
            const response = await fetch(`${this.baseURL}/categories`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('✅ Categories from backend:', data.categories);

            return {
                categories: data.categories || []
            };
        } catch (error) {
            console.error('❌ Get categories error:', error);
            throw error;
        }
    }

    /**
     * Health check
     * Endpoint: GET /api/health
     */
    async healthCheck() {
        try {
            const response = await fetch(`${this.baseURL}/health`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const health = await response.json();
            console.log('✅ Backend health check:', health);
            return health;
        } catch (error) {
            console.error('❌ Health check failed:', error);
            throw error;
        }
    }

    /**
     * ✅ Transform REAL backend result to frontend format
     * Backend returns: { product_id, category, score }
     * Frontend needs: { id, title, price, category, type, image_path, isPreorder, onSale, score }
     */
    transformBackendResult(item) {
        console.log('🔄 Transforming backend item:', item);

        // ✅ Get REAL data from backend
        const productId = item.product_id || item.id;
        const category = item.category || 'Unknown';
        const score = item.score || 0;

        // ✅ Extract metadata if backend provides it
        const metadata = item.metadata || {};

        return {
            // ✅ REAL product ID from backend
            id: productId,

            // ✅ Use metadata or generate from product_id + category
            title: metadata.title || this.generateTitle(productId, category),

            // ✅ Use metadata price or generate consistent price
            price: metadata.price || this.generateConsistentPrice(productId),

            // ✅ REAL category from backend
            category: category,

            // ✅ Extract type from category
            type: this.extractType(category),

            // ✅ REAL IMAGE PATH from backend static files
            image_path: this.getRealImagePath(productId),

            // ✅ Use metadata or default to false
            isPreorder: metadata.isPreorder || false,
            onSale: metadata.onSale || false,

            // ✅ REAL similarity score from backend
            score: score
        };
    }

    /**
     * ✅ Get REAL image path from backend static files
     * Images are at: D:\recommendation-systems\Semantic Image Fashion Recommender\data\fashion-mini\data
     * Served via: http://127.0.0.1:8000/static/images/{product_id}.jpg
     */
    getRealImagePath(productId) {
        // ✅ REAL images from backend static server
        return `${this.imageBaseURL}/${productId}.jpg`;
    }

    /**
     * Generate title from product_id and category
     * (Only used if backend doesn't provide metadata.title)
     */
    generateTitle(productId, category) {
        // Simple title: just use product_id and category
        return `${category} ${productId}`;
    }

    /**
     * Extract type from category for filtering
     */
    extractType(category) {
        if (!category) return 'unknown';

        const lower = category.toLowerCase();

        // Map backend categories to frontend types
        if (lower.includes('tshirt') || lower.includes('t-shirt')) return 'tshirt';
        if (lower.includes('shirt')) return 'shirt';
        if (lower.includes('jean') || lower.includes('trouser') || lower.includes('pant')) return 'pants';
        if (lower.includes('shoe') || lower.includes('sandal') || lower.includes('flip')) return 'shoes';
        if (lower.includes('watch') || lower.includes('bag') || lower.includes('belt') ||
            lower.includes('wallet') || lower.includes('sunglass') || lower.includes('cap') ||
            lower.includes('backpack')) return 'accessories';
        if (lower.includes('short')) return 'shorts';

        return lower.replace(/s$/, ''); // Remove trailing 's'
    }

    /**
     * Generate consistent price based on product ID
     * (Only used if backend doesn't provide metadata.price)
     */
    generateConsistentPrice(productId) {
        // Use hash of product_id for consistent pricing
        const hash = this.hashCode(productId);
        const basePrice = 20 + (hash % 80); // $20 - $100
        return basePrice.toFixed(2);
    }

    /**
     * Hash string to number for consistent values
     */
    hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }
}