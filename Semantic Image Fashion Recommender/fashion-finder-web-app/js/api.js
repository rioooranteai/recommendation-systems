class APIClient {
    constructor(baseURL = 'http://127.0.0.1:8000/api') {
        this.baseURL = baseURL;
        this.imageBaseURL = `${baseURL}/images`;
    }

    async searchByImage(formData) {
        const response = await fetch(`${this.baseURL}/search/image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        return {
            success: data.success,
            results: data.results.map(item => this.transformBackendResult(item)),
            count: data.total_results,
            search_mode: data.query_type === 'hybrid' ? 'hybrid' : 'image'
        };
    }

    async searchByText(params) {
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

        return {
            success: data.success,
            results: data.results.map(item => this.transformBackendResult(item)),
            count: data.total_results,
            search_mode: 'text'
        };
    }

    async hybridSearch(formData) {
        const response = await fetch(`${this.baseURL}/search/image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        return {
            success: data.success,
            results: data.results.map(item => this.transformBackendResult(item)),
            count: data.total_results,
            search_mode: 'hybrid'
        };
    }

    async getCategories() {
        const response = await fetch(`${this.baseURL}/categories`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return { categories: data.categories || [] };
    }

    async healthCheck() {
        const response = await fetch(`${this.baseURL}/health`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async fetchImageAsBlob(filename) {
        const response = await fetch(`${this.imageBaseURL}/${filename}`);
        if (!response.ok) throw new Error(`Failed to fetch image: ${filename}`);
        return await response.blob();
    }

    async getRecommendations(productId, filename, description, title, category) {
        const FETCH_K = 6;
        const textQuery = description || title || category || '';
        const promises = [];
        const modes = [];

        if (filename) {
            try {
                const blob = await this.fetchImageAsBlob(filename);
                const imageFile = new File([blob], filename, { type: 'image/jpeg' });
                const formData = new FormData();
                formData.append('file', imageFile);
                formData.append('top_k', FETCH_K);
                formData.append('alpha', 1.0);
                promises.push(this.searchByImage(formData));
                modes.push('image');
            } catch (e) {
                console.error('Failed to fetch image for recommendation:', e);
            }
        }

        if (textQuery) {
            promises.push(this.searchByText({ text_query: textQuery, top_k: FETCH_K }));
            modes.push('text');
        }

        if (promises.length === 0) return { byImage: [], byText: [] };

        const settled = await Promise.allSettled(promises);

        let byImage = [];
        let byText = [];

        settled.forEach((result, index) => {
            if (result.status !== 'fulfilled') {
                console.error(`Recommendation ${modes[index]} search failed:`, result.reason);
                return;
            }
            const filtered = result.value.results.filter(p => p.id !== productId).slice(0, 5);
            if (modes[index] === 'image') byImage = filtered;
            else byText = filtered;
        });

        return { byImage, byText };
    }

    transformBackendResult(item) {
        const productId = item.product_id || item.id;
        const category = item.category || 'Unknown';
        const score = item.score || 0;
        const metadata = item.metadata || {};

        const title = metadata.title
            || metadata.display_name
            || item.display_name
            || this.generateTitle(productId, category);

        const description = metadata.description || item.description || '';

        return {
            id: productId,
            title: title,
            description: description,
            price: metadata.price || this.generateConsistentPrice(productId),
            rating: metadata.rating || this.generateConsistentRating(productId),
            reviewCount: metadata.review_count || this.generateReviewCount(productId),
            stock: metadata.stock || this.generateStock(productId),
            category: category,
            type: this.extractType(category),
            image_path: this.getImagePath(item.filename || productId),
            isPreorder: metadata.isPreorder || false,
            onSale: metadata.onSale || false,
            score: score
        };
    }

    getImagePath(filename) {
        if (!filename) return null;
        return `${this.imageBaseURL}/${filename}`;
    }

    generateTitle(productId, category) {
        return `${category} ${productId}`;
    }

    extractType(category) {
        if (!category) return 'unknown';
        const lower = category.toLowerCase();
        if (lower.includes('tshirt') || lower.includes('t-shirt')) return 'tshirt';
        if (lower.includes('shirt')) return 'shirt';
        if (lower.includes('jean') || lower.includes('trouser') || lower.includes('pant')) return 'pants';
        if (lower.includes('shoe') || lower.includes('sandal') || lower.includes('flip')) return 'shoes';
        if (lower.includes('watch') || lower.includes('bag') || lower.includes('belt') ||
            lower.includes('wallet') || lower.includes('sunglass') || lower.includes('cap') ||
            lower.includes('backpack')) return 'accessories';
        if (lower.includes('short')) return 'shorts';
        return lower.replace(/s$/, '');
    }

    generateConsistentPrice(productId) {
        const hash = this.hashCode(productId);
        return (20 + (hash % 80)).toFixed(2);
    }

    generateConsistentRating(productId) {
        const hash = this.hashCode(productId + 'rating');
        return (3.5 + (hash % 15) / 10).toFixed(1);
    }

    generateReviewCount(productId) {
        const hash = this.hashCode(productId + 'reviews');
        return 10 + (hash % 490);
    }

    generateStock(productId) {
        const hash = this.hashCode(productId + 'stock');
        return 5 + (hash % 95);
    }

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
