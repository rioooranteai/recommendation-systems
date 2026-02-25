class FilterManager {
    constructor() {
        this.activeFilters = {
            category: '',
            types: [],
            preorder: false,
            sale: false
        };

        this.init();
    }

    init() {
        this.setupFilterSections();
        this.setupCategoryPills();
        this.setupCheckboxes();
    }

    /**
     * Setup collapsible filter sections
     */
    setupFilterSections() {
        const headers = document.querySelectorAll('.filter-section-header');

        headers.forEach(header => {
            header.addEventListener('click', () => {
                const section = header.dataset.section;
                const content = document.getElementById(`${section}Section`);

                // Toggle active state
                header.classList.toggle('active');
                content.classList.toggle('show');
            });
        });
    }

    /**
     * Setup category pills
     */
    setupCategoryPills() {
        const pills = document.querySelectorAll('.category-pill');

        pills.forEach(pill => {
            pill.addEventListener('click', () => {
                // Remove active from all
                pills.forEach(p => p.classList.remove('active'));

                // Add active to clicked
                pill.classList.add('active');

                // Update filter
                this.activeFilters.category = pill.dataset.category || '';
                this.updateFiltersCount();
            });
        });
    }

    /**
     * Setup checkboxes
     */
    setupCheckboxes() {
        // Type checkboxes
        const typeCheckboxes = document.querySelectorAll('input[name="type"]');
        typeCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateTypeFilters();
                this.updateFiltersCount();
            });
        });

        // Preorder checkbox
        const preorderCheckbox = document.getElementById('preorderOnly');
        if (preorderCheckbox) {
            preorderCheckbox.addEventListener('change', () => {
                this.activeFilters.preorder = preorderCheckbox.checked;
                this.updateFiltersCount();
            });
        }

        // Sale checkbox
        const saleCheckbox = document.getElementById('saleOnly');
        if (saleCheckbox) {
            saleCheckbox.addEventListener('change', () => {
                this.activeFilters.sale = saleCheckbox.checked;
                this.updateFiltersCount();
            });
        }
    }

    /**
     * Update type filters
     */
    updateTypeFilters() {
        const checked = document.querySelectorAll('input[name="type"]:checked');
        this.activeFilters.types = Array.from(checked).map(cb => cb.value);
    }

    /**
     * Update filters count badge
     */
    updateFiltersCount() {
        const count = this.getActiveFiltersCount();
        const badge = document.getElementById('filtersCount');
        if (badge) {
            badge.textContent = count;
        }
    }

    /**
     * Get active filters count
     */
    getActiveFiltersCount() {
        let count = 0;

        if (this.activeFilters.category) count++;
        count += this.activeFilters.types.length;
        if (this.activeFilters.preorder) count++;
        if (this.activeFilters.sale) count++;

        return count;
    }

    /**
     * Apply filters to results
     */
    applyFilters(products) {
        let filtered = [...products];

        // Filter by category
        if (this.activeFilters.category) {
            filtered = filtered.filter(p =>
                p.category === this.activeFilters.category
            );
        }

        // Filter by type
        if (this.activeFilters.types.length > 0) {
            filtered = filtered.filter(p =>
                this.activeFilters.types.includes(p.type)
            );
        }

        // Filter by preorder
        if (this.activeFilters.preorder) {
            filtered = filtered.filter(p => p.isPreorder);
        }

        // Filter by sale
        if (this.activeFilters.sale) {
            filtered = filtered.filter(p => p.onSale);
        }

        return filtered;
    }

    /**
     * Get active filters
     */
    getActiveFilters() {
        return { ...this.activeFilters };
    }

    /**
     * Clear all filters
     */
    clearAllFilters() {
        // Reset category
        this.activeFilters.category = '';
        const pills = document.querySelectorAll('.category-pill');
        pills.forEach(p => p.classList.remove('active'));
        document.querySelector('.category-pill[data-category=""]')?.classList.add('active');

        // Reset types
        this.activeFilters.types = [];
        document.querySelectorAll('input[name="type"]').forEach(cb => cb.checked = false);

        // Reset preorder
        this.activeFilters.preorder = false;
        const preorderCheckbox = document.getElementById('preorderOnly');
        if (preorderCheckbox) preorderCheckbox.checked = false;

        // Reset sale
        this.activeFilters.sale = false;
        const saleCheckbox = document.getElementById('saleOnly');
        if (saleCheckbox) saleCheckbox.checked = false;

        this.updateFiltersCount();
    }
}