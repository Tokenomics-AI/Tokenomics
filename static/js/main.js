// Main JavaScript utilities

// Utility functions
const utils = {
    formatNumber: (num) => {
        return new Intl.NumberFormat().format(num);
    },
    
    formatDate: (dateString) => {
        const date = new Date(dateString);
        return date.toLocaleString();
    },
    
    formatDuration: (ms) => {
        if (ms < 1000) return `${Math.round(ms)} ms`;
        return `${(ms / 1000).toFixed(2)} s`;
    },
};

// API client
const api = {
    baseUrl: '',
    
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        };
        
        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }
        
        try {
            const response = await fetch(url, config);
            
            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                const text = await response.text();
                throw new Error(`Server returned non-JSON response: ${text.substring(0, 100)}`);
            }
            
            const data = await response.json();
            
            if (!response.ok) {
                const error = new Error(data.error || 'Request failed');
                error.status = response.status;
                error.data = data;
                throw error;
            }
            
            return data;
        } catch (error) {
            console.error('API request failed:', error);
            // Provide more detailed error information
            if (error.message === 'Failed to fetch' || error.message.includes('NetworkError')) {
                const detailedError = new Error(`Cannot connect to server at ${url}. Is the Flask server running on port 5000?`);
                detailedError.originalError = error;
                throw detailedError;
            }
            throw error;
        }
    },
    
    async runExperiment(queries, mode, numQueries) {
        return this.request('/api/run', {
            method: 'POST',
            body: { queries, mode, num_queries: numQueries },
        });
    },
    
    async getStatus() {
        return this.request('/api/status');
    },
    
    async getRuns() {
        return this.request('/api/runs');
    },
    
    async getRun(runId) {
        return this.request(`/api/runs/${runId}`);
    },
    
    async getStats() {
        return this.request('/api/stats');
    },
    
    async compare(query) {
        return this.request('/api/compare', {
            method: 'POST',
            body: { query },
        });
    },
};

