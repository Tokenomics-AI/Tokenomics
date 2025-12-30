// Test Results Page JavaScript

let allTests = [];

// Load test results on page load
document.addEventListener('DOMContentLoaded', function() {
    loadTestResults();
});

async function loadTestResults() {
    const loadingState = document.getElementById('loading-state');
    const errorState = document.getElementById('error-state');
    const cardsContainer = document.getElementById('test-cards-container');
    
    // Show loading
    loadingState.classList.remove('hidden');
    errorState.classList.add('hidden');
    cardsContainer.classList.add('hidden');
    
    try {
        const response = await fetch('/api/test-results');
        const data = await response.json();
        
        if (data.success) {
            allTests = data.tests;
            displayTestCards(allTests);
            loadingState.classList.add('hidden');
            cardsContainer.classList.remove('hidden');
        } else {
            throw new Error(data.error || 'Failed to load test results');
        }
    } catch (error) {
        console.error('Error loading test results:', error);
        loadingState.classList.add('hidden');
        errorState.classList.remove('hidden');
        document.getElementById('error-message').textContent = error.message;
    }
}

function displayTestCards(tests) {
    const container = document.getElementById('test-cards-container');
    container.innerHTML = '';
    
    if (tests.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No test results found.</p>';
        return;
    }
    
    tests.forEach(test => {
        const card = createTestCard(test);
        container.appendChild(card);
    });
}

function createTestCard(test) {
    const card = document.createElement('div');
    card.className = 'test-card';
    card.onclick = () => showTestDetails(test.id);
    
    // Extract key metrics
    const metrics = test.metrics || {};
    const tokenSavings = metrics.token_savings || metrics.total_token_savings || 'N/A';
    const costSavings = metrics.cost_savings || metrics.total_cost_savings || 'N/A';
    const cacheHitRate = metrics.cache_hit_rate || 'N/A';
    const qualityPreservation = metrics.quality_preservation || metrics.quality_preservation_rate || 'N/A';
    
    // Format metrics
    const formatMetric = (value) => {
        if (value === 'N/A' || value === null || value === undefined) return 'N/A';
        if (typeof value === 'object' && value.value) {
            return `${formatNumber(value.value)} (${value.percent}%)`;
        }
        if (typeof value === 'number') {
            if (value < 1) return value.toFixed(2);
            return formatNumber(value);
        }
        return value;
    };
    
    card.innerHTML = `
        <div class="test-card-header">
            <h3 class="test-card-title">${test.name}</h3>
            <span class="test-card-type ${test.type}">${test.type.replace('_', ' ')}</span>
        </div>
        <p class="test-card-description">${test.description}</p>
        <div class="test-card-metrics">
            <div class="test-metric">
                <span class="test-metric-label">Token Savings</span>
                <span class="test-metric-value">${formatMetric(tokenSavings)}</span>
            </div>
            <div class="test-metric">
                <span class="test-metric-label">Cache Hit Rate</span>
                <span class="test-metric-value">${formatMetric(cacheHitRate)}%</span>
            </div>
            <div class="test-metric">
                <span class="test-metric-label">Cost Savings</span>
                <span class="test-metric-value">${formatMetric(costSavings)}</span>
            </div>
            <div class="test-metric">
                <span class="test-metric-label">Quality</span>
                <span class="test-metric-value">${formatMetric(qualityPreservation)}%</span>
            </div>
        </div>
        <div class="test-card-footer">
            <div class="test-card-date">
                <span>📅</span>
                <span>${test.date || 'Unknown date'}</span>
            </div>
            <span>Click to view details →</span>
        </div>
    `;
    
    return card;
}

function formatNumber(num) {
    if (typeof num !== 'number') return num;
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toLocaleString();
}

async function showTestDetails(testId) {
    const modal = document.getElementById('test-detail-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    
    // Show loading in modal
    modalBody.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Loading test details...</p></div>';
    modal.classList.remove('hidden');
    
    try {
        const response = await fetch(`/api/test-results/${testId}`);
        const data = await response.json();
        
        if (data.success) {
            const test = data.test;
            modalTitle.textContent = test.name;
            modalBody.innerHTML = generateTestDetailsHTML(test);
        } else {
            throw new Error(data.error || 'Failed to load test details');
        }
    } catch (error) {
        console.error('Error loading test details:', error);
        modalBody.innerHTML = `
            <div class="error-state">
                <h3>⚠️ Error Loading Test Details</h3>
                <p>${error.message}</p>
                <button onclick="showTestDetails('${testId}')" class="btn btn-primary">Retry</button>
            </div>
        `;
    }
}

function generateTestDetailsHTML(test) {
    const metrics = test.metrics || {};
    const jsonData = test.json_data || {};
    
    let html = `
        <div class="metric-section">
            <h3 class="metric-section-title">Overview</h3>
            <div class="explanation-section">
                <p><strong>Test Type:</strong> ${test.type.replace('_', ' ').toUpperCase()}</p>
                <p><strong>Description:</strong> ${test.description}</p>
                ${test.date ? `<p><strong>Date:</strong> ${test.date}</p>` : ''}
                ${test.files && test.files.length > 0 ? `
                    <p><strong>Result Files:</strong></p>
                    <ul>
                        ${test.files.map(f => `<li>${f.name} (${f.type})</li>`).join('')}
                    </ul>
                ` : ''}
            </div>
        </div>
    `;
    
    // Key Metrics Section
    html += `
        <div class="metric-section">
            <h3 class="metric-section-title">Key Metrics</h3>
            <div class="metric-grid">
    `;
    
    // Add metrics
    const keyMetrics = [
        { label: 'Total Queries', key: 'total_queries', format: 'number' },
        { label: 'Successful Queries', key: 'successful_queries', format: 'number' },
        { label: 'Token Savings', key: ['token_savings', 'total_token_savings'], format: 'tokens' },
        { label: 'Token Savings %', key: ['token_savings_percent', 'total_token_savings_percent'], format: 'percent' },
        { label: 'Cost Savings', key: ['cost_savings', 'total_cost_savings'], format: 'currency' },
        { label: 'Cost Savings %', key: ['cost_savings_percent', 'total_cost_savings_percent'], format: 'percent' },
        { label: 'Cache Hit Rate', key: 'cache_hit_rate', format: 'percent' },
        { label: 'Quality Preservation', key: ['quality_preservation', 'quality_preservation_rate'], format: 'percent' },
    ];
    
    keyMetrics.forEach(metric => {
        const value = getMetricValue(metrics, metric.key);
        if (value !== null && value !== undefined && value !== 'N/A') {
            html += `
                <div class="metric-card">
                    <div class="metric-card-label">${metric.label}</div>
                    <div class="metric-card-value">${formatMetricValue(value, metric.format)}</div>
                </div>
            `;
        }
    });
    
    html += `</div></div>`;
    
    // Component Analysis
    if (jsonData.aggregated_metrics) {
        const agg = jsonData.aggregated_metrics;
        html += generateComponentAnalysis(agg);
    }
    
    // Detailed Explanation
    html += `
        <div class="metric-section">
            <h3 class="metric-section-title">What This Test Measures</h3>
            <div class="explanation-section">
                ${getTestExplanation(test.type)}
            </div>
        </div>
    `;
    
    // Metric Explanations
    html += `
        <div class="metric-section">
            <h3 class="metric-section-title">Understanding the Metrics</h3>
            <div class="explanation-section">
                ${getMetricExplanations()}
            </div>
        </div>
    `;
    
    // Full Content (if markdown)
    if (test.content) {
        html += `
            <div class="metric-section">
                <h3 class="metric-section-title">Full Test Report</h3>
                <div class="markdown-content">
                    ${convertMarkdownToHTML(test.content)}
                </div>
            </div>
        `;
    }
    
    return html;
}

function generateComponentAnalysis(agg) {
    let html = `
        <div class="metric-section">
            <h3 class="metric-section-title">Component Analysis</h3>
    `;
    
    // Memory Layer
    if (agg.cache_metrics) {
        html += `
            <div class="explanation-section">
                <h3>Memory Layer (Caching)</h3>
                <p><strong>Exact Cache Hits:</strong> ${agg.cache_metrics.exact_cache_hits || 0}</p>
                <p><strong>Semantic Cache Hits:</strong> ${agg.cache_metrics.semantic_cache_hits || 0}</p>
                <p><strong>Cache Misses:</strong> ${agg.cache_metrics.cache_misses || 0}</p>
                <p><strong>Cache Hit Rate:</strong> ${agg.cache_metrics.cache_hit_rate || 0}%</p>
                <p class="metric-card-description">
                    The Memory Layer provides intelligent caching to avoid redundant LLM calls. 
                    Exact cache hits return instant results (0 tokens), while semantic cache hits 
                    match similar queries to previously answered questions.
                </p>
            </div>
        `;
    }
    
    // Orchestrator
    if (agg.orchestrator_metrics) {
        html += `
            <div class="explanation-section">
                <h3>Token Orchestrator</h3>
                <p><strong>Complexity Distribution:</strong> ${JSON.stringify(agg.orchestrator_metrics.complexity_distribution || {})}</p>
                <p><strong>Query Compression:</strong> ${agg.orchestrator_metrics.query_compressed_count || 0} queries</p>
                <p><strong>Context Compression:</strong> ${agg.orchestrator_metrics.context_compressed_count || 0} contexts</p>
                <p class="metric-card-description">
                    The Token Orchestrator analyzes query complexity and allocates tokens optimally 
                    across system prompt, user query, context, and response. It compresses content 
                    when needed to stay within budget limits.
                </p>
            </div>
        `;
    }
    
    // Bandit
    if (agg.bandit_metrics) {
        html += `
            <div class="explanation-section">
                <h3>Bandit Optimizer</h3>
                <p><strong>Strategy Distribution:</strong> ${JSON.stringify(agg.bandit_metrics.strategy_distribution || {})}</p>
                <p><strong>Context-Aware Routing:</strong> ${agg.bandit_metrics.context_aware_routing_count || 0} queries</p>
                <p class="metric-card-description">
                    The Bandit Optimizer learns which model strategy (cheap, balanced, premium) 
                    works best for different query types. It uses multi-armed bandit algorithms 
                    to balance exploration and exploitation.
                </p>
            </div>
        `;
    }
    
    html += `</div>`;
    return html;
}

function getMetricValue(metrics, key) {
    if (Array.isArray(key)) {
        for (const k of key) {
            if (metrics[k] !== undefined && metrics[k] !== null) {
                return metrics[k];
            }
        }
        return null;
    }
    return metrics[key];
}

function formatMetricValue(value, format) {
    if (value === null || value === undefined) return 'N/A';
    
    if (typeof value === 'object' && value.value) {
        value = value.value;
    }
    
    switch (format) {
        case 'percent':
            return typeof value === 'number' ? `${value.toFixed(1)}%` : value;
        case 'currency':
            return typeof value === 'number' ? `$${value.toFixed(4)}` : value;
        case 'tokens':
            return typeof value === 'number' ? formatNumber(value) + ' tokens' : value;
        case 'number':
            return typeof value === 'number' ? formatNumber(value) : value;
        default:
            return value;
    }
}

function getTestExplanation(testType) {
    const explanations = {
        'ab_comparison': `
            <p>This A/B comparison test runs queries through both a baseline (naive) approach and 
            the Tokenomics optimized path. It measures:</p>
            <ul>
                <li><strong>Token Savings:</strong> How many tokens were saved compared to baseline</li>
                <li><strong>Cost Savings:</strong> Financial savings from optimization</li>
                <li><strong>Quality Preservation:</strong> Whether optimized responses maintain quality</li>
                <li><strong>Component Performance:</strong> How each component (Memory, Orchestrator, Bandit) contributed</li>
            </ul>
            <p>The test uses diverse query types (simple, medium, complex, duplicates, paraphrases) 
            to validate optimization across different scenarios.</p>
        `,
        'validation': `
            <p>The Platform Validation Suite tests the interaction between Memory, Routing, and Compression 
            components using controlled prompts across 4 buckets:</p>
            <ul>
                <li><strong>Bucket A (Exact Duplicates):</strong> Tests exact cache efficiency</li>
                <li><strong>Bucket B (Semantic Paraphrases):</strong> Tests semantic cache matching</li>
                <li><strong>Bucket C (Context Injection):</strong> Tests context compression and injection</li>
                <li><strong>Bucket D (Routing Stress):</strong> Tests Bandit strategy selection</li>
            </ul>
            <p>The test includes pass/fail gates to ensure components meet performance thresholds.</p>
        `,
        'diagnostic': `
            <p>Comprehensive diagnostic tests trigger all platform components to verify they work correctly 
            and identify areas for improvement. These tests:</p>
            <ul>
                <li>Exercise all caching mechanisms (exact, semantic, context)</li>
                <li>Test compression on long queries and contexts</li>
                <li>Validate Bandit learning and strategy selection</li>
                <li>Measure component-level savings</li>
            </ul>
        `,
        'benchmark': `
            <p>Benchmark tests use standardized datasets to measure platform performance consistently 
            over time. They help track improvements and regressions.</p>
        `,
        'analysis': `
            <p>Performance impact analysis documents how specific fixes or improvements affected 
            overall platform performance, providing insights into optimization effectiveness.</p>
        `
    };
    
    return explanations[testType] || '<p>This test validates platform functionality and performance.</p>';
}

function getMetricExplanations() {
    return `
        <h4>Token Savings</h4>
        <p>The number of tokens saved by using the optimized path instead of baseline. 
        This includes savings from caching (avoiding LLM calls), compression (reducing input size), 
        and smart routing (selecting efficient models).</p>
        
        <h4>Cost Savings</h4>
        <p>Financial savings calculated from token usage and model pricing. Different models 
        have different costs per token, so smart routing can significantly reduce costs.</p>
        
        <h4>Cache Hit Rate</h4>
        <p>The percentage of queries that were answered from cache (exact or semantic) without 
        calling the LLM. Higher cache hit rates mean more savings and faster responses.</p>
        
        <h4>Quality Preservation</h4>
        <p>The percentage of queries where the optimized response maintained or improved quality 
        compared to baseline. This ensures optimizations don't degrade answer quality.</p>
        
        <h4>Latency</h4>
        <p>Response time in milliseconds. Cache hits provide near-instant responses, while 
        optimized queries may have slightly higher latency due to additional processing 
        (compression, routing decisions).</p>
        
        <h4>Strategy Distribution</h4>
        <p>Shows how often each Bandit strategy (cheap, balanced, premium) was selected. 
        This indicates how well the Bandit is learning to route queries efficiently.</p>
    `;
}

function convertMarkdownToHTML(markdown) {
    // Simple markdown to HTML conversion
    let html = markdown;
    
    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Lists
    html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    
    // Paragraphs
    html = html.split('\n\n').map(p => {
        if (!p.trim()) return '';
        if (p.startsWith('<')) return p;
        return `<p>${p}</p>`;
    }).join('');
    
    return html;
}

function closeModal() {
    document.getElementById('test-detail-modal').classList.add('hidden');
}

// Close modal on outside click
document.addEventListener('click', function(event) {
    const modal = document.getElementById('test-detail-modal');
    if (event.target === modal) {
        closeModal();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeModal();
    }
});







