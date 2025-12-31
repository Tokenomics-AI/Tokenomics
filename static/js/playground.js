// Tokenomics Playground

let currentMode = 'explore';
let exploreHistory = [];
let testResults = [];
let valueStats = {
    totalCostSaved: 0,
    totalTokensSaved: 0,
    queriesCount: 0,
    strategyCounts: {},
    cacheHits: 0,
    cacheMisses: 0,
};

// Initialize playground
document.addEventListener('DOMContentLoaded', () => {
    initializePlayground();
});

function initializePlayground() {
    // Mode toggle
    const modeButtons = document.querySelectorAll('.mode-btn');
    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            switchMode(mode);
        });
    });
    
    // Explore mode handlers
    setupExploreMode();
    
    // Test mode handlers
    setupTestMode();
    
    // Load saved state
    loadState();
}

function switchMode(mode) {
    currentMode = mode;
    
    // Update button states
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    
    // Show/hide modes
    document.getElementById('explore-mode').classList.toggle('hidden', mode !== 'explore');
    document.getElementById('test-mode').classList.toggle('hidden', mode !== 'test');
    
    // Save state
    saveState();
}

function setupExploreMode() {
    const chatInput = document.getElementById('explore-chat-input');
    const sendBtn = document.getElementById('explore-send-btn');
    
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleExploreQuery();
            }
        });
        
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = chatInput.scrollHeight + 'px';
        });
    }
    
    if (sendBtn) {
        sendBtn.addEventListener('click', handleExploreQuery);
    }
}

function setupTestMode() {
    const runBtn = document.getElementById('test-run-btn');
    const clearBtn = document.getElementById('test-clear-btn');
    
    if (runBtn) {
        runBtn.addEventListener('click', handleTestRun);
    }
    
    if (clearBtn) {
        clearBtn.addEventListener('click', handleTestClear);
    }
}

async function handleExploreQuery() {
    const chatInput = document.getElementById('explore-chat-input');
    const query = chatInput?.value.trim();
    
    if (!query) return;
    
    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Add user message
    addChatMessage('user', query);
    
    // Show loading
    const loadingMsg = addChatMessage('assistant', 'Analyzing query...', null, true);
    
    // Disable input
    setExploreInputEnabled(false);
    
    try {
        const response = await api.compare(query);
        
        // Remove loading message
        loadingMsg.remove();
        
        // Add AI response
        addChatMessage('assistant', response.tokenomics.response);
        
        // Add value card
        const valueCard = renderValueCard(response.tokenomics, response.comparison);
        const messagesContainer = document.getElementById('explore-chat-messages');
        messagesContainer.appendChild(valueCard);
        
        // Update value dashboard
        updateValueDashboard(response.comparison, response.tokenomics);
        
        // Save to history
        exploreHistory.push({ query, response });
        saveState();
        
    } catch (error) {
        console.error('Explore query error:', error);
        loadingMsg.remove();
        addChatMessage('assistant', `Error: ${error.message || 'Failed to get response'}`);
    } finally {
        setExploreInputEnabled(true);
        scrollChatToBottom();
    }
}

function addChatMessage(role, content, isLoading = false) {
    const messagesContainer = document.getElementById('explore-chat-messages');
    if (!messagesContainer) return null;
    
    // Remove welcome message if present
    const welcomeMsg = messagesContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message chat-message-${role}`;
    if (isLoading) {
        messageDiv.className += ' loading';
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">${escapeHtml(content)}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    scrollChatToBottom();
    
    return messageDiv;
}

function renderValueCard(tokenomics, comparison) {
    const template = document.getElementById('value-card-template');
    if (!template) return document.createElement('div');
    
    const card = template.content.cloneNode(true).querySelector('.value-card');
    
    // Cost savings
    const costSavings = card.querySelector('#value-cost-savings');
    const costPercent = card.querySelector('#value-cost-percent');
    if (costSavings && costPercent) {
        const savings = comparison.cost_savings || 0;
        costSavings.textContent = formatCost(Math.abs(savings));
        costPercent.textContent = `${comparison.cost_savings_percent || 0}% saved`;
        costSavings.className = `metric-value savings ${savings >= 0 ? 'positive' : 'negative'}`;
    }
    
    // Token savings
    const tokenSavings = card.querySelector('#value-token-savings');
    const tokenDesc = card.querySelector('#value-token-desc');
    if (tokenSavings && tokenDesc) {
        const savings = comparison.token_savings || 0;
        tokenSavings.textContent = `${Math.abs(savings).toLocaleString()} tokens`;
        tokenDesc.textContent = `${comparison.token_savings_percent || 0}% fewer than baseline`;
    }
    
    // Latency
    const latency = card.querySelector('#value-latency');
    const latencyDesc = card.querySelector('#value-latency-desc');
    if (latency && latencyDesc) {
        const reduction = comparison.latency_reduction || 0;
        latency.textContent = `${Math.abs(reduction).toFixed(0)}ms`;
        latencyDesc.textContent = `${comparison.latency_reduction_percent || 0}% faster`;
    }
    
    // Decisions
    const strategy = card.querySelector('#value-strategy');
    const model = card.querySelector('#value-model');
    const cacheItem = card.querySelector('#value-cache-item');
    
    if (strategy) {
        strategy.textContent = capitalize(tokenomics.strategy || 'none');
        strategy.className = `badge badge-${tokenomics.strategy || 'none'}`;
    }
    
    if (model) {
        model.textContent = tokenomics.model || 'unknown';
    }
    
    if (tokenomics.cache_hit && cacheItem) {
        cacheItem.style.display = 'block';
    }
    
    // Decision details
    const detailsContainer = card.querySelector('#value-decision-details');
    if (detailsContainer && tokenomics.decisions) {
        detailsContainer.innerHTML = renderDecisionChain(tokenomics.decisions);
    }
    
    return card;
}

function renderDecisionChain(decisions) {
    if (!decisions) return '';
    
    let html = '<div class="decision-chain">';
    
    // Complexity
    if (decisions.complexity_analysis) {
        html += `
            <div class="decision-step">
                <div class="step-label">Complexity Analysis</div>
                <div class="step-value">${capitalize(decisions.complexity_analysis.detected)}</div>
                <div class="step-reasoning">${decisions.complexity_analysis.reasoning}</div>
            </div>
        `;
    }
    
    // Routing
    if (decisions.routing_decision) {
        html += `
            <div class="decision-step">
                <div class="step-label">Routing Decision</div>
                <div class="step-value">${capitalize(decisions.routing_decision.strategy)} → ${decisions.routing_decision.model}</div>
                <div class="step-reasoning">${decisions.routing_decision.reasoning}</div>
            </div>
        `;
    }
    
    // Cache
    if (decisions.cache_decision) {
        html += `
            <div class="decision-step">
                <div class="step-label">Cache Decision</div>
                <div class="step-value">${decisions.cache_decision.hit ? 'Hit' : 'Miss'}</div>
                <div class="step-reasoning">${decisions.cache_decision.reasoning}</div>
            </div>
        `;
    }
    
    // Compression
    if (decisions.compression_decision) {
        html += `
            <div class="decision-step">
                <div class="step-label">Compression Decision</div>
                <div class="step-value">${decisions.compression_decision.applied ? 'Applied' : 'Not Applied'}</div>
                <div class="step-reasoning">${decisions.compression_decision.reasoning}</div>
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

function updateValueDashboard(comparison, tokenomics) {
    // Update totals
    valueStats.totalCostSaved += comparison.cost_savings || 0;
    valueStats.totalTokensSaved += comparison.token_savings || 0;
    valueStats.queriesCount += 1;
    
    // Update strategy counts
    const strategy = tokenomics.strategy || 'none';
    valueStats.strategyCounts[strategy] = (valueStats.strategyCounts[strategy] || 0) + 1;
    
    // Update cache stats
    if (tokenomics.cache_hit) {
        valueStats.cacheHits += 1;
    } else {
        valueStats.cacheMisses += 1;
    }
    
    // Update UI
    const totalCostSaved = document.getElementById('total-cost-saved');
    const totalCostPercent = document.getElementById('total-cost-saved-percent');
    const totalTokensSaved = document.getElementById('total-tokens-saved');
    const queriesCount = document.getElementById('queries-count');
    
    if (totalCostSaved) {
        totalCostSaved.textContent = formatCost(Math.abs(valueStats.totalCostSaved));
    }
    
    if (totalCostPercent) {
        const avgPercent = valueStats.queriesCount > 0 
            ? (valueStats.totalCostSaved / (valueStats.totalCostSaved + (valueStats.totalCostSaved / (comparison.cost_savings_percent / 100))) * 100)
            : 0;
        totalCostPercent.textContent = `~${Math.round(avgPercent)}% avg`;
    }
    
    if (totalTokensSaved) {
        totalTokensSaved.textContent = valueStats.totalTokensSaved.toLocaleString();
    }
    
    if (queriesCount) {
        queriesCount.textContent = valueStats.queriesCount;
    }
    
    // Update charts
    updateValueCharts();
}

function updateValueCharts() {
    // Strategy chart
    const strategyCtx = document.getElementById('strategy-chart');
    if (strategyCtx && Object.keys(valueStats.strategyCounts).length > 0) {
        if (window.strategyChart) {
            window.strategyChart.destroy();
        }
        window.strategyChart = new Chart(strategyCtx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(valueStats.strategyCounts),
                datasets: [{
                    data: Object.values(valueStats.strategyCounts),
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.5)',
                        'rgba(245, 158, 11, 0.5)',
                        'rgba(239, 68, 68, 0.5)'
                    ]
                }]
            }
        });
    }
    
    // Cache chart
    const cacheCtx = document.getElementById('cache-chart');
    if (cacheCtx && (valueStats.cacheHits > 0 || valueStats.cacheMisses > 0)) {
        if (window.cacheChart) {
            window.cacheChart.destroy();
        }
        window.cacheChart = new Chart(cacheCtx, {
            type: 'doughnut',
            data: {
                labels: ['Cache Hits', 'Cache Misses'],
                datasets: [{
                    data: [valueStats.cacheHits, valueStats.cacheMisses],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.5)',
                        'rgba(239, 68, 68, 0.5)'
                    ]
                }]
            }
        });
    }
}

async function handleTestRun() {
    const queryInput = document.getElementById('test-query-input');
    const batchMode = document.getElementById('batch-mode')?.checked;
    const queryText = queryInput?.value.trim();
    
    if (!queryText) {
        alert('Please enter a query');
        return;
    }
    
    const queries = batchMode 
        ? queryText.split('\n').map(q => q.trim()).filter(q => q)
        : [queryText];
    
    if (queries.length === 0) {
        alert('Please enter at least one query');
        return;
    }
    
    const resultsContainer = document.getElementById('test-results');
    const placeholder = resultsContainer.querySelector('.results-placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    // Show loading
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-indicator';
    loadingDiv.textContent = `Running ${queries.length} comparison${queries.length > 1 ? 's' : ''}...`;
    resultsContainer.appendChild(loadingDiv);
    
    // Disable button
    const runBtn = document.getElementById('test-run-btn');
    runBtn.disabled = true;
    runBtn.textContent = 'Running...';
    
    try {
        const results = [];
        for (const query of queries) {
            const response = await api.compare(query);
            results.push(response);
            
            // Render A/B card
            const card = renderABComparisonCard(response);
            resultsContainer.appendChild(card);
        }
        
        testResults = results;
        saveState();
        
    } catch (error) {
        console.error('Test run error:', error);
        alert(`Error: ${error.message || 'Failed to run comparison'}`);
    } finally {
        loadingDiv.remove();
        runBtn.disabled = false;
        runBtn.textContent = 'Run Comparison';
    }
}

function renderABComparisonCard(comparisonData) {
    const template = document.getElementById('ab-card-template');
    if (!template) return document.createElement('div');
    
    const card = template.content.cloneNode(true).querySelector('.ab-comparison-card');
    
    const { baseline, tokenomics, comparison, query } = comparisonData;
    
    // Query title
    const queryTitle = card.querySelector('#ab-query');
    if (queryTitle) {
        queryTitle.textContent = query.length > 60 ? query.substring(0, 60) + '...' : query;
    }
    
    // Baseline card
    const baselineResponse = card.querySelector('#ab-baseline-response');
    const baselineTokens = card.querySelector('#ab-baseline-tokens');
    const baselineCost = card.querySelector('#ab-baseline-cost');
    const baselineLatency = card.querySelector('#ab-baseline-latency');
    const baselineModel = card.querySelector('#ab-baseline-model');
    
    if (baselineResponse) baselineResponse.textContent = baseline.response.substring(0, 200) + (baseline.response.length > 200 ? '...' : '');
    if (baselineTokens) baselineTokens.textContent = baseline.tokens_used.toLocaleString();
    if (baselineCost) baselineCost.textContent = formatCost(baseline.cost);
    if (baselineLatency) baselineLatency.textContent = `${baseline.latency_ms.toFixed(0)}ms`;
    if (baselineModel) baselineModel.textContent = baseline.model;
    
    // Tokenomics card
    const tokenomicsResponse = card.querySelector('#ab-tokenomics-response');
    const tokenomicsTokens = card.querySelector('#ab-tokenomics-tokens');
    const tokenomicsCost = card.querySelector('#ab-tokenomics-cost');
    const tokenomicsLatency = card.querySelector('#ab-tokenomics-latency');
    const tokenomicsModel = card.querySelector('#ab-tokenomics-model');
    const tokenomicsStrategy = card.querySelector('#ab-tokenomics-strategy');
    
    if (tokenomicsResponse) tokenomicsResponse.textContent = tokenomics.response.substring(0, 200) + (tokenomics.response.length > 200 ? '...' : '');
    if (tokenomicsTokens) tokenomicsTokens.textContent = tokenomics.tokens_used.toLocaleString();
    if (tokenomicsCost) tokenomicsCost.textContent = formatCost(tokenomics.cost);
    if (tokenomicsLatency) tokenomicsLatency.textContent = `${tokenomics.latency_ms.toFixed(0)}ms`;
    if (tokenomicsModel) tokenomicsModel.textContent = tokenomics.model;
    if (tokenomicsStrategy) {
        tokenomicsStrategy.textContent = capitalize(tokenomics.strategy || 'none');
        tokenomicsStrategy.className = `badge badge-${tokenomics.strategy || 'none'}`;
    }
    
    // Comparison bar
    const tokenSavings = card.querySelector('#ab-token-savings');
    const tokenPercent = card.querySelector('#ab-token-percent');
    const costSavings = card.querySelector('#ab-cost-savings');
    const costPercent = card.querySelector('#ab-cost-percent');
    const latencyReduction = card.querySelector('#ab-latency-reduction');
    const latencyPercent = card.querySelector('#ab-latency-percent');
    
    if (tokenSavings) tokenSavings.textContent = `-${comparison.token_savings.toLocaleString()}`;
    if (tokenPercent) tokenPercent.textContent = `${comparison.token_savings_percent}%`;
    if (costSavings) costSavings.textContent = `-${formatCost(Math.abs(comparison.cost_savings))}`;
    if (costPercent) costPercent.textContent = `${comparison.cost_savings_percent}%`;
    if (latencyReduction) latencyReduction.textContent = `-${comparison.latency_reduction.toFixed(0)}ms`;
    if (latencyPercent) latencyPercent.textContent = `${comparison.latency_reduction_percent}%`;
    
    // Decision details
    const tokenomicsDetails = card.querySelector('#ab-tokenomics-details');
    if (tokenomicsDetails && tokenomics.decisions) {
        tokenomicsDetails.innerHTML = renderDecisionChain(tokenomics.decisions);
    }
    
    const baselineDetails = card.querySelector('#ab-baseline-details');
    if (baselineDetails && baseline.decisions) {
        baselineDetails.innerHTML = `
            <div class="decision-chain">
                <div class="decision-step">
                    <div class="step-label">Baseline Mode</div>
                    <div class="step-value">No Optimization</div>
                    <div class="step-reasoning">${baseline.decisions.reasoning || 'Direct LLM call with no optimizations'}</div>
                </div>
            </div>
        `;
    }
    
    return card;
}

function toggleDecisionDetails(btn) {
    const card = btn.closest('.value-card');
    const details = card.querySelector('.decision-details');
    if (details) {
        details.classList.toggle('hidden');
        btn.textContent = details.classList.contains('hidden') 
            ? 'Show Decision Details' 
            : 'Hide Decision Details';
    }
}

function toggleCardDetails(btn, type) {
    const card = btn.closest('.ab-card');
    const details = card.querySelector('.card-details');
    if (details) {
        details.classList.toggle('hidden');
        btn.textContent = details.classList.contains('hidden')
            ? (type === 'baseline' ? 'Show Details' : 'Show Decision Chain')
            : 'Hide Details';
    }
}

function handleTestClear() {
    const resultsContainer = document.getElementById('test-results');
    resultsContainer.innerHTML = `
        <div class="results-placeholder">
            <p>Enter a query and click "Run Comparison" to see A/B results</p>
        </div>
    `;
    testResults = [];
    saveState();
}

function setExploreInputEnabled(enabled) {
    const chatInput = document.getElementById('explore-chat-input');
    const sendBtn = document.getElementById('explore-send-btn');
    if (chatInput) chatInput.disabled = !enabled;
    if (sendBtn) sendBtn.disabled = !enabled;
}

function scrollChatToBottom() {
    const messagesContainer = document.getElementById('explore-chat-messages');
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// Utility functions
function formatCost(cost) {
    if (cost < 0.001) {
        return `$${(cost * 1000).toFixed(3)}m`;
    }
    return `$${cost.toFixed(4)}`;
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// State management
function saveState() {
    try {
        localStorage.setItem('tokenomics_playground_state', JSON.stringify({
            currentMode,
            exploreHistory: exploreHistory.slice(-10), // Keep last 10
            valueStats,
        }));
    } catch (e) {
        console.warn('Failed to save state:', e);
    }
}

function loadState() {
    try {
        const saved = localStorage.getItem('tokenomics_playground_state');
        if (saved) {
            const state = JSON.parse(saved);
            currentMode = state.currentMode || 'explore';
            exploreHistory = state.exploreHistory || [];
            valueStats = { ...valueStats, ...state.valueStats };
            
            // Restore mode
            switchMode(currentMode);
        }
    } catch (e) {
        console.warn('Failed to load state:', e);
    }
}

// API extension
if (typeof api !== 'undefined') {
    api.compare = async function(query) {
        return this.request('/api/compare', {
            method: 'POST',
            body: { query },
        });
    };
} else {
    // Fallback API
    const api = {
        async request(endpoint, options = {}) {
            const config = {
                method: options.method || 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers,
                },
            };
            
            if (options.body) {
                config.body = typeof options.body === 'string' 
                    ? options.body 
                    : JSON.stringify(options.body);
            }
            
            const response = await fetch(endpoint, config);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Request failed');
            }
            return response.json();
        },
        
        async compare(query) {
            return this.request('/api/compare', {
                method: 'POST',
                body: { query },
            });
        },
    };
    
    window.api = api;
}







