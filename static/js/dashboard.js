// Dashboard functionality

let charts = {};
let currentRun = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
});

function initializeDashboard() {
    const runBtn = document.getElementById('run-btn');
    const clearBtn = document.getElementById('clear-btn');
    
    runBtn.addEventListener('click', handleRunExperiment);
    clearBtn.addEventListener('click', handleClear);
    
    // Setup query mode handlers
    setupQueryModeHandlers();
    
    // Load platform status
    loadStatus();
}

function setupQueryModeHandlers() {
    const queryModeRadios = document.querySelectorAll('input[name="query-mode"]');
    const manualQueriesGroup = document.getElementById('manual-queries-group');
    const queryPreviewGroup = document.getElementById('query-preview-group');
    const numQueriesInput = document.getElementById('num-queries');
    const queryInput = document.getElementById('query-input');
    
    // Safety check: ensure elements exist
    if (!queryModeRadios.length || !manualQueriesGroup || !queryPreviewGroup || !numQueriesInput || !queryInput) {
        console.warn('Some dashboard elements not found, query mode handlers not initialized');
        return;
    }
    
    queryModeRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            const mode = radio.value;
            
            if (mode === 'manual') {
                manualQueriesGroup.classList.remove('hidden');
                queryPreviewGroup.classList.add('hidden');
                numQueriesInput.disabled = true;
            } else {
                manualQueriesGroup.classList.add('hidden');
                numQueriesInput.disabled = false;
                updateQueryPreview();
            }
        });
    });
    
    // Update preview when query or num queries changes
    queryInput.addEventListener('input', updateQueryPreview);
    numQueriesInput.addEventListener('input', updateQueryPreview);
}

function updateQueryPreview() {
    const queryMode = document.querySelector('input[name="query-mode"]:checked')?.value || 'same';
    const queryInput = document.getElementById('query-input');
    const numQueriesInput = document.getElementById('num-queries');
    const queryPreviewGroup = document.getElementById('query-preview-group');
    const queryPreview = document.getElementById('query-preview');
    
    if (!queryInput || !numQueriesInput || !queryPreviewGroup || !queryPreview) {
        return;
    }
    
    if (queryMode === 'manual') {
        queryPreviewGroup.classList.add('hidden');
        return;
    }
    
    const baseQuery = queryInput.value.trim();
    const numQueries = parseInt(numQueriesInput.value) || 1;
    
    if (!baseQuery || numQueries <= 1) {
        queryPreviewGroup.classList.add('hidden');
        return;
    }
    
    let queries = [];
    if (queryMode === 'same') {
        queries = Array(numQueries).fill(baseQuery);
    } else if (queryMode === 'similar') {
        queries = generateSimilarQueries(baseQuery, numQueries);
    }
    
    if (queries.length > 0) {
        queryPreviewGroup.classList.remove('hidden');
        queryPreview.innerHTML = queries.map((q, i) => 
            `<div class="preview-query">${i + 1}. ${q}</div>`
        ).join('');
    } else {
        queryPreviewGroup.classList.add('hidden');
    }
}

function generateSimilarQueries(baseQuery, count) {
    // Simple similar query generation
    const variations = [
        baseQuery,
        `Tell me about ${baseQuery}`,
        `Explain ${baseQuery}`,
        `What is ${baseQuery}?`,
        `Can you explain ${baseQuery}?`,
    ];
    
    return Array(count).fill(null).map((_, i) => {
        return variations[i % variations.length];
    });
}

async function handleRunExperiment() {
    const queryInput = document.getElementById('query-input');
    const modeSelect = document.getElementById('mode-select');
    const numQueriesInput = document.getElementById('num-queries');
    const manualQueriesInput = document.getElementById('manual-queries-input');
    const queryMode = document.querySelector('input[name="query-mode"]:checked')?.value || 'same';
    
    let queries = [];
    
    // Get queries based on mode
    if (queryMode === 'manual') {
        const manualQueries = manualQueriesInput.value.trim();
        if (!manualQueries) {
            alert('Please enter queries in the manual input field');
            return;
        }
        queries = manualQueries.split('\n')
            .map(q => q.trim())
            .filter(q => q.length > 0);
        
        if (queries.length === 0) {
            alert('Please enter at least one query');
            return;
        }
    } else {
        const query = queryInput.value.trim();
        if (!query) {
            alert('Please enter a query');
            return;
        }
        
        const numQueries = parseInt(numQueriesInput.value) || 1;
        
        if (queryMode === 'same') {
            queries = numQueries > 1 ? Array(numQueries).fill(query) : [query];
        } else if (queryMode === 'similar') {
            queries = generateSimilarQueries(query, numQueries);
        }
    }
    
    const mode = modeSelect.value;
    const numQueries = queries.length;
    
    // Show status indicator
    const statusIndicator = document.getElementById('status-indicator');
    statusIndicator.classList.remove('hidden');
    
    // Disable run button
    const runBtn = document.getElementById('run-btn');
    runBtn.disabled = true;
    runBtn.textContent = 'Running...';
    
    try {
        // Run experiment
        console.log('Running experiment with:', { queries, mode, numQueries });
        currentRun = await api.runExperiment(queries, mode, numQueries);
        console.log('Experiment completed. Run data:', currentRun);
        
        // Verify run structure
        if (!currentRun) {
            throw new Error('No data returned from API');
        }
        
        if (!currentRun.results || !Array.isArray(currentRun.results)) {
            console.error('Invalid results structure:', currentRun);
            throw new Error('Invalid results structure returned from API');
        }
        
        if (currentRun.results.length === 0) {
            console.warn('No results in response');
            alert('Experiment completed but no results were returned. Check console for details.');
            return;
        }
        
        // Display results
        displayResults(currentRun);
        
    } catch (error) {
        console.error('Experiment failed:', error);
        console.error('Error details:', {
            message: error.message,
            stack: error.stack,
            data: error.data
        });
        
        // Try to parse error response
        let errorMessage = 'Failed to run experiment';
        if (error.message) {
            errorMessage = error.message;
        } else if (error.data && error.data.error) {
            errorMessage = error.data.error;
        }
        
        alert(`Error: ${errorMessage}\n\nCheck browser console (F12) for more details.`);
    } finally {
        // Hide status indicator
        statusIndicator.classList.add('hidden');
        
        // Re-enable run button
        runBtn.disabled = false;
        runBtn.textContent = 'Run Experiment';
    }
}

function displayResults(run) {
    console.log('Displaying results for run:', run);
    
    if (!run) {
        console.error('No run data provided');
        alert('No results to display. Please check the console for errors.');
        return;
    }
    
    if (!run.results || !Array.isArray(run.results) || run.results.length === 0) {
        console.error('Invalid or empty results array:', run.results);
        alert('No results found in the response. The API may have returned an error.');
        return;
    }
    
    // Get results first
    const results = run.results;
    
    // Show summary section
    const summarySection = document.getElementById('summary-section');
    if (!summarySection) {
        console.error('Summary section element not found');
        alert('Dashboard error: Summary section not found. Please refresh the page.');
        return;
    }
    
    // Force remove hidden class (in case of CSS specificity issues)
    summarySection.classList.remove('hidden');
    summarySection.style.display = 'block'; // Explicitly set display
    summarySection.style.visibility = 'visible'; // Force visibility
    summarySection.style.opacity = '1'; // Force opacity
    
    // Also ensure the metrics grid is visible
    const metricsGrid = document.getElementById('metrics-grid');
    if (metricsGrid) {
        metricsGrid.style.display = 'grid';
        metricsGrid.style.visibility = 'visible';
    }
    
    // Verify the section is visible
    const computedStyle = window.getComputedStyle(summarySection);
    console.log('Summary section display:', computedStyle.display);
    console.log('Summary section visibility:', computedStyle.visibility);
    console.log('Summary section has hidden class:', summarySection.classList.contains('hidden'));
    
    console.log('Summary section shown, results count:', results.length);
    console.log('First result sample:', results[0] ? {
        tokens_used: results[0].tokens_used,
        latency_ms: results[0].latency_ms,
        cache_hit: results[0].cache_hit
    } : 'No results');
    
    // Calculate summary metrics
    const totalTokens = results.reduce((sum, r) => sum + (r.tokens_used || 0), 0);
    const avgTokens = results.length > 0 ? Math.round(totalTokens / results.length) : 0;
    const totalLatency = results.reduce((sum, r) => sum + (r.latency_ms || 0), 0);
    const avgLatency = results.length > 0 ? Math.round(totalLatency / results.length) : 0;
    const cacheHits = results.filter(r => r.cache_hit).length;
    const cacheHitRate = results.length > 0 ? Math.round((cacheHits / results.length) * 100) : 0;
    
    // Update summary metrics (with null checks)
    const totalTokensEl = document.getElementById('total-tokens');
    const avgTokensEl = document.getElementById('avg-tokens');
    const totalLatencyEl = document.getElementById('total-latency');
    const avgLatencyEl = document.getElementById('avg-latency');
    const cacheHitRateEl = document.getElementById('cache-hit-rate');
    const cacheHitsEl = document.getElementById('cache-hits');
    
    // Update metrics with validation
    console.log('About to update metrics. Elements found:', {
        totalTokensEl: !!totalTokensEl,
        avgTokensEl: !!avgTokensEl,
        totalLatencyEl: !!totalLatencyEl,
        avgLatencyEl: !!avgLatencyEl,
        cacheHitRateEl: !!cacheHitRateEl,
        cacheHitsEl: !!cacheHitsEl
    });
    
    if (totalTokensEl) {
        totalTokensEl.textContent = totalTokens.toLocaleString();
        totalTokensEl.style.display = 'block';
        totalTokensEl.style.visibility = 'visible';
        console.log('✅ Updated total-tokens:', totalTokens, '→', totalTokensEl.textContent);
        // Force a reflow to ensure visibility
        totalTokensEl.offsetHeight;
    } else {
        console.error('❌ total-tokens element not found');
    }
    
    if (avgTokensEl) {
        avgTokensEl.textContent = avgTokens.toLocaleString();
        avgTokensEl.style.display = 'block';
        avgTokensEl.style.visibility = 'visible';
        console.log('✅ Updated avg-tokens:', avgTokens);
    } else {
        console.error('❌ avg-tokens element not found');
    }
    
    if (totalLatencyEl) {
        totalLatencyEl.textContent = `${totalLatency.toLocaleString()} ms`;
        totalLatencyEl.style.display = 'block';
        totalLatencyEl.style.visibility = 'visible';
        console.log('✅ Updated total-latency:', totalLatency);
    } else {
        console.error('❌ total-latency element not found');
    }
    
    if (avgLatencyEl) {
        avgLatencyEl.textContent = `${avgLatency.toLocaleString()} ms`;
        avgLatencyEl.style.display = 'block';
        avgLatencyEl.style.visibility = 'visible';
        console.log('✅ Updated avg-latency:', avgLatency);
    } else {
        console.error('❌ avg-latency element not found');
    }
    
    if (cacheHitRateEl) {
        cacheHitRateEl.textContent = `${cacheHitRate}%`;
        cacheHitRateEl.style.display = 'block';
        cacheHitRateEl.style.visibility = 'visible';
        console.log('✅ Updated cache-hit-rate:', cacheHitRate);
    } else {
        console.error('❌ cache-hit-rate element not found');
    }
    
    if (cacheHitsEl) {
        cacheHitsEl.textContent = cacheHits;
        cacheHitsEl.style.display = 'block';
        cacheHitsEl.style.visibility = 'visible';
        console.log('✅ Updated cache-hits:', cacheHits);
    } else {
        console.error('❌ cache-hits element not found');
    }
    
    console.log('✅ Metrics updated:', {
        totalTokens,
        avgTokens,
        totalLatency,
        avgLatency,
        cacheHits,
        cacheHitRate
    });
    
    // Verify the values are actually in the DOM
    setTimeout(() => {
        const verifyTotalTokens = document.getElementById('total-tokens');
        console.log('🔍 Verification - total-tokens in DOM:', verifyTotalTokens?.textContent);
        console.log('🔍 Verification - summary section visible:', 
            window.getComputedStyle(summarySection).display !== 'none');
    }, 100);
    
    // Show component breakdown for A/B mode
    if (run.mode === 'ab' && run.summary && run.summary.component_breakdown) {
        const componentSection = document.getElementById('component-breakdown-section');
        componentSection.classList.remove('hidden');
        
        const breakdown = run.summary.component_breakdown;
        const memorySavings = breakdown.memory_layer?.tokens_saved || 0;
        const orchestratorSavings = breakdown.orchestrator?.tokens_saved || 0;
        const banditSavings = breakdown.bandit?.tokens_saved || 0;
        
        document.getElementById('memory-savings').textContent = memorySavings.toLocaleString();
        document.getElementById('orchestrator-savings').textContent = orchestratorSavings.toLocaleString();
        document.getElementById('bandit-savings').textContent = banditSavings.toLocaleString();
        
        const totalSavings = memorySavings + orchestratorSavings + banditSavings;
        if (totalSavings > 0) {
            document.getElementById('memory-percentage').textContent = `${Math.round((memorySavings / totalSavings) * 100)}%`;
            document.getElementById('orchestrator-percentage').textContent = `${Math.round((orchestratorSavings / totalSavings) * 100)}%`;
            document.getElementById('bandit-percentage').textContent = `${Math.round((banditSavings / totalSavings) * 100)}%`;
        } else {
            document.getElementById('memory-percentage').textContent = '0%';
            document.getElementById('orchestrator-percentage').textContent = '0%';
            document.getElementById('bandit-percentage').textContent = '0%';
        }
    } else {
        document.getElementById('component-breakdown-section').classList.add('hidden');
    }
    
    // Update semantic cache metrics
    const exactHits = results.filter(r => r.cache_type === 'exact' || r.cache_type === 'hit').length;
    const semanticHits = results.filter(r => r.cache_type === 'semantic_direct' || r.cache_type === 'semantic').length;
    const contextHits = results.filter(r => r.cache_type === 'context').length;
    const similarities = results.filter(r => r.similarity !== null && r.similarity !== undefined).map(r => r.similarity);
    const avgSimilarity = similarities.length > 0 
        ? (similarities.reduce((sum, s) => sum + s, 0) / similarities.length).toFixed(3)
        : '-';
    
    const exactHitsEl = document.getElementById('exact-hits');
    const semanticHitsEl = document.getElementById('semantic-hits');
    const contextHitsEl = document.getElementById('context-hits');
    const avgSimilarityEl = document.getElementById('avg-similarity');
    
    if (exactHitsEl) exactHitsEl.textContent = exactHits;
    if (semanticHitsEl) semanticHitsEl.textContent = semanticHits;
    if (contextHitsEl) contextHitsEl.textContent = contextHits;
    if (avgSimilarityEl) avgSimilarityEl.textContent = avgSimilarity;
    
    // Show charts section
    const chartsSection = document.getElementById('charts-section');
    if (chartsSection) {
        chartsSection.classList.remove('hidden');
        chartsSection.style.display = '';
        
        // Render charts
        console.log('Rendering charts for', results.length, 'results');
        renderCharts(results);
    } else {
        console.warn('Charts section element not found');
    }
    
    // Show results table
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.classList.remove('hidden');
        resultsSection.style.display = '';
        
        // Render results table
        console.log('Rendering results table for', results.length, 'results');
        renderResultsTable(results);
    } else {
        console.warn('Results section element not found');
    }
    
    console.log('Results displayed successfully. Summary visible:', !summarySection.classList.contains('hidden'));
    
    // Scroll to summary section to ensure it's visible
    setTimeout(() => {
        summarySection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
        // Add a temporary highlight to show the section
        summarySection.style.border = '3px solid #00d4ff';
        summarySection.style.boxShadow = '0 0 30px rgba(0, 212, 255, 0.5)';
        setTimeout(() => {
            summarySection.style.border = '';
            summarySection.style.boxShadow = '';
        }, 2000);
        
        // Final verification
        const finalCheck = {
            sectionVisible: window.getComputedStyle(summarySection).display !== 'none',
            totalTokensValue: document.getElementById('total-tokens')?.textContent,
            sectionInView: summarySection.getBoundingClientRect().top < window.innerHeight
        };
        console.log('🔍 Final visibility check:', finalCheck);
        
        if (!finalCheck.sectionVisible) {
            console.error('⚠️ Section is still hidden! Forcing display...');
            summarySection.style.display = 'block';
            summarySection.style.visibility = 'visible';
        }
        
        if (finalCheck.totalTokensValue === '0' || !finalCheck.totalTokensValue) {
            console.warn('⚠️ Total tokens value is 0 or empty. This might indicate no data was processed.');
        }
    }, 100);
}

function renderCharts(results) {
    // Token Usage Chart
    const tokensCtx = document.getElementById('tokens-chart');
    if (tokensCtx && results.length > 0) {
        if (charts.tokens) charts.tokens.destroy();
        charts.tokens = new Chart(tokensCtx, {
            type: 'bar',
            data: {
                labels: results.map((_, i) => `Query ${i + 1}`),
                datasets: [{
                    label: 'Tokens Used',
                    data: results.map(r => r.tokens_used || 0),
                    backgroundColor: 'rgba(0, 212, 255, 0.5)',
                    borderColor: 'rgba(0, 212, 255, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    // Latency Chart
    const latencyCtx = document.getElementById('latency-chart');
    if (latencyCtx && results.length > 0) {
        if (charts.latency) charts.latency.destroy();
        charts.latency = new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: results.map((_, i) => `Query ${i + 1}`),
                datasets: [{
                    label: 'Latency (ms)',
                    data: results.map(r => r.latency_ms || 0),
                    borderColor: 'rgba(124, 58, 237, 1)',
                    backgroundColor: 'rgba(124, 58, 237, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    // Cache Chart
    const cacheCtx = document.getElementById('cache-chart');
    if (cacheCtx && results.length > 0) {
        if (charts.cache) charts.cache.destroy();
        const hits = results.filter(r => r.cache_hit).length;
        const misses = results.length - hits;
        charts.cache = new Chart(cacheCtx, {
            type: 'doughnut',
            data: {
                labels: ['Cache Hits', 'Cache Misses'],
                datasets: [{
                    data: [hits, misses],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.5)',
                        'rgba(239, 68, 68, 0.5)'
                    ],
                    borderColor: [
                        'rgba(16, 185, 129, 1)',
                        'rgba(239, 68, 68, 1)'
                    ]
                }]
            }
        });
    }
    
    // Strategy Chart
    const strategyCtx = document.getElementById('strategy-chart');
    if (strategyCtx && results.length > 0) {
        if (charts.strategy) charts.strategy.destroy();
        const strategyCounts = {};
        results.forEach(r => {
            const strategy = r.strategy || 'none';
            strategyCounts[strategy] = (strategyCounts[strategy] || 0) + 1;
        });
        charts.strategy = new Chart(strategyCtx, {
            type: 'pie',
            data: {
                labels: Object.keys(strategyCounts),
                datasets: [{
                    data: Object.values(strategyCounts),
                    backgroundColor: [
                        'rgba(0, 212, 255, 0.5)',
                        'rgba(124, 58, 237, 0.5)',
                        'rgba(244, 114, 182, 0.5)'
                    ]
                }]
            }
        });
    }
}

function renderResultsTable(results) {
    const tbody = document.getElementById('results-tbody');
    if (!tbody) return;
    
    if (!results || results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-secondary);">No results to display</td></tr>';
        return;
    }
    
    tbody.innerHTML = results.map((result, index) => {
        const cacheType = result.cache_type || 'none';
        const similarity = result.similarity !== null && result.similarity !== undefined 
            ? result.similarity.toFixed(3) 
            : '-';
        const strategy = result.strategy || 'none';
        const model = result.model || 'unknown';
        const queryText = result.query || `Query ${index + 1}`;
        const responsePreview = (result.response || '').substring(0, 100) + (result.response && result.response.length > 100 ? '...' : '');
        
        return `
            <tr>
                <td>${index + 1}</td>
                <td>${escapeHtml(queryText.substring(0, 50))}${(queryText.length > 50 ? '...' : '')}</td>
                <td>${escapeHtml(responsePreview)}</td>
                <td>${(result.tokens_used || 0).toLocaleString()}</td>
                <td>${(result.latency_ms || 0).toLocaleString()} ms</td>
                <td>${cacheType}</td>
                <td>${similarity}</td>
                <td>${strategy}</td>
                <td>${model}</td>
                <td>
                    <button class="btn btn-sm" onclick="showDetails(${index})">Details</button>
                </td>
            </tr>
        `;
    }).join('');
}

function showDetails(index) {
    if (!currentRun || !currentRun.results || !currentRun.results[index]) {
        return;
    }
    
    const result = currentRun.results[index];
    const details = JSON.stringify(result, null, 2);
    alert(`Query ${index + 1} Details:\n\n${details}`);
}

function handleClear() {
    // Clear input fields
    document.getElementById('query-input').value = '';
    document.getElementById('manual-queries-input').value = '';
    document.getElementById('num-queries').value = '1';
    
    // Hide sections
    document.getElementById('summary-section').classList.add('hidden');
    document.getElementById('charts-section').classList.add('hidden');
    document.getElementById('results-section').classList.add('hidden');
    
    // Destroy charts
    Object.values(charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    charts = {};
    
    currentRun = null;
}

async function loadStatus() {
    try {
        const status = await api.getStatus();
        console.log('Platform status:', status);
        
        // Display API status
        const apiStatusIndicator = document.getElementById('api-status-indicator');
        if (!apiStatusIndicator) {
            console.warn('API status indicator element not found');
            return;
        }
        
        if (status.status === 'error') {
            apiStatusIndicator.classList.remove('hidden');
            apiStatusIndicator.className = 'api-status error';
            apiStatusIndicator.innerHTML = `
                <div class="api-status-content">
                    <strong>[WARNING] Configuration Issue:</strong>
                    <p>${status.message || 'API key not configured'}</p>
                    <p style="font-size: 0.9em; margin-top: 0.5em;">
                        Please set OPENAI_API_KEY in your .env file and restart the server.
                    </p>
                </div>
            `;
        } else if (status.status === 'ok') {
            apiStatusIndicator.classList.remove('hidden');
            apiStatusIndicator.className = 'api-status success';
            apiStatusIndicator.innerHTML = `
                <div class="api-status-content">
                    <strong>[OK] Platform Ready</strong>
                    <span style="font-size: 0.9em;">Cache: ${status.platform_stats?.cache_size || 0} entries</span>
                </div>
            `;
        }
    } catch (error) {
        console.error('Failed to load status:', error);
        const apiStatusIndicator = document.getElementById('api-status-indicator');
        if (apiStatusIndicator) {
            apiStatusIndicator.classList.remove('hidden');
            apiStatusIndicator.className = 'api-status error';
            apiStatusIndicator.innerHTML = `
                <div class="api-status-content">
                    <strong>[ERROR] Cannot connect to server</strong>
                    <p>${error.message || 'Unknown error'}</p>
                </div>
            `;
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
