// Results page functionality

let allRuns = [];
let filteredRuns = [];

document.addEventListener('DOMContentLoaded', () => {
    initializeResults();
});

async function initializeResults() {
    // Load aggregated stats
    await loadAggregatedStats();
    
    // Load runs
    await loadRuns();
    
    // Setup filters
    setupFilters();
    
    // Setup modal
    setupModal();
}

async function loadAggregatedStats() {
    try {
        const stats = await api.getStats();
        
        document.getElementById('total-runs').textContent = stats.total_runs;
        document.getElementById('avg-token-savings').textContent = 
            `${stats.average_token_savings.toFixed(1)}%`;
        const latencyReduction = stats.average_latency_reduction != null ? stats.average_latency_reduction.toFixed(1) : '0.0';
        document.getElementById('avg-latency-reduction').textContent = 
            `${latencyReduction}%`;
        document.getElementById('avg-cache-hit-rate').textContent = 
            `${stats.average_cache_hit_rate.toFixed(1)}%`;
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

async function loadRuns() {
    try {
        const data = await api.getRuns();
        allRuns = data.runs || [];
        filteredRuns = [...allRuns];
        
        displayRuns(filteredRuns);
    } catch (error) {
        console.error('Failed to load runs:', error);
        document.getElementById('runs-container').innerHTML = 
            '<div class="error">Failed to load runs</div>';
    }
}

function setupFilters() {
    const applyBtn = document.getElementById('apply-filters');
    applyBtn.addEventListener('click', applyFilters);
}

function applyFilters() {
    const modeFilter = document.getElementById('filter-mode').value;
    const sortBy = document.getElementById('sort-by').value;
    
    // Filter by mode
    filteredRuns = allRuns.filter(run => {
        if (modeFilter === 'all') return true;
        return run.mode === modeFilter;
    });
    
    // Sort
    filteredRuns.sort((a, b) => {
        switch (sortBy) {
            case 'date':
                return new Date(b.start_time) - new Date(a.start_time);
            case 'tokens':
                return (b.summary?.tokens_saved || 0) - (a.summary?.tokens_saved || 0);
            case 'latency':
                return (b.summary?.latency_reduction || 0) - (a.summary?.latency_reduction || 0);
            case 'cache':
                return (b.summary?.cache_hit_rate || 0) - (a.summary?.cache_hit_rate || 0);
            default:
                return 0;
        }
    });
    
    displayRuns(filteredRuns);
}

function displayRuns(runs) {
    const container = document.getElementById('runs-container');
    
    if (runs.length === 0) {
        container.innerHTML = '<div class="no-runs">No runs found</div>';
        return;
    }
    
    container.innerHTML = runs.map(run => createRunCard(run)).join('');
    
    // Add click listeners
    runs.forEach(run => {
        const card = document.querySelector(`[data-run-id="${run.id}"]`);
        if (card) {
            card.addEventListener('click', () => showRunDetails(run.id));
        }
    });
}

function createRunCard(run) {
    const summary = run.summary || {};
    const date = new Date(run.start_time).toLocaleString();
    
    return `
        <div class="run-card" data-run-id="${run.id}">
            <div class="run-header">
                <span class="run-id">${run.id}</span>
                <span class="run-date">${date}</span>
            </div>
            <div class="run-metrics">
                <div class="run-metric">
                    <div class="run-metric-value">${summary.total_queries || 0}</div>
                    <div class="run-metric-label">Queries</div>
                </div>
                <div class="run-metric">
                    <div class="run-metric-value">${summary.cache_hit_rate?.toFixed(1) || 0}%</div>
                    <div class="run-metric-label">Cache Hit Rate</div>
                </div>
                <div class="run-metric">
                    <div class="run-metric-value">${utils.formatNumber(summary.total_tokens || 0)}</div>
                    <div class="run-metric-label">Total Tokens</div>
                </div>
                <div class="run-metric">
                    <div class="run-metric-value">${utils.formatDuration(summary.average_latency_ms || 0)}</div>
                    <div class="run-metric-label">Avg Latency</div>
                </div>
            </div>
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-color);">
                <strong>Mode:</strong> ${run.mode || 'N/A'}
            </div>
        </div>
    `;
}

async function showRunDetails(runId) {
    try {
        const run = await api.getRun(runId);
        displayRunModal(run);
    } catch (error) {
        console.error('Failed to load run details:', error);
        alert('Failed to load run details');
    }
}

function displayRunModal(run) {
    const modal = document.getElementById('run-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    
    modalTitle.textContent = `Run Details: ${run.id}`;
    
    // Animate modal appearance
    modal.style.opacity = '0';
    setTimeout(() => {
        modal.style.transition = 'opacity 0.3s ease';
        modal.style.opacity = '1';
    }, 10);
    
    const summary = run.summary || {};
    const results = run.results || [];
    
    modalBody.innerHTML = `
        <div style="margin-bottom: 2rem;">
            <h3>Summary</h3>
            <div class="metrics-grid" style="margin-top: 1rem;">
                <div class="metric-card">
                    <div class="metric-label">Total Queries</div>
                    <div class="metric-value">${summary.total_queries || 0}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Tokens</div>
                    <div class="metric-value">${utils.formatNumber(summary.total_tokens || 0)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cache Hit Rate</div>
                    <div class="metric-value">${summary.cache_hit_rate?.toFixed(1) || 0}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Latency</div>
                    <div class="metric-value">${utils.formatDuration(summary.average_latency_ms || 0)}</div>
                </div>
            </div>
        </div>
        
        <div>
            <h3>Query Results</h3>
            <div class="table-container" style="margin-top: 1rem;">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Query</th>
                            <th>Tokens</th>
                            <th>Latency</th>
                            <th>Cache</th>
                            <th>Strategy</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${results.map(r => `
                            <tr>
                                <td>${r.query_num}</td>
                                <td>${r.query.substring(0, 60)}${r.query.length > 60 ? '...' : ''}</td>
                                <td>${utils.formatNumber(r.tokens_used)}</td>
                                <td>${utils.formatDuration(r.latency_ms)}</td>
                                <td>${r.cache_hit ? '✓ Hit' : '✗ Miss'}</td>
                                <td>${r.strategy || 'N/A'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    modal.classList.remove('hidden');
}

function setupModal() {
    const modal = document.getElementById('run-modal');
    const closeBtn = document.querySelector('.close-modal');
    
    closeBtn.addEventListener('click', () => {
        modal.style.opacity = '0';
        setTimeout(() => {
            modal.classList.add('hidden');
        }, 300);
    });
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.opacity = '0';
            setTimeout(() => {
                modal.classList.add('hidden');
            }, 300);
        }
    });
}

