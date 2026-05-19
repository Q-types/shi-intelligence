/**
 * SHI Dashboard - Main Application
 * Handles data fetching, chart rendering, and UI interactions
 */

const API_BASE = '/api/v1';

// State
let currentToken = null;
let analysisData = null;
let distributionChart = null;
let topHoldersChart = null;
let walletGraph = null;

// DOM Elements
const elements = {
    tokenInput: document.getElementById('tokenInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    welcomeSection: document.getElementById('welcomeSection'),
    tokenInfo: document.getElementById('tokenInfo'),
    riskSection: document.getElementById('riskSection'),
    metricsSection: document.getElementById('metricsSection'),
    chartsSection: document.getElementById('chartsSection'),
    graphSection: document.getElementById('graphSection'),
    holdersSection: document.getElementById('holdersSection'),
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    initCharts();
});

function initEventListeners() {
    elements.analyzeBtn.addEventListener('click', () => analyzeToken());
    elements.tokenInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') analyzeToken();
    });

    // Example token buttons
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            elements.tokenInput.value = btn.dataset.mint;
            analyzeToken();
        });
    });

    // Graph controls
    document.getElementById('resetGraph')?.addEventListener('click', () => {
        if (walletGraph) walletGraph.reset();
    });

    document.getElementById('showLabels')?.addEventListener('change', (e) => {
        if (walletGraph) walletGraph.toggleLabels(e.target.checked);
    });
}

function initCharts() {
    // Set Chart.js defaults
    Chart.defaults.color = '#8899a6';
    Chart.defaults.borderColor = '#38444d';
}

async function analyzeToken() {
    const mint = elements.tokenInput.value.trim();
    if (!mint) {
        alert('Please enter a token mint address');
        return;
    }

    // Validate mint format
    if (mint.length < 32 || mint.length > 44) {
        alert('Invalid mint address format (should be 32-44 characters)');
        return;
    }

    showLoading(true);
    currentToken = mint;

    try {
        // Fetch data from our analysis endpoint
        const data = await fetchAnalysis(mint);
        analysisData = data;
        console.log('Analysis data received:', Object.keys(data));

        // Update UI - show sections first
        showAnalysis(true);

        // Render each component with isolated error handling
        try { updateTokenInfo(data); } catch (e) { console.error('updateTokenInfo error:', e); }
        try { updateRiskSection(data); } catch (e) { console.error('updateRiskSection error:', e); }
        try { updateMetrics(data); } catch (e) { console.error('updateMetrics error:', e); }
        try { renderCharts(data); } catch (e) { console.error('renderCharts error:', e); }
        try { renderGraph(data); } catch (e) { console.error('renderGraph error:', e); }
        try { renderHoldersTable(data); } catch (e) { console.error('renderHoldersTable error:', e); }
        try { renderArchetypes(data); } catch (e) { console.error('renderArchetypes error:', e); }

        console.log('All render functions completed');

    } catch (error) {
        console.error('Analysis failed:', error);
        alert(`Analysis failed: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

async function fetchAnalysis(mint) {
    // Fetch from our dashboard API endpoint
    const response = await fetch(`${API_BASE}/dashboard/analyze/${mint}`);

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Analysis failed');
    }

    return response.json();
}

function showLoading(show) {
    elements.loadingOverlay.classList.toggle('hidden', !show);
}

function showAnalysis(show) {
    elements.welcomeSection.classList.toggle('hidden', show);
    elements.tokenInfo.classList.toggle('hidden', !show);
    elements.riskSection.classList.toggle('hidden', !show);
    elements.metricsSection.classList.toggle('hidden', !show);
    elements.chartsSection.classList.toggle('hidden', !show);
    elements.graphSection.classList.toggle('hidden', !show);
    elements.holdersSection.classList.toggle('hidden', !show);
    // Archetypes section is shown via renderArchetypes when data is available
    const archetypesSection = document.getElementById('archetypesSection');
    if (archetypesSection) archetypesSection.classList.toggle('hidden', !show);
}

function updateTokenInfo(data) {
    document.getElementById('tokenName').textContent = 'Token Analysis';
    document.getElementById('tokenMint').textContent = truncateAddress(data.token_mint);

    // Price info
    if (data.price) {
        document.getElementById('tokenPrice').textContent = `$${formatPrice(data.price.price_usd)}`;
        const changeEl = document.getElementById('priceChange');
        const change = data.price.price_change_24h_pct || 0;
        changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
        changeEl.className = `stat-value ${change >= 0 ? 'positive' : 'negative'}`;
    } else {
        document.getElementById('tokenPrice').textContent = '--';
        document.getElementById('priceChange').textContent = '--';
    }

    document.getElementById('holderCount').textContent = formatNumber(data.holder_count);
    document.getElementById('totalSupply').textContent = formatLargeNumber(data.total_supply);
}

function updateRiskSection(data) {
    const riskScore = data.risk_score || 0.5;
    const riskLevel = getRiskLevel(riskScore);

    document.getElementById('riskScore').textContent = riskScore.toFixed(2);

    const riskBadge = document.getElementById('riskLevel');
    riskBadge.textContent = riskLevel.label;
    riskBadge.className = `risk-badge ${riskLevel.class}`;

    document.getElementById('riskDescription').textContent = riskLevel.description;

    // Render gauge
    renderRiskGauge(riskScore);
}

function getRiskLevel(score) {
    if (score <= 0.3) {
        return {
            label: 'LOW RISK',
            class: 'low',
            description: 'Token shows healthy distribution with no major concentration concerns. Holder base appears well-distributed.'
        };
    } else if (score <= 0.6) {
        return {
            label: 'MEDIUM RISK',
            class: 'medium',
            description: 'Some concentration detected. Monitor large holders for potential coordinated activity.'
        };
    } else {
        return {
            label: 'HIGH RISK',
            class: 'high',
            description: 'High concentration risk detected. Large holders control significant supply, which may lead to price manipulation.'
        };
    }
}

function renderRiskGauge(score) {
    const svg = document.getElementById('riskGauge');
    const width = 200;
    const height = 120;
    const cx = width / 2;
    const cy = height - 10;
    const radius = 80;

    // Colors for gradient
    const colors = ['#17bf63', '#ffad1f', '#e0245e'];

    // Create arc
    const startAngle = -Math.PI;
    const endAngle = 0;
    const scoreAngle = startAngle + (score * Math.PI);

    // Background arc
    const bgArc = describeArc(cx, cy, radius, startAngle, endAngle);

    // Score arc
    const scoreArc = describeArc(cx, cy, radius, startAngle, scoreAngle);

    // Gradient
    const gradientId = 'riskGradient';

    svg.innerHTML = `
        <defs>
            <linearGradient id="${gradientId}" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:${colors[0]}" />
                <stop offset="50%" style="stop-color:${colors[1]}" />
                <stop offset="100%" style="stop-color:${colors[2]}" />
            </linearGradient>
        </defs>
        <path d="${bgArc}" fill="none" stroke="#38444d" stroke-width="12" stroke-linecap="round" />
        <path d="${scoreArc}" fill="none" stroke="url(#${gradientId})" stroke-width="12" stroke-linecap="round" />
        <circle cx="${cx + radius * Math.cos(scoreAngle)}" cy="${cy + radius * Math.sin(scoreAngle)}" r="8" fill="#fff" />
    `;
}

function describeArc(cx, cy, radius, startAngle, endAngle) {
    const start = {
        x: cx + radius * Math.cos(startAngle),
        y: cy + radius * Math.sin(startAngle)
    };
    const end = {
        x: cx + radius * Math.cos(endAngle),
        y: cy + radius * Math.sin(endAngle)
    };
    const largeArcFlag = endAngle - startAngle > Math.PI ? 1 : 0;

    return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
}

function updateMetrics(data) {
    const metrics = data.metrics || {};

    // HHI
    const hhi = metrics.hhi || 0;
    document.getElementById('hhiValue').textContent = hhi.toFixed(4);
    document.getElementById('hhiFill').style.width = `${Math.min(hhi * 400, 100)}%`;
    document.getElementById('hhiInterpretation').textContent = getHHIInterpretation(hhi);

    // Gini
    const gini = metrics.gini || 0;
    document.getElementById('giniValue').textContent = gini.toFixed(4);
    document.getElementById('giniFill').style.width = `${gini * 100}%`;
    document.getElementById('giniInterpretation').textContent = getGiniInterpretation(gini);

    // Entropy
    const entropy = metrics.entropy || 0;
    document.getElementById('entropyValue').textContent = entropy.toFixed(2);
    document.getElementById('entropyFill').style.width = `${Math.min(entropy / 10 * 100, 100)}%`;
    document.getElementById('entropyInterpretation').textContent = getEntropyInterpretation(entropy);

    // Whale Dominance
    const wdr = metrics.whale_dominance || 0;
    document.getElementById('wdrValue').textContent = `${(wdr * 100).toFixed(1)}%`;
    document.getElementById('wdrFill').style.width = `${wdr * 100}%`;
    document.getElementById('wdrInterpretation').textContent = getWDRInterpretation(wdr);
}

function getHHIInterpretation(hhi) {
    if (hhi < 0.1) return 'Low concentration - healthy distribution';
    if (hhi < 0.25) return 'Medium concentration - some whale presence';
    return 'High concentration - whale dominated';
}

function getGiniInterpretation(gini) {
    if (gini < 0.5) return 'Relatively equal distribution';
    if (gini < 0.8) return 'Unequal distribution';
    return 'Very unequal - high inequality';
}

function getEntropyInterpretation(entropy) {
    if (entropy > 4) return 'High diversity - many unique holders';
    if (entropy > 2) return 'Moderate diversity';
    return 'Low diversity - few holders dominate';
}

function getWDRInterpretation(wdr) {
    if (wdr < 0.3) return 'Well distributed - low whale control';
    if (wdr < 0.5) return 'Moderate whale presence';
    return 'Whale dominated - top 10 control majority';
}

function renderCharts(data) {
    renderDistributionChart(data);
    renderTopHoldersChart(data);
}

function renderDistributionChart(data) {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    const holders = data.holders || [];

    // Calculate distribution buckets
    const buckets = {
        'Whales (>5%)': 0,
        'Large (1-5%)': 0,
        'Medium (0.1-1%)': 0,
        'Small (<0.1%)': 0
    };

    holders.forEach(h => {
        const share = h.share * 100;
        if (share >= 5) buckets['Whales (>5%)'] += h.share;
        else if (share >= 1) buckets['Large (1-5%)'] += h.share;
        else if (share >= 0.1) buckets['Medium (0.1-1%)'] += h.share;
        else buckets['Small (<0.1%)'] += h.share;
    });

    // Destroy existing chart
    if (distributionChart) {
        distributionChart.destroy();
    }

    distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(buckets),
            datasets: [{
                data: Object.values(buckets).map(v => (v * 100).toFixed(2)),
                backgroundColor: ['#e74c3c', '#f39c12', '#3498db', '#95a5a6'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.label}: ${context.raw}%`
                    }
                }
            }
        }
    });
}

function renderTopHoldersChart(data) {
    const ctx = document.getElementById('topHoldersChart').getContext('2d');
    const holders = (data.holders || []).slice(0, 10);

    // Destroy existing chart
    if (topHoldersChart) {
        topHoldersChart.destroy();
    }

    topHoldersChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: holders.map((h, i) => `#${i + 1}`),
            datasets: [{
                label: 'Share %',
                data: holders.map(h => (h.share * 100).toFixed(2)),
                backgroundColor: holders.map(h => {
                    const share = h.share;
                    if (share >= 0.05) return '#e74c3c';
                    if (share >= 0.01) return '#f39c12';
                    if (share >= 0.001) return '#3498db';
                    return '#95a5a6';
                }),
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            const idx = items[0].dataIndex;
                            return truncateAddress(holders[idx].wallet);
                        },
                        label: (context) => `Share: ${context.raw}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: '#38444d'
                    },
                    ticks: {
                        callback: (value) => `${value}%`
                    }
                },
                y: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function renderGraph(data) {
    try {
        const holders = (data.holders || []).slice(0, 50); // Limit to top 50 for performance
        const fundingEdges = data.funding_edges || [];

        console.log('renderGraph: holders=', holders.length, 'edges=', fundingEdges.length);

        // Initialize graph if not exists
        if (!walletGraph) {
            walletGraph = new WalletGraph('walletGraphContainer');
        }

        walletGraph.init();
        walletGraph.setData(holders, fundingEdges);
    } catch (error) {
        console.error('Graph rendering error:', error);
        // Don't throw - just log and continue
    }
}

function renderHoldersTable(data) {
    const tbody = document.getElementById('holdersTableBody');
    if (!tbody) {
        console.error('holdersTableBody element not found');
        return;
    }

    const holders = (data.holders || []).slice(0, 20);
    console.log('Rendering holders table:', holders.length, 'holders');

    if (holders.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 20px;">No holder data available</td></tr>';
        return;
    }

    let cumulative = 0;
    const rows = [];
    for (let i = 0; i < holders.length; i++) {
        const h = holders[i];
        cumulative += h.share;
        const type = getHolderType(h.share);
        const archetype = h.archetype || 'unknown';
        const archetypeClass = getArchetypeClass(archetype);
        const sharedFunderBadge = h.shared_funders ? `<span class="shared-funder-badge" title="Shares funder with other wallets">🔗</span>` : '';

        rows.push(`
            <tr>
                <td>${i + 1}</td>
                <td>
                    <span class="wallet-address" title="${h.wallet}">${truncateAddress(h.wallet)}</span>
                    ${sharedFunderBadge}
                </td>
                <td>${formatLargeNumber(h.balance)}</td>
                <td>${(h.share * 100).toFixed(2)}%</td>
                <td>${(cumulative * 100).toFixed(2)}%</td>
                <td><span class="holder-type ${type}">${type.toUpperCase()}</span></td>
                <td><span class="archetype-badge ${archetypeClass}">${formatArchetypeName(archetype)}</span></td>
            </tr>
        `);
    }
    tbody.innerHTML = rows.join('');
    console.log('Table rendered with', rows.length, 'rows');
}

function renderArchetypes(data) {
    const archetypes = data.archetypes || {};
    const container = document.getElementById('archetypesContainer');
    if (!container) {
        console.log('archetypesContainer not found, skipping archetype render');
        return;
    }

    // Show the section
    const section = document.getElementById('archetypesSection');
    if (section) section.classList.remove('hidden');

    // Build archetype cards
    const archetypeOrder = [
        'sniper', 'long_term_accumulator', 'coordinated_cluster',
        'liquidity_actor', 'exchange_linked', 'dormant_whale', 'unknown'
    ];

    const archetypeInfo = {
        sniper: { icon: '🎯', description: 'Early entry, quick exits' },
        long_term_accumulator: { icon: '💎', description: 'Gradual accumulation, low churn' },
        coordinated_cluster: { icon: '🔗', description: 'Shared funders, sync behavior' },
        liquidity_actor: { icon: '💧', description: 'Frequent LP interactions' },
        exchange_linked: { icon: '🏦', description: 'CEX-linked wallets' },
        dormant_whale: { icon: '🐋', description: 'Large holder, inactive' },
        unknown: { icon: '❓', description: 'Unclassified behavior' },
    };

    let html = '';
    for (const archetype of archetypeOrder) {
        const proportion = archetypes[archetype] || 0;
        if (proportion > 0) {
            const info = archetypeInfo[archetype] || { icon: '❓', description: '' };
            const pct = (proportion * 100).toFixed(1);
            html += `
                <div class="archetype-card ${getArchetypeClass(archetype)}">
                    <div class="archetype-icon">${info.icon}</div>
                    <div class="archetype-name">${formatArchetypeName(archetype)}</div>
                    <div class="archetype-pct">${pct}%</div>
                    <div class="archetype-desc">${info.description}</div>
                </div>
            `;
        }
    }

    container.innerHTML = html || '<p>No archetype data available</p>';
}

function formatArchetypeName(archetype) {
    const names = {
        sniper: 'Sniper',
        long_term_accumulator: 'Accumulator',
        coordinated_cluster: 'Coordinated',
        liquidity_actor: 'LP Actor',
        exchange_linked: 'Exchange',
        dormant_whale: 'Dormant Whale',
        unknown: 'Unknown',
    };
    return names[archetype] || archetype;
}

function getArchetypeClass(archetype) {
    const classes = {
        sniper: 'archetype-sniper',
        long_term_accumulator: 'archetype-accumulator',
        coordinated_cluster: 'archetype-coordinated',
        liquidity_actor: 'archetype-lp',
        exchange_linked: 'archetype-exchange',
        dormant_whale: 'archetype-whale',
        unknown: 'archetype-unknown',
    };
    return classes[archetype] || 'archetype-unknown';
}

function getHolderType(share) {
    if (share >= 0.05) return 'whale';
    if (share >= 0.01) return 'large';
    if (share >= 0.001) return 'medium';
    return 'small';
}

// Utility functions
function truncateAddress(address, length = 10) {
    if (!address) return '';
    if (address.length <= length) return address;
    const half = Math.floor(length / 2);
    return `${address.slice(0, half)}...${address.slice(-half)}`;
}

function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function formatLargeNumber(num) {
    if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toFixed(2);
}

function formatPrice(price) {
    if (price >= 1) return price.toFixed(2);
    if (price >= 0.01) return price.toFixed(4);
    if (price >= 0.0001) return price.toFixed(6);
    return price.toExponential(2);
}
