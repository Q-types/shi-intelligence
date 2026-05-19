/**
 * SHI Dashboard - Wallet Graph Visualization
 * D3.js force-directed graph for wallet relationships
 */

class WalletGraph {
    constructor(containerId) {
        this.containerId = containerId;
        this.svg = null;
        this.g = null;
        this.simulation = null;
        this.width = 800;
        this.height = 500;
        this.showLabels = true;

        // Colors for share-based types
        this.colors = {
            whale: '#e74c3c',
            large: '#f39c12',
            medium: '#3498db',
            small: '#95a5a6',
            funder: '#9b59b6'
        };

        // Colors for archetypes (behavioral classification)
        this.archetypeColors = {
            sniper: '#e74c3c',
            long_term_accumulator: '#27ae60',
            coordinated_cluster: '#9b59b6',
            liquidity_actor: '#3498db',
            exchange_linked: '#f39c12',
            dormant_whale: '#1abc9c',
            unknown: '#95a5a6'
        };
    }

    init() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error('Graph container not found:', this.containerId);
            return;
        }

        // Clear existing content
        container.innerHTML = '';

        this.width = container.clientWidth || 800;
        this.height = container.clientHeight || 500;

        console.log('Graph init: width=', this.width, 'height=', this.height);

        // Create SVG
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .style('background', '#1a1f2e');

        // Add glow filter for funding links
        const defs = this.svg.append('defs');
        const filter = defs.append('filter')
            .attr('id', 'glow')
            .attr('x', '-50%')
            .attr('y', '-50%')
            .attr('width', '200%')
            .attr('height', '200%');
        filter.append('feGaussianBlur')
            .attr('stdDeviation', '3')
            .attr('result', 'coloredBlur');
        const feMerge = filter.append('feMerge');
        feMerge.append('feMergeNode').attr('in', 'coloredBlur');
        feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });

        this.svg.call(zoom);

        // Create main group for zoom/pan
        this.g = this.svg.append('g');

        // Create groups for links and nodes
        this.linksGroup = this.g.append('g').attr('class', 'links');
        this.nodesGroup = this.g.append('g').attr('class', 'nodes');

        // Remove old tooltip if exists
        d3.select('.graph-tooltip').remove();

        // Create tooltip
        this.tooltip = d3.select('body')
            .append('div')
            .attr('class', 'graph-tooltip tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('pointer-events', 'none');
    }

    setData(holders, fundingEdges = []) {
        if (!this.svg) {
            console.error('Graph not initialized');
            return;
        }

        console.log('Graph setData:', holders.length, 'holders,', fundingEdges.length, 'edges');

        // Create nodes with random initial positions
        const nodes = holders.map((holder, index) => ({
            id: holder.wallet,
            wallet: holder.wallet,
            balance: holder.balance,
            share: holder.share,
            rank: index + 1,
            type: this.getHolderType(holder.share),
            archetype: holder.archetype || 'unknown',
            hasSharedFunder: Boolean(holder.shared_funders && holder.shared_funders.length > 0),
            x: this.width / 2 + (Math.random() - 0.5) * this.width * 0.8,
            y: this.height / 2 + (Math.random() - 0.5) * this.height * 0.8
        }));

        // Create a Set of valid node IDs
        const nodeIds = new Set(nodes.map(n => n.id));

        // Filter funding edges to only valid node references
        const validFundingLinks = fundingEdges
            .filter(edge => nodeIds.has(edge.from) && nodeIds.has(edge.to))
            .map(edge => ({
                source: edge.from,
                target: edge.to,
                type: 'funding'
            }));

        console.log('Valid funding links:', validFundingLinks.length);

        // Create proximity links between top holders
        const links = [...validFundingLinks];
        const topHolders = nodes.slice(0, Math.min(10, nodes.length));
        for (let i = 0; i < topHolders.length - 1; i++) {
            for (let j = i + 1; j < topHolders.length; j++) {
                if (Math.random() < 0.25) {
                    links.push({
                        source: topHolders[i].id,
                        target: topHolders[j].id,
                        type: 'proximity'
                    });
                }
            }
        }

        console.log('Total links:', links.length);

        // Stop old simulation
        if (this.simulation) {
            this.simulation.stop();
        }

        // Create new simulation
        this.simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(80).strength(0.5))
            .force('charge', d3.forceManyBody().strength(-400))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => this.getNodeRadius(d.share) + 5))
            .force('x', d3.forceX(this.width / 2).strength(0.03))
            .force('y', d3.forceY(this.height / 2).strength(0.03));

        this.render(nodes, links);
    }

    getHolderType(share) {
        if (share >= 0.05) return 'whale';
        if (share >= 0.01) return 'large';
        if (share >= 0.001) return 'medium';
        return 'small';
    }

    getNodeRadius(share) {
        const baseRadius = 8;
        const maxRadius = 35;
        return Math.min(maxRadius, baseRadius + Math.log10((share || 0.0001) * 1000 + 1) * 8);
    }

    getNodeColor(d) {
        // Use archetype color if archetype is known and not 'unknown'
        if (d.archetype && d.archetype !== 'unknown' && this.archetypeColors[d.archetype]) {
            return this.archetypeColors[d.archetype];
        }
        // Fall back to share-based type color
        return this.colors[d.type];
    }

    render(nodes, links) {
        const self = this;

        // Clear existing elements
        this.linksGroup.selectAll('*').remove();
        this.nodesGroup.selectAll('*').remove();

        // Draw links
        const linkElements = this.linksGroup.selectAll('.link')
            .data(links)
            .enter()
            .append('line')
            .attr('class', d => `link ${d.type || ''}`)
            .attr('stroke', d => d.type === 'funding' ? '#00d4ff' : '#38444d')
            .attr('stroke-width', d => d.type === 'funding' ? 6 : 1)
            .attr('stroke-opacity', d => d.type === 'funding' ? 1.0 : 0.3)
            .attr('stroke-dasharray', d => d.type === 'funding' ? 'none' : 'none')
            .attr('filter', d => d.type === 'funding' ? 'url(#glow)' : 'none');

        // Draw nodes
        const nodeElements = this.nodesGroup.selectAll('.node')
            .data(nodes)
            .enter()
            .append('g')
            .attr('class', 'node')
            .attr('cursor', 'pointer')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)))
            .on('mouseover', function(event, d) { self.showTooltip(event, d); })
            .on('mouseout', function() { self.hideTooltip(); })
            .on('click', function(event, d) { self.nodeClicked(event, d); });

        nodeElements.append('circle')
            .attr('r', d => this.getNodeRadius(d.share))
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', d => d.hasSharedFunder ? '#9b59b6' : '#fff')
            .attr('stroke-width', d => d.hasSharedFunder ? 3 : 2)
            .attr('stroke-dasharray', d => d.archetype === 'coordinated_cluster' ? '3,2' : 'none');

        if (this.showLabels) {
            nodeElements.append('text')
                .attr('dy', d => this.getNodeRadius(d.share) + 14)
                .attr('text-anchor', 'middle')
                .attr('fill', '#8899a6')
                .attr('font-size', '10px')
                .text(d => this.truncateWallet(d.wallet));
        }

        // Update positions on tick
        this.simulation.on('tick', () => {
            linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodeElements.attr('transform', d => `translate(${d.x},${d.y})`);
        });

        // Store references
        this.nodeElements = nodeElements;
        this.linkElements = linkElements;
        this.nodes = nodes;
        this.links = links;
    }

    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    showTooltip(event, d) {
        const archetypeName = this.formatArchetypeName(d.archetype);
        const archetypeColor = this.archetypeColors[d.archetype] || '#95a5a6';
        const sharedFunderNote = d.hasSharedFunder
            ? `<div style="margin-top:8px;padding-top:8px;border-top:1px solid #38444d;color:#9b59b6;font-size:11px;">🔗 Shares funder with other wallets</div>`
            : '';

        this.tooltip
            .style('opacity', 1)
            .style('left', (event.pageX + 15) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
                <div style="font-weight:600;margin-bottom:8px;">${this.truncateWallet(d.wallet, 12)}</div>
                <div style="display:flex;justify-content:space-between;gap:16px;margin-bottom:4px;">
                    <span style="color:#8899a6;">Rank:</span>
                    <span>#${d.rank}</span>
                </div>
                <div style="display:flex;justify-content:space-between;gap:16px;margin-bottom:4px;">
                    <span style="color:#8899a6;">Balance:</span>
                    <span>${this.formatNumber(d.balance)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;gap:16px;margin-bottom:4px;">
                    <span style="color:#8899a6;">Share:</span>
                    <span>${(d.share * 100).toFixed(2)}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;gap:16px;margin-bottom:4px;">
                    <span style="color:#8899a6;">Size:</span>
                    <span style="color:${this.colors[d.type]};font-weight:600;">${d.type.toUpperCase()}</span>
                </div>
                <div style="display:flex;justify-content:space-between;gap:16px;">
                    <span style="color:#8899a6;">Behavior:</span>
                    <span style="color:${archetypeColor};font-weight:600;">${archetypeName}</span>
                </div>
                ${sharedFunderNote}
            `);
    }

    formatArchetypeName(archetype) {
        const names = {
            sniper: 'Sniper',
            long_term_accumulator: 'Accumulator',
            coordinated_cluster: 'Coordinated',
            liquidity_actor: 'LP Actor',
            exchange_linked: 'Exchange',
            dormant_whale: 'Dormant Whale',
            unknown: 'Unknown',
        };
        return names[archetype] || archetype || 'Unknown';
    }

    hideTooltip() {
        this.tooltip.style('opacity', 0);
    }

    nodeClicked(event, d) {
        navigator.clipboard.writeText(d.wallet).then(() => {
            const circle = d3.select(event.currentTarget).select('circle');
            const originalColor = circle.attr('fill');
            circle
                .transition().duration(150).attr('fill', '#ffffff')
                .transition().duration(150).attr('fill', originalColor);
        }).catch(err => console.log('Copy failed:', err));
    }

    truncateWallet(wallet, length = 8) {
        if (!wallet) return '';
        if (wallet.length <= length) return wallet;
        const half = Math.floor(length / 2);
        return `${wallet.slice(0, half)}...${wallet.slice(-half)}`;
    }

    formatNumber(num) {
        if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
        if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
        return num.toFixed(2);
    }

    toggleLabels(show) {
        this.showLabels = show;
        if (this.nodeElements) {
            this.nodeElements.selectAll('text').remove();
            if (show) {
                this.nodeElements.append('text')
                    .attr('dy', d => this.getNodeRadius(d.share) + 14)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#8899a6')
                    .attr('font-size', '10px')
                    .text(d => this.truncateWallet(d.wallet));
            }
        }
    }

    reset() {
        if (this.svg) {
            this.svg.transition()
                .duration(500)
                .call(d3.zoom().transform, d3.zoomIdentity);
        }
        if (this.simulation) {
            this.simulation.alpha(1).restart();
        }
    }

    destroy() {
        if (this.tooltip) {
            this.tooltip.remove();
        }
        if (this.simulation) {
            this.simulation.stop();
        }
    }
}

// Export for use in app.js
window.WalletGraph = WalletGraph;
