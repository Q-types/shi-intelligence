# Feature Ablation Results

Generated: 2026-05-20T16:40:42.154609+00:00

## Executive Summary

**Essential Groups:** temporal

**Redundant Groups:** distribution, flow, trading, graph, price_pnl, liquidity

**Harmful Groups:** None identified

**Recommendation:** KEEP: temporal (essential for performance); OPTIONAL: distribution, flow, trading, graph, price_pnl, liquidity (minimal impact)

---

## Baseline Metrics

- **Silhouette Score:** N/A
- **Noise Rate:** 1.0%
- **Concordance Index:** 0.847

## Feature Group Impact Analysis

| Group | Stability Δ | Noise Δ | Sell Pred Δ | Essential | Redundant | Harmful |
|-------|-------------|---------|-------------|-----------|-----------|---------|
| distribution | 0.000 | -0.0% | 0.001 |  | ✓ |  |
| temporal | 0.000 | +0.0% | -0.250 | ✓ |  |  |
| flow | 0.000 | -0.0% | -0.002 |  | ✓ |  |
| trading | 0.000 | -0.1% | 0.003 |  | ✓ |  |
| graph | 0.000 | -0.1% | 0.003 |  | ✓ |  |
| price_pnl | 0.000 | +0.0% | -0.002 |  | ✓ |  |
| liquidity | 0.000 | +0.0% | 0.001 |  | ✓ |  |

## Detailed Group Analysis

### Distribution

**Features:** balance, share, rank

**Impact Metrics:**
- Cluster Stability Change: +0.000
- Noise Rate Change: -0.0%
- UNKNOWN Rate Change: -0.0%
- Sell Prediction (C-index) Change: +0.001

**Assessment:**
- Redundant: Minimal impact when removed

### Temporal

**Features:** entry_time_relative, holding_duration, position_volatility

**Impact Metrics:**
- Cluster Stability Change: +0.000
- Noise Rate Change: +0.0%
- UNKNOWN Rate Change: +0.0%
- Sell Prediction (C-index) Change: -0.250

**Assessment:**
- Essential: Removing drops concordance by 0.250

### Flow

**Features:** delta_balance_7d, delta_balance_30d

**Impact Metrics:**
- Cluster Stability Change: +0.000
- Noise Rate Change: -0.0%
- UNKNOWN Rate Change: -0.0%
- Sell Prediction (C-index) Change: -0.002

**Assessment:**
- Redundant: Minimal impact when removed

### Trading

**Features:** trade_count, burstiness, swap_frequency, lp_interaction_ratio

**Impact Metrics:**
- Cluster Stability Change: +0.000
- Noise Rate Change: -0.1%
- UNKNOWN Rate Change: -0.0%
- Sell Prediction (C-index) Change: +0.003

**Assessment:**
- Redundant: Minimal impact when removed

### Graph

**Features:** in_degree, out_degree, eigenvector_centrality, shared_funder_count, total_funding_received, largest_funder_share, funding_hhi, funding_burst_score, weighted_in_degree, weighted_out_degree

**Impact Metrics:**
- Cluster Stability Change: +0.000
- Noise Rate Change: -0.1%
- UNKNOWN Rate Change: -0.1%
- Sell Prediction (C-index) Change: +0.003

**Assessment:**
- Redundant: Minimal impact when removed

### Price_Pnl

**Features:** entry_price_usd, current_price_usd, unrealized_pnl_ratio, unrealized_pnl_usd, price_change_1h_pct, price_change_24h_pct, price_change_7d_pct

**Impact Metrics:**
- Cluster Stability Change: +0.000
- Noise Rate Change: +0.0%
- UNKNOWN Rate Change: +0.0%
- Sell Prediction (C-index) Change: -0.002

**Assessment:**
- Redundant: Minimal impact when removed

### Liquidity

**Features:** liquidity_usd_current, liquidity_usd_1h_avg, liquidity_usd_24h_avg, sell_pressure_vs_liquidity, unrealized_profit_concentration

**Impact Metrics:**
- Cluster Stability Change: +0.000
- Noise Rate Change: +0.0%
- UNKNOWN Rate Change: +0.0%
- Sell Prediction (C-index) Change: +0.001

**Assessment:**
- Redundant: Minimal impact when removed

---

## Recommendations

KEEP: temporal (essential for performance); OPTIONAL: distribution, flow, trading, graph, price_pnl, liquidity (minimal impact)