# PRODUCT DESIGN REQUIREMENTS (PDR)  
# Solana Holder Intelligence (SHI)

---

# 1. PRODUCT VISION

Solana Holder Intelligence (SHI) is a probabilistic on-chain intelligence engine designed to analyze token holder structure and behavioral dynamics for tokens on the Solana blockchain.

The system will:

- Analyze holder distribution
- Cluster wallets into behavioral archetypes
- Detect coordination and Sybil-like funding patterns
- Estimate sell-risk probabilities for major holders
- Produce stability and manipulation risk indicators
- Deliver outputs via Telegram interface

The system provides probabilistic structural intelligence.  
It does NOT provide trading signals.  

All behavioral outputs must be uncertainty-aware and reproducible.

---

# 2. CORE DESIGN PRINCIPLES

1. Deterministic Metrics Layer (immutable).
2. Probabilistic Modeling Layer (interpretable).
3. Graph-First Architecture.
4. Explainable Outputs.
5. Telegram-First Interaction Model.
6. MCP-Based Modular Agent Orchestration.
7. Percentile-Based Baseline Normalization.

---

# 3. SYSTEM ARCHITECTURE

## 3.1 Layer 1 — Data Acquisition

Inputs:
- Token mint address
- Token holders (mint → token accounts)
- Wallet metadata (funded_by, first funded timestamp)
- Parsed transaction history
- Token balances over time

Funding Graph Definition:

```
G = (V, E)
```

Where:

```
V = set of wallet addresses
E = set of directed edges representing funding transfers
```

Additional derived graphs:

```
G_transfer = (V, E_transfer)
G_cotrade = (V, E_similarity)
```

---

## 3.2 Layer 2 — Feature Engineering

Per Wallet Features:

### Distribution Features

```
s_i = b_i / SUM(b_j)
```

Where:

```
b_i = balance of wallet i
```

### Temporal Features

- Entry time relative to token launch
- Holding duration
- Inter-trade intervals
- Position volatility

### Flow Features

```
Delta_b(t) = b(t) - b(t - Delta_t)
```

### Trade Burstiness

```
Burstiness = (sigma_tau - mu_tau) / (sigma_tau + mu_tau)
```

Where:

```
tau = inter-trade interval
mu_tau = mean interval
sigma_tau = standard deviation of intervals
```

### Graph Features

- In-degree
- Out-degree
- Eigenvector centrality
- Community membership
- Shared upstream funders

---

# 4. METRICS ENGINE (IMMUTABLE)

THE FOLLOWING METRICS ARE FROZEN.  
NO AGENT MAY MODIFY FORMULAS OR DEFINITIONS.

---

## 4.1 Herfindahl–Hirschman Index (HHI)

```
HHI = SUM( s_i^2 )
```

Where:

```
s_i = b_i / SUM(b_j)
```

---

## 4.2 Shannon Entropy

```
H = - SUM( s_i * log(s_i) )
```

---

## 4.3 Gini Coefficient

```
G = ( SUM_i SUM_j |b_i - b_j| ) / ( 2 * N * SUM(b_i) )
```

Where:

```
N = number of wallets
```

---

## 4.4 Whale Dominance Ratio (WDR)

```
WDR = ( SUM_{i=1 to k} b_i ) / Total_Supply
```

Where:

```
k = number of top wallets (default: 10)
```

---

## 4.5 Churn Rate

```
Churn = Wallets_Exited_in_Window / Wallets_At_Window_Start
```

---

## 4.6 Coordination Score (Cluster-Level)

```
Coord(C) = Shared_Funder_Count / Size_of_Cluster
```

Where:

```
Shared_Funder_Count = number of wallets in cluster sharing dominant upstream funder
```

---

## 4.7 Funding Density

```
Funding_Density = |E| / ( |V| * (|V| - 1) )
```

---

## 4.8 Sell Hazard Model

Survival Hazard Function:

```
lambda(t | x) = lambda_0(t) * exp( beta^T * x )
```

Where:

```
x = wallet feature vector
beta = learned coefficients
lambda_0(t) = baseline hazard
```

Sell Probability within Horizon T:

```
P_sell(T) = 1 - exp( - Integral_0_to_T lambda(t | x) dt )
```

---

## 4.9 Sell Pressure Index

```
Sell_Pressure = SUM_{i in Top_N} P_sell_i(T)
```

---

## 4.10 Z-Score Normalization

All metrics must be normalized against baseline reference datasets.

```
Z = ( X - mu ) / sigma
```

Where:

```
X = observed metric value
mu = baseline mean
sigma = baseline standard deviation
```

Baseline reference classes must include:

- Established Solana tokens
- Known rug tokens
- High-liquidity blue-chip tokens

---

# 5. WALLET ARCHETYPE DEFINITIONS (FIXED)

Behavioral clustering must assign wallets into one of the following archetypes.

These are behavioral classifications only.

---

## 5.1 Snipers

Characteristics:
- Early entry time
- Short holding duration
- High turnover
- High trade frequency

---

## 5.2 Long-Term Accumulators

Characteristics:
- Gradual position growth
- Low churn
- Low trade frequency

---

## 5.3 Coordinated Cluster Members

Characteristics:
- Shared upstream funders
- Temporal synchronization in entry
- Graph community clustering

---

## 5.4 Liquidity Actors

Characteristics:
- Frequent liquidity pool interactions
- Add/remove liquidity cycles

---

## 5.5 Exchange-Linked

Characteristics:
- High fan-out patterns
- Known CEX-linked transfer signatures

---

## 5.6 Dormant Whales

Characteristics:
- Large share of supply
- Low transaction activity
- Long holding duration

---

# 6. TOKEN-LEVEL RISK SCORES

## 6.1 Token Stability Score (0–100)

Weighted function of:

- HHI
- Gini
- Whale Dominance Ratio
- Churn Rate
- Coordination Score

Weights must be stored persistently.  
Metric formulas may NOT change.

---

## 6.2 Sybil Probability Index

Function of:

- Funding graph density
- Shared funder ratio
- Temporal clustering
- Coordination score

Must produce probabilistic output with uncertainty bounds.

---

# 7. TELEGRAM INTERFACE REQUIREMENTS

Mandatory command:

```
/analyze <token_mint>
```

Required Output:

- Distribution snapshot
- Whale breakdown
- Archetype proportions
- Stability score
- Sell pressure index
- Coordination alert flag
- Sybil probability
- Confidence intervals

Optional commands:

```
/summary
/top_holders
/risk
/graph
/history
```

Latency Target:

```
<= 30 seconds for typical token
```

---

# 8. STORAGE AND REPRODUCIBILITY

All of the following must be persistently stored:

- Metric definitions (locked)
- Baseline datasets
- Model parameters
- Training data hashes
- Iteration logs
- Version history

All modeling must be reproducible.

---

# 9. CONSTRAINTS

1. Metric formulas are immutable.
2. Archetype definitions are fixed.
3. No identity inference beyond behavioral classification.
4. Outputs must be uncertainty-aware.
5. Percentile normalization required.
6. Telegram delivery mandatory.

---

# END OF PDR