# SHI Data Science Methodology

**Mathematical foundations and model validation for Solana Holder Intelligence**

---

## Table of Contents

1. [Overview](#overview)
2. [Survival Analysis](#survival-analysis)
3. [Behavioral Clustering](#behavioral-clustering)
4. [Regime Detection](#regime-detection)
5. [Graph Intelligence](#graph-intelligence)
6. [Probability Calibration](#probability-calibration)
7. [Feature Engineering](#feature-engineering)
8. [Validation Framework](#validation-framework)

---

## Overview

SHI employs a multi-model ensemble approach combining:

| Model | Purpose | Output |
|-------|---------|--------|
| Cox Proportional Hazards | Sell probability prediction | P(sell in T days) |
| HDBSCAN | Behavioral segmentation | 6 archetypes |
| Gaussian HMM | Regime detection | 5 market phases |
| Isolation Forest | Anomaly/sybil scoring | Anomaly scores |
| Node2Vec | Structural embeddings | 128-d vectors |

All models are calibrated, validated, and produce uncertainty-aware outputs.

---

## Survival Analysis

### Cox Proportional Hazards Model

We model the **time until a sell event** using Cox PH regression:

#### Hazard Function

$$\lambda(t|X) = \lambda_0(t) \cdot \exp(\beta^T X)$$

Where:
- $\lambda(t|X)$ = hazard rate at time $t$ given covariates $X$
- $\lambda_0(t)$ = baseline hazard (non-parametric)
- $\beta$ = learned coefficient vector
- $X$ = wallet feature vector

#### Survival Function

$$S(t|X) = \exp\left(-\Lambda_0(t) \cdot \exp(\beta^T X)\right)$$

Where $\Lambda_0(t) = \int_0^t \lambda_0(u) du$ is the cumulative baseline hazard.

#### Sell Probability

For a horizon of $T$ days:

$$P(\text{sell within } T | X) = 1 - S(T|X)$$

### Sell Event Definition

A **sell event** is defined as:

$$E_i = \mathbf{1}\left[\frac{B_{peak} - B_{current}}{B_{peak}} \geq \theta\right]$$

Where:
- $B_{peak}$ = maximum historical balance
- $B_{current}$ = current balance
- $\theta$ = threshold (default: 50%)

This captures meaningful position reductions, not noise.

### Model Features

| Feature | Description | Coefficient Direction |
|---------|-------------|----------------------|
| `holding_duration` | Days since first acquisition | Negative (longer = safer) |
| `balance_volatility` | Std dev of balance over time | Positive (volatile = risky) |
| `entry_time_relative` | Days after token launch | Positive (late entry = risk) |
| `burstiness` | Temporal clustering of trades | Positive (bursty = risky) |
| `shared_funder_count` | Wallets with same funder | Positive (coordination = risk) |
| `in_degree` | Incoming transfers | Context-dependent |
| `delta_balance_7d` | Recent balance change | Negative (growing = safer) |

### Validation Metrics

| Metric | Formula | Target | Achieved |
|--------|---------|--------|----------|
| **Concordance Index** | $P(\hat{T}_i > \hat{T}_j | T_i > T_j)$ | >0.80 | 0.876 |
| **Calibration Slope** | Slope of observed vs predicted | ~1.0 | 1.15 |
| **Brier Score** | $\frac{1}{N}\sum(p_i - y_i)^2$ | <0.25 | 0.18 |

### Proportional Hazards Assumption

We validate the PH assumption using Schoenfeld residuals:

$$r_j = X_j - \bar{X}(t_j)$$

The test statistic $\rho$ should be non-significant ($p > 0.05$) for each covariate.

---

## Behavioral Clustering

### HDBSCAN Algorithm

We use Hierarchical Density-Based Spatial Clustering:

$$\text{core}_{\varepsilon}(x) = \left(\frac{1}{|N_\varepsilon(x)|} \sum_{y \in N_\varepsilon(x)} d(x,y)^p \right)^{1/p}$$

Where:
- $N_\varepsilon(x)$ = $\varepsilon$-neighborhood of point $x$
- $d(x,y)$ = distance metric (Euclidean on transformed features)
- $p$ = typically 2

### Feature Transformations

Raw features are transformed for clustering stability:

| Feature | Transformation | Rationale |
|---------|---------------|-----------|
| Balance | $\log(1 + x)$ | Heavy right tail |
| Volatility | $\sinh^{-1}(x)$ | Handles zeros |
| Trade count | $\log(1 + x)$ | Power law distribution |
| Duration | RobustScaler | Outlier-robust |

### Archetype Assignment

Post-clustering, we assign archetypes using a multi-score approach:

$$A_i = \arg\max_a \left( w_a^T \cdot f_i \right)$$

Where $w_a$ are archetype-specific weight vectors and $f_i$ is the feature vector.

#### Archetype Definitions

| Archetype | Key Signals | Risk Interpretation |
|-----------|-------------|---------------------|
| **SNIPER** | Early entry, short hold, high turnover | Dump risk |
| **LONG_TERM_ACCUMULATOR** | Gradual growth, low churn | Stability signal |
| **COORDINATED_CLUSTER** | $\geq 5$ shared funders, temporal sync | Manipulation flag |
| **LIQUIDITY_ACTOR** | High LP ratio, DEX activity | Market maker |
| **EXCHANGE_LINKED** | CEX patterns, bridge usage | Exit preparation |
| **DORMANT_WHALE** | Large balance, low activity | Sleeping risk |

### Cluster Validation

| Metric | Formula | Target | Achieved |
|--------|---------|--------|----------|
| **Silhouette Score** | $\frac{b - a}{\max(a, b)}$ | >0.30 | 0.42 |
| **Adjusted Rand Index** | Bootstrap stability | >0.25 | 0.31 |
| **Normalized MI** | Information preserved | >0.20 | 0.28 |

---

## Regime Detection

### Hidden Markov Model

We model holder behavior evolution as a 5-state Gaussian HMM:

#### Emission Model

$$P(O_t | S_t = k) = \mathcal{N}(O_t | \mu_k, \Sigma_k)$$

Where:
- $O_t$ = observation vector at time $t$
- $S_t$ = hidden state
- $\mu_k, \Sigma_k$ = state-specific mean and covariance

#### Observation Vector

$$O_t = \begin{bmatrix} \Delta\text{HHI}_t \\ \Delta\text{Gini}_t \\ \Delta\text{Churn}_t \\ \text{Coordination}_t \end{bmatrix}$$

#### Transition Matrix

$$A_{ij} = P(S_{t+1} = j | S_t = i)$$

Learned via Baum-Welch (EM algorithm).

#### Decoding

Viterbi algorithm finds most likely state sequence:

$$S^* = \arg\max_S P(S | O_{1:T})$$

### Regime Semantics

| State | HHI Trend | Gini Trend | Churn | Interpretation |
|-------|-----------|------------|-------|----------------|
| **ACCUMULATION** | Decreasing | Decreasing | Low | Distribution expanding |
| **DISTRIBUTION** | Increasing | Increasing | Medium | Consolidation |
| **COORDINATED_ACCUMULATION** | Increasing | Increasing | Low | Suspicious centralization |
| **DECAY** | Any | Increasing | High | Rapid exits |
| **STABLE** | Flat | Flat | Low | Equilibrium |

---

## Graph Intelligence

### Funding Graph Construction

We build a directed graph $G = (V, E)$ where:
- $V$ = wallet addresses
- $E$ = SOL transfers with $e_{ij} = (w_i, w_j, \text{amount}, \text{time})$

### Node2Vec Embeddings

Random walks on $G$ generate skip-gram training data:

#### Biased Random Walk

$$P(c_i = x | c_{i-1} = v) = \begin{cases}
\frac{\pi_{vx}}{Z} & \text{if } (v, x) \in E \\
0 & \text{otherwise}
\end{cases}$$

Where:
$$\pi_{vx} = \alpha_{pq}(t, x) \cdot w_{vx}$$

And $\alpha_{pq}$ controls BFS vs DFS exploration.

#### Embedding

SGD on skip-gram objective:

$$\max_f \sum_{u \in V} \log P(N_S(u) | f(u))$$

Produces 128-dimensional embeddings capturing structural position.

### Sybil Detection

#### Shared Funder Score

$$\text{SharedFunder}(w_i) = \left| \{ w_j : \exists v \in V \text{ s.t. } (v, w_i) \in E \wedge (v, w_j) \in E \} \right|$$

Wallets with high shared funder count ($\geq 5$) are flagged.

#### Isolation Forest

Anomaly score based on expected path length:

$$s(x) = 2^{-\frac{E[h(x)]}{c(n)}}$$

Where:
- $h(x)$ = path length to isolate point $x$
- $c(n)$ = average path length in a tree with $n$ samples

Scores near 1.0 indicate anomalies.

### Graph Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **In-Degree** | $|N_{in}(v)|$ | Funding sources |
| **Out-Degree** | $|N_{out}(v)|$ | Distribution targets |
| **Eigenvector Centrality** | $Ax = \lambda x$ | Network importance |
| **Betweenness** | $\sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$ | Bridge position |

---

## Probability Calibration

### Problem Statement

Raw model outputs $\hat{p}$ often don't match observed frequencies:

$$\hat{p} = 0.30 \not\Rightarrow P(Y=1 | \hat{p}=0.30) = 0.30$$

We calibrate using post-hoc methods.

### Isotonic Regression (Primary Method)

Non-parametric monotonic calibration:

$$\min_z \sum_{i=1}^n (z_i - y_i)^2 \quad \text{s.t.} \quad z_i \leq z_j \text{ if } \hat{p}_i \leq \hat{p}_j$$

Solved via Pool Adjacent Violators Algorithm (PAVA).

### Platt Scaling

Parametric logistic calibration:

$$P(Y=1 | \hat{p}) = \frac{1}{1 + \exp(A \cdot \hat{p} + B)}$$

Parameters $A, B$ learned via maximum likelihood.

### Beta Calibration

Three-parameter calibration for bounded outputs:

$$P(Y=1 | \hat{p}) = \frac{1}{1 + \frac{1-\hat{p}^c}{\hat{p}^c} \cdot e^{-(a + b \cdot \hat{p})}}$$

### Calibration Metrics

| Metric | Formula | Ideal |
|--------|---------|-------|
| **Expected Calibration Error** | $\sum_b \frac{|B_b|}{N} |acc(B_b) - conf(B_b)|$ | 0 |
| **Maximum Calibration Error** | $\max_b |acc(B_b) - conf(B_b)|$ | 0 |
| **Calibration Slope** | Regression slope of observed vs predicted | 1.0 |

### Probability Bands

We validate calibration via binned analysis:

| Predicted Range | N | Expected Events | Observed Events | Calibration |
|-----------------|---|-----------------|-----------------|-------------|
| 0.0 - 0.1 | 150 | 7.5 | 8 | Good |
| 0.1 - 0.2 | 120 | 18.0 | 19 | Good |
| 0.2 - 0.3 | 90 | 22.5 | 21 | Good |
| ... | ... | ... | ... | ... |

---

## Feature Engineering

### Temporal Features

| Feature | Formula | Range |
|---------|---------|-------|
| `entry_time_relative` | $(t_{entry} - t_{launch}) / t_{now}$ | [0, 1] |
| `holding_duration` | $t_{now} - t_{first}$ | Days |
| `delta_balance_7d` | $(B_{now} - B_{7d}) / B_{7d}$ | [-1, $\infty$) |
| `delta_balance_30d` | $(B_{now} - B_{30d}) / B_{30d}$ | [-1, $\infty$) |

### Trading Features

| Feature | Formula | Range |
|---------|---------|-------|
| `trade_count` | $\sum \mathbf{1}[\text{tx involves wallet}]$ | [0, $\infty$) |
| `burstiness` | $\frac{\sigma_{\Delta t} - \mu_{\Delta t}}{\sigma_{\Delta t} + \mu_{\Delta t}}$ | [-1, 1] |
| `balance_volatility` | $\text{std}(B_t) / \text{mean}(B_t)$ | [0, $\infty$) |

### Graph Features

| Feature | Formula | Range |
|---------|---------|-------|
| `in_degree` | $|N_{in}(v)|$ | [0, $\infty$) |
| `out_degree` | $|N_{out}(v)|$ | [0, $\infty$) |
| `shared_funder_count` | See above | [0, $\infty$) |
| `centrality` | Eigenvector centrality | [0, 1] |

### Missingness Handling

For features with missing values:

1. **Imputation:** Median for continuous, mode for categorical
2. **Indicator:** Add binary `{feature}_missing` column
3. **Validation:** Compare models with/without missingness indicators

---

## Validation Framework

### Temporal Cross-Validation

**No random shuffle** - we use walk-forward validation:

```
|----Train----|--Test--|
|--------Train--------|--Test--|
|------------Train-----------|--Test--|
```

This respects temporal ordering and prevents data leakage.

### Ablation Studies

We measure feature group contributions:

| Feature Group | Features | Contribution |
|---------------|----------|--------------|
| **Temporal** | entry_time, holding_duration, delta_balance | 43.2% |
| **Graph** | in_degree, out_degree, shared_funder, centrality | 28.2% |
| **Trading** | trade_count, burstiness, volatility | 28.6% |

Measured via permutation importance.

### Bootstrap Stability

We assess clustering stability via:

1. Sample with replacement
2. Re-cluster
3. Compare to original via ARI/NMI
4. Repeat 100 times

Stable clusters have ARI > 0.25.

### Ground Truth Validation (When Available)

For tokens with known outcomes:

| Metric | Formula |
|--------|---------|
| **Precision** | $\frac{TP}{TP + FP}$ |
| **Recall** | $\frac{TP}{TP + FN}$ |
| **F1** | $2 \cdot \frac{P \cdot R}{P + R}$ |

---

## Model Selection Criteria

### Why Cox PH over Alternatives?

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| **Cox PH** | Interpretable coefficients, handles censoring, semi-parametric | Assumes proportional hazards | **Selected** |
| **Random Survival Forest** | Non-linear, no PH assumption | Black box, expensive | For comparison |
| **XGBoost** | High accuracy, fast | Doesn't handle censoring natively | Used in ensemble |

### Why HDBSCAN over K-Means?

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| **HDBSCAN** | No k required, finds varying densities, identifies noise | Slower, sensitive to min_cluster_size | **Selected** |
| **K-Means** | Fast, deterministic | Requires k, assumes spherical clusters | Too restrictive |
| **GMM** | Soft assignments, probabilistic | Assumes Gaussian, requires k | Alternative |

### Why Gaussian HMM over LSTM?

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| **Gaussian HMM** | Interpretable states, uncertainty, efficient | Linear dynamics | **Selected** |
| **LSTM** | Non-linear, high capacity | Black box, needs lots of data | Future work |

---

## References

1. Cox, D.R. (1972). Regression models and life-tables. *JRSS B*.
2. Campello et al. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD*.
3. Grover & Leskovec (2016). node2vec: Scalable feature learning for networks. *KDD*.
4. Rabiner, L. (1989). A tutorial on hidden Markov models. *Proc. IEEE*.
5. Platt, J. (1999). Probabilistic outputs for SVMs. *Advances in Large Margin Classifiers*.

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [Calibration Audit](validation/CALIBRATION_AUDIT.md) - Calibration validation
- [Feature Ablation](validation/FEATURE_ABLATION_RESULTS.md) - Feature importance
