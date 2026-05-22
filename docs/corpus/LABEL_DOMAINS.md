# Label Domains

## Overview

The Intelligence Corpus organizes labels into 6 distinct domains, each targeting a specific aspect of on-chain behaviour analysis.

## Domain 1: Exit Events

**Domain ID:** `exit_event`

Classification of how tokens left a wallet.

### Labels

| Label | Description |
|-------|-------------|
| `dex_sell` | Sold via DEX (Raydium, Jupiter, etc.) |
| `transfer_out` | Transferred to another wallet |
| `lp_add` | Added to liquidity pool |
| `lp_remove` | Removed from liquidity pool |
| `cex_deposit` | Deposited to CEX |
| `burn` | Tokens burned |
| `swap_intermediate` | Intermediate swap in a route |
| `stake` | Staked in protocol |
| `bridge_out` | Bridged to another chain |
| `unknown` | Cannot classify |
| `ambiguous` | Multiple valid interpretations |
| `needs_more_context` | Insufficient evidence |

### Impact

Exit classification directly affects PnL calculation. Misclassifying a transfer as a sell would create phantom profits/losses.

---

## Domain 2: Coordination

**Domain ID:** `coordination`

Assessment of whether wallet clusters exhibit coordinated behaviour.

### Labels

| Label | Description |
|-------|-------------|
| `true_coordinated` | Genuine coordinated behaviour |
| `false_positive` | Mistakenly grouped, not coordinated |
| `partially_coordinated` | Some coordination, some independent |
| `unknown_coordination` | Cannot determine |
| `legitimate_coordination` | Coordinated but legitimate (e.g., fund operations) |

### Impact

False positives can incorrectly flag legitimate activity. True coordination detection is critical for identifying manipulation.

---

## Domain 3: Wallet Behaviour

**Domain ID:** `wallet_behaviour`

Classification of wallet behaviour patterns.

### Labels

| Label | Description |
|-------|-------------|
| `sniper` | Quick entry/exit, targets launches |
| `accumulator` | Gradual buying, long holds |
| `whale` | Large position sizes |
| `retail` | Small, irregular trades |
| `bot` | Automated trading patterns |
| `market_maker` | Provides liquidity, tight spreads |
| `institutional` | Large, structured operations |
| `unknown_behaviour` | Cannot classify |

### Impact

Behaviour classification affects trust scoring and anomaly detection.

---

## Domain 4: Token Outcomes

**Domain ID:** `token_outcome`

Classification of token lifecycle outcomes.

### Labels

| Label | Description |
|-------|-------------|
| `rug_pull` | Deliberate exit scam |
| `organic_failure` | Failed despite legitimate effort |
| `success` | Achieved sustainable activity |
| `slow_rug` | Gradual extraction by insiders |
| `abandoned` | Team abandoned without extracting |
| `ongoing` | Still active, outcome unclear |
| `unknown_outcome` | Cannot determine |

### Impact

Outcome labels are high-stakes - mislabeling affects trust in the entire system.

---

## Domain 5: Launch Trajectories

**Domain ID:** `launch_trajectory`

Classification of token launch patterns.

### Labels

| Label | Description |
|-------|-------------|
| `organic` | Natural launch, no suspicious activity |
| `insider_coordinated` | Launch with insider coordination |
| `botted` | Heavy bot activity at launch |
| `fair_launch` | Designed for fair distribution |
| `pre_mined` | Significant pre-mine before launch |
| `unknown_trajectory` | Cannot classify |

### Impact

Launch trajectory affects initial trust assessment for new tokens.

---

## Domain 6: Entity Resolution

**Domain ID:** `entity_resolution`

Decisions about wallet entity relationships.

### Labels

| Label | Description |
|-------|-------------|
| `same_entity` | Wallets controlled by same entity |
| `different_entity` | Wallets controlled by different entities |
| `uncertain_link` | Cannot determine relationship |
| `partial_link` | Some evidence of connection |

### Impact

Entity resolution affects coordination detection and whale tracking.

---

## Domain Selection Guide

| Use Case | Domain |
|----------|--------|
| PnL calculation accuracy | exit_event |
| Cluster validation | coordination |
| Wallet profiling | wallet_behaviour |
| Token risk assessment | token_outcome |
| New token evaluation | launch_trajectory |
| Whale tracking | entity_resolution |

## Cross-Domain Relationships

```
exit_event ← affects → wallet_behaviour
    ↓
coordination ← validates → entity_resolution
    ↓
token_outcome ← depends on → launch_trajectory
```

Labels in one domain often provide evidence for labels in another. The corpus maintains these relationships through evidence packages.
