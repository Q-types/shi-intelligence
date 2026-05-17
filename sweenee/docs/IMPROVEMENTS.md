# SWEENEE Dashboard Improvements Plan

**Created:** 2026-05-17
**Based on:** UX_AUDIT.md findings
**Goal:** Transform dashboard from data presentation to confidence communication

---

## Implementation Priority

### Tier 1: Quick Wins (High Impact, Low Effort)
*Implement first - immediate user experience improvement*

| # | Improvement | Impact | Effort | Est. Lines |
|---|-------------|--------|--------|------------|
| 1 | Hero Confidence Banner | 10/10 | Small | ~30 |
| 2 | Color-Coded Transaction Rows | 8/10 | Small | ~20 |
| 3 | Remove/Hide HHI Metric | 7/10 | Tiny | ~5 |
| 4 | Buy Alert Banner | 9/10 | Small | ~25 |
| 5 | Accumulation Phase Indicator | 9/10 | Small | ~20 |

### Tier 2: Strategic Improvements (High Impact, Medium Effort)
*Implement second - significant UX transformation*

| # | Improvement | Impact | Effort | Est. Lines |
|---|-------------|--------|--------|------------|
| 6 | Whale Confidence Score | 8/10 | Medium | ~50 |
| 7 | Buy Streak Counter | 8/10 | Medium | ~30 |
| 8 | Supply Context Display | 7/10 | Medium | ~25 |
| 9 | Plain English Summary | 8/10 | Medium | ~40 |
| 10 | Positive Framing Rewrites | 7/10 | Medium | ~30 |

### Tier 3: Nice-to-Have (Lower Priority)
*Implement if time allows*

| # | Improvement | Impact | Effort |
|---|-------------|--------|--------|
| 11 | Tooltips for Metrics | 5/10 | Medium |
| 12 | Historical Comparison | 6/10 | Large |
| 13 | Celebration Animations | 4/10 | Medium |

---

## Detailed Implementation Specs

### 1. Hero Confidence Banner

**Goal:** First thing users see answers "Are whales holding?"

**Location:** After header, before metrics (line ~186 in app.py)

**Implementation:**

```python
# Add after line 186 (after st.caption)

def render_confidence_banner(metrics: DashboardMetrics, transactions: list):
    """Render the hero confidence banner."""

    # Determine status
    all_holding = metrics.wallets_holding == metrics.total_tracked_wallets
    net_positive_24h = metrics.net_flow_24h >= 0
    net_positive_7d = metrics.net_flow_7d >= 0

    # Count recent buys
    recent_buys = sum(1 for tx in transactions
                     if tx.classification.value == "buy"
                     and tx.block_time and
                     (datetime.now(timezone.utc) - tx.block_time).total_seconds() < 86400)

    # Build message
    if all_holding and net_positive_7d:
        status = "strong"
        icon = "💎"
        color = "#4CAF50"
        message = f"All {metrics.total_tracked_wallets} tracked whales still holding"
        submessage = f"+{format_number(metrics.net_flow_7d)} SWEENEE net inflow this week"
    elif all_holding:
        status = "steady"
        icon = "🐳"
        color = "#2196F3"
        message = f"All {metrics.total_tracked_wallets} tracked whales still holding"
        submessage = "Positions stable"
    else:
        status = "neutral"
        icon = "📊"
        color = "#9E9E9E"
        message = f"{metrics.wallets_holding} of {metrics.total_tracked_wallets} whales holding"
        submessage = "Monitor for changes"

    # Add buy activity
    if recent_buys > 0:
        submessage += f" • {recent_buys} whale{'s' if recent_buys > 1 else ''} bought today"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
        border-left: 4px solid {color};
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    ">
        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">
            {icon} {message}
        </div>
        <div style="font-size: 1rem; color: #666; margin-top: 0.25rem;">
            {submessage}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Call after st.caption on line 186
render_confidence_banner(metrics, transactions)
```

**Acceptance Criteria:**
- [ ] Banner appears immediately after header
- [ ] Shows "All X whales still holding" when true
- [ ] Green styling when net positive
- [ ] Shows buy count when whales bought today

---

### 2. Color-Coded Transaction Rows

**Goal:** Make buys visually exciting, sells muted

**Location:** Transaction feed section (lines 399-431)

**Implementation:**

```python
# Replace the transaction dataframe rendering (lines 418-431)

if transactions:
    tx_data = []
    for tx in transactions[:50]:
        # Find wallet label
        label = None
        for w in wallets:
            if w["address"] == tx.wallet_address:
                label = w.get("label")
                break

        # Determine row styling
        tx_type = tx.classification.value.upper()
        if tx_type == "BUY":
            row_emoji = "🟢"
            amount_prefix = "+"
        elif tx_type == "SELL":
            row_emoji = "🔴"
            amount_prefix = ""
        elif tx_type == "TRANSFER_IN":
            row_emoji = "⬆️"
            amount_prefix = "+"
        elif tx_type == "TRANSFER_OUT":
            row_emoji = "⬇️"
            amount_prefix = ""
        else:
            row_emoji = "⚪"
            amount_prefix = ""

        tx_data.append({
            "": row_emoji,
            "Time": tx.block_time.strftime("%m/%d %H:%M") if tx.block_time else "—",
            "Wallet": label or short_address(tx.wallet_address),
            "Action": tx_type,
            "Amount": f"{amount_prefix}{format_number(abs(tx.amount_change))}",
            "Explorer": tx.explorer_url,
        })

    df_tx = pd.DataFrame(tx_data)
    st.dataframe(
        df_tx,
        column_config={
            "": st.column_config.TextColumn("", width="small"),
            "Time": st.column_config.TextColumn("Time", width="small"),
            "Wallet": st.column_config.TextColumn("Whale", width="medium"),
            "Action": st.column_config.TextColumn("Action", width="small"),
            "Amount": st.column_config.TextColumn("SWEENEE", width="medium"),
            "Explorer": st.column_config.LinkColumn("🔗", display_text="View", width="small"),
        },
        hide_index=True,
        use_container_width=True,
    )
```

**Acceptance Criteria:**
- [ ] Buy rows have green indicator
- [ ] Sell rows have red indicator
- [ ] Transfer directions are clear
- [ ] Amount shows + prefix for inflows

---

### 3. Remove/Hide HHI Metric

**Goal:** Remove confusing technical metric

**Location:** Metrics row 2 (line 262)

**Implementation:**

```python
# Replace the HHI metric at line 261-262 with something useful

# BEFORE:
# with m6:
#     st.metric("HHI (Concentration)", f"{metrics.hhi:.4f}")

# AFTER:
with m6:
    # Show days since last major sell instead
    days_accumulating = compute_accumulation_days(transactions)
    if days_accumulating > 0:
        st.metric(
            "Accumulation Streak",
            f"{days_accumulating} days",
            "Net buyers",
        )
    else:
        st.metric(
            "Activity",
            "Mixed",
            "Monitor",
        )
```

**Helper function to add:**

```python
def compute_accumulation_days(transactions: list) -> int:
    """Count consecutive days of net positive flow."""
    if not transactions:
        return 0

    # Group by day and sum
    from collections import defaultdict
    daily_flow = defaultdict(float)

    for tx in transactions:
        if tx.block_time:
            day = tx.block_time.date()
            daily_flow[day] += tx.amount_change

    # Count streak from today backwards
    today = datetime.now(timezone.utc).date()
    streak = 0

    for i in range(30):  # Check last 30 days
        check_date = today - timedelta(days=i)
        if check_date in daily_flow:
            if daily_flow[check_date] >= 0:
                streak += 1
            else:
                break
        # Skip days with no activity

    return streak
```

**Acceptance Criteria:**
- [ ] HHI no longer visible by default
- [ ] Replaced with "Accumulation Streak" counter
- [ ] Shows days of consecutive net buying

---

### 4. Buy Alert Banner

**Goal:** Celebrate recent large buys prominently

**Location:** After confidence banner, before metrics

**Implementation:**

```python
def render_buy_alerts(transactions: list, wallets: list, threshold: float = 1_000_000):
    """Show prominent alerts for recent large buys."""

    # Find large buys in last 24h
    now = datetime.now(timezone.utc)
    large_buys = []

    for tx in transactions:
        if (tx.classification.value == "buy" and
            tx.amount_change >= threshold and
            tx.block_time and
            (now - tx.block_time).total_seconds() < 86400):

            # Find label
            label = None
            for w in wallets:
                if w["address"] == tx.wallet_address:
                    label = w.get("label")
                    break

            large_buys.append({
                "wallet": label or short_address(tx.wallet_address),
                "amount": tx.amount_change,
                "time": tx.block_time,
            })

    if not large_buys:
        return

    # Sort by amount descending
    large_buys.sort(key=lambda x: x["amount"], reverse=True)

    for buy in large_buys[:3]:  # Show top 3
        hours_ago = (now - buy["time"]).total_seconds() / 3600
        time_str = f"{hours_ago:.0f}h ago" if hours_ago >= 1 else "Just now"

        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #4CAF5022 0%, transparent 100%);
            border-left: 3px solid #4CAF50;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            display: flex;
            align-items: center;
        ">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">🐳</span>
            <span style="font-weight: 600; color: #4CAF50;">
                {buy["wallet"]} bought +{format_number(buy["amount"])} SWEENEE
            </span>
            <span style="color: #888; margin-left: auto; font-size: 0.85rem;">
                {time_str}
            </span>
        </div>
        """, unsafe_allow_html=True)

# Call after confidence banner
render_buy_alerts(transactions, wallets, threshold=500_000)
```

**Acceptance Criteria:**
- [ ] Large buys (>500K) shown prominently
- [ ] Green styling with whale emoji
- [ ] Shows time ago
- [ ] Maximum 3 alerts shown

---

### 5. Accumulation Phase Indicator

**Goal:** Instantly show if whales are accumulating or distributing

**Location:** Top metrics row or hero section

**Implementation:**

```python
def compute_market_phase(metrics: DashboardMetrics) -> tuple[str, str, str]:
    """Determine accumulation/distribution phase.

    Returns: (phase_name, emoji, color)
    """
    net_7d = metrics.net_flow_7d
    net_24h = metrics.net_flow_24h

    if net_7d > 0 and net_24h >= 0:
        return ("Accumulation", "📈", "#4CAF50")
    elif net_7d > 0 and net_24h < 0:
        return ("Accumulation (pause)", "📊", "#8BC34A")
    elif net_7d < 0 and net_24h <= 0:
        return ("Distribution", "📉", "#F44336")
    elif net_7d < 0 and net_24h > 0:
        return ("Distribution (slowing)", "📊", "#FF9800")
    else:
        return ("Neutral", "➡️", "#9E9E9E")

# Add to metrics display
phase, phase_emoji, phase_color = compute_market_phase(metrics)

# Replace one of the less useful metrics slots
with m5:  # Was "Top 10 Share"
    st.markdown(f"""
    <div style="text-align: center;">
        <div style="font-size: 0.85rem; color: #666;">Whale Phase</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {phase_color};">
            {phase_emoji} {phase}
        </div>
    </div>
    """, unsafe_allow_html=True)
```

**Acceptance Criteria:**
- [ ] Phase indicator visible in top metrics
- [ ] Shows "Accumulation" when net positive
- [ ] Green for accumulation, red for distribution
- [ ] Clear emoji indicators

---

### 6. Whale Confidence Score

**Goal:** Single number summarizing whale conviction

**Location:** Hero section or new prominent card

**Implementation:**

```python
def compute_confidence_score(metrics: DashboardMetrics, transactions: list) -> tuple[int, str]:
    """Compute whale confidence score 0-100.

    Factors:
    - % of wallets holding (30%)
    - 7d net flow direction (25%)
    - 24h net flow direction (20%)
    - Buy vs sell ratio (25%)

    Returns: (score, description)
    """
    score = 0

    # Factor 1: Holding ratio (30 points)
    holding_ratio = metrics.wallets_holding / max(metrics.total_tracked_wallets, 1)
    score += int(holding_ratio * 30)

    # Factor 2: 7d flow (25 points)
    if metrics.net_flow_7d > 0:
        score += 25
    elif metrics.net_flow_7d == 0:
        score += 12
    else:
        score += 0

    # Factor 3: 24h flow (20 points)
    if metrics.net_flow_24h > 0:
        score += 20
    elif metrics.net_flow_24h == 0:
        score += 10
    else:
        score += 0

    # Factor 4: Buy/sell ratio (25 points)
    buys = sum(1 for tx in transactions if tx.classification.value == "buy")
    sells = sum(1 for tx in transactions if tx.classification.value == "sell")
    total = buys + sells
    if total > 0:
        buy_ratio = buys / total
        score += int(buy_ratio * 25)
    else:
        score += 12  # Neutral if no trades

    # Description
    if score >= 80:
        desc = "Very Strong"
    elif score >= 60:
        desc = "Strong"
    elif score >= 40:
        desc = "Neutral"
    elif score >= 20:
        desc = "Weak"
    else:
        desc = "Very Weak"

    return (score, desc)

# Render as a prominent card
score, desc = compute_confidence_score(metrics, transactions)

# Color based on score
if score >= 70:
    score_color = "#4CAF50"
elif score >= 40:
    score_color = "#FF9800"
else:
    score_color = "#F44336"

st.markdown(f"""
<div style="
    background: white;
    border: 2px solid {score_color};
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
">
    <div style="font-size: 0.9rem; color: #666; text-transform: uppercase; letter-spacing: 1px;">
        Whale Confidence Score
    </div>
    <div style="font-size: 3rem; font-weight: 700; color: {score_color};">
        {score}
    </div>
    <div style="font-size: 1.1rem; color: {score_color}; font-weight: 600;">
        {desc}
    </div>
</div>
""", unsafe_allow_html=True)
```

**Acceptance Criteria:**
- [ ] Single 0-100 score visible
- [ ] Color-coded (green/yellow/red)
- [ ] Clear description (Strong/Neutral/Weak)
- [ ] Prominently placed

---

### 7. Buy Streak Counter

**Goal:** Show consecutive days of net buying

**Location:** Metrics row or hero section

**Implementation:**

```python
def compute_buy_streak(transactions: list) -> tuple[int, str]:
    """Count consecutive days of net positive whale activity.

    Returns: (streak_days, description)
    """
    from collections import defaultdict

    if not transactions:
        return (0, "No data")

    # Group transactions by day
    daily_net = defaultdict(float)
    for tx in transactions:
        if tx.block_time:
            day = tx.block_time.date()
            daily_net[day] += tx.amount_change

    if not daily_net:
        return (0, "No data")

    # Sort days and count streak from most recent
    today = datetime.now(timezone.utc).date()
    streak = 0

    for i in range(30):
        check_date = today - timedelta(days=i)
        if check_date in daily_net:
            if daily_net[check_date] > 0:
                streak += 1
            elif daily_net[check_date] < 0:
                break  # Streak broken
            # Zero days don't break streak but don't count

    if streak >= 7:
        desc = "Strong buying pressure"
    elif streak >= 3:
        desc = "Consistent accumulation"
    elif streak >= 1:
        desc = "Recent buying"
    else:
        desc = "Mixed activity"

    return (streak, desc)

# Display
streak, streak_desc = compute_buy_streak(transactions)

if streak > 0:
    st.markdown(f"""
    <div style="
        background: #E8F5E9;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        display: inline-block;
    ">
        <span style="font-size: 1.2rem;">🔥</span>
        <span style="font-weight: 600; color: #2E7D32;">
            {streak} day{'s' if streak != 1 else ''} of net buying
        </span>
        <span style="color: #666; font-size: 0.9rem; margin-left: 0.5rem;">
            {streak_desc}
        </span>
    </div>
    """, unsafe_allow_html=True)
```

**Acceptance Criteria:**
- [ ] Shows consecutive buy days
- [ ] Fire emoji for visual impact
- [ ] Description of what streak means
- [ ] Only shows when streak > 0

---

### 8. Supply Context Display

**Goal:** Make holdings meaningful with supply context

**Location:** After total holdings metric

**Implementation:**

```python
# Add supply context
SWEENEE_TOTAL_SUPPLY = 1_000_000_000  # 1 billion - UPDATE WITH ACTUAL VALUE

def render_supply_context(total_held: float, total_supply: float = SWEENEE_TOTAL_SUPPLY):
    """Show what % of supply whales are tracking."""

    pct_of_supply = (total_held / total_supply) * 100

    st.markdown(f"""
    <div style="
        font-size: 0.85rem;
        color: #666;
        margin-top: -0.5rem;
    ">
        {pct_of_supply:.1f}% of total supply tracked
    </div>
    """, unsafe_allow_html=True)

# Call after Total SWEENEE Held metric
render_supply_context(metrics.total_sweenee)
```

**Acceptance Criteria:**
- [ ] Shows % of total supply
- [ ] Placed contextually near holdings
- [ ] Uses actual supply number

---

### 9. Plain English Summary

**Goal:** Explain what the data means in simple terms

**Location:** New section after metrics, before charts

**Implementation:**

```python
def generate_plain_english_summary(metrics: DashboardMetrics, transactions: list) -> str:
    """Generate simple summary for non-technical users."""

    parts = []

    # Holding status
    if metrics.wallets_holding == metrics.total_tracked_wallets:
        parts.append(f"All {metrics.total_tracked_wallets} tracked whale wallets are still holding SWEENEE.")
    else:
        parts.append(f"{metrics.wallets_holding} out of {metrics.total_tracked_wallets} whale wallets are holding SWEENEE.")

    # Flow interpretation
    if metrics.net_flow_7d > 0:
        parts.append(f"This week, whales have bought more than they sold ({format_number(metrics.net_flow_7d)} net inflow).")
    elif metrics.net_flow_7d < 0:
        parts.append(f"This week, whales have sold more than they bought ({format_number(abs(metrics.net_flow_7d))} net outflow).")
    else:
        parts.append("This week, whale buying and selling is balanced.")

    # 24h activity
    if metrics.transaction_count_24h == 0:
        parts.append("No whale movements in the last 24 hours.")
    elif metrics.net_flow_24h > 0:
        parts.append(f"In the last 24 hours: {metrics.transaction_count_24h} transactions, net buying.")
    else:
        parts.append(f"In the last 24 hours: {metrics.transaction_count_24h} transactions.")

    # Largest holder
    if metrics.largest_holder_label:
        parts.append(f"The biggest tracked holder is {metrics.largest_holder_label} with {format_number(metrics.largest_holder_balance)} SWEENEE.")

    return " ".join(parts)

# Render summary
st.markdown("### 📝 What This Means")
summary = generate_plain_english_summary(metrics, transactions)
st.info(summary)
```

**Acceptance Criteria:**
- [ ] Simple language, no jargon
- [ ] Answers "are whales holding?"
- [ ] Interprets flows (buying vs selling)
- [ ] Placed prominently

---

### 10. Positive Framing Rewrites

**Goal:** Reframe neutral labels to emphasize positive news

**Changes:**

| Current | Reframe To | Location |
|---------|-----------|----------|
| "24h Net Flow" | "24h Whale Activity" or "Whales Bought/Sold" | line 246 |
| "Tracked Wallets" | "Whales Monitored" | line 233 |
| "Total SWEENEE Held" | "Whale Holdings" | line 239 |
| "Wallets holding" | "Diamond Hands" | line 237 |
| "Recent Whale Movements" | "Whale Activity Feed" | line 397 |
| "Largest Holder" | "Top Whale" | line 252 |

**Implementation:**

```python
# Rewrite metric labels
with m1:
    st.metric(
        "🐳 Whales Monitored",
        metrics.total_tracked_wallets,
        f"💎 {metrics.wallets_holding} diamond hands",
    )
with m2:
    st.metric(
        "💰 Whale Holdings",
        format_number(metrics.total_sweenee),
    )
with m3:
    # Frame positively when possible
    if metrics.net_flow_24h >= 0:
        label = "24h Net Buying"
        value = f"+{format_number(metrics.net_flow_24h)}"
    else:
        label = "24h Net Selling"
        value = format_number(metrics.net_flow_24h)

    st.metric(
        label,
        value,
        f"{metrics.transaction_count_24h} moves",
    )
with m4:
    st.metric(
        "👑 Top Whale",
        format_number(metrics.largest_holder_balance),
        metrics.largest_holder_label or short_address(metrics.largest_holder_address or ""),
    )
```

**Acceptance Criteria:**
- [ ] Labels feel positive/exciting
- [ ] "Diamond hands" terminology used
- [ ] Whale emoji on key metrics
- [ ] Buying framed positively

---

## Implementation Order

Execute improvements in this order for maximum impact:

```
Phase 1 (Immediate - Do First):
├── 1. Hero Confidence Banner
├── 4. Buy Alert Banner
└── 5. Accumulation Phase Indicator

Phase 2 (Quick Wins):
├── 2. Color-Coded Transaction Rows
├── 3. Remove HHI / Add Accumulation Streak
└── 10. Positive Framing Rewrites

Phase 3 (Enhanced Features):
├── 6. Whale Confidence Score
├── 7. Buy Streak Counter
├── 9. Plain English Summary
└── 8. Supply Context Display

Phase 4 (If Time Permits):
├── 11. Tooltips
├── 12. Historical Comparison
└── 13. Animations
```

---

## Testing Checklist

After implementation, verify:

- [ ] Dashboard loads without errors
- [ ] Confidence banner shows correct status
- [ ] Buy alerts appear for recent large buys
- [ ] Transaction rows are color-coded
- [ ] Phase indicator shows correct direction
- [ ] All text is non-technical
- [ ] Mobile view still works
- [ ] Performance not degraded

---

## Files to Modify

| File | Changes |
|------|---------|
| `app.py` | Main improvements (lines 175-460) |
| `src/metrics.py` | Add new computation functions |

**No new files needed** - all improvements are additions to existing code.

---

## Success Metrics

After implementation, the dashboard should:

1. **3-Second Test:** User knows "whales are holding" within 3 seconds
2. **Buy Visibility:** Large buys are impossible to miss
3. **Emotional Impact:** Dashboard feels reassuring, not clinical
4. **Accessibility:** No unexplained technical terms visible
5. **Confidence Score:** Single number summarizes whale conviction
