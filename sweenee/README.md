# SWEENEE Whale Wallet Dashboard 🐳

A community-facing dashboard for tracking whale wallet activity for the **SWEENEE** token on Solana.

## What This Dashboard Does

- **Tracks Whale Wallets**: Monitors a configured list of high-interest wallet addresses
- **Shows Current Holdings**: Displays SWEENEE token balances for each tracked wallet
- **Transaction Feed**: Shows recent SWEENEE movements (buys, sells, transfers)
- **Summary Metrics**: Total holdings, net flows, concentration metrics
- **Telegram Summaries**: Generates copy-paste ready updates for Telegram

## What This Dashboard Does NOT Do

- Track all SWEENEE holders (only configured wallets)
- Predict prices or provide trading signals
- Guarantee accuracy of transaction classifications
- Provide financial advice

## Quick Start

### 1. Add Wallet Addresses

Create wallet files in the `wallets/` directory:

**Plain text (`wallets/whales.txt`):**
```
# One address per line
9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU

# Or with labels
Whale1: 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

**CSV (`wallets/tracked.csv`):**
```csv
label,address,notes
Whale 1,9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM,Top holder
Whale 2,7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU,
```

**JSON (`wallets/list.json`):**
```json
[
  {"label": "Whale 1", "address": "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"},
  {"label": "Whale 2", "address": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"}
]
```

### 2. Set Up Environment

```bash
cd /Users/q/PythonScript/Python/Vibe/SHI/sweenee

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys if needed
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## Configuration

Edit `.env` or `config.py` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `SWEENEE_MINT` | `FkAtYamtEMtgnsTeUhzhTCiT2Svyxw63UdUYp1T7pump` | SWEENEE token mint address |
| `HELIUS_API_KEY` | (from parent SHI) | Helius RPC API key |
| `BALANCE_CACHE_TTL` | `300` | Seconds before refreshing balances |
| `TRANSACTION_CACHE_TTL` | `300` | Seconds before refreshing transactions |

## Metrics Explained

### Balance Metrics

| Metric | Description |
|--------|-------------|
| **Total Tracked Wallets** | Number of configured wallet addresses |
| **Wallets Holding** | Wallets with SWEENEE balance > 0 |
| **Total SWEENEE Held** | Sum of all tracked wallet balances |
| **Share %** | Wallet's percentage of tracked holdings |

### Concentration Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Top 10 Share** | Σ top_10 / total | Combined share of top 10 holders |
| **HHI** | Σ share² | Herfindahl-Hirschman Index (0-1, higher = more concentrated) |

**Note:** These metrics only reflect tracked wallets, not all SWEENEE holders.

### Flow Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Net Flow 24h** | Σ inflows - Σ outflows | Net SWEENEE movement in last 24 hours |
| **Net Flow 7d** | Σ inflows - Σ outflows | Net SWEENEE movement in last 7 days |

### Transaction Classification

| Type | Meaning |
|------|---------|
| **BUY** | DEX swap: SWEENEE received, SOL/USDC sent |
| **SELL** | DEX swap: SWEENEE sent, SOL/USDC received |
| **TRANSFER_IN** | SWEENEE received without swap context |
| **TRANSFER_OUT** | SWEENEE sent without swap context |
| **UNKNOWN** | Classification uncertain |

Classification is conservative. If unsure, transactions are marked as UNKNOWN.

## Project Structure

```
sweenee/
├── app.py                 # Streamlit dashboard entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── README.md              # This file
│
├── wallets/               # Wallet address files
│   └── sample_whales.txt  # Example wallet list
│
├── src/
│   ├── __init__.py
│   ├── wallet_loader.py   # Load wallets from txt/csv/json
│   ├── solana_client.py   # Solana RPC client
│   ├── token_balances.py  # Balance fetching
│   ├── transactions.py    # Transaction parsing/classification
│   ├── cache.py           # SQLite persistence
│   ├── metrics.py         # Dashboard metrics
│   ├── telegram_summary.py # Telegram text generation
│   └── utils.py           # Helper functions
│
├── data/
│   ├── sweenee.sqlite     # Cache database (auto-created)
│   ├── cache/
│   ├── raw/
│   └── processed/
│
└── tests/
    └── (test files)
```

## Generating Telegram Summaries

The dashboard includes a "Telegram Summary" section that generates copy-paste ready messages.

**Daily format:**
```
🐳 SWEENEE Whale Wallet Watch

Tracked wallets: 24
Wallets holding SWEENEE: 17
Total tracked SWEENEE: 12,450,000

24h net flow: +320,000 SWEENEE
24h transactions: 8

Largest holder:
ABCD...wxyz — 3,100,000 SWEENEE

Not financial advice. Community transparency only.
```

## API Dependencies

The dashboard uses:
- **Solana RPC** (free tier available)
- **Helius API** (optional, for enhanced data)

It reuses API keys from the parent SHI project when available.

## Known Limitations

1. **Selected wallets only**: Only tracks configured addresses, not all holders
2. **Transaction classification**: May not correctly identify all swap types
3. **Rate limits**: High-frequency refreshes may hit API limits
4. **Cache staleness**: Data may be up to 5 minutes old by default
5. **No historical charts**: Currently shows snapshot only

## Future Improvements

- [ ] Historical balance tracking charts
- [ ] Alert system for large movements
- [ ] Webhook integration for automated Telegram posting
- [ ] Additional DEX support (Orca, Jupiter v6+)
- [ ] Wallet grouping/entity detection
- [ ] Export to CSV/JSON

## Disclaimer

This dashboard is for **community transparency and research only**.

- It tracks selected configured wallets, **not all SWEENEE holders**
- Transaction classifications are **probabilistic and may be incorrect**
- This is **not financial advice**
- Use at your own risk

---

Built as part of the SHI (Solana Holder Intelligence) ecosystem.
