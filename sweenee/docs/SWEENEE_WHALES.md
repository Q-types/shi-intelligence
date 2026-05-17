Below is a copy-paste-ready agent prompt for building the separate SWEENEE whale-wallet dashboard.

# Agent Task: Build SWEENEE Whale Wallet Dashboard
You are acting as a senior full-stack data science engineering team with additional marketing, community, and crypto analytics advisors.
Your task is to build a separate dashboard for tracking whale-wallet activity for the Solana token **SWEENEE**.
## Project Context
This project sits within the broader Solana Holder Intelligence / SHI ecosystem, but it must be implemented as a separate dashboard/module so it can be shared regularly with the SWEENEE Telegram community.
The dashboard should allow the community to see:
1. Which tracked whale wallets hold SWEENEE.
2. How much SWEENEE each wallet currently holds.
3. All recent SWEENEE-related transactions involving those wallets.
4. Whether each transaction is a buy, sell, transfer in, transfer out, or other token movement.
5. Summary metrics that are easy for non-technical Telegram members to understand.
6. A shareable dashboard view, suitable for linking or screenshotting into Telegram.
## Token Details
Token name: SWEENEE  
Contract address / mint address:
```text
FkAtYamtEMtgnsTeUhzhTCiT2Svyxw63UdUYp1T7pump

Local Project Location

Create the dashboard project inside:

/Users/q/PythonScript/Python/Vibe/SHI/sweenee

The tracked whale wallet addresses are currently located in:

/Users/q/PythonScript/Python/Vibe/SHI/sweenee/wallets

You must inspect this directory and infer the wallet file format. It may contain one or more files. The implementation should be robust to common formats such as:

* .txt with one wallet address per line
* .csv with wallet columns
* .json with wallet arrays or labelled wallet objects

Do not hard-code wallet addresses unless required for testing. The wallet directory should remain the source of truth.

⸻

Main Goal

Build a working SWEENEE whale-wallet dashboard that:

* Loads tracked wallet addresses from /Users/q/PythonScript/Python/Vibe/SHI/sweenee/wallets
* Fetches current SWEENEE token balances for each wallet
* Fetches recent SWEENEE token transactions for each wallet
* Stores/cache results locally
* Presents the data in a clean dashboard
* Includes Telegram-friendly summary outputs
* Is easy to run locally and later deploy

⸻

Recommended Architecture

Use the existing Solana API/key setup already available in the SHI environment. Reuse existing provider/client infrastructure if appropriate, but keep the SWEENEE dashboard cleanly separated.

Suggested folder structure:

/Users/q/PythonScript/Python/Vibe/SHI/sweenee
│
├── README.md
├── requirements.txt or pyproject.toml
├── .env.example
│
├── app.py                         # Main Streamlit/Dash dashboard entrypoint
├── config.py                      # Token config, paths, API options
│
├── wallets/
│   └── existing wallet files
│
├── src/
│   ├── __init__.py
│   ├── wallet_loader.py           # Load wallet addresses from txt/csv/json
│   ├── solana_client.py           # API wrapper or adapter to existing SHI provider
│   ├── token_balances.py          # Current SWEENEE balance logic
│   ├── transactions.py            # Transaction fetching/parsing/classification
│   ├── cache.py                   # Local cache/database layer
│   ├── metrics.py                 # Aggregated dashboard metrics
│   ├── telegram_summary.py        # Telegram-friendly text generation
│   └── utils.py
│
├── data/
│   ├── cache/
│   ├── raw/
│   └── processed/
│
└── tests/
    ├── test_wallet_loader.py
    ├── test_transaction_parser.py
    └── test_metrics.py

⸻

Dashboard Requirements

Use Streamlit unless the existing project already has a preferred dashboard framework.

The dashboard should include the following pages or sections.

1. Header / Hero Section

Display:

* SWEENEE Whale Wallet Watch
* Token mint address
* Last updated timestamp
* Number of tracked wallets
* Total SWEENEE held by tracked wallets

The dashboard should make clear that it tracks only the configured whale wallets, not all holders.

Suggested wording:

SWEENEE Whale Wallet Watch tracks selected high-interest wallets and shows their SWEENEE balances and token movements in near-real time.

Avoid overclaiming. Do not imply financial advice.

Add a small disclaimer:

This dashboard is for community transparency and research only. It is not financial advice.

⸻

2. Summary Metrics

Create clear metric cards:

* Total tracked wallets
* Wallets currently holding SWEENEE
* Total SWEENEE held by tracked wallets
* Largest SWEENEE wallet among tracked wallets
* Net SWEENEE flow over last 24h
* Number of SWEENEE transactions over last 24h
* Largest recent buy / inflow
* Largest recent sell / outflow

Where price data is available, optionally include estimated USD value. If reliable price data is not available, leave USD values out rather than guessing.

⸻

3. Wallet Balance Table

Create a sortable/filterable table with:

Column	Meaning
Wallet label	If available from file, otherwise shortened address
Wallet address	Full or copyable address
Short address	e.g. ABCD…wxyz
Current SWEENEE balance	Token amount
Percentage of tracked-wallet holdings	wallet_balance / tracked_total
Last SWEENEE transaction time	Most recent relevant transaction
Net flow 24h	SWEENEE in minus SWEENEE out
Net flow 7d	SWEENEE in minus SWEENEE out
Transaction count 24h	Number of relevant token movements

Use sensible formatting for large numbers.

For percentage of tracked-wallet holdings:

p_i = \frac{B_i}{\sum_{j=1}^{N} B_j}

Plain spoken form:

p sub i equals B sub i divided by the sum from j equals one to N of B sub j.

Where:

* B_i = SWEENEE balance of wallet i
* N = number of tracked wallets

⸻

4. Transaction Feed

Create a live/recent transaction feed filtered to SWEENEE movements involving tracked wallets.

Columns:

Column	Meaning
Time	Transaction timestamp
Wallet label	Tracked wallet involved
Wallet address	Address
Type	Buy, sell, transfer in, transfer out, unknown
SWEENEE amount	Token amount
Counterparty	Other wallet/account if available
Signature	Solana transaction signature
Explorer link	Solscan or equivalent

Transaction classification should be conservative. If unsure, label as unknown rather than forcing a buy/sell classification.

Suggested classification hierarchy:

1. If DEX swap data is available and wallet receives SWEENEE while giving SOL/USDC/other token, classify as buy.
2. If DEX swap data is available and wallet sends SWEENEE while receiving SOL/USDC/other token, classify as sell.
3. If SWEENEE moves into the wallet without clear swap context, classify as transfer_in.
4. If SWEENEE moves out of the wallet without clear swap context, classify as transfer_out.
5. Otherwise classify as unknown.

Net flow formula:

F_i = \sum \text{SWEENEE}_{\text{in}, i} - \sum \text{SWEENEE}_{\text{out}, i}

Plain spoken form:

F sub i equals the sum of SWEENEE into wallet i minus the sum of SWEENEE out of wallet i.

⸻

5. Charts

Add simple, Telegram-screenshot-friendly charts.

Required charts:

1. Top tracked wallets by SWEENEE balance
    * Horizontal bar chart
    * Top 10 or top 20 wallets
2. Net SWEENEE flow by wallet
    * 24h and 7d options
3. SWEENEE transaction activity over time
    * Hourly or daily counts
4. Cumulative tracked-wallet SWEENEE holdings
    * If historical cache exists

Avoid clutter. This is for a Telegram community, not an institutional terminal. The dashboard should feel transparent, fast, and readable.

⸻

6. Telegram Summary Generator

Create a function that produces a short Telegram-ready update.

Example output:

🐳 SWEENEE Whale Wallet Watch
Tracked wallets: 24
Wallets holding SWEENEE: 17
Total tracked SWEENEE: 12,450,000
24h net flow: +320,000 SWEENEE
24h transactions: 8
Largest holder:
ABCD...wxyz — 3,100,000 SWEENEE
Largest 24h movement:
+180,000 SWEENEE into EFGH...1234
Dashboard:
[insert dashboard URL when deployed]
Not financial advice. Community transparency only.

Also create a more detailed version for weekly summaries.

The summary should be generated from the same processed data as the dashboard, not manually typed.

⸻

Data Engineering Requirements

Wallet Loading

Implement robust wallet loading.

The loader should support:

Plain text

wallet_address_1
wallet_address_2
wallet_address_3

CSV

Possible columns:

wallet
address
wallet_address
label
name
notes

JSON

Possible structures:

[
  "wallet_address_1",
  "wallet_address_2"
]

or:

[
  {
    "label": "Whale 1",
    "address": "wallet_address_1"
  }
]

Return a standard structure:

@dataclass
class TrackedWallet:
    address: str
    label: str | None = None
    notes: str | None = None
    source_file: str | None = None

Deduplicate wallet addresses.

Validate Solana address format where possible.

⸻

Balance Fetching

Fetch token accounts for each wallet and calculate current SWEENEE balance.

Important:

* Use the SWEENEE mint address only.
* Handle wallets with zero balance.
* Handle wallets with closed token accounts.
* Handle API errors gracefully.
* Use retries and rate-limit handling.
* Cache results with timestamps.

Output schema:

@dataclass
class TokenBalance:
    wallet_address: str
    token_mint: str
    raw_amount: int
    decimals: int
    ui_amount: float
    fetched_at: datetime

⸻

Transaction Fetching

Fetch recent transactions for each tracked wallet and identify SWEENEE token movements.

Requirements:

* Fetch signatures for each wallet.
* Fetch parsed transaction details.
* Extract token balance changes involving the SWEENEE mint.
* Associate movements with tracked wallet.
* Classify movement type where possible.
* Store transaction signature to prevent duplicates.
* Support configurable lookback windows, e.g. 24h, 7d, 30d.

Output schema:

@dataclass
class SweeneeTransaction:
    signature: str
    block_time: datetime | None
    wallet_address: str
    token_mint: str
    amount_change: float
    direction: str  # in, out, neutral, unknown
    classification: str  # buy, sell, transfer_in, transfer_out, unknown
    counterparty: str | None
    explorer_url: str
    raw: dict | None = None

⸻

Caching / Persistence

Use SQLite initially unless the existing SHI project already has a preferred local database layer.

Create tables for:

tracked_wallets
wallet_balances
sweenee_transactions
dashboard_runs

The dashboard should not hammer APIs on every UI refresh. Use cached data unless manually refreshed or stale.

Suggested cache policy:

* Balances: refresh every 1 to 5 minutes
* Transactions: refresh every 1 to 5 minutes
* Historical summaries: append on each successful refresh

Make refresh interval configurable.

⸻

Analytical Metrics

Implement the following metrics:

Total tracked holdings

B_{\text{total}} = \sum_{i=1}^{N} B_i

Plain spoken form:

B total equals the sum from i equals one to N of B sub i.

Wallet share of tracked holdings

p_i = \frac{B_i}{B_{\text{total}}}

Plain spoken form:

p sub i equals B sub i divided by B total.

Net flow over window

F_i(\Delta t) = \sum_{\Delta t} I_i - \sum_{\Delta t} O_i

Plain spoken form:

F sub i over time window delta t equals the sum of inflows into wallet i over delta t minus the sum of outflows from wallet i over delta t.

Concentration among tracked wallets

Implement Herfindahl-Hirschman Index:

HHI = \sum_{i=1}^{N} p_i^2

Plain spoken form:

H H I equals the sum from i equals one to N of p sub i squared.

This should be labelled carefully:

Concentration among tracked wallets only.

Do not imply this is the full-holder concentration unless full-holder data is available.

⸻

UX / Marketing Guidance

The audience is a Telegram token community. The dashboard must be:

* Clear
* Fast
* Screenshot-friendly
* Mobile-readable
* Honest about uncertainty
* Not overly technical on the surface
* Able to show detail when needed

Tone:

* Transparent
* Community-facing
* No hype
* No price predictions
* No “guaranteed bullish/bearish” wording
* No financial advice

Suggested section titles:

🐳 Whale Wallet Snapshot
📊 Current SWEENEE Holdings
🔁 Recent Whale Movements
📈 Net Flow
🧭 Wallet Details
📣 Telegram Summary

Use emoji lightly. The dashboard should look credible, not like a casino flyer wearing sunglasses.

⸻

Reliability Requirements

The system must handle:

* API rate limits
* Missing token accounts
* Empty wallet list
* Invalid wallet addresses
* Failed transaction fetches
* Duplicate transactions
* Unknown transaction types
* Large wallets with many transactions
* Token decimals correctly
* Timezones consistently

All errors should be logged. The dashboard should show friendly messages rather than crashing.

⸻

Configuration

Create a config file or .env support for:

SWEENEE_MINT=FkAtYamtEMtgnsTeUhzhTCiT2Svyxw63UdUYp1T7pump
WALLETS_DIR=/Users/q/PythonScript/Python/Vibe/SHI/sweenee/wallets
DATABASE_PATH=/Users/q/PythonScript/Python/Vibe/SHI/sweenee/data/sweenee.sqlite
REFRESH_INTERVAL_SECONDS=300
SOLANA_RPC_URL=
HELIUS_API_KEY=
BIRDEYE_API_KEY=
SOLSCAN_API_KEY=

Use whatever API keys already exist in the wider SHI environment where possible. Do not duplicate secrets into code.

⸻

Testing Requirements

Create tests for:

1. Loading wallets from txt
2. Loading wallets from csv
3. Loading wallets from json
4. Deduplicating wallets
5. Handling invalid wallet entries
6. Balance aggregation
7. Net flow calculation
8. HHI calculation
9. Transaction classification from mocked transaction data
10. Telegram summary generation

Use mocked API responses for tests. Do not require live API calls in unit tests.

⸻

README Requirements

Create a README.md explaining:

* What the dashboard does
* What it does not do
* How to add wallets
* How to run locally
* How to refresh data
* How to generate Telegram summaries
* What each metric means
* Known limitations
* Future improvements

Include commands such as:

cd /Users/q/PythonScript/Python/Vibe/SHI/sweenee
streamlit run app.py

or equivalent, depending on the chosen framework.

⸻

Implementation Priorities

Build in this order:

Phase 1: Local MVP

* Create project structure
* Load wallet addresses
* Fetch current SWEENEE balances
* Show balance table
* Show total tracked holdings
* Add manual refresh button

Phase 2: Transaction Feed

* Fetch recent transactions
* Parse SWEENEE token balance changes
* Classify transaction types conservatively
* Add transaction table
* Add explorer links

Phase 3: Metrics and Charts

* Add top wallet chart
* Add net flow chart
* Add transaction activity chart
* Add HHI / concentration metric

Phase 4: Telegram Sharing

* Add Telegram summary generator
* Add copy-to-clipboard summary box if possible
* Add clean screenshot-friendly dashboard layout

Phase 5: Deployment Prep

* Add caching/persistence
* Add README
* Add tests
* Add .env.example
* Make it deployable locally or to a lightweight cloud host

⸻

Quality Bar

The final implementation should be:

* Runnable
* Modular
* Well-documented
* Safe with API keys
* Resistant to API failures
* Clear enough for a non-technical token holder
* Technically honest enough for serious analysis

Before finishing, provide:

1. A summary of files created.
2. Setup/run instructions.
3. Any assumptions made.
4. Any limitations.
5. Suggested next improvements.

Do not overstate what the dashboard proves. It tracks selected configured wallets only.

A useful extra instruction to add to the agent, because this is community-facing:
```md
Important: keep all analytics descriptive, not predictive. The dashboard may say “tracked whale wallets had net inflow of X SWEENEE over 24h,” but must not say “this means price will rise” or similar. We are building transparency infrastructure, not a crystal ball with a token account.