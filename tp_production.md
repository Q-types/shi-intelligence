# SHI Production Deployment Checklist

## Overview
This document outlines all steps required to deploy SHI (Solana Holder Intelligence) to production. Tasks are categorized by what can be automated vs. what requires manual action.

---

## 1. Environment Configuration

### 1.1 Create Environment File
- [ ] **Create `.env` file in project root**
  ```bash
  cp .env.example .env
  ```

### 1.2 Required Environment Variables
| Variable | Description | Where to Get |
|----------|-------------|--------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot API token | [@BotFather](https://t.me/BotFather) on Telegram |
| `HELIUS_API_KEY` | Helius RPC API key | [helius.dev](https://www.helius.dev/) |
| `SOLANA_RPC_URL` | Solana RPC endpoint | [QuickNode](https://quicknode.com), [Alchemy](https://alchemy.com), or public RPC |
| `DATABASE_URL` | PostgreSQL connection string | Your database provider |
| `REDIS_URL` | Redis connection string | [Redis Cloud](https://redis.com/cloud/), [Upstash](https://upstash.com/), or self-hosted |
| `ADMIN_USER_IDS` | Comma-separated Telegram user IDs for admins | Your Telegram user ID |

### 1.3 Manual Steps Required
- [ ] **Go to Telegram:**
  1. Open Telegram, search for `@BotFather`
  2. Send `/newbot` and follow prompts
  3. Copy the API token provided
  4. Send `/setcommands` and paste:
     ```
     analyze - Full token analysis
     summary - Quick overview
     top_holders - Holder breakdown
     risk - Risk scores only
     help - Show help
     ```

- [ ] **Go to Helius:**
  1. Visit [helius.dev](https://www.helius.dev/)
  2. Create account and verify email
  3. Create new project
  4. Copy API key from dashboard
  5. Note: Free tier = 100k requests/month

- [ ] **Database Setup (choose one):**
  - **Option A: Supabase (recommended)**
    1. Go to [supabase.com](https://supabase.com)
    2. Create new project
    3. Go to Settings → Database → Connection string
    4. Copy the URI (use "Transaction pooler" for serverless)

  - **Option B: Local PostgreSQL**
    ```bash
    # macOS
    brew install postgresql@15
    brew services start postgresql@15
    createdb shi_production
    ```

- [ ] **Redis Setup (choose one):**
  - **Option A: Upstash (serverless)**
    1. Go to [upstash.com](https://upstash.com)
    2. Create Redis database
    3. Copy the Redis URL

  - **Option B: Local Redis**
    ```bash
    # macOS
    brew install redis
    brew services start redis
    # URL: redis://localhost:6379
    ```

---

## 2. Database Migrations

### 2.1 Automated Steps
```bash
# Install dependencies
cd /Users/q/PythonScript/Python/Vibe/SHI
pip install -e .

# Run migrations
alembic upgrade head
```

### 2.2 Verify Migration
```bash
# Check current revision
alembic current

# List all tables (PostgreSQL)
psql $DATABASE_URL -c "\dt"
```

### 2.3 Expected Tables
- [ ] `tokens` - Token metadata
- [ ] `wallets` - Wallet information
- [ ] `funding_edges` - Funding graph edges
- [ ] `holder_snapshots` - Point-in-time holder data
- [ ] `balances` - Wallet balances
- [ ] `wallet_features` - Computed features
- [ ] `archetype_assignments` - Clustering results
- [ ] `metrics` - Computed metrics with Z-scores
- [ ] `baseline_datasets` - Training baselines
- [ ] `hazard_models` - Trained Cox PH models
- [ ] `audit_logs` - System audit trail

---

## 3. Telegram Bot Deployment

### 3.1 Local Testing
```bash
# Set environment
export TELEGRAM_BOT_TOKEN="your_token_here"

# Run bot
python -m shi.telegram.bot
```

### 3.2 Production Deployment Options

- [ ] **Option A: systemd service (Linux VPS)**
  ```bash
  # Create service file
  sudo nano /etc/systemd/system/shi-bot.service
  ```
  ```ini
  [Unit]
  Description=SHI Telegram Bot
  After=network.target

  [Service]
  Type=simple
  User=shi
  WorkingDirectory=/opt/shi
  Environment=PYTHONUNBUFFERED=1
  EnvironmentFile=/opt/shi/.env
  ExecStart=/opt/shi/venv/bin/python -m shi.telegram.bot
  Restart=always
  RestartSec=10

  [Install]
  WantedBy=multi-user.target
  ```
  ```bash
  sudo systemctl enable shi-bot
  sudo systemctl start shi-bot
  ```

- [ ] **Option B: Docker**
  ```dockerfile
  # Dockerfile already exists - build and run
  docker build -t shi-bot .
  docker run -d --env-file .env --name shi shi-bot
  ```

- [ ] **Option C: Railway/Render (PaaS)**
  1. Push code to GitHub
  2. Connect repository to [Railway](https://railway.app) or [Render](https://render.com)
  3. Add environment variables in dashboard
  4. Deploy

---

## 4. Redis Cache Configuration

### 4.1 Verify Connection
```python
# Test Redis connection
import redis
r = redis.from_url("your_redis_url")
r.ping()  # Should return True
```

### 4.2 Cache TTL Configuration
Edit `src/infra/cache.py` if needed:
```python
ttl_by_type: dict[str, int] = {
    "holder_snapshot": 60,      # 1 minute
    "liquidity": 30,            # 30 seconds
    "analysis_result": 300,     # 5 minutes
    "token_metadata": 3600,     # 1 hour
    "wallet_features": 600,     # 10 minutes
}
```

### 4.3 Memory Limits
- [ ] Set Redis `maxmemory` policy:
  ```bash
  redis-cli CONFIG SET maxmemory 256mb
  redis-cli CONFIG SET maxmemory-policy allkeys-lru
  ```

---

## 5. Prometheus Monitoring Setup

### 5.1 Metrics Endpoint
The `/metrics` endpoint is exposed by the monitoring module.

### 5.2 Prometheus Configuration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'shi'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 15s
```

### 5.3 Key Metrics to Monitor
| Metric | Alert Threshold |
|--------|-----------------|
| `shi_http_request_duration_seconds_p99` | > 25s |
| `shi_analysis_total{status="error"}` | > 10% of total |
| `shi_brier_score` | > 0.25 |
| `shi_circuit_state{state="open"}` | Any occurrence |

### 5.4 Grafana Dashboard (Manual)
- [ ] Import dashboard from `monitoring/grafana-dashboard.json` (to be created)
- [ ] Set up alerts for SLA violations

---

## 6. Formal Validation Benchmark Protocol

### 6.1 Benchmark Dataset Requirements
- [ ] **Collect baseline dataset:**
  - Minimum 500 tokens with known outcomes
  - At least 50 "sell events" per archetype
  - 90 days of historical data minimum
  - Include both "rugged" and "stable" tokens

### 6.2 Validation Metrics (Per INITIAL_PROMPT)
| Metric | Threshold | Measurement |
|--------|-----------|-------------|
| Concordance Index | ≥ 0.55 | `fitter.concordance_index_` |
| Brier Score | ≤ 0.25 | `brier_score_loss(actual, predicted)` |
| ROC-AUC | ≥ 0.60 | `roc_auc_score(actual, predicted)` |
| PH Assumption p-value | ≥ 0.01 | `proportional_hazard_test()` |
| Calibration Slope | 0.7 - 1.3 | Linear regression on calibration curve |
| CV Score Std | ≤ 0.1 | 5-fold cross-validation |

### 6.3 Benchmark Execution Steps
```bash
# 1. Prepare benchmark dataset
python -m shi.calibration.prepare_benchmark \
  --output data/benchmark_v1.parquet \
  --min-tokens 500 \
  --min-events 250

# 2. Run validation suite
python -m shi.models.validation \
  --dataset data/benchmark_v1.parquet \
  --output reports/validation_report.json

# 3. Generate calibration curves
python -m shi.calibration.plot_curves \
  --input reports/validation_report.json \
  --output reports/calibration_plots/
```

### 6.4 Benchmark Schedule
- [ ] **Pre-deployment:** Full validation on historical data
- [ ] **Weekly:** Drift detection report
- [ ] **Monthly:** Full revalidation with new data
- [ ] **On regime change:** Triggered automatically

### 6.5 Manual Validation Steps
1. **Data Collection (You must do):**
   - [ ] Export token list from DexScreener/Birdeye for tokens > 30 days old
   - [ ] Label tokens as "rugged" (>80% drawdown) or "stable"
   - [ ] Record actual sell events from on-chain data

2. **Ground Truth Labeling (You must do):**
   - [ ] For each token, identify major sell events
   - [ ] Label wallet archetypes manually for 100 sample wallets
   - [ ] Compare against system classifications

3. **Expert Review (You must do):**
   - [ ] Review 20 random analysis reports
   - [ ] Score accuracy of risk assessments (1-5)
   - [ ] Document false positives/negatives

---

## 7. Pre-Flight Checklist

### 7.1 Before Going Live
- [ ] All environment variables set
- [ ] Database migrations complete
- [ ] Redis connection verified
- [ ] Telegram bot responds to `/start`
- [ ] At least one successful `/analyze` test
- [ ] Monitoring endpoints accessible
- [ ] Backup strategy in place
- [ ] Rate limits configured appropriately

### 7.2 Security Checklist
- [ ] `.env` file is in `.gitignore`
- [ ] Admin user IDs configured
- [ ] Rate limiting active
- [ ] Audit logging enabled
- [ ] No hardcoded secrets in code

### 7.3 Performance Checklist
- [ ] 30-second SLA verified with load test
- [ ] Circuit breakers configured for external APIs
- [ ] Cache hit rate > 50%
- [ ] Memory usage stable under load

---

## 8. Implementation I Cannot Do (Requires Your Action)

### 8.1 External Account Creation
| Service | URL | Action Required |
|---------|-----|-----------------|
| Telegram BotFather | t.me/BotFather | Create bot, get token |
| Helius | helius.dev | Create account, get API key |
| Database Host | supabase.com / railway.app | Create PostgreSQL instance |
| Redis Host | upstash.com / redis.com | Create Redis instance |
| VPS (if self-hosting) | digitalocean.com / hetzner.com | Provision server |

### 8.2 Data Collection
- [ ] Historical token data for training
- [ ] Known rug-pull examples for validation
- [ ] Ground truth labels for benchmarking

### 8.3 Domain-Specific Configuration
- [ ] Token whitelist (if restricting analysis)
- [ ] Premium user list (if monetizing)
- [ ] Custom alert thresholds based on your risk tolerance

### 8.4 Legal/Compliance
- [ ] Privacy policy for Telegram bot
- [ ] Terms of service
- [ ] Disclaimer that outputs are not financial advice

---

## 9. Quick Start Commands

```bash
# 1. Clone and install
cd /Users/q/PythonScript/Python/Vibe/SHI
pip install -e .

# 2. Set up environment
cp .env.example .env
# Edit .env with your values

# 3. Run migrations
alembic upgrade head

# 4. Test locally
python -c "from shi.core.config import settings; print(settings)"

# 5. Start bot
python -m shi.telegram.bot

# 6. Run tests
pytest tests/ -v

# 7. Run validation benchmark
pytest tests/integration/ -v --benchmark
```

---

## 10. Support & Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `TELEGRAM_BOT_TOKEN not set` | Check `.env` file exists and is loaded |
| `Database connection refused` | Verify DATABASE_URL and network access |
| `Rate limit exceeded` | Reduce request frequency or upgrade API tier |
| `Analysis timeout` | Token may have too many holders; check logs |
| `Circuit breaker open` | External API down; wait for recovery |

### Log Locations
- Application logs: stdout (structured JSON)
- Audit logs: `audit_logs` database table
- Metrics: `/metrics` endpoint (Prometheus format)

---

## Document Info
- **Created:** 2024-02-24
- **Version:** 1.0
- **Status:** Production Ready
