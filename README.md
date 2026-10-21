# yfinance-ai

### World's First AI-Powered Yahoo Finance Integration

**68 financial data tools** for any AI assistant: real-time stock prices, crypto, forex, commodities, ETFs, fundamentals, dividends, analyst ratings, options, holders, calendars, news and everything yfinance offers, through plain-language questions.

Use it as an **MCP server** (Claude, Cursor, VS Code and any MCP client), an **OpenClaw skill** or an **OpenWebUI tool**.

[![PyPI](https://img.shields.io/pypi/v/yfinance-ai)](https://pypi.org/project/yfinance-ai/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

https://github.com/user-attachments/assets/f856e26a-4022-418a-98c7-f01e6947d885

Also try [sec-finance-ai](https://github.com/lkcair/sec-finance-ai) for SEC EDGAR filings. ⭐ Star the repo or the skill if you like it.

---

# 🚀 Thousands of downloads on ClawHub and OpenWebUI

# Quick Start

## 🔌 MCP server (Claude, Cursor, VS Code, any MCP client)

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/). Nothing to configure, no API key.

**Claude Code**
```bash
claude mcp add yfinance-ai -- uvx yfinance-ai
```

**Claude Desktop, Cursor and other clients** (`claude_desktop_config.json`, `.cursor/mcp.json`, ...)
```json
{
  "mcpServers": {
    "yfinance-ai": {
      "command": "uvx",
      "args": ["yfinance-ai"]
    }
  }
}
```

Without uv: `pip install yfinance-ai`, then use `"command": "yfinance-ai"` with no args.

All 68 tools are exposed, each marked read-only. Listed in the [MCP Registry](https://registry.modelcontextprotocol.io) as `io.github.lkcair/yfinance-ai`. MCP server source: [lkcair/yfinance-ai-mcp](https://github.com/lkcair/yfinance-ai-mcp).

## 🦞 OpenClaw (ClawHub): thousands of downloads

```bash
openclaw skills install @lkcair/stocks
```
Skill page: [clawhub.ai/lkcair/stocks](https://clawhub.ai/lkcair/stocks). You can also download the skill there and upload it to your OpenClaw skills folder.

## 🌐 OpenWebUI: thousands of downloads

1. Open yfinance-ai on the [OpenWebUI website](https://openwebui.com/posts/finance_data_pull_ai_tool_yahoo_finance_yfinance_fc928bb1) and click **Get**.
2. Enter your OpenWebUI address and click **Save** inside your instance.
3. **Enable** the tool on the model (Admin Panel) or in each new chat. This step is required.

Manual install: copy [yfinance_ai.py](https://github.com/lkcair/yfinance-ai/blob/main/yfinance_ai.py), go to **Workspace → Tools → +**, paste, **Save**, then enable it as above.

## 🐍 Python

```python
import asyncio
from yfinance_ai import Tools
print(asyncio.run(Tools().get_stock_quote("AAPL")))  # every tool is an async method
```

---

## ✨ Why yfinance-ai

✅ **The most complete yfinance integration**: 68 tools, from quotes to calendars, funds and options

✅ **One file, three platforms**: the same `yfinance_ai.py` runs as MCP server, OpenClaw skill and OpenWebUI tool

✅ **Every release checked against real data**: each tool is called and its numbers compared with independent sources

✅ **Multi-asset**: stocks, ETFs, mutual funds, crypto (BTC-USD), forex (EURUSD=X), commodities (GC=F)

✅ **Market-wide**: movers, sectors, industries, earnings, IPO, economic and split calendars

✅ **Built for AI**: readable answers with context, retries, validation and rate limiting

✅ **Self-test**: ask the AI to "run self-test on yfinance tools" (64 checks)

---

## 💬 Example prompts

- "What's Apple's stock price?" / "Compare AAPL, MSFT and GOOGL"
- "Show Tesla's income statement" / "Is Apple buying back shares?"
- "What are analysts saying about NVIDIA?" / "Tesla upgrades and downgrades"
- "Show insider transactions for Amazon" / "Get options chain for SPY"
- "Bitcoin price" / "EUR/USD rate" / "Gold price today"
- "SPY top holdings" / "QQQ sector allocation"
- "Top gainers today" / "Who reports earnings this week?" / "Economic calendar for this week"

---

## 🛠️ Available Tools (68)

### Quotes and history (6)
- get_stock_price - Current price and key metrics
- get_stock_quote - Detailed quote with bid/ask, day range and trading info
- get_fast_info - Lightweight quick quote
- get_historical_data - Historical prices for any period and interval
- get_historical_price_on_date - Price on a specific date
- download_multiple_tickers - Historical data for several tickers at once

### Company (4)
- get_company_info - Company profile and business description
- get_company_overview - Company overview with context for every metric
- get_company_officers - Officers and key management
- get_isin - ISIN for a ticker

### Financials (8)
- get_income_statement - Income statement
- get_balance_sheet - Balance sheet
- get_cash_flow - Cash flow statement
- get_ttm_financials - Trailing-twelve-month income and cash flow
- get_key_ratios - Key ratios and valuation metrics
- get_valuation_history - Market cap, EV, P/E, PEG, P/S, P/B, EV/EBITDA by quarter
- get_financial_summary - One-page financial summary
- get_shares_history - Shares outstanding over time (dilution or buybacks)

### Earnings and estimates (7)
- get_earnings_dates - Upcoming and past earnings dates with estimates
- get_earnings_history - Historical earnings, annual and quarterly
- get_earnings_calendar - Next earnings date and dividend dates
- get_analyst_estimates - EPS and revenue estimates
- get_growth_estimates - Analyst growth estimates
- get_eps_trend - How EPS estimates have changed
- get_eps_revisions - EPS revisions up and down

### Analysts (3)
- get_analyst_recommendations - Recommendations and ratings
- get_analyst_price_targets - Low, mean, median and high targets
- get_upgrades_downgrades - Rating changes with dates

### Holders and insiders (6)
- get_institutional_holders - Major institutional holders
- get_major_holders - Institutions, insiders and float summary
- get_mutualfund_holders - Mutual fund holders
- get_insider_transactions - Recent insider trades
- get_insider_purchases - Insider purchases only
- get_insider_roster_holders - Company insiders

### Dividends and actions (4)
- get_dividends - Dividend history and current yield
- get_stock_splits - Split history
- get_corporate_actions - Dividends and splits combined
- get_capital_gains - Capital gains distributions (funds and ETFs)

### Options (2)
- get_options_chain - Calls and puts
- get_options_expirations - Available expiration dates

### Market-wide (8)
- get_market_status - Are US markets open
- get_market_summary - Region overview: status, indices, futures, bonds and FX
- get_market_indices - Major market indices
- get_market_movers - Gainers, losers, most active, most shorted and more
- get_trending_tickers - Most active US tickers today
- get_sector_performance - Performance of major sectors
- get_sector_overview - Sector size, largest companies, industries and ETFs
- get_industry_overview - Largest, best-performing and fastest-growing companies in an industry

### Calendars (4)
- get_upcoming_earnings - Who reports in the next N days
- get_ipo_calendar - Upcoming IPOs
- get_economic_calendar - CPI, jobs, GDP, rate decisions with expected and prior values
- get_splits_calendar - Upcoming stock splits

### Funds and ETFs (3)
- get_fund_overview - ETF or mutual fund overview
- get_fund_holdings - Top holdings
- get_fund_sector_weights - Sector allocation

### Crypto, forex, commodities (3)
- get_crypto_price - Cryptocurrency price and market data
- get_forex_rate - Currency pair rate
- get_commodity_price - Commodity and futures prices

### Compare (4)
- compare_stocks - Key metrics across several stocks
- get_peer_comparison - A stock against its peers
- get_historical_comparison - Price performance of several stocks
- get_complete_analysis - Full analysis calling every relevant tool

### News and filings (2)
- get_stock_news - Recent news
- get_sec_filings - Recent SEC filings (10-K, 10-Q, 8-K, ...)

### Utility (4)
- search_ticker - Find symbols by company name
- validate_ticker - Check a symbol exists and trades
- get_api_status - API status and usage
- run_self_test - Self-test of all tools (64 checks)

---

## 📝 License and credits

MIT. Built on [yfinance](https://github.com/ranaroussi/yfinance) by ranaroussi, pandas, pydantic and the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk). Author: lucas0 ([lucas0.com](https://lucas0.com)).

⚠️ For educational and research purposes only. Data comes from Yahoo Finance and may be delayed. Trading and investing involve substantial risk of loss. Not affiliated with Yahoo.
