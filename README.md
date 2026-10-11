# yfinance-ai

### World's First AI-Powered Yahoo Finance Integration

**Natural Language Financial Data** - Fetch real-time stock prices, crypto, forex, commodities, ETFs, fundamentals, dividends, analyst ratings, options, news, and everything yfinance offers via simple AI prompts. 68 financial tools for OpenWebUI, OpenClaw, and any AI assistant with function calling.

**Author**: lucas0

**Github**: [Github](https://github.com/lkcair)

**License**: MIT

**Requirements**: yfinance>=1.7.0, pandas>=2.2.0, pydantic>=2.0.0, requests>=2.28.0

**Repository**: https://github.com/lkcair/yfinance-ai

**Author URL**: https://lucas0.com

**OpenWebUI Page**: Download yfinance on the OpenWebUI Website: https://openwebui.com/posts/finance_data_pull_ai_tool_yahoo_finance_yfinance_fc928bb1

**OpenClaw Page**: https://clawhub.ai/lkcair/stocks

---



https://github.com/user-attachments/assets/f856e26a-4022-418a-98c7-f01e6947d885



# 🚀 3k+ Downloads
# Star the Github or the Skill if you like it.
**Repository**: https://github.com/lkcair/yfinance-ai

Also, try [SEC-ai](https://github.com/lkcair/sec-finance-ai) for SEC data.

# Quick Start
## 🦞 (Main) Integration with OpenClaw 
```bash
openclaw skills install @lkcair/stocks
```
On OpenClaw, start asking questions like:
   - "What's Apple's stock price?"
   - "Compare AAPL vs MSFT"

Skill page: [Openclaw Stocks](https://clawhub.ai/lkcair/stocks)

- **(Alternative) OpenClaw Manual**: Download Skill at [https://clawhub.ai/lkcair/stocks](https://clawhub.ai/lkcair/stocks) and upload manually to OpenClaw skills location.

- **(Obsolete versions) Clawhub CLI**: Add as Skill with clawhub command:
    ```bash
  # To install with npm
  npx clawhub@latest install stocks
    
  # To install with pnpm:
  pnpm dlx clawhub@latest install stocks

  # To install with bun:
  bunx clawhub@latest install stocks    
    ```

## 🚀 OpenWebUI (Main) Automatic Installation
1. Download yfinance-ai on the [OpenWebUI Website](https://openwebui.com/posts/finance_data_pull_ai_tool_yahoo_finance_yfinance_fc928bb1)
2. Click on **Get**
3. Write your OpenWebUI address for import to work.
4. Click on **Save** inside your OpenWebUI installation.
5. **Enable** on Model or on Chat (VERY IMPORTANT - Either enable the yfinance TOOL on the model itself via Admin Panel; or enable everytime on every new chat).
6. Start asking questions like:
   - "What's Apple's stock price?"
   - "Compare AAPL vs MSFT"

## 🚀 OpenWebUI (Alternative) Manual Installation

1. Copy the entire - [yfinance_ai.py](https://github.com/lkcair/yfinance-ai/blob/main/yfinance_ai.py) file
2. Go to OpenWebUI → **Workspace** → **Tools**
3. Click **"+"** to create new TOOL
4. Paste this code
5. Click on **Save**
6. **Enable** on Model or on Chat (VERY IMPORTANT - Either enable the yfinance TOOL on the model itself via Admin Panel; or enable everytime on every new chat).
7. Start asking questions like:
   - "What's Apple's stock price?"
   - "Compare AAPL vs MSFT"

---

## 🔌 Other Ways to Use It

- **OpenClaw**: `openclaw skills install @lkcair/stocks`
- **Python**: `from yfinance_ai import Tools; tools = Tools()` (every tool is an async method)

---

## ✨ Features

✅ **Monolithic Codebase:** single-file script that allows skill integration on OpenWebUI and OpenClaw

✅ **68 financial data tools** built on yfinance 1.7

✅ **Real-time stock prices** and quotes

✅ **Historical data** with customizable periods/intervals

✅ **Financial statements** (income, balance sheet, cash flow)

✅ **Key ratios** and valuation metrics

✅ **Dividends, splits**, and corporate actions

✅ **Analyst recommendations** and price targets

✅ **Institutional and insider** holdings

✅ **Options chains** and derivatives data

✅ **Company news** and SEC filings

✅ **Market indices** and sector performance

✅ **Bulk operations** and stock comparison

✅ **Built-in testing** function for validation

✅ **Comprehensive error handling**

✅ **Rate limiting** protection

✅ **Natural language** query support


### 🆕 Market-wide, valuation and estimates

✅ **Market movers** - Day gainers, losers, most active, most shorted, undervalued and more

✅ **Market summary** - Open/closed status (holiday-aware) and headline indices for US, Europe, Asia

✅ **Sectors and industries** - Largest companies, top performers, growth leaders, sector ETFs

✅ **Calendars** - Market-wide earnings, IPOs, economic events (CPI, jobs, rates) and splits

✅ **Valuation history** - P/E, PEG, P/S, P/B, EV/EBITDA by quarter

✅ **TTM financials** - Trailing-twelve-month revenue, profit, EBITDA and cash flow

✅ **Shares history** - Buybacks and dilution over time

✅ **Real analyst estimates** - EPS and revenue (avg, low, high, growth, analyst count) for quarters and years

✅ **Ticker search** - Find symbols by company name

✅ **Fixed percentages** - Margins, ROE and growth now display correctly

### 🌍 Multi-asset, funds and analyst data
✅ **Cryptocurrency prices** - BTC, ETH, any crypto (BTC-USD format)

✅ **Forex rates** - EUR/USD, GBP/USD, any currency pair

✅ **Commodities** - Gold, Oil, Silver, Natural Gas futures

✅ **Fund/ETF data** - Holdings, sector weights, expense ratios

✅ **Analyst price targets** - Low, mean, median, high targets

✅ **Upgrades/Downgrades** - Analyst rating changes with dates

✅ **EPS trends & revisions** - Earnings estimate changes

✅ **Peer comparison** - Compare stocks in same sector

✅ **Financial summary** - One-call comprehensive overview

✅ **Historical comparison** - Multi-ticker performance analysis

✅ **Market status** - Check if markets are open

---

## 💬 Example Prompts for AI

### Basic Queries
- "What's the current price of Apple stock?"
- "Show me Tesla's income statement"
- "Compare valuations of AAPL, MSFT, and GOOGL"

### Financial Analysis
- "Get dividend history for Coca-Cola"
- "What are analysts saying about NVIDIA?"
- "Show insider transactions for Amazon"

### Advanced Queries
- "Get options chain for SPY"
- "What's the latest news on Bitcoin?"
- "Compare tech sector vs healthcare sector"
- "Show me the major market indices"

### Crypto, Forex & Commodities
- "Get Bitcoin price"
- "What's Ethereum trading at?"
- "EUR/USD exchange rate"
- "Gold price today"
- "Oil futures price"

### Fund/ETF Queries
- "SPY top holdings"
- "QQQ sector allocation"
- "VOO expense ratio"

### Market-Wide
- "Top gainers today"
- "Who reports earnings this week?"
- "Upcoming IPOs"
- "Economic calendar for this week"
- "Semiconductor industry top performers"
- "Is Apple buying back shares?"

### Analyst Data
- "NVIDIA analyst price targets"
- "Tesla upgrades and downgrades"
- "Apple EPS trend"

---

## 🧪 Testing

AI can self-test by asking:

```
"Run self-test on yfinance tools"
```

This runs 64 checks across all 68 functions and reports results with **100% pass rate**.

---

## 🛠️ Available Tools (68)


### Quotes and history (6 tools)
- get_stock_price - Get current stock price and key metrics for a ticker symbol
- get_stock_quote - Get detailed quote with bid/ask, day range, and trading info
- get_fast_info - Get fast-access ticker information (lightweight, quick response)
- get_historical_data - Get historical price data for a stock
- get_historical_price_on_date - Get historical price for a specific date
- download_multiple_tickers - Download historical data for multiple tickers at once (bulk operation)

### Company (4 tools)
- get_company_info - Get detailed company information and business description
- get_company_overview - Get a beautiful, insightful company overview with context for every metric
- get_company_officers - Get company officers and key management
- get_isin - Get International Securities Identification Number (ISIN) for a ticker

### Financials (8 tools)
- get_income_statement - Get income statement data
- get_balance_sheet - Get balance sheet data
- get_cash_flow - Get cash flow statement data
- get_ttm_financials - Get trailing-twelve-month (TTM) income statement and cash flow
- get_key_ratios - Get key financial ratios and valuation metrics
- get_valuation_history - Get valuation measures now and for the last quarters (market cap, EV, P/E, PEG, P/S, P/B, EV/EBITDA)
- get_financial_summary - Get a comprehensive one-page financial summary of a stock
- get_shares_history - Get shares outstanding over time (dilution or buybacks)

### Earnings and estimates (7 tools)
- get_earnings_dates - Get upcoming and past earnings dates with estimates
- get_earnings_history - Get historical earnings data with annual and quarterly figures
- get_earnings_calendar - Get earnings calendar including next earnings date and dividend dates
- get_analyst_estimates - Get analyst estimates for earnings, revenue, and EPS
- get_growth_estimates - Get analyst growth estimates
- get_eps_trend - Get EPS (Earnings Per Share) trend data showing how estimates have changed
- get_eps_revisions - Get EPS revision data showing analyst estimate changes

### Analysts (3 tools)
- get_analyst_recommendations - Get analyst recommendations and ratings
- get_analyst_price_targets - Get detailed analyst price targets for a stock
- get_upgrades_downgrades - Get detailed analyst upgrades and downgrades history

### Holders and insiders (6 tools)
- get_institutional_holders - Get major institutional holders and their ownership
- get_major_holders - Get major holders summary (institutions, insiders, public float)
- get_mutualfund_holders - Get mutual fund holders
- get_insider_transactions - Get recent insider trading activity
- get_insider_purchases - Get insider purchases only (filtered for buys)
- get_insider_roster_holders - Get insider roster (list of company insiders)

### Dividends and actions (4 tools)
- get_dividends - Get dividend payment history and current metrics
- get_stock_splits - Get stock split history
- get_corporate_actions - Get all corporate actions (dividends + splits combined)
- get_capital_gains - Get capital gains distributions (primarily for funds/ETFs)

### Options (2 tools)
- get_options_chain - Get options chain data (calls and puts)
- get_options_expirations - Get all available options expiration dates

### Market-wide (8 tools)
- get_market_status - Check if US stock markets are currently open
- get_market_summary - Get a market overview for a region: open/closed status plus main indices, futures, bonds and FX
- get_market_indices - Get current prices for major market indices
- get_market_movers - Get today's market movers from Yahoo Finance screeners
- get_trending_tickers - Get today's most active (trending) US tickers
- get_sector_performance - Get performance of major market sectors
- get_sector_overview - Get a sector overview: size, weight, largest companies, industries and sector ETFs
- get_industry_overview - Get an industry overview: largest, best-performing and fastest-growing companies

### Calendars (4 tools)
- get_upcoming_earnings - Get the market-wide earnings calendar (which companies report in the next N days)
- get_ipo_calendar - Get upcoming IPOs (exchange, expected date, price range, shares)
- get_economic_calendar - Get upcoming economic events (CPI, jobs, GDP, rate decisions) with expected and prior values
- get_splits_calendar - Get upcoming stock splits

### Funds and ETFs (3 tools)
- get_fund_overview - Get overview data for an ETF or mutual fund
- get_fund_holdings - Get top holdings for an ETF or mutual fund
- get_fund_sector_weights - Get sector allocation/weights for an ETF or mutual fund

### Crypto, forex, commodities (3 tools)
- get_crypto_price - Get cryptocurrency price and market data
- get_forex_rate - Get foreign exchange (forex) rate for a currency pair
- get_commodity_price - Get commodity/futures price data

### Compare (4 tools)
- compare_stocks - Compare key metrics across multiple stocks
- get_peer_comparison - Compare a stock with its peers on key metrics
- get_historical_comparison - Compare historical price performance of multiple stocks
- get_complete_analysis - Run a complete analysis of a stock by calling ALL available functions

### News and filings (2 tools)
- get_stock_news - Get recent news articles about a stock
- get_sec_filings - Get recent SEC filings (10-K, 10-Q, 8-K, etc.)

### Utility (4 tools)
- search_ticker - Search for ticker symbols by company name or keyword
- validate_ticker - Validate if a ticker symbol exists and is tradeable
- get_api_status - Get API status and usage information
- run_self_test - Run comprehensive self-test of the yfinance-ai tools (64 checks)

---


## 🎯 What Makes This Special

🌟 **World's First** - The first AI-powered Yahoo Finance integration, and the most complete one
📊 **Complete Coverage** - 68 tools covering stocks, crypto, forex, commodities, ETFs
🚀 **Production Ready** - Full error handling, retry logic, validation, testing
📄 **Single File** - Everything in one ~230 KB file
🔌 **Multi-Platform** - Works with OpenWebUI, OpenClaw and plain Python
📚 **Well Documented** - Comprehensive guides and examples
✅ **100% Tested** - All 64 self-test checks pass

---

## 📞 Support & Resources

**File Location**: `/yfinance_ai.py`
**Documentation**: `/README.md`
**Built on**: yfinance, pandas, pydantic, requests
**Status**: ✅ Production Ready

---

## 📝 License

MIT License - Free to use, modify, and distribute

---

## 🙏 Credits

**Built with**:
- [yfinance](https://github.com/ranaroussi/yfinance) by ranaroussi
- pandas, pydantic
- OpenWebUI tool framework

**Author**: lucas0 (https://lucas0.com)
**Project**: InvestAI
**Made with** ❤️ **for the AI community**

⚠️ DISCLAIMER: This software is for educational and research purposes only. Trading and investing involves substantial risk of loss. Use at your own risk.

---

**⭐ Star the repo if you find it useful!**
[Repo on Github](https://github.com/lkcair/yfinance-ai)

---

*Status: Production Ready ✅*
*Test Pass Rate: 100% (64/64) ✅*
