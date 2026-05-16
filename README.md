# yfinance-ai

### World's Best AI-Powered Yahoo Finance Integration

**Natural Language Financial Data** - Fetch real-time stock prices, crypto, forex, commodities, ETFs, fundamentals, dividends, analyst ratings, options, news, and everything yfinance offers via simple AI prompts. 56+ financial tools for seamless integration with OpenWebUI, Claude, ChatGPT, and any AI assistant.

**Author**: lucas0

**Github**: [Github](https://github.com/lkcair)

**Version**: 3.0.4

**License**: MIT

**Requirements**: yfinance>=0.2.66, pandas>=2.2.0, pydantic>=2.0.0, requests>=2.28.0

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
openclaw skills install stocks
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

## 🔌 Integration with Other AI Tools

- **Claude**: Add as skill or as MCP server
- **ChatGPT**: Import as action or skill
- **LangChain**: Use as Tool
- **Python**: `from yfinance_ai import Tools; tools = Tools()`
- **API**: Deploy as FastAPI/Flask endpoint

---

## ✨ Features

✅ **Monolithic Codebase:** single-file script that allows skill integration on OpenWebUI and OpenClaw

✅ **56+ financial data tools** covering every yfinance capability

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


### 🆕 New in v3.0.0
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

### Crypto, Forex & Commodities (New!)
- "Get Bitcoin price"
- "What's Ethereum trading at?"
- "EUR/USD exchange rate"
- "Gold price today"
- "Oil futures price"

### Fund/ETF Queries (New!)
- "SPY top holdings"
- "QQQ sector allocation"
- "VOO expense ratio"

### Analyst Data (New!)
- "NVIDIA analyst price targets"
- "Tesla upgrades and downgrades"
- "Apple EPS trend"

---

## 🧪 Testing

AI can self-test by asking:

```
"Run self-test on yfinance tools"
```

This will test all 56+ functions and report results with **100% pass rate**.

---

## 📊 Test Results (v3.0.4)

```
Total Tests: 53
Passed: 53 ✅
Failed: 0 ❌
Success Rate: 100.0%

All 56+ tools are fully operational!
```

### Test Categories (All 16 Passing):
- ✅ Stock Quotes & Prices (5/5)
- ✅ Company Information (2/2)
- ✅ Financial Statements (3/3)
- ✅ Key Ratios & Metrics (1/1)
- ✅ Dividends & Corporate Actions (4/4)
- ✅ Earnings & Estimates (7/7)
- ✅ Analyst Data (3/3)
- ✅ Institutional & Insider Data (6/6)
- ✅ Options & Derivatives (2/2)
- ✅ News & SEC Filings (2/2)
- ✅ Market Indices & Comparison (3/3)
- ✅ Fund/ETF Data (3/3)
- ✅ Crypto, Forex & Commodities (3/3)
- ✅ Analysis & Comparison (4/4)
- ✅ Bulk Operations (2/2)
- ✅ Utility Functions (3/3)

---

## 🛠️ Available Tools (56+)

### Stock Data (5 tools)
- get_stock_price - Get current stock price
- get_fast_info - Get fast access to key stock metrics
- get_stock_quote - Get detailed real-time quote
- get_historical_data - Get historical price data
- get_isin - Get ISIN identifier

### Company Info (2 tools)
- get_company_info - Get company profile and details
- get_company_officers - Get company officers and executives

### Financials (3 tools)
- get_income_statement - Get income statement
- get_balance_sheet - Get balance sheet
- get_cash_flow - Get cash flow statement

### Analysis (5 tools)
- get_key_ratios - Get key financial ratios
- get_analyst_recommendations - Get analyst recommendations
- get_analyst_estimates - Get analyst estimates
- get_growth_estimates - Get growth estimates
- get_earnings_dates - Get earnings calendar dates

### Ownership (6 tools)
- get_institutional_holders - Get institutional holders
- get_major_holders - Get major shareholders
- get_mutualfund_holders - Get mutual fund holders
- get_insider_transactions - Get insider transactions
- get_insider_purchases - Get insider purchases
- get_insider_roster_holders - Get insider roster

### Options (2 tools)
- get_options_chain - Get options chain data
- get_options_expirations - Get options expiration dates

### Market Data (3 tools)
- get_market_indices - Get major market indices
- compare_stocks - Compare multiple stocks
- get_sector_performance - Get sector performance data

### Dividends & Splits (4 tools)
- get_dividends - Get dividend history
- get_splits - Get stock split history
- get_capital_gains - Get capital gains distributions
- get_actions - Get all corporate actions

### Earnings History (4 tools)
- get_earnings - Get earnings history
- get_quarterly_earnings - Get quarterly earnings
- get_earnings_history - Get earnings history with estimates
- get_revenue_estimates - Get revenue estimates

### News & Filings (2 tools)
- get_stock_news - Get latest news articles
- get_sec_filings - Get SEC filings

### Bulk Operations (2 tools)
- get_multiple_quotes - Get quotes for multiple tickers
- compare_fundamentals - Compare fundamentals across stocks

### Utilities (3 tools)
- get_stock_info - Get comprehensive stock information
- get_calendar - Get upcoming events calendar
- validate_ticker - Validate ticker symbol

### Analyst Data (3 tools) 🆕
- get_analyst_price_targets - Get analyst price targets (low, mean, median, high)
- get_upgrades_downgrades - Get recent analyst upgrades/downgrades
- get_eps_trend - Get EPS trend data

### Earnings & Estimates (3 new tools) 🆕
- get_eps_revisions - Get EPS revision data
- get_earnings_calendar - Get earnings calendar and dates

### Fund/ETF Data (3 tools) 🆕
- get_fund_overview - Get ETF/mutual fund overview
- get_fund_holdings - Get top holdings for funds
- get_fund_sector_weights - Get sector allocation for funds

### Crypto, Forex & Commodities (3 tools) 🆕
- get_crypto_price - Get cryptocurrency prices (BTC, ETH, etc.)
- get_forex_rate - Get forex exchange rates (EUR/USD, etc.)
- get_commodity_price - Get commodity prices (gold, oil, etc.)

### Analysis & Comparison (4 tools) 🆕
- get_peer_comparison - Compare stock with sector peers
- get_financial_summary - Get comprehensive financial summary
- get_historical_comparison - Compare historical performance
- get_market_status - Check if markets are open

### Comprehensive Analysis (2 tools) 🆕
- get_company_overview - Beautiful, insightful company overview with context
- get_complete_analysis - Run ALL functions for complete analysis (AI provides peers)

### Testing (1 tool)
- run_self_test - Run comprehensive self-test

---

## 🎯 What Makes This Special

🌟 **World's Best** - Most comprehensive AI-powered Yahoo Finance integration
📊 **Complete Coverage** - 56+ tools covering stocks, crypto, forex, commodities, ETFs
🚀 **Production Ready** - Full error handling, retry logic, validation, testing
📄 **Single File** - Everything in one ~150 KB file
🔌 **Multi-Platform** - Works with OpenWebUI, Claude, ChatGPT, LangChain
📚 **Well Documented** - Comprehensive guides and examples
✅ **100% Tested** - All 55 tests pass comprehensive self-test

---

## 📞 Support & Resources

**File Location**: `/yfinance_ai.py`
**Documentation**: `/README.md`
**Built on**: yfinance, pandas, pydantic, requests
**Status**: ✅ Production Ready (v3.0.4)

---

## 🏆 What's New in v3.0.0

✅ **15 new tools** - Crypto, forex, commodities, funds, analyst data, analysis
✅ **Fixed yfinance 0.2.66+ compatibility** - Works with latest yfinance
✅ **Added retry logic** - Exponential backoff for network resilience
✅ **Improved news function** - Multiple fallback parsing strategies
✅ **Natural language support** - "gold" → GC=F, "bitcoin" → BTC-USD
✅ **100% test pass rate** - All 54 tests passing

---

## 📝 License

MIT License - Free to use, modify, and distribute

---

## 🙏 Credits

**Built with**:
- [yfinance](https://github.com/ranaroussi/yfinance) by ranaroussi
- pandas, pydantic
- OpenWebUI function framework

**Author**: lucas0 (https://lucas0.com)
**Project**: InvestAI
**Made with** ❤️ **for the AI community**

⚠️ DISCLAIMER: This software is for educational and research purposes only. Trading and investing involves substantial risk of loss. Use at your own risk.

---

**⭐ Star the repo if you find it useful!**
[Repo on Github](https://github.com/lkcair/yfinance-ai)

---

*Version: 3.0.4*
*Status: Production Ready ✅*
*Test Pass Rate: 100% (55/55) ✅*
