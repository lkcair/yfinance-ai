# yfinance-ai

World's first AI-powered Yahoo Finance integration: 68 tools for stocks, ETFs, mutual funds, crypto, forex and commodities, in one Python file.

🚀 **Thousands of downloads on ClawHub and OpenWebUI**

Prices and history, financial statements, ratios and valuation history, analyst estimates and price targets, earnings, dividends, options, holders and insiders, funds and ETFs, market movers, sectors and industries, and earnings, IPO, economic and split calendars.

| Use it in | Install | Repo |
|---|---|---|
| Claude, Cursor, VS Code, [LangChain](https://docs.langchain.com/oss/python/langchain/mcp), [Google ADK](https://google.github.io/adk-docs/tools-custom/mcp-tools/), [Strands Agents](https://strandsagents.com/docs/user-guide/sdk/tools/mcp-tools/) and other MCP clients | `uvx yfinance-ai@latest` | [yfinance-ai-mcp](https://github.com/lkcair/yfinance-ai-mcp) |
| ClawHub skill for OpenClaw 🚀 Thousands of downloads | `openclaw skills install @lkcair/stocks` | [yfinance-ai-skill](https://github.com/lkcair/yfinance-ai-skill) |
| OpenWebUI tool 🚀 Thousands of downloads | [Get it on openwebui.com](https://openwebui.com/posts/finance_data_pull_ai_tool_yahoo_finance_yfinance_fc928bb1) | [yfinance-ai](https://github.com/lkcair/yfinance-ai) |
| Python | `from yfinance_ai import Tools` | [yfinance-ai](https://github.com/lkcair/yfinance-ai) |


https://github.com/user-attachments/assets/f856e26a-4022-418a-98c7-f01e6947d885

## Install

**MCP**
```bash
claude mcp add yfinance-ai -- uvx yfinance-ai@latest
```
Setup for VS Code, Claude Desktop, Cursor and other clients, plus LangChain, Google ADK and Strands Agents examples: [yfinance-ai-mcp](https://github.com/lkcair/yfinance-ai-mcp).

**ClawHub (OpenClaw)**
```bash
openclaw skills install @lkcair/stocks
```
Skill page: [clawhub.ai/lkcair/stocks](https://clawhub.ai/lkcair/stocks).

**OpenWebUI**
1. Open the [OpenWebUI page](https://openwebui.com/posts/finance_data_pull_ai_tool_yahoo_finance_yfinance_fc928bb1), click **Get** and save it in your OpenWebUI.
2. Enable the tool on the model or in the chat.

Or copy [`yfinance_ai.py`](https://github.com/lkcair/yfinance-ai/blob/main/yfinance_ai.py) into **Workspace > Tools > +** and save.

**Python**
```python
import asyncio
from yfinance_ai import Tools
print(asyncio.run(Tools().get_stock_quote("AAPL")))
```

## Tools (68)

**Quotes and history**
- `get_stock_price`: Get current stock price and key metrics for a ticker symbol
- `get_stock_quote`: Get detailed quote with bid/ask, day range, and trading info
- `get_fast_info`: Get fast-access ticker information (lightweight, quick response)
- `get_historical_data`: Get historical price data for a stock
- `get_historical_price_on_date`: Get historical price for a specific date
- `download_multiple_tickers`: Historical data for several tickers at once

**Company**
- `get_company_info`: Get detailed company information and business description
- `get_company_overview`: Company overview with context for every metric
- `get_company_officers`: Get company officers and key management
- `get_isin`: Get International Securities Identification Number (ISIN) for a ticker

**Financials**
- `get_income_statement`: Get income statement data
- `get_balance_sheet`: Get balance sheet data
- `get_cash_flow`: Get cash flow statement data
- `get_ttm_financials`: Get trailing-twelve-month (TTM) income statement and cash flow
- `get_key_ratios`: Get key financial ratios and valuation metrics
- `get_valuation_history`: Get valuation measures now and for the last quarters (market cap, EV, P/E, PEG, P/S, P/B, EV/EBITDA)
- `get_financial_summary`: Get a comprehensive one-page financial summary of a stock
- `get_shares_history`: Get shares outstanding over time (dilution or buybacks)

**Earnings and estimates**
- `get_earnings_dates`: Get upcoming and past earnings dates with estimates
- `get_earnings_history`: Get historical earnings data with annual and quarterly figures
- `get_earnings_calendar`: Get earnings calendar including next earnings date and dividend dates
- `get_analyst_estimates`: Get analyst estimates for earnings, revenue, and EPS
- `get_growth_estimates`: Get analyst growth estimates
- `get_eps_trend`: Get EPS (Earnings Per Share) trend data showing how estimates have changed
- `get_eps_revisions`: Get EPS revision data showing analyst estimate changes

**Analysts**
- `get_analyst_recommendations`: Get analyst recommendations and ratings
- `get_analyst_price_targets`: Get detailed analyst price targets for a stock
- `get_upgrades_downgrades`: Get detailed analyst upgrades and downgrades history

**Holders and insiders**
- `get_institutional_holders`: Get major institutional holders and their ownership
- `get_major_holders`: Get major holders summary (institutions, insiders, public float)
- `get_mutualfund_holders`: Get mutual fund holders
- `get_insider_transactions`: Get recent insider trading activity
- `get_insider_purchases`: Get insider purchases only (filtered for buys)
- `get_insider_roster_holders`: Get insider roster (list of company insiders)

**Dividends and corporate actions**
- `get_dividends`: Get dividend payment history and current metrics
- `get_stock_splits`: Get stock split history
- `get_corporate_actions`: Get all corporate actions (dividends + splits combined)
- `get_capital_gains`: Get capital gains distributions (primarily for funds/ETFs)

**Options**
- `get_options_chain`: Get options chain data (calls and puts)
- `get_options_expirations`: Get all available options expiration dates

**Market-wide**
- `get_market_status`: Check if US stock markets are currently open
- `get_market_summary`: Get a market overview for a region: open/closed status plus main indices, futures, bonds and FX
- `get_market_indices`: Get current prices for major market indices
- `get_market_movers`: Get today's market movers from Yahoo Finance screeners
- `get_trending_tickers`: Get today's most active (trending) US tickers
- `get_sector_performance`: Get performance of major market sectors
- `get_sector_overview`: Get a sector overview: size, weight, largest companies, industries and sector ETFs
- `get_industry_overview`: Get an industry overview: largest, best-performing and fastest-growing companies

**Calendars**
- `get_upcoming_earnings`: Get the market-wide earnings calendar (which companies report in the next N days)
- `get_ipo_calendar`: Get upcoming IPOs (exchange, expected date, price range, shares)
- `get_economic_calendar`: Get upcoming economic events (CPI, jobs, GDP, rate decisions) with expected and prior values
- `get_splits_calendar`: Get upcoming stock splits

**Funds and ETFs**
- `get_fund_overview`: Get overview data for an ETF or mutual fund
- `get_fund_holdings`: Get top holdings for an ETF or mutual fund
- `get_fund_sector_weights`: Get sector allocation/weights for an ETF or mutual fund

**Crypto, forex and commodities**
- `get_crypto_price`: Get cryptocurrency price and market data
- `get_forex_rate`: Get foreign exchange (forex) rate for a currency pair
- `get_commodity_price`: Get commodity/futures price data

**Compare**
- `compare_stocks`: Compare key metrics across multiple stocks
- `get_peer_comparison`: Compare a stock with its peers on key metrics
- `get_historical_comparison`: Compare historical price performance of multiple stocks
- `get_complete_analysis`: Complete analysis of a stock, calling every relevant tool

**News and filings**
- `get_stock_news`: Get recent news articles about a stock
- `get_sec_filings`: Get recent SEC filings (10-K, 10-Q, 8-K, etc.)

**Utility**
- `search_ticker`: Search for ticker symbols by company name or keyword
- `validate_ticker`: Validate if a ticker symbol exists and is tradeable
- `get_api_status`: Get API status and usage information
- `run_self_test`: Run comprehensive self-test of the yfinance-ai tools (64 checks)

## Example questions

- What is Apple's P/E, and how does it compare with Microsoft and Google?
- What are analysts saying about NVIDIA? Price targets and recent upgrades.
- Who reports earnings this week?
- Bitcoin price, EUR/USD rate and gold today.

See also [sec-finance-ai](https://github.com/lkcair/sec-finance-ai) for SEC EDGAR filings.

## License

MIT. Data comes from Yahoo Finance through [yfinance](https://github.com/ranaroussi/yfinance) and may be delayed. Not affiliated with Yahoo. For research and education, not investment advice.
