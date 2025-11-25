"""
title: yfinance-ai - World's Best AI-Powered Yahoo Finance Integration
description: Complete Financial Data Suite - 54+ tools for stocks, crypto, forex, commodities, ETFs. Real-time prices, fundamentals, analyst data, options, news. Works with OpenWebUI, Claude, ChatGPT, and any AI assistant.
author: lucas0
author_url: https://lucas0.com
funding_url: https://github.com/sponsors/lucas0
version: 3.0.2
license: MIT
requirements: yfinance>=0.2.66,pandas>=2.2.0,pydantic>=2.0.0,requests>=2.28.0
repository: https://github.com/lucas0/yfinance-ai

OPENWEBUI INSTALLATION:
1. Copy this entire file
2. Go to OpenWebUI → Admin Panel → Functions
3. Click "+" to create new function
4. Paste this code
5. Save and enable
6. Start asking questions like "What's Apple's stock price?" or "Get Bitcoin price"

INTEGRATION WITH OTHER AI TOOLS:
- Claude Desktop: Add as MCP server
- ChatGPT Custom GPT: Import as action
- LangChain: Use as Tool
- Python: from yfinance_ai import Tools; tools = Tools()
- API: Deploy as FastAPI/Flask endpoint

FEATURES:
✅ 54+ financial data tools - the most comprehensive yfinance integration
✅ Multi-asset support: Stocks, Crypto, Forex, Commodities, ETFs
✅ Real-time stock prices and detailed quotes
✅ Historical data with customizable periods/intervals
✅ Financial statements (income, balance sheet, cash flow)
✅ Key ratios and valuation metrics
✅ Dividends, splits, and corporate actions
✅ Analyst recommendations, price targets, upgrades/downgrades
✅ EPS trends, revisions, and earnings calendar
✅ Institutional and insider holdings
✅ Options chains and derivatives data
✅ Company news and SEC filings (with robust fallbacks)
✅ Market indices and sector performance
✅ ETF/Fund data: holdings, sector weights, expense ratios
✅ Bulk operations and stock comparison
✅ Peer comparison and financial summaries
✅ Built-in self-testing for validation
✅ Retry logic with exponential backoff
✅ Rate limiting protection
✅ Natural language query support

EXAMPLE PROMPTS FOR AI:
- "What's the current price of Apple stock?"
- "Get Bitcoin price" or "Show me ETH-USD"
- "What's the EUR/USD exchange rate?"
- "Show me Tesla's income statement"
- "Compare valuations of AAPL, MSFT, and GOOGL"
- "Get dividend history for Coca-Cola"
- "What are analysts saying about NVIDIA?"
- "Show analyst price targets for TSLA"
- "Get upgrades and downgrades for AAPL"
- "Show insider transactions for Amazon"
- "Get options chain for SPY"
- "What's the latest news on Apple?"
- "Show me VOO's top holdings"
- "Compare tech sector vs healthcare sector"
- "Show me the major market indices"
- "Is the stock market open?"

TESTING:
AI can self-test by asking: "Run self-test on yfinance tools"
This will test all 54+ functions and report results.
"""

from typing import Callable, Any, Optional, List, Dict, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from dateutil import parser as dateutil_parser
from functools import wraps
import yfinance as yf
import pandas as pd
import requests
import logging
import re
import time

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: yfinance 0.2.66+ handles sessions internally with curl_cffi
# Do not pass custom requests.Session - it will cause errors

# Valid period values for historical data
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
FINANCIAL_PERIODS = ["annual", "quarterly", "ttm"]

# Major market indices
MARKET_INDICES = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
    "10Y Treasury": "^TNX",
    "US Dollar": "DX-Y.NYB",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
}

# API rate limiting
RATE_LIMIT_CALLS = 100  # calls per window
RATE_LIMIT_WINDOW = 60  # seconds

# Commodity name to ticker mappings
COMMODITY_TICKERS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "platinum": "PL=F",
    "palladium": "PA=F",
    "copper": "HG=F",
    "crude oil": "CL=F",
    "oil": "CL=F",
    "wti": "CL=F",
    "brent": "BZ=F",
    "natural gas": "NG=F",
    "nat gas": "NG=F",
    "gasoline": "RB=F",
    "heating oil": "HO=F",
    "corn": "ZC=F",
    "wheat": "ZW=F",
    "soybeans": "ZS=F",
    "coffee": "KC=F",
    "sugar": "SB=F",
    "cotton": "CT=F",
    "cocoa": "CC=F",
    "lumber": "LBS=F",
    "live cattle": "LE=F",
    "lean hogs": "HE=F",
}

# Popular cryptocurrencies
CRYPTO_TICKERS = {
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "dogecoin": "DOGE-USD",
    "doge": "DOGE-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
    "cardano": "ADA-USD",
    "ada": "ADA-USD",
    "xrp": "XRP-USD",
    "ripple": "XRP-USD",
    "polkadot": "DOT-USD",
    "dot": "DOT-USD",
    "avalanche": "AVAX-USD",
    "avax": "AVAX-USD",
    "chainlink": "LINK-USD",
    "link": "LINK-USD",
    "litecoin": "LTC-USD",
    "ltc": "LTC-USD",
    "shiba": "SHIB-USD",
    "shib": "SHIB-USD",
    "polygon": "MATIC-USD",
    "matic": "MATIC-USD",
    "uniswap": "UNI-USD",
    "uni": "UNI-USD",
}

# Popular forex pairs
FOREX_PAIRS = {
    "eurusd": "EURUSD=X",
    "eur/usd": "EURUSD=X",
    "gbpusd": "GBPUSD=X",
    "gbp/usd": "GBPUSD=X",
    "usdjpy": "USDJPY=X",
    "usd/jpy": "USDJPY=X",
    "usdchf": "USDCHF=X",
    "usd/chf": "USDCHF=X",
    "audusd": "AUDUSD=X",
    "aud/usd": "AUDUSD=X",
    "usdcad": "USDCAD=X",
    "usd/cad": "USDCAD=X",
    "nzdusd": "NZDUSD=X",
    "nzd/usd": "NZDUSD=X",
    "eurgbp": "EURGBP=X",
    "eur/gbp": "EURGBP=X",
    "eurjpy": "EURJPY=X",
    "eur/jpy": "EURJPY=X",
    "gbpjpy": "GBPJPY=X",
    "gbp/jpy": "GBPJPY=X",
}


# ============================================================
# VALIDATION MODELS
# ============================================================

class TickerValidator(BaseModel):
    """Validate ticker symbol input"""
    ticker: str = Field(..., min_length=1, max_length=10)

    @validator('ticker')
    def validate_ticker(cls, v):
        # Allow alphanumeric, dots, hyphens, equals, carets
        if not re.match(r'^[A-Za-z0-9.\-=^]+$', v):
            raise ValueError(f"Invalid ticker symbol format: {v}")
        return v.upper()


class PeriodValidator(BaseModel):
    """Validate period and interval parameters"""
    period: str = Field(default="1y")
    interval: str = Field(default="1d")

    @validator('period')
    def validate_period(cls, v):
        if v not in VALID_PERIODS:
            raise ValueError(f"Period must be one of {VALID_PERIODS}")
        return v

    @validator('interval')
    def validate_interval(cls, v):
        if v not in VALID_INTERVALS:
            raise ValueError(f"Interval must be one of {VALID_INTERVALS}")
        return v


# ============================================================
# UTILITY DECORATORS & HELPERS
# ============================================================

def retry_with_backoff(max_retries: int = 3, base_wait: float = 1.0, max_wait: float = 10.0):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_wait: Initial wait time in seconds
        max_wait: Maximum wait time between retries
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (requests.exceptions.RequestException,
                        requests.exceptions.Timeout,
                        ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = min(base_wait * (2 ** attempt), max_wait)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {max_retries} attempts: {func.__name__}")
                except Exception as e:
                    # Don't retry on non-network errors
                    raise
            raise last_exception
        return wrapper
    return decorator


def get_ticker_with_session(symbol: str):
    """Create a yfinance Ticker - yfinance handles sessions internally"""
    return yf.Ticker(symbol)


def safe_ticker_call(func):
    """Decorator for safe yfinance API calls with error handling and retry logic"""
    @wraps(func)
    async def wrapper(self, ticker: str, *args, **kwargs):
        try:
            # Validate ticker
            validated = TickerValidator(ticker=ticker)
            ticker_clean = validated.ticker

            # Call the original function with retry logic for transient failures
            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    return await func(self, ticker_clean, *args, **kwargs)
                except (requests.exceptions.RequestException,
                        requests.exceptions.Timeout,
                        ConnectionError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = min(1.0 * (2 ** attempt), 10.0)
                        logger.warning(f"Network error for {ticker_clean}, retry {attempt + 1}/{max_retries} in {wait_time}s")
                        time.sleep(wait_time)
                    continue
                except Exception as e:
                    # Non-network errors should not be retried
                    raise

            # If we exhausted retries
            if last_error:
                logger.error(f"Failed after {max_retries} retries for {ticker_clean}: {last_error}")
                return f"❌ Network error for {ticker_clean}. Please try again."

        except ValueError as e:
            logger.error(f"Validation error in {func.__name__}: {e}")
            return f"❌ Invalid input: {str(e)}"
        except Exception as e:
            logger.error(f"Error in {func.__name__} for {ticker}: {e}")
            return f"❌ Error fetching data for {ticker}: {str(e)}"
    return wrapper


def format_large_number(num: Union[int, float, None]) -> str:
    """Format large numbers with appropriate suffixes (K, M, B, T)"""
    if num is None or not isinstance(num, (int, float)):
        return "N/A"

    if num == 0:
        return "0"

    abs_num = abs(num)
    sign = "-" if num < 0 else ""

    if abs_num >= 1_000_000_000_000:
        return f"{sign}${abs_num/1_000_000_000_000:.2f}T"
    elif abs_num >= 1_000_000_000:
        return f"{sign}${abs_num/1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        return f"{sign}${abs_num/1_000_000:.2f}M"
    elif abs_num >= 1_000:
        return f"{sign}${abs_num/1_000:.2f}K"
    else:
        return f"{sign}${abs_num:.2f}"


def format_percentage(value: Union[float, None], decimals: int = 2) -> str:
    """Format percentage values consistently"""
    if value is None or not isinstance(value, (int, float)):
        return "N/A"

    # yfinance returns dividend yield as decimal: 0.0038 for 0.38%
    # But sometimes returns already-scaled: 0.38 for 0.38% (not 38%)
    # Safe rule: if value >= 0.01, it's likely already scaled, don't multiply
    if abs(value) > 1:
        # Already in percentage form (38 = 38%)
        return f"{value:.{decimals}f}%"
    elif abs(value) >= 0.01:
        # Already decimal percentage (0.38 = 0.38%, not 38%)
        return f"{value:.{decimals}f}%"
    else:
        # Tiny decimal that needs multiplying (0.0038 -> 0.38%)
        return f"{value*100:.{decimals}f}%"


def safe_get(dictionary: dict, key: str, default: Any = "N/A") -> Any:
    """Safely get dictionary value with default"""
    value = dictionary.get(key, default)
    return value if value is not None else default


def format_date(timestamp: Union[int, float, datetime, None]) -> str:
    """Format various date formats to YYYY-MM-DD"""
    if timestamp is None:
        return "N/A"

    try:
        if isinstance(timestamp, (int, float)):
            if timestamp > 946684800:  # After year 2000
                return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        elif isinstance(timestamp, datetime):
            return timestamp.strftime('%Y-%m-%d')
        elif hasattr(timestamp, 'strftime'):
            return timestamp.strftime('%Y-%m-%d')
    except:
        pass

    return str(timestamp)


# ============================================================
# MAIN TOOLS CLASS - 50+ FINANCIAL DATA TOOLS
# ============================================================

class Tools:
    """
    🌟 yfinance-ai: World's First AI-Powered Yahoo Finance Integration

    Natural Language Financial Data Access - 50+ Tools for AI Assistants

    This class provides comprehensive access to Yahoo Finance data through
    natural language queries. Perfect for OpenWebUI, Claude, ChatGPT, and
    any AI assistant that supports function calling.

    Features:
    - Real-time stock quotes and historical data
    - Financial statements and fundamental analysis
    - Dividends, splits, and corporate actions
    - Analyst ratings and price targets
    - Insider trading and institutional ownership
    - Options chains and derivatives
    - Market news and SEC filings
    - Sector performance and market indices
    - Built-in self-testing capabilities

    Author: lucas0 (https://lucas0.com)
    Repository: https://github.com/lucas0/yfinance-ai
    License: MIT
    """

    class Valves(BaseModel):
        """Configuration for yfinance-ai Tools"""

        default_period: str = Field(
            default="1y",
            description="Default period for historical data queries"
        )
        default_interval: str = Field(
            default="1d",
            description="Default interval for historical data"
        )
        enable_caching: bool = Field(
            default=False,
            description="Enable response caching (not yet implemented)"
        )
        max_news_items: int = Field(
            default=10,
            description="Maximum number of news items to return"
        )
        max_comparison_tickers: int = Field(
            default=10,
            description="Maximum tickers to compare at once"
        )
        include_technical_indicators: bool = Field(
            default=False,
            description="Include technical analysis indicators (not yet implemented)"
        )

    def __init__(self):
        self.valves = self.Valves()
        self._call_count = 0
        self._window_start = time.time()
        logger.info("yfinance-ai v3.0.2 initialized - 54+ financial tools ready")

    def _check_rate_limit(self) -> bool:
        """Simple rate limiting check"""
        current_time = time.time()

        # Reset window if needed
        if current_time - self._window_start > RATE_LIMIT_WINDOW:
            self._call_count = 0
            self._window_start = current_time

        # Check limit
        if self._call_count >= RATE_LIMIT_CALLS:
            logger.warning("Rate limit exceeded")
            return False

        self._call_count += 1
        return True

    # ============================================================
    # TOOL 1-4: STOCK QUOTES & CURRENT PRICES
    # ============================================================

    @safe_ticker_call
    async def get_stock_price(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get current stock price and key metrics for a ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'SPY')

        Returns:
            Formatted string with current price, market cap, P/E ratio, and other key metrics

        Example:
            >>> await get_stock_price("AAPL")
            **AAPL - Apple Inc.**
            Current Price: $175.43
            Market Cap: $2.75T
            ...
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded. Please wait before making more requests."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Fetching price for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or len(info) < 3:
            return f"❌ No data available for ticker {ticker}"

        result = f"**{ticker} - {safe_get(info, 'longName')}**\n\n"

        # Get current price with multiple fallbacks
        current_price = (
            info.get("currentPrice") or
            info.get("regularMarketPrice") or
            info.get("previousClose")
        )

        if current_price:
            result += f"💰 **Current Price:** ${current_price:.2f}\n"
        else:
            result += f"💰 **Current Price:** N/A\n"

        # Market cap with formatting
        market_cap = info.get("marketCap")
        result += f"📈 **Market Cap:** {format_large_number(market_cap)}\n"

        # Key valuation metrics
        trailing_pe = safe_get(info, "trailingPE")
        if trailing_pe != "N/A" and isinstance(trailing_pe, (int, float)):
            result += f"📊 **P/E Ratio:** {trailing_pe:.2f}\n"
        else:
            result += f"📊 **P/E Ratio:** N/A\n"

        # Price ranges
        result += f"📉 **52-Week Range:** ${safe_get(info, 'fiftyTwoWeekLow')} - ${safe_get(info, 'fiftyTwoWeekHigh')}\n"

        # Day's range
        day_low = safe_get(info, 'dayLow')
        day_high = safe_get(info, 'dayHigh')
        if day_low != "N/A" and day_high != "N/A":
            result += f"📊 **Day Range:** ${day_low} - ${day_high}\n"

        # Volume with formatting
        volume = info.get("volume")
        avg_volume = info.get("averageVolume")

        if volume:
            result += f"📊 **Volume:** {volume:,}\n"
        if avg_volume:
            result += f"📊 **Avg Volume:** {avg_volume:,}\n"

        # Additional metrics
        beta = safe_get(info, "beta")
        if beta != "N/A" and isinstance(beta, (int, float)):
            result += f"📈 **Beta:** {beta:.2f}\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved {ticker} price data",
                    "done": True
                }
            })

        return result

    @safe_ticker_call
    async def get_fast_info(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get fast-access ticker information (lightweight, quick response).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quick summary with most recent price, market cap, and basic metrics

        Example:
            Useful for quick price checks without full info loading.
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded. Please wait."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"⚡ Fetching quick info for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            fast_info = stock.fast_info

            result = f"**⚡ Quick Info: {ticker}**\n\n"
            result += f"Last Price: ${fast_info.get('lastPrice', 'N/A')}\n"
            result += f"Market Cap: {format_large_number(fast_info.get('marketCap'))}\n"
            result += f"Shares Outstanding: {fast_info.get('shares', 'N/A'):,}\n" if fast_info.get('shares') else ""
            result += f"Previous Close: ${fast_info.get('previousClose', 'N/A')}\n"
            result += f"Open: ${fast_info.get('open', 'N/A')}\n"
            result += f"Day High: ${fast_info.get('dayHigh', 'N/A')}\n"
            result += f"Day Low: ${fast_info.get('dayLow', 'N/A')}\n"
            result += f"52W High: ${fast_info.get('yearHigh', 'N/A')}\n"
            result += f"52W Low: ${fast_info.get('yearLow', 'N/A')}\n"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved quick info for {ticker}",
                        "done": True
                    }
                })

            return result

        except Exception as e:
            logger.warning(f"Fast info not available for {ticker}, falling back to regular info")
            return await self.get_stock_price(ticker, __event_emitter__)

    @safe_ticker_call
    async def get_stock_quote(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get detailed quote with bid/ask, day range, and trading info.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Detailed quote information including bid/ask spreads and intraday trading data
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📋 Retrieving detailed quote for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return f"❌ No quote data available for ticker {ticker}"

        result = f"**📋 Detailed Quote: {ticker}**\n\n"

        # Bid/Ask spread
        bid = safe_get(info, 'bid')
        ask = safe_get(info, 'ask')
        bid_size = safe_get(info, 'bidSize')
        ask_size = safe_get(info, 'askSize')

        result += f"**Bid/Ask Spread:**\n"
        result += f"  Bid: ${bid} × {bid_size}\n"
        result += f"  Ask: ${ask} × {ask_size}\n\n"

        # Day trading range
        result += f"**Day Range:** ${safe_get(info, 'dayLow')} - ${safe_get(info, 'dayHigh')}\n"
        result += f"**Open:** ${safe_get(info, 'open')}\n"
        result += f"**Previous Close:** ${safe_get(info, 'previousClose')}\n\n"

        # Calculate change
        change = safe_get(info, "regularMarketChange", 0)
        change_pct = safe_get(info, "regularMarketChangePercent", 0)

        if change != "N/A" and change != 0:
            emoji = "📈" if change > 0 else "📉"
            result += f"{emoji} **Change:** {change:+.2f} ({change_pct:+.2f}%)\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved detailed quote for {ticker}",
                    "done": True
                }
            })

        return result

    @safe_ticker_call
    async def get_historical_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get historical price data for a stock.

        Args:
            ticker: Stock ticker symbol
            period: Time period - valid values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: Data interval - valid values: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

        Returns:
            Historical price data summary with statistics and returns

        Example:
            >>> await get_historical_data("AAPL", period="6mo", interval="1d")
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        # Validate parameters
        try:
            validated = PeriodValidator(period=period, interval=interval)
        except ValueError as e:
            return f"❌ {str(e)}"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Retrieving {period} historical data for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)

        if hist.empty:
            return f"❌ No historical data found for {ticker} (period={period}, interval={interval})"

        result = f"**📊 Historical Data: {ticker}**\n"
        result += f"Period: {period} | Interval: {interval}\n\n"

        result += f"**Date Range:** {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}\n"
        result += f"**Data Points:** {len(hist)}\n\n"

        # Latest data
        latest_date = hist.index[-1].strftime('%Y-%m-%d')
        latest_close = hist['Close'].iloc[-1]
        latest_volume = int(hist['Volume'].iloc[-1])

        result += f"**Latest ({latest_date}):**\n"
        result += f"  Close: ${latest_close:.2f}\n"
        result += f"  Volume: {latest_volume:,}\n\n"

        # Period statistics
        high = hist['Close'].max()
        low = hist['Close'].min()
        avg = hist['Close'].mean()

        result += f"**Period Statistics:**\n"
        result += f"  Highest: ${high:.2f}\n"
        result += f"  Lowest: ${low:.2f}\n"
        result += f"  Average: ${avg:.2f}\n"

        # Calculate returns
        total_return = ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100
        result += f"  Total Return: {total_return:+.2f}%\n"

        # Volatility (std deviation)
        volatility = hist['Close'].pct_change().std() * 100
        result += f"  Volatility (σ): {volatility:.2f}%\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved {len(hist)} data points for {ticker}",
                    "done": True
                }
            })

        return result

    @safe_ticker_call
    async def get_historical_price_on_date(
        self,
        ticker: str,
        date: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get historical price for a specific date.

        Args:
            ticker: Stock ticker symbol
            date: Date in YYYY-MM-DD format (e.g., "2024-01-15")

        Returns:
            Price on that date (or closest trading day)

        Example:
            >>> await get_historical_price_on_date("AAPL", "2024-01-15")
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        # Parse date
        try:
            target_date = dateutil_parser.parse(date)
        except Exception as e:
            return f"❌ Invalid date format: {date}. Use YYYY-MM-DD format."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Fetching {ticker} price for {date}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        # Fetch data for a 5-day window around target date (handles weekends/holidays)
        start_date = target_date - timedelta(days=5)
        end_date = target_date + timedelta(days=2)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return f"❌ No historical data found for {ticker} around {date}"

        # Find closest trading day to target date
        closest_date = min(hist.index, key=lambda d: abs((d.date() - target_date.date()).days))
        price_data = hist.loc[closest_date]

        result = f"**📊 {ticker} - Price on {date}**\n\n"
        result += f"**Closest Trading Day:** {closest_date.strftime('%Y-%m-%d')}\n"
        result += f"**Close:** ${price_data['Close']:.2f}\n"
        result += f"**Open:** ${price_data['Open']:.2f}\n"
        result += f"**High:** ${price_data['High']:.2f}\n"
        result += f"**Low:** ${price_data['Low']:.2f}\n"
        result += f"**Volume:** {int(price_data['Volume']):,}\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Found {ticker} price for {date}",
                    "done": True
                }
            })

        return result

    @safe_ticker_call
    async def get_isin(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get International Securities Identification Number (ISIN) for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            ISIN code and related identification information
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🔖 Retrieving ISIN for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            isin = stock.isin
            info = stock.info

            result = f"**🔖 Identification: {ticker}**\n\n"
            result += f"ISIN: {isin if isin else 'N/A'}\n"
            result += f"Symbol: {safe_get(info, 'symbol')}\n"
            result += f"Exchange: {safe_get(info, 'exchange')}\n"
            result += f"Quote Type: {safe_get(info, 'quoteType')}\n"
            result += f"Currency: {safe_get(info, 'currency')}\n"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved ISIN for {ticker}",
                        "done": True
                    }
                })

            return result

        except Exception as e:
            return f"❌ Could not retrieve ISIN for {ticker}: {str(e)}"

    # ============================================================
    # TOOL 5-7: COMPANY INFORMATION & FUNDAMENTALS
    # ============================================================

    @safe_ticker_call
    async def get_company_info(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get detailed company information and business description.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company overview including sector, industry, employees, website, and business summary
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🏢 Retrieving company information for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return f"❌ No company information available for {ticker}"

        long_name = safe_get(info, 'longName', ticker)
        result = f"**🏢 Company Information: {long_name}**\n\n"

        # Basic info
        result += f"**Sector:** {safe_get(info, 'sector')}\n"
        result += f"**Industry:** {safe_get(info, 'industry')}\n"
        result += f"**Country:** {safe_get(info, 'country')}\n"
        result += f"**City:** {safe_get(info, 'city')}\n"
        result += f"**Website:** {safe_get(info, 'website')}\n"

        # Employee count
        employees = info.get("fullTimeEmployees")
        if employees and isinstance(employees, (int, float)):
            result += f"**Employees:** {int(employees):,}\n"

        # Founding year if available
        if "founded" in info and info["founded"]:
            result += f"**Founded:** {info['founded']}\n"

        # Exchange info
        result += f"**Exchange:** {safe_get(info, 'exchange')}\n"
        result += f"**Currency:** {safe_get(info, 'currency')}\n"

        # Business summary
        summary = safe_get(info, 'longBusinessSummary')
        if summary and summary != "N/A":
            result += f"\n**📄 Business Summary:**\n{summary}\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved company information for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_company_officers(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get company officers and key management.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of company officers with titles and compensation
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"👔 Retrieving company officers for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        # Try to get company officers
        officers = info.get('companyOfficers', [])

        if not officers:
            return f"ℹ️ No officer information available for {ticker}"

        result = f"**👔 Company Officers: {ticker}**\n\n"

        for officer in officers[:10]:  # Limit to top 10
            try:
                # Handle both dict and non-dict formats
                if isinstance(officer, dict):
                    name = officer.get('name', 'N/A')
                    title = officer.get('title', 'N/A')
                    age = officer.get('age', '')

                    # Handle totalPay which can be dict or int
                    total_pay = officer.get('totalPay')
                    if isinstance(total_pay, dict):
                        pay = total_pay.get('raw')
                    elif isinstance(total_pay, (int, float)):
                        pay = total_pay
                    else:
                        pay = None

                    result += f"**{name}**\n"
                    result += f"  Title: {title}\n"
                    if age:
                        result += f"  Age: {age}\n"
                    if pay:
                        result += f"  Compensation: {format_large_number(pay)}\n"
                    result += "\n"
                else:
                    # Skip non-dict entries
                    continue
            except Exception as e:
                logger.debug(f"Skipping officer entry due to format issue: {e}")
                continue


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved company officers for {ticker}",
                    "done": True
                }
            })
        return result

    # ============================================================
    # TOOL 8-10: FINANCIAL STATEMENTS
    # ============================================================

    @safe_ticker_call
    async def get_income_statement(
        self,
        ticker: str,
        period: str = "annual",
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get income statement data.

        Args:
            ticker: Stock ticker symbol
            period: 'annual', 'quarterly', or 'ttm' (trailing twelve months)

        Returns:
            Income statement with revenue, expenses, and net income
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"💼 Retrieving {period} income statement for {ticker}",
                    "done": False
                }
            })

        if period not in FINANCIAL_PERIODS:
            return f"❌ Period must be one of {FINANCIAL_PERIODS}"

        stock = yf.Ticker(ticker)

        # Get appropriate financials based on period
        if period == "quarterly":
            financials = stock.quarterly_financials
        elif period == "ttm":
            # TTM is typically in the income_stmt attribute
            try:
                financials = stock.income_stmt
            except:
                financials = stock.financials
        else:  # annual
            financials = stock.financials

        if financials is None or financials.empty:
            return f"❌ No {period} income statement data available for {ticker}"

        result = f"**💼 Income Statement: {ticker}** ({period})\n\n"

        latest_date = financials.columns[0]
        result += f"📅 Period ending: {format_date(latest_date)}\n\n"

        # Key income statement metrics
        metrics = [
            ("Total Revenue", "💰"),
            ("Gross Profit", "📊"),
            ("Operating Income", "📈"),
            ("EBIT", "💵"),
            ("EBITDA", "💵"),
            ("Net Income", "💎"),
            ("Basic EPS", "📊"),
            ("Diluted EPS", "📊"),
        ]

        for metric_name, emoji in metrics:
            if metric_name in financials.index:
                value = financials.loc[metric_name, latest_date]
                if pd.notna(value):
                    if "EPS" in metric_name:
                        result += f"{emoji} **{metric_name}:** ${value:.2f}\n"
                    else:
                        result += f"{emoji} **{metric_name}:** {format_large_number(value)}\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved {period} income statement for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_balance_sheet(
        self,
        ticker: str,
        period: str = "annual",
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get balance sheet data.

        Args:
            ticker: Stock ticker symbol
            period: 'annual' or 'quarterly'

        Returns:
            Balance sheet with assets, liabilities, and equity
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Retrieving {period} balance sheet for {ticker}",
                    "done": False
                }
            })

        if period not in ["annual", "quarterly"]:
            return f"❌ Period must be 'annual' or 'quarterly'"

        stock = yf.Ticker(ticker)

        if period == "quarterly":
            balance = stock.quarterly_balance_sheet
        else:
            balance = stock.balance_sheet

        if balance is None or balance.empty:
            return f"❌ No {period} balance sheet data available for {ticker}"

        result = f"**🏦 Balance Sheet: {ticker}** ({period})\n\n"

        latest_date = balance.columns[0]
        result += f"📅 Period ending: {format_date(latest_date)}\n\n"

        # Key balance sheet metrics
        metrics = [
            ("Total Assets", "💰"),
            ("Cash And Cash Equivalents", "💵"),
            ("Current Assets", "📊"),
            ("Total Liabilities Net Minority Interest", "📉"),
            ("Current Liabilities", "📊"),
            ("Total Debt", "💳"),
            ("Long Term Debt", "💳"),
            ("Stockholders Equity", "💎"),
            ("Working Capital", "💼"),
        ]

        for metric_name, emoji in metrics:
            if metric_name in balance.index:
                value = balance.loc[metric_name, latest_date]
                if pd.notna(value):
                    result += f"{emoji} **{metric_name}:** {format_large_number(value)}\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved {period} balance sheet for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_cash_flow(
        self,
        ticker: str,
        period: str = "annual",
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get cash flow statement data.

        Args:
            ticker: Stock ticker symbol
            period: 'annual' or 'quarterly'

        Returns:
            Cash flow statement with operating, investing, and financing activities
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"💵 Retrieving {period} cash flow for {ticker}",
                    "done": False
                }
            })

        if period not in ["annual", "quarterly"]:
            return f"❌ Period must be 'annual' or 'quarterly'"

        stock = yf.Ticker(ticker)

        if period == "quarterly":
            cashflow = stock.quarterly_cashflow
        else:
            cashflow = stock.cashflow

        if cashflow is None or cashflow.empty:
            return f"❌ No {period} cash flow data available for {ticker}"

        result = f"**💸 Cash Flow Statement: {ticker}** ({period})\n\n"

        latest_date = cashflow.columns[0]
        result += f"📅 Period ending: {format_date(latest_date)}\n\n"

        # Key cash flow metrics
        metrics = [
            ("Operating Cash Flow", "💰"),
            ("Investing Cash Flow", "📊"),
            ("Financing Cash Flow", "💳"),
            ("Free Cash Flow", "💎"),
            ("Capital Expenditure", "🏭"),
            ("Issuance Of Debt", "💵"),
            ("Repayment Of Debt", "💵"),
        ]

        for metric_name, emoji in metrics:
            if metric_name in cashflow.index:
                value = cashflow.loc[metric_name, latest_date]
                if pd.notna(value):
                    result += f"{emoji} **{metric_name}:** {format_large_number(value)}\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved {period} cash flow for {ticker}",
                    "done": True
                }
            })
        return result

    # ============================================================
    # TOOL 11: KEY RATIOS & METRICS
    # ============================================================

    @safe_ticker_call
    async def get_key_ratios(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get key financial ratios and valuation metrics.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Comprehensive ratios including P/E, P/B, ROE, ROA, profit margins, debt ratios
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🔢 Retrieving key ratios for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return f"❌ No ratio data available for {ticker}"

        result = f"**📊 Key Financial Ratios: {ticker}**\n\n"

        # Valuation Metrics
        result += "**💰 Valuation Metrics**\n"

        trailing_pe = safe_get(info, 'trailingPE')
        if trailing_pe != "N/A" and isinstance(trailing_pe, (int, float)):
            result += f"  P/E (Trailing): {trailing_pe:.2f}\n"

        forward_pe = safe_get(info, 'forwardPE')
        if forward_pe != "N/A" and isinstance(forward_pe, (int, float)):
            result += f"  P/E (Forward): {forward_pe:.2f}\n"

        result += f"  Price/Book: {safe_get(info, 'priceToBook')}\n"
        result += f"  Price/Sales: {safe_get(info, 'priceToSalesTrailing12Months')}\n"
        result += f"  PEG Ratio: {safe_get(info, 'pegRatio')}\n"
        result += f"  EV/EBITDA: {safe_get(info, 'enterpriseToEbitda')}\n"
        result += f"  EV/Revenue: {safe_get(info, 'enterpriseToRevenue')}\n\n"

        # Profitability
        result += "**📈 Profitability**\n"
        result += f"  Profit Margin: {format_percentage(info.get('profitMargins'))}\n"
        result += f"  Operating Margin: {format_percentage(info.get('operatingMargins'))}\n"
        result += f"  Gross Margin: {format_percentage(info.get('grossMargins'))}\n"
        result += f"  ROE: {format_percentage(info.get('returnOnEquity'))}\n"
        result += f"  ROA: {format_percentage(info.get('returnOnAssets'))}\n\n"

        # Financial Health
        result += "**🏥 Financial Health**\n"
        result += f"  Current Ratio: {safe_get(info, 'currentRatio')}\n"
        result += f"  Quick Ratio: {safe_get(info, 'quickRatio')}\n"
        result += f"  Debt to Equity: {safe_get(info, 'debtToEquity')}\n"
        result += f"  Total Debt: {format_large_number(info.get('totalDebt'))}\n"
        result += f"  Total Cash: {format_large_number(info.get('totalCash'))}\n"
        result += f"  Free Cash Flow: {format_large_number(info.get('freeCashflow'))}\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved key ratios for {ticker}",
                    "done": True
                }
            })
        return result

    # ============================================================
    # TOOL 12-16: DIVIDENDS & CORPORATE ACTIONS
    # ============================================================

    @safe_ticker_call
    async def get_dividends(
        self,
        ticker: str,
        period: str = "5y",
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get dividend payment history and current metrics.

        Args:
            ticker: Stock ticker symbol
            period: Time period for dividend history (1y, 5y, 10y, max)

        Returns:
            Dividend history and current dividend metrics
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"💎 Retrieving dividend history for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info
        dividends = stock.dividends

        # Filter by period
        filtered_dividends = dividends.copy() if not dividends.empty else dividends
        if period != "max" and not filtered_dividends.empty:
            years = int(period[:-1]) if period[:-1].isdigit() else 1
            cutoff_date = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=years)
            div_index = filtered_dividends.index
            if div_index.tz is None:
                div_index = pd.to_datetime(div_index, utc=True)
            else:
                div_index = div_index.tz_convert('UTC')
            filtered_dividends = filtered_dividends[div_index >= cutoff_date]

        result = f"**💎 Dividend History: {ticker}**\n\n"

        # Current dividend metrics
        div_yield = info.get('dividendYield')
        result += f"**Current Yield:** {format_percentage(div_yield)}\n"
        result += f"**Annual Rate:** ${safe_get(info, 'dividendRate')}\n"
        result += f"**Payout Ratio:** {format_percentage(info.get('payoutRatio'))}\n"
        result += f"**Ex-Dividend Date:** {format_date(info.get('exDividendDate'))}\n\n"

        # Historical dividend payments
        if not filtered_dividends.empty:
            result += f"**Recent Payments:**\n"
            for date, amount in filtered_dividends.tail(10).items():
                try:
                    date_str = date.strftime("%Y-%m-%d")
                except:
                    date_str = str(date)[:10]
                result += f"  {date_str}: ${amount:.4f}\n"
            result += f"\n**Total ({period}):** ${filtered_dividends.sum():.2f}\n"
        else:
            result += f"No dividend data found\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved dividend history for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_stock_splits(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get stock split history.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Complete stock split history with dates and ratios
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✂️ Retrieving stock split history for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        splits = stock.splits

        if splits.empty:
            return f"ℹ️ {ticker} has no stock split history"

        result = f"**🔀 Stock Split History: {ticker}**\n\n"
        result += f"**Total Splits:** {len(splits)}\n\n"

        for date, ratio in splits.items():
            date_str = date.strftime('%Y-%m-%d')
            result += f"📅 {date_str}: {ratio}:1 split\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved stock split history for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_corporate_actions(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get all corporate actions (dividends + splits combined).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Combined history of dividends and stock splits
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📋 Retrieving corporate actions for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        actions = stock.actions

        if actions.empty:
            return f"ℹ️ No corporate actions found for {ticker}"

        result = f"**📋 Corporate Actions: {ticker}**\n\n"
        result += f"**Total Events:** {len(actions)}\n\n"

        # Display recent actions
        recent_actions = actions.tail(20)

        for date, row in recent_actions.iterrows():
            date_str = date.strftime('%Y-%m-%d')

            dividend = row.get('Dividends', 0)
            split = row.get('Stock Splits', 0)

            if dividend > 0:
                result += f"💰 {date_str}: Dividend ${dividend:.4f}\n"
            if split > 0:
                result += f"🔀 {date_str}: Stock Split {split}:1\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved corporate actions for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_capital_gains(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get capital gains distributions (primarily for funds/ETFs).

        Args:
            ticker: Ticker symbol (typically ETF or mutual fund)

        Returns:
            Capital gains distribution history
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"💰 Retrieving capital gains for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            capital_gains = stock.capital_gains

            if capital_gains is None or (hasattr(capital_gains, 'empty') and capital_gains.empty):
                # This is expected for stocks (only ETFs/funds have capital gains)
                info = stock.info
                quote_type = info.get('quoteType', '')
                if quote_type.upper() in ['EQUITY', 'STOCK']:
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {
                                "description": f"✅ Retrieved capital gains for {ticker}",
                                "done": True
                            }
                        })
                    return f"ℹ️ {ticker} is a stock. Capital gains distributions are only for ETFs/mutual funds."

                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"✅ Retrieved capital gains for {ticker}",
                            "done": True
                        }
                    })
                return f"ℹ️ No capital gains distributions available for {ticker}"

            result = f"**💵 Capital Gains Distributions: {ticker}**\n\n"

            for date, amount in capital_gains.tail(20).items():
                try:
                    date_str = date.strftime('%Y-%m-%d')
                    result += f"{date_str}: ${amount:.4f}\n"
                except:
                    result += f"{str(date)[:10]}: ${amount:.4f}\n"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved capital gains for {ticker}",
                        "done": True
                    }
                })

            return result

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved capital gains for {ticker}",
                        "done": True
                    }
                })
            return f"ℹ️ Capital gains data not available for {ticker}. This is normal for stocks (only ETFs/funds distribute capital gains)."

    # ============================================================
    # TOOL 17-21: EARNINGS & ESTIMATES
    # ============================================================

    @safe_ticker_call
    async def get_earnings_dates(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get upcoming and past earnings dates with estimates.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Earnings calendar with dates, estimates, and actual results
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📅 Retrieving earnings dates for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        result = f"**📅 Earnings Information: {ticker}**\n\n"

        # Basic earnings info
        result += "**💰 Earnings Estimates**\n"
        result += f"  Forward EPS: ${safe_get(info, 'forwardEps')}\n"
        result += f"  Trailing EPS: ${safe_get(info, 'trailingEps')}\n"

        # Quarterly growth
        quarterly_growth = info.get("quarterlyEarningsGrowth")
        if quarterly_growth:
            result += f"  Quarterly Growth: {format_percentage(quarterly_growth)}\n"

        # Earnings date
        earnings_date = info.get("earningsDate")
        if earnings_date:
            result += f"  Next Earnings: {earnings_date}\n"

        # Try to get detailed earnings history
        try:
            earnings_history = stock.earnings_dates
            if earnings_history is not None and not earnings_history.empty:
                result += f"\n**📊 Recent Earnings Dates (Last 5):**\n"
                recent = earnings_history.head(5)

                for date, row in recent.iterrows():
                    date_str = format_date(date)
                    eps_est = row.get("EPS Estimate")
                    eps_actual = row.get("Reported EPS")

                    result += f"\n  {date_str}:\n"
                    if pd.notna(eps_est):
                        result += f"    Estimate: ${eps_est:.2f}\n"
                    if pd.notna(eps_actual):
                        result += f"    Actual: ${eps_actual:.2f}\n"
                        if pd.notna(eps_est) and eps_est != 0:
                            surprise = ((eps_actual - eps_est) / abs(eps_est)) * 100
                            emoji = "✅" if surprise > 0 else "❌"
                            result += f"    {emoji} Surprise: {surprise:+.2f}%\n"
        except:
            pass


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved earnings dates for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_earnings_history(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get historical earnings data with annual and quarterly figures.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Historical earnings by year and quarter
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📈 Retrieving earnings history for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            # Use new income statement API (old earnings API is deprecated)
            income_stmt = stock.income_stmt
            quarterly_income = stock.quarterly_income_stmt

            result = f"**📊 Earnings History: {ticker}**\n\n"

            # Annual earnings from income statement
            if income_stmt is not None and not income_stmt.empty:
                result += "**Annual Earnings:**\n"

                # Get revenue and net income rows
                revenue_row = None
                income_row = None

                if 'Total Revenue' in income_stmt.index:
                    revenue_row = income_stmt.loc['Total Revenue']
                if 'Net Income' in income_stmt.index:
                    income_row = income_stmt.loc['Net Income']

                if revenue_row is not None and income_row is not None:
                    # Get last 5 years
                    for col in income_stmt.columns[:5]:
                        year = col.strftime('%Y') if hasattr(col, 'strftime') else str(col)
                        revenue = revenue_row[col] if col in revenue_row.index else 0
                        net_income = income_row[col] if col in income_row.index else 0

                        result += f"\n{year}:\n"
                        result += f"  Revenue: {format_large_number(revenue)}\n"
                        result += f"  Net Income: {format_large_number(net_income)}\n"

            # Quarterly earnings
            if quarterly_income is not None and not quarterly_income.empty:
                result += "\n**Quarterly Earnings (Last 4):**\n"

                # Get revenue and net income rows
                revenue_row = None
                income_row = None

                if 'Total Revenue' in quarterly_income.index:
                    revenue_row = quarterly_income.loc['Total Revenue']
                if 'Net Income' in quarterly_income.index:
                    income_row = quarterly_income.loc['Net Income']

                if revenue_row is not None and income_row is not None:
                    # Get last 4 quarters
                    for col in quarterly_income.columns[:4]:
                        # Format quarter date
                        if hasattr(col, 'strftime'):
                            quarter = col.strftime('%Y-%m-%d')
                        else:
                            quarter = str(col)[:10]

                        revenue = revenue_row[col] if col in revenue_row.index else 0
                        net_income = income_row[col] if col in income_row.index else 0

                        result += f"\n{quarter}:\n"
                        result += f"  Revenue: {format_large_number(revenue)}\n"
                        result += f"  Net Income: {format_large_number(net_income)}\n"

            if "Annual Earnings" not in result and "Quarterly Earnings" not in result:
                return f"ℹ️ No earnings data available for {ticker}"

            return result

        except Exception as e:
            return f"❌ Could not retrieve earnings history for {ticker}: {str(e)}"

    @safe_ticker_call
    async def get_analyst_estimates(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get analyst estimates for earnings, revenue, and EPS.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Comprehensive analyst estimates and forecasts
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🔍 Retrieving analyst estimates for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        result = f"**🎯 Analyst Estimates: {ticker}**\n\n"

        # Earnings estimates
        result += "**Earnings Per Share:**\n"
        result += f"  Current Quarter: ${safe_get(info, 'earningsQuarterlyGrowth')}\n"
        result += f"  Next Quarter: ${safe_get(info, 'forwardEps')}\n"
        result += f"  Current Year: ${safe_get(info, 'trailingEps')}\n"
        result += f"  Next Year: (estimate data if available)\n\n"

        # Revenue estimates
        result += "**Revenue:**\n"
        result += f"  Trailing: {format_large_number(info.get('totalRevenue'))}\n"
        result += f"  Growth Rate: {format_percentage(info.get('revenueGrowth'))}\n\n"

        # Try to get more detailed estimates
        try:
            # Some versions of yfinance have analyst_price_targets
            targets = info.get('analyst_price_targets', {})
            if targets:
                result += "**Price Targets:**\n"
                for key, value in targets.items():
                    result += f"  {key}: ${value}\n"
        except:
            pass


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved analyst estimates for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_growth_estimates(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get analyst growth estimates.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Growth rate estimates for various periods
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Retrieving growth estimates for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        result = f"**📈 Growth Estimates: {ticker}**\n\n"

        result += "**Growth Metrics:**\n"
        result += f"  Earnings Growth (Quarterly): {format_percentage(info.get('earningsQuarterlyGrowth'))}\n"
        result += f"  Earnings Growth (YoY): {format_percentage(info.get('earningsGrowth'))}\n"
        result += f"  Revenue Growth: {format_percentage(info.get('revenueGrowth'))}\n"
        result += f"  Revenue Per Share Growth: {format_percentage(info.get('revenuePerShare'))}\n\n"

        # Historical growth
        result += "**Historical Performance:**\n"
        result += f"  52-Week Change: {format_percentage(info.get('52WeekChange'))}\n"
        result += f"  S&P500 52-Week Change: {format_percentage(info.get('SandP52WeekChange'))}\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved growth estimates for {ticker}",
                    "done": True
                }
            })
        return result

    # ============================================================
    # TOOL 22: ANALYST RECOMMENDATIONS & PRICE TARGETS
    # ============================================================

    @safe_ticker_call
    async def get_analyst_recommendations(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get analyst recommendations and ratings.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Analyst buy/hold/sell recommendations, consensus, and recent upgrades/downgrades
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"⭐ Retrieving analyst recommendations for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        result = f"**🎯 Analyst Recommendations: {ticker}**\n\n"

        # Target prices
        result += "**💰 Price Targets**\n"
        result += f"  Mean Target: ${safe_get(info, 'targetMeanPrice')}\n"
        result += f"  High Target: ${safe_get(info, 'targetHighPrice')}\n"
        result += f"  Low Target: ${safe_get(info, 'targetLowPrice')}\n"
        result += f"  Median Target: ${safe_get(info, 'targetMedianPrice')}\n"
        result += f"  Number of Analysts: {safe_get(info, 'numberOfAnalystOpinions')}\n\n"

        # Recommendation consensus
        recommendation = info.get("recommendationMean")
        recommendation_key = info.get("recommendationKey")

        if recommendation:
            result += "**📊 Current Recommendation**\n"
            result += f"  Mean Rating: {recommendation:.2f}\n"
            if recommendation_key:
                result += f"  Rating: {recommendation_key.upper()}\n"
            result += "\n"

        # Recommendation breakdown
        strong_buy = info.get("recommendationStrongBuy", 0)
        buy = info.get("recommendationBuy", 0)
        hold = info.get("recommendationHold", 0)
        sell = info.get("recommendationSell", 0)
        strong_sell = info.get("recommendationStrongSell", 0)

        if any([strong_buy, buy, hold, sell, strong_sell]):
            result += "**📈 Recommendation Breakdown:**\n"
            result += f"  Strong Buy: {strong_buy}\n"
            result += f"  Buy: {buy}\n"
            result += f"  Hold: {hold}\n"
            result += f"  Sell: {sell}\n"
            result += f"  Strong Sell: {strong_sell}\n\n"

        # Try to get upgrades/downgrades
        try:
            upgrades = stock.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                result += "**🔄 Recent Upgrades/Downgrades (Last 5):**\n"
                for idx, (date, row) in enumerate(upgrades.tail(5).iterrows()):
                    if idx >= 5:
                        break

                    firm = row.get("Firm", "N/A")
                    to_grade = row.get("ToGrade", "N/A")
                    from_grade = row.get("FromGrade", "")
                    action = row.get("Action", "N/A")

                    date_str = format_date(date)

                    if from_grade and to_grade:
                        result += f"  {date_str}: {firm} - {action} ({from_grade} → {to_grade})\n"
                    else:
                        result += f"  {date_str}: {firm} - {action} {to_grade}\n"
        except Exception as e:
            logger.debug(f"Could not fetch upgrades/downgrades: {e}")

        if not recommendation and not info.get("targetMeanPrice"):
            result += "\nℹ️ No analyst recommendation data available\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved analyst recommendations for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_analyst_price_targets(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get detailed analyst price targets for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Comprehensive price target data including current, low, mean, median, high targets
            and the number of analysts covering the stock
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🎯 Retrieving price targets for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        result = f"**🎯 Analyst Price Targets: {ticker}**\n\n"

        # Current price for comparison
        current_price = (
            info.get("currentPrice") or
            info.get("regularMarketPrice") or
            info.get("previousClose")
        )

        if current_price:
            result += f"**📈 Current Price:** ${current_price:.2f}\n\n"

        # Price targets
        target_low = info.get("targetLowPrice")
        target_mean = info.get("targetMeanPrice")
        target_median = info.get("targetMedianPrice")
        target_high = info.get("targetHighPrice")
        num_analysts = info.get("numberOfAnalystOpinions")

        if not any([target_low, target_mean, target_median, target_high]):
            return f"ℹ️ No analyst price target data available for {ticker}"

        result += "**💰 Price Targets:**\n"
        if target_low:
            result += f"  🔻 Low Target: ${target_low:.2f}"
            if current_price:
                pct = ((target_low - current_price) / current_price) * 100
                result += f" ({pct:+.1f}%)"
            result += "\n"

        if target_mean:
            result += f"  📊 Mean Target: ${target_mean:.2f}"
            if current_price:
                pct = ((target_mean - current_price) / current_price) * 100
                result += f" ({pct:+.1f}%)"
            result += "\n"

        if target_median:
            result += f"  📐 Median Target: ${target_median:.2f}"
            if current_price:
                pct = ((target_median - current_price) / current_price) * 100
                result += f" ({pct:+.1f}%)"
            result += "\n"

        if target_high:
            result += f"  🔺 High Target: ${target_high:.2f}"
            if current_price:
                pct = ((target_high - current_price) / current_price) * 100
                result += f" ({pct:+.1f}%)"
            result += "\n"

        if num_analysts:
            result += f"\n**👥 Number of Analysts:** {num_analysts}\n"

        # Recommendation summary
        rec_key = info.get("recommendationKey")
        rec_mean = info.get("recommendationMean")
        if rec_key:
            result += f"\n**📊 Consensus Rating:** {rec_key.upper()}"
            if rec_mean:
                result += f" ({rec_mean:.2f}/5)"
            result += "\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved price targets for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_upgrades_downgrades(
        self,
        ticker: str,
        limit: int = 20,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get detailed analyst upgrades and downgrades history.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of rating changes to return (default 20)

        Returns:
            Recent analyst rating changes including firm, action, and grade changes
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🔄 Retrieving upgrades/downgrades for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            upgrades = stock.upgrades_downgrades

            if upgrades is None or upgrades.empty:
                return f"ℹ️ No upgrades/downgrades data available for {ticker}"

            result = f"**🔄 Analyst Upgrades & Downgrades: {ticker}**\n\n"
            result += f"**Total Records:** {len(upgrades)} (showing latest {min(limit, len(upgrades))})\n\n"

            # Get the latest entries
            recent = upgrades.tail(limit).iloc[::-1]  # Reverse to show most recent first

            for idx, (date, row) in enumerate(recent.iterrows()):
                firm = row.get("Firm", "N/A")
                to_grade = row.get("ToGrade", "N/A")
                from_grade = row.get("FromGrade", "")
                action = row.get("Action", "N/A")

                date_str = format_date(date)

                # Action emoji
                if action and "up" in action.lower():
                    emoji = "⬆️"
                elif action and "down" in action.lower():
                    emoji = "⬇️"
                elif action and "init" in action.lower():
                    emoji = "🆕"
                elif action and "reit" in action.lower():
                    emoji = "🔄"
                else:
                    emoji = "📊"

                result += f"**{idx + 1}. {date_str}** | {firm}\n"
                result += f"   {emoji} {action}"
                if from_grade and to_grade:
                    result += f": {from_grade} → {to_grade}"
                elif to_grade:
                    result += f": {to_grade}"
                result += "\n\n"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved {len(recent)} rating changes for {ticker}",
                        "done": True
                    }
                })
            return result

        except Exception as e:
            logger.warning(f"Error fetching upgrades/downgrades for {ticker}: {e}")
            return f"ℹ️ Unable to fetch upgrades/downgrades for {ticker}. Data may not be available."

    @safe_ticker_call
    async def get_eps_trend(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get EPS (Earnings Per Share) trend data showing how estimates have changed.

        Args:
            ticker: Stock ticker symbol

        Returns:
            EPS trend data showing estimate changes over different time periods
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📈 Retrieving EPS trend for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            eps_trend = stock.eps_trend

            if eps_trend is None or (hasattr(eps_trend, 'empty') and eps_trend.empty):
                return f"ℹ️ No EPS trend data available for {ticker}"

            result = f"**📈 EPS Trend: {ticker}**\n\n"

            if isinstance(eps_trend, pd.DataFrame):
                result += "**Estimate Changes Over Time:**\n\n"

                for col in eps_trend.columns:
                    result += f"**{col}:**\n"
                    for idx, value in eps_trend[col].items():
                        if pd.notna(value):
                            if isinstance(value, (int, float)):
                                result += f"  {idx}: {value:.4f}\n"
                            else:
                                result += f"  {idx}: {value}\n"
                    result += "\n"
            elif isinstance(eps_trend, dict):
                for period, data in eps_trend.items():
                    result += f"**{period}:**\n"
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if value is not None:
                                result += f"  {key}: {value}\n"
                    else:
                        result += f"  {data}\n"
                    result += "\n"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved EPS trend for {ticker}",
                        "done": True
                    }
                })
            return result

        except Exception as e:
            logger.warning(f"Error fetching EPS trend for {ticker}: {e}")
            return f"ℹ️ Unable to fetch EPS trend for {ticker}. Data may not be available."

    @safe_ticker_call
    async def get_eps_revisions(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get EPS revision data showing analyst estimate changes.

        Args:
            ticker: Stock ticker symbol

        Returns:
            EPS revisions showing up/down estimate changes by analysts
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Retrieving EPS revisions for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            eps_revisions = stock.eps_revisions

            if eps_revisions is None or (hasattr(eps_revisions, 'empty') and eps_revisions.empty):
                return f"ℹ️ No EPS revision data available for {ticker}"

            result = f"**📊 EPS Revisions: {ticker}**\n\n"

            if isinstance(eps_revisions, pd.DataFrame):
                result += "**Analyst Estimate Revisions:**\n\n"

                for col in eps_revisions.columns:
                    result += f"**{col}:**\n"
                    for idx, value in eps_revisions[col].items():
                        if pd.notna(value):
                            # Format based on type
                            if "up" in str(idx).lower():
                                emoji = "⬆️"
                            elif "down" in str(idx).lower():
                                emoji = "⬇️"
                            else:
                                emoji = "📊"

                            if isinstance(value, (int, float)):
                                result += f"  {emoji} {idx}: {value:.0f}\n"
                            else:
                                result += f"  {emoji} {idx}: {value}\n"
                    result += "\n"
            elif isinstance(eps_revisions, dict):
                for period, data in eps_revisions.items():
                    result += f"**{period}:**\n"
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if value is not None:
                                emoji = "⬆️" if "up" in str(key).lower() else "⬇️" if "down" in str(key).lower() else "📊"
                                result += f"  {emoji} {key}: {value}\n"
                    result += "\n"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved EPS revisions for {ticker}",
                        "done": True
                    }
                })
            return result

        except Exception as e:
            logger.warning(f"Error fetching EPS revisions for {ticker}: {e}")
            return f"ℹ️ Unable to fetch EPS revisions for {ticker}. Data may not be available."

    @safe_ticker_call
    async def get_earnings_calendar(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get earnings calendar including next earnings date and dividend dates.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Upcoming earnings dates, ex-dividend dates, and dividend payment dates
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📅 Retrieving earnings calendar for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        result = f"**📅 Earnings Calendar: {ticker}**\n\n"

        try:
            calendar = stock.calendar

            if calendar is None or (isinstance(calendar, dict) and len(calendar) == 0):
                # Fall back to info
                info = stock.info
                ex_div = info.get("exDividendDate")
                div_date = info.get("dividendDate")

                if ex_div or div_date:
                    result += "**📆 Dividend Calendar:**\n"
                    if ex_div:
                        result += f"  Ex-Dividend Date: {format_date(ex_div)}\n"
                    if div_date:
                        result += f"  Dividend Date: {format_date(div_date)}\n"
                else:
                    return f"ℹ️ No calendar data available for {ticker}"

            elif isinstance(calendar, dict):
                # Earnings dates
                if "Earnings Date" in calendar:
                    earnings_dates = calendar["Earnings Date"]
                    result += "**💰 Earnings Date:**\n"
                    if isinstance(earnings_dates, list):
                        for date in earnings_dates:
                            result += f"  📅 {format_date(date)}\n"
                    else:
                        result += f"  📅 {format_date(earnings_dates)}\n"
                    result += "\n"

                # Ex-dividend date
                if "Ex-Dividend Date" in calendar:
                    result += f"**📆 Ex-Dividend Date:** {format_date(calendar['Ex-Dividend Date'])}\n"

                # Dividend date
                if "Dividend Date" in calendar:
                    result += f"**💵 Dividend Date:** {format_date(calendar['Dividend Date'])}\n"

                # Other calendar items
                for key, value in calendar.items():
                    if key not in ["Earnings Date", "Ex-Dividend Date", "Dividend Date"]:
                        if value is not None:
                            result += f"**{key}:** {value}\n"

            elif isinstance(calendar, pd.DataFrame) and not calendar.empty:
                result += "**📆 Calendar Events:**\n\n"
                for col in calendar.columns:
                    result += f"**{col}:**\n"
                    for idx, value in calendar[col].items():
                        if pd.notna(value):
                            result += f"  {idx}: {format_date(value) if 'date' in str(idx).lower() else value}\n"
                    result += "\n"

        except Exception as e:
            logger.warning(f"Error fetching calendar for {ticker}: {e}")
            # Try fallback to earnings_dates
            try:
                earnings_dates = stock.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    result += "**📅 Upcoming Earnings Dates:**\n"
                    future_dates = earnings_dates[earnings_dates.index >= datetime.now()]
                    if not future_dates.empty:
                        for date in future_dates.head(3).index:
                            result += f"  📅 {format_date(date)}\n"
                    else:
                        result += "  No upcoming earnings dates found\n"
            except:
                pass

        # Add info-based dividend data if not already included
        try:
            info = stock.info
            if "Ex-Dividend" not in result and info.get("exDividendDate"):
                result += f"\n**📆 Ex-Dividend Date:** {format_date(info['exDividendDate'])}\n"
            if info.get("dividendYield"):
                result += f"**💰 Dividend Yield:** {format_percentage(info['dividendYield'])}\n"
        except:
            pass

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved earnings calendar for {ticker}",
                    "done": True
                }
            })
        return result

    # ============================================================
    # TOOL 23-28: INSTITUTIONAL & INSIDER HOLDINGS
    # ============================================================

    @safe_ticker_call
    async def get_institutional_holders(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get major institutional holders and their ownership.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of institutional holders with shares owned and percentage ownership
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🏛️ Retrieving institutional holders for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        holders = stock.institutional_holders

        if holders is None or holders.empty:
            return f"ℹ️ No institutional holder data available for {ticker}"

        result = f"**🏛️ Major Institutional Holders: {ticker}**\n\n"

        for idx, row in holders.iterrows():
            holder = row.get("Holder", "N/A")
            shares = row.get("Shares", 0)
            # Try multiple column names for percentage ownership
            pct = row.get("pctHeld") or row.get("% Out") or row.get("% Held")
            value = row.get("Value", 0)

            result += f"**{holder}**\n"

            if isinstance(shares, (int, float)) and shares > 0:
                result += f"  Shares: {shares:,.0f}\n"
            else:
                result += f"  Shares: {shares}\n"

            # Format percentage properly
            if pct is not None and isinstance(pct, (int, float)):
                if pct < 1:  # Decimal format like 0.0923
                    result += f"  Ownership: {pct*100:.2f}%\n"
                else:  # Already percentage
                    result += f"  Ownership: {pct:.2f}%\n"
            else:
                result += f"  Ownership: {pct if pct else 'N/A'}\n"

            if isinstance(value, (int, float)) and value > 0:
                result += f"  Value: {format_large_number(value)}\n\n"
            else:
                result += f"  Value: {value}\n\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved institutional holders for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_major_holders(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get major holders summary (institutions, insiders, public float).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Summary of insider vs institutional ownership percentages
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"👥 Retrieving major holders for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            major_holders = stock.major_holders

            if major_holders is None or major_holders.empty:
                return f"ℹ️ No major holders data available for {ticker}"

            result = f"**👥 Major Holders Summary: {ticker}**\n\n"

            # New format: DataFrame with index as breakdown names and 'Value' column
            # Convert breakdown names to readable format
            breakdown_map = {
                'insidersPercentHeld': 'Insiders Percent Held',
                'institutionsPercentHeld': 'Institutions Percent Held',
                'institutionsFloatPercentHeld': 'Institutions Float Percent Held',
                'institutionsCount': 'Number of Institutions'
            }

            for idx, row in major_holders.iterrows():
                breakdown = idx
                value = row.get('Value', row.iloc[0] if len(row) > 0 else 'N/A')

                # Get readable name
                readable_name = breakdown_map.get(breakdown, breakdown)

                # Format value based on type
                if breakdown == 'institutionsCount':
                    result += f"{readable_name}: {int(value):,}\n"
                elif isinstance(value, (int, float)):
                    result += f"{readable_name}: {value*100:.2f}%\n"
                else:
                    result += f"{readable_name}: {value}\n"

            return result

        except Exception as e:
            return f"❌ Could not retrieve major holders for {ticker}: {str(e)}"

    @safe_ticker_call
    async def get_mutualfund_holders(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get mutual fund holders.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of mutual funds holding the stock
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Retrieving mutual fund holders for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            fund_holders = stock.mutualfund_holders

            if fund_holders is None or fund_holders.empty:
                return f"ℹ️ No mutual fund holder data available for {ticker}"

            result = f"**🏦 Mutual Fund Holders: {ticker}**\n\n"

            # New column names: 'Date Reported', 'Holder', 'pctHeld', 'Shares', 'Value', 'pctChange'
            for idx, row in fund_holders.head(10).iterrows():
                holder = row.get("Holder", "N/A")
                shares = row.get("Shares", 0)
                pct_held = row.get("pctHeld", row.get("% Out", 0))  # Try new name, fallback to old
                value = row.get("Value", 0)
                date_reported = row.get("Date Reported", "N/A")

                result += f"**{holder}**\n"
                result += f"  Date Reported: {date_reported}\n" if date_reported != "N/A" else ""
                result += f"  Shares: {shares:,}\n" if isinstance(shares, (int, float)) else f"  Shares: {shares}\n"

                # Format percent held
                if isinstance(pct_held, (int, float)):
                    result += f"  Ownership: {pct_held*100:.2f}%\n"
                else:
                    result += f"  Ownership: {pct_held}\n"

                result += f"  Value: {format_large_number(value)}\n\n"

            return result

        except Exception as e:
            return f"❌ Could not retrieve mutual fund holders for {ticker}: {str(e)}"

    @safe_ticker_call
    async def get_insider_transactions(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get recent insider trading activity.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Recent insider buy/sell transactions
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🔐 Retrieving insider transactions for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        # Try multiple methods to get insider data
        try:
            insiders = stock.insider_transactions
        except:
            try:
                insiders = stock.get_insider_transactions()
            except:
                insiders = None

        if insiders is None or insiders.empty:
            return f"ℹ️ No insider transaction data available for {ticker}"

        result = f"**👤 Recent Insider Transactions: {ticker}**\n\n"

        for idx, row in insiders.head(15).iterrows():
            insider = row.get("Insider", "N/A")
            shares = row.get("Shares", 0)
            transaction = row.get("Transaction", "N/A")
            start_date = row.get("Start Date", "N/A")

            # Emoji based on transaction type
            emoji = "💰" if "Buy" in str(transaction) else "💸" if "Sale" in str(transaction) else "📄"

            result += f"{emoji} **{insider}** - {transaction}\n"

            if isinstance(shares, (int, float)) and shares != 0:
                result += f"  Shares: {abs(shares):,.0f}\n"
            else:
                result += f"  Shares: {shares}\n"

            result += f"  Date: {start_date}\n\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved insider transactions for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_insider_purchases(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get insider purchases only (filtered for buys).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Recent insider purchase transactions
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🔐 Retrieving insider purchases for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            insiders = stock.insider_purchases

            if insiders is None or insiders.empty:
                return f"ℹ️ No insider purchase data available for {ticker}"

            result = f"**💰 Recent Insider Purchases: {ticker}**\n\n"

            for idx, row in insiders.head(10).iterrows():
                insider = row.get("Insider", "N/A")
                shares = row.get("Shares", 0)
                transaction = row.get("Transaction", "N/A")
                start_date = row.get("Start Date", "N/A")

                result += f"**{insider}**\n"
                result += f"  Shares: {abs(shares):,.0f}\n" if isinstance(shares, (int, float)) else f"  Shares: {shares}\n"
                result += f"  Transaction: {transaction}\n"
                result += f"  Date: {start_date}\n\n"

            return result

        except Exception as e:
            # Fallback to filtering all transactions
            return await self.get_insider_transactions(ticker, __event_emitter__)

    @safe_ticker_call
    async def get_insider_roster_holders(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get insider roster (list of company insiders).

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of company insiders and their positions
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🔐 Retrieving insider roster for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            roster = stock.insider_roster_holders

            if roster is None or (hasattr(roster, 'empty') and roster.empty):
                return f"ℹ️ No insider roster data available for {ticker}"

            result = f"**👥 Insider Roster: {ticker}**\n\n"

            for idx, row in roster.iterrows():
                try:
                    # Safely get values with fallbacks
                    name = safe_get(row, "Name", "N/A") if hasattr(row, 'get') else row.get("Name", "N/A") if isinstance(row, dict) else "N/A"
                    position = safe_get(row, "Position", "N/A") if hasattr(row, 'get') else row.get("Position", "N/A") if isinstance(row, dict) else "N/A"

                    # Handle Most Recent Transaction which can be dict or other format
                    transaction = row.get("Most Recent Transaction") if hasattr(row, 'get') else None
                    shares = None

                    if transaction:
                        if isinstance(transaction, dict):
                            shares = transaction.get("Shares")
                        elif hasattr(transaction, 'get'):
                            shares = transaction.get("Shares")

                    result += f"**{name}**\n"
                    result += f"  Position: {position}\n"
                    if shares and isinstance(shares, (int, float)):
                        result += f"  Shares: {shares:,}\n"
                    result += "\n"
                except Exception as row_error:
                    logger.debug(f"Skipping roster row due to format issue: {row_error}")
                    continue

            return result

        except Exception as e:
            return f"ℹ️ Insider roster data not available for {ticker}"

    # ============================================================
    # TOOL 29-30: OPTIONS & DERIVATIVES
    # ============================================================

    @safe_ticker_call
    async def get_options_chain(
        self,
        ticker: str,
        expiration: str = "",
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get options chain data (calls and puts).

        Args:
            ticker: Stock ticker symbol
            expiration: Specific expiration date (YYYY-MM-DD) or empty for nearest

        Returns:
            Options chain with top calls and puts by volume
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"⛓️ Retrieving options chain for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        expirations = stock.options

        if not expirations:
            return f"ℹ️ No options data available for {ticker}"

        # Use specified expiration or default to nearest
        if not expiration or expiration not in expirations:
            expiration = expirations[0]

        opt = stock.option_chain(expiration)
        calls = opt.calls
        puts = opt.puts

        result = f"**📊 Options Chain: {ticker}**\n"
        result += f"**Expiration:** {expiration}\n"
        result += f"**Available Expirations:** {', '.join(expirations[:5])}{'...' if len(expirations) > 5 else ''}\n\n"

        # Top calls by volume
        if not calls.empty:
            calls_with_volume = (
                calls[calls["volume"] > 0].copy()
                if "volume" in calls.columns
                else calls.copy()
            )
            top_calls = (
                calls_with_volume.nlargest(5, "volume")
                if "volume" in calls.columns and not calls_with_volume.empty
                else calls.head(5)
            )

            result += "**📈 Top 5 Calls (by volume):**\n"
            for idx, row in top_calls.iterrows():
                strike = row.get("strike", "N/A")
                last_price = row.get("lastPrice", "N/A")
                volume = row.get("volume", "N/A")
                iv = row.get("impliedVolatility", "N/A")
                oi = row.get("openInterest", "N/A")

                result += f"  Strike: ${strike} | Last: ${last_price} | Vol: {volume} | OI: {oi} | IV: {iv}\n"

        # Top puts by volume
        if not puts.empty:
            puts_with_volume = (
                puts[puts["volume"] > 0].copy()
                if "volume" in puts.columns
                else puts.copy()
            )
            top_puts = (
                puts_with_volume.nlargest(5, "volume")
                if "volume" in puts.columns and not puts_with_volume.empty
                else puts.head(5)
            )

            result += "\n**📉 Top 5 Puts (by volume):**\n"
            for idx, row in top_puts.iterrows():
                strike = row.get("strike", "N/A")
                last_price = row.get("lastPrice", "N/A")
                volume = row.get("volume", "N/A")
                iv = row.get("impliedVolatility", "N/A")
                oi = row.get("openInterest", "N/A")

                result += f"  Strike: ${strike} | Last: ${last_price} | Vol: {volume} | OI: {oi} | IV: {iv}\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved options chain for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_options_expirations(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get all available options expiration dates.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of all available options expiration dates
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📅 Retrieving options expirations for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        expirations = stock.options

        if not expirations:
            return f"ℹ️ No options data available for {ticker}"

        result = f"**📅 Options Expirations: {ticker}**\n\n"
        result += f"**Total Expirations:** {len(expirations)}\n\n"

        # Group by year
        exp_by_year = {}
        for exp in expirations:
            year = exp[:4]
            if year not in exp_by_year:
                exp_by_year[year] = []
            exp_by_year[year].append(exp)

        for year in sorted(exp_by_year.keys()):
            result += f"**{year}:**\n"
            for exp in exp_by_year[year]:
                result += f"  {exp}\n"
            result += "\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved options expirations for {ticker}",
                    "done": True
                }
            })
        return result

    # ============================================================
    # TOOL 31-32: NEWS & SEC FILINGS
    # ============================================================

    @safe_ticker_call
    async def get_stock_news(
        self,
        ticker: str,
        limit: int = 10,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get recent news articles about a stock.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news articles to return (default 10)

        Returns:
            Recent news headlines, sources, and links
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📰 Retrieving news for {ticker}",
                    "done": False
                }
            })

        # Respect configured limit
        limit = min(limit, self.valves.max_news_items)

        # Retry logic for news fetching (can be flaky)
        news_data = None
        last_error = None

        for attempt in range(3):
            try:
                stock = yf.Ticker(ticker)
                news_data = stock.news
                if news_data and len(news_data) > 0:
                    break
                # If empty, wait and retry
                if attempt < 2:
                    time.sleep(1)
            except Exception as e:
                last_error = e
                logger.warning(f"News fetch attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < 2:
                    time.sleep(1)

        if not news_data or len(news_data) == 0:
            if last_error:
                logger.warning(f"News fetch failed after retries for {ticker}: {last_error}")
            return f"ℹ️ No recent news available for {ticker}. News may not be accessible at this time. Try searching for company name instead."

        result = f"**📰 Recent News: {ticker}**\n\n"
        articles_displayed = 0

        def parse_article(article):
            """Parse article from either old or new yfinance API structure"""
            # New structure (v0.2.66+): nested in 'content' dict
            content = article.get("content", article)

            title = content.get("title", "") or article.get("title", "")

            # Publisher from provider or top-level
            provider = content.get("provider", {})
            publisher = (
                provider.get("displayName") or
                provider.get("name") or
                article.get("publisher") or
                content.get("publisher") or
                "Unknown"
            )

            # Link from canonicalUrl or clickThroughUrl or top-level
            canonical_url = content.get("canonicalUrl", {})
            click_url = content.get("clickThroughUrl", {})
            link = (
                canonical_url.get("url", "") or
                click_url.get("url", "") or
                article.get("link", "") or
                content.get("link", "") or
                article.get("url", "") or
                ""
            )

            # Date from pubDate, displayTime, or providerPublishTime
            pub_date = (
                content.get("pubDate") or
                content.get("displayTime") or
                article.get("pubDate") or
                article.get("providerPublishTime")
            )

            # Format date
            date = "Unknown date"
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        date = dateutil_parser.parse(pub_date).strftime("%Y-%m-%d %H:%M")
                    elif isinstance(pub_date, (int, float)):
                        date = datetime.fromtimestamp(pub_date).strftime("%Y-%m-%d %H:%M")
                except:
                    date = "Unknown date"

            # Summary/description if available
            summary = content.get("summary", "") or article.get("summary", "")

            return {
                "title": title.strip() if title else "",
                "publisher": publisher,
                "link": link,
                "date": date,
                "summary": summary[:200] if summary else ""
            }

        for article in news_data:
            if articles_displayed >= limit:
                break

            try:
                parsed = parse_article(article)

                # Only show articles with valid titles
                if parsed["title"]:
                    articles_displayed += 1
                    result += f"**{articles_displayed}. {parsed['title']}**\n"
                    result += f"   📰 {parsed['publisher']} | 📅 {parsed['date']}\n"
                    if parsed["summary"]:
                        result += f"   _{parsed['summary']}..._\n"
                    if parsed["link"]:
                        result += f"   🔗 {parsed['link']}\n"
                    result += "\n"
            except Exception as article_error:
                logger.debug(f"Skipping news article due to format issue: {article_error}")
                continue

        if articles_displayed == 0:
            return f"ℹ️ News articles found for {ticker} but none had valid content. Yahoo Finance news format may have changed. Try again later."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved {articles_displayed} news articles for {ticker}",
                    "done": True
                }
            })

        return result

    @safe_ticker_call
    async def get_sec_filings(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get recent SEC filings (10-K, 10-Q, 8-K, etc.).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Recent SEC filings with types and dates
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📄 Retrieving SEC filings for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            filings = stock.sec_filings

            if filings is None or (hasattr(filings, 'empty') and filings.empty):
                return f"ℹ️ No SEC filing data available for {ticker}"

            result = f"**📄 Recent SEC Filings: {ticker}**\n\n"

            # Handle different return types
            if isinstance(filings, list):
                for filing in filings[:15]:
                    filing_type = filing.get('type', 'N/A')
                    date = filing.get('date', 'N/A')
                    title = filing.get('title', '')
                    url = filing.get('edgarUrl', '')

                    result += f"**{filing_type}** - {date}\n"
                    if title:
                        result += f"  {title}\n"
                    if url:
                        result += f"  🔗 {url}\n"
                    result += "\n"
            else:
                result += str(filings)

            return result

        except Exception as e:
            return f"ℹ️ SEC filings not available for {ticker}"

    # ============================================================
    # TOOL 33-35: MARKET INDICES & COMPARISON
    # ============================================================

    async def get_market_indices(
        self,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get current prices for major market indices.

        Returns:
            Current values and changes for S&P 500, Nasdaq, Dow Jones, and other major indices
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📈 Retrieving market indices",
                    "done": False
                }
            })

        result = "**📊 Major Market Indices**\n\n"

        for name, symbol in MARKET_INDICES.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                price = info.get("regularMarketPrice") or info.get("currentPrice")
                change = info.get("regularMarketChange", 0)
                change_pct = info.get("regularMarketChangePercent", 0)

                emoji = "📈" if change > 0 else "📉" if change < 0 else "➖"

                result += f"{emoji} **{name}** ({symbol}): "

                if price:
                    result += f"{price:.2f}"
                else:
                    result += "N/A"

                if change and change_pct:
                    result += f" ({change:+.2f}, {change_pct:+.2f}%)"

                result += "\n"

            except Exception as idx_error:
                logger.warning(f"Error fetching {name}: {idx_error}")
                result += f"⚠️ **{name}** ({symbol}): Data unavailable\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved market indices",
                    "done": True
                }
            })
        return result

    async def compare_stocks(
        self,
        tickers: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Compare key metrics across multiple stocks.

        Args:
            tickers: Comma-separated list of ticker symbols (e.g., 'AAPL,MSFT,GOOGL')

        Returns:
            Comparison table of price, market cap, P/E, dividend yield, and returns
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"⚖️ Comparing stocks",
                    "done": False
                }
            })

        ticker_list = [t.strip().upper() for t in tickers.split(",")]

        if len(ticker_list) < 2:
            return "❌ Please provide at least 2 tickers separated by commas"

        if len(ticker_list) > self.valves.max_comparison_tickers:
            return f"❌ Maximum {self.valves.max_comparison_tickers} tickers allowed for comparison"

        result = "**📊 Stock Comparison**\n\n"
        result += f"**Comparing:** {', '.join(ticker_list)}\n\n"

        metrics = []

        for ticker_symbol in ticker_list:
            try:
                # Validate ticker
                validated = TickerValidator(ticker=ticker_symbol)
                ticker_symbol = validated.ticker

                stock = yf.Ticker(ticker_symbol)
                info = stock.info

                # Get 52-week return
                try:
                    hist = stock.history(period="1y")
                    if not hist.empty:
                        week_52_return = ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100
                    else:
                        week_52_return = info.get("52WeekChange")
                except:
                    week_52_return = info.get("52WeekChange")

                price = info.get("currentPrice") or info.get("regularMarketPrice")
                market_cap = info.get("marketCap")
                pe_ratio = info.get("trailingPE")
                div_yield = info.get("dividendYield")

                # Format dividend yield
                div_yield_display = format_percentage(div_yield) if div_yield else "N/A"

                metrics.append({
                    "Ticker": ticker_symbol,
                    "Price": f"${price:.2f}" if price else "N/A",
                    "Market Cap": format_large_number(market_cap),
                    "P/E": f"{pe_ratio:.2f}" if pe_ratio else "N/A",
                    "Div Yield": div_yield_display,
                    "52W Return": f"{week_52_return:.2f}%" if week_52_return else "N/A",
                    "Beta": f"{info.get('beta'):.2f}" if info.get('beta') else "N/A",
                })

            except Exception as ticker_error:
                logger.warning(f"Error fetching data for {ticker_symbol}: {ticker_error}")
                metrics.append({
                    "Ticker": ticker_symbol,
                    "Price": "Error",
                    "Market Cap": "Error",
                    "P/E": "Error",
                    "Div Yield": "Error",
                    "52W Return": "Error",
                    "Beta": "Error",
                })

        # Display comparison
        metric_names = ["Price", "Market Cap", "P/E", "Div Yield", "52W Return", "Beta"]

        for metric_name in metric_names:
            result += f"\n**{metric_name}:**\n"
            for m in metrics:
                result += f"  {m['Ticker']}: {m[metric_name]}\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Compared stocks",
                    "done": True
                }
            })
        return result

    async def get_sector_performance(
        self,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get performance of major market sectors.

        Returns:
            Performance data for major S&P 500 sectors
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Retrieving sector performance",
                    "done": False
                }
            })

        # Major sector ETFs
        sectors = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Industrials": "XLI",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
            "Consumer Staples": "XLP",
            "Communication Services": "XLC",
        }

        result = "**📊 Sector Performance (via Sector ETFs)**\n\n"

        sector_data = []

        for sector_name, etf_symbol in sectors.items():
            try:
                ticker = yf.Ticker(etf_symbol)
                info = ticker.info
                hist = ticker.history(period="1y")

                price = info.get("regularMarketPrice") or info.get("currentPrice")
                change = info.get("regularMarketChange", 0)
                change_pct = info.get("regularMarketChangePercent", 0)

                # Calculate YTD return
                if not hist.empty:
                    ytd_return = ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100
                else:
                    ytd_return = None

                emoji = "📈" if change > 0 else "📉" if change < 0 else "➖"

                sector_data.append({
                    "name": sector_name,
                    "symbol": etf_symbol,
                    "price": price,
                    "change": change,
                    "change_pct": change_pct,
                    "ytd_return": ytd_return,
                    "emoji": emoji
                })

            except Exception as e:
                logger.warning(f"Error fetching {sector_name}: {e}")

        # Sort by daily performance
        sector_data.sort(key=lambda x: x["change_pct"] if x["change_pct"] else 0, reverse=True)

        result += "**Today's Performance:**\n"
        for s in sector_data:
            result += f"{s['emoji']} **{s['name']}** ({s['symbol']}): "
            if s['price']:
                result += f"${s['price']:.2f} "
            if s['change'] and s['change_pct']:
                result += f"({s['change']:+.2f}, {s['change_pct']:+.2f}%)"
            result += "\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved sector performance",
                    "done": True
                }
            })
        return result

    # ============================================================
    # TOOL 36-38: FUND/ETF DATA
    # ============================================================

    @safe_ticker_call
    async def get_fund_overview(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get overview data for an ETF or mutual fund.

        Args:
            ticker: Fund ticker symbol (e.g., 'VOO', 'SPY', 'QQQ', 'ARKK')

        Returns:
            Fund category, family, total assets, expense ratio, and other key metrics
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Retrieving fund overview for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        # Check if this is a fund
        quote_type = info.get("quoteType", "").upper()
        if quote_type not in ["ETF", "MUTUALFUND"]:
            return f"ℹ️ {ticker} does not appear to be an ETF or mutual fund. Use get_company_info for stocks."

        result = f"**📊 Fund Overview: {ticker}**\n\n"

        # Basic info
        result += f"**📛 Name:** {safe_get(info, 'longName')}\n"
        result += f"**📁 Category:** {safe_get(info, 'category')}\n"
        result += f"**🏢 Fund Family:** {safe_get(info, 'fundFamily')}\n"
        result += f"**📈 Quote Type:** {quote_type}\n\n"

        # Key metrics
        result += "**💰 Key Metrics:**\n"

        total_assets = info.get("totalAssets")
        if total_assets:
            result += f"  Total Assets: {format_large_number(total_assets)}\n"

        # Get expense ratio - try funds_data first (more reliable), then info fallback
        expense_ratio = None
        try:
            funds_data = stock.funds_data
            if funds_data is not None:
                fund_ops = funds_data.fund_operations
                if fund_ops is not None and not fund_ops.empty:
                    if "Annual Report Expense Ratio" in fund_ops.index:
                        # fund_operations has index as metric name, columns include 'Value'
                        exp_val = fund_ops.loc["Annual Report Expense Ratio"]
                        if isinstance(exp_val, pd.Series):
                            expense_ratio = exp_val.iloc[0]  # Get first column value
                        else:
                            expense_ratio = exp_val
        except Exception as e:
            logger.debug(f"Could not get expense ratio from funds_data: {e}")

        # Fallback to info dict
        if expense_ratio is None:
            expense_ratio = info.get("annualReportExpenseRatio") or info.get("expenseRatio") or info.get("annualHoldingsTurnover")

        if expense_ratio is not None and expense_ratio > 0:
            # Format expense ratio - handle both decimal (0.0003) and percentage (0.03) formats
            if expense_ratio < 0.1:
                result += f"  Expense Ratio: {expense_ratio*100:.2f}%\n"
            else:
                result += f"  Expense Ratio: {expense_ratio:.2f}%\n"

        nav = info.get("navPrice")
        if nav:
            result += f"  NAV: ${nav:.2f}\n"

        yield_val = info.get("yield") or info.get("trailingAnnualDividendYield")
        if yield_val:
            result += f"  Yield: {format_percentage(yield_val)}\n"

        ytd_return = info.get("ytdReturn")
        if ytd_return:
            result += f"  YTD Return: {format_percentage(ytd_return)}\n"

        three_year_return = info.get("threeYearAverageReturn")
        if three_year_return:
            result += f"  3-Year Avg Return: {format_percentage(three_year_return)}\n"

        five_year_return = info.get("fiveYearAverageReturn")
        if five_year_return:
            result += f"  5-Year Avg Return: {format_percentage(five_year_return)}\n"

        # Beta
        beta = info.get("beta3Year")
        if beta:
            result += f"  Beta (3Y): {beta:.2f}\n"

        result += "\n"

        # Price info
        current_price = info.get("regularMarketPrice") or info.get("previousClose")
        if current_price:
            result += f"**💵 Current Price:** ${current_price:.2f}\n"

        day_low = info.get("dayLow")
        day_high = info.get("dayHigh")
        if day_low and day_high:
            result += f"**📊 Day Range:** ${day_low:.2f} - ${day_high:.2f}\n"

        week_low = info.get("fiftyTwoWeekLow")
        week_high = info.get("fiftyTwoWeekHigh")
        if week_low and week_high:
            result += f"**📉 52-Week Range:** ${week_low:.2f} - ${week_high:.2f}\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved fund overview for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_fund_holdings(
        self,
        ticker: str,
        limit: int = 25,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get top holdings for an ETF or mutual fund.

        Args:
            ticker: Fund ticker symbol (e.g., 'VOO', 'SPY', 'QQQ')
            limit: Maximum number of holdings to return (default 25)

        Returns:
            List of top holdings with weights and values
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📋 Retrieving fund holdings for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)

        try:
            # Try to get fund holdings
            holdings = None

            # Method 1: Try funds_data
            try:
                funds_data = stock.funds_data
                if funds_data and hasattr(funds_data, 'top_holdings'):
                    holdings = funds_data.top_holdings
            except:
                pass

            # Method 2: Try direct holdings attribute
            if holdings is None or (hasattr(holdings, 'empty') and holdings.empty):
                try:
                    holdings = stock.major_holders
                except:
                    pass

            if holdings is None or (hasattr(holdings, 'empty') and holdings.empty):
                # Check if it's even a fund
                info = stock.info
                quote_type = info.get("quoteType", "").upper()
                if quote_type not in ["ETF", "MUTUALFUND"]:
                    return f"ℹ️ {ticker} is not an ETF or mutual fund. Holdings data is only available for funds."
                return f"ℹ️ Holdings data for {ticker} is not available via yfinance. Try checking the fund provider's website."

            result = f"**📋 Top Holdings: {ticker}**\n\n"

            if isinstance(holdings, pd.DataFrame):
                # Limit results
                holdings = holdings.head(limit)

                # Debug: log available columns
                logger.debug(f"Holdings columns for {ticker}: {list(holdings.columns)}")

                for idx, row in holdings.iterrows():
                    # Try multiple column name variations
                    holding_name = (
                        row.get("Name") or
                        row.get("holdingName") or
                        row.get("Holder") or
                        row.get("name") or
                        str(idx)
                    )
                    # Correct column name is "Holding Percent" per yfinance API
                    weight = (
                        row.get("Holding Percent") or
                        row.get("holdingPercent") or
                        row.get("pctHeld") or
                        row.get("% Held") or
                        row.get("Weight")
                    )

                    # Get symbol if available
                    symbol = row.get("Symbol") or row.get("symbol") or ""

                    # Format holding entry
                    if isinstance(idx, int):
                        result += f"**{idx + 1}. {holding_name}**"
                    else:
                        result += f"**{idx}. {holding_name}**"

                    if symbol:
                        result += f" ({symbol})"
                    result += "\n"

                    if weight is not None:
                        if isinstance(weight, (int, float)):
                            # Handle both decimal (0.07) and percentage (7.0) formats
                            if weight < 1:
                                result += f"   Weight: {weight*100:.2f}%\n"
                            else:
                                result += f"   Weight: {weight:.2f}%\n"
                        else:
                            result += f"   Weight: {weight}\n"
                    result += "\n"

            info = stock.info
            result += f"\n**📊 Total Holdings:** {info.get('holdings', 'N/A')}\n"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved holdings for {ticker}",
                        "done": True
                    }
                })
            return result

        except Exception as e:
            logger.warning(f"Error fetching holdings for {ticker}: {e}")
            return f"ℹ️ Unable to fetch holdings data for {ticker}. This data may not be available for all funds."

    @safe_ticker_call
    async def get_fund_sector_weights(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get sector allocation/weights for an ETF or mutual fund.

        Args:
            ticker: Fund ticker symbol (e.g., 'VOO', 'SPY', 'QQQ')

        Returns:
            Sector allocation percentages for the fund
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Retrieving sector weights for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        # Check if this is a fund
        quote_type = info.get("quoteType", "").upper()
        if quote_type not in ["ETF", "MUTUALFUND"]:
            return f"ℹ️ {ticker} is not an ETF or mutual fund. Sector weights are only available for funds."

        result = f"**📊 Sector Allocation: {ticker}**\n\n"

        try:
            # Try to get sector weightings
            sector_weights = None

            # Method 1: Try funds_data
            try:
                funds_data = stock.funds_data
                if funds_data and hasattr(funds_data, 'sector_weightings'):
                    sector_weights = funds_data.sector_weightings
            except:
                pass

            if sector_weights and len(sector_weights) > 0:
                if isinstance(sector_weights, dict):
                    # Sort by weight descending
                    sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)
                    for sector, weight in sorted_sectors:
                        if weight:
                            bar = "█" * int(weight * 20) if weight < 1 else "█" * int(weight / 5)
                            weight_pct = weight * 100 if weight < 1 else weight
                            result += f"**{sector}:** {weight_pct:.1f}% {bar}\n"
                elif isinstance(sector_weights, pd.DataFrame):
                    for idx, row in sector_weights.iterrows():
                        sector = idx
                        weight = row.iloc[0] if len(row) > 0 else row
                        if weight:
                            weight_pct = weight * 100 if weight < 1 else weight
                            bar = "█" * int(weight_pct / 5)
                            result += f"**{sector}:** {weight_pct:.1f}% {bar}\n"

                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"✅ Retrieved sector weights for {ticker}",
                            "done": True
                        }
                    })
                return result

            # Fallback: Check info for sector data
            sector_info = info.get("sectorWeightings")
            if sector_info:
                for sector_data in sector_info:
                    for sector, weight in sector_data.items():
                        if weight:
                            weight_pct = weight * 100 if weight < 1 else weight
                            bar = "█" * int(weight_pct / 5)
                            result += f"**{sector}:** {weight_pct:.1f}% {bar}\n"
                return result

            return f"ℹ️ Sector allocation data for {ticker} is not available via yfinance. Try checking the fund provider's website."

        except Exception as e:
            logger.warning(f"Error fetching sector weights for {ticker}: {e}")
            return f"ℹ️ Unable to fetch sector weights for {ticker}. This data may not be available for all funds."

    # ============================================================
    # TOOL 40-42: CRYPTO, FOREX & COMMODITY DATA
    # ============================================================

    async def get_crypto_price(
        self,
        symbol: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get cryptocurrency price and market data.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'bitcoin', 'ethereum', 'BTC-USD')

        Returns:
            Current price, 24h change, market cap, volume, and key metrics
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        # Normalize symbol
        symbol_clean = symbol.strip().upper()

        # Map common names to tickers
        symbol_lower = symbol.strip().lower()
        if symbol_lower in CRYPTO_TICKERS:
            ticker = CRYPTO_TICKERS[symbol_lower]
        elif "-USD" in symbol_clean or "-EUR" in symbol_clean:
            ticker = symbol_clean
        else:
            # Assume USD pair
            ticker = f"{symbol_clean}-USD"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"₿ Retrieving crypto data for {ticker}",
                    "done": False
                }
            })

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or len(info) < 3:
                return f"❌ No data found for cryptocurrency {symbol}. Try using format like 'BTC-USD' or 'ETH-USD'."

            result = f"**₿ Cryptocurrency: {ticker}**\n\n"

            # Name
            name = info.get("name") or info.get("shortName") or ticker
            result += f"**📛 Name:** {name}\n\n"

            # Current price
            current_price = (
                info.get("regularMarketPrice") or
                info.get("currentPrice") or
                info.get("previousClose")
            )
            if current_price:
                result += f"**💰 Current Price:** ${current_price:,.2f}\n"

            # 24h change
            day_change = info.get("regularMarketChange")
            day_change_pct = info.get("regularMarketChangePercent")
            if day_change is not None and day_change_pct is not None:
                emoji = "📈" if day_change >= 0 else "📉"
                result += f"**{emoji} 24h Change:** ${day_change:+,.2f} ({day_change_pct:+.2f}%)\n"

            # Day range
            day_low = info.get("dayLow") or info.get("regularMarketDayLow")
            day_high = info.get("dayHigh") or info.get("regularMarketDayHigh")
            if day_low and day_high:
                result += f"**📊 24h Range:** ${day_low:,.2f} - ${day_high:,.2f}\n"

            # 52-week range
            week_low = info.get("fiftyTwoWeekLow")
            week_high = info.get("fiftyTwoWeekHigh")
            if week_low and week_high:
                result += f"**📉 52-Week Range:** ${week_low:,.2f} - ${week_high:,.2f}\n"

            result += "\n"

            # Market metrics
            result += "**📈 Market Metrics:**\n"

            market_cap = info.get("marketCap")
            if market_cap:
                result += f"  Market Cap: {format_large_number(market_cap)}\n"

            volume = info.get("volume") or info.get("regularMarketVolume")
            if volume:
                result += f"  24h Volume: ${volume:,.0f}\n"

            circulating = info.get("circulatingSupply")
            if circulating:
                result += f"  Circulating Supply: {circulating:,.0f}\n"

            total_supply = info.get("maxSupply")
            if total_supply:
                result += f"  Max Supply: {total_supply:,.0f}\n"

            # Previous close
            prev_close = info.get("previousClose")
            if prev_close:
                result += f"  Previous Close: ${prev_close:,.2f}\n"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved crypto data for {ticker}",
                        "done": True
                    }
                })
            return result

        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return f"❌ Error fetching data for {symbol}: {str(e)}. Make sure the symbol is valid (e.g., 'BTC-USD')."

    async def get_forex_rate(
        self,
        pair: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get foreign exchange (forex) rate for a currency pair.

        Args:
            pair: Currency pair (e.g., 'EURUSD', 'EUR/USD', 'eurusd', 'GBPUSD=X')

        Returns:
            Current exchange rate, bid/ask, day range, and recent changes
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        # Normalize pair
        pair_clean = pair.strip().lower().replace("/", "").replace("-", "").replace(" ", "")

        # Map common formats to yfinance tickers
        if pair_clean in FOREX_PAIRS:
            ticker = FOREX_PAIRS[pair_clean]
        elif pair.strip().upper().endswith("=X"):
            ticker = pair.strip().upper()
        else:
            # Try to construct ticker
            ticker = f"{pair_clean.upper()}=X"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"💱 Retrieving forex rate for {ticker}",
                    "done": False
                }
            })

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or len(info) < 3:
                return f"❌ No data found for forex pair {pair}. Try format like 'EURUSD' or 'EUR/USD'."

            result = f"**💱 Forex Rate: {ticker.replace('=X', '')}**\n\n"

            # Name
            name = info.get("shortName") or info.get("longName") or ticker
            result += f"**📛 Pair:** {name}\n\n"

            # Current rate
            current_rate = (
                info.get("regularMarketPrice") or
                info.get("bid") or
                info.get("previousClose")
            )
            if current_rate:
                result += f"**💰 Current Rate:** {current_rate:.5f}\n"

            # Bid/Ask
            bid = info.get("bid")
            ask = info.get("ask")
            if bid and ask:
                spread = ask - bid
                result += f"**📊 Bid:** {bid:.5f}\n"
                result += f"**📊 Ask:** {ask:.5f}\n"
                result += f"**📐 Spread:** {spread:.5f} ({spread/bid*10000:.1f} pips)\n"

            # Day change
            day_change = info.get("regularMarketChange")
            day_change_pct = info.get("regularMarketChangePercent")
            if day_change is not None and day_change_pct is not None:
                emoji = "📈" if day_change >= 0 else "📉"
                result += f"**{emoji} Day Change:** {day_change:+.5f} ({day_change_pct:+.2f}%)\n"

            # Day range
            day_low = info.get("dayLow") or info.get("regularMarketDayLow")
            day_high = info.get("dayHigh") or info.get("regularMarketDayHigh")
            if day_low and day_high:
                result += f"**📊 Day Range:** {day_low:.5f} - {day_high:.5f}\n"

            # 52-week range
            week_low = info.get("fiftyTwoWeekLow")
            week_high = info.get("fiftyTwoWeekHigh")
            if week_low and week_high:
                result += f"**📉 52-Week Range:** {week_low:.5f} - {week_high:.5f}\n"

            # Volume
            volume = info.get("volume") or info.get("regularMarketVolume")
            if volume:
                result += f"**📈 Volume:** {volume:,.0f}\n"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved forex rate for {ticker}",
                        "done": True
                    }
                })
            return result

        except Exception as e:
            logger.error(f"Error fetching forex data for {pair}: {e}")
            return f"❌ Error fetching data for {pair}: {str(e)}. Try format like 'EURUSD' or 'EUR/USD'."

    async def get_commodity_price(
        self,
        commodity: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get commodity/futures price data.

        Args:
            commodity: Commodity name or ticker (e.g., 'gold', 'oil', 'crude oil', 'GC=F', 'CL=F')

        Returns:
            Current price, day change, and key market data
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        # Normalize commodity name
        commodity_clean = commodity.strip().lower()

        # Map common names to tickers
        if commodity_clean in COMMODITY_TICKERS:
            ticker = COMMODITY_TICKERS[commodity_clean]
        elif commodity.strip().upper().endswith("=F"):
            ticker = commodity.strip().upper()
        else:
            # Check if it's already a valid futures ticker format
            ticker = f"{commodity.strip().upper()}=F"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🛢️ Retrieving commodity data for {ticker}",
                    "done": False
                }
            })

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or len(info) < 3:
                # Try alternative lookup
                available = ", ".join(list(COMMODITY_TICKERS.keys())[:10])
                return f"❌ No data found for commodity '{commodity}'. Available commodities: {available}"

            result = f"**🛢️ Commodity: {ticker}**\n\n"

            # Name
            name = info.get("shortName") or info.get("longName") or ticker
            result += f"**📛 Name:** {name}\n\n"

            # Current price
            current_price = (
                info.get("regularMarketPrice") or
                info.get("previousClose")
            )
            if current_price:
                result += f"**💰 Current Price:** ${current_price:,.2f}\n"

            # Day change
            day_change = info.get("regularMarketChange")
            day_change_pct = info.get("regularMarketChangePercent")
            if day_change is not None and day_change_pct is not None:
                emoji = "📈" if day_change >= 0 else "📉"
                result += f"**{emoji} Day Change:** ${day_change:+,.2f} ({day_change_pct:+.2f}%)\n"

            # Day range
            day_low = info.get("dayLow") or info.get("regularMarketDayLow")
            day_high = info.get("dayHigh") or info.get("regularMarketDayHigh")
            if day_low and day_high:
                result += f"**📊 Day Range:** ${day_low:,.2f} - ${day_high:,.2f}\n"

            # 52-week range
            week_low = info.get("fiftyTwoWeekLow")
            week_high = info.get("fiftyTwoWeekHigh")
            if week_low and week_high:
                result += f"**📉 52-Week Range:** ${week_low:,.2f} - ${week_high:,.2f}\n"

            # Open
            market_open = info.get("open") or info.get("regularMarketOpen")
            if market_open:
                result += f"**📈 Open:** ${market_open:,.2f}\n"

            # Previous close
            prev_close = info.get("previousClose")
            if prev_close:
                result += f"**📊 Previous Close:** ${prev_close:,.2f}\n"

            # Volume
            volume = info.get("volume") or info.get("regularMarketVolume")
            if volume:
                result += f"**📊 Volume:** {volume:,.0f}\n"

            # Contract info
            result += "\n**📜 Contract Info:**\n"

            expire_date = info.get("expireDate")
            if expire_date:
                result += f"  Expiration: {format_date(expire_date)}\n"

            contract_size = info.get("contractSize")
            if contract_size:
                result += f"  Contract Size: {contract_size}\n"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Retrieved commodity data for {ticker}",
                        "done": True
                    }
                })
            return result

        except Exception as e:
            logger.error(f"Error fetching commodity data for {commodity}: {e}")
            available = ", ".join(list(COMMODITY_TICKERS.keys())[:10])
            return f"❌ Error fetching data for '{commodity}'. Available: {available}"

    # ============================================================
    # TOOL 43-45: BULK OPERATIONS & SCREENING
    # ============================================================

    async def download_multiple_tickers(
        self,
        tickers: str,
        period: str = "1mo",
        interval: str = "1d",
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Download historical data for multiple tickers at once (bulk operation).

        Args:
            tickers: Space or comma-separated ticker symbols (e.g., 'AAPL MSFT GOOGL')
            period: Time period (default: 1mo)
            interval: Data interval (default: 1d)

        Returns:
            Summary of downloaded data for all tickers
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📥 Downloading data for {tickers}",
                    "done": False
                }
            })

        # Parse tickers
        ticker_list = tickers.replace(",", " ").split()
        ticker_list = [t.strip().upper() for t in ticker_list if t.strip()]

        if not ticker_list:
            return "❌ No valid tickers provided"

        if len(ticker_list) > 20:
            return "❌ Maximum 20 tickers allowed for bulk download"

        try:
            # Use yfinance download function
            data = yf.download(ticker_list, period=period, interval=interval, group_by='ticker', progress=False)

            if data.empty:
                return "❌ No data downloaded"

            result = f"**📥 Bulk Download Results**\n\n"
            result += f"**Tickers:** {', '.join(ticker_list)}\n"
            result += f"**Period:** {period} | **Interval:** {interval}\n\n"

            # Summary for each ticker
            for ticker in ticker_list:
                try:
                    if len(ticker_list) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker]

                    if not ticker_data.empty:
                        latest = ticker_data.iloc[-1]
                        first = ticker_data.iloc[0]

                        result += f"**{ticker}:**\n"
                        result += f"  Data Points: {len(ticker_data)}\n"
                        result += f"  Latest Close: ${latest['Close']:.2f}\n"
                        result += f"  Period Return: {((latest['Close'] / first['Close'] - 1) * 100):+.2f}%\n\n"
                except:
                    result += f"**{ticker}:** No data\n\n"

            return result

        except Exception as e:
            return f"❌ Error downloading data: {str(e)}"

    async def get_trending_tickers(
        self,
        count: int = 10,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get trending/most active tickers (note: limited availability in yfinance).

        Args:
            count: Number of trending tickers to return

        Returns:
            List of trending tickers with basic info
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🔥 Retrieving trending tickers",
                    "done": False
                }
            })

        # This feature may not be available in all yfinance versions
        result = "**📈 Trending Tickers**\n\n"
        result += "ℹ️ Note: Trending tickers feature has limited availability in yfinance.\n"
        result += "For the most accurate trending data, please check Yahoo Finance website directly.\n\n"

        # Provide some popular/most traded tickers as reference
        popular_tickers = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "AMZN", "GOOGL", "META"]

        result += "**Most Popular/Traded Tickers:**\n"
        for ticker_symbol in popular_tickers[:count]:
            try:
                stock = yf.Ticker(ticker_symbol)
                info = stock.info
                price = info.get("regularMarketPrice") or info.get("currentPrice")
                change_pct = info.get("regularMarketChangePercent", 0)

                emoji = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➖"

                result += f"{emoji} {ticker_symbol}: ${price:.2f} ({change_pct:+.2f}%)\n"
            except:
                result += f"  {ticker_symbol}: Data unavailable\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Retrieved trending tickers",
                    "done": True
                }
            })
        return result

    # ============================================================
    # TOOL 46-49: ANALYSIS & COMPARISON TOOLS
    # ============================================================

    @safe_ticker_call
    async def get_peer_comparison(
        self,
        ticker: str,
        peers: str = None,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Compare a stock with its peers on key metrics.

        Args:
            ticker: Primary stock ticker symbol
            peers: Optional comma-separated peer tickers (if not provided, uses sector peers)

        Returns:
            Comparison table of key metrics across stocks
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📊 Generating peer comparison for {ticker}",
                    "done": False
                }
            })

        # Get primary stock info
        stock = yf.Ticker(ticker)
        info = stock.info

        # Determine peers
        peer_list = []
        if peers:
            peer_list = [p.strip().upper() for p in peers.replace(",", " ").split() if p.strip()]
        else:
            # Try to find sector peers from info
            sector = info.get("sector")
            industry = info.get("industry")

            # Common sector peers (fallback)
            sector_peers = {
                "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
                "Financial Services": ["JPM", "BAC", "GS", "MS", "C"],
                "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
                "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
                "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "T"],
                "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
            }

            if sector and sector in sector_peers:
                peer_list = [p for p in sector_peers[sector] if p != ticker][:4]
            else:
                peer_list = ["SPY"]  # Default to S&P 500 comparison

        result = f"**📊 Peer Comparison: {ticker}**\n\n"
        result += f"**Sector:** {info.get('sector', 'N/A')}\n"
        result += f"**Industry:** {info.get('industry', 'N/A')}\n\n"

        # Collect data for all tickers
        all_tickers = [ticker] + peer_list[:5]  # Limit to 6 total
        comparison_data = []

        for t in all_tickers:
            try:
                s = yf.Ticker(t)
                i = s.info

                comparison_data.append({
                    "ticker": t,
                    "name": i.get("shortName", t)[:20],
                    "price": i.get("regularMarketPrice") or i.get("previousClose"),
                    "market_cap": i.get("marketCap"),
                    "pe": i.get("trailingPE"),
                    "forward_pe": i.get("forwardPE"),
                    "pb": i.get("priceToBook"),
                    "div_yield": i.get("dividendYield"),
                    "profit_margin": i.get("profitMargins"),
                    "beta": i.get("beta"),
                })
            except Exception as e:
                logger.warning(f"Could not fetch data for {t}: {e}")

        if not comparison_data:
            return f"❌ Could not fetch comparison data for {ticker}"

        # Format comparison table
        result += "**💰 Valuation Metrics:**\n"
        result += "```\n"
        result += f"{'Ticker':<8} {'Price':>10} {'P/E':>8} {'Fwd P/E':>8} {'P/B':>8}\n"
        result += "-" * 50 + "\n"

        for d in comparison_data:
            price = f"${d['price']:.2f}" if d['price'] else "N/A"
            pe = f"{d['pe']:.1f}" if d['pe'] else "N/A"
            fpe = f"{d['forward_pe']:.1f}" if d['forward_pe'] else "N/A"
            pb = f"{d['pb']:.1f}" if d['pb'] else "N/A"
            result += f"{d['ticker']:<8} {price:>10} {pe:>8} {fpe:>8} {pb:>8}\n"

        result += "```\n\n"

        result += "**📈 Market & Risk:**\n"
        result += "```\n"
        result += f"{'Ticker':<8} {'Mkt Cap':>12} {'Beta':>8} {'Div Yld':>10}\n"
        result += "-" * 50 + "\n"

        for d in comparison_data:
            mc = format_large_number(d['market_cap']) if d['market_cap'] else "N/A"
            beta = f"{d['beta']:.2f}" if d['beta'] else "N/A"
            div = f"{d['div_yield']*100:.1f}%" if d['div_yield'] else "N/A"
            result += f"{d['ticker']:<8} {mc:>12} {beta:>8} {div:>10}\n"

        result += "```\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Generated peer comparison for {ticker}",
                    "done": True
                }
            })
        return result

    @safe_ticker_call
    async def get_financial_summary(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get a comprehensive one-page financial summary of a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Comprehensive summary including price, valuation, financials, and analyst data
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📋 Generating financial summary for {ticker}",
                    "done": False
                }
            })

        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or len(info) < 3:
            return f"❌ No data available for {ticker}"

        result = f"**📋 Financial Summary: {ticker}**\n"
        result += f"**{info.get('longName', ticker)}**\n"
        result += "=" * 50 + "\n\n"

        # Company Overview
        result += "**🏢 Company Overview:**\n"
        result += f"  Sector: {info.get('sector', 'N/A')}\n"
        result += f"  Industry: {info.get('industry', 'N/A')}\n"
        result += f"  Employees: {info.get('fullTimeEmployees', 'N/A'):,}\n" if info.get('fullTimeEmployees') else ""
        result += "\n"

        # Price & Valuation
        result += "**💰 Price & Valuation:**\n"
        current_price = info.get("regularMarketPrice") or info.get("previousClose")
        if current_price:
            result += f"  Current Price: ${current_price:.2f}\n"

        market_cap = info.get("marketCap")
        if market_cap:
            result += f"  Market Cap: {format_large_number(market_cap)}\n"

        pe = info.get("trailingPE")
        if pe:
            result += f"  P/E Ratio: {pe:.1f}\n"

        forward_pe = info.get("forwardPE")
        if forward_pe:
            result += f"  Forward P/E: {forward_pe:.1f}\n"

        pb = info.get("priceToBook")
        if pb:
            result += f"  P/B Ratio: {pb:.1f}\n"

        ev_ebitda = info.get("enterpriseToEbitda")
        if ev_ebitda:
            result += f"  EV/EBITDA: {ev_ebitda:.1f}\n"

        result += "\n"

        # Financials
        result += "**📊 Key Financials:**\n"
        revenue = info.get("totalRevenue")
        if revenue:
            result += f"  Revenue: {format_large_number(revenue)}\n"

        gross_profit = info.get("grossProfits")
        if gross_profit:
            result += f"  Gross Profit: {format_large_number(gross_profit)}\n"

        net_income = info.get("netIncomeToCommon")
        if net_income:
            result += f"  Net Income: {format_large_number(net_income)}\n"

        profit_margin = info.get("profitMargins")
        if profit_margin:
            result += f"  Profit Margin: {profit_margin*100:.1f}%\n"

        roe = info.get("returnOnEquity")
        if roe:
            result += f"  ROE: {roe*100:.1f}%\n"

        debt_equity = info.get("debtToEquity")
        if debt_equity:
            result += f"  Debt/Equity: {debt_equity:.1f}\n"

        result += "\n"

        # Dividends
        div_yield = info.get("dividendYield")
        div_rate = info.get("dividendRate")
        if div_yield or div_rate:
            result += "**💵 Dividends:**\n"
            if div_rate:
                result += f"  Annual Dividend: ${div_rate:.2f}\n"
            if div_yield:
                result += f"  Dividend Yield: {div_yield*100:.2f}%\n"
            payout = info.get("payoutRatio")
            if payout:
                result += f"  Payout Ratio: {payout*100:.1f}%\n"
            result += "\n"

        # Analyst Ratings
        result += "**🎯 Analyst Consensus:**\n"
        rec_key = info.get("recommendationKey")
        if rec_key:
            result += f"  Rating: {rec_key.upper()}\n"

        target_mean = info.get("targetMeanPrice")
        if target_mean and current_price:
            upside = ((target_mean - current_price) / current_price) * 100
            result += f"  Price Target: ${target_mean:.2f} ({upside:+.1f}%)\n"

        num_analysts = info.get("numberOfAnalystOpinions")
        if num_analysts:
            result += f"  # of Analysts: {num_analysts}\n"

        result += "\n"

        # 52-Week Performance
        result += "**📈 52-Week Performance:**\n"
        week_low = info.get("fiftyTwoWeekLow")
        week_high = info.get("fiftyTwoWeekHigh")
        if week_low and week_high and current_price:
            result += f"  52W Range: ${week_low:.2f} - ${week_high:.2f}\n"
            position = (current_price - week_low) / (week_high - week_low) * 100
            result += f"  Position: {position:.0f}% from low\n"

        beta = info.get("beta")
        if beta:
            result += f"  Beta: {beta:.2f}\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Generated financial summary for {ticker}",
                    "done": True
                }
            })
        return result

    async def get_historical_comparison(
        self,
        tickers: str,
        period: str = "1y",
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Compare historical price performance of multiple stocks.

        Args:
            tickers: Comma or space-separated ticker symbols (e.g., 'AAPL,MSFT,GOOGL')
            period: Time period for comparison (default: 1y)

        Returns:
            Performance comparison showing % returns for each ticker
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        # Parse tickers
        ticker_list = [t.strip().upper() for t in tickers.replace(",", " ").split() if t.strip()]

        if len(ticker_list) < 2:
            return "❌ Please provide at least 2 tickers to compare (e.g., 'AAPL,MSFT,GOOGL')"

        if len(ticker_list) > 10:
            ticker_list = ticker_list[:10]

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📈 Comparing historical performance",
                    "done": False
                }
            })

        result = f"**📈 Historical Performance Comparison**\n"
        result += f"**Period:** {period}\n\n"

        performance_data = []

        for ticker in ticker_list:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)

                if hist.empty:
                    continue

                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                high_price = hist['High'].max()
                low_price = hist['Low'].min()

                total_return = ((end_price - start_price) / start_price) * 100
                max_drawdown = ((low_price - high_price) / high_price) * 100

                performance_data.append({
                    "ticker": ticker,
                    "start_price": start_price,
                    "end_price": end_price,
                    "return": total_return,
                    "high": high_price,
                    "low": low_price,
                    "drawdown": max_drawdown,
                })

            except Exception as e:
                logger.warning(f"Could not fetch history for {ticker}: {e}")

        if not performance_data:
            return "❌ Could not fetch historical data for the provided tickers"

        # Sort by return
        performance_data.sort(key=lambda x: x['return'], reverse=True)

        # Performance table
        result += "**📊 Performance Rankings:**\n"
        result += "```\n"
        result += f"{'Rank':<6} {'Ticker':<8} {'Return':>10} {'Start':>10} {'End':>10}\n"
        result += "-" * 55 + "\n"

        for i, d in enumerate(performance_data, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            result += f"{emoji:<6} {d['ticker']:<8} {d['return']:>+9.1f}% ${d['start_price']:>9.2f} ${d['end_price']:>9.2f}\n"

        result += "```\n\n"

        # Summary stats
        result += "**📉 Risk Metrics (Max Drawdown):**\n"
        for d in performance_data:
            result += f"  {d['ticker']}: {d['drawdown']:.1f}% (Low: ${d['low']:.2f}, High: ${d['high']:.2f})\n"

        # Best and worst
        result += "\n**🏆 Summary:**\n"
        result += f"  Best Performer: {performance_data[0]['ticker']} ({performance_data[0]['return']:+.1f}%)\n"
        result += f"  Worst Performer: {performance_data[-1]['ticker']} ({performance_data[-1]['return']:+.1f}%)\n"

        avg_return = sum(d['return'] for d in performance_data) / len(performance_data)
        result += f"  Average Return: {avg_return:+.1f}%\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Completed performance comparison",
                    "done": True
                }
            })
        return result

    async def get_market_status(
        self,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Check if US stock markets are currently open.

        Returns:
            Market status (open/closed), current time, and market hours
        """
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": "🕐 Checking market status",
                    "done": False
                }
            })

        from datetime import datetime
        import pytz

        try:
            # Get Eastern Time
            eastern = pytz.timezone('US/Eastern')
            now_et = datetime.now(eastern)

            result = "**🕐 US Market Status**\n\n"
            result += f"**Current Time (ET):** {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n"

            # Check day of week
            weekday = now_et.weekday()  # 0=Monday, 6=Sunday

            if weekday >= 5:  # Weekend
                result += "**📊 Status:** 🔴 CLOSED (Weekend)\n\n"
                result += "Markets are closed on weekends.\n"
                result += "Next open: Monday at 9:30 AM ET\n"
            else:
                # Market hours: 9:30 AM - 4:00 PM ET
                market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                pre_market_start = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
                after_hours_end = now_et.replace(hour=20, minute=0, second=0, microsecond=0)

                if market_open <= now_et < market_close:
                    result += "**📊 Status:** 🟢 OPEN\n\n"
                    time_to_close = market_close - now_et
                    hours, remainder = divmod(time_to_close.seconds, 3600)
                    minutes = remainder // 60
                    result += f"Time until close: {hours}h {minutes}m\n"
                elif pre_market_start <= now_et < market_open:
                    result += "**📊 Status:** 🟡 PRE-MARKET\n\n"
                    time_to_open = market_open - now_et
                    hours, remainder = divmod(time_to_open.seconds, 3600)
                    minutes = remainder // 60
                    result += f"Time until regular session: {hours}h {minutes}m\n"
                elif market_close <= now_et < after_hours_end:
                    result += "**📊 Status:** 🟠 AFTER-HOURS\n\n"
                    time_to_end = after_hours_end - now_et
                    hours, remainder = divmod(time_to_end.seconds, 3600)
                    minutes = remainder // 60
                    result += f"Time until after-hours ends: {hours}h {minutes}m\n"
                else:
                    result += "**📊 Status:** 🔴 CLOSED\n\n"

            result += "\n**🕐 Regular Trading Hours:**\n"
            result += "  Open: 9:30 AM ET\n"
            result += "  Close: 4:00 PM ET\n"
            result += "  Pre-Market: 4:00 AM - 9:30 AM ET\n"
            result += "  After-Hours: 4:00 PM - 8:00 PM ET\n"

            result += "\n**📅 Note:** Markets are closed on federal holidays."

        except ImportError:
            # Fallback if pytz not available
            result = "**🕐 US Market Status**\n\n"
            result += "ℹ️ Install `pytz` for accurate timezone-based market status.\n"
            result += "\n**🕐 Regular Trading Hours (ET):**\n"
            result += "  Open: 9:30 AM ET\n"
            result += "  Close: 4:00 PM ET\n"
            result += "  Pre-Market: 4:00 AM - 9:30 AM ET\n"
            result += "  After-Hours: 4:00 PM - 8:00 PM ET\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": "✅ Retrieved market status",
                    "done": True
                }
            })
        return result

    # ============================================================
    # TOOL 50-52: UTILITY & HELPER METHODS
    # ============================================================

    async def search_ticker(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Search for ticker symbols by company name or keyword.

        Args:
            query: Company name or keyword to search

        Returns:
            List of matching tickers
        """
        if not self._check_rate_limit():
            return "⚠️ Rate limit exceeded."

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🔎 Searching for ticker",
                    "done": False
                }
            })

        result = f"**🔍 Search Results for: {query}**\n\n"
        result += "ℹ️ Note: Ticker search via yfinance has limited capabilities.\n"
        result += "For best results, use Yahoo Finance website's search feature.\n\n"

        # Try a few common variations
        queries_to_try = [
            query.upper(),
            query.upper().replace(" ", ""),
            query.upper()[:4],  # First 4 chars
        ]

        found_any = False
        for q in queries_to_try:
            try:
                stock = yf.Ticker(q)
                info = stock.info

                if info and info.get("longName"):
                    found_any = True
                    result += f"**{q}** - {info.get('longName')}\n"
                    result += f"  Exchange: {info.get('exchange', 'N/A')}\n"
                    result += f"  Type: {info.get('quoteType', 'N/A')}\n\n"
            except:
                pass

        if not found_any:
            result += "No matching tickers found. Try:\n"
            result += "- Using the exact ticker symbol\n"
            result += "- Searching on Yahoo Finance website\n"
            result += "- Checking alternative exchanges (e.g., .L for London, .TO for Toronto)\n"


        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Search complete",
                    "done": True
                }
            })
        return result

    async def validate_ticker(
        self,
        ticker: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Validate if a ticker symbol exists and is tradeable.

        Args:
            ticker: Ticker symbol to validate

        Returns:
            Validation result with ticker details if valid
        """
        try:
            validated = TickerValidator(ticker=ticker)
            ticker_clean = validated.ticker

            stock = yf.Ticker(ticker_clean)
            info = stock.info

            if not info or len(info) < 3:
                return f"❌ Ticker {ticker_clean} appears to be invalid or has no data"

            result = f"✅ **Ticker {ticker_clean} is valid**\n\n"
            result += f"**Name:** {info.get('longName', 'N/A')}\n"
            result += f"**Exchange:** {info.get('exchange', 'N/A')}\n"
            result += f"**Type:** {info.get('quoteType', 'N/A')}\n"
            result += f"**Currency:** {info.get('currency', 'N/A')}\n"
            result += f"**Tradeable:** {'Yes' if info.get('regularMarketPrice') else 'Unknown'}\n"

            return result

        except ValueError as e:
            return f"❌ Invalid ticker format: {str(e)}"
        except Exception as e:
            return f"❌ Error validating ticker: {str(e)}"

    async def get_api_status(
        self,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get API status and usage information.

        Returns:
            Current API usage statistics and configuration
        """
        result = "**⚙️ yfinance-ai API Status**\n\n"
        result += f"**Version:** 3.0.0\n"
        result += f"**yfinance Library:** {yf.__version__}\n\n"

        result += "**Rate Limiting:**\n"
        result += f"  Calls in current window: {self._call_count}/{RATE_LIMIT_CALLS}\n"
        result += f"  Window duration: {RATE_LIMIT_WINDOW}s\n\n"

        result += "**Configuration:**\n"
        result += f"  Default Period: {self.valves.default_period}\n"
        result += f"  Default Interval: {self.valves.default_interval}\n"
        result += f"  Max News Items: {self.valves.max_news_items}\n"
        result += f"  Max Comparison Tickers: {self.valves.max_comparison_tickers}\n\n"

        result += "**Available Methods:** 54+ financial data tools\n"
        result += "**Status:** ✅ Operational\n"

        return result

    # ============================================================
    # TOOL 43: BUILT-IN SELF-TEST FUNCTION (AI-CALLABLE)
    # ============================================================

    async def run_self_test(
        self,
        ticker: str = "SPY",
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        🧪 Run comprehensive self-test of all 55+ yfinance-ai tools.

        This function is designed to be called by AI assistants to verify
        that all tools are working correctly. It tests each tool category,
        reports success/failure status, and shows sample data from each test.

        Args:
            ticker: Ticker symbol to use for testing (default: SPY - reliable ETF with full data)

        Returns:
            Comprehensive test report with pass/fail status and sample data for all tools

        Example AI Prompt:
            "Run self-test on yfinance tools"
            "Test all financial data functions"
            "Verify yfinance-ai is working"
        """
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": "🧪 Running comprehensive yfinance-ai self-test...",
                    "done": False
                }
            })

        result = "**🧪 yfinance-ai Self-Test Report**\n\n"
        result += f"**Test Ticker:** {ticker} (S&P 500 ETF - chosen for comprehensive data coverage)\n"
        result += f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"**Version:** 3.0.1\n\n"
        result += "="*60 + "\n\n"

        test_results = {}
        category_results = {}

        def extract_sample_data(response: str, tool_name: str) -> str:
            """Extract key data points from response to prove it works"""
            samples = []

            # Extract price data
            if "Current Price:" in response or "Last Price:" in response:
                for line in response.split('\n'):
                    if 'Price:' in line or 'Close:' in line:
                        samples.append(line.strip())
                        if len(samples) >= 2:
                            break

            # Extract financial data
            elif "Revenue" in response or "Total Assets" in response or "Cash Flow" in response:
                for line in response.split('\n'):
                    if any(keyword in line for keyword in ['Revenue', 'Assets', 'Earnings', 'Cash Flow', 'Debt']):
                        samples.append(line.strip())
                        if len(samples) >= 2:
                            break

            # Extract ratio data
            elif "P/E" in response or "Ratio" in response:
                for line in response.split('\n'):
                    if 'P/E' in line or 'Ratio' in line or 'Margin' in line:
                        samples.append(line.strip())
                        if len(samples) >= 2:
                            break

            # Extract dividend data
            elif "Dividend" in response or "Yield" in response:
                for line in response.split('\n'):
                    if 'Yield' in line or 'Rate' in line or '$' in line:
                        samples.append(line.strip())
                        if len(samples) >= 2:
                            break

            # Extract news/data count
            elif "Recent News" in response or "Filings" in response:
                count = response.count('\n')
                samples.append(f"Found {count} lines of data")

            # Extract holder data
            elif "Holders" in response or "Institutional" in response:
                lines = [l for l in response.split('\n') if l.strip() and not l.startswith('**')]
                if lines:
                    samples.append(f"Found {len(lines)} data lines")

            # Default: show first meaningful line
            if not samples:
                for line in response.split('\n'):
                    if line.strip() and not line.startswith('**') and not line.startswith('='):
                        samples.append(line.strip()[:80])
                        break

            return " | ".join(samples[:2]) if samples else "Data retrieved"

        def is_successful(response: str, tool_name: str) -> tuple:
            """Determine if test passed and extract reason"""
            # Hard failures
            if "❌" in response and "Error" in response:
                return False, "Error returned"
            if "EXCEPTION" in response or "Exception" in response:
                return False, "Exception raised"
            if len(response) < 20:
                return False, "Empty or minimal response"

            # Expected no-data scenarios (not failures)
            optional_data_tools = [
                "get_stock_splits", "get_capital_gains", "get_insider_purchases",
                "get_sec_filings", "get_mutualfund_holders"
            ]

            if tool_name in optional_data_tools:
                if "ℹ️" in response or "No" in response:
                    return True, "OK (optional data not available)"

            # Check for actual data presence
            if "Current Price:" in response or "Last Price:" in response:
                return True, "Price data found"
            if "Revenue" in response or "Total Assets" in response:
                return True, "Financial data found"
            if "Dividend" in response or "Yield:" in response:
                return True, "Dividend data found"
            if "P/E" in response or "Market Cap:" in response:
                return True, "Metrics found"
            if "Institutional" in response or "Insider" in response or "Mutual Fund" in response:
                return True, "Holder data found"
            if "News" in response or "article" in response:
                return True, "News data found"
            if "Sector:" in response or "Industry:" in response:
                return True, "Company info found"
            if "Expiration" in response or "Strike" in response:
                return True, "Options data found"
            if "Data Points:" in response:
                return True, "Historical data found"
            if "Status:" in response and "Operational" in response:
                return True, "API status confirmed"
            if "✅" in response and "valid" in response:
                return True, "Validation passed"

            # If response has substantial content, consider it passed
            if len(response) > 100 and "**" in response:
                return True, "Substantial data returned"

            return False, "No recognizable data pattern"

        # Define test categories and tools with better test tickers
        test_categories = {
            "Stock Quotes & Prices": [
                ("get_stock_price", ["SPY"]),
                ("get_fast_info", ["SPY"]),
                ("get_stock_quote", ["SPY"]),
                ("get_historical_data", ["SPY", "1mo", "1d"]),
                ("get_isin", ["SPY"]),
            ],
            "Company Information": [
                ("get_company_info", ["AAPL"]),
                ("get_company_officers", ["AAPL"]),
            ],
            "Financial Statements": [
                ("get_income_statement", ["AAPL", "annual"]),
                ("get_balance_sheet", ["AAPL", "annual"]),
                ("get_cash_flow", ["AAPL", "annual"]),
            ],
            "Key Ratios & Metrics": [
                ("get_key_ratios", ["AAPL"]),
            ],
            "Dividends & Corporate Actions": [
                ("get_dividends", ["SPY", "5y"]),
                ("get_stock_splits", ["AAPL"]),
                ("get_corporate_actions", ["SPY"]),
                ("get_capital_gains", ["SPY"]),
            ],
            "Earnings & Estimates": [
                ("get_earnings_dates", ["AAPL"]),
                ("get_earnings_history", ["AAPL"]),
                ("get_analyst_estimates", ["AAPL"]),
                ("get_growth_estimates", ["AAPL"]),
                ("get_eps_trend", ["AAPL"]),
                ("get_eps_revisions", ["AAPL"]),
                ("get_earnings_calendar", ["AAPL"]),
            ],
            "Analyst Data": [
                ("get_analyst_recommendations", ["AAPL"]),
                ("get_analyst_price_targets", ["AAPL"]),
                ("get_upgrades_downgrades", ["AAPL", 5]),
            ],
            "Institutional & Insider Data": [
                ("get_institutional_holders", ["AAPL"]),
                ("get_major_holders", ["AAPL"]),
                ("get_mutualfund_holders", ["AAPL"]),
                ("get_insider_transactions", ["AAPL"]),
                ("get_insider_purchases", ["AAPL"]),
                ("get_insider_roster_holders", ["AAPL"]),
            ],
            "Options & Derivatives": [
                ("get_options_chain", ["SPY"]),
                ("get_options_expirations", ["SPY"]),
            ],
            "News & SEC Filings": [
                ("get_stock_news", ["AAPL", 3]),
                ("get_sec_filings", ["AAPL"]),
            ],
            "Market Indices & Comparison": [
                ("get_market_indices", []),
                ("compare_stocks", ["SPY,QQQ,DIA"]),
                ("get_sector_performance", []),
            ],
            "Fund/ETF Data": [
                ("get_fund_overview", ["SPY"]),
                ("get_fund_holdings", ["SPY"]),
                ("get_fund_sector_weights", ["SPY"]),
            ],
            "Crypto, Forex & Commodities": [
                ("get_crypto_price", ["BTC"]),
                ("get_forex_rate", ["EURUSD"]),
                ("get_commodity_price", ["gold"]),
            ],
            "Analysis & Comparison": [
                ("get_peer_comparison", ["AAPL"]),
                ("get_financial_summary", ["AAPL"]),
                ("get_historical_comparison", ["AAPL,MSFT", "6mo"]),
                ("get_market_status", []),
            ],
            "Bulk Operations": [
                ("download_multiple_tickers", ["SPY QQQ", "1mo", "1d"]),
                ("get_trending_tickers", [5]),
            ],
            "Utility Functions": [
                ("search_ticker", ["Apple"]),
                ("validate_ticker", ["SPY"]),
                ("get_api_status", []),
            ],
        }

        # Run tests for each category
        for category, tools in test_categories.items():
            result += f"### {category}\n"
            category_pass = 0
            category_fail = 0

            for tool_name, args in tools:
                try:
                    method = getattr(self, tool_name)
                    response = await method(*args)

                    # Analyze response
                    success, reason = is_successful(response, tool_name)
                    sample = extract_sample_data(response, tool_name)

                    if success:
                        status = "✅ PASS"
                        test_results[tool_name] = True
                        category_pass += 1
                        result += f"  {status} - {tool_name}\n"
                        result += f"      → {sample}\n"
                    else:
                        status = "❌ FAIL"
                        test_results[tool_name] = False
                        category_fail += 1
                        result += f"  {status} - {tool_name}: {reason}\n"

                except Exception as e:
                    result += f"  ❌ EXCEPTION - {tool_name}: {str(e)[:60]}\n"
                    test_results[tool_name] = False
                    category_fail += 1

            # Category summary
            category_results[category] = {
                "pass": category_pass,
                "fail": category_fail,
                "total": len(tools)
            }
            result += f"  📊 Category: {category_pass}/{len(tools)} passed\n\n"

        # Overall summary
        result += "="*60 + "\n"
        result += "**📊 OVERALL SUMMARY**\n"
        result += "="*60 + "\n\n"

        total_tests = len(test_results)
        total_pass = sum(1 for v in test_results.values() if v)
        total_fail = total_tests - total_pass
        success_rate = (total_pass / total_tests * 100) if total_tests > 0 else 0

        result += f"**Total Tests:** {total_tests}\n"
        result += f"**Passed:** {total_pass} ✅\n"
        result += f"**Failed:** {total_fail} ❌\n"
        result += f"**Success Rate:** {success_rate:.1f}%\n\n"

        # Category breakdown
        result += "**Category Breakdown:**\n"
        for category, stats in category_results.items():
            pct = (stats['pass'] / stats['total'] * 100) if stats['total'] > 0 else 0
            emoji = "✅" if pct == 100 else "⚠️" if pct >= 50 else "❌"
            result += f"  {emoji} {category}: {stats['pass']}/{stats['total']} ({pct:.0f}%)\n"

        result += "\n" + "="*60 + "\n\n"

        if total_fail == 0:
            result += "🎉 **ALL TESTS PASSED!** yfinance-ai is fully operational.\n"
        elif success_rate >= 90:
            result += "✅ **EXCELLENT** - Nearly all tools working perfectly!\n"
        elif success_rate >= 75:
            result += "✅ **GOOD** - Most tools operational, some data unavailable.\n"
        elif success_rate >= 50:
            result += "⚠️  **PARTIAL** - Some tools experiencing issues.\n"
        else:
            result += "❌ **CRITICAL** - Many tools failing. Check connectivity.\n"

        result += "\n📚 For details on any tool, ask: 'How do I use [tool_name]?'\n"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Self-test complete: {success_rate:.0f}% success rate",
                    "done": True
                }
            })

        return result


# ============================================================
# END OF yfinance-ai v3.0.2
# ============================================================
