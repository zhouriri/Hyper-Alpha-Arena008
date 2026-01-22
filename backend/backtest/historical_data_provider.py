"""
Historical Data Provider for Backtest

Provides historical market data for backtesting.
Interface compatible with DataProvider for strategy code reuse.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class HistoricalDataProvider:
    """
    Provides historical data for backtesting.

    Key differences from DataProvider:
    - Reads from database instead of real-time API
    - Filters data by current_time_ms (only data before this time)
    - Used by backtest engine to simulate historical market state
    - Tracks data queries for logging
    """

    def __init__(
        self,
        db: Session,
        symbols: List[str],
        start_time_ms: int,
        end_time_ms: int,
    ):
        """
        Initialize historical data provider.

        Args:
            db: Database session
            symbols: List of symbols to load data for
            start_time_ms: Backtest start time (milliseconds)
            end_time_ms: Backtest end time (milliseconds)
        """
        self.db = db
        self.symbols = symbols
        self.start_time_ms = start_time_ms
        self.end_time_ms = end_time_ms
        self.current_time_ms = start_time_ms

        # Caches
        self._kline_cache: Dict[str, List[Dict]] = {}
        self._flow_cache: Dict[str, Dict] = {}
        self._price_cache: Dict[str, float] = {}

        # Query tracking for logging
        self._query_log: List[str] = []

        # Preload data for better performance
        self._preload_data()

    def _preload_data(self):
        """Preload all kline data for symbols to avoid repeated DB queries."""
        logger.info(f"Preloading data for {len(self.symbols)} symbols...")

        for symbol in self.symbols:
            # Preload common periods
            for period in ["5m", "15m", "1h", "4h", "1d"]:
                self._load_klines_to_cache(symbol, period)

        logger.info(f"Preload complete. Cache size: {len(self._kline_cache)} entries")

    def set_current_time(self, timestamp_ms: int):
        """Set current simulation time."""
        self.current_time_ms = timestamp_ms
        # Clear price cache when time changes
        self._price_cache = {}

    def clear_query_log(self):
        """Clear query log for new trigger."""
        self._query_log = []

    def get_query_log(self) -> List[str]:
        """Get list of data queries made since last clear."""
        return self._query_log.copy()

    def _log_query(self, query_type: str, symbol: str, params: str = ""):
        """Log a data query."""
        if params:
            self._query_log.append(f"{query_type}({symbol}, {params})")
        else:
            self._query_log.append(f"{query_type}({symbol})")

    def get_current_prices(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get current prices for symbols at current_time_ms."""
        target_symbols = symbols or self.symbols
        prices = {}

        for symbol in target_symbols:
            if symbol in self._price_cache:
                prices[symbol] = self._price_cache[symbol]
                continue

            price = self._get_price_at_time(symbol, self.current_time_ms)
            if price:
                prices[symbol] = price
                self._price_cache[symbol] = price

        return prices

    def _get_price_at_time(self, symbol: str, timestamp_ms: int) -> Optional[float]:
        """Get price at specific timestamp from cached klines."""
        timestamp_sec = timestamp_ms // 1000
        cache_key = f"{symbol}_5m"

        # Try to get from cache first
        if cache_key in self._kline_cache:
            klines = self._kline_cache[cache_key]
            # Binary search for the closest kline <= timestamp
            for k in reversed(klines):
                if k["timestamp"] <= timestamp_sec:
                    return k["close"]
            # If no kline found before timestamp, return first available
            if klines:
                return klines[0]["close"]

        # Fallback to DB query if not in cache
        try:
            result = self.db.execute(text("""
                SELECT close_price FROM crypto_klines
                WHERE symbol = :symbol AND period = '5m'
                AND timestamp <= :ts
                ORDER BY timestamp DESC LIMIT 1
            """), {"symbol": symbol, "ts": timestamp_sec})
            row = result.fetchone()
            if row:
                return float(row[0])
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")

        return None

    def get_klines(self, symbol: str, period: str, count: int = 50) -> List[Any]:
        """
        Get historical K-line data up to current_time_ms.

        Returns Kline objects compatible with DataProvider.
        """
        from program_trader.models import Kline

        # Log the query
        self._log_query("get_klines", symbol, f"period={period}, count={count}")

        timestamp_sec = self.current_time_ms // 1000
        cache_key = f"{symbol}_{period}"

        # Load and cache all klines for this symbol/period if not cached
        if cache_key not in self._kline_cache:
            self._load_klines_to_cache(symbol, period)

        # Filter klines up to current time and return last 'count'
        all_klines = self._kline_cache.get(cache_key, [])
        filtered = [k for k in all_klines if k["timestamp"] <= timestamp_sec]

        # Convert to Kline objects
        result = []
        for k in filtered[-count:]:
            result.append(Kline(
                timestamp=k["timestamp"],
                open=k["open"],
                high=k["high"],
                low=k["low"],
                close=k["close"],
                volume=k["volume"],
            ))

        return result

    def _load_klines_to_cache(self, symbol: str, period: str):
        """Load all klines for symbol/period into cache."""
        cache_key = f"{symbol}_{period}"

        # Calculate time range with buffer for indicator calculation
        start_sec = (self.start_time_ms // 1000) - (500 * 300)  # 500 candles * 5min
        end_sec = self.end_time_ms // 1000

        try:
            result = self.db.execute(text("""
                SELECT timestamp, open_price, high_price, low_price, close_price, volume
                FROM crypto_klines
                WHERE symbol = :symbol AND period = :period
                AND timestamp >= :start_ts AND timestamp <= :end_ts
                ORDER BY timestamp ASC
            """), {"symbol": symbol, "period": period, "start_ts": start_sec, "end_ts": end_sec})

            klines = []
            for row in result:
                klines.append({
                    "timestamp": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]) if row[5] else 0.0,
                })

            self._kline_cache[cache_key] = klines
            logger.debug(f"Loaded {len(klines)} klines for {symbol} {period}")

        except Exception as e:
            logger.error(f"Failed to load klines for {symbol} {period}: {e}")
            self._kline_cache[cache_key] = []

    def get_indicator(self, symbol: str, indicator: str, period: str) -> Dict[str, Any]:
        """
        Calculate technical indicator from historical klines.

        Uses same calculation as DataProvider for consistency.
        """
        from services.technical_indicators import calculate_indicators

        # Log the query
        self._log_query("get_indicator", symbol, f"indicator={indicator}, period={period}")

        # Get klines (enough for indicator calculation)
        klines = self.get_klines(symbol, period, 500)
        if not klines:
            return {}

        # Convert to format expected by calculate_indicators
        kline_data = [
            {
                "timestamp": k.timestamp,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }
            for k in klines
        ]

        try:
            indicator_upper = indicator.upper()
            calculated = calculate_indicators(kline_data, [indicator_upper])

            if indicator_upper in calculated and calculated[indicator_upper] is not None:
                value = calculated[indicator_upper]
                if isinstance(value, list):
                    return {'value': value[-1] if value else None, 'series': value}
                elif isinstance(value, dict):
                    latest = {}
                    for k, v in value.items():
                        if isinstance(v, list) and v:
                            latest[k] = v[-1]
                        else:
                            latest[k] = v
                    return latest
                else:
                    return {'value': value}
        except Exception as e:
            logger.warning(f"Failed to calculate {indicator} for {symbol}: {e}")

        return {}

    def get_flow(self, symbol: str, metric: str, period: str) -> Dict[str, Any]:
        """
        Get historical flow data (CVD, OI, TAKER, etc.).

        Queries aggregated market data tables.
        """
        from services.market_flow_indicators import get_flow_indicators_for_prompt

        # Log the query
        self._log_query("get_flow", symbol, f"metric={metric}, period={period}")

        try:
            results = get_flow_indicators_for_prompt(
                self.db, symbol, period, [metric.upper()], self.current_time_ms
            )
            return results.get(metric.upper(), {}) or {}
        except Exception as e:
            logger.warning(f"Failed to get flow {metric} for {symbol}: {e}")
            return {}

    def get_regime(self, symbol: str, period: str) -> Any:
        """
        Get market regime at current time.

        Uses historical data to calculate regime.
        """
        from program_trader.models import RegimeInfo
        from services.market_regime_service import get_market_regime

        # Log the query
        self._log_query("get_regime", symbol, f"period={period}")

        try:
            result = get_market_regime(
                self.db, symbol, period,
                use_realtime=False,
                timestamp_ms=self.current_time_ms
            )
            if result:
                return RegimeInfo(
                    regime=result.get("regime", "noise"),
                    conf=result.get("confidence", 0.0),
                    direction=result.get("direction", "neutral"),
                    reason=result.get("reason", ""),
                    indicators=result.get("indicators", {}),
                )
        except Exception as e:
            logger.warning(f"Failed to get regime for {symbol}: {e}")

        return RegimeInfo(regime="noise", conf=0.0)

    def get_price_change(self, symbol: str, period: str) -> Dict[str, float]:
        """Get price change over period."""
        from services.market_flow_indicators import get_indicator_value

        # Log the query
        self._log_query("get_price_change", symbol, f"period={period}")

        try:
            raw = get_indicator_value(
                self.db, symbol, "PRICE_CHANGE", period, self.current_time_ms
            )
            if raw:
                return {
                    "change_percent": raw.get("change_percent", 0.0),
                    "change_usd": raw.get("change_usd", 0.0),
                }
        except Exception:
            pass

        return {"change_percent": 0.0, "change_usd": 0.0}

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data snapshot at current time."""
        # Log the query
        self._log_query("get_market_data", symbol)

        price = self._get_price_at_time(symbol, self.current_time_ms)

        # Get additional data from market_asset_metrics if available
        try:
            result = self.db.execute(text("""
                SELECT mark_price, funding_rate, open_interest
                FROM market_asset_metrics
                WHERE symbol = :symbol AND timestamp <= :ts
                ORDER BY timestamp DESC LIMIT 1
            """), {"symbol": symbol, "ts": self.current_time_ms})
            row = result.fetchone()
            if row:
                return {
                    "symbol": symbol,
                    "price": price or float(row[0] or 0),
                    "mark_price": float(row[0] or 0),
                    "funding_rate": float(row[1] or 0),
                    "open_interest": float(row[2] or 0),
                }
        except Exception as e:
            logger.warning(f"Failed to get market data for {symbol}: {e}")

        return {"symbol": symbol, "price": price or 0.0}


