"""
Signal Backtest Service

Backtests signals against historical data to show where triggers would occur.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Timeframe to milliseconds mapping
TIMEFRAME_MS = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}


class SignalBacktestService:
    """Service for backtesting signals against historical data."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def backtest_signal(
        self, db: Session, signal_id: int, symbol: str,
        kline_min_ts: int = None, kline_max_ts: int = None
    ) -> Dict[str, Any]:
        """
        Backtest a signal against historical data.
        Returns only trigger points - K-lines should be fetched separately via market API.

        Args:
            db: Database session
            signal_id: Signal definition ID
            symbol: Trading symbol (e.g., 'BTC')
            kline_min_ts: Minimum K-line timestamp in milliseconds (for filtering triggers)
            kline_max_ts: Maximum K-line timestamp in milliseconds (for filtering triggers)
        """
        # Clear bucket cache for fresh data
        self._bucket_cache = {}

        # Get signal definition
        result = db.execute(
            text("""
                SELECT id, signal_name, description, trigger_condition, enabled
                FROM signal_definitions WHERE id = :id
            """),
            {"id": signal_id}
        )
        row = result.fetchone()
        if not row:
            return {"error": "Signal not found"}

        signal_def = {
            "id": row[0],
            "signal_name": row[1],
            "description": row[2],
            "trigger_condition": row[3],
            "enabled": row[4]
        }

        condition = signal_def.get("trigger_condition", {})
        metric = condition.get("metric")
        time_window = condition.get("time_window", "5m")

        if not metric:
            return {"error": "Signal has no metric configured"}

        # Find triggers within the specified time range
        triggers = self._find_triggers_in_range(
            db, signal_def, symbol, time_window, kline_min_ts, kline_max_ts
        )

        return {
            "signal_id": signal_id,
            "signal_name": signal_def["signal_name"],
            "symbol": symbol,
            "time_window": time_window,
            "condition": condition,
            "trigger_count": len(triggers),
            "triggers": triggers,
        }

    def backtest_temp_signal(
        self, db: Session, symbol: str, trigger_condition: Dict,
        kline_min_ts: int = None, kline_max_ts: int = None
    ) -> Dict[str, Any]:
        """
        Backtest a temporary signal configuration without saving to database.
        Used for AI signal creation preview.

        Args:
            db: Database session
            symbol: Trading symbol (e.g., 'BTC')
            trigger_condition: Signal trigger condition dict
            kline_min_ts: Minimum K-line timestamp in milliseconds
            kline_max_ts: Maximum K-line timestamp in milliseconds
        """
        # Clear bucket cache for fresh data
        self._bucket_cache = {}

        # Build temporary signal definition
        signal_def = {
            "id": None,
            "signal_name": "Temporary Preview",
            "description": "AI-generated signal preview",
            "trigger_condition": trigger_condition,
            "enabled": True
        }

        metric = trigger_condition.get("metric")
        time_window = trigger_condition.get("time_window", "5m")

        if not metric:
            return {"error": "Signal has no metric configured"}

        # Find triggers within the specified time range
        triggers = self._find_triggers_in_range(
            db, signal_def, symbol, time_window, kline_min_ts, kline_max_ts
        )

        return {
            "signal_id": None,
            "signal_name": "Temporary Preview",
            "symbol": symbol,
            "time_window": time_window,
            "condition": trigger_condition,
            "trigger_count": len(triggers),
            "triggers": triggers,
        }

    def _find_triggers_in_range(
        self, db: Session, signal_def: Dict, symbol: str, time_window: str,
        kline_min_ts: int = None, kline_max_ts: int = None
    ) -> List[Dict]:
        """
        Find trigger points within a time range.

        Args:
            db: Database session
            signal_def: Signal definition dict
            symbol: Trading symbol
            time_window: Time window (e.g., '5m', '15m')
            kline_min_ts: Minimum timestamp in milliseconds (optional)
            kline_max_ts: Maximum timestamp in milliseconds (optional)
        """
        condition = signal_def.get("trigger_condition", {})
        metric = condition.get("metric")
        operator = condition.get("operator")
        threshold = condition.get("threshold")

        # Handle taker_volume composite signal
        if metric == "taker_volume":
            return self._find_taker_triggers_in_range(
                db, signal_def, symbol, time_window, kline_min_ts, kline_max_ts
            )

        if not all([metric, operator, threshold is not None]):
            return []

        # Map metric names for backward compatibility
        metric_map = {
            "oi_delta_percent": "oi_delta",
            "funding_rate": "funding",
            "taker_buy_ratio": "taker_ratio",
        }
        metric = metric_map.get(metric, metric)

        interval_ms = TIMEFRAME_MS.get(time_window, 300000)

        # Get ALL bucket values from market flow data
        cache_key = f"{symbol}_{metric}_{interval_ms}"
        if cache_key not in self._bucket_cache:
            self._bucket_cache[cache_key] = self._compute_all_bucket_values(
                db, symbol, metric, interval_ms
            )
        bucket_values = self._bucket_cache[cache_key]

        if not bucket_values:
            return []

        # Iterate over all buckets and find triggers
        triggers = []
        for bucket_ts, value in sorted(bucket_values.items()):
            # Filter by time range if provided
            if kline_min_ts is not None and bucket_ts < kline_min_ts:
                continue
            if kline_max_ts is not None and bucket_ts > kline_max_ts:
                continue

            if value is not None and self._evaluate_condition(value, operator, threshold):
                triggers.append({
                    "timestamp": bucket_ts,
                    "value": value,
                    "threshold": threshold,
                    "operator": operator,
                })

        return triggers

    def _find_taker_triggers_in_range(
        self, db: Session, signal_def: Dict, symbol: str, time_window: str,
        kline_min_ts: int = None, kline_max_ts: int = None
    ) -> List[Dict]:
        """Find taker_volume composite signal triggers within a time range."""
        condition = signal_def.get("trigger_condition", {})
        direction = condition.get("direction", "any")
        ratio_threshold = condition.get("ratio_threshold", 1.5)
        volume_threshold = condition.get("volume_threshold", 0)

        interval_ms = TIMEFRAME_MS.get(time_window, 300000)

        # Compute all taker volume buckets from market flow data
        taker_buckets = self._compute_taker_volume_buckets(db, symbol, interval_ms)
        if not taker_buckets:
            return []

        # Iterate over all buckets and find triggers
        triggers = []
        for bucket_ts, data in sorted(taker_buckets.items()):
            # Filter by time range if provided
            if kline_min_ts is not None and bucket_ts < kline_min_ts:
                continue
            if kline_max_ts is not None and bucket_ts > kline_max_ts:
                continue

            ratio = data["ratio"]
            total = data["volume"]

            if total < volume_threshold:
                continue

            triggered = False
            actual_dir = None

            if direction == "buy" and ratio >= ratio_threshold:
                triggered, actual_dir = True, "buy"
            elif direction == "sell" and ratio <= 1 / ratio_threshold:
                triggered, actual_dir = True, "sell"
            elif direction == "any":
                if ratio >= ratio_threshold:
                    triggered, actual_dir = True, "buy"
                elif ratio <= 1 / ratio_threshold:
                    triggered, actual_dir = True, "sell"

            if triggered:
                triggers.append({
                    "timestamp": bucket_ts,
                    "direction": actual_dir,
                    "ratio": ratio,
                    "ratio_threshold": ratio_threshold,
                    "volume": total,
                    "volume_threshold": volume_threshold,
                })

        return triggers

    # Legacy method - kept for backward compatibility but no longer used
    def _find_triggers(
        self, db: Session, signal_def: Dict, symbol: str, klines: List[Dict], time_window: str
    ) -> List[Dict]:
        """
        Find trigger points in historical data.

        IMPORTANT: This method iterates over MARKET FLOW DATA (buckets), not K-lines.
        K-lines are only used as a visual background - the actual trigger detection
        is based on market flow indicator values from the database.

        The trigger timestamp is the bucket timestamp, and we find the closest
        K-line to display the price context.
        """
        condition = signal_def.get("trigger_condition", {})
        metric = condition.get("metric")
        operator = condition.get("operator")
        threshold = condition.get("threshold")

        if not all([metric, operator, threshold is not None]):
            # Handle taker_volume composite signal
            if metric == "taker_volume":
                return self._find_taker_triggers(db, signal_def, symbol, klines, time_window)
            return []

        # Map metric names for backward compatibility
        metric_map = {
            "oi_delta_percent": "oi_delta",
            "funding_rate": "funding",
            "taker_buy_ratio": "taker_ratio",
        }
        metric = metric_map.get(metric, metric)

        interval_ms = TIMEFRAME_MS.get(time_window, 300000)

        # Get ALL bucket values from market flow data (this is the PRIMARY data source)
        cache_key = f"{symbol}_{metric}_{interval_ms}"
        if cache_key not in self._bucket_cache:
            self._bucket_cache[cache_key] = self._compute_all_bucket_values(
                db, symbol, metric, interval_ms
            )
        bucket_values = self._bucket_cache[cache_key]

        if not bucket_values:
            return []

        # Build a lookup for K-line prices (for display only)
        from services.market_flow_indicators import floor_timestamp
        kline_prices = {}
        for kline in klines:
            bucket_ts = floor_timestamp(kline["timestamp"], interval_ms)
            kline_prices[bucket_ts] = kline["close"]

        # Get K-line time range for filtering triggers to display
        if klines:
            kline_min_ts = min(floor_timestamp(k["timestamp"], interval_ms) for k in klines)
            kline_max_ts = max(floor_timestamp(k["timestamp"], interval_ms) for k in klines)
        else:
            return []

        # Iterate over ALL buckets and find triggers
        triggers = []
        for bucket_ts, value in sorted(bucket_values.items()):
            # Only include triggers within K-line display range
            if bucket_ts < kline_min_ts or bucket_ts > kline_max_ts:
                continue

            if value is not None and self._evaluate_condition(value, operator, threshold):
                # Get price from K-line (for display context)
                price = kline_prices.get(bucket_ts, 0)
                triggers.append({
                    "timestamp": bucket_ts,
                    "value": value,
                    "threshold": threshold,
                    "operator": operator,
                    "price": price,
                })

        return triggers

    def _get_indicator_at_time(
        self, db: Session, symbol: str, metric: str, timestamp_ms: int, interval_ms: int
    ) -> Optional[float]:
        """
        Get indicator value at a specific timestamp using bucket aggregation.

        IMPORTANT: This method uses the same bucket aggregation logic as
        signal_analysis_service to ensure consistency between:
        - Statistical analysis (threshold suggestions)
        - Preview backtest (trigger visualization)
        - K-line chart indicators

        The key insight is that we need to return the indicator value FOR that
        specific bucket, not the "current" value looking back from that timestamp.
        """
        # Use pre-computed bucket values if available
        cache_key = f"{symbol}_{metric}_{interval_ms}"
        if not hasattr(self, '_bucket_cache'):
            self._bucket_cache = {}

        if cache_key not in self._bucket_cache:
            self._bucket_cache[cache_key] = self._compute_all_bucket_values(
                db, symbol, metric, interval_ms
            )

        bucket_values = self._bucket_cache[cache_key]
        if not bucket_values:
            return None

        # Find the bucket that contains this timestamp
        from services.market_flow_indicators import floor_timestamp
        bucket_ts = floor_timestamp(timestamp_ms, interval_ms)

        return bucket_values.get(bucket_ts)

    def _compute_all_bucket_values(
        self, db: Session, symbol: str, metric: str, interval_ms: int
    ) -> Dict[int, float]:
        """
        Compute indicator values for all buckets using the same logic as
        signal_analysis_service. This ensures consistency between statistical
        analysis and backtest preview.

        Returns a dict mapping bucket_timestamp -> indicator_value
        """
        from services.market_flow_indicators import floor_timestamp
        from database.models import MarketAssetMetrics, MarketTradesAggregated
        from database.models import MarketOrderbookSnapshots
        from datetime import datetime

        # Query 7 days of data (same as signal_analysis_service)
        current_time_ms = int(datetime.utcnow().timestamp() * 1000)
        start_time_ms = current_time_ms - (7 * 24 * 60 * 60 * 1000)

        if metric == "oi_delta":
            return self._compute_oi_delta_buckets(
                db, symbol, interval_ms, start_time_ms, current_time_ms
            )
        elif metric == "cvd":
            return self._compute_cvd_buckets(
                db, symbol, interval_ms, start_time_ms, current_time_ms
            )
        elif metric == "depth_ratio":
            return self._compute_depth_ratio_buckets(
                db, symbol, interval_ms, start_time_ms, current_time_ms
            )
        elif metric == "order_imbalance":
            return self._compute_imbalance_buckets(
                db, symbol, interval_ms, start_time_ms, current_time_ms
            )
        elif metric == "taker_ratio":
            return self._compute_taker_ratio_buckets(
                db, symbol, interval_ms, start_time_ms, current_time_ms
            )
        elif metric == "funding":
            return self._compute_funding_buckets(
                db, symbol, interval_ms, start_time_ms, current_time_ms
            )
        elif metric == "oi":
            return self._compute_oi_buckets(
                db, symbol, interval_ms, start_time_ms, current_time_ms
            )
        else:
            logger.warning(f"Unknown metric for bucket computation: {metric}")
            return {}

    def _compute_oi_delta_buckets(
        self, db, symbol, interval_ms, start_time_ms, current_time_ms
    ) -> Dict[int, float]:
        """Compute OI delta percentage for each bucket (same as signal_analysis)."""
        from services.market_flow_indicators import floor_timestamp
        from database.models import MarketAssetMetrics

        records = db.query(
            MarketAssetMetrics.timestamp,
            MarketAssetMetrics.open_interest
        ).filter(
            MarketAssetMetrics.symbol == symbol.upper(),
            MarketAssetMetrics.timestamp >= start_time_ms,
            MarketAssetMetrics.timestamp <= current_time_ms
        ).order_by(MarketAssetMetrics.timestamp).all()

        if not records:
            return {}

        # Bucket by period
        buckets = {}
        for ts, oi in records:
            bucket_ts = floor_timestamp(ts, interval_ms)
            buckets[bucket_ts] = float(oi) if oi else None

        # Calculate deltas - map each bucket to its delta value
        sorted_times = sorted(buckets.keys())
        result = {}
        for i in range(1, len(sorted_times)):
            prev_oi = buckets[sorted_times[i-1]]
            curr_oi = buckets[sorted_times[i]]
            if prev_oi and curr_oi and prev_oi != 0:
                delta_pct = ((curr_oi - prev_oi) / prev_oi) * 100
                # The delta is associated with the CURRENT bucket
                result[sorted_times[i]] = delta_pct

        return result

    def _compute_cvd_buckets(
        self, db, symbol, interval_ms, start_time_ms, current_time_ms
    ) -> Dict[int, float]:
        """Compute CVD for each bucket."""
        from services.market_flow_indicators import floor_timestamp
        from database.models import MarketTradesAggregated

        records = db.query(
            MarketTradesAggregated.timestamp,
            MarketTradesAggregated.taker_buy_notional,
            MarketTradesAggregated.taker_sell_notional
        ).filter(
            MarketTradesAggregated.symbol == symbol.upper(),
            MarketTradesAggregated.timestamp >= start_time_ms,
            MarketTradesAggregated.timestamp <= current_time_ms
        ).order_by(MarketTradesAggregated.timestamp).all()

        if not records:
            return {}

        buckets = {}
        for ts, buy, sell in records:
            bucket_ts = floor_timestamp(ts, interval_ms)
            if bucket_ts not in buckets:
                buckets[bucket_ts] = {"buy": 0, "sell": 0}
            buckets[bucket_ts]["buy"] += float(buy or 0)
            buckets[bucket_ts]["sell"] += float(sell or 0)

        result = {}
        for ts in buckets:
            result[ts] = buckets[ts]["buy"] - buckets[ts]["sell"]

        return result

    def _compute_depth_ratio_buckets(
        self, db, symbol, interval_ms, start_time_ms, current_time_ms
    ) -> Dict[int, float]:
        """Compute depth ratio (bid/ask) for each bucket."""
        from services.market_flow_indicators import floor_timestamp
        from database.models import MarketOrderbookSnapshots

        records = db.query(
            MarketOrderbookSnapshots.timestamp,
            MarketOrderbookSnapshots.bid_depth_5,
            MarketOrderbookSnapshots.ask_depth_5
        ).filter(
            MarketOrderbookSnapshots.symbol == symbol.upper(),
            MarketOrderbookSnapshots.timestamp >= start_time_ms,
            MarketOrderbookSnapshots.timestamp <= current_time_ms
        ).order_by(MarketOrderbookSnapshots.timestamp).all()

        if not records:
            return {}

        buckets = {}
        for ts, bid, ask in records:
            bucket_ts = floor_timestamp(ts, interval_ms)
            buckets[bucket_ts] = {"bid": float(bid or 0), "ask": float(ask or 0)}

        result = {}
        for ts in buckets:
            ask = buckets[ts]["ask"]
            if ask > 0:
                result[ts] = buckets[ts]["bid"] / ask

        return result

    def _compute_imbalance_buckets(
        self, db, symbol, interval_ms, start_time_ms, current_time_ms
    ) -> Dict[int, float]:
        """Compute order imbalance for each bucket."""
        from services.market_flow_indicators import floor_timestamp
        from database.models import MarketOrderbookSnapshots

        records = db.query(
            MarketOrderbookSnapshots.timestamp,
            MarketOrderbookSnapshots.bid_depth_5,
            MarketOrderbookSnapshots.ask_depth_5
        ).filter(
            MarketOrderbookSnapshots.symbol == symbol.upper(),
            MarketOrderbookSnapshots.timestamp >= start_time_ms,
            MarketOrderbookSnapshots.timestamp <= current_time_ms
        ).order_by(MarketOrderbookSnapshots.timestamp).all()

        if not records:
            return {}

        buckets = {}
        for ts, bid, ask in records:
            bucket_ts = floor_timestamp(ts, interval_ms)
            buckets[bucket_ts] = {"bid": float(bid or 0), "ask": float(ask or 0)}

        result = {}
        for ts in buckets:
            bid, ask = buckets[ts]["bid"], buckets[ts]["ask"]
            total = bid + ask
            if total > 0:
                result[ts] = (bid - ask) / total

        return result

    def _compute_taker_ratio_buckets(
        self, db, symbol, interval_ms, start_time_ms, current_time_ms
    ) -> Dict[int, float]:
        """Compute taker buy/sell ratio for each bucket."""
        from services.market_flow_indicators import floor_timestamp
        from database.models import MarketTradesAggregated

        records = db.query(
            MarketTradesAggregated.timestamp,
            MarketTradesAggregated.taker_buy_notional,
            MarketTradesAggregated.taker_sell_notional
        ).filter(
            MarketTradesAggregated.symbol == symbol.upper(),
            MarketTradesAggregated.timestamp >= start_time_ms,
            MarketTradesAggregated.timestamp <= current_time_ms
        ).order_by(MarketTradesAggregated.timestamp).all()

        if not records:
            return {}

        buckets = {}
        for ts, buy, sell in records:
            bucket_ts = floor_timestamp(ts, interval_ms)
            if bucket_ts not in buckets:
                buckets[bucket_ts] = {"buy": 0, "sell": 0}
            buckets[bucket_ts]["buy"] += float(buy or 0)
            buckets[bucket_ts]["sell"] += float(sell or 0)

        result = {}
        for ts in buckets:
            sell = buckets[ts]["sell"]
            if sell > 0:
                result[ts] = buckets[ts]["buy"] / sell

        return result

    def _compute_funding_buckets(
        self, db, symbol, interval_ms, start_time_ms, current_time_ms
    ) -> Dict[int, float]:
        """Compute funding rate for each bucket."""
        from services.market_flow_indicators import floor_timestamp
        from database.models import MarketAssetMetrics

        records = db.query(
            MarketAssetMetrics.timestamp,
            MarketAssetMetrics.funding_rate
        ).filter(
            MarketAssetMetrics.symbol == symbol.upper(),
            MarketAssetMetrics.timestamp >= start_time_ms,
            MarketAssetMetrics.timestamp <= current_time_ms,
            MarketAssetMetrics.funding_rate.isnot(None)
        ).order_by(MarketAssetMetrics.timestamp).all()

        if not records:
            return {}

        buckets = {}
        for ts, funding in records:
            bucket_ts = floor_timestamp(ts, interval_ms)
            buckets[bucket_ts] = float(funding) * 100  # Convert to percentage

        return buckets

    def _compute_oi_buckets(
        self, db, symbol, interval_ms, start_time_ms, current_time_ms
    ) -> Dict[int, float]:
        """Compute absolute OI for each bucket."""
        from services.market_flow_indicators import floor_timestamp
        from database.models import MarketAssetMetrics

        records = db.query(
            MarketAssetMetrics.timestamp,
            MarketAssetMetrics.open_interest
        ).filter(
            MarketAssetMetrics.symbol == symbol.upper(),
            MarketAssetMetrics.timestamp >= start_time_ms,
            MarketAssetMetrics.timestamp <= current_time_ms,
            MarketAssetMetrics.open_interest.isnot(None)
        ).order_by(MarketAssetMetrics.timestamp).all()

        if not records:
            return {}

        buckets = {}
        for ts, oi in records:
            bucket_ts = floor_timestamp(ts, interval_ms)
            buckets[bucket_ts] = float(oi)

        return buckets

    def _find_taker_triggers(
        self, db: Session, signal_def: Dict, symbol: str, klines: List[Dict], time_window: str
    ) -> List[Dict]:
        """
        Find taker_volume composite signal triggers.

        IMPORTANT: This method iterates over MARKET FLOW DATA (buckets), not K-lines.
        K-lines are only used as a visual background.
        """
        condition = signal_def.get("trigger_condition", {})
        direction = condition.get("direction", "any")
        ratio_threshold = condition.get("ratio_threshold", 1.5)
        volume_threshold = condition.get("volume_threshold", 0)

        interval_ms = TIMEFRAME_MS.get(time_window, 300000)

        # Compute all taker volume buckets from market flow data
        taker_buckets = self._compute_taker_volume_buckets(db, symbol, interval_ms)
        if not taker_buckets:
            return []

        # Build K-line price lookup
        from services.market_flow_indicators import floor_timestamp
        kline_prices = {}
        for kline in klines:
            bucket_ts = floor_timestamp(kline["timestamp"], interval_ms)
            kline_prices[bucket_ts] = kline["close"]

        # Get K-line time range
        if not klines:
            return []
        kline_min_ts = min(floor_timestamp(k["timestamp"], interval_ms) for k in klines)
        kline_max_ts = max(floor_timestamp(k["timestamp"], interval_ms) for k in klines)

        # Iterate over all buckets and find triggers
        triggers = []
        for bucket_ts, data in sorted(taker_buckets.items()):
            # Only include triggers within K-line display range
            if bucket_ts < kline_min_ts or bucket_ts > kline_max_ts:
                continue

            ratio = data["ratio"]
            total = data["volume"]

            if total < volume_threshold:
                continue

            triggered = False
            actual_dir = None

            if direction == "buy" and ratio >= ratio_threshold:
                triggered, actual_dir = True, "buy"
            elif direction == "sell" and ratio <= 1 / ratio_threshold:
                triggered, actual_dir = True, "sell"
            elif direction == "any":
                if ratio >= ratio_threshold:
                    triggered, actual_dir = True, "buy"
                elif ratio <= 1 / ratio_threshold:
                    triggered, actual_dir = True, "sell"

            if triggered:
                triggers.append({
                    "timestamp": bucket_ts,
                    "direction": actual_dir,
                    "ratio": ratio,
                    "ratio_threshold": ratio_threshold,
                    "volume": total,
                    "volume_threshold": volume_threshold,
                    "price": kline_prices.get(bucket_ts, 0),
                })

        return triggers

    def _compute_taker_volume_buckets(
        self, db, symbol, interval_ms
    ) -> Dict[int, Dict]:
        """Compute taker volume data (ratio and volume) for each bucket."""
        from services.market_flow_indicators import floor_timestamp
        from database.models import MarketTradesAggregated
        from datetime import datetime

        current_time_ms = int(datetime.utcnow().timestamp() * 1000)
        start_time_ms = current_time_ms - (7 * 24 * 60 * 60 * 1000)

        records = db.query(
            MarketTradesAggregated.timestamp,
            MarketTradesAggregated.taker_buy_notional,
            MarketTradesAggregated.taker_sell_notional
        ).filter(
            MarketTradesAggregated.symbol == symbol.upper(),
            MarketTradesAggregated.timestamp >= start_time_ms,
            MarketTradesAggregated.timestamp <= current_time_ms
        ).order_by(MarketTradesAggregated.timestamp).all()

        if not records:
            return {}

        buckets = {}
        for ts, buy, sell in records:
            bucket_ts = floor_timestamp(ts, interval_ms)
            if bucket_ts not in buckets:
                buckets[bucket_ts] = {"buy": 0, "sell": 0}
            buckets[bucket_ts]["buy"] += float(buy or 0)
            buckets[bucket_ts]["sell"] += float(sell or 0)

        # Calculate ratio and volume for each bucket
        result = {}
        for ts, data in buckets.items():
            buy, sell = data["buy"], data["sell"]
            total = buy + sell
            if sell > 0 and total > 0:
                result[ts] = {
                    "ratio": buy / sell,
                    "volume": total
                }

        return result

    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate if a condition is met."""
        # Support both symbol and text forms of operators
        if operator in (">", "greater_than", "gt"):
            return value > threshold
        elif operator in (">=", "greater_than_or_equal", "gte"):
            return value >= threshold
        elif operator in ("<", "less_than", "lt"):
            return value < threshold
        elif operator in ("<=", "less_than_or_equal", "lte"):
            return value <= threshold
        elif operator in ("==", "equal", "eq"):
            return abs(value - threshold) < 1e-9
        elif operator in ("!=", "not_equal", "ne"):
            return abs(value - threshold) >= 1e-9
        elif operator in ("abs_greater_than", "abs_gt"):
            return abs(value) > threshold
        elif operator in ("abs_less_than", "abs_lt"):
            return abs(value) < threshold
        return False


# Singleton instance
signal_backtest_service = SignalBacktestService()
