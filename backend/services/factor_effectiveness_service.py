"""
Factor Effectiveness Service

Computes IC, ICIR, win rate for each factor using full local K-line history.
Uses vectorized extraction (no per-bar loops). Auto-backfills from exchange
APIs when local data is insufficient.

Runs daily via CronTrigger at UTC 01:00, also callable on-demand.
"""

import logging
import time
from datetime import date
from typing import List, Dict, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.connection import SessionLocal
from services.factor_registry import FACTOR_REGISTRY
from services.technical_indicators import calculate_indicators
from services.scheduler import task_scheduler

logger = logging.getLogger(__name__)


class FactorEffectivenessService:
    def __init__(self):
        self._running = False
        self._progress: Dict = {"status": "idle"}

    def start(self):
        if self._running:
            return
        self._running = True
        from apscheduler.triggers.cron import CronTrigger
        task_scheduler.scheduler.add_job(
            func=self._run,
            trigger=CronTrigger(hour=1, minute=0),
            id="factor_effectiveness_daily",
            replace_existing=True,
            max_instances=1, coalesce=True,
        )
        print("[FactorEffectiveness] Scheduled daily at UTC 01:00", flush=True)

    def stop(self):
        self._running = False

    def _run(self):
        db: Session = SessionLocal()
        try:
            for ex in ["hyperliquid", "binance"]:
                self.compute_for_exchange(db, ex)
        except Exception as e:
            logger.error(f"[FactorEffectiveness] error: {e}")
        finally:
            db.close()

    def get_progress(self) -> Dict:
        return dict(self._progress)

    def compute_for_exchange(self, db: Session, exchange: str, period: str = "1h"):
        """Compute effectiveness for all factors on one exchange. Public API."""
        symbols = self._get_symbols(db, exchange)
        if not symbols:
            self._progress = {"status": "idle"}
            return {"computed": 0, "exchange": exchange}
        today = date.today()
        total = len(symbols)
        self._progress = {
            "status": "running", "phase": "effectiveness",
            "completed": 0, "total": total, "current_symbol": "",
        }
        count = 0
        for i, symbol in enumerate(symbols):
            self._progress["current_symbol"] = symbol
            self._progress["completed"] = i
            try:
                n = self._compute_symbol(db, exchange, symbol, period, today)
                count += n
            except Exception as e:
                logger.warning(f"[FactorEffectiveness] {exchange}/{symbol}: {e}")
        db.commit()
        self._progress = {"status": "idle"}
        print(f"[FactorEffectiveness] {exchange}: {count} records", flush=True)
        return {"computed": count, "exchange": exchange}

    # ── internal ──

    def _get_symbols(self, db, exchange):
        try:
            if exchange == "binance":
                from services.binance_symbol_service import get_selected_symbols
            else:
                from services.hyperliquid_symbol_service import get_selected_symbols
            return get_selected_symbols()
        except Exception:
            rows = db.execute(text(
                "SELECT DISTINCT symbol FROM crypto_klines "
                "WHERE exchange = :ex AND period = '1h' LIMIT 50"
            ), {"ex": exchange}).fetchall()
            return [r[0] for r in rows]

    def _compute_symbol(self, db, exchange, symbol, period, today) -> int:
        """Vectorized factor effectiveness computation using full local kline history."""
        from services.factor_data_provider import ensure_kline_coverage

        FORWARD_PERIODS = {"1h": 1, "4h": 4, "12h": 12, "24h": 24}

        klines = ensure_kline_coverage(db, exchange, symbol, period)
        if not klines or len(klines) < 50:
            return 0

        n_bars = len(klines)
        closes = [float(k["close"]) for k in klines]

        # Batch compute technical indicators (vectorized, one pass)
        tech_keys = list({f["indicator_key"] for f in FACTOR_REGISTRY if f["compute_type"] == "technical"})
        indicators = calculate_indicators(klines, tech_keys)

        # Extract full series for each factor (vectorized, no per-bar loop)
        factor_series: Dict[str, List[Optional[float]]] = {}
        for fdef in FACTOR_REGISTRY:
            series = self._extract_full_series(fdef, indicators, klines, n_bars, db, symbol, exchange)
            if series is not None:
                factor_series[fdef["name"]] = (series, fdef["category"])

        # Compute IC for each factor × forward period
        count = 0
        for fname, (fvals, fcat) in factor_series.items():
            for fp_label, fp_hours in FORWARD_PERIODS.items():
                aligned_fv, aligned_rt = self._align_series(fvals, closes, fp_hours, n_bars)
                if len(aligned_fv) < 10:
                    continue
                metrics = self._calc_metrics(aligned_fv, aligned_rt)
                self._upsert(db, exchange, fname, fcat,
                             symbol, period, fp_label, today, n_bars, metrics)
                count += 1

        # Custom factors via expression engine (already vectorized)
        count += self._compute_custom_effectiveness(
            db, exchange, symbol, period, klines, closes, n_bars, today,
        )
        return count

    def _extract_full_series(self, factor_def, indicators, klines, n_bars, db=None, symbol=None, exchange=None):
        """Extract a complete factor value series (vectorized). Returns list or None."""
        ctype = factor_def["compute_type"]

        if ctype == "technical":
            return self._extract_technical_series(factor_def, indicators, klines, n_bars)

        if ctype == "derived":
            return self._extract_derived_series(factor_def, klines, n_bars)

        if ctype == "microstructure" and db and symbol:
            return self._extract_microstructure_series(
                factor_def, klines, db, symbol, exchange or "hyperliquid"
            )

        return None

    def _extract_microstructure_series(self, factor_def, klines, db, symbol, exchange):
        """Extract historical series for microstructure factors aligned to kline timestamps."""
        from sqlalchemy import text

        indicator_key = factor_def.get("indicator_key", "")
        # Build hourly timestamp list from klines (ms)
        ts_list = [int(k["timestamp"]) * 1000 if k["timestamp"] < 1e12 else int(k["timestamp"]) for k in klines]
        hour_ms = 3600 * 1000
        ts_min, ts_max = ts_list[0], ts_list[-1] + hour_ms

        if indicator_key == "OI_DELTA":
            rows = db.execute(text("""
                SELECT timestamp, open_interest FROM market_asset_metrics
                WHERE symbol = :s AND exchange = :e AND open_interest IS NOT NULL
                    AND timestamp >= :tmin AND timestamp < :tmax
                ORDER BY timestamp ASC
            """), {"s": symbol, "e": exchange, "tmin": ts_min, "tmax": ts_max}).fetchall()
            if len(rows) < 10:
                return None
            return self._align_flow_to_klines_delta(rows, ts_list, hour_ms, col_idx=1)

        if indicator_key == "FUNDING":
            rows = db.execute(text("""
                SELECT timestamp, funding_rate FROM market_asset_metrics
                WHERE symbol = :s AND exchange = :e AND funding_rate IS NOT NULL
                    AND timestamp >= :tmin AND timestamp < :tmax
                ORDER BY timestamp ASC
            """), {"s": symbol, "e": exchange, "tmin": ts_min, "tmax": ts_max}).fetchall()
            if len(rows) < 10:
                return None
            return self._align_flow_to_klines_avg(rows, ts_list, hour_ms, col_idx=1)

        if indicator_key in ("CVD", "TAKER"):
            rows = db.execute(text("""
                SELECT timestamp, taker_buy_volume, taker_sell_volume
                FROM market_trades_aggregated
                WHERE symbol = :s AND exchange = :e
                    AND timestamp >= :tmin AND timestamp < :tmax
                ORDER BY timestamp ASC
            """), {"s": symbol, "e": exchange, "tmin": ts_min, "tmax": ts_max}).fetchall()
            if len(rows) < 10:
                return None
            if indicator_key == "CVD":
                return self._align_flow_to_klines_cvd(rows, ts_list, hour_ms)
            else:
                return self._align_flow_to_klines_taker_ratio(rows, ts_list, hour_ms)

        if indicator_key == "DEPTH":
            rows = db.execute(text("""
                SELECT timestamp, bid_depth_5, ask_depth_5
                FROM market_orderbook_snapshots
                WHERE symbol = :s AND exchange = :e
                    AND timestamp >= :tmin AND timestamp < :tmax
                ORDER BY timestamp ASC
            """), {"s": symbol, "e": exchange, "tmin": ts_min, "tmax": ts_max}).fetchall()
            if len(rows) < 10:
                return None
            return self._align_flow_to_klines_depth(rows, ts_list, hour_ms)

        return None

    def _align_flow_to_klines_avg(self, rows, ts_list, hour_ms, col_idx):
        """Average value per kline hour bucket."""
        result = []
        row_idx = 0
        n_rows = len(rows)
        for ts in ts_list:
            vals = []
            while row_idx < n_rows and rows[row_idx][0] < ts:
                row_idx += 1
            j = row_idx
            while j < n_rows and rows[j][0] < ts + hour_ms:
                v = rows[j][col_idx]
                if v is not None:
                    vals.append(float(v))
                j += 1
            result.append(sum(vals) / len(vals) if vals else None)
        return result

    def _align_flow_to_klines_delta(self, rows, ts_list, hour_ms, col_idx):
        """Percentage change of value over each kline hour."""
        result = []
        row_idx = 0
        n_rows = len(rows)
        for ts in ts_list:
            while row_idx < n_rows and rows[row_idx][0] < ts:
                row_idx += 1
            start_val = None
            end_val = None
            j = row_idx
            while j < n_rows and rows[j][0] < ts + hour_ms:
                v = rows[j][col_idx]
                if v is not None:
                    fv = float(v)
                    if start_val is None:
                        start_val = fv
                    end_val = fv
                j += 1
            if start_val and end_val and start_val != 0:
                result.append((end_val - start_val) / start_val * 100)
            else:
                result.append(None)
        return result

    def _align_flow_to_klines_cvd(self, rows, ts_list, hour_ms):
        """Cumulative volume delta per hour: sum(buy - sell)."""
        result = []
        row_idx = 0
        n_rows = len(rows)
        for ts in ts_list:
            cvd = 0.0
            count = 0
            while row_idx < n_rows and rows[row_idx][0] < ts:
                row_idx += 1
            j = row_idx
            while j < n_rows and rows[j][0] < ts + hour_ms:
                buy = float(rows[j][1] or 0)
                sell = float(rows[j][2] or 0)
                cvd += buy - sell
                count += 1
                j += 1
            result.append(cvd if count > 0 else None)
        return result

    def _align_flow_to_klines_taker_ratio(self, rows, ts_list, hour_ms):
        """Taker buy ratio per hour: sum(buy) / sum(buy+sell)."""
        result = []
        row_idx = 0
        n_rows = len(rows)
        for ts in ts_list:
            total_buy = 0.0
            total_sell = 0.0
            count = 0
            while row_idx < n_rows and rows[row_idx][0] < ts:
                row_idx += 1
            j = row_idx
            while j < n_rows and rows[j][0] < ts + hour_ms:
                total_buy += float(rows[j][1] or 0)
                total_sell += float(rows[j][2] or 0)
                count += 1
                j += 1
            total = total_buy + total_sell
            result.append(total_buy / total if total > 0 and count > 0 else None)
        return result

    def _align_flow_to_klines_depth(self, rows, ts_list, hour_ms):
        """Average depth ratio per hour: bid_depth / ask_depth."""
        result = []
        row_idx = 0
        n_rows = len(rows)
        for ts in ts_list:
            ratios = []
            while row_idx < n_rows and rows[row_idx][0] < ts:
                row_idx += 1
            j = row_idx
            while j < n_rows and rows[j][0] < ts + hour_ms:
                bid = float(rows[j][1] or 0)
                ask = float(rows[j][2] or 0)
                if ask > 0:
                    ratios.append(bid / ask)
                j += 1
            result.append(sum(ratios) / len(ratios) if ratios else None)
        return result

    def _extract_technical_series(self, factor_def, indicators, klines, n_bars):
        """Extract full series from technical indicator results."""
        key = factor_def["indicator_key"]
        raw = indicators.get(key)
        if raw is None:
            return None

        extract_key = factor_def.get("extract")
        normalize = factor_def.get("normalize")

        if isinstance(raw, dict):
            if extract_key == "width" and "upper" in raw and "middle" in raw and "lower" in raw:
                u, m, lo = np.array(raw["upper"], dtype=float), np.array(raw["middle"], dtype=float), np.array(raw["lower"], dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.where(m != 0, (u - lo) / m, np.nan)
                return [None if np.isnan(v) else float(v) for v in result]

            if extract_key == "percent_b" and "upper" in raw and "lower" in raw:
                u, lo = np.array(raw["upper"], dtype=float), np.array(raw["lower"], dtype=float)
                cl = np.array([float(k["close"]) for k in klines], dtype=float)
                denom = u - lo
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.where(denom != 0, (cl - lo) / denom, np.nan)
                return [None if np.isnan(v) else float(v) for v in result]

            if extract_key and extract_key in raw:
                series = raw[extract_key]
                if series and len(series) >= n_bars:
                    return [float(v) if v is not None and not np.isnan(v) else None for v in series[:n_bars]]
                return None

            # Stochastic K/D fallback
            if extract_key == "k":
                for k_key in ("STOCHk", "k", "%K"):
                    if k_key in raw and raw[k_key]:
                        return [float(v) if v is not None and not np.isnan(v) else None for v in raw[k_key][:n_bars]]
            if extract_key == "d":
                for d_key in ("STOCHd", "d", "%D"):
                    if d_key in raw and raw[d_key]:
                        return [float(v) if v is not None and not np.isnan(v) else None for v in raw[d_key][:n_bars]]
            return None

        if isinstance(raw, list) and len(raw) >= n_bars:
            if normalize == "price_deviation":
                cl = [float(k["close"]) for k in klines]
                return [
                    ((cl[i] - raw[i]) / cl[i]) if cl[i] != 0 and raw[i] is not None else None
                    for i in range(n_bars)
                ]
            return [float(v) if v is not None else None for v in raw[:n_bars]]

        return None

    def _extract_derived_series(self, factor_def, klines, n_bars):
        """Extract full series for derived factors (ROC, volume ratio)."""
        derive = factor_def.get("derive_from")
        plen = factor_def.get("period_len", 10)

        if derive == "close":
            closes = np.array([float(k["close"]) for k in klines], dtype=float)
            result = [None] * min(plen, n_bars)
            for i in range(plen, n_bars):
                prev = closes[i - plen]
                result.append((closes[i] - prev) / prev if prev != 0 else None)
            return result

        if derive == "volume_ratio":
            vols = np.array([float(k["volume"]) for k in klines], dtype=float)
            result = [None] * min(plen, n_bars)
            for i in range(plen, n_bars):
                avg = vols[i - plen:i].mean()
                result.append(vols[i] / avg if avg != 0 else None)
            return result

        return None

    def _compute_custom_effectiveness(self, db, exchange, symbol, period, klines, closes, n_bars, today):
        """Compute IC/ICIR/win_rate for active custom factors."""
        import pandas as pd
        from database.models import CustomFactor
        from services.factor_expression_engine import factor_expression_engine

        forward_periods = {"1h": 1, "4h": 4, "12h": 12, "24h": 24}

        try:
            custom_factors = db.query(CustomFactor).filter(CustomFactor.is_active == True).all()
        except Exception:
            return 0

        count = 0
        for cf in custom_factors:
            try:
                series, err = factor_expression_engine.execute(cf.expression, klines)
                if series is None or len(series) != n_bars:
                    continue
                fvals = [None if pd.isna(v) else float(v) for v in series.tolist()]
                for fp_label, fp_hours in forward_periods.items():
                    aligned_fv, aligned_rt = self._align_series(fvals, closes, fp_hours, n_bars)
                    if len(aligned_fv) < 10:
                        continue
                    metrics = self._calc_metrics(aligned_fv, aligned_rt)
                    self._upsert(db, exchange, cf.name, "custom",
                                 symbol, period, fp_label, today, n_bars, metrics)
                    count += 1
            except Exception as e:
                logger.warning(f"[FactorEffectiveness] custom '{cf.name}' err: {e}")
        return count

    def _align_series(self, fvals, closes, offset, n_bars):
        """Align factor values with forward returns, filtering None."""
        aligned_fv, aligned_rt = [], []
        for i in range(n_bars - offset):
            fv = fvals[i]
            if fv is None or closes[i] == 0:
                continue
            ret = (closes[i + offset] - closes[i]) / closes[i]
            aligned_fv.append(fv)
            aligned_rt.append(ret)
        return aligned_fv, aligned_rt

    def _calc_metrics(self, factor_vals, returns):
        """Compute IC, ICIR, win_rate from aligned factor values and returns."""
        from scipy.stats import spearmanr
        import pandas as pd

        n = len(factor_vals)
        fv = np.array(factor_vals, dtype=float)
        rt = np.array(returns, dtype=float)

        ic, _ = spearmanr(fv, rt)
        ic = float(ic) if not np.isnan(ic) else 0.0

        # Rolling IC for ICIR using rank-based Pearson (vectorized via pandas)
        # Use non-overlapping windows for large datasets to avoid O(n) scipy calls
        window = min(50, max(20, n // 20))
        ics = []
        if window >= 10 and n >= window * 3:
            fv_rank = pd.Series(fv).rolling(window).rank()
            rt_rank = pd.Series(rt).rolling(window).rank()
            # Sample at window-stride intervals to avoid redundant overlapping windows
            stride = max(1, window // 2)
            for i in range(window - 1, n, stride):
                fr = fv_rank.iloc[i - window + 1:i + 1].values
                rr = rt_rank.iloc[i - window + 1:i + 1].values
                valid = ~(np.isnan(fr) | np.isnan(rr))
                if valid.sum() < 10:
                    continue
                fr_v, rr_v = fr[valid], rr[valid]
                denom = np.std(fr_v) * np.std(rr_v)
                if denom > 1e-10:
                    corr = np.corrcoef(fr_v, rr_v)[0, 1]
                    if not np.isnan(corr):
                        ics.append(corr)

        ic_mean = float(np.mean(ics)) if ics else ic
        ic_std = float(np.std(ics)) if ics else 0.0
        icir = ic_mean / ic_std if ic_std > 1e-8 else 0.0

        signs_match = np.sign(fv) == np.sign(rt)
        win_rate = float(signs_match.mean())

        return {
            "ic_mean": round(ic_mean, 6), "ic_std": round(ic_std, 6),
            "icir": round(icir, 4), "win_rate": round(win_rate, 4),
            "sample_count": n, "decay_half_life": None,
        }

    def _upsert(self, db, exchange, fname, fcat, symbol, period, fp, calc_date, lookback, m):
        db.execute(text("""
            INSERT INTO factor_effectiveness
                (exchange, factor_name, factor_category, symbol, period, forward_period,
                 calc_date, lookback_days, ic_mean, ic_std, icir,
                 win_rate, decay_half_life, sample_count)
            VALUES
                (:ex, :fn, :fc, :sym, :p, :fp, :cd, :lb, :icm, :ics, :icir,
                 :wr, :dhl, :sc)
            ON CONFLICT (exchange, factor_name, symbol, period, forward_period, calc_date)
            DO UPDATE SET
                ic_mean = EXCLUDED.ic_mean, ic_std = EXCLUDED.ic_std,
                icir = EXCLUDED.icir, win_rate = EXCLUDED.win_rate,
                sample_count = EXCLUDED.sample_count
        """), {
            "ex": exchange, "fn": fname, "fc": fcat, "sym": symbol,
            "p": period, "fp": fp, "cd": calc_date, "lb": lookback,
            "icm": m["ic_mean"], "ics": m["ic_std"], "icir": m["icir"],
            "wr": m["win_rate"], "dhl": m["decay_half_life"], "sc": m["sample_count"],
        })


# Singleton
factor_effectiveness_service = FactorEffectivenessService()
