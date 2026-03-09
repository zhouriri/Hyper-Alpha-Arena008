"""
Factor Effectiveness Service

Computes IC, ICIR, win rate for each factor by back-calculating factor values
directly from K-line data (500 bars lookback). Does NOT depend on factor_values
table accumulation.

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
from services.market_data import get_kline_data
from services.technical_indicators import calculate_indicators
from services.scheduler import task_scheduler

logger = logging.getLogger(__name__)

FORWARD_PERIODS = {"1h": 1, "4h": 4, "12h": 12, "24h": 24}
LOOKBACK_BARS = 500


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
        """Back-calculate factor values from K-lines, then compute IC for each factor."""
        market = "binance" if exchange == "binance" else "CRYPTO"
        klines = get_kline_data(symbol, market=market, period=period, count=LOOKBACK_BARS)
        if not klines or len(klines) < 50:
            return 0

        # Batch compute technical indicators for all klines
        tech_keys = list({f["indicator_key"] for f in FACTOR_REGISTRY if f["compute_type"] == "technical"})
        indicators = calculate_indicators(klines, tech_keys)

        # Build factor value series: for each bar, extract all factor values
        from services.factor_computation_service import FactorComputationService
        svc = FactorComputationService()

        n_bars = len(klines)
        factor_series: Dict[str, List[Optional[float]]] = {f["name"]: [] for f in FACTOR_REGISTRY}

        for i in range(n_bars):
            sub_klines = klines[:i+1]
            sub_indicators = self._slice_indicators(indicators, i)
            for fdef in FACTOR_REGISTRY:
                val = svc._extract_value(fdef, sub_indicators, sub_klines, db, symbol, period, exchange)
                factor_series[fdef["name"]].append(val)

        # Compute forward returns for each bar
        closes = [float(k["close"]) for k in klines]
        count = 0
        for fdef in FACTOR_REGISTRY:
            fvals = factor_series[fdef["name"]]
            for fp_label, fp_hours in FORWARD_PERIODS.items():
                offset = fp_hours  # bars offset (1h klines, so 1 bar = 1h)
                aligned_fv, aligned_rt = self._align_series(fvals, closes, offset, n_bars)
                if len(aligned_fv) < 10:
                    continue
                metrics = self._calc_metrics(aligned_fv, aligned_rt)
                self._upsert(db, exchange, fdef["name"], fdef["category"],
                             symbol, period, fp_label, today, n_bars, metrics)
                count += 1
        return count

    def _slice_indicators(self, indicators, idx):
        """Slice indicator results up to index idx (inclusive)."""
        sliced = {}
        for key, val in indicators.items():
            if isinstance(val, dict):
                sliced[key] = {}
                for k, v in val.items():
                    if isinstance(v, (list, np.ndarray)):
                        sliced[key][k] = v[:idx+1] if len(v) > idx else v
                    else:
                        sliced[key][k] = v
            elif isinstance(val, (list, np.ndarray)):
                sliced[key] = val[:idx+1] if len(val) > idx else val
            else:
                sliced[key] = val
        return sliced

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
        n = len(factor_vals)
        fv = np.array(factor_vals)
        rt = np.array(returns)

        ic, _ = spearmanr(fv, rt)
        ic = float(ic) if not np.isnan(ic) else 0.0

        # Rolling IC for ICIR (window = 20 bars or n//4, whichever smaller)
        window = min(20, max(5, n // 4))
        ics = []
        if window >= 5 and n >= window * 2:
            for i in range(n - window + 1):
                c, _ = spearmanr(fv[i:i+window], rt[i:i+window])
                if not np.isnan(c):
                    ics.append(c)
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
