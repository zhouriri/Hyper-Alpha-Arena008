"""
Factor System API Routes

GET  /api/factors/library                       → Factor registry
GET  /api/factors/values?exchange=&symbol=&period= → Latest factor values
GET  /api/factors/effectiveness?exchange=&symbol=  → Effectiveness ranking
GET  /api/factors/effectiveness/{name}/history     → IC trend
GET  /api/factors/status                          → Engine status
GET  /api/factors/compute/estimate                → Pre-compute estimation
POST /api/factors/compute                         → Manual trigger (async)
GET  /api/factors/compute/progress                → Computation progress
"""

import threading
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.connection import SessionLocal
from database.models import CustomFactor
from services.factor_registry import FACTOR_REGISTRY, FACTOR_CATEGORIES, CATEGORY_LABELS
from services.factor_expression_engine import factor_expression_engine

# Track background compute task
_compute_lock = threading.Lock()
_compute_result: Optional[dict] = None
_compute_running = False

router = APIRouter(prefix="/api/factors", tags=["factors"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/library")
async def get_factor_library(db: Session = Depends(get_db)):
    """Return full factor registry: built-in + custom factors."""
    # Built-in factors with source tag
    builtin = [{**f, "source": "builtin"} for f in FACTOR_REGISTRY]

    # Custom factors from DB
    custom_rows = db.query(CustomFactor).filter(CustomFactor.is_active == True).all()
    custom = [
        {
            "name": cf.name, "category": "custom",
            "display_name": cf.name, "display_name_zh": cf.name,
            "description": cf.expression, "description_zh": cf.expression,
            "source": cf.source or "custom",
            "expression": cf.expression,
            "custom_id": cf.id,
        }
        for cf in custom_rows
    ]

    return {
        "factors": builtin + custom,
        "categories": FACTOR_CATEGORIES,
        "category_labels": {**CATEGORY_LABELS, "custom": {"en": "Custom", "zh": "自定义"}},
    }


@router.get("/values")
async def get_factor_values(
    symbol: str = Query(...),
    period: str = Query("1h"),
    exchange: str = Query("hyperliquid"),
    db: Session = Depends(get_db),
):
    """Return latest factor values for a symbol/period/exchange."""
    rows = db.execute(text("""
        SELECT DISTINCT ON (factor_name)
            factor_name, factor_category, value, timestamp
        FROM factor_values
        WHERE symbol = :sym AND period = :p AND exchange = :ex
        ORDER BY factor_name, timestamp DESC
    """), {"sym": symbol, "p": period, "ex": exchange}).fetchall()

    return {
        "symbol": symbol, "period": period, "exchange": exchange,
        "values": [
            {"factor_name": r[0], "category": r[1], "value": r[2], "timestamp": r[3]}
            for r in rows
        ],
    }


@router.get("/effectiveness")
async def get_factor_effectiveness(
    symbol: str = Query(...),
    period: str = Query("1h"),
    forward_period: str = Query("4h"),
    exchange: str = Query("hyperliquid"),
    sort_by: str = Query("icir"),
    db: Session = Depends(get_db),
):
    """Return effectiveness ranking for a symbol/period/forward_period."""
    valid_sorts = {"icir", "ic_mean", "win_rate", "sample_count"}
    col = sort_by if sort_by in valid_sorts else "icir"

    rows = db.execute(text(f"""
        SELECT DISTINCT ON (factor_name)
            factor_name, factor_category, ic_mean, ic_std, icir,
            win_rate, decay_half_life, sample_count, calc_date
        FROM factor_effectiveness
        WHERE symbol = :sym AND period = :p AND forward_period = :fp AND exchange = :ex
        ORDER BY factor_name, calc_date DESC
    """), {"sym": symbol, "p": period, "fp": forward_period, "ex": exchange}).fetchall()

    items = [
        {
            "factor_name": r[0], "category": r[1],
            "ic_mean": r[2], "ic_std": r[3], "icir": r[4],
            "win_rate": r[5], "decay_half_life": r[6],
            "sample_count": r[7], "calc_date": str(r[8]),
        }
        for r in rows
    ]
    items.sort(key=lambda x: abs(x.get(col) or 0), reverse=True)

    return {
        "symbol": symbol, "period": period,
        "forward_period": forward_period, "exchange": exchange,
        "items": items,
    }


@router.get("/effectiveness/{factor_name}/history")
async def get_effectiveness_history(
    factor_name: str,
    symbol: str = Query(...),
    period: str = Query("1h"),
    forward_period: str = Query("4h"),
    exchange: str = Query("hyperliquid"),
    days: int = Query(30),
    db: Session = Depends(get_db),
):
    """Return IC trend over time for a specific factor."""
    cutoff = date.today() - timedelta(days=days)
    rows = db.execute(text("""
        SELECT calc_date, ic_mean, icir, win_rate, sample_count
        FROM factor_effectiveness
        WHERE factor_name = :fn AND symbol = :sym AND period = :p
            AND forward_period = :fp AND calc_date >= :cutoff
        ORDER BY calc_date
    """), {"fn": factor_name, "sym": symbol, "p": period,
           "fp": forward_period, "cutoff": cutoff}).fetchall()

    return {
        "factor_name": factor_name,
        "history": [
            {"date": str(r[0]), "ic_mean": r[1], "icir": r[2],
             "win_rate": r[3], "sample_count": r[4]}
            for r in rows
        ],
    }


@router.get("/status")
async def get_factor_status(db: Session = Depends(get_db)):
    """Return engine status with per-exchange last compute time."""
    import os
    enabled = os.getenv("FACTOR_ENGINE_ENABLED", "false").lower() == "true"

    from services.factor_computation_service import factor_computation_service

    stats = db.execute(text("""
        SELECT COUNT(*), COUNT(DISTINCT symbol), MAX(timestamp), MAX(created_at)
        FROM factor_values
    """)).fetchone()

    eff_stats = db.execute(text("""
        SELECT COUNT(*), MAX(calc_date) FROM factor_effectiveness
    """)).fetchone()

    # Use DB created_at as persistent last compute time, fallback to in-memory
    db_last_compute = None
    if stats and stats[3]:
        db_last_compute = stats[3].timestamp() if hasattr(stats[3], 'timestamp') else None

    mem_hl = factor_computation_service.get_last_compute_time("hyperliquid")
    mem_bn = factor_computation_service.get_last_compute_time("binance")

    return {
        "enabled": enabled,
        "total_factor_values": stats[0] if stats else 0,
        "symbols_covered": stats[1] if stats else 0,
        "latest_computation_ts": stats[2] if stats else None,
        "total_effectiveness_records": eff_stats[0] if eff_stats else 0,
        "latest_effectiveness_date": str(eff_stats[1]) if eff_stats and eff_stats[1] else None,
        "registered_factors": len(FACTOR_REGISTRY),
        "last_compute_time": {
            "hyperliquid": mem_hl or db_last_compute,
            "binance": mem_bn or db_last_compute,
        },
        "compute_interval_seconds": 3600,
    }


class ComputeRequest(BaseModel):
    exchange: str = "hyperliquid"
    period: str = "1h"


def _run_compute_background(exchange: str, period: str):
    """Run factor computation in a background thread."""
    global _compute_result, _compute_running
    from services.factor_computation_service import factor_computation_service
    from services.factor_effectiveness_service import factor_effectiveness_service

    try:
        val_result = factor_computation_service.compute_now(exchange, period)
        db = SessionLocal()
        try:
            eff_result = factor_effectiveness_service.compute_for_exchange(db, exchange, period)
        finally:
            db.close()
        _compute_result = {
            "status": "done",
            "exchange": exchange,
            "values_computed": val_result.get("computed", 0),
            "effectiveness_computed": eff_result.get("computed", 0),
        }
    except Exception as e:
        _compute_result = {"status": "error", "error": str(e)}
    finally:
        _compute_running = False


@router.get("/compute/estimate")
async def compute_estimate(exchange: str = Query("hyperliquid"), db: Session = Depends(get_db)):
    """Return symbol list, data coverage, and estimated duration."""
    from services.factor_computation_service import factor_computation_service
    from services.factor_registry import FACTOR_REGISTRY

    symbols = factor_computation_service.get_symbols(exchange)
    factor_count = len(FACTOR_REGISTRY)

    # Query actual data coverage per symbol
    coverage = {}
    if symbols:
        rows = db.execute(text("""
            SELECT symbol, COUNT(*) as cnt
            FROM crypto_klines
            WHERE exchange = :ex AND period = '1h' AND symbol = ANY(:syms)
            GROUP BY symbol
        """), {"ex": exchange, "syms": symbols}).fetchall()
        coverage = {r[0]: r[1] for r in rows}

    total_bars = sum(coverage.values())
    avg_bars = total_bars // len(symbols) if symbols else 0
    # Vectorized: ~2s per symbol for indicator computation + IC calculation
    estimated_seconds = len(symbols) * 2

    return {
        "exchange": exchange,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "factor_count": factor_count,
        "forward_periods": ["1h", "4h", "12h", "24h"],
        "avg_bars_per_symbol": avg_bars,
        "total_bars": total_bars,
        "estimated_seconds": estimated_seconds,
    }


@router.post("/compute")
async def trigger_compute(req: ComputeRequest):
    """Start factor computation in background thread. Returns immediately."""
    global _compute_result, _compute_running

    with _compute_lock:
        if _compute_running:
            return {"status": "already_running"}
        _compute_running = True
        _compute_result = None

    t = threading.Thread(
        target=_run_compute_background,
        args=(req.exchange, req.period),
        daemon=True,
    )
    t.start()
    return {"status": "started", "exchange": req.exchange}


@router.get("/compute/progress")
async def compute_progress():
    """Return current computation progress."""
    from services.factor_computation_service import factor_computation_service
    from services.factor_effectiveness_service import factor_effectiveness_service

    if not _compute_running:
        return _compute_result or {"status": "idle"}

    val_prog = factor_computation_service.get_progress()
    eff_prog = factor_effectiveness_service.get_progress()

    if eff_prog.get("status") == "running":
        return {
            "status": "running",
            "phase": "effectiveness",
            "current_symbol": eff_prog.get("current_symbol", ""),
            "completed": eff_prog.get("completed", 0),
            "total": eff_prog.get("total", 0),
        }
    if val_prog.get("status") == "running":
        return {
            "status": "running",
            "phase": "values",
            "current_symbol": val_prog.get("current_symbol", ""),
            "completed": val_prog.get("completed", 0),
            "total": val_prog.get("total", 0),
        }
    return {"status": "running", "phase": "starting"}


# ── Custom Factor CRUD ──


class CustomFactorRequest(BaseModel):
    name: str
    expression: str
    description: str = ""
    category: str = "custom"
    source: str = "manual"


@router.get("/custom")
async def list_custom_factors(db: Session = Depends(get_db)):
    """List all custom factors."""
    rows = db.query(CustomFactor).order_by(CustomFactor.created_at.desc()).all()
    return {
        "items": [
            {
                "id": r.id, "name": r.name, "expression": r.expression,
                "description": r.description, "category": r.category,
                "source": r.source, "is_active": r.is_active,
                "created_at": str(r.created_at) if r.created_at else None,
            }
            for r in rows
        ]
    }


@router.post("/custom")
async def create_custom_factor(req: CustomFactorRequest, db: Session = Depends(get_db)):
    """Save a custom factor expression."""
    # Validate expression syntax
    ok, err = factor_expression_engine.validate(req.expression)
    if not ok:
        return {"status": "error", "error": err}

    # Check duplicate name
    existing = db.query(CustomFactor).filter(CustomFactor.name == req.name).first()
    if existing:
        return {"status": "error", "error": f"Factor name '{req.name}' already exists"}

    factor = CustomFactor(
        name=req.name,
        expression=req.expression,
        description=req.description,
        category=req.category,
        source=req.source,
    )
    db.add(factor)
    db.commit()
    db.refresh(factor)
    return {"status": "ok", "id": factor.id, "name": factor.name}


@router.delete("/custom/{factor_id}")
async def delete_custom_factor(factor_id: int, db: Session = Depends(get_db)):
    """Delete a custom factor."""
    factor = db.query(CustomFactor).filter(CustomFactor.id == factor_id).first()
    if not factor:
        return {"status": "error", "error": "Factor not found"}
    db.delete(factor)
    db.commit()
    return {"status": "ok"}


class EditCustomFactorRequest(BaseModel):
    name: Optional[str] = None
    expression: Optional[str] = None
    description: Optional[str] = None


@router.put("/custom/{factor_id}")
async def edit_custom_factor(factor_id: int, req: EditCustomFactorRequest, db: Session = Depends(get_db)):
    """Edit an existing custom factor."""
    factor = db.query(CustomFactor).filter(CustomFactor.id == factor_id).first()
    if not factor:
        return {"status": "error", "error": "Factor not found"}

    if req.expression:
        ok, err = factor_expression_engine.validate(req.expression)
        if not ok:
            return {"status": "error", "error": err}
        factor.expression = req.expression

    if req.name:
        dup = db.query(CustomFactor).filter(
            CustomFactor.name == req.name, CustomFactor.id != factor_id
        ).first()
        if dup:
            return {"status": "error", "error": f"Factor name '{req.name}' already exists"}
        factor.name = req.name

    if req.description is not None:
        factor.description = req.description

    db.commit()
    db.refresh(factor)
    return {"status": "ok", "id": factor.id, "name": factor.name}


# ── Expression Evaluation ──


class EvaluateRequest(BaseModel):
    expression: str
    symbol: str
    exchange: str = "hyperliquid"
    period: str = "1h"


@router.post("/evaluate")
async def evaluate_expression(req: EvaluateRequest):
    """Evaluate a factor expression on-demand (no save required)."""
    from services.market_data import get_kline_data

    market = "binance" if req.exchange == "binance" else "CRYPTO"
    klines = get_kline_data(req.symbol, market=market, period=req.period, count=500)
    if not klines or len(klines) < 50:
        return {"status": "error", "error": f"Insufficient K-line data for {req.symbol}"}

    results, err = factor_expression_engine.evaluate_ic(req.expression, klines)
    if results is None:
        return {"status": "error", "error": err}

    # Also get latest value
    series, _ = factor_expression_engine.execute(req.expression, klines)
    latest_value = None
    if series is not None and len(series) > 0:
        last = series.iloc[-1]
        latest_value = float(last) if not pd.isna(last) else None

    return {
        "status": "ok",
        "expression": req.expression,
        "symbol": req.symbol,
        "exchange": req.exchange,
        "latest_value": latest_value,
        "effectiveness": results,
    }


class ValidateExpressionRequest(BaseModel):
    expression: str


@router.post("/validate-expression")
async def validate_expression(req: ValidateExpressionRequest):
    """Quick syntax check for an expression."""
    ok, err = factor_expression_engine.validate(req.expression)
    return {"valid": ok, "error": err if not ok else None}


@router.get("/expression-functions")
async def list_expression_functions():
    """Return available functions for expression building."""
    return {"functions": factor_expression_engine.FUNCTION_DOCS}
