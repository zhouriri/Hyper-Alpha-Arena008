"""
Strategy Analytics API routes.
Provides multi-dimensional analysis of trading decisions and performance.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, case, and_, or_
from sqlalchemy.orm import Session

from database.connection import SessionLocal
from database.models import AIDecisionLog, Account, PromptTemplate
from database.snapshot_connection import SnapshotSessionLocal
from database.snapshot_models import HyperliquidTrade, HyperliquidAccountSnapshot
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============== Pydantic Models ==============

class MetricsResponse(BaseModel):
    total_pnl: float
    total_fee: float
    net_pnl: float
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    avg_win: Optional[float]
    avg_loss: Optional[float]
    profit_factor: Optional[float]


class DataCompleteness(BaseModel):
    total_decisions: int
    with_strategy: int
    with_signal: int
    with_pnl: int


class TriggerTypeBreakdown(BaseModel):
    count: int
    net_pnl: float


# ============== Helper Functions ==============

def get_fees_for_decisions(decisions: List[AIDecisionLog]) -> Dict[int, float]:
    """
    Batch query HyperliquidTrade to get total fees for each decision.
    Returns a dict mapping decision_id -> total_fee.
    """
    if not decisions:
        return {}

    # Collect all order IDs (main, tp, sl)
    order_ids = set()
    decision_orders: Dict[int, List[str]] = {}  # decision_id -> list of order_ids

    for d in decisions:
        orders = []
        if d.hyperliquid_order_id:
            order_ids.add(d.hyperliquid_order_id)
            orders.append(d.hyperliquid_order_id)
        if d.tp_order_id:
            order_ids.add(d.tp_order_id)
            orders.append(d.tp_order_id)
        if d.sl_order_id:
            order_ids.add(d.sl_order_id)
            orders.append(d.sl_order_id)
        decision_orders[d.id] = orders

    if not order_ids:
        return {d.id: 0.0 for d in decisions}

    # Batch query fees from HyperliquidTrade
    fee_map: Dict[str, float] = {}
    try:
        snapshot_db = SnapshotSessionLocal()
        trades = snapshot_db.query(HyperliquidTrade).filter(
            HyperliquidTrade.order_id.in_(list(order_ids))
        ).all()
        for t in trades:
            if t.order_id:
                fee_map[str(t.order_id)] = float(t.fee or 0)
        snapshot_db.close()
    except Exception as e:
        logger.warning(f"Failed to fetch fees from HyperliquidTrade: {e}")

    # Calculate total fee for each decision
    result: Dict[int, float] = {}
    for d in decisions:
        total_fee = 0.0
        for oid in decision_orders.get(d.id, []):
            total_fee += fee_map.get(oid, 0.0)
        result[d.id] = total_fee

    return result


def calculate_metrics(records: List[Dict]) -> Dict[str, Any]:
    """Calculate standard metrics from a list of decision records."""
    if not records:
        return {
            "total_pnl": 0.0,
            "total_fee": 0.0,
            "net_pnl": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0.0,
            "avg_win": None,
            "avg_loss": None,
            "profit_factor": None,
        }

    total_pnl = sum(r.get("pnl", 0) or 0 for r in records)
    total_fee = sum(r.get("fee", 0) or 0 for r in records)
    net_pnl = total_pnl - total_fee

    wins = [r for r in records if (r.get("pnl") or 0) > 0]
    losses = [r for r in records if (r.get("pnl") or 0) < 0]

    win_count = len(wins)
    loss_count = len(losses)
    trade_count = len(records)
    win_rate = win_count / trade_count if trade_count > 0 else 0.0

    total_win = sum(r.get("pnl", 0) or 0 for r in wins)
    total_loss = abs(sum(r.get("pnl", 0) or 0 for r in losses))

    avg_win = total_win / win_count if win_count > 0 else None
    avg_loss = -total_loss / loss_count if loss_count > 0 else None
    profit_factor = total_win / total_loss if total_loss > 0 else None

    return {
        "total_pnl": round(total_pnl, 2),
        "total_fee": round(total_fee, 2),
        "net_pnl": round(net_pnl, 2),
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2) if avg_win else None,
        "avg_loss": round(avg_loss, 2) if avg_loss else None,
        "profit_factor": round(profit_factor, 2) if profit_factor else None,
    }


def get_trigger_type(decision: AIDecisionLog) -> str:
    """Determine trigger type for a decision."""
    if decision.signal_trigger_id is not None:
        return "signal"
    elif decision.executed == "true" and decision.operation in ("buy", "sell", "close"):
        return "scheduled"
    return "unknown"


def build_base_query(
    db: Session,
    start_date: Optional[date],
    end_date: Optional[date],
    environment: Optional[str],
    account_id: Optional[int],
):
    """Build base query with common filters.

    Only includes decisions with non-zero realized_pnl (i.e., actually closed positions).
    This ensures statistics only count trades that have settled PnL,
    excluding opening trades (pnl=0) and unsync trades (pnl=NULL).
    """
    query = db.query(AIDecisionLog).filter(
        AIDecisionLog.operation.in_(["buy", "sell", "close"]),
        AIDecisionLog.executed == "true",
        AIDecisionLog.realized_pnl.isnot(None),  # Exclude unsync trades
        AIDecisionLog.realized_pnl != 0,  # Exclude opening trades (no settled PnL)
    )

    if start_date:
        query = query.filter(AIDecisionLog.decision_time >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        query = query.filter(AIDecisionLog.decision_time <= datetime.combine(end_date, datetime.max.time()))
    if environment and environment != "all":
        query = query.filter(AIDecisionLog.hyperliquid_environment == environment)
    if account_id:
        query = query.filter(AIDecisionLog.account_id == account_id)

    return query


# ============== API Endpoints ==============

@router.get("/summary")
def get_analytics_summary(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    environment: Optional[str] = Query("all"),
    account_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Get overall analytics summary."""
    query = build_base_query(db, start_date, end_date, environment, account_id)
    decisions = query.all()

    # Get fees for all decisions
    fee_map = get_fees_for_decisions(decisions)

    # Convert to records for metrics calculation
    records = []
    signal_records = []
    scheduled_records = []
    unknown_records = []

    with_strategy = 0
    with_signal = 0
    with_pnl = 0

    for d in decisions:
        pnl = float(d.realized_pnl) if d.realized_pnl else 0
        fee = fee_map.get(d.id, 0.0)
        record = {"pnl": pnl, "fee": fee}
        records.append(record)

        trigger_type = get_trigger_type(d)
        if trigger_type == "signal":
            signal_records.append(record)
        elif trigger_type == "scheduled":
            scheduled_records.append(record)
        else:
            unknown_records.append(record)

        if d.prompt_template_id:
            with_strategy += 1
        if d.signal_trigger_id:
            with_signal += 1
        if d.realized_pnl:
            with_pnl += 1

    overview = calculate_metrics(records)

    return {
        "period": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "overview": overview,
        "data_completeness": {
            "total_decisions": len(decisions),
            "with_strategy": with_strategy,
            "with_signal": with_signal,
            "with_pnl": with_pnl,
        },
        "by_trigger_type": {
            "signal": {
                "count": len(signal_records),
                "net_pnl": round(sum(r["pnl"] - r["fee"] for r in signal_records), 2),
            },
            "scheduled": {
                "count": len(scheduled_records),
                "net_pnl": round(sum(r["pnl"] - r["fee"] for r in scheduled_records), 2),
            },
            "unknown": {
                "count": len(unknown_records),
                "net_pnl": round(sum(r["pnl"] - r["fee"] for r in unknown_records), 2),
            },
        },
    }


@router.get("/by-strategy")
def get_analytics_by_strategy(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    environment: Optional[str] = Query("all"),
    account_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Get analytics grouped by strategy (prompt template)."""
    query = build_base_query(db, start_date, end_date, environment, account_id)
    decisions = query.all()

    # Get fees for all decisions
    fee_map = get_fees_for_decisions(decisions)

    # Group by strategy
    by_strategy: Dict[Optional[int], List[Dict]] = {}
    strategy_names: Dict[int, str] = {}

    for d in decisions:
        strategy_id = d.prompt_template_id
        pnl = float(d.realized_pnl) if d.realized_pnl else 0
        fee = fee_map.get(d.id, 0.0)
        record = {
            "pnl": pnl,
            "fee": fee,
            "trigger_type": get_trigger_type(d),
        }

        if strategy_id not in by_strategy:
            by_strategy[strategy_id] = []
        by_strategy[strategy_id].append(record)

    # Get strategy names
    strategy_ids = [sid for sid in by_strategy.keys() if sid is not None]
    if strategy_ids:
        templates = db.query(PromptTemplate).filter(PromptTemplate.id.in_(strategy_ids)).all()
        strategy_names = {t.id: t.name for t in templates}

    # Build response
    items = []
    for strategy_id, records in by_strategy.items():
        if strategy_id is None:
            continue

        signal_records = [r for r in records if r["trigger_type"] == "signal"]
        scheduled_records = [r for r in records if r["trigger_type"] == "scheduled"]

        items.append({
            "strategy_id": strategy_id,
            "strategy_name": strategy_names.get(strategy_id, f"Strategy {strategy_id}"),
            "metrics": calculate_metrics(records),
            "by_trigger_type": {
                "signal": {"count": len(signal_records), "net_pnl": round(sum(r["pnl"] - r["fee"] for r in signal_records), 2)},
                "scheduled": {"count": len(scheduled_records), "net_pnl": round(sum(r["pnl"] - r["fee"] for r in scheduled_records), 2)},
            },
        })

    # Sort by net_pnl descending
    items.sort(key=lambda x: x["metrics"]["net_pnl"], reverse=True)

    # Unattributed (no strategy)
    unattributed_records = by_strategy.get(None, [])

    return {
        "items": items,
        "unattributed": {
            "count": len(unattributed_records),
            "metrics": calculate_metrics(unattributed_records) if unattributed_records else None,
        },
    }


@router.get("/by-account")
def get_analytics_by_account(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    environment: Optional[str] = Query("all"),
    db: Session = Depends(get_db),
):
    """Get analytics grouped by account."""
    query = build_base_query(db, start_date, end_date, environment, None)
    decisions = query.all()

    # Get fees for all decisions
    fee_map = get_fees_for_decisions(decisions)

    # Group by account
    by_account: Dict[Optional[int], List[Dict]] = {}

    for d in decisions:
        account_id = d.account_id
        pnl = float(d.realized_pnl) if d.realized_pnl else 0
        fee = fee_map.get(d.id, 0.0)
        record = {"pnl": pnl, "fee": fee, "trigger_type": get_trigger_type(d)}

        if account_id not in by_account:
            by_account[account_id] = []
        by_account[account_id].append(record)

    # Get account info (name, current model)
    account_ids = [aid for aid in by_account.keys() if aid is not None]
    account_info: Dict[int, Dict] = {}
    if account_ids:
        accounts = db.query(Account).filter(Account.id.in_(account_ids)).all()
        account_info = {
            a.id: {"name": a.name, "model": a.model, "environment": a.hyperliquid_environment}
            for a in accounts
        }

    # Build response
    items = []
    for account_id, records in by_account.items():
        if account_id is None:
            continue

        info = account_info.get(account_id, {})
        signal_records = [r for r in records if r["trigger_type"] == "signal"]
        scheduled_records = [r for r in records if r["trigger_type"] == "scheduled"]

        items.append({
            "account_id": account_id,
            "account_name": info.get("name", f"Account {account_id}"),
            "model": info.get("model"),
            "environment": info.get("environment"),
            "metrics": calculate_metrics(records),
            "by_trigger_type": {
                "signal": {"count": len(signal_records), "net_pnl": round(sum(r["pnl"] - r["fee"] for r in signal_records), 2)},
                "scheduled": {"count": len(scheduled_records), "net_pnl": round(sum(r["pnl"] - r["fee"] for r in scheduled_records), 2)},
            },
        })

    # Sort by net_pnl descending
    items.sort(key=lambda x: x["metrics"]["net_pnl"], reverse=True)

    # Unattributed (no account)
    unattributed_records = by_account.get(None, [])

    return {
        "items": items,
        "unattributed": {
            "count": len(unattributed_records),
            "metrics": calculate_metrics(unattributed_records) if unattributed_records else None,
        },
    }


@router.get("/by-symbol")
def get_analytics_by_symbol(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    environment: Optional[str] = Query("all"),
    account_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Get analytics grouped by trading symbol."""
    query = build_base_query(db, start_date, end_date, environment, account_id)
    decisions = query.all()

    # Get fees for all decisions
    fee_map = get_fees_for_decisions(decisions)

    # Group by symbol
    by_symbol: Dict[Optional[str], List[Dict]] = {}

    for d in decisions:
        symbol = d.symbol
        pnl = float(d.realized_pnl) if d.realized_pnl else 0
        fee = fee_map.get(d.id, 0.0)
        record = {"pnl": pnl, "fee": fee, "trigger_type": get_trigger_type(d)}

        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(record)

    # Build response
    items = []
    for symbol, records in by_symbol.items():
        if symbol is None:
            continue

        signal_records = [r for r in records if r["trigger_type"] == "signal"]
        scheduled_records = [r for r in records if r["trigger_type"] == "scheduled"]

        items.append({
            "symbol": symbol,
            "metrics": calculate_metrics(records),
            "by_trigger_type": {
                "signal": {"count": len(signal_records), "net_pnl": round(sum(r["pnl"] - r["fee"] for r in signal_records), 2)},
                "scheduled": {"count": len(scheduled_records), "net_pnl": round(sum(r["pnl"] - r["fee"] for r in scheduled_records), 2)},
            },
        })

    # Sort by net_pnl descending
    items.sort(key=lambda x: x["metrics"]["net_pnl"], reverse=True)

    # Unattributed (no symbol)
    unattributed_records = by_symbol.get(None, [])

    return {
        "items": items,
        "unattributed": {
            "count": len(unattributed_records),
            "metrics": calculate_metrics(unattributed_records) if unattributed_records else None,
        },
    }


@router.get("/by-operation")
def get_analytics_by_operation(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    environment: Optional[str] = Query("all"),
    account_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Get analytics grouped by operation type (buy/sell/close)."""
    query = build_base_query(db, start_date, end_date, environment, account_id)
    decisions = query.all()

    # Get fees for all decisions
    fee_map = get_fees_for_decisions(decisions)

    # Group by operation
    by_operation: Dict[str, List[Dict]] = {}

    for d in decisions:
        operation = d.operation or "unknown"
        pnl = float(d.realized_pnl) if d.realized_pnl else 0
        fee = fee_map.get(d.id, 0.0)
        record = {"pnl": pnl, "fee": fee, "trigger_type": get_trigger_type(d)}

        if operation not in by_operation:
            by_operation[operation] = []
        by_operation[operation].append(record)

    # Build response
    items = []
    for operation, records in by_operation.items():
        signal_records = [r for r in records if r["trigger_type"] == "signal"]
        scheduled_records = [r for r in records if r["trigger_type"] == "scheduled"]

        items.append({
            "operation": operation,
            "metrics": calculate_metrics(records),
            "by_trigger_type": {
                "signal": {"count": len(signal_records), "net_pnl": round(sum(r["pnl"] - r["fee"] for r in signal_records), 2)},
                "scheduled": {"count": len(scheduled_records), "net_pnl": round(sum(r["pnl"] - r["fee"] for r in scheduled_records), 2)},
            },
        })

    # Sort by trade_count descending
    items.sort(key=lambda x: x["metrics"]["trade_count"], reverse=True)

    return {"items": items}


@router.get("/by-trigger-type")
def get_analytics_by_trigger_type(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    environment: Optional[str] = Query("all"),
    account_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Get analytics grouped by trigger type (signal/scheduled/unknown)."""
    query = build_base_query(db, start_date, end_date, environment, account_id)
    decisions = query.all()

    # Get fees for all decisions
    fee_map = get_fees_for_decisions(decisions)

    # Group by trigger type
    by_trigger: Dict[str, List[Dict]] = {"signal": [], "scheduled": [], "unknown": []}

    for d in decisions:
        trigger_type = get_trigger_type(d)
        pnl = float(d.realized_pnl) if d.realized_pnl else 0
        fee = fee_map.get(d.id, 0.0)
        record = {"pnl": pnl, "fee": fee}
        by_trigger[trigger_type].append(record)

    # Build response
    items = []
    for trigger_type in ["signal", "scheduled", "unknown"]:
        records = by_trigger[trigger_type]
        if records:
            items.append({
                "trigger_type": trigger_type,
                "metrics": calculate_metrics(records),
            })

    # Sort by trade_count descending
    items.sort(key=lambda x: x["metrics"]["trade_count"], reverse=True)

    return {"items": items}


# ============== AI Attribution Analysis Routes ==============

from fastapi.responses import StreamingResponse
from pydantic import BaseModel as PydanticBaseModel
from services.ai_attribution_service import (
    generate_attribution_analysis_stream,
    get_attribution_conversations,
    get_attribution_messages
)


class AiAttributionChatRequest(PydanticBaseModel):
    accountId: int
    userMessage: str
    conversationId: Optional[int] = None


@router.post("/ai-attribution/chat-stream")
async def ai_attribution_chat_stream(
    request: AiAttributionChatRequest,
    db: Session = Depends(get_db)
):
    """SSE streaming endpoint for AI attribution analysis chat."""
    return StreamingResponse(
        generate_attribution_analysis_stream(
            db=db,
            account_id=request.accountId,
            user_message=request.userMessage,
            conversation_id=request.conversationId
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/ai-attribution/conversations")
async def list_attribution_conversations(db: Session = Depends(get_db)):
    """Get list of AI attribution analysis conversations."""
    conversations = get_attribution_conversations(db)
    return {"conversations": conversations}


@router.get("/ai-attribution/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get messages for a specific conversation."""
    messages = get_attribution_messages(db, conversation_id)
    return {"messages": messages}


# ============== Trade Details API ==============

def get_exit_type(decision: AIDecisionLog) -> str:
    """Determine exit type based on operation and order IDs."""
    if decision.operation == 'close':
        return 'CLOSE'
    elif decision.tp_order_id or decision.sl_order_id:
        if decision.realized_pnl and float(decision.realized_pnl) > 0:
            return 'TP'
        else:
            return 'SL'
    return 'CLOSE'


def get_entry_type(decision: AIDecisionLog, db: Session) -> str:
    """Determine entry type (BUY/SELL/-).

    For close operations, look up the corresponding opening trade.
    """
    if decision.operation in ('buy', 'sell'):
        return decision.operation.upper()

    # For close operation, find the corresponding opening trade
    if decision.operation == 'close' and decision.symbol and decision.wallet_address:
        # Find the most recent buy/sell for the same symbol before this close
        opening_trade = db.query(AIDecisionLog).filter(
            AIDecisionLog.symbol == decision.symbol,
            AIDecisionLog.wallet_address == decision.wallet_address,
            AIDecisionLog.operation.in_(['buy', 'sell']),
            AIDecisionLog.decision_time < decision.decision_time
        ).order_by(AIDecisionLog.decision_time.desc()).first()

        if opening_trade:
            return opening_trade.operation.upper()

    return '-'


def get_entry_decision(decision: AIDecisionLog, db: Session) -> Optional[AIDecisionLog]:
    """Get the entry decision for a trade.

    For buy/sell operations, return the decision itself.
    For close operations, find the corresponding opening trade.
    """
    if decision.operation in ('buy', 'sell'):
        return decision

    # For close operation, find the corresponding opening trade
    if decision.operation == 'close' and decision.symbol and decision.wallet_address:
        return db.query(AIDecisionLog).filter(
            AIDecisionLog.symbol == decision.symbol,
            AIDecisionLog.wallet_address == decision.wallet_address,
            AIDecisionLog.operation.in_(['buy', 'sell']),
            AIDecisionLog.decision_time < decision.decision_time
        ).order_by(AIDecisionLog.decision_time.desc()).first()

    return None


def calculate_trade_tags(
    decisions: List[AIDecisionLog],
    account_equity: float,
    equity_threshold: float = 0.05
) -> Dict[int, List[str]]:
    """Calculate rule-based tags for each trade."""
    tags: Dict[int, List[str]] = {}
    loss_threshold = account_equity * equity_threshold if account_equity > 0 else 50.0

    # Sort by time for consecutive loss detection
    sorted_decisions = sorted(decisions, key=lambda d: d.decision_time or datetime.min)

    consecutive_losses = 0
    consecutive_loss_ids = []

    for d in sorted_decisions:
        d_tags = []
        pnl = float(d.realized_pnl) if d.realized_pnl else 0

        # Large loss: |pnl| > threshold
        if pnl < 0 and abs(pnl) > loss_threshold:
            d_tags.append('large_loss')

        # SL triggered: has sl_order_id and pnl < 0
        if d.sl_order_id and pnl < 0:
            d_tags.append('sl_triggered')

        # Consecutive losses tracking
        if pnl < 0:
            consecutive_losses += 1
            consecutive_loss_ids.append(d.id)
        else:
            # Mark previous consecutive losses if >= 3
            if consecutive_losses >= 3:
                for loss_id in consecutive_loss_ids:
                    if loss_id not in tags:
                        tags[loss_id] = []
                    if 'consecutive_loss' not in tags[loss_id]:
                        tags[loss_id].append('consecutive_loss')
            consecutive_losses = 0
            consecutive_loss_ids = []

        tags[d.id] = d_tags

    # Handle trailing consecutive losses
    if consecutive_losses >= 3:
        for loss_id in consecutive_loss_ids:
            if 'consecutive_loss' not in tags.get(loss_id, []):
                tags.setdefault(loss_id, []).append('consecutive_loss')

    return tags


@router.get("/trades")
def get_trade_details(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    environment: Optional[str] = Query("all"),
    account_id: Optional[int] = Query(None),
    tag_filter: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    db: Session = Depends(get_db),
):
    """Get trade details with rule-based tags for micro-analysis."""
    # Build query for closed trades
    query = db.query(AIDecisionLog).filter(
        AIDecisionLog.realized_pnl.isnot(None),
        AIDecisionLog.realized_pnl != 0,
        AIDecisionLog.hyperliquid_order_id.isnot(None)
    )

    if start_date:
        query = query.filter(AIDecisionLog.decision_time >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        query = query.filter(AIDecisionLog.decision_time <= datetime.combine(end_date, datetime.max.time()))
    if environment and environment != "all":
        query = query.filter(AIDecisionLog.hyperliquid_environment == environment)
    if account_id:
        query = query.filter(AIDecisionLog.account_id == account_id)

    # Get total count before pagination
    total_count = query.count()

    # Get all decisions for tag calculation (need full list for consecutive loss)
    all_decisions = query.order_by(AIDecisionLog.decision_time.desc()).all()

    # Get account equity for threshold calculation
    account_equity = 0.0
    if all_decisions:
        first_account_id = all_decisions[0].account_id
        env = environment if environment != "all" else "mainnet"
        try:
            snapshot_db = SnapshotSessionLocal()
            snapshot = snapshot_db.query(HyperliquidAccountSnapshot).filter(
                HyperliquidAccountSnapshot.account_id == first_account_id,
                HyperliquidAccountSnapshot.environment == env
            ).order_by(HyperliquidAccountSnapshot.created_at.desc()).first()
            if snapshot and snapshot.total_equity:
                account_equity = float(snapshot.total_equity)
            snapshot_db.close()
        except Exception as e:
            logger.warning(f"Failed to get account equity: {e}")

    # Calculate tags
    trade_tags = calculate_trade_tags(all_decisions, account_equity)

    # Get fees
    fee_map = get_fees_for_decisions(all_decisions)

    # Apply tag filter if specified
    if tag_filter:
        filtered_ids = [d.id for d in all_decisions if tag_filter in trade_tags.get(d.id, [])]
        all_decisions = [d for d in all_decisions if d.id in filtered_ids]
        total_count = len(all_decisions)

    # Apply pagination
    paginated = all_decisions[offset:offset + limit]

    # Build response
    trades = []
    for d in paginated:
        pnl = float(d.realized_pnl) if d.realized_pnl else 0
        fee = fee_map.get(d.id, 0.0)

        # Get entry decision and time
        entry_decision = get_entry_decision(d, db)
        entry_time = None
        if entry_decision and entry_decision.decision_time:
            entry_time = entry_decision.decision_time.isoformat()

        # Determine exit time: use pnl_updated_at for TP/SL triggers, otherwise decision_time
        exit_type = get_exit_type(d)
        if exit_type in ('TP', 'SL') and d.pnl_updated_at:
            exit_time = d.pnl_updated_at.isoformat()
        else:
            exit_time = d.decision_time.isoformat() if d.decision_time else None

        trades.append({
            "id": d.id,
            "symbol": d.symbol,
            "decision_time": d.decision_time.isoformat() if d.decision_time else None,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_type": entry_decision.operation.upper() if entry_decision else '-',
            "exit_type": exit_type,
            "gross_pnl": round(pnl, 2),
            "fees": round(fee, 2),
            "net_pnl": round(pnl - fee, 2),
            "tags": trade_tags.get(d.id, []),
            "hyperliquid_order_id": d.hyperliquid_order_id,
            "tp_order_id": d.tp_order_id,
            "sl_order_id": d.sl_order_id,
        })

    return {
        "trades": trades,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "account_equity": round(account_equity, 2),
        "loss_threshold": round(account_equity * 0.05, 2) if account_equity > 0 else 50.0,
    }


@router.get("/trades/{trade_id}/replay")
def get_trade_replay(
    trade_id: int,
    db: Session = Depends(get_db),
):
    """Get trade replay data including decision chain and trade details."""
    # Get the main trade record
    trade = db.query(AIDecisionLog).filter(AIDecisionLog.id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    # Determine if this is an entry or exit record
    is_entry = trade.operation in ('buy', 'sell')
    is_exit = trade.operation == 'close' or (trade.realized_pnl is not None and trade.realized_pnl != 0)

    entry_decision = None
    exit_decision = None
    entry_time = None
    exit_time = None

    if is_entry and trade.realized_pnl:
        # This is an entry with PnL (TP/SL triggered) - entry and exit are same record
        entry_decision = trade
        exit_decision = trade
        entry_time = trade.decision_time
        exit_time = trade.pnl_updated_at or trade.decision_time
    elif is_entry:
        # Entry without PnL - need to find exit
        entry_decision = trade
        entry_time = trade.decision_time
        # Find corresponding close
        exit_decision = db.query(AIDecisionLog).filter(
            AIDecisionLog.symbol == trade.symbol,
            AIDecisionLog.wallet_address == trade.wallet_address,
            AIDecisionLog.operation == 'close',
            AIDecisionLog.decision_time > trade.decision_time
        ).order_by(AIDecisionLog.decision_time.asc()).first()
        if exit_decision:
            exit_time = exit_decision.decision_time
    else:
        # This is an exit (close) - find corresponding entry
        exit_decision = trade
        exit_time = trade.decision_time
        entry_decision = db.query(AIDecisionLog).filter(
            AIDecisionLog.symbol == trade.symbol,
            AIDecisionLog.wallet_address == trade.wallet_address,
            AIDecisionLog.operation.in_(['buy', 'sell']),
            AIDecisionLog.decision_time < trade.decision_time
        ).order_by(AIDecisionLog.decision_time.desc()).first()
        if entry_decision:
            entry_time = entry_decision.decision_time

    # Build decision chain (all decisions between entry and exit)
    decisions_chain = []
    if entry_time and exit_time:
        chain_query = db.query(AIDecisionLog).filter(
            AIDecisionLog.symbol == trade.symbol,
            AIDecisionLog.wallet_address == trade.wallet_address,
            AIDecisionLog.decision_time >= entry_time,
            AIDecisionLog.decision_time <= exit_time
        ).order_by(AIDecisionLog.decision_time.asc()).all()

        for d in chain_query:
            decisions_chain.append({
                "id": d.id,
                "operation": d.operation,
                "decision_time": d.decision_time.isoformat() if d.decision_time else None,
                "reason": d.reason,
                "target_portion": float(d.target_portion) if d.target_portion else 0,
                "realized_pnl": float(d.realized_pnl) if d.realized_pnl else None,
            })

    # Calculate summary
    entry_price = None
    exit_price = None
    hold_duration = None

    if entry_time and exit_time:
        hold_duration = str(exit_time - entry_time)

    # Get PnL (fees are already deducted in realized_pnl from Hyperliquid)
    pnl = float(trade.realized_pnl) if trade.realized_pnl else 0

    return {
        "trade": {
            "id": trade.id,
            "symbol": trade.symbol,
            "operation": trade.operation,
            "decision_time": trade.decision_time.isoformat() if trade.decision_time else None,
            "wallet_address": trade.wallet_address,
            "hyperliquid_environment": trade.hyperliquid_environment,
            "account_id": trade.account_id,
        },
        "entry_decision": {
            "id": entry_decision.id,
            "operation": entry_decision.operation,
            "decision_time": entry_decision.decision_time.isoformat() if entry_decision.decision_time else None,
            "reason": entry_decision.reason,
        } if entry_decision else None,
        "exit_decision": {
            "id": exit_decision.id,
            "operation": exit_decision.operation,
            "decision_time": exit_decision.decision_time.isoformat() if exit_decision.decision_time else None,
            "reason": exit_decision.reason,
            "exit_type": get_exit_type(exit_decision),
        } if exit_decision else None,
        "decisions_chain": decisions_chain,
        "summary": {
            "entry_time": entry_time.isoformat() if entry_time else None,
            "exit_time": exit_time.isoformat() if exit_time else None,
            "hold_duration": hold_duration,
            "pnl": round(pnl, 2),
        },
        "kline_params": {
            "symbol": trade.symbol,
            "start_time": (entry_time - timedelta(hours=1)).isoformat() if entry_time else None,
            "end_time": (exit_time + timedelta(hours=1)).isoformat() if exit_time else None,
        } if entry_time and exit_time else None,
    }


# ============== Trade Replay K-line API ==============

def _parse_decision_prices(decision: AIDecisionLog) -> dict:
    """Extract price info from decision_snapshot JSON"""
    prices = {"entry_price": None, "tp_price": None, "sl_price": None, "exit_price": None}
    if not decision.decision_snapshot:
        return prices
    try:
        import json
        snapshot = json.loads(decision.decision_snapshot) if isinstance(decision.decision_snapshot, str) else decision.decision_snapshot
        operation = snapshot.get("operation", decision.operation)

        # For BUY: max_price is entry limit (buy no higher than this)
        # For SELL: min_price is entry limit (sell no lower than this)
        if operation == "buy":
            prices["entry_price"] = snapshot.get("max_price") or snapshot.get("entry_price")
        elif operation == "sell":
            prices["entry_price"] = snapshot.get("min_price") or snapshot.get("entry_price")
        else:
            prices["entry_price"] = snapshot.get("max_price") or snapshot.get("min_price") or snapshot.get("entry_price")

        prices["tp_price"] = snapshot.get("take_profit_price") or snapshot.get("tp_price")
        prices["sl_price"] = snapshot.get("stop_loss_price") or snapshot.get("sl_price")
        # For exit/close: min_price is the exit price limit
        prices["exit_price"] = snapshot.get("min_price") if operation == "close" else None
    except:
        pass
    return prices

@router.get("/trades/{trade_id}/kline")
def get_trade_replay_kline(
    trade_id: int,
    period: str = Query("5m", description="K-line period: 5m, 15m, 1h, 4h"),
    db: Session = Depends(get_db)
):
    """
    Get K-line data for trade replay with entry/exit markers.
    Returns historical K-line data centered around the trade's entry and exit times.
    """
    from services.hyperliquid_market_data import get_historical_kline_data_from_hyperliquid

    # Validate period
    valid_periods = ['5m', '15m', '1h', '4h']
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail=f"Invalid period. Valid: {valid_periods}")

    # Get the main trade record (same logic as get_trade_replay)
    trade = db.query(AIDecisionLog).filter(AIDecisionLog.id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    # Determine entry and exit decisions (same logic as get_trade_replay)
    is_entry = trade.operation in ('buy', 'sell')

    entry_decision = None
    exit_decision = None
    entry_time = None
    exit_time = None

    if is_entry and trade.realized_pnl:
        entry_decision = trade
        exit_decision = trade
        entry_time = trade.decision_time
        exit_time = trade.pnl_updated_at or trade.decision_time
    elif is_entry:
        entry_decision = trade
        entry_time = trade.decision_time
        exit_decision = db.query(AIDecisionLog).filter(
            AIDecisionLog.symbol == trade.symbol,
            AIDecisionLog.wallet_address == trade.wallet_address,
            AIDecisionLog.operation == 'close',
            AIDecisionLog.decision_time > trade.decision_time
        ).order_by(AIDecisionLog.decision_time.asc()).first()
        if exit_decision:
            exit_time = exit_decision.decision_time
    else:
        exit_decision = trade
        exit_time = trade.decision_time
        entry_decision = db.query(AIDecisionLog).filter(
            AIDecisionLog.symbol == trade.symbol,
            AIDecisionLog.wallet_address == trade.wallet_address,
            AIDecisionLog.operation.in_(['buy', 'sell']),
            AIDecisionLog.decision_time < trade.decision_time
        ).order_by(AIDecisionLog.decision_time.desc()).first()
        if entry_decision:
            entry_time = entry_decision.decision_time

    if not entry_time:
        raise HTTPException(status_code=400, detail="Trade has no entry time")

    # Calculate buffer based on period
    period_buffer = {'5m': 30, '15m': 60, '1h': 120, '4h': 480}
    buffer_minutes = period_buffer.get(period, 30)

    # Calculate time range
    start_time = entry_time - timedelta(minutes=buffer_minutes)
    if exit_time:
        end_time = exit_time + timedelta(minutes=buffer_minutes)
    else:
        end_time = entry_time + timedelta(hours=4)

    # Convert to milliseconds
    since_ms = int(start_time.timestamp() * 1000)
    until_ms = int(end_time.timestamp() * 1000)

    # Fetch K-line data from Hyperliquid API first
    environment = trade.hyperliquid_environment or "mainnet"
    klines = get_historical_kline_data_from_hyperliquid(
        symbol=trade.symbol,
        period=period,
        since_ms=since_ms,
        until_ms=until_ms,
        environment=environment
    )

    # Fallback to local database if API returns empty
    if not klines:
        from database.models import CryptoKline

        # Clean symbol (remove -PERP suffix)
        symbol_clean = trade.symbol.upper()
        if symbol_clean.endswith('-PERP'):
            symbol_clean = symbol_clean[:-5]

        # Query local klines
        local_klines = db.query(CryptoKline).filter(
            CryptoKline.symbol == symbol_clean,
            CryptoKline.period == period,
            CryptoKline.timestamp >= int(since_ms / 1000),
            CryptoKline.timestamp <= int(until_ms / 1000),
            CryptoKline.environment == environment
        ).order_by(CryptoKline.timestamp).all()

        if local_klines:
            # Convert to API format
            klines = []
            for k in local_klines:
                open_p = float(k.open_price) if k.open_price else None
                close_p = float(k.close_price) if k.close_price else None
                chg = (close_p - open_p) if open_p and close_p else 0
                pct = (chg / open_p * 100) if open_p else 0
                klines.append({
                    'timestamp': k.timestamp,
                    'datetime': k.datetime_str,
                    'open': open_p,
                    'high': float(k.high_price) if k.high_price else None,
                    'low': float(k.low_price) if k.low_price else None,
                    'close': close_p,
                    'volume': float(k.volume) if k.volume else None,
                    'amount': float(k.amount) if k.amount else None,
                    'chg': chg,
                    'percent': pct,
                })
            logger.info(f"Using {len(klines)} local klines for {symbol_clean} {period}")

    if not klines:
        raise HTTPException(status_code=404, detail="Historical K-line data not available (exchange only keeps recent data)")

    # Build markers with full decision info including prices
    markers = []

    if entry_decision:
        entry_prices = _parse_decision_prices(entry_decision)
        markers.append({
            "type": "entry",
            "time": entry_decision.decision_time.isoformat() if entry_decision.decision_time else None,
            "timestamp": int(entry_decision.decision_time.timestamp()) if entry_decision.decision_time else None,
            "operation": entry_decision.operation,
            "reason": entry_decision.reason,
            "target_portion": float(entry_decision.target_portion) if entry_decision.target_portion else None,
            "symbol": trade.symbol,
            "entry_price": entry_prices["entry_price"],
            "tp_price": entry_prices["tp_price"],
            "sl_price": entry_prices["sl_price"],
        })

    if exit_decision and exit_decision.id != entry_decision.id:
        exit_prices = _parse_decision_prices(exit_decision)
        markers.append({
            "type": "exit",
            "time": exit_decision.decision_time.isoformat() if exit_decision.decision_time else None,
            "timestamp": int(exit_decision.decision_time.timestamp()) if exit_decision.decision_time else None,
            "operation": exit_decision.operation,
            "reason": exit_decision.reason,
            "exit_type": get_exit_type(exit_decision),
            "realized_pnl": float(trade.realized_pnl) if trade.realized_pnl else None,
            "symbol": trade.symbol,
            "exit_price": exit_prices["exit_price"] or exit_prices["entry_price"],  # min_price for close, max_price fallback
        })
    elif exit_decision and exit_decision.id == entry_decision.id and trade.realized_pnl:
        # TP/SL triggered - add exit marker at pnl_updated_at time
        entry_prices = _parse_decision_prices(entry_decision)
        markers.append({
            "type": "exit",
            "time": (trade.pnl_updated_at or trade.decision_time).isoformat(),
            "timestamp": int((trade.pnl_updated_at or trade.decision_time).timestamp()),
            "operation": "close",
            "reason": "TP/SL triggered",
            "exit_type": get_exit_type(trade),
            "realized_pnl": float(trade.realized_pnl),
            "symbol": trade.symbol,
            "exit_price": entry_prices["tp_price"] if float(trade.realized_pnl) > 0 else entry_prices["sl_price"],
        })

    # Add HOLD decision markers
    if entry_time and exit_time:
        hold_decisions = db.query(AIDecisionLog).filter(
            AIDecisionLog.symbol == trade.symbol,
            AIDecisionLog.wallet_address == trade.wallet_address,
            AIDecisionLog.operation == 'hold',
            AIDecisionLog.decision_time > entry_time,
            AIDecisionLog.decision_time < exit_time
        ).order_by(AIDecisionLog.decision_time.asc()).all()

        for hold in hold_decisions:
            markers.append({
                "type": "hold",
                "time": hold.decision_time.isoformat() if hold.decision_time else None,
                "timestamp": int(hold.decision_time.timestamp()) if hold.decision_time else None,
                "operation": "hold",
                "reason": hold.reason,
                "symbol": trade.symbol,
            })

    # Calculate default period based on hold duration
    default_period = "5m"
    if entry_time and exit_time:
        hold_minutes = (exit_time - entry_time).total_seconds() / 60
        if hold_minutes > 1440:
            default_period = "4h"
        elif hold_minutes > 240:
            default_period = "1h"
        elif hold_minutes > 60:
            default_period = "15m"

    return {
        "symbol": trade.symbol,
        "period": period,
        "default_period": default_period,
        "klines": klines,
        "markers": markers,
        "time_range": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        }
    }
