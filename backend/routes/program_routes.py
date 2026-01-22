"""
API routes for Program Trader with N:N binding architecture.

- Programs: Reusable strategy code templates
- Bindings: N:N relationship between AI Traders and Programs with trigger config
- Executions: Execution history logs
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import asyncio

from database.connection import get_db
from database.models import (
    TradingProgram, AccountProgramBinding, ProgramExecutionLog,
    User, Account, SignalPool, BacktestResult, BacktestTriggerLog
)
from program_trader import validate_strategy_code, BacktestEngine
from program_trader.models import Kline


router = APIRouter(prefix="/api/programs", tags=["Program Trader"])


# ============================================================================
# Pydantic Models
# ============================================================================

class ProgramCreate(BaseModel):
    name: str
    description: Optional[str] = None
    code: str
    params: Optional[Dict[str, Any]] = None
    icon: Optional[str] = None


class ProgramUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    code: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    icon: Optional[str] = None


class ProgramResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    code: str
    params: Optional[Dict[str, Any]]
    icon: Optional[str]
    binding_count: int = 0  # How many AI Traders use this program
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class WalletInfo(BaseModel):
    environment: str  # "testnet" | "mainnet"
    address: str      # wallet address


class BindingCreate(BaseModel):
    program_id: int
    signal_pool_ids: List[int] = []
    trigger_interval: int = 300
    scheduled_trigger_enabled: bool = False
    is_active: bool = True
    params_override: Optional[Dict[str, Any]] = None


class BindingUpdate(BaseModel):
    signal_pool_ids: Optional[List[int]] = None
    trigger_interval: Optional[int] = None
    scheduled_trigger_enabled: Optional[bool] = None
    is_active: Optional[bool] = None
    params_override: Optional[Dict[str, Any]] = None


class BindingResponse(BaseModel):
    id: int
    account_id: int
    account_name: str
    program_id: int
    program_name: str
    signal_pool_ids: List[int]
    signal_pool_names: List[str] = []  # Human-readable pool names
    trigger_interval: int
    scheduled_trigger_enabled: bool
    is_active: bool
    last_trigger_at: Optional[str]
    params_override: Optional[Dict[str, Any]]
    wallets: List[WalletInfo] = []  # AI Trader's wallets
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class ValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class BacktestRequest(BaseModel):
    symbol: str = "BTC"
    period: str = "5m"
    days: int = 7
    initial_balance: float = 10000.0


class BacktestResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    equity_curve: List[Dict[str, Any]] = []


class SignalPoolInfo(BaseModel):
    id: int
    pool_name: str
    symbols: List[str]
    enabled: bool


class AccountInfo(BaseModel):
    id: int
    name: str
    model: Optional[str]


# ============================================================================
# Test Run Models (for AI-friendly error reporting)
# ============================================================================

class TestRunRequest(BaseModel):
    """Request model for test-run API."""
    code: str  # Program code to test (can be unsaved)
    symbol: str = "BTC"  # Symbol for market data
    period: str = "1h"  # K-line period for indicators


class ErrorLocation(BaseModel):
    """Location of error in code."""
    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    function: Optional[str] = None
    code_context: Optional[str] = None  # The actual line of code


class DecisionResult(BaseModel):
    """Decision returned by strategy."""
    action: str  # BUY, SELL, HOLD, CLOSE
    symbol: Optional[str] = None
    size_usd: Optional[float] = None
    leverage: Optional[int] = None
    reason: Optional[str] = None


class MarketDataSummary(BaseModel):
    """Summary of market data used in test."""
    symbol: str
    current_price: Optional[float] = None
    price_change_1h: Optional[float] = None
    klines_count: int = 0
    indicators_loaded: List[str] = []


class TestRunResponse(BaseModel):
    """
    Comprehensive test-run response designed for AI analysis.

    When success=True: Contains decision details
    When success=False: Contains detailed error info for debugging
    """
    success: bool

    # Success fields
    decision: Optional[DecisionResult] = None
    execution_time_ms: float = 0.0
    market_data: Optional[MarketDataSummary] = None

    # Error fields (populated when success=False)
    error_type: Optional[str] = None  # SyntaxError, ImportError, RuntimeError, etc.
    error_message: Optional[str] = None  # Human-readable error message
    error_traceback: Optional[str] = None  # Full traceback for debugging
    error_location: Optional[ErrorLocation] = None  # Where the error occurred

    # AI-friendly suggestions
    suggestions: List[str] = []  # Possible fixes or hints
    available_apis: Optional[Dict[str, Any]] = None  # Available functions/methods


# ============================================================================
# Helper Functions
# ============================================================================

def get_default_user(db: Session) -> User:
    """Get or create default user."""
    user = db.query(User).filter(User.username == "default").first()
    if not user:
        user = User(username="default", email="default@local")
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def _program_to_response(program: TradingProgram, db: Session) -> ProgramResponse:
    """Convert TradingProgram to response model."""
    binding_count = db.query(AccountProgramBinding).filter(
        AccountProgramBinding.program_id == program.id
    ).count()

    return ProgramResponse(
        id=program.id,
        name=program.name,
        description=program.description,
        code=program.code,
        params=json.loads(program.params) if program.params else None,
        icon=program.icon,
        binding_count=binding_count,
        created_at=program.created_at.isoformat() if program.created_at else "",
        updated_at=program.updated_at.isoformat() if program.updated_at else "",
    )


def _binding_to_response(binding: AccountProgramBinding, db: Session) -> BindingResponse:
    """Convert AccountProgramBinding to response model."""
    pool_ids = []
    if binding.signal_pool_ids:
        try:
            pool_ids = json.loads(binding.signal_pool_ids)
        except:
            pass

    # Query signal pool names (include disabled pools for display)
    pool_names = []
    if pool_ids:
        pools = db.query(SignalPool).filter(SignalPool.id.in_(pool_ids)).all()
        pool_map = {p.id: p.pool_name for p in pools}
        pool_names = [pool_map.get(pid, f"Pool #{pid}") for pid in pool_ids]

    params_override = None
    if binding.params_override:
        try:
            params_override = json.loads(binding.params_override)
        except:
            pass

    # Query wallets for this AI Trader
    from database.models import HyperliquidWallet
    wallets = []
    wallet_rows = db.query(HyperliquidWallet).filter(
        HyperliquidWallet.account_id == binding.account_id,
        HyperliquidWallet.is_active == "true"
    ).all()
    for w in wallet_rows:
        if w.wallet_address:
            wallets.append(WalletInfo(environment=w.environment, address=w.wallet_address))

    return BindingResponse(
        id=binding.id,
        account_id=binding.account_id,
        account_name=binding.account.name if binding.account else "Unknown",
        program_id=binding.program_id,
        program_name=binding.program.name if binding.program else "Unknown",
        signal_pool_ids=pool_ids,
        signal_pool_names=pool_names,
        trigger_interval=binding.trigger_interval,
        scheduled_trigger_enabled=binding.scheduled_trigger_enabled,
        is_active=binding.is_active,
        last_trigger_at=binding.last_trigger_at.isoformat() if binding.last_trigger_at else None,
        params_override=params_override,
        wallets=wallets,
        created_at=binding.created_at.isoformat() if binding.created_at else "",
        updated_at=binding.updated_at.isoformat() if binding.updated_at else "",
    )


# ============================================================================
# Test Run API (must be before /{program_id} routes)
# ============================================================================

def _parse_error_location(traceback_str: str, code: str) -> ErrorLocation:
    """Extract error location from traceback."""
    import re

    location = ErrorLocation()

    # Look for line number in <string>
    match = re.search(r'File "<string>", line (\d+)', traceback_str)
    if match:
        location.file = "<string>"
        location.line = int(match.group(1))

        # Extract the code context
        lines = code.split('\n')
        if 0 < location.line <= len(lines):
            location.code_context = lines[location.line - 1].strip()

    # Look for function name
    match = re.search(r'in (\w+)\n', traceback_str)
    if match:
        location.function = match.group(1)

    return location


def _generate_suggestions(error_type: str, error_msg: str, traceback_str: str) -> List[str]:
    """Generate AI-friendly suggestions based on error type."""
    suggestions = []

    if error_type == "ImportError":
        suggestions.append("Check if the module/function name is spelled correctly")
        if "calculate_indicator" in error_msg:
            suggestions.append("Available indicator functions: get_indicator(symbol, indicator, period)")
        if "services" in error_msg:
            suggestions.append("Use MarketData methods instead of direct service imports")

    elif error_type == "SyntaxError":
        suggestions.append("Check for missing colons, parentheses, or indentation errors")
        suggestions.append("Ensure proper Python 3 syntax")

    elif error_type == "NameError":
        suggestions.append("Check if the variable/function is defined before use")
        suggestions.append("Available in sandbox: MarketData, Decision, ActionType, math functions")

    elif error_type == "AttributeError":
        suggestions.append("Check if the method/attribute exists on the object")
        suggestions.append("MarketData methods: get_price(), get_indicator(), get_klines(), get_flow()")

    elif error_type == "TypeError":
        suggestions.append("Check function arguments - wrong number or type of arguments")

    elif error_type == "KeyError":
        suggestions.append("Check if the dictionary key exists before accessing")
        suggestions.append("Use .get(key, default) for safe dictionary access")

    elif error_type == "ValidationError":
        suggestions.append("Ensure your class has a should_trade(self, data: MarketData) method")
        suggestions.append("should_trade must return a Decision object")

    elif error_type == "TimeoutError":
        suggestions.append("Strategy execution took too long (>5 seconds)")
        suggestions.append("Avoid infinite loops or expensive computations")

    return suggestions


def _get_available_apis() -> Dict[str, Any]:
    """Return documentation of available APIs for AI reference."""
    return {
        "MarketData_properties": {
            "data.trigger_symbol": "Symbol that triggered this evaluation",
            "data.trigger_type": "Trigger type: 'signal' or 'scheduled'",
            "data.available_balance": "Available balance in USD",
            "data.total_equity": "Total account equity",
            "data.positions": "Dict[str, Position] of current open positions",
        },
        "Position_fields": {
            "symbol": "Trading symbol",
            "side": "'long' or 'short'",
            "size": "Position size",
            "entry_price": "Entry price",
            "unrealized_pnl": "Unrealized PnL",
            "leverage": "Leverage used",
            "liquidation_price": "Liquidation price",
        },
        "MarketData_methods": {
            "get_market_data(symbol)": "Returns {symbol, price, oracle_price, change24h, volume24h, percentage24h, open_interest, funding_rate}",
            "get_indicator(symbol, indicator, period)": "Indicators: RSI14, RSI7, MA5, MA10, MA20, EMA20, EMA50, EMA100, MACD, BOLL, ATR14, VWAP, STOCH, OBV",
            "get_klines(symbol, period, count=50)": "Returns list of Kline(timestamp, open, high, low, close, volume)",
            "get_flow(symbol, metric, period)": "Metrics: CVD, OI, OI_DELTA, TAKER, FUNDING, DEPTH, IMBALANCE",
            "get_regime(symbol, period)": "Returns RegimeInfo(regime, conf, direction, reason, indicators)",
            "get_price_change(symbol, period)": "Returns {change_percent, change_usd}",
        },
        "Decision_fields": {
            "operation": "Required: 'buy', 'sell', 'hold', 'close'",
            "symbol": "Required: Trading symbol string",
            "target_portion_of_balance": "Required for buy/sell/close: 0.1-1.0",
            "leverage": "Required for buy/sell/close: 1-50 (default: 10)",
            "max_price": "Required for buy or close short: maximum entry price",
            "min_price": "Required for sell or close long: minimum entry price",
            "time_in_force": "Optional: 'Ioc', 'Gtc', 'Alo' (default: 'Ioc')",
            "take_profit_price": "Optional: TP trigger price",
            "stop_loss_price": "Optional: SL trigger price",
            "tp_execution": "Optional: 'market' or 'limit' (default: 'limit')",
            "sl_execution": "Optional: 'market' or 'limit' (default: 'limit')",
            "reason": "Optional: Explanation string",
            "trading_strategy": "Optional: Entry thesis, risk controls",
        },
        "operation_values": ["buy", "sell", "hold", "close"],
        "supported_periods": ["1m", "5m", "15m", "1h", "4h"],
        "math_functions": {
            "usage": "Call via math.xxx (e.g., math.pow(10, 2))",
            "functions": ["sqrt", "log", "log10", "exp", "pow", "floor", "ceil", "fabs"],
        },
        "available_builtins": ["abs", "min", "max", "sum", "len", "round", "int", "float", "str", "bool", "list", "dict", "range", "enumerate", "zip", "sorted", "any", "all"],
        "debug_function": "log(message) - Print debug output",
    }


@router.post("/test-run", response_model=TestRunResponse)
def test_run_program(request: TestRunRequest, db: Session = Depends(get_db)):
    """
    Test-run a program in sandbox environment without saving.

    This API is designed for:
    1. Syntax validation before saving/binding
    2. AI-assisted debugging with detailed error info
    3. Quick iteration during development

    The endpoint only provides execution environment (MarketData with data_provider).
    Strategy code internally calls data_provider methods to get market data as needed.

    Returns comprehensive error information when execution fails.
    """
    import time
    import traceback
    from program_trader.executor import SandboxExecutor
    from program_trader.data_provider import DataProvider
    from program_trader.models import MarketData

    start_time = time.time()

    # Step 1: Validate code syntax first
    validation = validate_strategy_code(request.code)
    if not validation.is_valid:
        return TestRunResponse(
            success=False,
            error_type="ValidationError",
            error_message=f"Code validation failed: {'; '.join(validation.errors)}",
            suggestions=[
                "Ensure your code defines a class with should_trade(self, data: MarketData) method",
                "The method must return a Decision object",
            ],
            available_apis=_get_available_apis(),
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    # Step 2: Prepare execution environment
    # Note: test-run is for syntax validation only, no AI Trader is bound yet,
    # so there's no real wallet address. We use simulated account data here.
    # Strategy code can still call data_provider methods to get market data
    # (e.g., get_klines, get_indicator) which don't require a wallet.
    try:
        data_provider = DataProvider(db, account_id=0, environment="mainnet")
        market_data = MarketData(
            available_balance=10000.0,  # Simulated balance for testing
            total_equity=10000.0,
            used_margin=0.0,
            margin_usage_percent=0.0,
            maintenance_margin=0.0,
            positions={},  # No positions in test mode
            trigger_symbol=request.symbol,
            trigger_type="manual_test",
            _data_provider=data_provider,
        )

    except Exception as e:
        return TestRunResponse(
            success=False,
            error_type="DataError",
            error_message=f"Failed to initialize execution environment: {str(e)}",
            error_traceback=traceback.format_exc(),
            suggestions=["Check database connection"],
            available_apis=_get_available_apis(),
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    # Step 3: Execute in sandbox
    try:
        executor = SandboxExecutor(timeout_seconds=5)
        result = executor.execute(request.code, market_data, params={})

        execution_time = (time.time() - start_time) * 1000

        if result.success and result.decision:
            return TestRunResponse(
                success=True,
                decision=DecisionResult(
                    action=result.decision.operation,
                    symbol=result.decision.symbol,
                    size_usd=getattr(result.decision, 'size_usd', None),
                    leverage=result.decision.leverage,
                    reason=result.decision.reason,
                ),
                execution_time_ms=result.execution_time_ms,
            )
        else:
            # Execution failed - parse error details
            error_str = result.error or "Unknown error"
            tb_str = error_str

            # Determine error type
            error_type = "RuntimeError"
            if "ImportError" in error_str:
                error_type = "ImportError"
            elif "SyntaxError" in error_str:
                error_type = "SyntaxError"
            elif "NameError" in error_str:
                error_type = "NameError"
            elif "AttributeError" in error_str:
                error_type = "AttributeError"
            elif "TypeError" in error_str:
                error_type = "TypeError"
            elif "KeyError" in error_str:
                error_type = "KeyError"
            elif "timed out" in error_str.lower():
                error_type = "TimeoutError"
            elif "Validation failed" in error_str:
                error_type = "ValidationError"

            # Extract just the error message (first line after "Error:")
            error_msg = error_str.split('\n')[0] if '\n' in error_str else error_str
            if ": " in error_msg:
                error_msg = error_msg.split(": ", 1)[1]

            return TestRunResponse(
                success=False,
                error_type=error_type,
                error_message=error_msg,
                error_traceback=tb_str,
                error_location=_parse_error_location(tb_str, request.code),
                suggestions=_generate_suggestions(error_type, error_msg, tb_str),
                available_apis=_get_available_apis(),
                execution_time_ms=execution_time,
            )

    except Exception as e:
        tb_str = traceback.format_exc()
        error_type = type(e).__name__

        return TestRunResponse(
            success=False,
            error_type=error_type,
            error_message=str(e),
            error_traceback=tb_str,
            error_location=_parse_error_location(tb_str, request.code),
            suggestions=_generate_suggestions(error_type, str(e), tb_str),
            available_apis=_get_available_apis(),
            execution_time_ms=(time.time() - start_time) * 1000,
        )


# ============================================================================
# Reference Data Endpoints (must be before /{program_id} routes)
# ============================================================================

@router.get("/dev-guide")
def get_program_dev_guide(lang: str = "en") -> dict:
    """
    Get Program Trader development guide documentation.
    Supports English (default) and Chinese.
    """
    import os

    if lang == "zh":
        filename = "PROGRAM_DEV_GUIDE_ZH.md"
    else:
        filename = "PROGRAM_DEV_GUIDE.md"

    doc_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        filename
    )

    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Documentation file not found: {filename}")


@router.get("/signal-pools/", response_model=List[SignalPoolInfo])
def list_signal_pools(db: Session = Depends(get_db)):
    """List available signal pools."""
    pools = db.query(SignalPool).filter(SignalPool.enabled == True).all()
    result = []
    for pool in pools:
        symbols = pool.symbols
        if isinstance(symbols, str):
            try:
                symbols = json.loads(symbols)
            except:
                symbols = []
        result.append(SignalPoolInfo(
            id=pool.id,
            pool_name=pool.pool_name,
            symbols=symbols or [],
            enabled=pool.enabled,
        ))
    return result


@router.get("/accounts/", response_model=List[AccountInfo])
def list_accounts(db: Session = Depends(get_db)):
    """List available AI Traders for binding."""
    accounts = db.query(Account).filter(
        Account.is_active == "true",
        Account.account_type == "AI"
    ).all()
    return [AccountInfo(id=a.id, name=a.name, model=a.model) for a in accounts]


# ============================================================================
# Program CRUD Endpoints
# ============================================================================

@router.get("/", response_model=List[ProgramResponse])
def list_programs(db: Session = Depends(get_db)):
    """List all trading programs (code templates)."""
    user = get_default_user(db)
    programs = db.query(TradingProgram).filter(
        TradingProgram.user_id == user.id
    ).order_by(TradingProgram.updated_at.desc()).all()

    return [_program_to_response(p, db) for p in programs]


@router.post("/", response_model=ProgramResponse)
def create_program(data: ProgramCreate, db: Session = Depends(get_db)):
    """Create a new trading program."""
    user = get_default_user(db)

    validation = validate_strategy_code(data.code)
    if not validation.is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid code: {'; '.join(validation.errors)}")

    program = TradingProgram(
        user_id=user.id,
        name=data.name,
        description=data.description,
        code=data.code,
        params=json.dumps(data.params) if data.params else None,
        icon=data.icon,
    )
    db.add(program)
    db.commit()
    db.refresh(program)

    return _program_to_response(program, db)


# ============================================================================
# AI Program Coding API (MUST be before /{program_id} routes)
# ============================================================================

class AiProgramChatRequest(BaseModel):
    message: str
    account_id: int
    conversation_id: Optional[int] = None
    program_id: Optional[int] = None


class ConversationResponse(BaseModel):
    id: int
    program_id: Optional[int]
    title: str
    created_at: str
    updated_at: str


class SaveSuggestionResponse(BaseModel):
    code: str
    name: str
    description: str


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    saveSuggestion: Optional[SaveSuggestionResponse] = None
    reasoning_snapshot: Optional[str] = None
    tool_calls_log: Optional[List[Dict[str, Any]]] = None
    created_at: str
    is_complete: bool = True  # False = interrupted, can retry


@router.post("/ai-chat")
async def ai_program_chat(
    request: AiProgramChatRequest,
    db: Session = Depends(get_db)
):
    """
    AI-assisted program coding with SSE streaming.
    Returns Server-Sent Events for real-time updates.
    """
    from fastapi.responses import StreamingResponse
    from services.ai_program_service import generate_program_with_ai_stream

    user = db.query(User).first()
    user_id = user.id if user else 1

    def event_generator():
        yield from generate_program_with_ai_stream(
            db=db,
            account_id=request.account_id,
            user_message=request.message,
            conversation_id=request.conversation_id,
            program_id=request.program_id,
            user_id=user_id
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/ai-conversations", response_model=List[ConversationResponse])
async def list_ai_conversations(
    program_id: Optional[int] = None,
    limit: int = Query(default=20, le=100),
    db: Session = Depends(get_db)
):
    """List AI program coding conversations."""
    from database.models import AiProgramConversation

    user = db.query(User).first()
    user_id = user.id if user else 1

    query = db.query(AiProgramConversation).filter(
        AiProgramConversation.user_id == user_id
    )

    if program_id:
        query = query.filter(AiProgramConversation.program_id == program_id)

    conversations = query.order_by(
        AiProgramConversation.updated_at.desc()
    ).limit(limit).all()

    return [
        ConversationResponse(
            id=c.id,
            program_id=c.program_id,
            title=c.title,
            created_at=c.created_at.isoformat() if c.created_at else "",
            updated_at=c.updated_at.isoformat() if c.updated_at else ""
        )
        for c in conversations
    ]


@router.get("/ai-conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get messages for a specific conversation."""
    from database.models import AiProgramConversation, AiProgramMessage

    user = db.query(User).first()
    user_id = user.id if user else 1

    conversation = db.query(AiProgramConversation).filter(
        AiProgramConversation.id == conversation_id,
        AiProgramConversation.user_id == user_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = db.query(AiProgramMessage).filter(
        AiProgramMessage.conversation_id == conversation_id
    ).order_by(AiProgramMessage.created_at).all()

    result = []
    for m in messages:
        # Parse code_suggestion JSON to saveSuggestion object
        save_suggestion = None
        if m.code_suggestion:
            try:
                parsed = json.loads(m.code_suggestion)
                if isinstance(parsed, dict) and "code" in parsed:
                    save_suggestion = SaveSuggestionResponse(
                        code=parsed.get("code", ""),
                        name=parsed.get("name", "Saved Code"),
                        description=parsed.get("description", "")
                    )
                else:
                    # Old format: just code string, construct default object
                    save_suggestion = SaveSuggestionResponse(
                        code=m.code_suggestion,
                        name="Saved Code",
                        description=""
                    )
            except json.JSONDecodeError:
                # Not JSON, treat as plain code string
                save_suggestion = SaveSuggestionResponse(
                    code=m.code_suggestion,
                    name="Saved Code",
                    description=""
                )

        result.append(MessageResponse(
            id=m.id,
            role=m.role,
            content=m.content,
            saveSuggestion=save_suggestion,
            reasoning_snapshot=m.reasoning_snapshot,
            tool_calls_log=json.loads(m.tool_calls_log) if m.tool_calls_log else None,
            created_at=m.created_at.isoformat() if m.created_at else "",
            is_complete=m.is_complete if m.is_complete is not None else True
        ))

    return result


# ============================================================================
# Program CRUD with path parameters (MUST be after /ai-* routes)
# ============================================================================

@router.get("/{program_id}", response_model=ProgramResponse)
def get_program(program_id: int, db: Session = Depends(get_db)):
    """Get a trading program by ID."""
    user = get_default_user(db)
    program = db.query(TradingProgram).filter(
        TradingProgram.id == program_id,
        TradingProgram.user_id == user.id
    ).first()

    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    return _program_to_response(program, db)


@router.put("/{program_id}", response_model=ProgramResponse)
def update_program(program_id: int, data: ProgramUpdate, db: Session = Depends(get_db)):
    """Update a trading program."""
    user = get_default_user(db)
    program = db.query(TradingProgram).filter(
        TradingProgram.id == program_id,
        TradingProgram.user_id == user.id
    ).first()

    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    if data.code:
        validation = validate_strategy_code(data.code)
        if not validation.is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid code: {'; '.join(validation.errors)}")
        program.code = data.code

    if data.name is not None:
        program.name = data.name
    if data.description is not None:
        program.description = data.description
    if data.params is not None:
        program.params = json.dumps(data.params)
    if data.icon is not None:
        program.icon = data.icon

    db.commit()
    db.refresh(program)

    return _program_to_response(program, db)


@router.delete("/{program_id}")
def delete_program(program_id: int, db: Session = Depends(get_db)):
    """Delete a trading program."""
    user = get_default_user(db)
    program = db.query(TradingProgram).filter(
        TradingProgram.id == program_id,
        TradingProgram.user_id == user.id
    ).first()

    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    # Check if program has active bindings
    binding_count = db.query(AccountProgramBinding).filter(
        AccountProgramBinding.program_id == program_id
    ).count()
    if binding_count > 0:
        raise HTTPException(status_code=400, detail=f"Cannot delete: program is bound to {binding_count} AI Trader(s)")

    db.delete(program)
    db.commit()

    return {"success": True, "message": "Program deleted"}


@router.post("/validate", response_model=ValidationResponse)
def validate_code(data: dict, db: Session = Depends(get_db)):
    """Validate strategy code without saving."""
    code = data.get("code", "")
    if not code:
        raise HTTPException(status_code=400, detail="Code is required")

    validation = validate_strategy_code(code)
    return ValidationResponse(
        is_valid=validation.is_valid,
        errors=validation.errors,
        warnings=validation.warnings,
    )


# ============================================================================
# Binding Endpoints (N:N relationship between AI Traders and Programs)
# ============================================================================

@router.get("/bindings/", response_model=List[BindingResponse])
def list_bindings(
    program_id: Optional[int] = Query(None),
    account_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    """List program bindings, optionally filtered by program_id or account_id."""
    query = db.query(AccountProgramBinding)

    if program_id:
        query = query.filter(AccountProgramBinding.program_id == program_id)
    if account_id:
        query = query.filter(AccountProgramBinding.account_id == account_id)

    bindings = query.order_by(AccountProgramBinding.created_at.desc()).all()
    return [_binding_to_response(b, db) for b in bindings]


@router.post("/bindings/", response_model=BindingResponse)
def create_binding(data: BindingCreate, account_id: int = Query(...), db: Session = Depends(get_db)):
    """Create a new binding between an AI Trader and a Program."""
    # Verify account exists
    account = db.query(Account).filter(Account.id == account_id).first()
    if not account:
        raise HTTPException(status_code=404, detail="AI Trader not found")

    # Verify program exists
    program = db.query(TradingProgram).filter(TradingProgram.id == data.program_id).first()
    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    # Check for duplicate binding
    existing = db.query(AccountProgramBinding).filter(
        AccountProgramBinding.account_id == account_id,
        AccountProgramBinding.program_id == data.program_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Binding already exists")

    binding = AccountProgramBinding(
        account_id=account_id,
        program_id=data.program_id,
        signal_pool_ids=json.dumps(data.signal_pool_ids) if data.signal_pool_ids else None,
        trigger_interval=data.trigger_interval,
        scheduled_trigger_enabled=data.scheduled_trigger_enabled,
        is_active=data.is_active,
        params_override=json.dumps(data.params_override) if data.params_override else None,
    )
    db.add(binding)
    db.commit()
    db.refresh(binding)

    return _binding_to_response(binding, db)


@router.put("/bindings/{binding_id}", response_model=BindingResponse)
def update_binding(binding_id: int, data: BindingUpdate, db: Session = Depends(get_db)):
    """Update a program binding's trigger configuration."""
    binding = db.query(AccountProgramBinding).filter(
        AccountProgramBinding.id == binding_id
    ).first()

    if not binding:
        raise HTTPException(status_code=404, detail="Binding not found")

    if data.signal_pool_ids is not None:
        binding.signal_pool_ids = json.dumps(data.signal_pool_ids)
    if data.trigger_interval is not None:
        binding.trigger_interval = data.trigger_interval
    if data.scheduled_trigger_enabled is not None:
        binding.scheduled_trigger_enabled = data.scheduled_trigger_enabled
    if data.is_active is not None:
        binding.is_active = data.is_active
    if data.params_override is not None:
        binding.params_override = json.dumps(data.params_override)

    db.commit()
    db.refresh(binding)

    return _binding_to_response(binding, db)


@router.delete("/bindings/{binding_id}")
def delete_binding(binding_id: int, db: Session = Depends(get_db)):
    """Delete a program binding."""
    binding = db.query(AccountProgramBinding).filter(
        AccountProgramBinding.id == binding_id
    ).first()

    if not binding:
        raise HTTPException(status_code=404, detail="Binding not found")

    db.delete(binding)
    db.commit()

    return {"success": True, "message": "Binding deleted"}


# ============================================================================
# Preview Run API (for testing with real account data)
# ============================================================================

class PreviewRunResponse(BaseModel):
    """Response for preview-run API with full execution details."""
    success: bool
    error: Optional[str] = None

    # Input data snapshot
    input_data: Optional[Dict[str, Any]] = None

    # Data queries made during execution
    data_queries: List[Dict[str, Any]] = []

    # Execution logs from log() calls
    execution_logs: List[str] = []

    # Decision result
    decision: Optional[Dict[str, Any]] = None

    # Timing
    execution_time_ms: float = 0.0


@router.post("/bindings/{binding_id}/preview-run", response_model=PreviewRunResponse)
def preview_run_binding(binding_id: int, db: Session = Depends(get_db)):
    """
    Preview-run a program binding with real account data in sandbox environment.

    Uses the bound AI Trader's real account data (positions, balance, etc.)
    to test the strategy without actually executing trades.

    The endpoint provides execution environment with real account state.
    Strategy code internally calls data_provider methods to get market data as needed.

    Returns detailed execution info including:
    - Input data snapshot (account state, positions)
    - All data queries made (indicators, flow metrics)
    - Execution logs from log() calls
    - Final decision
    """
    import time
    import traceback
    from program_trader.executor import SandboxExecutor
    from program_trader.data_provider import DataProvider
    from program_trader.models import MarketData
    from services.hyperliquid_environment import get_hyperliquid_client, get_global_trading_mode
    from database.models import HyperliquidWallet

    start_time = time.time()

    # Get binding
    binding = db.query(AccountProgramBinding).filter(
        AccountProgramBinding.id == binding_id
    ).first()
    if not binding:
        raise HTTPException(status_code=404, detail="Binding not found")

    # Get program
    program = db.query(TradingProgram).filter(
        TradingProgram.id == binding.program_id
    ).first()
    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    # Get wallet for this AI Trader based on current global environment
    global_environment = get_global_trading_mode(db)
    wallet = db.query(HyperliquidWallet).filter(
        HyperliquidWallet.account_id == binding.account_id,
        HyperliquidWallet.is_active == "true",
        HyperliquidWallet.environment == global_environment
    ).first()

    if not wallet:
        return PreviewRunResponse(
            success=False,
            error=f"No active {global_environment} wallet found for this AI Trader",
            execution_time_ms=(time.time() - start_time) * 1000
        )

    # Determine trigger context from binding config
    trigger_symbol = "BTC"  # Default
    trigger_type = "scheduled" if binding.scheduled_trigger_enabled else "signal"

    # If signal pools configured, use first pool's first symbol
    if binding.signal_pool_ids:
        try:
            pool_ids = json.loads(binding.signal_pool_ids)
            if pool_ids:
                pool = db.query(SignalPool).filter(SignalPool.id == pool_ids[0]).first()
                if pool and pool.symbols:
                    symbols = pool.symbols
                    if isinstance(symbols, str):
                        symbols = json.loads(symbols)
                    if symbols:
                        trigger_symbol = symbols[0]
        except:
            pass

    # Create trading client and data provider with query recording
    try:
        trading_client = get_hyperliquid_client(
            db,
            binding.account_id,
            override_environment=wallet.environment
        )
        data_provider = DataProvider(
            db,
            account_id=binding.account_id,
            environment=wallet.environment,
            trading_client=trading_client,
            record_queries=True  # Enable query logging
        )
    except Exception as e:
        return PreviewRunResponse(
            success=False,
            error=f"Failed to initialize trading client: {str(e)}",
            execution_time_ms=(time.time() - start_time) * 1000
        )

    # Build MarketData with real account data
    try:
        account_info = data_provider.get_account_info()
        positions = data_provider.get_positions()
        recent_trades = data_provider.get_recent_trades()
        open_orders = data_provider.get_open_orders()

        market_data = MarketData(
            available_balance=account_info.get("available_balance", 0.0),
            total_equity=account_info.get("total_equity", 0.0),
            used_margin=account_info.get("used_margin", 0.0),
            margin_usage_percent=account_info.get("margin_usage_percent", 0.0),
            maintenance_margin=account_info.get("maintenance_margin", 0.0),
            positions=positions,
            recent_trades=recent_trades,
            open_orders=open_orders,
            trigger_symbol=trigger_symbol,
            trigger_type=trigger_type,
            _data_provider=data_provider,
        )

        # Build input data snapshot for response
        input_data = {
            "trigger_symbol": trigger_symbol,
            "trigger_type": trigger_type,
            "environment": wallet.environment,
            "signal_pool_name": "",  # Preview run doesn't have signal context
            "pool_logic": "OR",
            "triggered_signals": [],
            "trigger_market_regime": None,
            "max_leverage": 20,  # Default max
            "default_leverage": 3,  # Default
            "available_balance": account_info.get("available_balance", 0.0),
            "total_equity": account_info.get("total_equity", 0.0),
            "used_margin": account_info.get("used_margin", 0.0),
            "margin_usage_percent": account_info.get("margin_usage_percent", 0.0),
            "positions": {k: {"side": v.side, "size": v.size, "entry_price": v.entry_price,
                            "unrealized_pnl": v.unrealized_pnl, "leverage": getattr(v, 'leverage', None)} for k, v in positions.items()},
            "positions_count": len(positions),
            "open_orders": [
                {
                    "order_id": o.order_id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "direction": o.direction,
                    "order_type": o.order_type,
                    "size": o.size,
                    "price": o.price,
                    "trigger_price": o.trigger_price,
                    "reduce_only": o.reduce_only,
                    "timestamp": o.timestamp,
                }
                for o in open_orders
            ],
            "open_orders_count": len(open_orders),
            "recent_trades_count": len(recent_trades),
        }
    except Exception as e:
        return PreviewRunResponse(
            success=False,
            error=f"Failed to load account data: {str(e)}",
            execution_time_ms=(time.time() - start_time) * 1000
        )

    # Execute in sandbox
    try:
        # Get params override if any
        params = {}
        if binding.params_override:
            try:
                params = json.loads(binding.params_override)
            except:
                pass

        executor = SandboxExecutor(timeout_seconds=5)
        result = executor.execute(program.code, market_data, params=params)

        execution_time = (time.time() - start_time) * 1000

        if result.success and result.decision:
            return PreviewRunResponse(
                success=True,
                input_data=input_data,
                data_queries=data_provider.get_query_log(),
                execution_logs=result.logs if hasattr(result, 'logs') else [],
                decision=result.decision.to_dict(),
                execution_time_ms=execution_time
            )
        else:
            return PreviewRunResponse(
                success=False,
                error=result.error or "Unknown execution error",
                input_data=input_data,
                data_queries=data_provider.get_query_log(),
                execution_logs=result.logs if hasattr(result, 'logs') else [],
                execution_time_ms=execution_time
            )
    except Exception as e:
        return PreviewRunResponse(
            success=False,
            error=f"Execution failed: {str(e)}",
            input_data=input_data,
            data_queries=data_provider.get_query_log(),
            execution_time_ms=(time.time() - start_time) * 1000
        )


# ============================================================================
# Backtest Endpoint
# ============================================================================

@router.post("/{program_id}/backtest", response_model=BacktestResponse)
def run_backtest(program_id: int, request: BacktestRequest, db: Session = Depends(get_db)):
    """Run backtest on a trading program."""
    from database.models import CryptoKline
    from datetime import datetime, timedelta

    user = get_default_user(db)
    program = db.query(TradingProgram).filter(
        TradingProgram.id == program_id,
        TradingProgram.user_id == user.id
    ).first()

    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    start_time = datetime.utcnow() - timedelta(days=request.days)
    rows = db.query(CryptoKline).filter(
        CryptoKline.symbol == request.symbol,
        CryptoKline.period == request.period,
        CryptoKline.timestamp >= start_time,
    ).order_by(CryptoKline.timestamp).all()

    if len(rows) < 100:
        raise HTTPException(status_code=400, detail="Insufficient historical data")

    klines = [
        Kline(
            timestamp=int(row.timestamp.timestamp() * 1000),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        for row in rows
    ]

    engine = BacktestEngine(initial_balance=request.initial_balance)
    kline_dict = {f"{request.symbol}_{request.period}": klines}
    params = json.loads(program.params) if program.params else {}

    result = engine.run(
        code=program.code,
        klines=kline_dict,
        symbol=request.symbol,
        period=request.period,
        params=params,
    )

    program.last_backtest_result = json.dumps({
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "total_pnl": result.total_pnl,
        "max_drawdown": result.max_drawdown,
    })
    program.last_backtest_at = datetime.utcnow()
    db.commit()

    return BacktestResponse(
        success=result.success,
        error=result.error,
        total_trades=result.total_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        win_rate=result.win_rate,
        total_pnl=result.total_pnl,
        max_drawdown=result.max_drawdown,
        equity_curve=result.equity_curve[-100:] if result.equity_curve else [],
    )


# ============================================================================
# Execution Log Endpoints
# ============================================================================

class ExecutionLogResponse(BaseModel):
    id: int
    binding_id: Optional[int]  # Can be NULL if binding was deleted
    account_id: int
    account_name: str
    program_id: Optional[int]  # Can be NULL if program was deleted
    program_name: str
    trigger_type: str  # "signal" or "scheduled"
    trigger_symbol: Optional[str]
    signal_pool_id: Optional[int]
    signal_pool_name: Optional[str]
    wallet_address: Optional[str]
    success: bool
    decision_action: Optional[str]
    decision_symbol: Optional[str]
    decision_size_usd: Optional[float]
    decision_leverage: Optional[int]
    decision_reason: Optional[str]
    error_message: Optional[str]
    execution_time_ms: Optional[float]
    # Execution snapshots for debugging/analysis
    market_context: Optional[Dict[str, Any]] = None  # Market data at execution time
    params_snapshot: Optional[Dict[str, Any]] = None  # Params used for execution
    decision_json: Optional[Dict[str, Any]] = None  # Full decision object
    created_at: str

    class Config:
        from_attributes = True


@router.get("/executions/", response_model=List[ExecutionLogResponse])
def list_executions(
    account_id: Optional[int] = Query(None),
    program_id: Optional[int] = Query(None),
    environment: Optional[str] = Query(None, regex="^(testnet|mainnet)$"),
    before: Optional[str] = Query(None, description="ISO timestamp for pagination, returns logs before this time"),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    """List program execution logs for Feed display."""
    query = db.query(ProgramExecutionLog)

    if account_id:
        query = query.filter(ProgramExecutionLog.account_id == account_id)
    if program_id:
        query = query.filter(ProgramExecutionLog.program_id == program_id)
    if environment:
        query = query.filter(ProgramExecutionLog.environment == environment)
    if before:
        from datetime import datetime
        before_dt = datetime.fromisoformat(before.replace('Z', '+00:00'))
        query = query.filter(ProgramExecutionLog.created_at < before_dt)

    logs = query.order_by(ProgramExecutionLog.created_at.desc()).limit(limit).all()

    result = []
    for log in logs:
        # Get account name
        account = db.query(Account).filter(Account.id == log.account_id).first()
        account_name = account.name if account else "Unknown"

        # Get program name
        program = db.query(TradingProgram).filter(TradingProgram.id == log.program_id).first()
        program_name = program.name if program else "Unknown"

        # Get signal pool name if applicable
        signal_pool_name = None
        if log.signal_pool_id:
            pool = db.query(SignalPool).filter(SignalPool.id == log.signal_pool_id).first()
            signal_pool_name = pool.pool_name if pool else None

        result.append(ExecutionLogResponse(
            id=log.id,
            binding_id=log.binding_id,
            account_id=log.account_id,
            account_name=account_name,
            program_id=log.program_id,
            program_name=program_name,
            trigger_type=log.trigger_type,
            trigger_symbol=log.trigger_symbol,
            signal_pool_id=log.signal_pool_id,
            signal_pool_name=signal_pool_name,
            wallet_address=log.wallet_address,
            success=log.success,
            decision_action=log.decision_action,
            decision_symbol=log.decision_symbol,
            decision_size_usd=log.decision_size_usd,
            decision_leverage=log.decision_leverage,
            decision_reason=log.decision_reason,
            error_message=log.error_message,
            execution_time_ms=log.execution_time_ms,
            market_context=json.loads(log.market_context) if log.market_context else None,
            params_snapshot=json.loads(log.params_snapshot) if log.params_snapshot else None,
            decision_json=json.loads(log.decision_json) if log.decision_json else None,
            created_at=log.created_at.isoformat() if log.created_at else "",
        ))

    return result


# ============================================================================
# Market Data Query API - For AI to query current market data before writing code
# ============================================================================

class MarketDataQueryRequest(BaseModel):
    """Request for querying market data."""
    symbol: str = "BTC"
    period: str = "1h"
    indicators: Optional[List[str]] = None  # If None, return all
    flow_metrics: Optional[List[str]] = None  # If None, return all


class MarketDataQueryResponse(BaseModel):
    """Response with current market data snapshot."""
    symbol: str
    period: str
    price: Optional[float]
    indicators: Dict[str, Any]
    flow_metrics: Dict[str, Any]
    regime: Optional[Dict[str, Any]]
    klines_sample: Optional[List[Dict[str, Any]]]
    timestamp: str


@router.post("/query-market-data", response_model=MarketDataQueryResponse)
async def query_market_data(
    request: MarketDataQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Query current market data for a symbol.

    Use this API to check current indicator values before setting thresholds
    in your strategy code. Returns real-time data for all available indicators.
    """
    import time
    from datetime import datetime
    from services.technical_indicators import calculate_indicator
    from services.market_flow_indicators import get_flow_indicators_for_prompt
    from services.market_regime_service import get_market_regime
    from services.hyperliquid_market_data import get_last_price_from_hyperliquid
    from database.models import CryptoKline
    from sqlalchemy import desc

    symbol = request.symbol.upper()
    period = request.period
    current_time_ms = int(time.time() * 1000)

    # Get current price
    price = None
    try:
        price = get_last_price_from_hyperliquid(symbol, "mainnet")
        if price:
            price = float(price)
    except Exception:
        pass

    # Default indicators and flow metrics
    all_indicators = ["RSI14", "RSI7", "MA5", "MA10", "MA20", "EMA20", "EMA50",
                      "EMA100", "MACD", "BOLL", "ATR14", "VWAP", "STOCH", "OBV"]
    all_flow_metrics = ["CVD", "OI", "OI_DELTA", "TAKER", "FUNDING", "DEPTH", "IMBALANCE"]

    indicators_to_query = request.indicators or all_indicators
    flow_metrics_to_query = request.flow_metrics or all_flow_metrics

    # Query indicators
    indicators_result = {}
    for ind in indicators_to_query:
        try:
            result = calculate_indicator(db, symbol, ind, period, current_time_ms)
            # Note: empty dict {} or 0.0 values are valid results
            indicators_result[ind] = result if result is not None else None
        except Exception as e:
            indicators_result[ind] = {"error": str(e)}

    # Query flow metrics - return full structure (not simplified single value)
    flow_result = {}
    try:
        full_flow_data = get_flow_indicators_for_prompt(
            db, symbol, period, flow_metrics_to_query, current_time_ms
        )
        for metric in flow_metrics_to_query:
            flow_result[metric] = full_flow_data.get(metric) if full_flow_data else None
    except Exception as e:
        for metric in flow_metrics_to_query:
            flow_result[metric] = {"error": str(e)}

    # Query regime - return full structure including indicators
    regime_result = None
    try:
        regime = get_market_regime(db, symbol, period, timestamp_ms=current_time_ms)
        if regime:
            regime_result = {
                "regime": regime.get("regime", "noise"),
                "confidence": regime.get("confidence", 0.0),
                "direction": regime.get("direction", "neutral"),
                "reason": regime.get("reason", ""),
                "indicators": regime.get("indicators", {})
            }
    except Exception:
        pass

    # Get sample klines (last 5)
    klines_sample = None
    try:
        rows = (
            db.query(CryptoKline)
            .filter(CryptoKline.symbol == symbol, CryptoKline.period == period)
            .order_by(desc(CryptoKline.timestamp))
            .limit(5)
            .all()
        )
        if rows:
            klines_sample = [
                {
                    "timestamp": row.timestamp,
                    "open": float(row.open_price) if row.open_price else 0,
                    "high": float(row.high_price) if row.high_price else 0,
                    "low": float(row.low_price) if row.low_price else 0,
                    "close": float(row.close_price) if row.close_price else 0,
                    "volume": float(row.volume) if row.volume else 0,
                }
                for row in reversed(rows)
            ]
    except Exception:
        pass

    return MarketDataQueryResponse(
        symbol=symbol,
        period=period,
        price=price,
        indicators=indicators_result,
        flow_metrics=flow_result,
        regime=regime_result,
        klines_sample=klines_sample,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@router.get("/available-symbols")
async def get_available_symbols(db: Session = Depends(get_db)):
    """
    Get list of symbols with available market data.
    """
    from database.models import CryptoKline
    from sqlalchemy import distinct

    try:
        symbols = db.query(distinct(CryptoKline.symbol)).all()
        return {"symbols": sorted([s[0] for s in symbols])}
    except Exception as e:
        return {"symbols": ["BTC", "ETH", "SOL"], "error": str(e)}


# ============================================================================
# Backtest API
# ============================================================================

class BacktestRequest(BaseModel):
    """Request model for program backtest."""
    binding_id: int
    start_time: str  # ISO format: "2024-01-01" or "2024-01-01T00:00:00"
    end_time: str
    initial_balance: float = 10000.0
    slippage_percent: float = 0.05
    fee_rate: float = 0.035


@router.post("/backtest")
async def run_backtest(request: BacktestRequest, db: Session = Depends(get_db)):
    """
    Run backtest for a program binding with SSE progress updates.

    Returns Server-Sent Events with:
    - type: "init" - Initial trigger count
    - type: "progress" - Execution progress updates
    - type: "complete" - Final results
    - type: "error" - Error message
    """
    from backtest import (
        BacktestConfig, ProgramBacktestEngine,
        BacktestResult as BacktestResultData, TriggerEvent
    )
    from backtest.engine import INTERVAL_MS

    # Get binding info
    binding = db.query(AccountProgramBinding).filter(
        AccountProgramBinding.id == request.binding_id
    ).first()

    if not binding:
        raise HTTPException(status_code=404, detail="Binding not found")

    # Get program
    program = db.query(TradingProgram).filter(
        TradingProgram.id == binding.program_id
    ).first()

    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    # Get signal pool symbols
    signal_pool_ids = []
    symbols = set()

    if binding.signal_pool_ids:
        pool_ids = binding.signal_pool_ids
        if isinstance(pool_ids, str):
            pool_ids = json.loads(pool_ids)
        signal_pool_ids = pool_ids

        for pool_id in pool_ids:
            pool = db.query(SignalPool).filter(SignalPool.id == pool_id).first()
            if pool and pool.symbols:
                # symbols is a list field
                pool_symbols = pool.symbols
                if isinstance(pool_symbols, str):
                    pool_symbols = json.loads(pool_symbols)
                for sym in pool_symbols:
                    symbols.add(sym)

    # Default to BTC if no symbols found
    if not symbols:
        symbols = {"BTC"}

    # Determine scheduled interval
    scheduled_interval = None
    if binding.scheduled_trigger_enabled and binding.trigger_interval:
        interval_sec = binding.trigger_interval
        for name, ms in INTERVAL_MS.items():
            if ms // 1000 == interval_sec:
                scheduled_interval = name
                break

    # Create backtest config
    config = BacktestConfig(
        code=program.code,
        signal_pool_ids=signal_pool_ids,
        symbols=list(symbols),
        start_time=request.start_time,
        end_time=request.end_time,
        scheduled_interval=scheduled_interval,
        initial_balance=request.initial_balance,
        slippage_percent=request.slippage_percent,
        fee_rate=request.fee_rate,
    )

    async def generate_events():
        """Generate SSE events during backtest execution."""
        engine = ProgramBacktestEngine(db)
        backtest_record = None

        try:
            # Phase 1: Generate triggers (send calculating status)
            yield f"data: {json.dumps({'type': 'calculating', 'message': 'Calculating trigger points...'})}\n\n"
            await asyncio.sleep(0.01)

            triggers = engine._generate_trigger_events(config)

            if not triggers:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No triggers generated'})}\n\n"
                return

            # Create backtest record in database
            backtest_record = BacktestResult(
                backtest_type="program",
                binding_id=request.binding_id,
                user_id=binding.account.user_id if binding.account else None,
                config=json.dumps({
                    "signal_pool_ids": signal_pool_ids,
                    "symbols": list(symbols),
                    "scheduled_interval": scheduled_interval,
                    "slippage_percent": request.slippage_percent,
                    "fee_rate": request.fee_rate,
                }),
                start_time=config.start_time,
                end_time=config.end_time,
                initial_balance=config.initial_balance,
                total_triggers=len(triggers),
                status="running",
            )
            db.add(backtest_record)
            db.commit()
            db.refresh(backtest_record)

            # Send init event with trigger count and backtest_id
            yield f"data: {json.dumps({'type': 'init', 'total_triggers': len(triggers), 'backtest_id': backtest_record.id})}\n\n"
            await asyncio.sleep(0.01)

            # Phase 2: Run backtest with progress updates
            async for event in _run_backtest_with_progress(engine, config, triggers, db, backtest_record.id):
                yield event

        except Exception as e:
            if backtest_record:
                backtest_record.status = "error"
                backtest_record.error_message = str(e)
                db.commit()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


async def _run_backtest_with_progress(engine, config, triggers, db, backtest_id):
    """
    Run backtest event loop with progress updates and database logging.

    Yields SSE events for progress and final results.
    Records each trigger's decision to BacktestTriggerLog.
    """
    from backtest import VirtualAccount, ExecutionSimulator, HistoricalDataProvider
    from backtest.models import BacktestTradeRecord
    from program_trader.executor import SandboxExecutor
    from program_trader.models import MarketData, Position
    from datetime import datetime, timezone
    import time

    start_time = time.time()

    # Initialize components
    account = VirtualAccount(initial_balance=config.initial_balance)
    simulator = ExecutionSimulator(
        slippage_percent=config.slippage_percent,
        fee_rate=config.fee_rate,
    )
    data_provider = HistoricalDataProvider(
        db=engine.db,
        symbols=config.symbols,
        start_time_ms=config.start_time_ms,
        end_time_ms=config.end_time_ms,
    )
    executor = SandboxExecutor(timeout_seconds=5)

    trades = []
    equity_curve = []
    total = len(triggers)
    progress_interval = max(1, total // 50)  # Update every 2%

    # Use continuous index for trigger logs (not enumerate's i which may have gaps)
    trigger_log_index = 0

    for i, trigger in enumerate(triggers):
        equity_before = account.equity

        # Set current time and clear query log for this trigger
        data_provider.set_current_time(trigger.timestamp)
        data_provider.clear_query_log()

        # Get current prices
        prices = data_provider.get_current_prices(config.symbols)
        if not prices:
            continue

        # Check TP/SL triggers and record them
        tp_sl_trades = simulator.check_tp_sl_triggers(account, prices, trigger.timestamp)
        for tp_sl_trade in tp_sl_trades:
            trades.append(tp_sl_trade)
            # Record TP/SL trigger as separate log entry
            tp_sl_time = datetime.fromtimestamp(tp_sl_trade.exit_timestamp / 1000, tz=timezone.utc)
            tp_sl_log = BacktestTriggerLog(
                backtest_id=backtest_id,
                trigger_index=trigger_log_index,  # Use continuous index
                trigger_type=tp_sl_trade.exit_reason,  # "tp" or "sl"
                trigger_time=tp_sl_time,
                symbol=tp_sl_trade.symbol,
                decision_type="program",
                decision_action="close",
                decision_symbol=tp_sl_trade.symbol,
                decision_side=tp_sl_trade.side,
                decision_size=tp_sl_trade.size,
                decision_reason=tp_sl_trade.reason,
                entry_price=tp_sl_trade.entry_price,
                exit_price=tp_sl_trade.exit_price,
                pnl=0,  # Will set realized_pnl instead
                fee=tp_sl_trade.fee,
                unrealized_pnl=0,
                realized_pnl=tp_sl_trade.pnl,
                equity_before=equity_before,
                equity_after=account.equity,
                decision_input=json.dumps({
                    "trigger": tp_sl_trade.exit_reason.upper(),
                    "entry_price": tp_sl_trade.entry_price,
                    "exit_price": tp_sl_trade.exit_price,
                }),
                decision_output=None,
            )
            db.add(tp_sl_log)
            equity_before = account.equity  # Update for next iteration

        # Update equity
        account.update_equity(prices)

        # Determine trigger symbol
        trigger_symbol = trigger.symbol if trigger.symbol else config.symbols[0]

        # Build MarketData
        market_data = engine._build_market_data(
            account, data_provider, trigger, trigger_symbol
        )

        # Execute strategy
        result = executor.execute(config.code, market_data, {})

        # Prepare trigger log data
        trigger_time = datetime.fromtimestamp(trigger.timestamp / 1000, tz=timezone.utc)
        decision_action = "hold"
        decision_symbol = trigger_symbol
        decision_side = None
        decision_size = None
        decision_reason = ""
        entry_price = prices.get(trigger_symbol, 0)
        trade_fee = 0.0
        trade_realized_pnl = 0.0
        execution_error = None

        # Build decision input snapshot (minimal for storage)
        decision_input = {
            "balance": account.balance,
            "equity": account.equity,
            "trigger_type": trigger.trigger_type,
            "trigger_symbol": trigger_symbol,
            "prices": prices,
            "positions": {k: {"side": v.side, "size": v.size, "entry_price": v.entry_price}
                         for k, v in account.positions.items()},
        }

        decision_output = None
        trade = None

        if result.success and result.decision:
            decision = result.decision
            symbol = decision.symbol or trigger_symbol
            current_price = prices.get(symbol, 0)

            decision_action = decision.operation
            decision_symbol = symbol
            decision_reason = decision.reason

            # Build decision output
            decision_output = decision.to_dict() if hasattr(decision, 'to_dict') else {
                "operation": decision.operation,
                "symbol": decision.symbol,
                "reason": decision.reason,
            }

            if current_price > 0 and decision.operation != "hold":
                signal_names = [
                    s.get("signal_name", "") for s in (trigger.triggered_signals or [])
                ]
                trade = simulator.execute_decision(
                    decision=decision,
                    account=account,
                    current_price=current_price,
                    timestamp=trigger.timestamp,
                    trigger_type=trigger.trigger_type,
                    pool_name=trigger.pool_name,
                    triggered_signals=signal_names,
                )
                if trade:
                    trades.append(trade)
                    decision_side = trade.side
                    decision_size = trade.size
                    entry_price = trade.entry_price
                    trade_fee = trade.fee
                    trade_realized_pnl = trade.pnl if trade.operation == "close" else 0
        elif not result.success:
            execution_error = result.error

        # Update equity and record
        account.update_equity(prices)
        equity_after = account.equity

        # Calculate current unrealized PnL for this symbol
        current_unrealized_pnl = account.unrealized_pnl_total

        equity_curve.append({
            "timestamp": trigger.timestamp,
            "equity": account.equity,
            "balance": account.balance,
        })

        # Save trigger log to database
        # Capture data queries and execution logs
        data_queries = data_provider.get_query_log()
        execution_logs = result.logs if result else []

        trigger_log = BacktestTriggerLog(
            backtest_id=backtest_id,
            trigger_index=trigger_log_index,  # Use continuous index
            trigger_type=trigger.trigger_type,
            trigger_time=trigger_time,
            symbol=trigger_symbol,
            decision_type="program",
            decision_action=decision_action,
            decision_symbol=decision_symbol,
            decision_side=decision_side,
            decision_size=decision_size,
            decision_reason=decision_reason,
            entry_price=entry_price,
            pnl=0,  # Deprecated, use realized_pnl instead
            fee=trade_fee,
            unrealized_pnl=current_unrealized_pnl,
            realized_pnl=trade_realized_pnl,
            equity_before=equity_before,
            equity_after=equity_after,
            decision_input=json.dumps(decision_input),
            decision_output=json.dumps(decision_output) if decision_output else None,
            data_queries=json.dumps(data_queries) if data_queries else None,
            execution_logs=json.dumps(execution_logs) if execution_logs else None,
            execution_error=execution_error,
        )
        db.add(trigger_log)

        # Increment continuous index after saving
        trigger_log_index += 1

        # Commit periodically to avoid large transactions
        if trigger_log_index % 100 == 0:
            db.commit()

        # Yield progress update periodically
        if (i + 1) % progress_interval == 0 or i == total - 1:
            yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': total, 'equity': account.equity})}\n\n"
            await asyncio.sleep(0.001)  # Allow other tasks

    # Final commit for remaining logs
    db.commit()

    # Calculate final statistics
    calc_result = engine._calculate_result(trades, equity_curve, triggers, account, config)
    calc_result.execution_time_ms = (time.time() - start_time) * 1000

    # Update backtest record with final results
    backtest_record = db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()
    if backtest_record:
        backtest_record.final_equity = account.equity
        backtest_record.total_pnl = calc_result.total_pnl
        backtest_record.total_pnl_percent = calc_result.total_pnl_percent
        backtest_record.max_drawdown = calc_result.max_drawdown
        backtest_record.max_drawdown_percent = calc_result.max_drawdown_percent
        backtest_record.total_trades = calc_result.total_trades
        backtest_record.winning_trades = calc_result.winning_trades
        backtest_record.losing_trades = calc_result.losing_trades
        backtest_record.win_rate = calc_result.win_rate
        backtest_record.profit_factor = calc_result.profit_factor
        backtest_record.sharpe_ratio = calc_result.sharpe_ratio
        backtest_record.equity_curve = json.dumps(equity_curve)
        backtest_record.execution_time_ms = int(calc_result.execution_time_ms)
        backtest_record.status = "completed"
        backtest_record.completed_at = datetime.now(timezone.utc)
        db.commit()

    # Yield complete event with results
    complete_data = {
        "type": "complete",
        "backtest_id": backtest_id,
        "success": calc_result.success,
        "total_pnl": calc_result.total_pnl,
        "total_pnl_percent": calc_result.total_pnl_percent,
        "max_drawdown": calc_result.max_drawdown,
        "max_drawdown_percent": calc_result.max_drawdown_percent,
        "sharpe_ratio": calc_result.sharpe_ratio,
        "total_trades": calc_result.total_trades,
        "winning_trades": calc_result.winning_trades,
        "losing_trades": calc_result.losing_trades,
        "win_rate": calc_result.win_rate,
        "profit_factor": calc_result.profit_factor,
        "avg_win": calc_result.avg_win,
        "avg_loss": calc_result.avg_loss,
        "largest_win": calc_result.largest_win,
        "largest_loss": calc_result.largest_loss,
        "total_triggers": calc_result.total_triggers,
        "signal_triggers": calc_result.signal_triggers,
        "scheduled_triggers": calc_result.scheduled_triggers,
        "execution_time_ms": calc_result.execution_time_ms,
        "equity_curve": calc_result.equity_curve,
        "trades": [_trade_to_dict(t) for t in calc_result.trades],
    }
    yield f"data: {json.dumps(complete_data)}\n\n"


def _trade_to_dict(trade):
    """Convert BacktestTradeRecord to dict."""
    return {
        "timestamp": trade.timestamp,
        "trigger_type": trade.trigger_type,
        "symbol": trade.symbol,
        "operation": trade.operation,
        "side": trade.side,
        "entry_price": trade.entry_price,
        "size": trade.size,
        "leverage": trade.leverage,
        "exit_price": trade.exit_price,
        "exit_timestamp": trade.exit_timestamp,
        "exit_reason": trade.exit_reason,
        "pnl": trade.pnl,
        "pnl_percent": trade.pnl_percent,
        "fee": trade.fee,
        "reason": trade.reason,
        "pool_name": trade.pool_name,
        "triggered_signals": trade.triggered_signals,
    }


# ============================================================================
# Backtest Detail Query APIs
# ============================================================================

@router.get("/backtest/history")
def get_backtest_history(
    binding_id: int = Query(..., description="Binding ID to get history for"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Get backtest history for a specific binding.

    Returns a list of past backtest results (summary only, no equity curve).
    """
    query = db.query(BacktestResult).filter(
        BacktestResult.binding_id == binding_id,
        BacktestResult.backtest_type == 'program'
    ).order_by(BacktestResult.created_at.desc())

    total = query.count()
    backtests = query.offset(offset).limit(limit).all()

    results = []
    for bt in backtests:
        config = bt.config
        if isinstance(config, str):
            config = json.loads(config)

        results.append({
            "id": bt.id,
            "config": config,
            "start_time": bt.start_time.isoformat() if bt.start_time else None,
            "end_time": bt.end_time.isoformat() if bt.end_time else None,
            "initial_balance": bt.initial_balance,
            "final_equity": bt.final_equity,
            "total_pnl": bt.total_pnl,
            "total_pnl_percent": bt.total_pnl_percent,
            "max_drawdown_percent": bt.max_drawdown_percent,
            "total_triggers": bt.total_triggers,
            "total_trades": bt.total_trades,
            "win_rate": bt.win_rate,
            "status": bt.status,
            "created_at": bt.created_at.isoformat() if bt.created_at else None,
        })

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": results
    }


@router.get("/backtest/{backtest_id}")
def get_backtest_result(backtest_id: int, db: Session = Depends(get_db)):
    """
    Get backtest result summary by ID.

    Returns the backtest result without trigger logs (use /triggers endpoint for logs).
    """
    backtest = db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()
    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    # Parse equity_curve if stored as JSON string
    equity_curve = backtest.equity_curve
    if isinstance(equity_curve, str):
        equity_curve = json.loads(equity_curve)

    # Parse config if stored as JSON string
    config = backtest.config
    if isinstance(config, str):
        config = json.loads(config)

    return {
        "id": backtest.id,
        "backtest_type": backtest.backtest_type,
        "binding_id": backtest.binding_id,
        "prompt_id": backtest.prompt_id,
        "user_id": backtest.user_id,
        "config": config,
        "start_time": backtest.start_time.isoformat() if backtest.start_time else None,
        "end_time": backtest.end_time.isoformat() if backtest.end_time else None,
        "initial_balance": backtest.initial_balance,
        "final_equity": backtest.final_equity,
        "total_pnl": backtest.total_pnl,
        "total_pnl_percent": backtest.total_pnl_percent,
        "max_drawdown": backtest.max_drawdown,
        "max_drawdown_percent": backtest.max_drawdown_percent,
        "total_triggers": backtest.total_triggers,
        "total_trades": backtest.total_trades,
        "winning_trades": backtest.winning_trades,
        "losing_trades": backtest.losing_trades,
        "win_rate": backtest.win_rate,
        "profit_factor": backtest.profit_factor,
        "sharpe_ratio": backtest.sharpe_ratio,
        "equity_curve": equity_curve,
        "execution_time_ms": backtest.execution_time_ms,
        "status": backtest.status,
        "error_message": backtest.error_message,
        "created_at": backtest.created_at.isoformat() if backtest.created_at else None,
        "completed_at": backtest.completed_at.isoformat() if backtest.completed_at else None,
    }


@router.get("/backtest/{backtest_id}/triggers")
def get_backtest_triggers(
    backtest_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    action_filter: Optional[str] = Query(None, description="Filter by action: buy, sell, close, hold"),
    db: Session = Depends(get_db)
):
    """
    Get trigger logs for a backtest (summary list).

    Returns paginated list of trigger logs without full decision_input/output.
    Use /triggers/{trigger_id} for full details.
    """
    backtest = db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()
    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    query = db.query(BacktestTriggerLog).filter(
        BacktestTriggerLog.backtest_id == backtest_id
    )

    if action_filter:
        query = query.filter(BacktestTriggerLog.decision_action == action_filter)

    total = query.count()
    triggers = query.order_by(BacktestTriggerLog.trigger_index).offset(offset).limit(limit).all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "triggers": [
            {
                "id": t.id,
                "trigger_index": t.trigger_index,
                "trigger_type": t.trigger_type,
                "trigger_time": t.trigger_time.isoformat() if t.trigger_time else None,
                "symbol": t.symbol,
                "decision_action": t.decision_action,
                "decision_symbol": t.decision_symbol,
                "decision_side": t.decision_side,
                "decision_size": t.decision_size,
                "decision_reason": t.decision_reason,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "fee": t.fee,
                "unrealized_pnl": t.unrealized_pnl,
                "realized_pnl": t.realized_pnl,
                "equity_before": t.equity_before,
                "equity_after": t.equity_after,
                "execution_error": t.execution_error,
            }
            for t in triggers
        ]
    }


@router.get("/backtest/{backtest_id}/markers")
def get_backtest_markers(backtest_id: int, db: Session = Depends(get_db)):
    """
    Get chart markers for a backtest.

    Returns all non-HOLD triggers with minimal data for chart display.
    """
    backtest = db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()
    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    triggers = db.query(
        BacktestTriggerLog.trigger_index,
        BacktestTriggerLog.decision_action,
        BacktestTriggerLog.trigger_type
    ).filter(
        BacktestTriggerLog.backtest_id == backtest_id,
        BacktestTriggerLog.decision_action != 'hold'
    ).order_by(BacktestTriggerLog.trigger_index).all()

    return {
        "total": len(triggers),
        "markers": [
            {
                "index": t.trigger_index,
                "action": t.decision_action,
                "trigger_type": t.trigger_type
            }
            for t in triggers
        ]
    }


@router.get("/backtest/trigger/{trigger_id}")
def get_trigger_detail(trigger_id: int, db: Session = Depends(get_db)):
    """
    Get full details for a single trigger log.

    Returns complete decision_input and decision_output for debugging.
    """
    trigger = db.query(BacktestTriggerLog).filter(BacktestTriggerLog.id == trigger_id).first()
    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger log not found")

    # Parse JSON fields
    decision_input = trigger.decision_input
    if isinstance(decision_input, str):
        decision_input = json.loads(decision_input)

    decision_output = trigger.decision_output
    if isinstance(decision_output, str) and decision_output:
        decision_output = json.loads(decision_output)

    data_queries = trigger.data_queries
    if isinstance(data_queries, str) and data_queries:
        data_queries = json.loads(data_queries)

    execution_logs = trigger.execution_logs
    if isinstance(execution_logs, str) and execution_logs:
        execution_logs = json.loads(execution_logs)

    return {
        "id": trigger.id,
        "backtest_id": trigger.backtest_id,
        "trigger_index": trigger.trigger_index,
        "trigger_type": trigger.trigger_type,
        "trigger_time": trigger.trigger_time.isoformat() if trigger.trigger_time else None,
        "symbol": trigger.symbol,
        "decision_type": trigger.decision_type,
        "decision_action": trigger.decision_action,
        "decision_symbol": trigger.decision_symbol,
        "decision_side": trigger.decision_side,
        "decision_size": trigger.decision_size,
        "decision_reason": trigger.decision_reason,
        "entry_price": trigger.entry_price,
        "exit_price": trigger.exit_price,
        "pnl": trigger.pnl,
        "fee": trigger.fee,
        "unrealized_pnl": trigger.unrealized_pnl,
        "realized_pnl": trigger.realized_pnl,
        "equity_before": trigger.equity_before,
        "equity_after": trigger.equity_after,
        "decision_input": decision_input,
        "decision_output": decision_output,
        "data_queries": data_queries,
        "execution_logs": execution_logs,
        "execution_error": trigger.execution_error,
        "created_at": trigger.created_at.isoformat() if trigger.created_at else None,
    }
