"""
Backtest Data Models

Core data structures for the backtest engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    code: str                              # Strategy code
    signal_pool_ids: List[int]             # Signal pool IDs to use
    symbols: List[str]                     # Trading symbols ["BTC", "ETH"]
    start_time: Any                        # Backtest start time (datetime or str)
    end_time: Any                          # Backtest end time (datetime or str)

    # Scheduled trigger (optional)
    scheduled_interval: Optional[str] = None  # "1h", "4h", "1d", None

    # Capital and risk settings
    initial_balance: float = 10000.0
    slippage_percent: float = 0.05         # 0.05% default slippage
    fee_rate: float = 0.035                # 0.035% Hyperliquid taker fee

    # Execution assumptions
    execution_price: str = "close"         # "close", "open", "vwap"

    def __post_init__(self):
        """Convert string times to datetime if needed."""
        if isinstance(self.start_time, str):
            self.start_time = datetime.fromisoformat(self.start_time)
        if isinstance(self.end_time, str):
            self.end_time = datetime.fromisoformat(self.end_time)

    @property
    def start_time_ms(self) -> int:
        """Start time in milliseconds."""
        return int(self.start_time.timestamp() * 1000)

    @property
    def end_time_ms(self) -> int:
        """End time in milliseconds."""
        return int(self.end_time.timestamp() * 1000)


@dataclass
class TriggerEvent:
    """Unified trigger event (signal or scheduled)."""
    timestamp: int                         # Millisecond timestamp
    trigger_type: str                      # "signal" or "scheduled"
    symbol: str                            # Trigger symbol (empty for scheduled)

    # Signal trigger specific fields
    pool_id: Optional[int] = None
    pool_name: Optional[str] = None
    pool_logic: Optional[str] = None       # "AND" or "OR"
    triggered_signals: Optional[List[Dict[str, Any]]] = None
    market_regime: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.triggered_signals is None:
            self.triggered_signals = []


@dataclass
class BacktestTradeRecord:
    """Record of a single trade in backtest."""
    timestamp: int                         # Trade timestamp (ms)
    trigger_type: str                      # "signal" or "scheduled"
    symbol: str
    operation: str                         # "buy", "sell", "close"
    side: str                              # "long" or "short"
    entry_price: float
    size: float
    leverage: int = 1

    # Exit info (filled when position closed)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[int] = None
    exit_reason: Optional[str] = None      # "decision", "tp", "sl", "liquidation"

    # PnL (filled when position closed)
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fee: float = 0.0

    # Context
    reason: str = ""                       # Strategy reason
    pool_name: Optional[str] = None
    triggered_signals: Optional[List[str]] = None

    def __post_init__(self):
        if self.triggered_signals is None:
            self.triggered_signals = []


@dataclass
class BacktestResult:
    """Backtest execution result with statistics."""
    success: bool
    error: Optional[str] = None

    # Core metrics
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0             # Total profit / Total loss
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Trigger statistics
    total_triggers: int = 0
    signal_triggers: int = 0
    scheduled_triggers: int = 0

    # Detailed data
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[BacktestTradeRecord] = field(default_factory=list)
    trigger_log: List[TriggerEvent] = field(default_factory=list)

    # Execution info
    execution_time_ms: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
