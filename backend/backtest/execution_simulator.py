"""
Execution Simulator for Backtest

Simulates order execution with realistic conditions:
- Slippage calculation
- Fee calculation
- TP/SL order checking
- Position management
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .virtual_account import VirtualAccount, VirtualPosition
from .models import BacktestTradeRecord

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of order execution simulation."""
    success: bool
    executed_price: float = 0.0
    executed_size: float = 0.0
    fee: float = 0.0
    slippage: float = 0.0
    error: Optional[str] = None


class ExecutionSimulator:
    """
    Simulates order execution for backtesting.

    Handles:
    - Slippage calculation based on order side
    - Fee calculation (maker/taker)
    - TP/SL order trigger checking
    - Position opening/closing
    """

    def __init__(
        self,
        slippage_percent: float = 0.05,
        fee_rate: float = 0.035,
    ):
        """
        Initialize execution simulator.

        Args:
            slippage_percent: Default slippage in percent (0.05 = 0.05%)
            fee_rate: Trading fee rate in percent (0.035 = 0.035%)
        """
        self.slippage_percent = slippage_percent
        self.fee_rate = fee_rate

    def calculate_execution_price(
        self,
        price: float,
        side: str,
        slippage_pct: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate execution price with slippage.

        Args:
            price: Base price
            side: "buy" or "sell"
            slippage_pct: Override slippage percent

        Returns:
            (executed_price, slippage_amount)
        """
        slippage = slippage_pct if slippage_pct is not None else self.slippage_percent

        if side.lower() == "buy":
            # Buying pushes price up
            executed_price = price * (1 + slippage / 100)
        else:
            # Selling pushes price down
            executed_price = price * (1 - slippage / 100)

        slippage_amount = abs(executed_price - price)
        return executed_price, slippage_amount

    def calculate_fee(
        self,
        notional: float,
        fee_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate trading fee.

        Args:
            notional: Trade notional value (size * price)
            fee_rate: Override fee rate

        Returns:
            Fee amount
        """
        rate = fee_rate if fee_rate is not None else self.fee_rate
        return notional * rate / 100

    def check_tp_sl_triggers(
        self,
        account: VirtualAccount,
        prices: Dict[str, float],
        timestamp: int,
    ) -> List[BacktestTradeRecord]:
        """
        Check if any TP/SL orders should trigger.

        Args:
            account: Virtual account state
            prices: Current prices {symbol: price}
            timestamp: Current timestamp

        Returns:
            List of trade records for triggered orders
        """
        triggered_trades = []

        for symbol, pos in list(account.positions.items()):
            if symbol not in prices:
                continue

            current_price = prices[symbol]
            exit_reason = None
            exit_price = None

            # Check take profit
            if pos.take_profit_price:
                if pos.side == "long" and current_price >= pos.take_profit_price:
                    exit_reason = "tp"
                    exit_price = pos.take_profit_price
                elif pos.side == "short" and current_price <= pos.take_profit_price:
                    exit_reason = "tp"
                    exit_price = pos.take_profit_price

            # Check stop loss
            if not exit_reason and pos.stop_loss_price:
                if pos.side == "long" and current_price <= pos.stop_loss_price:
                    exit_reason = "sl"
                    exit_price = pos.stop_loss_price
                elif pos.side == "short" and current_price >= pos.stop_loss_price:
                    exit_reason = "sl"
                    exit_price = pos.stop_loss_price

            if exit_reason and exit_price:
                # Apply slippage to exit
                close_side = "sell" if pos.side == "long" else "buy"
                executed_price, _ = self.calculate_execution_price(exit_price, close_side)

                # Calculate fee
                notional = pos.size * executed_price
                fee = self.calculate_fee(notional)

                # Close position
                pnl = account.close_position(symbol, executed_price, fee)

                # Calculate PnL percent
                entry_notional = pos.size * pos.entry_price
                pnl_percent = (pnl / entry_notional * 100) if entry_notional > 0 else 0

                trade = BacktestTradeRecord(
                    timestamp=pos.entry_timestamp,
                    trigger_type="",
                    symbol=symbol,
                    operation="close",
                    side=pos.side,
                    entry_price=pos.entry_price,
                    size=pos.size,
                    leverage=pos.leverage,
                    exit_price=executed_price,
                    exit_timestamp=timestamp,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    fee=fee,
                    reason=f"{'Take Profit' if exit_reason == 'tp' else 'Stop Loss'} triggered",
                )
                triggered_trades.append(trade)

        return triggered_trades

    def execute_decision(
        self,
        decision: Any,
        account: VirtualAccount,
        current_price: float,
        timestamp: int,
        trigger_type: str = "",
        pool_name: Optional[str] = None,
        triggered_signals: Optional[List[str]] = None,
    ) -> Optional[BacktestTradeRecord]:
        """
        Execute a trading decision.

        Args:
            decision: Decision object from strategy
            account: Virtual account state
            current_price: Current market price
            timestamp: Current timestamp
            trigger_type: "signal" or "scheduled"
            pool_name: Signal pool name (if signal trigger)
            triggered_signals: List of triggered signal names

        Returns:
            Trade record if trade executed, None otherwise
        """
        operation = decision.operation.lower() if decision.operation else "hold"

        if operation == "hold":
            return None

        symbol = decision.symbol
        has_position = account.has_position(symbol)

        # Handle close operation
        if operation == "close":
            if not has_position:
                return None
            return self._execute_close(
                account, symbol, current_price, timestamp,
                trigger_type, pool_name, triggered_signals, decision.reason
            )

        # Handle buy/sell operations
        if operation in ("buy", "sell"):
            if has_position:
                pos = account.get_position(symbol)
                # If same direction, add to position (averaging)
                if (operation == "buy" and pos.side == "long") or \
                   (operation == "sell" and pos.side == "short"):
                    return self._execute_add_position(
                        account, decision, current_price, timestamp,
                        trigger_type, pool_name, triggered_signals
                    )
                # If opposite direction, close existing first
                self._execute_close(
                    account, symbol, current_price, timestamp,
                    trigger_type, pool_name, triggered_signals, "Reverse position"
                )

            return self._execute_open(
                account, decision, current_price, timestamp,
                trigger_type, pool_name, triggered_signals
            )

        return None

    def _execute_add_position(
        self,
        account: VirtualAccount,
        decision: Any,
        current_price: float,
        timestamp: int,
        trigger_type: str,
        pool_name: Optional[str],
        triggered_signals: Optional[List[str]],
    ) -> Optional[BacktestTradeRecord]:
        """Execute adding to existing position (averaging entry price)."""
        operation = decision.operation.lower()
        symbol = decision.symbol
        side = "long" if operation == "buy" else "short"

        # Calculate execution price with slippage
        exec_price, slippage = self.calculate_execution_price(current_price, operation)

        # Calculate additional position size
        portion = getattr(decision, 'target_portion_of_balance', 0.5)
        leverage = getattr(decision, 'leverage', 1)
        available = account.balance * portion
        add_size = (available * leverage) / exec_price

        if add_size <= 0:
            return None

        # Calculate fee
        notional = add_size * exec_price
        fee = self.calculate_fee(notional)

        # Get TP/SL prices (will override existing)
        tp_price = getattr(decision, 'take_profit_price', None)
        sl_price = getattr(decision, 'stop_loss_price', None)

        # Get position info before adding
        pos = account.get_position(symbol)
        old_size = pos.size
        old_entry = pos.entry_price

        # Add to position
        account.add_to_position(
            symbol=symbol,
            size=add_size,
            entry_price=exec_price,
            fee=fee,
            take_profit=tp_price,
            stop_loss=sl_price,
        )

        # Get updated position info
        updated_pos = account.get_position(symbol)

        return BacktestTradeRecord(
            timestamp=timestamp,
            trigger_type=trigger_type,
            symbol=symbol,
            operation="add_position",
            side=side,
            entry_price=exec_price,
            size=add_size,
            leverage=leverage,
            fee=fee,
            reason=getattr(decision, 'reason', '') + f" (Added to position, avg entry: {updated_pos.entry_price:.2f})",
            pool_name=pool_name,
            triggered_signals=triggered_signals or [],
        )

    def _execute_open(
        self,
        account: VirtualAccount,
        decision: Any,
        current_price: float,
        timestamp: int,
        trigger_type: str,
        pool_name: Optional[str],
        triggered_signals: Optional[List[str]],
    ) -> Optional[BacktestTradeRecord]:
        """Execute position open."""
        operation = decision.operation.lower()
        symbol = decision.symbol
        side = "long" if operation == "buy" else "short"

        # Calculate execution price with slippage
        exec_price, slippage = self.calculate_execution_price(current_price, operation)

        # Calculate position size
        portion = getattr(decision, 'target_portion_of_balance', 0.5)
        leverage = getattr(decision, 'leverage', 1)
        available = account.balance * portion
        size = (available * leverage) / exec_price

        if size <= 0:
            return None

        # Calculate entry fee
        notional = size * exec_price
        fee = self.calculate_fee(notional)

        # Get TP/SL prices
        tp_price = getattr(decision, 'take_profit_price', None)
        sl_price = getattr(decision, 'stop_loss_price', None)

        # Open position (fee is tracked inside open_position)
        account.open_position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=exec_price,
            leverage=leverage,
            timestamp=timestamp,
            take_profit=tp_price,
            stop_loss=sl_price,
            fee=fee,
        )

        return BacktestTradeRecord(
            timestamp=timestamp,
            trigger_type=trigger_type,
            symbol=symbol,
            operation=operation,
            side=side,
            entry_price=exec_price,
            size=size,
            leverage=leverage,
            fee=fee,
            reason=getattr(decision, 'reason', ''),
            pool_name=pool_name,
            triggered_signals=triggered_signals or [],
        )

    def _execute_close(
        self,
        account: VirtualAccount,
        symbol: str,
        current_price: float,
        timestamp: int,
        trigger_type: str,
        pool_name: Optional[str],
        triggered_signals: Optional[List[str]],
        reason: str = "",
    ) -> Optional[BacktestTradeRecord]:
        """Execute position close."""
        pos = account.get_position(symbol)
        if not pos:
            return None

        # Calculate execution price with slippage
        close_side = "sell" if pos.side == "long" else "buy"
        exec_price, _ = self.calculate_execution_price(current_price, close_side)

        # Calculate fee
        notional = pos.size * exec_price
        fee = self.calculate_fee(notional)

        # Store position info before closing
        entry_price = pos.entry_price
        size = pos.size
        leverage = pos.leverage
        side = pos.side
        entry_ts = pos.entry_timestamp

        # Close position
        pnl = account.close_position(symbol, exec_price, fee)

        # Calculate PnL percent
        entry_notional = size * entry_price
        pnl_percent = (pnl / entry_notional * 100) if entry_notional > 0 else 0

        return BacktestTradeRecord(
            timestamp=entry_ts,
            trigger_type=trigger_type,
            symbol=symbol,
            operation="close",
            side=side,
            entry_price=entry_price,
            size=size,
            leverage=leverage,
            exit_price=exec_price,
            exit_timestamp=timestamp,
            exit_reason="decision",
            pnl=pnl,
            pnl_percent=pnl_percent,
            fee=fee,
            reason=reason,
            pool_name=pool_name,
            triggered_signals=triggered_signals or [],
        )

