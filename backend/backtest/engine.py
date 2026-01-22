"""
Program Backtest Engine

Event-driven backtest engine for Program Trader strategies.
Orchestrates trigger generation, strategy execution, and result calculation.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from .models import BacktestConfig, TriggerEvent, BacktestResult, BacktestTradeRecord
from .virtual_account import VirtualAccount
from .execution_simulator import ExecutionSimulator
from .historical_data_provider import HistoricalDataProvider

logger = logging.getLogger(__name__)

# Interval to milliseconds mapping
INTERVAL_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


class ProgramBacktestEngine:
    """
    Event-driven backtest engine for Program Trader.

    Flow:
    1. Generate trigger events (signal + scheduled)
    2. Sort events by timestamp
    3. For each event:
       - Set historical data provider time
       - Check TP/SL triggers
       - Build MarketData
       - Execute strategy code
       - Simulate order execution
       - Update virtual account
    4. Calculate statistics
    """

    def __init__(self, db: Session):
        self.db = db

    def run(self, config: BacktestConfig) -> BacktestResult:
        """
        Run backtest with given configuration.

        Args:
            config: Backtest configuration

        Returns:
            BacktestResult with statistics and trade history
        """
        start_time = time.time()

        try:
            # 1. Generate all trigger events
            triggers = self._generate_trigger_events(config)
            if not triggers:
                return BacktestResult(
                    success=False,
                    error="No trigger events generated. Check signal pools and time range."
                )

            # 2. Initialize components
            account = VirtualAccount(initial_balance=config.initial_balance)
            simulator = ExecutionSimulator(
                slippage_percent=config.slippage_percent,
                fee_rate=config.fee_rate,
            )
            data_provider = HistoricalDataProvider(
                db=self.db,
                symbols=config.symbols,
                start_time_ms=config.start_time_ms,
                end_time_ms=config.end_time_ms,
            )

            # 3. Run event loop
            trades, equity_curve = self._run_event_loop(
                config, triggers, account, simulator, data_provider
            )

            # 4. Calculate statistics
            result = self._calculate_result(
                trades=trades,
                equity_curve=equity_curve,
                triggers=triggers,
                account=account,
                config=config,
            )
            result.execution_time_ms = (time.time() - start_time) * 1000

            return result

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return BacktestResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _generate_trigger_events(self, config: BacktestConfig) -> List[TriggerEvent]:
        """Generate all trigger events (signal + scheduled)."""
        from services.signal_backtest_service import signal_backtest_service

        events = []

        # Generate signal triggers
        for pool_id in config.signal_pool_ids:
            for symbol in config.symbols:
                try:
                    pool_result = signal_backtest_service.backtest_pool(
                        self.db, pool_id, symbol,
                        config.start_time_ms, config.end_time_ms
                    )

                    if "error" in pool_result:
                        logger.warning(f"Signal backtest error for pool {pool_id}: {pool_result['error']}")
                        continue

                    for t in pool_result.get("triggers", []):
                        events.append(TriggerEvent(
                            timestamp=t["timestamp"],
                            trigger_type="signal",
                            symbol=symbol,
                            pool_id=pool_id,
                            pool_name=pool_result.get("pool_name"),
                            pool_logic=pool_result.get("logic"),
                            triggered_signals=t.get("triggered_signals", []),
                        ))
                except Exception as e:
                    logger.error(f"Failed to get signal triggers for pool {pool_id}: {e}")

        # Generate scheduled triggers
        if config.scheduled_interval and config.scheduled_interval in INTERVAL_MS:
            interval_ms = INTERVAL_MS[config.scheduled_interval]
            ts = config.start_time_ms

            while ts <= config.end_time_ms:
                events.append(TriggerEvent(
                    timestamp=ts,
                    trigger_type="scheduled",
                    symbol="",
                ))
                ts += interval_ms

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        logger.info(f"Generated {len(events)} trigger events "
                   f"({sum(1 for e in events if e.trigger_type == 'signal')} signal, "
                   f"{sum(1 for e in events if e.trigger_type == 'scheduled')} scheduled)")

        return events

    def _run_event_loop(
        self,
        config: BacktestConfig,
        triggers: List[TriggerEvent],
        account: VirtualAccount,
        simulator: ExecutionSimulator,
        data_provider: HistoricalDataProvider,
    ) -> tuple:
        """Run the main event loop."""
        from program_trader.executor import SandboxExecutor
        from program_trader.models import MarketData

        executor = SandboxExecutor(timeout_seconds=5)
        trades: List[BacktestTradeRecord] = []
        equity_curve: List[Dict[str, Any]] = []

        for trigger in triggers:
            # Set current time
            data_provider.set_current_time(trigger.timestamp)

            # Get current prices
            prices = data_provider.get_current_prices(config.symbols)
            if not prices:
                continue

            # Check TP/SL triggers first
            tp_sl_trades = simulator.check_tp_sl_triggers(account, prices, trigger.timestamp)
            trades.extend(tp_sl_trades)

            # Update equity after TP/SL
            account.update_equity(prices)

            # Determine trigger symbol
            trigger_symbol = trigger.symbol if trigger.symbol else config.symbols[0]

            # Build MarketData for strategy
            market_data = self._build_market_data(
                account, data_provider, trigger, trigger_symbol
            )

            # Execute strategy
            result = executor.execute(config.code, market_data, {})

            if result.success and result.decision:
                decision = result.decision
                symbol = decision.symbol or trigger_symbol
                current_price = prices.get(symbol, 0)

                if current_price > 0:
                    # Get signal names for logging
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

            # Update equity and record
            account.update_equity(prices)
            equity_curve.append({
                "timestamp": trigger.timestamp,
                "equity": account.equity,
                "balance": account.balance,
                "drawdown": account.max_drawdown,
            })

        return trades, equity_curve

    def _build_market_data(
        self,
        account: VirtualAccount,
        data_provider: HistoricalDataProvider,
        trigger: TriggerEvent,
        trigger_symbol: str,
    ) -> Any:
        """Build MarketData object for strategy execution."""
        from program_trader.models import MarketData, Position

        # Convert virtual positions to Position objects
        positions = {}
        for symbol, vpos in account.positions.items():
            positions[symbol] = Position(
                symbol=symbol,
                side=vpos.side,
                size=vpos.size,
                entry_price=vpos.entry_price,
                unrealized_pnl=vpos.unrealized_pnl,
                leverage=vpos.leverage,
                liquidation_price=0,
            )

        # Build triggered signals info
        triggered_signals = []
        if trigger.triggered_signals:
            for sig in trigger.triggered_signals:
                triggered_signals.append({
                    "signal_name": sig.get("signal_name", ""),
                    "metric": sig.get("metric", ""),
                    "value": sig.get("value", 0),
                })

        return MarketData(
            available_balance=account.balance,
            total_equity=account.equity,
            trigger_symbol=trigger_symbol,
            trigger_type=trigger.trigger_type,
            positions=positions,
            triggered_signals=triggered_signals,
            _data_provider=data_provider,
        )

    def _calculate_result(
        self,
        trades: List[BacktestTradeRecord],
        equity_curve: List[Dict[str, Any]],
        triggers: List[TriggerEvent],
        account: VirtualAccount,
        config: BacktestConfig,
    ) -> BacktestResult:
        """Calculate backtest statistics."""
        # Filter closed trades (with exit_price)
        closed_trades = [t for t in trades if t.exit_price is not None]

        # Basic counts
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]

        # PnL calculations
        total_pnl = sum(t.pnl for t in closed_trades)
        total_pnl_percent = (total_pnl / config.initial_balance * 100) if config.initial_balance > 0 else 0

        # Win rate
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf') if total_profit > 0 else 0

        # Average win/loss
        avg_win = (total_profit / len(winning_trades)) if winning_trades else 0
        avg_loss = (total_loss / len(losing_trades)) if losing_trades else 0

        # Largest win/loss
        largest_win = max((t.pnl for t in winning_trades), default=0)
        largest_loss = min((t.pnl for t in losing_trades), default=0)

        # Trigger counts
        signal_triggers = sum(1 for t in triggers if t.trigger_type == "signal")
        scheduled_triggers = sum(1 for t in triggers if t.trigger_type == "scheduled")

        # Sharpe ratio (simplified - annualized)
        sharpe_ratio = 0.0
        if equity_curve and len(equity_curve) > 1:
            returns = []
            for i in range(1, len(equity_curve)):
                prev_eq = equity_curve[i-1]["equity"]
                curr_eq = equity_curve[i]["equity"]
                if prev_eq > 0:
                    returns.append((curr_eq - prev_eq) / prev_eq)

            if returns:
                import statistics
                mean_return = statistics.mean(returns)
                std_return = statistics.stdev(returns) if len(returns) > 1 else 0
                if std_return > 0:
                    # Annualize (assume daily returns, 252 trading days)
                    sharpe_ratio = (mean_return / std_return) * (252 ** 0.5)

        return BacktestResult(
            success=True,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            max_drawdown=account.max_drawdown,
            max_drawdown_percent=account.max_drawdown_percent * 100,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_triggers=len(triggers),
            signal_triggers=signal_triggers,
            scheduled_triggers=scheduled_triggers,
            equity_curve=equity_curve,
            trades=trades,
            trigger_log=triggers,
            start_time=config.start_time,
            end_time=config.end_time,
        )


