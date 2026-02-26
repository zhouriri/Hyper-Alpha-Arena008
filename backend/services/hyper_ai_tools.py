"""
Hyper AI Tools - Tools for Hyper AI main agent

Provides tools for:
- System overview and diagnostics
- Wallet status queries
- API reference documentation
- Market data (klines, regime, flow)
- Create/save operations (signal pool, prompt, program, AI trader)
- Sub-agent calls (Prompt AI, Program AI, Signal AI, Attribution AI)
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import text, func

from services.hyper_ai_subagents import SUBAGENT_TOOLS, execute_subagent_tool

logger = logging.getLogger(__name__)


# Tool definitions in OpenAI format
# IMPORTANT: When adding/removing/modifying tools, you MUST also update:
#   1. The JSON schema definition below (this list)
#   2. The execute_xxx() implementation function
#   3. The execute_hyper_ai_tool() dispatcher at the bottom of this file
#   4. The system prompt: backend/config/hyper_ai_system_prompt.md (Available Tools section)
HYPER_AI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_system_overview",
            "description": "Get high-level system status: wallets, AI traders, strategies, signal pools, positions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_wallet_status",
            "description": "Get wallet balance and position summary (read-only, no credentials exposed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "exchange": {
                        "type": "string",
                        "enum": ["hyperliquid", "binance", "all"],
                        "description": "Filter by exchange (default: all)"
                    },
                    "environment": {
                        "type": "string",
                        "enum": ["testnet", "mainnet", "all"],
                        "description": "Filter by environment (default: all)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_api_reference",
            "description": "Get API reference docs for Prompt variables or Program MarketData/Decision APIs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_type": {
                        "type": "string",
                        "enum": ["prompt", "program"],
                        "description": "Document type: prompt (variables) or program (MarketData/Decision API)"
                    },
                    "api_type": {
                        "type": "string",
                        "enum": ["market", "decision", "all"],
                        "description": "For program only: which API docs (default: all)"
                    },
                    "lang": {
                        "type": "string",
                        "enum": ["en", "zh"],
                        "description": "Language (default: en)"
                    }
                },
                "required": ["doc_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_klines",
            "description": "Get K-line/candlestick data for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading symbol (e.g., BTC, ETH)"},
                    "period": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h", "1d"], "description": "K-line period (default: 1h)"},
                    "limit": {"type": "integer", "description": "Number of candles (default: 50, max: 200)"},
                    "exchange": {"type": "string", "enum": ["hyperliquid", "binance"], "description": "Exchange (default: hyperliquid)"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_regime",
            "description": "Get current market regime classification for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading symbol"},
                    "period": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h"], "description": "Time period (default: 1h)"},
                    "exchange": {"type": "string", "enum": ["hyperliquid", "binance"], "description": "Exchange (default: hyperliquid)"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_flow",
            "description": "Get market flow data (CVD, OI, Funding, etc.) for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading symbol"},
                    "period": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h"], "description": "Time period (default: 1h)"},
                    "exchange": {"type": "string", "enum": ["hyperliquid", "binance"], "description": "Exchange (default: hyperliquid)"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_logs",
            "description": "Get recent system error/warning logs for troubleshooting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["error", "warning", "all"], "description": "Log level filter (default: error)"},
                    "limit": {"type": "integer", "description": "Max entries (default: 20, max: 50)"},
                    "trader_id": {"type": "integer", "description": "Filter by AI Trader ID"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_contact_config",
            "description": "Get support channel URLs (Twitter, Telegram, GitHub).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_trader_issues",
            "description": "Check why an AI Trader is not triggering and provide actionable suggestions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "integer", "description": "AI Trader ID to diagnose"}
                },
                "required": ["trader_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_signal_pool",
            "description": "Create a signal pool from complete signal configuration. Automatically creates signal definitions and combines them into a pool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pool_name": {"type": "string", "description": "Display name for the pool"},
                    "symbol": {"type": "string", "description": "Symbol to monitor (e.g., BTC, ETH)"},
                    "signals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "metric": {"type": "string", "description": "Metric name (e.g., cvd, oi_delta_percent, order_imbalance, taker_volume)"},
                                "operator": {"type": "string", "description": "Comparison operator (greater_than, less_than, etc.). NOT used for taker_volume."},
                                "threshold": {"type": "number", "description": "Threshold value. NOT used for taker_volume."},
                                "time_window": {"type": "string", "description": "Time window (e.g., 5m, 15m, 1h)"},
                                "direction": {"type": "string", "enum": ["buy", "sell", "any"], "description": "taker_volume ONLY: dominant side"},
                                "ratio_threshold": {"type": "number", "description": "taker_volume ONLY: buy/sell ratio multiplier (e.g., 1.5 = 50% more)"},
                                "volume_threshold": {"type": "number", "description": "taker_volume ONLY: minimum total volume in USD"}
                            }
                        },
                        "description": "Array of signal conditions. Standard signals use metric/operator/threshold/time_window. taker_volume uses metric/direction/ratio_threshold/volume_threshold/time_window instead."
                    },
                    "logic": {"type": "string", "enum": ["AND", "OR"], "description": "Logic operator (default: AND)"},
                    "exchange": {"type": "string", "enum": ["hyperliquid", "binance"], "description": "Exchange (default: hyperliquid)"},
                    "description": {"type": "string", "description": "Optional description for the pool"}
                },
                "required": ["pool_name", "symbol", "signals"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_prompt",
            "description": "Create or update a trading prompt template.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt_id": {"type": "integer", "description": "Prompt ID to update (omit for create)"},
                    "name": {"type": "string", "description": "Display name"},
                    "description": {"type": "string", "description": "Brief description"},
                    "template_text": {"type": "string", "description": "Main prompt content"}
                },
                "required": ["name", "template_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_program",
            "description": "Create or update a trading program.",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_id": {"type": "integer", "description": "Program ID to update (omit for create)"},
                    "name": {"type": "string", "description": "Display name"},
                    "description": {"type": "string", "description": "Brief description"},
                    "code": {"type": "string", "description": "Python strategy code"}
                },
                "required": ["name", "code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_ai_trader",
            "description": "Create a new AI Trader with LLM config. Tests LLM connection before saving. Strategy binding and wallet setup are done separately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Display name for the trader"},
                    "model": {"type": "string", "description": "LLM model name (e.g., gpt-4o, deepseek-chat, claude-3.5-sonnet)"},
                    "base_url": {"type": "string", "description": "LLM API base URL (e.g., https://api.openai.com/v1)"},
                    "api_key": {"type": "string", "description": "LLM API key"}
                },
                "required": ["name", "model", "base_url", "api_key"]
            }
        }
    },
    # --- Query Tools: list resources ---
    {
        "type": "function",
        "function": {
            "name": "list_traders",
            "description": "List all AI Traders with bindings, strategies, wallet and trading status. Pass trader_id to get one trader's full detail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "integer", "description": "Optional: specific AI Trader ID for detail view"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_signal_pools",
            "description": "List all signal pools with IDs, symbols, exchange, and trigger conditions. Pass pool_id to get one pool's full detail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pool_id": {"type": "integer", "description": "Optional: specific signal pool ID for detail view"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_strategies",
            "description": "List all trading prompts and programs with IDs, names, and binding status. Pass strategy_id + strategy_type to get full content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_id": {"type": "integer", "description": "Optional: specific strategy ID for detail view"},
                    "strategy_type": {"type": "string", "enum": ["prompt", "program"], "description": "Required when strategy_id is provided"}
                },
                "required": []
            }
        }
    },
    # --- Binding Tools: assemble components ---
    {
        "type": "function",
        "function": {
            "name": "bind_prompt_to_trader",
            "description": "Bind a prompt template to an AI Trader (one-to-one, replaces existing binding).",
            "parameters": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "integer", "description": "AI Trader ID"},
                    "prompt_id": {"type": "integer", "description": "Prompt template ID to bind"}
                },
                "required": ["trader_id", "prompt_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bind_program_to_trader",
            "description": "Create a program binding for an AI Trader with trigger config (many-to-many).",
            "parameters": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "integer", "description": "AI Trader ID"},
                    "program_id": {"type": "integer", "description": "Trading program ID"},
                    "signal_pool_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Signal pool IDs for triggering"
                    },
                    "trigger_interval": {"type": "integer", "description": "Scheduled trigger interval in seconds (default: 300)"},
                    "is_active": {"type": "boolean", "description": "Whether binding is active (default: true)"}
                },
                "required": ["trader_id", "program_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_trader_strategy",
            "description": "Update trigger configuration for a Prompt-based AI Trader (signal pools, scheduled trigger, interval).",
            "parameters": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "integer", "description": "AI Trader ID"},
                    "signal_pool_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Signal pool IDs to bind"
                    },
                    "scheduled_trigger_enabled": {"type": "boolean", "description": "Enable scheduled trigger"},
                    "trigger_interval": {"type": "integer", "description": "Trigger interval in seconds"}
                },
                "required": ["trader_id"]
            }
        }
    },
    # --- Update Tools ---
    {
        "type": "function",
        "function": {
            "name": "update_ai_trader",
            "description": "Update AI Trader settings (name, LLM config). Tests LLM connection if model/base_url/api_key changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "integer", "description": "AI Trader ID"},
                    "name": {"type": "string", "description": "New display name"},
                    "model": {"type": "string", "description": "New LLM model name"},
                    "base_url": {"type": "string", "description": "New LLM API base URL"},
                    "api_key": {"type": "string", "description": "New LLM API key"}
                },
                "required": ["trader_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_program_binding",
            "description": "Update a program binding's configuration (signal pools, trigger interval, activation, params).",
            "parameters": {
                "type": "object",
                "properties": {
                    "binding_id": {"type": "integer", "description": "Program binding ID"},
                    "signal_pool_ids": {"type": "array", "items": {"type": "integer"}, "description": "New signal pool IDs"},
                    "trigger_interval": {"type": "integer", "description": "New trigger interval in seconds"},
                    "scheduled_trigger_enabled": {"type": "boolean", "description": "Enable/disable scheduled trigger"},
                    "is_active": {"type": "boolean", "description": "Activate or deactivate the binding"},
                    "params_override": {"type": "object", "description": "Parameter overrides for the program"}
                },
                "required": ["binding_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_signal_pool",
            "description": "Update signal pool settings (name, enabled, logic, signal_ids). Signal IDs must belong to the same exchange as the pool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pool_id": {"type": "integer", "description": "Signal pool ID"},
                    "pool_name": {"type": "string", "description": "New display name"},
                    "enabled": {"type": "boolean", "description": "Enable or disable the pool"},
                    "logic": {"type": "string", "enum": ["AND", "OR"], "description": "Logic operator"},
                    "signal_ids": {"type": "array", "items": {"type": "integer"}, "description": "Replace signal definitions in this pool. All signals must match the pool's exchange."}
                },
                "required": ["pool_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_prompt_binding",
            "description": "Update which prompt template is bound to an AI Trader. Replaces the current binding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "integer", "description": "AI Trader ID"},
                    "prompt_id": {"type": "integer", "description": "New prompt template ID to bind"}
                },
                "required": ["trader_id", "prompt_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save or update long-term memory with intelligent deduplication. The system automatically compares against existing memories and decides to ADD, UPDATE (merge/replace), or SKIP. To update an existing memory, just call this with the corrected content — the old version will be replaced automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["preference", "decision", "lesson", "insight", "context"],
                        "description": "Memory category: preference (trading style/risk), decision (config changes), lesson (from wins/losses), insight (market patterns), context (general)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Concise, self-contained memory content. Should be understandable without conversation context."
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score 0.0-1.0. Default 0.5. Use 0.7+ for key lessons/preferences, 0.3 for minor context."
                    }
                },
                "required": ["category", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_trader",
            "description": "Soft-delete an AI Trader. Checks for bindings and open positions first. Returns dependency list if blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "integer", "description": "AI Trader ID to delete"}
                },
                "required": ["trader_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_prompt_template",
            "description": "Soft-delete a Prompt Template. Checks for active bindings first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt_id": {"type": "integer", "description": "Prompt Template ID to delete"}
                },
                "required": ["prompt_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_signal_definition",
            "description": "Soft-delete a Signal Definition. Checks for signal pool references first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_id": {"type": "integer", "description": "Signal Definition ID to delete"}
                },
                "required": ["signal_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_signal_pool",
            "description": "Soft-delete a Signal Pool. Checks for strategy and program binding references first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pool_id": {"type": "integer", "description": "Signal Pool ID to delete"}
                },
                "required": ["pool_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_trading_program",
            "description": "Soft-delete a Trading Program. Checks for active bindings first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_id": {"type": "integer", "description": "Trading Program ID to delete"}
                },
                "required": ["program_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_prompt_binding",
            "description": "Soft-delete a Prompt Binding (unbind prompt from trader).",
            "parameters": {
                "type": "object",
                "properties": {
                    "binding_id": {"type": "integer", "description": "Prompt Binding ID to delete"}
                },
                "required": ["binding_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_program_binding",
            "description": "Soft-delete a Program Binding. Must be deactivated (is_active=false) first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "binding_id": {"type": "integer", "description": "Program Binding ID to delete"}
                },
                "required": ["binding_id"]
            }
        }
    }
]

# --- Skill System Tools ---
# These tools load workflow guidance into the AI's context (Level 2 & 3 loading).
# They do NOT perform any actions — they provide step-by-step instructions
# for the AI to follow when executing complex multi-step tasks.
# See backend/skills/*/SKILL.md for skill definitions.
SKILL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load a skill workflow guide into your context. This does NOT perform any action — it provides you with step-by-step instructions for a specific task type. Use this when a user's request matches one of your available skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to load (e.g., 'prompt-strategy-setup', 'trader-diagnosis')"
                    }
                },
                "required": ["skill_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill_reference",
            "description": "Load a reference document from a skill's references/ directory. Use this when a loaded skill mentions additional reference materials you should consult.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill"
                    },
                    "reference_file": {
                        "type": "string",
                        "description": "Filename of the reference document (e.g., 'signal-design-guide.md')"
                    }
                },
                "required": ["skill_name", "reference_file"]
            }
        }
    }
]

# Combine base tools + skill tools + sub-agent tools
HYPER_AI_TOOLS = HYPER_AI_TOOLS + SKILL_TOOLS + SUBAGENT_TOOLS


# =============================================================================
# Tool Execution Functions
# =============================================================================

def execute_get_system_overview(db: Session) -> str:
    """Get high-level system status summary."""
    from database.models import (
        Account, HyperliquidWallet, BinanceWallet, PromptTemplate,
        TradingProgram, SignalPool, AccountPromptBinding, AccountProgramBinding,
        HyperliquidPosition
    )

    try:
        result = {
            "wallets": {"hyperliquid": {}, "binance": {}},
            "ai_traders": {"total": 0, "active": 0, "using_prompt": 0, "using_program": 0},
            "strategies": {"prompts": 0, "programs": 0},
            "signal_pools": {"hyperliquid": 0, "binance": 0},
            "open_positions": {}
        }

        # Count wallets by exchange and environment
        hl_wallets = db.query(
            HyperliquidWallet.environment, func.count(HyperliquidWallet.id)
        ).filter(HyperliquidWallet.is_active == "true").group_by(HyperliquidWallet.environment).all()
        for env, count in hl_wallets:
            result["wallets"]["hyperliquid"][env] = count

        bn_wallets = db.query(
            BinanceWallet.environment, func.count(BinanceWallet.id)
        ).filter(BinanceWallet.is_active == "true").group_by(BinanceWallet.environment).all()
        for env, count in bn_wallets:
            result["wallets"]["binance"][env] = count

        # Count AI Traders
        total_traders = db.query(Account).filter(Account.is_active == "true", Account.is_deleted != True).count()
        active_traders = db.query(Account).filter(
            Account.is_active == "true",
            Account.auto_trading_enabled == "true",
            Account.is_deleted != True
        ).count()
        result["ai_traders"]["total"] = total_traders
        result["ai_traders"]["active"] = active_traders

        # Count by strategy type
        prompt_bindings = db.query(AccountPromptBinding).filter(AccountPromptBinding.is_deleted != True).count()
        program_bindings = db.query(AccountProgramBinding).filter(
            AccountProgramBinding.is_active == True,
            AccountProgramBinding.is_deleted != True
        ).count()
        result["ai_traders"]["using_prompt"] = prompt_bindings
        result["ai_traders"]["using_program"] = program_bindings

        # Count strategies
        user_prompts = db.query(PromptTemplate).filter(
            PromptTemplate.is_system == "false",
            PromptTemplate.is_deleted == "false"
        ).count()
        programs = db.query(TradingProgram).filter(TradingProgram.is_deleted != True).count()
        result["strategies"]["prompts"] = user_prompts
        result["strategies"]["programs"] = programs

        # Count signal pools by exchange
        pools = db.query(
            SignalPool.exchange, func.count(SignalPool.id)
        ).filter(SignalPool.enabled == True, SignalPool.is_deleted != True).group_by(SignalPool.exchange).all()
        for exchange, count in pools:
            result["signal_pools"][exchange or "hyperliquid"] = count

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"[get_system_overview] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_get_wallet_status(db: Session, exchange: str = "all", environment: str = "all") -> str:
    """Get wallet balance and position summary using real-time API (same as frontend)."""
    from database.models import HyperliquidWallet, BinanceWallet, Account
    from services.hyperliquid_trading_client import get_cached_trading_client
    from services.binance_trading_client import BinanceTradingClient
    from utils.encryption import decrypt_private_key

    try:
        wallets = []

        # Query Hyperliquid wallets - use real-time API
        if exchange in ["all", "hyperliquid"]:
            hl_query = db.query(HyperliquidWallet, Account).join(
                Account, HyperliquidWallet.account_id == Account.id
            ).filter(HyperliquidWallet.is_active == "true")

            if environment != "all":
                hl_query = hl_query.filter(HyperliquidWallet.environment == environment)

            for wallet, account in hl_query.all():
                try:
                    # Decrypt private key and use cached client
                    private_key = decrypt_private_key(wallet.private_key_encrypted)
                    client = get_cached_trading_client(
                        account_id=account.id,
                        private_key=private_key,
                        environment=wallet.environment,
                        wallet_address=wallet.wallet_address
                    )
                    account_state = client.get_account_state(db)

                    wallet_info = {
                        "exchange": "hyperliquid",
                        "environment": wallet.environment,
                        "wallet_address": wallet.wallet_address[:10] + "..." + wallet.wallet_address[-6:],
                        "trader_id": account.id,
                        "trader_name": account.name,
                        "balance": {
                            "total_equity": float(account_state.get("total_equity", 0)),
                            "available_balance": float(account_state.get("available_balance", 0)),
                            "used_margin": float(account_state.get("used_margin", 0))
                        },
                        "positions": [],
                        "last_updated": "real-time"
                    }

                    # Get positions from API response
                    for pos in account_state.get("positions", []):
                        szi = float(pos.get("szi", 0) or 0)
                        if szi != 0:
                            wallet_info["positions"].append({
                                "symbol": pos.get("coin", ""),
                                "size": abs(szi),
                                "side": "long" if szi > 0 else "short",
                                "unrealized_pnl": float(pos.get("unrealized_pnl", 0) or 0)
                            })

                    wallets.append(wallet_info)
                except Exception as e:
                    logger.warning(f"[get_wallet_status] Failed to get Hyperliquid data for {account.name}: {e}")
                    wallets.append({
                        "exchange": "hyperliquid",
                        "environment": wallet.environment,
                        "wallet_address": wallet.wallet_address[:10] + "..." + wallet.wallet_address[-6:],
                        "trader_id": account.id,
                        "trader_name": account.name,
                        "balance": {"total_equity": 0, "available_balance": 0, "used_margin": 0},
                        "positions": [],
                        "error": str(e)
                    })

        # Query Binance wallets - use real-time API
        if exchange in ["all", "binance"]:
            bn_query = db.query(BinanceWallet, Account).join(
                Account, BinanceWallet.account_id == Account.id
            ).filter(BinanceWallet.is_active == "true")

            if environment != "all":
                bn_query = bn_query.filter(BinanceWallet.environment == environment)

            for wallet, account in bn_query.all():
                try:
                    # Decrypt API keys
                    api_key = decrypt_private_key(wallet.api_key_encrypted)
                    secret_key = decrypt_private_key(wallet.secret_key_encrypted)
                    client = BinanceTradingClient(api_key, secret_key, wallet.environment)
                    balance = client.get_balance()

                    wallet_info = {
                        "exchange": "binance",
                        "environment": wallet.environment,
                        "trader_id": account.id,
                        "trader_name": account.name,
                        "balance": {
                            "total_equity": float(balance.get("total_equity", 0)),
                            "available_balance": float(balance.get("available_balance", 0)),
                            "unrealized_pnl": float(balance.get("unrealized_pnl", 0))
                        },
                        "positions": [],
                        "last_updated": "real-time"
                    }
                    wallets.append(wallet_info)
                except Exception as e:
                    logger.warning(f"[get_wallet_status] Failed to get Binance data for {account.name}: {e}")
                    wallets.append({
                        "exchange": "binance",
                        "environment": wallet.environment,
                        "trader_id": account.id,
                        "trader_name": account.name,
                        "balance": {"total_equity": 0, "available_balance": 0, "unrealized_pnl": 0},
                        "positions": [],
                        "error": str(e)
                    })

        return json.dumps({"wallets": wallets}, indent=2)

    except Exception as e:
        logger.error(f"[get_wallet_status] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_get_api_reference(doc_type: str, api_type: str = "all", lang: str = "en") -> str:
    """Get API reference documentation."""
    try:
        if doc_type == "prompt":
            # Read prompt variables reference document
            filename = "PROMPT_VARIABLES_REFERENCE_ZH.md" if lang == "zh" else "PROMPT_VARIABLES_REFERENCE.md"
            doc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", filename)

            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return json.dumps({"doc_type": "prompt", "lang": lang, "content": content})
            except FileNotFoundError:
                return json.dumps({"error": f"Document not found: {filename}"})

        elif doc_type == "program":
            # Return MarketData/Decision API docs from ai_program_service
            from services.ai_program_service import MARKET_API_DOCS, DECISION_API_DOCS

            if api_type == "market":
                content = MARKET_API_DOCS
            elif api_type == "decision":
                content = DECISION_API_DOCS
            else:
                content = MARKET_API_DOCS + "\n\n" + DECISION_API_DOCS

            return json.dumps({"doc_type": "program", "api_type": api_type, "content": content})

        else:
            return json.dumps({"error": f"Invalid doc_type: {doc_type}"})

    except Exception as e:
        logger.error(f"[get_api_reference] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_get_klines(db: Session, symbol: str, period: str = "1h", limit: int = 50, exchange: str = "hyperliquid") -> str:
    """Get K-line data for a symbol."""
    from database.models import CryptoKline

    try:
        limit = min(max(limit, 1), 200)

        klines = db.query(CryptoKline).filter(
            CryptoKline.exchange == exchange,
            CryptoKline.symbol == symbol.upper(),
            CryptoKline.period == period,
            CryptoKline.environment == "mainnet"
        ).order_by(CryptoKline.timestamp.desc()).limit(limit).all()

        candles = []
        for k in reversed(klines):
            candles.append({
                "time": datetime.utcfromtimestamp(k.timestamp).strftime("%Y-%m-%d %H:%M UTC"),
                "open": float(k.open_price) if k.open_price else 0,
                "high": float(k.high_price) if k.high_price else 0,
                "low": float(k.low_price) if k.low_price else 0,
                "close": float(k.close_price) if k.close_price else 0,
                "volume": float(k.volume) if k.volume else 0
            })

        return json.dumps({
            "symbol": symbol.upper(),
            "period": period,
            "exchange": exchange,
            "candles": candles,
            "count": len(candles)
        }, indent=2)

    except Exception as e:
        logger.error(f"[get_klines] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_get_market_regime(db: Session, symbol: str, period: str = "1h", exchange: str = "hyperliquid") -> str:
    """Get market regime classification for a symbol."""
    try:
        from program_trader.data_provider import DataProvider

        data_provider = DataProvider(db=db, account_id=0, environment="mainnet", exchange=exchange)
        regime = data_provider.get_regime(symbol.upper(), period)

        if regime:
            return json.dumps({
                "symbol": symbol.upper(),
                "period": period,
                "exchange": exchange,
                "regime": regime.regime,
                "confidence": regime.conf
            }, indent=2)
        else:
            return json.dumps({
                "symbol": symbol.upper(),
                "period": period,
                "exchange": exchange,
                "regime": "unknown",
                "confidence": 0,
                "note": "Unable to determine market regime"
            })

    except Exception as e:
        logger.error(f"[get_market_regime] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_get_market_flow(db: Session, symbol: str, period: str = "1h", exchange: str = "hyperliquid") -> str:
    """Get market flow data for a symbol."""
    try:
        from program_trader.data_provider import DataProvider

        data_provider = DataProvider(db=db, account_id=0, environment="mainnet", exchange=exchange)

        flow = {}
        for metric in ["CVD", "OI", "OI_DELTA", "TAKER", "FUNDING"]:
            result = data_provider.get_flow(symbol.upper(), metric, period)
            if result:
                flow[metric] = result

        return json.dumps({
            "symbol": symbol.upper(),
            "period": period,
            "exchange": exchange,
            "flow": flow
        }, indent=2)

    except Exception as e:
        logger.error(f"[get_market_flow] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_get_system_logs(db: Session, level: str = "error", limit: int = 20, trader_id: int = None) -> str:
    """Get recent system logs (same as System Logs page)."""
    from services.system_logger import system_logger

    try:
        limit = min(max(limit, 1), 50)

        # Map level to min_level for system_logger
        min_level = None
        if level == "error":
            min_level = "ERROR"
        elif level == "warning":
            min_level = "WARNING"
        # "all" means no filter

        # Get logs from system_logger (same as frontend System Logs page)
        raw_logs = system_logger.get_logs(limit=limit, min_level=min_level)

        # Format for AI consumption
        logs = []
        for log in raw_logs:
            logs.append({
                "time": log.get("timestamp", ""),
                "level": log.get("level", "INFO"),
                "category": log.get("category", ""),
                "message": log.get("message", ""),
                "details": log.get("details", {})
            })

        return json.dumps({"logs": logs, "total": len(logs)}, indent=2)

    except Exception as e:
        logger.error(f"[get_system_logs] Error: {e}")
        return json.dumps({"error": str(e)})
        return json.dumps({"error": str(e)})


def execute_get_contact_config() -> str:
    """Get support channel URLs."""
    import requests

    try:
        # Try to fetch from external API
        resp = requests.get("https://www.akooi.com/api/config/contact", timeout=5)
        if resp.status_code == 200:
            return json.dumps(resp.json(), indent=2)
    except Exception as e:
        logger.warning(f"[get_contact_config] Failed to fetch from API: {e}")

    # Fallback to defaults
    return json.dumps({
        "twitter": {"url": "https://x.com/GptHammer3309", "enabled": True},
        "telegram": {"url": "https://t.me/+RqxjT7Gttm9hOGEx", "enabled": True},
        "github": {"url": "https://github.com/HammerGPT/Hyper-Alpha-Arena", "enabled": True}
    }, indent=2)


def execute_diagnose_trader_issues(db: Session, trader_id: int) -> str:
    """Diagnose why an AI Trader is not triggering."""
    from database.models import (
        Account, HyperliquidWallet, BinanceWallet, AccountPromptBinding,
        AccountProgramBinding, AccountStrategyConfig, SignalPool,
        HyperliquidAccountSnapshot, BinanceAccountSnapshot, AIDecisionLog, ProgramExecutionLog
    )

    try:
        # Get account
        account = db.query(Account).filter(Account.id == trader_id, Account.is_deleted != True).first()
        if not account:
            return json.dumps({"error": f"AI Trader with id {trader_id} not found"})

        checks = []
        issues = []

        # Check 1: Trader enabled
        trader_enabled = account.is_active == "true"
        checks.append({"check": "trader_enabled", "passed": trader_enabled})
        if not trader_enabled:
            issues.append("AI Trader is disabled")

        # Check 2: Auto trading enabled
        auto_enabled = account.auto_trading_enabled == "true"
        checks.append({"check": "auto_trading_enabled", "passed": auto_enabled})
        if not auto_enabled:
            issues.append("Auto trading is disabled")

        # Check 3: Strategy bound
        prompt_binding = db.query(AccountPromptBinding).filter(
            AccountPromptBinding.account_id == trader_id,
            AccountPromptBinding.is_deleted != True
        ).first()
        program_binding = db.query(AccountProgramBinding).filter(
            AccountProgramBinding.account_id == trader_id,
            AccountProgramBinding.is_active == True,
            AccountProgramBinding.is_deleted != True
        ).first()

        strategy_bound = prompt_binding is not None or program_binding is not None
        strategy_type = "prompt" if prompt_binding else ("program" if program_binding else None)
        checks.append({"check": "strategy_bound", "passed": strategy_bound, "type": strategy_type})
        if not strategy_bound:
            issues.append("No strategy (prompt or program) bound to this trader")

        # Check 4: Wallet bound and has balance
        strategy_config = db.query(AccountStrategyConfig).filter(
            AccountStrategyConfig.account_id == trader_id
        ).first()

        exchange = strategy_config.exchange if strategy_config else "hyperliquid"
        wallet_bound = False
        wallet_balance = 0
        wallet_env = None

        if exchange == "hyperliquid":
            wallet = db.query(HyperliquidWallet).filter(
                HyperliquidWallet.account_id == trader_id,
                HyperliquidWallet.is_active == "true"
            ).first()
            if wallet:
                wallet_bound = True
                wallet_env = wallet.environment
                snapshot = db.query(HyperliquidAccountSnapshot).filter(
                    HyperliquidAccountSnapshot.account_id == trader_id,
                    HyperliquidAccountSnapshot.environment == wallet.environment
                ).order_by(HyperliquidAccountSnapshot.snapshot_time.desc()).first()
                if snapshot:
                    wallet_balance = float(snapshot.available_balance)
        else:
            wallet = db.query(BinanceWallet).filter(
                BinanceWallet.account_id == trader_id,
                BinanceWallet.is_active == "true"
            ).first()
            if wallet:
                wallet_bound = True
                wallet_env = wallet.environment
                snapshot = db.query(BinanceAccountSnapshot).filter(
                    BinanceAccountSnapshot.account_id == trader_id,
                    BinanceAccountSnapshot.environment == wallet.environment
                ).order_by(BinanceAccountSnapshot.snapshot_time.desc()).first()
                if snapshot:
                    wallet_balance = float(snapshot.available_balance)

        checks.append({
            "check": "wallet_bound",
            "passed": wallet_bound,
            "wallet": f"{exchange}/{wallet_env}" if wallet_bound else None
        })
        if not wallet_bound:
            issues.append(f"No {exchange} wallet bound. Go to Settings → Wallets to configure.")

        checks.append({
            "check": "wallet_balance",
            "passed": wallet_balance > 0,
            "balance": wallet_balance,
            "suggestion": "Deposit funds to wallet" if wallet_balance == 0 else None
        })
        if wallet_balance == 0 and wallet_bound:
            issues.append("Wallet balance is 0. Deposit funds to enable trading.")

        # Check 5: Signal pool or scheduled trigger
        if strategy_config:
            has_signal = bool(strategy_config.signal_pool_ids)
            has_scheduled = strategy_config.scheduled_trigger_enabled
            checks.append({
                "check": "trigger_configured",
                "passed": has_signal or has_scheduled,
                "signal_pools": strategy_config.signal_pool_ids,
                "scheduled_enabled": has_scheduled,
                "interval": strategy_config.trigger_interval
            })
            if not has_signal and not has_scheduled:
                issues.append("No trigger configured (neither signal pool nor scheduled)")

        # Check 6: Recent errors
        recent_errors = []
        ai_errors = db.query(AIDecisionLog).filter(
            AIDecisionLog.account_id == trader_id,
            AIDecisionLog.executed == "false"
        ).order_by(AIDecisionLog.decision_time.desc()).limit(3).all()

        for err in ai_errors:
            recent_errors.append({
                "time": err.decision_time.strftime("%Y-%m-%d %H:%M UTC") if err.decision_time else None,
                "type": "ai_decision",
                "message": err.reason[:100] if err.reason else "Unknown"
            })

        status = "healthy" if not issues else "issues_found"
        summary = issues[0] if issues else "All checks passed. Trader should be operational."

        return json.dumps({
            "trader_id": trader_id,
            "trader_name": account.name,
            "status": status,
            "checks": checks,
            "summary": summary,
            "recent_errors": recent_errors
        }, indent=2)

    except Exception as e:
        logger.error(f"[diagnose_trader_issues] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_save_signal_pool(
    db: Session,
    pool_name: str,
    symbol: str,
    signals: List[Dict[str, Any]],
    logic: str = "AND",
    exchange: str = "hyperliquid",
    description: str = None
) -> str:
    """Create a signal pool by calling the existing API handler."""
    from api.signal_routes import create_pool_from_config, SignalPoolConfigRequest

    try:
        # Defensive validation for taker_volume signals
        for i, sig in enumerate(signals):
            if sig.get("metric") == "taker_volume":
                missing = [f for f in ("direction", "ratio_threshold", "volume_threshold", "time_window")
                           if not sig.get(f) and sig.get(f) != 0]
                if missing or sig.get("operator"):
                    return json.dumps({
                        "error": f"Signal {i+1} (taker_volume) format error. "
                                 f"taker_volume requires: direction, ratio_threshold, volume_threshold, time_window. "
                                 f"Do NOT use operator/threshold for taker_volume.",
                        "correct_example": {
                            "metric": "taker_volume", "direction": "buy",
                            "ratio_threshold": 1.5, "volume_threshold": 100000, "time_window": "5m"
                        }
                    })

        # Build request object for the existing API handler
        request = SignalPoolConfigRequest(
            name=pool_name,
            symbol=symbol,
            signals=signals,
            logic=logic,
            exchange=exchange,
            description=description
        )

        # Call the existing API handler directly
        result = create_pool_from_config(request, db)

        return json.dumps({
            "success": True,
            "pool_id": result["pool"]["id"],
            "pool_name": result["pool"]["pool_name"],
            "symbol": symbol.upper(),
            "signals_created": len(result["signals"]),
            "logic": logic,
            "exchange": exchange,
            "note": "Signal pool created. Bind it to an AI Trader to start receiving triggers."
        })

    except Exception as e:
        db.rollback()
        logger.error(f"[save_signal_pool] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_save_prompt(
    db: Session,
    name: str,
    template_text: str,
    prompt_id: int = None,
    description: str = None
) -> str:
    """Create or update a trading prompt template."""
    from database.models import PromptTemplate
    import re

    try:
        # Extract variables from template
        variables = re.findall(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', template_text)
        variables = list(set(variables))

        if prompt_id:
            # Update existing prompt
            prompt = db.query(PromptTemplate).filter(
                PromptTemplate.id == prompt_id,
                PromptTemplate.is_deleted == "false"
            ).first()
            if not prompt:
                return json.dumps({"error": f"Prompt {prompt_id} not found"})

            prompt.name = name
            prompt.template_text = template_text
            if description:
                prompt.description = description
            prompt.updated_by = "hyper_ai"
            db.commit()
            action = "updated"
        else:
            # Create new prompt
            # Generate unique key
            import uuid
            key = f"hyper_ai_{uuid.uuid4().hex[:8]}"

            # Get default system template
            default_system = db.query(PromptTemplate).filter(
                PromptTemplate.is_system == "true"
            ).first()
            system_text = default_system.system_template_text if default_system else ""

            prompt = PromptTemplate(
                key=key,
                name=name,
                description=description or "",
                template_text=template_text,
                system_template_text=system_text,
                is_system="false",
                is_deleted="false",
                created_by="hyper_ai"
            )
            db.add(prompt)
            db.commit()
            db.refresh(prompt)
            action = "created"

        return json.dumps({
            "success": True,
            "prompt_id": prompt.id,
            "name": name,
            "action": action,
            "variables_detected": variables[:20],
            "note": "Prompt saved. Changes apply to bound AI Traders on next trigger."
        }, indent=2)

    except Exception as e:
        db.rollback()
        logger.error(f"[save_prompt] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_save_program(
    db: Session,
    name: str,
    code: str,
    program_id: int = None,
    description: str = None
) -> str:
    """Create or update a trading program by calling existing API handlers."""
    from routes.program_routes import (
        create_program, update_program, ProgramCreate, ProgramUpdate
    )
    from fastapi import HTTPException

    try:
        if program_id:
            # Update existing program
            data = ProgramUpdate(name=name, code=code, description=description)
            result = update_program(program_id, data, db)
            action = "updated"
        else:
            # Create new program
            data = ProgramCreate(name=name, code=code, description=description)
            result = create_program(data, db)
            action = "created"

        return json.dumps({
            "success": True,
            "program_id": result.id,
            "name": result.name,
            "action": action,
            "validation": {"syntax_valid": True, "security_check": "passed"},
            "note": "Program saved. Use test_run_code to verify logic before binding."
        }, indent=2)

    except HTTPException as e:
        db.rollback()
        logger.error(f"[save_program] HTTPException: {e.detail}")
        return json.dumps({"success": False, "error": e.detail})
    except Exception as e:
        db.rollback()
        logger.error(f"[save_program] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_create_ai_trader(
    db: Session,
    name: str,
    model: str,
    base_url: str,
    api_key: str
) -> str:
    """Create a new AI Trader with LLM config only. Strategy and wallet binding done separately."""
    from database.models import Account

    try:
        # Step 1: Test LLM connection first
        from api.account_routes import test_llm_connection
        import asyncio

        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        test_result = loop.run_until_complete(test_llm_connection({
            "model": model,
            "base_url": base_url,
            "api_key": api_key
        }))

        if not test_result.get("success"):
            return json.dumps({
                "success": False,
                "error": "LLM connection test failed",
                "details": test_result.get("message", "Unknown error"),
                "note": "Please check your LLM credentials and try again."
            })

        # Step 2: Create Account with LLM config only
        account = Account(
            user_id=1,
            name=name,
            account_type="AI",
            is_active="true",
            auto_trading_enabled="false",  # Disabled until strategy/wallet configured
            model=model,
            base_url=base_url,
            api_key=api_key
        )
        db.add(account)
        db.commit()

        return json.dumps({
            "success": True,
            "trader_id": account.id,
            "trader_name": name,
            "llm_config": {
                "model": model,
                "base_url": base_url,
                "connection_tested": True
            },
            "next_steps": [
                "1. Go to Settings → Wallets to bind a wallet",
                "2. Go to Strategies to create/select a trading strategy",
                "3. Enable auto-trading when ready"
            ],
            "note": "AI Trader created with LLM config. Complete wallet and strategy setup to start trading."
        }, indent=2)

    except Exception as e:
        db.rollback()
        logger.error(f"[create_ai_trader] Error: {e}")
        return json.dumps({"error": str(e)})


# =============================================================================
# Query Tools: list resources
# =============================================================================

def execute_list_traders(db: Session, trader_id: int = None) -> str:
    """List all AI Traders with bindings, wallet status, and trading status.
    Pass trader_id to get a single trader's detail."""
    from database.models import (
        Account, HyperliquidWallet, BinanceWallet,
        AccountProgramBinding, AccountPromptBinding,
        TradingProgram, PromptTemplate
    )

    try:
        query = db.query(Account).filter(
            Account.is_active == "true",
            Account.account_type == "AI",
            Account.is_deleted != True
        )
        if trader_id:
            query = query.filter(Account.id == trader_id)
        accounts = query.all()
        if trader_id and not accounts:
            return json.dumps({"error": f"AI Trader {trader_id} not found"})

        traders = []
        for acc in accounts:
            # Wallet info
            hl_wallets = db.query(HyperliquidWallet).filter(
                HyperliquidWallet.account_id == acc.id
            ).all()
            bn_wallets = db.query(BinanceWallet).filter(
                BinanceWallet.account_id == acc.id
            ).all()

            wallet_info = []
            for w in hl_wallets:
                wallet_info.append({
                    "exchange": "hyperliquid",
                    "environment": w.environment
                })
            for w in bn_wallets:
                wallet_info.append({
                    "exchange": "binance",
                    "environment": w.environment
                })

            # Prompt binding
            prompt_binding = None
            pb = db.query(AccountPromptBinding).filter(
                AccountPromptBinding.account_id == acc.id,
                AccountPromptBinding.is_deleted != True
            ).first()
            if pb:
                tpl = db.get(PromptTemplate, pb.prompt_template_id)
                prompt_binding = {
                    "prompt_id": pb.prompt_template_id,
                    "prompt_name": tpl.name if tpl else "Unknown"
                }

            # Program bindings
            prog_bindings = db.query(AccountProgramBinding).filter(
                AccountProgramBinding.account_id == acc.id,
                AccountProgramBinding.is_deleted != True
            ).all()
            program_bindings = []
            for pgb in prog_bindings:
                prog = db.get(TradingProgram, pgb.program_id)
                pool_ids = json.loads(pgb.signal_pool_ids) if pgb.signal_pool_ids else []
                program_bindings.append({
                    "binding_id": pgb.id,
                    "program_id": pgb.program_id,
                    "program_name": prog.name if prog else "Unknown",
                    "signal_pool_ids": pool_ids,
                    "trigger_interval": pgb.trigger_interval,
                    "is_active": pgb.is_active
                })

            traders.append({
                "trader_id": acc.id,
                "name": acc.name,
                "model": acc.model,
                "auto_trading_enabled": acc.auto_trading_enabled == "true",
                "wallets": wallet_info,
                "prompt_binding": prompt_binding,
                "program_bindings": program_bindings
            })

        return json.dumps({"traders": traders, "count": len(traders)}, indent=2)

    except Exception as e:
        logger.error(f"[list_traders] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_list_signal_pools(db: Session, pool_id: int = None) -> str:
    """List all signal pools. Pass pool_id for single pool detail."""
    from database.models import SignalPool, SignalDefinition

    try:
        query = db.query(SignalPool).filter(SignalPool.is_deleted != True)
        if pool_id:
            query = query.filter(SignalPool.id == pool_id)
        pools = query.all()
        if pool_id and not pools:
            return json.dumps({"error": f"Signal pool {pool_id} not found"})

        result = []
        for pool in pools:
            # Parse signal_ids
            signal_ids = []
            if pool.signal_ids:
                try:
                    raw = pool.signal_ids
                    signal_ids = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    signal_ids = []

            # Parse symbols
            symbols = []
            if pool.symbols:
                try:
                    raw = pool.symbols
                    symbols = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    symbols = []

            # Get signal details from trigger_condition
            signals = []
            for sid in signal_ids:
                sig = db.query(SignalDefinition).filter(
                    SignalDefinition.id == sid,
                    SignalDefinition.is_deleted != True
                ).first()
                if sig:
                    cond = {}
                    if sig.trigger_condition:
                        try:
                            raw = sig.trigger_condition
                            cond = json.loads(raw) if isinstance(raw, str) else raw
                        except Exception:
                            cond = {"raw": sig.trigger_condition}
                    signals.append({
                        "signal_id": sig.id,
                        "signal_name": sig.signal_name,
                        "trigger_condition": cond,
                        "enabled": sig.enabled
                    })

            result.append({
                "pool_id": pool.id,
                "name": pool.pool_name,
                "symbols": symbols,
                "exchange": pool.exchange or "hyperliquid",
                "logic": pool.logic or "OR",
                "enabled": pool.enabled,
                "signals": signals
            })

        return json.dumps({"signal_pools": result, "count": len(result)}, indent=2)

    except Exception as e:
        logger.error(f"[list_signal_pools] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_list_strategies(db: Session, strategy_id: int = None, strategy_type: str = None) -> str:
    """List all prompts and programs with binding status.
    Pass strategy_id + strategy_type to get full content of a specific strategy."""
    from database.models import (
        PromptTemplate, TradingProgram,
        AccountProgramBinding, AccountPromptBinding, Account
    )

    try:
        # Single strategy detail mode
        if strategy_id and strategy_type:
            if strategy_type == "prompt":
                tpl = db.query(PromptTemplate).filter(
                    PromptTemplate.id == strategy_id,
                    PromptTemplate.is_deleted == "false"
                ).first()
                if not tpl:
                    return json.dumps({"error": f"Prompt {strategy_id} not found"})
                bindings = db.query(AccountPromptBinding).filter(
                    AccountPromptBinding.prompt_template_id == tpl.id,
                    AccountPromptBinding.is_deleted != True
                ).all()
                bound_traders = []
                for b in bindings:
                    acc = db.get(Account, b.account_id)
                    if acc:
                        bound_traders.append({"trader_id": acc.id, "trader_name": acc.name})
                return json.dumps({
                    "prompt_id": tpl.id,
                    "name": tpl.name,
                    "description": getattr(tpl, "description", None),
                    "template_text": tpl.template_text,
                    "bound_traders": bound_traders
                }, indent=2)
            elif strategy_type == "program":
                prog = db.query(TradingProgram).filter(
                    TradingProgram.id == strategy_id,
                    TradingProgram.is_deleted != True
                ).first()
                if not prog:
                    return json.dumps({"error": f"Program {strategy_id} not found"})
                bindings = db.query(AccountProgramBinding).filter(
                    AccountProgramBinding.program_id == prog.id,
                    AccountProgramBinding.is_deleted != True
                ).all()
                bound_traders = []
                for b in bindings:
                    acc = db.get(Account, b.account_id)
                    if acc:
                        bound_traders.append({
                            "trader_id": acc.id, "trader_name": acc.name,
                            "is_active": b.is_active
                        })
                return json.dumps({
                    "program_id": prog.id,
                    "name": prog.name,
                    "description": prog.description,
                    "code": prog.code,
                    "bound_traders": bound_traders
                }, indent=2)

        # List all mode (original behavior)
        # Prompts
        templates = db.query(PromptTemplate).filter(
            PromptTemplate.is_deleted == "false"
        ).all()
        prompts = []
        for tpl in templates:
            bindings = db.query(AccountPromptBinding).filter(
                AccountPromptBinding.prompt_template_id == tpl.id,
                AccountPromptBinding.is_deleted != True
            ).all()
            bound_traders = []
            for b in bindings:
                acc = db.get(Account, b.account_id)
                if acc:
                    bound_traders.append({"trader_id": acc.id, "trader_name": acc.name})
            prompts.append({
                "prompt_id": tpl.id,
                "name": tpl.name,
                "description": getattr(tpl, "description", None),
                "bound_traders": bound_traders
            })

        # Programs
        programs_db = db.query(TradingProgram).filter(TradingProgram.is_deleted != True).all()
        programs = []
        for prog in programs_db:
            bindings = db.query(AccountProgramBinding).filter(
                AccountProgramBinding.program_id == prog.id,
                AccountProgramBinding.is_deleted != True
            ).all()
            bound_traders = []
            for b in bindings:
                acc = db.get(Account, b.account_id)
                if acc:
                    bound_traders.append({
                        "trader_id": acc.id,
                        "trader_name": acc.name,
                        "is_active": b.is_active
                    })
            programs.append({
                "program_id": prog.id,
                "name": prog.name,
                "description": prog.description,
                "bound_traders": bound_traders
            })

        return json.dumps({
            "prompts": prompts,
            "programs": programs,
            "prompt_count": len(prompts),
            "program_count": len(programs)
        }, indent=2)

    except Exception as e:
        logger.error(f"[list_strategies] Error: {e}")
        return json.dumps({"error": str(e)})


# =============================================================================
# Binding Tools: assemble components
# =============================================================================

def execute_bind_prompt_to_trader(db: Session, trader_id: int, prompt_id: int) -> str:
    """Bind a prompt template to an AI Trader. Reuses prompt_repo.upsert_binding."""
    from database.models import Account, PromptTemplate
    from repositories import prompt_repo

    try:
        account = db.query(Account).filter(Account.id == trader_id, Account.is_deleted != True).first()
        if not account:
            return json.dumps({"error": f"AI Trader {trader_id} not found"})

        template = db.get(PromptTemplate, prompt_id)
        if not template:
            return json.dumps({"error": f"Prompt template {prompt_id} not found"})

        binding = prompt_repo.upsert_binding(
            db,
            account_id=trader_id,
            prompt_template_id=prompt_id,
            updated_by="hyper_ai"
        )

        return json.dumps({
            "success": True,
            "binding_id": binding.id,
            "trader_id": trader_id,
            "trader_name": account.name,
            "prompt_id": prompt_id,
            "prompt_name": template.name
        }, indent=2)

    except Exception as e:
        db.rollback()
        logger.error(f"[bind_prompt_to_trader] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_bind_program_to_trader(
    db: Session, trader_id: int, program_id: int,
    signal_pool_ids: list = None, trigger_interval: int = 300,
    is_active: bool = True
) -> str:
    """Create a program binding for an AI Trader. Reuses AccountProgramBinding model."""
    from database.models import Account, TradingProgram, AccountProgramBinding

    try:
        account = db.query(Account).filter(Account.id == trader_id, Account.is_deleted != True).first()
        if not account:
            return json.dumps({"error": f"AI Trader {trader_id} not found"})

        program = db.get(TradingProgram, program_id)
        if not program:
            return json.dumps({"error": f"Program {program_id} not found"})

        # Check duplicate
        existing = db.query(AccountProgramBinding).filter(
            AccountProgramBinding.account_id == trader_id,
            AccountProgramBinding.program_id == program_id,
            AccountProgramBinding.is_deleted != True
        ).first()
        if existing:
            return json.dumps({
                "error": f"Binding already exists (binding_id={existing.id})",
                "binding_id": existing.id
            })

        binding = AccountProgramBinding(
            account_id=trader_id,
            program_id=program_id,
            signal_pool_ids=json.dumps(signal_pool_ids) if signal_pool_ids else None,
            trigger_interval=trigger_interval,
            is_active=is_active
        )
        db.add(binding)
        db.commit()
        db.refresh(binding)

        return json.dumps({
            "success": True,
            "binding_id": binding.id,
            "trader_id": trader_id,
            "trader_name": account.name,
            "program_id": program_id,
            "program_name": program.name,
            "signal_pool_ids": signal_pool_ids or [],
            "trigger_interval": trigger_interval,
            "is_active": is_active
        }, indent=2)

    except Exception as e:
        db.rollback()
        logger.error(f"[bind_program_to_trader] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_update_trader_strategy(
    db: Session, trader_id: int,
    signal_pool_ids: list = None,
    scheduled_trigger_enabled: bool = None,
    trigger_interval: int = None
) -> str:
    """Update trigger config for a Prompt-based AI Trader. Reuses upsert_strategy."""
    from database.models import Account
    from repositories.strategy_repo import upsert_strategy

    try:
        account = db.query(Account).filter(Account.id == trader_id, Account.is_deleted != True).first()
        if not account:
            return json.dumps({"error": f"AI Trader {trader_id} not found"})

        strategy = upsert_strategy(
            db,
            account_id=trader_id,
            signal_pool_ids=signal_pool_ids,
            scheduled_trigger_enabled=scheduled_trigger_enabled,
            trigger_interval=trigger_interval
        )

        return json.dumps({
            "success": True,
            "trader_id": trader_id,
            "trader_name": account.name,
            "signal_pool_ids": signal_pool_ids,
            "scheduled_trigger_enabled": scheduled_trigger_enabled,
            "trigger_interval": trigger_interval
        }, indent=2)

    except Exception as e:
        db.rollback()
        logger.error(f"[update_trader_strategy] Error: {e}")
        return json.dumps({"error": str(e)})


# =============================================================================
# Update Tools
# =============================================================================

def execute_update_ai_trader(
    db: Session, trader_id: int,
    name: str = None, model: str = None,
    base_url: str = None, api_key: str = None
) -> str:
    """Update AI Trader settings. Tests LLM connection if credentials change."""
    from database.models import Account

    try:
        account = db.query(Account).filter(
            Account.id == trader_id, Account.is_active == "true",
            Account.is_deleted != True
        ).first()
        if not account:
            return json.dumps({"error": f"AI Trader {trader_id} not found"})

        # Test LLM connection if any credential field changes
        new_model = model or account.model
        new_base_url = base_url or account.base_url
        new_api_key = api_key or account.api_key
        need_test = any([model, base_url, api_key])

        if need_test and new_model and new_base_url and new_api_key:
            from api.account_routes import test_llm_connection
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            test_result = loop.run_until_complete(test_llm_connection({
                "model": new_model,
                "base_url": new_base_url,
                "api_key": new_api_key
            }))
            if not test_result.get("success"):
                return json.dumps({
                    "success": False,
                    "error": "LLM connection test failed",
                    "details": test_result.get("message", "Unknown error")
                })

        updated = []
        if name:
            account.name = name
            updated.append("name")
        if model:
            account.model = model
            updated.append("model")
        if base_url:
            account.base_url = base_url
            updated.append("base_url")
        if api_key:
            account.api_key = api_key
            updated.append("api_key")

        db.commit()
        return json.dumps({
            "success": True, "trader_id": trader_id,
            "trader_name": account.name,
            "updated_fields": updated,
            "llm_tested": need_test
        }, indent=2)

    except Exception as e:
        db.rollback()
        logger.error(f"[update_ai_trader] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_update_program_binding(
    db: Session, binding_id: int,
    signal_pool_ids: list = None, trigger_interval: int = None,
    scheduled_trigger_enabled: bool = None, is_active: bool = None,
    params_override: dict = None
) -> str:
    """Update a program binding's configuration."""
    from database.models import AccountProgramBinding, Account

    try:
        binding = db.query(AccountProgramBinding).filter(
            AccountProgramBinding.id == binding_id,
            AccountProgramBinding.is_deleted != True
        ).first()
        if not binding:
            return json.dumps({"error": f"Program binding {binding_id} not found"})

        updated = []
        if signal_pool_ids is not None:
            binding.signal_pool_ids = json.dumps(signal_pool_ids)
            updated.append("signal_pool_ids")
        if trigger_interval is not None:
            binding.trigger_interval = trigger_interval
            updated.append("trigger_interval")
        if scheduled_trigger_enabled is not None:
            binding.scheduled_trigger_enabled = scheduled_trigger_enabled
            updated.append("scheduled_trigger_enabled")
        if is_active is not None:
            binding.is_active = is_active
            updated.append("is_active")
        if params_override is not None:
            binding.params_override = json.dumps(params_override)
            updated.append("params_override")

        db.commit()
        account = db.get(Account, binding.account_id)
        return json.dumps({
            "success": True, "binding_id": binding_id,
            "trader_name": account.name if account else "unknown",
            "updated_fields": updated
        }, indent=2)

    except Exception as e:
        db.rollback()
        logger.error(f"[update_program_binding] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_update_signal_pool(
    db: Session, pool_id: int,
    pool_name: str = None, enabled: bool = None, logic: str = None,
    signal_ids: list = None
) -> str:
    """Update signal pool settings."""
    from database.models import SignalPool, SignalDefinition

    try:
        pool = db.query(SignalPool).filter(SignalPool.id == pool_id, SignalPool.is_deleted != True).first()
        if not pool:
            return json.dumps({"error": f"Signal pool {pool_id} not found"})

        updated = []

        # Validate signal_ids exchange match
        if signal_ids is not None:
            pool_exchange = pool.exchange or "hyperliquid"
            mismatched = []
            for sid in signal_ids:
                sig = db.query(SignalDefinition).filter(SignalDefinition.id == sid, SignalDefinition.is_deleted != True).first()
                if not sig:
                    return json.dumps({"error": f"Signal definition {sid} not found"})
                sig_exchange = sig.exchange or "hyperliquid"
                if sig_exchange != pool_exchange:
                    mismatched.append(f"Signal {sid} ({sig_exchange})")
            if mismatched:
                return json.dumps({
                    "error": f"Exchange mismatch: pool is {pool_exchange}, but {', '.join(mismatched)}"
                })
            pool.signal_ids = json.dumps(signal_ids)
            updated.append("signal_ids")

        if pool_name is not None:
            pool.pool_name = pool_name
            updated.append("pool_name")
        if enabled is not None:
            pool.enabled = enabled
            updated.append("enabled")
        if logic is not None:
            pool.logic = logic
            updated.append("logic")

        db.commit()
        return json.dumps({
            "success": True, "pool_id": pool_id,
            "pool_name": pool.pool_name,
            "updated_fields": updated
        }, indent=2)

    except Exception as e:
        db.rollback()
        logger.error(f"[update_signal_pool] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_update_prompt_binding(db: Session, trader_id: int, prompt_id: int) -> str:
    """Update which prompt is bound to a trader. Reuses upsert_binding."""
    from database.models import Account, PromptTemplate
    from repositories import prompt_repo

    try:
        account = db.query(Account).filter(Account.id == trader_id, Account.is_deleted != True).first()
        if not account:
            return json.dumps({"error": f"AI Trader {trader_id} not found"})

        template = db.get(PromptTemplate, prompt_id)
        if not template:
            return json.dumps({"error": f"Prompt template {prompt_id} not found"})

        binding = prompt_repo.upsert_binding(
            db, account_id=trader_id,
            prompt_template_id=prompt_id, updated_by="hyper_ai"
        )
        return json.dumps({
            "success": True, "binding_id": binding.id,
            "trader_id": trader_id, "trader_name": account.name,
            "prompt_id": prompt_id, "prompt_name": template.name
        }, indent=2)

    except Exception as e:
        db.rollback()
        logger.error(f"[update_prompt_binding] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_save_memory(
    db: Session, category: str, content: str,
    importance: float = 0.5, api_config: Optional[Dict[str, Any]] = None
) -> str:
    """Save a memory with LLM-powered dedup (same logic as compression).

    When api_config is provided, uses batch_dedup_memories to intelligently
    ADD/UPDATE/DELETE memories. Falls back to simple add if no api_config.
    """
    from services.hyper_ai_memory_service import (
        add_memory, MEMORY_CATEGORIES, enforce_memory_limit,
        batch_dedup_memories
    )

    try:
        if category not in MEMORY_CATEGORIES:
            return json.dumps({"error": f"Invalid category. Must be one of: {MEMORY_CATEGORIES}"})

        content = content.strip()
        if len(content) < 10:
            return json.dumps({"error": "Content must be at least 10 characters"})

        importance = max(0.0, min(1.0, importance))

        new_memory = [{"category": category, "content": content, "importance": importance}]

        if api_config and api_config.get("api_key"):
            count = batch_dedup_memories(db, new_memory, api_config, source="ai_tool")
            action = "deduped" if count > 0 else "skipped (redundant)"
        else:
            add_memory(db, category, content, source="ai_tool", importance=importance)
            enforce_memory_limit(db)
            action = "added"

        return json.dumps({
            "success": True,
            "action": action,
            "note": "Memory processed with intelligent dedup. It will be included in future conversations."
        }, indent=2)

    except Exception as e:
        db.rollback()
        logger.error(f"[save_memory] Error: {e}")
        return json.dumps({"error": str(e)})


def execute_hyper_ai_tool(
    db: Session, tool_name: str, arguments: Dict[str, Any],
    user_id: int = 1, api_config: Optional[Dict[str, Any]] = None
) -> str:
    """Execute a Hyper AI tool by name."""
    try:
        if tool_name == "get_system_overview":
            return execute_get_system_overview(db)

        elif tool_name == "get_wallet_status":
            return execute_get_wallet_status(
                db,
                exchange=arguments.get("exchange", "all"),
                environment=arguments.get("environment", "all")
            )

        elif tool_name == "get_api_reference":
            return execute_get_api_reference(
                doc_type=arguments.get("doc_type", "prompt"),
                api_type=arguments.get("api_type", "all"),
                lang=arguments.get("lang", "en")
            )

        elif tool_name == "get_klines":
            return execute_get_klines(
                db,
                symbol=arguments.get("symbol", "BTC"),
                period=arguments.get("period", "1h"),
                limit=arguments.get("limit", 50),
                exchange=arguments.get("exchange", "hyperliquid")
            )

        elif tool_name == "get_market_regime":
            return execute_get_market_regime(
                db,
                symbol=arguments.get("symbol", "BTC"),
                period=arguments.get("period", "1h"),
                exchange=arguments.get("exchange", "hyperliquid")
            )

        elif tool_name == "get_market_flow":
            return execute_get_market_flow(
                db,
                symbol=arguments.get("symbol", "BTC"),
                period=arguments.get("period", "1h"),
                exchange=arguments.get("exchange", "hyperliquid")
            )

        elif tool_name == "get_system_logs":
            return execute_get_system_logs(
                db,
                level=arguments.get("level", "error"),
                limit=arguments.get("limit", 20),
                trader_id=arguments.get("trader_id")
            )

        elif tool_name == "get_contact_config":
            return execute_get_contact_config()

        elif tool_name == "diagnose_trader_issues":
            return execute_diagnose_trader_issues(db, trader_id=arguments.get("trader_id"))

        elif tool_name == "save_signal_pool":
            return execute_save_signal_pool(
                db,
                pool_name=arguments.get("pool_name"),
                symbol=arguments.get("symbol", "BTC"),
                signals=arguments.get("signals", []),
                logic=arguments.get("logic", "AND"),
                exchange=arguments.get("exchange", "hyperliquid"),
                description=arguments.get("description")
            )

        elif tool_name == "save_prompt":
            return execute_save_prompt(
                db,
                name=arguments.get("name"),
                template_text=arguments.get("template_text"),
                prompt_id=arguments.get("prompt_id"),
                description=arguments.get("description")
            )

        elif tool_name == "save_program":
            return execute_save_program(
                db,
                name=arguments.get("name"),
                code=arguments.get("code"),
                program_id=arguments.get("program_id"),
                description=arguments.get("description")
            )

        elif tool_name == "create_ai_trader":
            return execute_create_ai_trader(
                db,
                name=arguments.get("name"),
                model=arguments.get("model"),
                base_url=arguments.get("base_url"),
                api_key=arguments.get("api_key")
            )

        # --- Query tools: list resources ---
        elif tool_name == "list_traders":
            return execute_list_traders(db, trader_id=arguments.get("trader_id"))

        elif tool_name == "list_signal_pools":
            return execute_list_signal_pools(db, pool_id=arguments.get("pool_id"))

        elif tool_name == "list_strategies":
            return execute_list_strategies(
                db,
                strategy_id=arguments.get("strategy_id"),
                strategy_type=arguments.get("strategy_type")
            )

        # --- Binding tools: assemble components ---
        elif tool_name == "bind_prompt_to_trader":
            return execute_bind_prompt_to_trader(
                db,
                trader_id=arguments.get("trader_id"),
                prompt_id=arguments.get("prompt_id")
            )

        elif tool_name == "bind_program_to_trader":
            return execute_bind_program_to_trader(
                db,
                trader_id=arguments.get("trader_id"),
                program_id=arguments.get("program_id"),
                signal_pool_ids=arguments.get("signal_pool_ids"),
                trigger_interval=arguments.get("trigger_interval", 300),
                is_active=arguments.get("is_active", True)
            )

        elif tool_name == "update_trader_strategy":
            return execute_update_trader_strategy(
                db,
                trader_id=arguments.get("trader_id"),
                signal_pool_ids=arguments.get("signal_pool_ids"),
                scheduled_trigger_enabled=arguments.get("scheduled_trigger_enabled"),
                trigger_interval=arguments.get("trigger_interval")
            )

        # --- Update tools ---
        elif tool_name == "update_ai_trader":
            return execute_update_ai_trader(
                db, trader_id=arguments.get("trader_id"),
                name=arguments.get("name"), model=arguments.get("model"),
                base_url=arguments.get("base_url"), api_key=arguments.get("api_key")
            )

        elif tool_name == "update_program_binding":
            return execute_update_program_binding(
                db, binding_id=arguments.get("binding_id"),
                signal_pool_ids=arguments.get("signal_pool_ids"),
                trigger_interval=arguments.get("trigger_interval"),
                scheduled_trigger_enabled=arguments.get("scheduled_trigger_enabled"),
                is_active=arguments.get("is_active"),
                params_override=arguments.get("params_override")
            )

        elif tool_name == "update_signal_pool":
            return execute_update_signal_pool(
                db, pool_id=arguments.get("pool_id"),
                pool_name=arguments.get("pool_name"),
                enabled=arguments.get("enabled"),
                logic=arguments.get("logic"),
                signal_ids=arguments.get("signal_ids")
            )

        elif tool_name == "update_prompt_binding":
            return execute_update_prompt_binding(
                db, trader_id=arguments.get("trader_id"),
                prompt_id=arguments.get("prompt_id")
            )

        # --- Skill tools: load workflow guidance (no side effects) ---
        elif tool_name == "load_skill":
            from services.hyper_ai_skill_engine import load_skill
            return json.dumps(load_skill(skill_name=arguments.get("skill_name", "")))

        elif tool_name == "load_skill_reference":
            from services.hyper_ai_skill_engine import load_skill_reference
            return json.dumps(load_skill_reference(
                skill_name=arguments.get("skill_name", ""),
                reference_file=arguments.get("reference_file", "")
            ))

        elif tool_name == "save_memory":
            return execute_save_memory(
                db,
                category=arguments.get("category", "context"),
                content=arguments.get("content", ""),
                importance=arguments.get("importance", 0.5),
                api_config=api_config
            )

        # --- Delete tools ---
        elif tool_name == "delete_trader":
            from services.entity_deletion_service import delete_trader
            return json.dumps(delete_trader(db, trader_id=arguments.get("trader_id")), indent=2)

        elif tool_name == "delete_prompt_template":
            from services.entity_deletion_service import delete_prompt_template
            return json.dumps(delete_prompt_template(db, prompt_id=arguments.get("prompt_id")), indent=2)

        elif tool_name == "delete_signal_definition":
            from services.entity_deletion_service import delete_signal_definition
            return json.dumps(delete_signal_definition(db, signal_id=arguments.get("signal_id")), indent=2)

        elif tool_name == "delete_signal_pool":
            from services.entity_deletion_service import delete_signal_pool
            return json.dumps(delete_signal_pool(db, pool_id=arguments.get("pool_id")), indent=2)

        elif tool_name == "delete_trading_program":
            from services.entity_deletion_service import delete_trading_program
            return json.dumps(delete_trading_program(db, program_id=arguments.get("program_id")), indent=2)

        elif tool_name == "delete_prompt_binding":
            from services.entity_deletion_service import delete_prompt_binding
            return json.dumps(delete_prompt_binding(db, binding_id=arguments.get("binding_id")), indent=2)

        elif tool_name == "delete_program_binding":
            from services.entity_deletion_service import delete_program_binding
            return json.dumps(delete_program_binding(db, binding_id=arguments.get("binding_id")), indent=2)

        # Sub-agent tools are handled directly in hyper_ai_service.py main loop
        # via _execute_tool_with_progress() which uses yield from for progress events.
        # This branch should not be reached but kept as safety fallback.
        elif tool_name in ("call_prompt_ai", "call_program_ai", "call_signal_ai", "call_attribution_ai"):
            logger.warning(f"[execute_hyper_ai_tool] Sub-agent {tool_name} reached fallback path")
            return execute_subagent_tool(db, tool_name, arguments, user_id=user_id)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.error(f"[execute_hyper_ai_tool] Error executing {tool_name}: {e}")
        return json.dumps({"error": str(e)})
