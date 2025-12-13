"""
AI Signal Generation Service

Handles AI-assisted signal creation conversations using LLM.
Supports Function Calling for AI to query real market data.
"""

import json
import logging
import re
import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

from sqlalchemy.orm import Session

from database.models import AiSignalConversation, AiSignalMessage, Account
from services.ai_decision_service import build_chat_completion_endpoints, _extract_text_from_message
from services.signal_backtest_service import signal_backtest_service, TIMEFRAME_MS

logger = logging.getLogger(__name__)

# System prompt for AI signal generation with Function Calling
SIGNAL_SYSTEM_PROMPT = """You are an expert trading signal designer for cryptocurrency perpetual futures.
You have access to TOOLS to query real market data. You MUST use them before setting thresholds.

## MANDATORY WORKFLOW (Follow these steps, MAX 4 tool calls total)
1. User describes need → Select appropriate indicator and time_window
2. Call `get_indicator_statistics` to see actual data distribution
3. Based on statistics, choose a candidate threshold (use p90-p99 for rare events)
4. Call `backtest_threshold` to test trigger frequency
5. If triggers not ideal, adjust threshold ONCE more (don't keep iterating)
6. **STOP CALLING TOOLS** and output final signal config with explanation

**IMPORTANT**: After 3-4 tool calls, you MUST output the final signal config. Do NOT keep adjusting thresholds indefinitely. A "reasonable" trigger count (5-50) is acceptable.

## CRITICAL RULES
- **ONE SIGNAL = ONE INDICATOR**: Each signal must have exactly ONE metric
- **MUST USE TOOLS**: Do NOT guess thresholds. Always query real data first
- **EXPLAIN YOUR REASONING**: After getting tool results, explain why you chose the threshold

## Available Indicators
- oi_delta_percent: OI change % over time window
- funding_rate: Perpetual funding rate %
- cvd: Cumulative Volume Delta
- depth_ratio: Bid/Ask depth ratio
- order_imbalance: Normalized imbalance (-1 to +1)
- taker_buy_ratio: Taker buy/sell volume ratio

## Operators
- greater_than, less_than, greater_than_or_equal, less_than_or_equal, abs_greater_than

## Time Windows
- 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h

## Output Format (include explanation before config)
Explain your analysis, then output:
```signal-config
{
  "name": "BTC_OI_Surge",
  "symbol": "BTC",
  "description": "Detects significant OI increase",
  "trigger_condition": {
    "metric": "oi_delta_percent",
    "operator": "greater_than",
    "threshold": 1.0,
    "time_window": "5m"
  }
}
```
"""

# Tools schema for Function Calling
SIGNAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_indicator_statistics",
            "description": "Get statistical distribution of an indicator over the past 7 days. Returns min, max, mean, and percentiles (p50, p75, p90, p95, p99).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading symbol, e.g., BTC, ETH"},
                    "indicator": {"type": "string", "enum": ["oi_delta_percent", "funding_rate", "cvd", "depth_ratio", "order_imbalance", "taker_buy_ratio"], "description": "Indicator metric name"},
                    "time_window": {"type": "string", "enum": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"], "description": "Aggregation time window"}
                },
                "required": ["symbol", "indicator", "time_window"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "backtest_threshold",
            "description": "Backtest a threshold on historical market flow data. Returns trigger count and sample timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading symbol"},
                    "indicator": {"type": "string", "description": "Indicator metric name"},
                    "operator": {"type": "string", "enum": ["greater_than", "less_than", "greater_than_or_equal", "less_than_or_equal", "abs_greater_than"], "description": "Comparison operator"},
                    "threshold": {"type": "number", "description": "Threshold value to test"},
                    "time_window": {"type": "string", "description": "Time window for aggregation"}
                },
                "required": ["symbol", "indicator", "operator", "threshold", "time_window"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_kline_context",
            "description": "Get K-line price data around specific timestamps to verify if triggers align with meaningful price movements. Time window matches the signal's time_window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading symbol"},
                    "timestamps": {"type": "array", "items": {"type": "integer"}, "description": "List of trigger timestamps (max 10)"},
                    "time_window": {"type": "string", "description": "K-line interval matching signal time_window"}
                },
                "required": ["symbol", "timestamps", "time_window"]
            }
        }
    }
]


def generate_signal_with_ai(
    db: Session,
    account_id: int,
    user_message: str,
    conversation_id: Optional[int] = None,
    user_id: int = 1
) -> Dict[str, Any]:
    """
    Generate signal configuration using AI.
    Follows the same pattern as ai_prompt_generation_service.generate_prompt_with_ai
    """
    start_time = time.time()
    request_id = f"signal_gen_{int(start_time)}"

    logger.info(f"[AI Signal Gen {request_id}] Starting: account_id={account_id}, "
                f"conversation_id={conversation_id}, user_message_length={len(user_message)}")

    try:
        # Get the specified AI account
        account = db.query(Account).filter(
            Account.id == account_id,
            Account.account_type == "AI"
        ).first()

        if not account:
            return {"success": False, "error": "AI account not found"}

        # Get or create conversation
        conversation = None
        if conversation_id:
            conversation = db.query(AiSignalConversation).filter(
                AiSignalConversation.id == conversation_id,
                AiSignalConversation.user_id == user_id
            ).first()
            if not conversation:
                logger.warning(f"[AI Signal Gen {request_id}] Conversation {conversation_id} not found")

        if not conversation:
            title = user_message[:50] + "..." if len(user_message) > 50 else user_message
            conversation = AiSignalConversation(user_id=user_id, title=title)
            db.add(conversation)
            db.flush()
            logger.info(f"[AI Signal Gen {request_id}] Created new conversation: id={conversation.id}")

        # Save user message
        user_msg = AiSignalMessage(
            conversation_id=conversation.id,
            role="user",
            content=user_message
        )
        db.add(user_msg)
        db.flush()

        # Build message history (same pattern as ai_prompt_generation_service)
        messages = [{"role": "system", "content": SIGNAL_SYSTEM_PROMPT}]

        history_messages = db.query(AiSignalMessage).filter(
            AiSignalMessage.conversation_id == conversation.id,
            AiSignalMessage.id != user_msg.id
        ).order_by(AiSignalMessage.created_at).limit(10).all()

        for msg in history_messages:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        logger.info(f"[AI Signal Gen {request_id}] Built message context: {len(messages)} messages total")

        # Call LLM API with Function Calling support
        endpoints = build_chat_completion_endpoints(account.base_url, account.model)
        if not endpoints:
            return {"success": False, "error": "Invalid base_url configuration"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {account.api_key}"
        }

        # Function Calling loop (max 6 rounds, last round forces no tools)
        max_tool_rounds = 6
        tool_round = 0
        assistant_content = None

        while tool_round < max_tool_rounds:
            tool_round += 1
            is_last_round = (tool_round == max_tool_rounds)
            logger.info(f"[AI Signal Gen {request_id}] Tool round {tool_round}/{max_tool_rounds} (last={is_last_round})")

            request_payload = {
                "model": account.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4096,
            }

            # On last round, force model to give final answer without tools
            if is_last_round:
                # Don't include tools, force text response
                messages.append({
                    "role": "user",
                    "content": "You have used enough tools. Now output the final signal configuration based on your analysis. Include the ```signal-config``` block."
                })
            else:
                request_payload["tools"] = SIGNAL_TOOLS
                request_payload["tool_choice"] = "auto"

            response = None
            last_error = None

            for endpoint in endpoints:
                try:
                    logger.info(f"[AI Signal Gen {request_id}] Trying endpoint: {endpoint}")
                    api_start = time.time()
                    response = requests.post(endpoint, json=request_payload, headers=headers, timeout=120)
                    api_elapsed = time.time() - api_start

                    if response.status_code == 200:
                        logger.info(f"[AI Signal Gen {request_id}] Success in {api_elapsed:.2f}s")
                        break
                    else:
                        logger.warning(f"[AI Signal Gen {request_id}] Endpoint failed: {response.status_code}")
                        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                except requests.exceptions.Timeout:
                    last_error = "Request timeout"
                    logger.warning(f"[AI Signal Gen {request_id}] Timeout on {endpoint}")
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"[AI Signal Gen {request_id}] Error on {endpoint}: {e}")

            if not response or response.status_code != 200:
                return {"success": False, "error": f"All endpoints failed. Last error: {last_error}"}

            # Parse response
            try:
                response_json = response.json()
                message = response_json["choices"][0]["message"]
            except Exception as e:
                logger.error(f"[AI Signal Gen {request_id}] Failed to parse response: {e}")
                return {"success": False, "error": f"Failed to parse AI response: {str(e)}"}

            # Check for tool calls
            tool_calls = message.get("tool_calls", [])
            reasoning_content = message.get("reasoning_content", "")
            content = message.get("content", "")

            # Log for debugging
            logger.info(f"[AI Signal Gen {request_id}] Response: tool_calls={len(tool_calls) if tool_calls else 0}, "
                       f"has_reasoning={bool(reasoning_content)}, has_content={bool(content)}")

            if tool_calls:
                # Add assistant message with tool calls AND reasoning_content to history
                # DeepSeek Reasoner requires reasoning_content to be passed back
                assistant_msg_dict = {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls
                }
                # Include reasoning_content if present (required for DeepSeek Reasoner)
                if reasoning_content:
                    assistant_msg_dict["reasoning_content"] = reasoning_content
                messages.append(assistant_msg_dict)

                # Execute each tool and add results
                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    try:
                        func_args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

                    logger.info(f"[AI Signal Gen {request_id}] Executing tool: {func_name}({func_args})")
                    tool_result = _execute_tool(db, func_name, func_args)
                    logger.info(f"[AI Signal Gen {request_id}] Tool result: {tool_result[:200]}...")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result
                    })
                # Continue loop for next round
            else:
                # No tool calls - AI returned final response
                # Combine reasoning_content and content for full response
                full_content = ""
                if reasoning_content:
                    full_content += f"**Reasoning:**\n{reasoning_content}\n\n"
                if content:
                    full_content += content
                assistant_content = _extract_text_from_message(full_content) if full_content else ""
                break

        # If we exhausted tool rounds, use the last content we received
        if assistant_content is None:
            # Try to get content from the last message in the loop
            if 'message' in dir() and message:
                last_content = message.get("content", "")
                last_reasoning = message.get("reasoning_content", "")
                if last_content or last_reasoning:
                    full_content = ""
                    if last_reasoning:
                        full_content += f"**Reasoning:**\n{last_reasoning}\n\n"
                    if last_content:
                        full_content += last_content
                    assistant_content = _extract_text_from_message(full_content)
                    logger.info(f"[AI Signal Gen {request_id}] Using last round content after limit reached")

            if not assistant_content:
                assistant_content = "Tool calling limit reached. Please try again with a simpler request."

        # Extract signal configs from response
        signal_configs = extract_signal_configs(assistant_content)

        # Save assistant message
        assistant_msg = AiSignalMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=assistant_content,
            signal_configs=json.dumps(signal_configs) if signal_configs else None
        )
        db.add(assistant_msg)
        db.commit()

        total_elapsed = time.time() - start_time
        logger.info(f"[AI Signal Gen {request_id}] Completed in {total_elapsed:.2f}s: "
                   f"conversation_id={conversation.id}, configs_found={len(signal_configs)}")

        return {
            "success": True,
            "conversation_id": conversation.id,
            "message_id": assistant_msg.id,
            "content": assistant_content,
            "signal_configs": signal_configs
        }

    except Exception as e:
        logger.error(f"[AI Signal Gen {request_id}] Unexpected error: {type(e).__name__}: {str(e)}",
                    exc_info=True)
        db.rollback()
        return {"success": False, "error": f"Internal error: {type(e).__name__}"}


def extract_signal_configs(content: str) -> List[Dict]:
    """Extract signal configurations from AI response."""
    configs = []
    pattern = r"```signal-config\s*([\s\S]*?)```"
    matches = re.findall(pattern, content)

    for match in matches:
        try:
            config = json.loads(match.strip())
            configs.append(config)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse signal config: {e}")
            continue

    return configs


def get_signal_conversation_history(
    db: Session,
    user_id: int,
    limit: int = 20
) -> List[Dict]:
    """Get list of AI signal conversations for a user."""
    conversations = db.query(AiSignalConversation).filter(
        AiSignalConversation.user_id == user_id
    ).order_by(AiSignalConversation.updated_at.desc()).limit(limit).all()

    return [
        {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None
        }
        for conv in conversations
    ]


def get_signal_conversation_messages(
    db: Session,
    conversation_id: int,
    user_id: int
) -> Optional[List[Dict]]:
    """Get all messages in a specific conversation."""
    conversation = db.query(AiSignalConversation).filter(
        AiSignalConversation.id == conversation_id,
        AiSignalConversation.user_id == user_id
    ).first()

    if not conversation:
        return None

    messages = db.query(AiSignalMessage).filter(
        AiSignalMessage.conversation_id == conversation_id
    ).order_by(AiSignalMessage.created_at).all()

    return [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "signal_configs": json.loads(msg.signal_configs) if msg.signal_configs else None,
            "created_at": msg.created_at.isoformat() if msg.created_at else None
        }
        for msg in messages
    ]


# ============== Tool Function Implementations ==============

def _tool_get_indicator_statistics(
    db: Session, symbol: str, indicator: str, time_window: str
) -> Dict[str, Any]:
    """Get statistical distribution of an indicator."""
    import numpy as np

    # Map indicator names
    metric_map = {
        "oi_delta_percent": "oi_delta",
        "funding_rate": "funding",
        "taker_buy_ratio": "taker_ratio",
    }
    metric = metric_map.get(indicator, indicator)
    interval_ms = TIMEFRAME_MS.get(time_window, 300000)

    # Get bucket values using backtest service's method
    signal_backtest_service._bucket_cache = {}
    bucket_values = signal_backtest_service._compute_all_bucket_values(
        db, symbol.upper(), metric, interval_ms
    )

    if not bucket_values:
        return {"error": f"No data found for {indicator} on {symbol}"}

    values = [v for v in bucket_values.values() if v is not None]
    if not values:
        return {"error": "No valid values found"}

    arr = np.array(values)
    return {
        "symbol": symbol.upper(),
        "indicator": indicator,
        "time_window": time_window,
        "data_points": len(values),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _tool_backtest_threshold(
    db: Session, symbol: str, indicator: str, operator: str,
    threshold: float, time_window: str
) -> Dict[str, Any]:
    """Backtest a threshold on historical market flow data."""
    # Build trigger condition
    trigger_condition = {
        "metric": indicator,
        "operator": operator,
        "threshold": threshold,
        "time_window": time_window
    }

    # Use existing backtest service
    result = signal_backtest_service.backtest_temp_signal(
        db=db,
        symbol=symbol.upper(),
        trigger_condition=trigger_condition,
        kline_min_ts=None,
        kline_max_ts=None
    )

    if "error" in result:
        return {"error": result["error"]}

    triggers = result.get("triggers", [])
    trigger_count = len(triggers)

    # Return sample timestamps (max 10 for AI to analyze)
    sample_timestamps = [t["timestamp"] for t in triggers[:10]]

    return {
        "symbol": symbol.upper(),
        "indicator": indicator,
        "operator": operator,
        "threshold": threshold,
        "time_window": time_window,
        "trigger_count": trigger_count,
        "sample_timestamps": sample_timestamps,
        "assessment": (
            "too_many" if trigger_count > 50 else
            "too_few" if trigger_count < 5 else
            "reasonable"
        )
    }


def _tool_get_kline_context(
    db: Session, symbol: str, timestamps: List[int], time_window: str
) -> Dict[str, Any]:
    """Get K-line price data around specific timestamps."""
    # Map time_window to Hyperliquid interval format
    interval_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
        "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h"
    }
    interval = interval_map.get(time_window, "5m")
    interval_ms = TIMEFRAME_MS.get(time_window, 300000)

    # Limit to 10 timestamps
    timestamps = timestamps[:10]
    if not timestamps:
        return {"error": "No timestamps provided"}

    # Fetch K-lines from Hyperliquid API
    try:
        # Get range covering all timestamps with some buffer
        min_ts = min(timestamps) - (10 * interval_ms)
        max_ts = max(timestamps) + (10 * interval_ms)

        url = "https://api.hyperliquid.xyz/info"
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol.upper(),
                "interval": interval,
                "startTime": min_ts,
                "endTime": max_ts
            }
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            return {"error": f"Failed to fetch K-lines: HTTP {resp.status_code}"}

        klines = resp.json()
        if not klines:
            return {"error": "No K-line data returned"}

        # Build K-line lookup by timestamp
        kline_map = {}
        for k in klines:
            ts = k.get("t", k.get("T", 0))
            kline_map[ts] = {
                "open": float(k.get("o", 0)),
                "high": float(k.get("h", 0)),
                "low": float(k.get("l", 0)),
                "close": float(k.get("c", 0)),
                "volume": float(k.get("v", 0))
            }

        # For each trigger timestamp, get context (before, at, after)
        contexts = []
        sorted_kline_ts = sorted(kline_map.keys())
        for trigger_ts in timestamps:
            # Find closest K-line
            closest_ts = min(sorted_kline_ts, key=lambda x: abs(x - trigger_ts))
            idx = sorted_kline_ts.index(closest_ts)

            context = {"trigger_ts": trigger_ts, "klines": []}
            # Get 3 K-lines before, the trigger, and 3 after
            for i in range(max(0, idx - 3), min(len(sorted_kline_ts), idx + 4)):
                ts = sorted_kline_ts[i]
                k = kline_map[ts]
                context["klines"].append({
                    "ts": ts,
                    "o": k["open"], "h": k["high"], "l": k["low"], "c": k["close"]
                })
            contexts.append(context)

        return {
            "symbol": symbol.upper(),
            "time_window": time_window,
            "contexts": contexts
        }
    except Exception as e:
        logger.error(f"Error fetching K-line context: {e}")
        return {"error": str(e)}


def _execute_tool(db: Session, tool_name: str, arguments: Dict) -> str:
    """Execute a tool and return JSON result."""
    try:
        if tool_name == "get_indicator_statistics":
            result = _tool_get_indicator_statistics(
                db=db,
                symbol=arguments.get("symbol", "BTC"),
                indicator=arguments.get("indicator", "depth_ratio"),
                time_window=arguments.get("time_window", "5m")
            )
        elif tool_name == "backtest_threshold":
            result = _tool_backtest_threshold(
                db=db,
                symbol=arguments.get("symbol", "BTC"),
                indicator=arguments.get("indicator", "depth_ratio"),
                operator=arguments.get("operator", "greater_than"),
                threshold=arguments.get("threshold", 1.0),
                time_window=arguments.get("time_window", "5m")
            )
        elif tool_name == "get_kline_context":
            result = _tool_get_kline_context(
                db=db,
                symbol=arguments.get("symbol", "BTC"),
                timestamps=arguments.get("timestamps", []),
                time_window=arguments.get("time_window", "5m")
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        return json.dumps(result)
    except Exception as e:
        logger.error(f"Tool execution error: {tool_name} - {e}")
        return json.dumps({"error": str(e)})
