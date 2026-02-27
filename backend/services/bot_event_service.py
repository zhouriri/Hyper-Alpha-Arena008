"""Bot Event Service - System event queue for Bot conversations.

Events (signal triggers, AI decisions) are saved as assistant messages
to the Bot conversation and pushed to all bound channels.
Push = broadcast to ALL bound channels; Reply = unicast to originating channel.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from database.models import HyperAiConversation, HyperAiMessage, BotConfig

logger = logging.getLogger(__name__)


def get_bot_conversations(db: Session) -> List[HyperAiConversation]:
    """Get all Bot conversations (one per platform)."""
    return db.query(HyperAiConversation).filter(
        HyperAiConversation.is_bot_conversation == True
    ).all()


def get_connected_bot_configs(db: Session) -> List[BotConfig]:
    """Get all connected bot configurations."""
    return db.query(BotConfig).filter(
        BotConfig.status == "connected"
    ).all()


def enqueue_system_event(
    db: Session,
    event_type: str,
    event_data: Dict[str, Any],
    format_message: bool = True
) -> List[Dict[str, Any]]:
    """
    Enqueue a system event to all Bot conversations.

    Args:
        db: Database session
        event_type: Type of event (signal_triggered, ai_decision, etc.)
        event_data: Event payload
        format_message: If True, format event_data into readable message

    Returns:
        List of dicts with conversation_id, message_id, platform for push
    """
    # Format event into readable message
    if format_message:
        content = _format_event_message(event_type, event_data)
    else:
        content = event_data.get("message", str(event_data))

    results = []
    bot_convs = get_bot_conversations(db)

    for conv in bot_convs:
        # Save as assistant message (Bot-initiated)
        message = HyperAiMessage(
            conversation_id=conv.id,
            role="assistant",
            content=content,
            is_complete=True
        )
        db.add(message)
        db.flush()

        results.append({
            "conversation_id": conv.id,
            "message_id": message.id,
            "platform": conv.bot_platform,
            "content": content
        })

    db.commit()
    return results


def _format_event_message(event_type: str, data: Dict[str, Any]) -> str:
    """Format event data into human-readable message."""
    header = "【🤖 Hyper AI Notification】\n\n"

    if event_type == "signal_triggered":
        pool_name = data.get('pool_name', 'Unknown')
        symbol = data.get('symbol', 'N/A')
        triggered = data.get('triggered_signals', [])
        signals_text = ", ".join(
            f"{s.get('signal_name', s.get('metric', 'N/A'))}={s.get('current_value', 'N/A'):.4f}"
            if isinstance(s.get('current_value'), (int, float))
            else f"{s.get('signal_name', s.get('metric', 'N/A'))}={s.get('current_value', 'N/A')}"
            for s in triggered[:3]
        ) if triggered else "N/A"
        return (
            f"{header}"
            f"🔔 【{pool_name}】 {symbol} triggered\n"
            f"{signals_text}"
        )
    elif event_type == "ai_decision":
        trader = data.get('trader_name', 'AI Trader')
        op = data.get('operation', 'HOLD')
        symbol = data.get('symbol', 'N/A')
        portion = data.get('target_portion', 'N/A')
        reason = data.get('reason', '')
        return (
            f"{header}"
            f"🤖 【{trader}】 {op} {symbol} {portion}\n"
            f"{reason}"
        )
    elif event_type == "program_decision":
        program = data.get('program_name', 'Program')
        op = data.get('operation', 'HOLD')
        symbol = data.get('symbol', 'N/A')
        size = data.get('size_usd', 'N/A')
        leverage = data.get('leverage', '')
        reason = data.get('reason', '')
        return (
            f"{header}"
            f"⚙️ 【{program}】 {op} {symbol} {size} {leverage}\n"
            f"{reason}"
        )
    elif event_type == "trade_executed":
        return (
            f"{header}"
            f"✅ **Trade Executed**\n"
            f"Symbol: {data.get('symbol', 'N/A')}\n"
            f"Side: {data.get('side', 'N/A')}\n"
            f"Size: {data.get('size', 'N/A')}\n"
            f"Price: {data.get('price', 'N/A')}"
        )
    else:
        return f"{header}📢 {event_type}: {data.get('message', str(data))}"


async def push_event_to_all_channels(
    db: Session,
    event_results: List[Dict[str, Any]]
):
    """
    Push broadcast: send event notification to ALL bound channels on ALL connected platforms.
    Called after enqueue_system_event to notify users on Telegram/Discord.
    """
    from database.models import BotConfig

    if not event_results:
        return

    # Get content from first result (all results have the same content)
    content = event_results[0].get("content", "")
    if not content:
        return

    # Query all connected platforms
    configs = db.query(BotConfig).filter(BotConfig.status == "connected").all()

    for config in configs:
        if config.platform == "telegram":
            await _push_to_telegram(db, content)
        elif config.platform == "discord":
            # TODO: Implement Discord push
            pass


async def _push_to_telegram(db: Session, content: str):
    """Push message to all known Telegram chat_ids."""
    from services.bot_service import get_decrypted_bot_token
    from services.telegram_bot_service import send_telegram_message
    from database.models import BotConfig

    token = get_decrypted_bot_token(db, "telegram")
    if not token:
        return

    # Get stored chat_ids from bot config metadata
    config = db.query(BotConfig).filter(
        BotConfig.platform == "telegram"
    ).first()
    if not config:
        return

    # For now, we store chat_ids in a separate tracking mechanism
    # This will be populated when users first message the bot
    from database.models import BotChatBinding
    bindings = db.query(BotChatBinding).filter(
        BotChatBinding.platform == "telegram",
        BotChatBinding.is_active == True
    ).all()

    for binding in bindings:
        try:
            await send_telegram_message(token, binding.chat_id, content)
        except Exception as e:
            logger.error(f"Failed to push to Telegram chat {binding.chat_id}: {e}")
