"""Bot Integration API Routes - Manage Telegram/Discord bot configurations"""
import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from database.connection import get_db
from services.bot_service import (
    get_bot_config,
    get_all_bot_configs,
    save_bot_config,
    get_decrypted_bot_token,
    update_bot_status,
    delete_bot_config,
)
from services.telegram_bot_service import (
    validate_telegram_token,
    setup_telegram_webhook,
    remove_telegram_webhook,
    send_telegram_message,
)

router = APIRouter(prefix="/api/bot", tags=["Bot Integration"])


class BotConfigRequest(BaseModel):
    platform: str  # telegram / discord
    bot_token: str
    bot_username: Optional[str] = None
    bot_app_id: Optional[str] = None


class BotStatusRequest(BaseModel):
    platform: str
    status: str  # disconnected / connecting / connected / error
    error_message: Optional[str] = None


@router.get("/configs")
def list_bot_configs(db: Session = Depends(get_db)):
    """List all bot configurations."""
    return {"configs": get_all_bot_configs(db)}


@router.get("/config/{platform}")
def get_bot_config_endpoint(platform: str, db: Session = Depends(get_db)):
    """Get bot configuration for a specific platform."""
    config = get_bot_config(db, platform)
    if not config:
        return {"config": None, "configured": False}
    return {"config": config, "configured": True}


@router.post("/config")
def save_bot_config_endpoint(request: BotConfigRequest, db: Session = Depends(get_db)):
    """Save or update bot configuration."""
    if request.platform not in ["telegram", "discord"]:
        raise HTTPException(status_code=400, detail="Invalid platform. Must be 'telegram' or 'discord'")

    if not request.bot_token:
        raise HTTPException(status_code=400, detail="Bot token is required")

    config = save_bot_config(
        db=db,
        platform=request.platform,
        bot_token=request.bot_token,
        bot_username=request.bot_username,
        bot_app_id=request.bot_app_id,
    )
    return {"success": True, "config": config}


@router.put("/status")
def update_bot_status_endpoint(request: BotStatusRequest, db: Session = Depends(get_db)):
    """Update bot connection status."""
    success = update_bot_status(
        db=db,
        platform=request.platform,
        status=request.status,
        error_message=request.error_message,
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Bot config for {request.platform} not found")
    return {"success": True}


@router.delete("/config/{platform}")
def delete_bot_config_endpoint(platform: str, db: Session = Depends(get_db)):
    """Delete bot configuration."""
    success = delete_bot_config(db, platform)
    if not success:
        raise HTTPException(status_code=404, detail=f"Bot config for {platform} not found")
    return {"success": True}


# ============================================================================
# Notification Configuration
# ============================================================================

NOTIFICATION_CONFIG_KEY = "bot_notification_config"
DEFAULT_NOTIFICATION_CONFIG = {
    "ai_trader": True,
    "program_trader": True,
    "signal_pools": {}  # pool_id -> bool, default all False
}


class NotificationConfigRequest(BaseModel):
    ai_trader: bool = True
    program_trader: bool = True
    signal_pools: dict = {}  # {"pool_id": true/false}


@router.get("/notification-config")
def get_notification_config(db: Session = Depends(get_db)):
    """Get bot push notification configuration."""
    from database.models import SystemConfig
    config = db.query(SystemConfig).filter(
        SystemConfig.key == NOTIFICATION_CONFIG_KEY
    ).first()
    if not config:
        return {"config": DEFAULT_NOTIFICATION_CONFIG}
    try:
        return {"config": json.loads(config.value)}
    except json.JSONDecodeError:
        return {"config": DEFAULT_NOTIFICATION_CONFIG}


@router.put("/notification-config")
def update_notification_config(
    request: NotificationConfigRequest,
    db: Session = Depends(get_db)
):
    """Update bot push notification configuration."""
    from database.models import SystemConfig
    config_data = {
        "ai_trader": request.ai_trader,
        "program_trader": request.program_trader,
        "signal_pools": request.signal_pools,
    }
    config = db.query(SystemConfig).filter(
        SystemConfig.key == NOTIFICATION_CONFIG_KEY
    ).first()
    if config:
        config.value = json.dumps(config_data)
    else:
        config = SystemConfig(
            key=NOTIFICATION_CONFIG_KEY,
            value=json.dumps(config_data)
        )
        db.add(config)
    db.commit()
    return {"success": True, "config": config_data}


def get_notification_config_dict(db: Session) -> dict:
    """Helper function to get notification config as dict (for use in hooks)."""
    from database.models import SystemConfig
    config = db.query(SystemConfig).filter(
        SystemConfig.key == NOTIFICATION_CONFIG_KEY
    ).first()
    if not config:
        return DEFAULT_NOTIFICATION_CONFIG.copy()
    try:
        return json.loads(config.value)
    except json.JSONDecodeError:
        return DEFAULT_NOTIFICATION_CONFIG.copy()


# ============================================================================
# Tool name mapping for Telegram progress messages
# ============================================================================

TOOL_LABELS = {
    "get_system_overview":    {"en": "Checking system status",      "zh": "正在检查系统状态"},
    "get_wallet_status":      {"en": "Querying wallet balances",    "zh": "正在查询钱包余额"},
    "get_klines":             {"en": "Fetching K-line data",        "zh": "正在获取K线数据"},
    "get_market_regime":      {"en": "Analyzing market regime",     "zh": "正在分析市场状态"},
    "get_market_flow":        {"en": "Analyzing market flow",       "zh": "正在分析资金流向"},
    "get_system_logs":        {"en": "Reading system logs",         "zh": "正在读取系统日志"},
    "get_api_reference":      {"en": "Loading API reference",       "zh": "正在加载API文档"},
    "get_contact_config":     {"en": "Loading contact info",        "zh": "正在加载联系方式"},
    "diagnose_trader_issues": {"en": "Diagnosing trader issues",    "zh": "正在诊断交易员问题"},
    "list_traders":           {"en": "Listing AI traders",          "zh": "正在列出AI交易员"},
    "list_signal_pools":      {"en": "Listing signal pools",        "zh": "正在列出信号池"},
    "list_strategies":        {"en": "Listing strategies",          "zh": "正在列出策略"},
    "save_signal_pool":       {"en": "Saving signal pool",          "zh": "正在保存信号池"},
    "save_prompt":            {"en": "Saving prompt strategy",      "zh": "正在保存提示词策略"},
    "save_program":           {"en": "Saving program strategy",     "zh": "正在保存程序化策略"},
    "create_ai_trader":       {"en": "Creating AI trader",          "zh": "正在创建AI交易员"},
    "update_ai_trader":       {"en": "Updating AI trader",          "zh": "正在更新AI交易员"},
    "bind_prompt_to_trader":  {"en": "Binding prompt to trader",    "zh": "正在绑定提示词到交易员"},
    "bind_program_to_trader": {"en": "Binding program to trader",   "zh": "正在绑定程序到交易员"},
    "update_trader_strategy": {"en": "Updating trader strategy",    "zh": "正在更新交易员策略"},
    "load_skill":             {"en": "Loading skill module",        "zh": "正在加载技能模块"},
    "load_skill_reference":   {"en": "Loading skill reference",     "zh": "正在加载技能参考"},
    "save_memory":            {"en": "Saving memory",               "zh": "正在保存记忆"},
    "call_prompt_ai":         {"en": "Running prompt AI analysis",  "zh": "正在运行提示词AI分析"},
    "call_program_ai":        {"en": "Running program AI analysis", "zh": "正在运行程序AI分析"},
    "call_signal_ai":         {"en": "Running signal AI analysis",  "zh": "正在运行信号AI分析"},
    "call_attribution_ai":    {"en": "Running trade attribution",   "zh": "正在运行交易归因分析"},
}


def _get_ui_language(db_session) -> str:
    """Get UI language from SystemConfig, default to 'en'."""
    from database.models import SystemConfig
    config = db_session.query(SystemConfig).filter(
        SystemConfig.key == "ui_language"
    ).first()
    return config.value if config and config.value in ("en", "zh") else "en"


def _get_tool_label(tool_name: str, lang: str) -> str:
    """Get user-friendly tool label in the specified language."""
    labels = TOOL_LABELS.get(tool_name)
    if labels:
        return labels.get(lang, labels["en"])
    # Fallback: humanize the function name
    return tool_name.replace("_", " ").title()


# ============================================================================
# Telegram-specific endpoints
# ============================================================================

class TelegramConnectRequest(BaseModel):
    bot_token: str
    webhook_base_url: Optional[str] = None  # If not provided, uses request host


def _build_webhook_url(explicit_base: Optional[str], http_request: Request) -> str:
    """Build webhook URL, detecting HTTPS from reverse proxy headers."""
    base_url = explicit_base
    if not base_url:
        base_url = str(http_request.base_url).rstrip("/")
        forwarded_proto = http_request.headers.get("X-Forwarded-Proto")
        if forwarded_proto == "https" and base_url.startswith("http://"):
            base_url = "https://" + base_url[7:]
    return f"{base_url}/api/bot/telegram/webhook"


@router.post("/telegram/connect")
async def connect_telegram_bot(
    request: TelegramConnectRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """Validate token, save config, and setup webhook for Telegram bot."""
    # Validate new token first
    result = await validate_telegram_token(request.bot_token)
    if not result["valid"]:
        raise HTTPException(status_code=400, detail=f"Invalid bot token: {result.get('error')}")

    # If rebinding: remove old webhook with old token (best-effort)
    old_token = get_decrypted_bot_token(db, "telegram")
    if old_token and old_token != request.bot_token:
        try:
            await remove_telegram_webhook(old_token)
        except Exception:
            pass  # Old token may be invalid, that's fine

    # Save config (creates or updates)
    save_bot_config(
        db=db,
        platform="telegram",
        bot_token=request.bot_token,
        bot_username=result.get("username"),
        bot_app_id=result.get("bot_id"),
    )

    # Build webhook URL - detect real protocol from reverse proxy headers
    webhook_url = _build_webhook_url(request.webhook_base_url, http_request)

    webhook_result = await setup_telegram_webhook(request.bot_token, webhook_url)
    if not webhook_result["success"]:
        update_bot_status(db, "telegram", "error", webhook_result.get("error"))
        raise HTTPException(status_code=500, detail=f"Failed to setup webhook: {webhook_result.get('error')}")

    update_bot_status(db, "telegram", "connected")

    # Save webhook URL for auto-restore on restart
    from database.models import BotConfig
    bot_cfg = db.query(BotConfig).filter(BotConfig.platform == "telegram").first()
    if bot_cfg:
        bot_cfg.webhook_url = webhook_url
        db.commit()

    # Create or get the shared Bot conversation (one per user, shared across platforms)
    from database.models import HyperAiConversation
    bot_conv = db.query(HyperAiConversation).filter(
        HyperAiConversation.is_bot_conversation == True
    ).first()
    if not bot_conv:
        bot_conv = HyperAiConversation(
            title="Hyper AI Bot",
            is_bot_conversation=True
        )
        db.add(bot_conv)
        db.commit()
        db.refresh(bot_conv)

    return {
        "success": True,
        "bot_username": result.get("username"),
        "webhook_url": webhook_url,
        "conversation_id": bot_conv.id,
    }


@router.post("/telegram/disconnect")
async def disconnect_telegram_bot(db: Session = Depends(get_db)):
    """Remove webhook and disconnect Telegram bot."""
    token = get_decrypted_bot_token(db, "telegram")
    if not token:
        raise HTTPException(status_code=404, detail="Telegram bot not configured")

    # Remove webhook
    result = await remove_telegram_webhook(token)
    if not result["success"]:
        update_bot_status(db, "telegram", "error", result.get("error"))
        raise HTTPException(status_code=500, detail=f"Failed to remove webhook: {result.get('error')}")

    update_bot_status(db, "telegram", "disconnected")
    return {"success": True}


@router.post("/telegram/retry-webhook")
async def retry_telegram_webhook(
    http_request: Request,
    db: Session = Depends(get_db)
):
    """Retry webhook setup for an already-configured Telegram bot."""
    token = get_decrypted_bot_token(db, "telegram")
    if not token:
        raise HTTPException(status_code=404, detail="Telegram bot not configured")

    webhook_url = _build_webhook_url(None, http_request)
    webhook_result = await setup_telegram_webhook(token, webhook_url)
    if not webhook_result["success"]:
        update_bot_status(db, "telegram", "error", webhook_result.get("error"))
        raise HTTPException(status_code=500, detail=f"Failed to setup webhook: {webhook_result.get('error')}")

    update_bot_status(db, "telegram", "connected")

    # Save webhook URL for auto-restore on restart
    from database.models import BotConfig
    bot_cfg = db.query(BotConfig).filter(BotConfig.platform == "telegram").first()
    if bot_cfg:
        bot_cfg.webhook_url = webhook_url
        db.commit()

    return {"success": True, "webhook_url": webhook_url}


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Receive incoming updates from Telegram.
    This is called by Telegram servers when a user sends a message to the bot.
    """
    try:
        update = await request.json()
    except Exception:
        return {"ok": True}  # Return ok to avoid Telegram retries

    # Extract message if present
    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")
    user = message.get("from", {})

    if not chat_id or not text:
        return {"ok": True}

    # Get bot token
    token = get_decrypted_bot_token(db, "telegram")
    if not token:
        return {"ok": True}

    # Process message with Hyper AI (async, non-blocking)
    asyncio.create_task(
        _process_telegram_message(token, chat_id, text, user, db)
    )
    print(f"[TG-WEBHOOK] Dispatched task for chat_id={chat_id} text={text[:50]}", flush=True)

    return {"ok": True}


async def _process_telegram_message(
    token: str,
    chat_id: int,
    text: str,
    user: dict,
    db: Session
):
    """
    Process a Telegram message through Hyper AI and send response.
    Runs as a background task to avoid blocking the webhook response.
    Reply unicast: response only goes to the originating chat_id.
    """
    from services.hyper_ai_service import (
        get_or_create_conversation,
        stream_chat_response,
    )
    from database.models import HyperAiConversation, BotChatBinding
    from database.connection import SessionLocal
    from sqlalchemy import func

    # Use a new db session for async context
    db_session = SessionLocal()
    try:
        # Record chat binding for push broadcast
        binding = db_session.query(BotChatBinding).filter(
            BotChatBinding.platform == "telegram",
            BotChatBinding.chat_id == str(chat_id)
        ).first()
        if not binding:
            binding = BotChatBinding(
                platform="telegram",
                chat_id=str(chat_id),
                username=user.get("username"),
                display_name=user.get("first_name", "") + " " + user.get("last_name", "")
            )
            db_session.add(binding)
        else:
            binding.last_message_at = func.current_timestamp()
            binding.is_active = True
        db_session.commit()

        # Find the shared Bot conversation (shared across all platforms)
        conv = db_session.query(HyperAiConversation).filter(
            HyperAiConversation.is_bot_conversation == True
        ).first()

        if not conv:
            # Fallback: create one if not exists (shouldn't happen normally)
            conv = HyperAiConversation(
                title="Hyper AI Bot",
                is_bot_conversation=True
            )
            db_session.add(conv)
            db_session.commit()
            db_session.refresh(conv)

        print(f"[TG-PROCESS] Using conv id={conv.id}, processing text: {text[:50]}", flush=True)

        # Determine UI language for tool progress messages
        lang = _get_ui_language(db_session)

        # Collect AI response, pushing tool_call progress to Telegram in real-time
        full_response = ""
        for event in stream_chat_response(db_session, conv.id, text):
            # SSE format: "event: <type>\ndata: <json>\n\n"
            event_type = None
            data_str = None
            for line in event.split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_str = line[6:]

            if not data_str:
                continue
            try:
                data = json.loads(data_str)
                if event_type == "tool_call" and data.get("name"):
                    label = _get_tool_label(data["name"], lang)
                    await send_telegram_message(token, chat_id, f"【🤖Hyper AI】{label}...")
                elif event_type == "content":
                    full_response += data.get("text", "")
                elif event_type == "error":
                    full_response = f"Error: {data.get('message', 'Unknown error')}"
            except json.JSONDecodeError:
                pass

        print(f"[TG-PROCESS] AI response length={len(full_response)}", flush=True)

        # Send response back to Telegram
        if full_response:
            result = await send_telegram_message(token, chat_id, full_response)
            print(f"[TG-PROCESS] Send result={result}", flush=True)

    except Exception as e:
        import logging
        print(f"[TG-PROCESS] ERROR: {type(e).__name__}: {e}", flush=True)
        logging.getLogger(__name__).error(f"Telegram message processing failed: {e}")
        # Try to send error message
        try:
            await send_telegram_message(token, chat_id, "Sorry, an error occurred while processing your message.")
        except Exception:
            pass
    finally:
        db_session.close()
