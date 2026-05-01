from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from .composer import Composer, LLMClient
from .context_store import ContextStore, ConversationMeta
from .reply_handler import ReplyHandler

load_dotenv()

app = FastAPI()
START_TIME = time.time()

store = ContextStore()
composer = Composer(LLMClient())
reply_handler = ReplyHandler(store, composer)


class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: int


@app.get("/v1/healthz")
async def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": store.count_by_scope(),
    }


@app.get("/v1/metadata")
async def metadata() -> Dict[str, Any]:
    team_name = os.getenv("TEAM_NAME", "Your Team")
    team_members = os.getenv("TEAM_MEMBERS", "Your Name").split(",")
    model_name = os.getenv("MODEL_NAME", os.getenv("LLM_MODEL", "")) or "unknown"
    approach = os.getenv(
        "APPROACH",
        "LLM-powered composer with trigger-aware prompts, deterministic settings, and reply handling.",
    )
    contact_email = os.getenv("CONTACT_EMAIL", "you@example.com")
    version = os.getenv("BOT_VERSION", "0.1.0")
    submitted_at = os.getenv("SUBMITTED_AT", "2026-05-02T00:00:00Z")

    return {
        "team_name": team_name,
        "team_members": [m.strip() for m in team_members if m.strip()],
        "model": model_name,
        "approach": approach,
        "contact_email": contact_email,
        "version": version,
        "submitted_at": submitted_at,
    }


@app.post("/v1/context")
async def push_context(body: ContextBody) -> Dict[str, Any]:
    accepted, reason, current_version = store.push_context(
        body.scope, body.context_id, body.version, body.payload
    )
    if not accepted:
        if reason == "invalid_scope":
            raise HTTPException(status_code=400, detail="invalid_scope")
        if reason == "stale_version":
            return JSONResponse(
                status_code=409,
                content={
                    "accepted": False,
                    "reason": "stale_version",
                    "current_version": current_version,
                },
            )

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/tick")
async def tick(body: TickBody) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for trigger_id in body.available_triggers:
        if len(candidates) >= 20:
            break

        trigger = store.get_context("trigger", trigger_id)
        if not trigger:
            continue

        suppression_key = trigger.get("suppression_key", "")
        if store.is_suppressed(suppression_key):
            continue

        merchant_id = trigger.get("merchant_id")
        if not merchant_id:
            continue

        merchant = store.get_context("merchant", merchant_id)
        if not merchant:
            continue

        category = store.get_context("category", merchant.get("category_slug", ""))
        if not category:
            continue

        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = store.get_context("customer", customer_id)
            if not customer:
                continue

        candidates.append(
            {
                "trigger_id": trigger_id,
                "trigger": trigger,
                "suppression_key": suppression_key,
                "merchant_id": merchant_id,
                "merchant": merchant,
                "category": category,
                "customer_id": customer_id,
                "customer": customer,
            }
        )

    if not candidates:
        return {"actions": actions}

    worker_count = tick_compose_workers(len(candidates))
    if worker_count == 1:
        results = [compose_candidate(candidate) for candidate in candidates]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(compose_candidate, candidates))

    for candidate, result in zip(candidates, results):
        if len(actions) >= 20:
            break
        if store.is_suppressed(candidate["suppression_key"]):
            continue
        if not result or not result.get("body"):
            continue

        trigger = candidate["trigger"]
        merchant = candidate["merchant"]
        customer = candidate["customer"]
        merchant_id = candidate["merchant_id"]
        customer_id = candidate["customer_id"]
        trigger_id = candidate["trigger_id"]
        suppression_key = candidate["suppression_key"]
        send_as = result.get("send_as", "vera")
        conversation_id = self_conversation_id(merchant_id, trigger_id)
        store.set_conversation_meta(
            conversation_id,
            ConversationMeta(
                merchant_id=merchant_id,
                customer_id=customer_id,
                trigger_id=trigger_id,
                send_as=send_as,
            ),
        )

        body_text = result["body"]
        store.record_sent(conversation_id, body_text)
        store.add_suppression(suppression_key)
        store.append_turn(conversation_id, "vera", body_text, body.now)

        template_name, template_params = build_template(result, merchant, customer)

        actions.append(
            {
                "conversation_id": conversation_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": send_as,
                "trigger_id": trigger_id,
                "template_name": template_name,
                "template_params": template_params,
                "body": body_text,
                "cta": result.get("cta", "open_ended"),
                "suppression_key": suppression_key,
                "rationale": result.get("rationale", "Composed from context."),
            }
        )

    return {"actions": actions}


def compose_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return composer.compose(
        candidate["category"],
        candidate["merchant"],
        candidate["trigger"],
        candidate["customer"],
    )


def tick_compose_workers(total: int) -> int:
    try:
        configured = int(os.getenv("TICK_COMPOSE_WORKERS", "5"))
    except ValueError:
        configured = 5
    return max(1, min(total, configured))


@app.post("/v1/reply")
async def reply(body: ReplyBody) -> Dict[str, Any]:
    return reply_handler.handle_reply(body.dict())


def self_conversation_id(merchant_id: str, trigger_id: str) -> str:
    safe_trigger = trigger_id.replace(":", "_")
    return f"conv_{merchant_id}_{safe_trigger}"


def build_template(result: Dict[str, Any], merchant: Dict[str, Any], customer: Optional[Dict[str, Any]]) -> tuple[str, List[str]]:
    template_name = "vera_generic_v1"
    if result.get("send_as") == "merchant_on_behalf":
        template_name = "merchant_on_behalf_generic_v1"

    body = result.get("body", "")
    merchant_name = merchant.get("identity", {}).get("name", "Merchant")
    owner = merchant.get("identity", {}).get("owner_first_name") or merchant_name

    if customer:
        customer_name = customer.get("identity", {}).get("name", "Customer")
        template_params = [customer_name, owner, truncate(body, 200)]
    else:
        template_params = [owner, truncate(body, 200)]

    return template_name, template_params


def truncate(text: str, max_len: int) -> str:
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
