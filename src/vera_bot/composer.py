from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from .prompts import BASE_SYSTEM_PROMPT, build_prompt, format_context_summary


VALID_CTA = {"binary_yes_no", "binary_confirm_cancel", "open_ended", "none", "multi_choice_slot"}


class LLMClient:
    def __init__(self) -> None:
        self.provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        self.api_key = (os.getenv("LLM_API_KEY") or "").strip()
        self.model = (os.getenv("LLM_MODEL") or "").strip()
        self.base_url = (os.getenv("LLM_BASE_URL") or "").strip()
        self.timeout = float(os.getenv("LLM_TIMEOUT", "8"))

    def is_configured(self) -> bool:
        if not self.provider:
            return False
        if self.provider in {"gemini", "openai", "anthropic", "deepseek", "groq", "openrouter", "openai-compatible"}:
            return bool(self.api_key)
        return bool(self.api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "gemini":
            return self._complete_gemini(system_prompt, user_prompt)
        if self.provider == "anthropic":
            return self._complete_anthropic(system_prompt, user_prompt)
        return self._complete_openai(system_prompt, user_prompt)

    def _complete_openai(self, system_prompt: str, user_prompt: str) -> str:
        default_bases = {
            "deepseek": "https://api.deepseek.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        base = self.base_url or default_bases.get(self.provider, "https://api.openai.com/v1")
        url = base.rstrip("/") + "/chat/completions"
        default_models = {
            "deepseek": "deepseek-chat",
            "groq": "llama-3.1-8b-instant",
            "openrouter": "anthropic/claude-3-haiku",
        }
        model = self.model or default_models.get(self.provider, "gpt-4o-mini")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 800,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _complete_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        model = self.model or "claude-3-haiku-20240307"
        payload = {
            "model": model,
            "max_tokens": 800,
            "temperature": 0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        chunks = data.get("content", [])
        return "".join(part.get("text", "") for part in chunks if part.get("type") == "text")

    def _complete_gemini(self, system_prompt: str, user_prompt: str) -> str:
        model = self.model or "gemini-2.0-flash"
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            + model
            + ":generateContent?key="
            + self.api_key
        )
        text = system_prompt + "\n\n" + user_prompt
        generation_config: Dict[str, Any] = {"maxOutputTokens": 800}
        if model.startswith("gemini-3"):
            generation_config["thinkingConfig"] = {
                "thinkingLevel": os.getenv("GEMINI_THINKING_LEVEL", "low")
            }
        else:
            generation_config["temperature"] = 0
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": generation_config,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


class Composer:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or LLMClient()

    def compose(
        self,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        facts = self._collect_facts(category, merchant, trigger, customer)
        prompt = build_prompt(category, merchant, trigger, customer, facts)

        if self._llm_compose_enabled() and self.llm.is_configured():
            try:
                raw = self.llm.complete(BASE_SYSTEM_PROMPT, prompt)
                parsed = self._extract_json(raw)
                if parsed:
                    result = self._normalize_result(parsed, category, merchant, trigger, customer)
                    if self._is_valid(result, category, merchant, trigger, customer):
                        return result
            except Exception:
                pass

        return self._fallback_compose(category, merchant, trigger, customer)

    def _llm_compose_enabled(self) -> bool:
        raw_value = os.getenv("LLM_COMPOSE_ENABLED")
        if raw_value is None or not raw_value.strip():
            return True
        value = raw_value.strip().lower()
        return value in {"1", "true", "yes", "on"}

    def reply(
        self,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
        conversation: List[Dict[str, Any]],
        last_message: str,
    ) -> Dict[str, Any]:
        facts = self._collect_facts(category, merchant, trigger, customer)
        convo_text = json.dumps(conversation[-6:], ensure_ascii=False, indent=2)
        context_text = format_context_summary(category, merchant, trigger, customer, facts)
        user_prompt = (
            "Continue the conversation in Vera's voice. Reply to the last merchant message with a concrete next step. "
            "Return JSON with keys: body, cta, rationale, send_as.\n\n"
            "Context summary (use only these facts):\n" + context_text + "\n\n"
            "Conversation so far:\n" + convo_text + "\n\n"
            "Last merchant message:\n" + (last_message or "") + "\n\n"
            "Return ONLY valid JSON."
        )

        if self.llm.is_configured():
            try:
                raw = self.llm.complete(BASE_SYSTEM_PROMPT, user_prompt)
                parsed = self._extract_json(raw)
                if parsed:
                    result = self._normalize_result(parsed, category, merchant, trigger, customer)
                    if self._is_valid(result, category, merchant, trigger, customer):
                        return result
            except Exception:
                pass

        body = self._fallback_reply(category, merchant, trigger, customer, last_message)
        return {
            "body": body,
            "cta": "binary_yes_no",
            "send_as": "merchant_on_behalf" if customer else "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Fallback reply when LLM is unavailable or invalid.",
        }

    def _collect_facts(
        self,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
    ) -> List[str]:
        facts: List[str] = []
        identity = merchant.get("identity", {})
        owner = identity.get("owner_first_name")
        if owner:
            facts.append(f"Owner first name: {owner}")
        if identity.get("name"):
            facts.append(f"Merchant name: {identity['name']}")
        if identity.get("locality"):
            facts.append(f"Locality: {identity['locality']}")
        if identity.get("city"):
            facts.append(f"City: {identity['city']}")

        perf = merchant.get("performance", {})
        for key in ("views", "calls", "directions", "ctr"):
            if key in perf:
                facts.append(f"Performance {key}: {perf[key]}")
        delta = perf.get("delta_7d", {})
        for key, val in delta.items():
            facts.append(f"Delta 7d {key}: {val}")

        offers = merchant.get("offers", [])
        for offer in offers:
            title = offer.get("title")
            status = offer.get("status")
            if title:
                facts.append(f"Offer ({status}): {title}")

        digest = category.get("digest", [])
        for item in digest:
            title = item.get("title")
            source = item.get("source")
            if title and source:
                facts.append(f"Digest: {title} ({source})")

        if trigger.get("kind"):
            facts.append(f"Trigger kind: {trigger['kind']}")
        if trigger.get("urgency") is not None:
            facts.append(f"Trigger urgency: {trigger.get('urgency')}")

        if customer:
            cust_id = customer.get("customer_id")
            if cust_id:
                facts.append(f"Customer id: {cust_id}")
            cust_name = customer.get("identity", {}).get("name")
            if cust_name:
                facts.append(f"Customer name: {cust_name}")
            state = customer.get("state")
            if state:
                facts.append(f"Customer state: {state}")

        return facts

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except Exception:
                return None
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _normalize_result(
        self,
        result: Dict[str, Any],
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        body = (result.get("body") or "").strip()
        cta = (result.get("cta") or "").strip()
        rationale = (result.get("rationale") or "").strip()
        send_as = (result.get("send_as") or "").strip()

        if cta not in VALID_CTA:
            cta = "open_ended"
        if customer:
            send_as = "merchant_on_behalf"
        else:
            send_as = "vera"
        if not rationale:
            rationale = "Composed from category, merchant, and trigger context."

        return {
            "body": body,
            "cta": cta,
            "send_as": send_as,
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": rationale,
        }

    def _is_valid(
        self,
        result: Dict[str, Any],
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
    ) -> bool:
        body = result.get("body", "")
        if not body:
            return False
        if re.search(r"https?://", body):
            return False

        taboos = []
        voice = category.get("voice", {})
        if isinstance(voice, dict):
            taboos.extend(voice.get("vocab_taboo", []) or [])
            taboos.extend(voice.get("taboos", []) or [])
        for taboo in taboos:
            if taboo and taboo.lower() in body.lower():
                return False

        return True

    def _fallback_compose(
        self,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        kind = trigger.get("kind", "unknown")
        owner = self._owner_name(merchant)
        body = ""
        cta = "open_ended"

        if kind == "research_digest":
            item = self._select_digest_item(category, trigger)
            if item:
                title = item.get("title", "")
                source = item.get("source", "")
                number = self._digest_number(item)
                body = (
                    f"{owner}, {source}: {title}{number}. "
                    "Want me to pull the abstract and draft a short message?"
                )
            else:
                category_name = category.get("display_name") or category.get("slug", "your category")
                body = f"{owner}, I found a sourced {category_name} research update for your profile. Want the 3-line summary and draft?"
            cta = "binary_yes_no"

        elif kind in {"regulation_change", "cde_opportunity"}:
            item = self._select_digest_item(category, trigger)
            payload = trigger.get("payload", {}) or {}
            deadline = payload.get("deadline_iso") or payload.get("date")
            if item:
                source = item.get("source", "category source")
                title = item.get("title", "new update")
                date_text = f" Deadline: {deadline}." if deadline else ""
                body = f"{owner}, {source}: {title}.{date_text} Want me to make a 5-point action checklist?"
            else:
                body = f"{owner}, there is an external update for your category. Want me to make the action checklist?"
            cta = "binary_yes_no"

        elif kind in {"perf_dip", "perf_spike"}:
            perf = merchant.get("performance", {})
            views = perf.get("views")
            calls = perf.get("calls")
            payload = trigger.get("payload", {}) or {}
            metric = payload.get("metric")
            delta = self._format_pct(payload.get("delta_pct"))
            cta = "binary_yes_no"
            metrics = []
            if views is not None:
                metrics.append(f"views {views}")
            if calls is not None:
                metrics.append(f"calls {calls}")
            metric_text = ", ".join(metrics) if metrics else "recent performance"
            if kind == "perf_dip":
                dip_text = f"{metric} {delta}" if metric and delta else metric_text
                body = f"{owner}, {dip_text} is down in the latest window ({metric_text}). Want me to draft a recovery post?"
            else:
                spike_text = f"{metric} {delta}" if metric and delta else metric_text
                body = f"{owner}, nice spike: {spike_text} ({metric_text}). Want me to draft a follow-up post to keep momentum?"

        elif kind == "seasonal_perf_dip":
            payload = trigger.get("payload", {}) or {}
            metric = payload.get("metric", "views")
            delta = self._format_pct(payload.get("delta_pct")) or "recent"
            note = payload.get("season_note", "seasonal window")
            cta = "binary_yes_no"
            body = f"{owner}, {metric} is {delta} in this {note}. Expected, but worth acting. Want a 2-line acquisition post?"

        elif kind in {"renewal_due", "winback_eligible", "dormant_with_vera"}:
            payload = trigger.get("payload", {}) or {}
            days = payload.get("days_remaining") or payload.get("days_since_expiry") or payload.get("days_since_last_merchant_message")
            active_offer = self._first_active_offer(merchant)
            cta = "binary_yes_no"
            if kind == "renewal_due":
                body = f"{owner}, Pro has {days} days left. Pausing now risks losing profile momentum. Want me to prep the renewal note?"
            elif kind == "winback_eligible":
                body = f"{owner}, {days} days since expiry and the dip signal is still active. Want a comeback plan using {active_offer or 'a fresh offer'}?"
            else:
                merchant_name = merchant.get("identity", {}).get("name", "your profile")
                body = f"{owner}, it has been {days} days since we last acted. I found one useful signal for {merchant_name}. Want it?"

        elif kind == "recall_due" and customer:
            cust_name = customer.get("identity", {}).get("name", "there")
            payload = trigger.get("payload", {}) or {}
            pref = self._slot_text(payload) or customer.get("preferences", {}).get("preferred_slots", "your preferred time")
            service = str(payload.get("service_due", "recall")).replace("_", " ")
            cta = "multi_choice_slot" if self._slot_text(payload) else "open_ended"
            clinic_name = merchant.get("identity", {}).get("name", "the clinic")
            body = (
                f"Hi {cust_name}, {clinic_name} here. Your {service} is due. "
                f"We can do {pref}. Which slot works?"
            )

        elif kind in {"customer_lapsed_soft", "customer_lapsed_hard"} and customer:
            cust_name = customer.get("identity", {}).get("name", "there")
            payload = trigger.get("payload", {}) or {}
            days = payload.get("days_since_last_visit")
            focus = payload.get("previous_focus")
            offer = self._first_active_offer(merchant)
            cta = "binary_yes_no"
            body = f"Hi {cust_name}, checking in from {owner}'s team"
            if days:
                body += f" after {days} days"
            if focus:
                body += f" on your {focus} goal"
            body += f". Want to restart with {offer or 'a simple trial'} this week? Reply YES."

        elif kind == "appointment_tomorrow" and customer:
            cust_name = customer.get("identity", {}).get("name", "there")
            cta = "binary_confirm_cancel"
            body = f"Hi {cust_name}, reminder for your appointment tomorrow. Reply CONFIRM to keep it or CANCEL to reschedule."

        elif kind == "chronic_refill_due" and customer:
            cust_name = customer.get("identity", {}).get("name", "there")
            payload = trigger.get("payload", {}) or {}
            meds = ", ".join(payload.get("molecule_list", [])[:3])
            stock_date = payload.get("stock_runs_out_iso")
            cta = "binary_confirm_cancel"
            body = f"Hi {cust_name}, your {meds} refill looks due before {stock_date}. Reply CONFIRM for delivery or CANCEL if not needed."

        elif kind in {"trial_followup", "wedding_package_followup"} and customer:
            cust_name = customer.get("identity", {}).get("name", "there")
            payload = trigger.get("payload", {}) or {}
            cta = "binary_yes_no"
            slot_text = self._slot_text(payload)
            if kind == "trial_followup":
                body = f"Hi {cust_name}, thanks for trying the session. Next option is {slot_text or 'this week'}. Should we hold it for you?"
            else:
                body = f"Hi {cust_name}, your wedding date {payload.get('wedding_date', 'is coming up')} is on our prep tracker. Want to plan the next skin/hair step?"

        elif kind in {"review_theme_emerged", "competitor_opened", "festival_upcoming", "ipl_match_today", "milestone_reached", "curious_ask_due", "active_planning_intent", "supply_alert", "category_seasonal", "gbp_unverified"}:
            cta = "binary_yes_no"
            body = self._fallback_merchant_signal(owner, merchant, trigger)

        else:
            body = f"{owner}, I have a quick update for you based on current signals. Want the details?"
            cta = "binary_yes_no"

        return {
            "body": body,
            "cta": cta,
            "send_as": "merchant_on_behalf" if customer else "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Deterministic fallback composition.",
        }

    def _fallback_reply(
        self,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
        last_message: str,
    ) -> str:
        owner = self._owner_name(merchant)
        kind = trigger.get("kind", "unknown")
        merchant_name = merchant.get("identity", {}).get("name", "your business")
        if kind == "research_digest":
            item = self._select_digest_item(category, trigger)
            if item and item.get("title"):
                return f"Thanks {owner}. I can draft a message about {item['title']}. Reply YES and I'll prepare it."
            return f"Thanks {owner}. I can draft a message from the latest research. Reply YES and I'll prepare it."
        if kind in {"perf_dip", "perf_spike", "seasonal_perf_dip"}:
            return f"Thanks {owner}. I can draft a performance update for {merchant_name}. Reply YES and I'll prepare it."
        if kind == "recall_due" and customer:
            cust_name = customer.get("identity", {}).get("name", "there")
            return f"Thanks {owner}. I can draft the recall note for {cust_name}. Reply YES and I'll prepare it."
        if kind == "active_planning_intent":
            topic = (trigger.get("payload", {}) or {}).get("intent_topic", "this idea").replace("_", " ")
            return f"Thanks {owner}. I can turn {topic} into a ready-to-send post and offer. Reply YES and I'll prepare it."
        if last_message:
            snippet = last_message.strip().splitlines()[0][:80]
            return f"Thanks {owner}. I can respond to '{snippet}'. Reply YES and I'll prepare it."
        return f"Thanks {owner}. I can draft the next step for {merchant_name}. Reply YES and I'll prepare it."

    def _owner_name(self, merchant: Dict[str, Any]) -> str:
        identity = merchant.get("identity", {})
        owner = identity.get("owner_first_name")
        if owner:
            return owner
        return identity.get("name", "there")

    def _first_active_offer(self, merchant: Dict[str, Any]) -> Optional[str]:
        for offer in merchant.get("offers", []) or []:
            if offer.get("status") == "active" and offer.get("title"):
                return offer.get("title")
        return None

    def _select_digest_item(self, category: Dict[str, Any], trigger: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        digest = category.get("digest", []) or []
        payload = trigger.get("payload", {}) or {}
        target_id = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
        if target_id:
            for item in digest:
                if item.get("id") == target_id:
                    return item
        return digest[0] if digest else None

    def _digest_number(self, item: Dict[str, Any]) -> str:
        if item.get("trial_n") is not None:
            return f" ({item.get('trial_n')} patients)"
        if item.get("credits") is not None:
            return f" ({item.get('credits')} credits)"
        summary = item.get("summary", "")
        match = re.search(r"\b\d+(?:\.\d+)?%?\b", summary)
        if match:
            return f" ({match.group(0)})"
        return ""

    def _format_pct(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            pct = float(value) * 100
        except (TypeError, ValueError):
            return str(value)
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.0f}%"

    def _slot_text(self, payload: Dict[str, Any]) -> str:
        for key in ("available_slots", "next_session_options"):
            slots = payload.get(key) or []
            labels = [slot.get("label") for slot in slots if isinstance(slot, dict) and slot.get("label")]
            if labels:
                return " or ".join(labels[:2])
        return ""

    def _fallback_merchant_signal(self, owner: str, merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
        kind = trigger.get("kind", "")
        payload = trigger.get("payload", {}) or {}
        if kind == "review_theme_emerged":
            theme = payload.get("theme", "a review theme").replace("_", " ")
            count = payload.get("occurrences_30d")
            return f"{owner}, {theme} appeared in reviews {count} times in 30 days. Want me to draft a calm public reply?"
        if kind == "competitor_opened":
            return (
                f"{owner}, {payload.get('competitor_name', 'a competitor')} opened {payload.get('distance_km', 'nearby')} km away "
                f"with {payload.get('their_offer', 'a launch offer')}. Want a sharper counter-post?"
            )
        if kind == "festival_upcoming":
            return f"{owner}, {payload.get('festival', 'the festival')} is coming up. Want me to draft a seasonal offer post?"
        if kind == "ipl_match_today":
            return f"{owner}, {payload.get('match', 'match night')} at {payload.get('match_time_iso', 'tonight')} can lift local demand. Want a same-day post?"
        if kind == "milestone_reached":
            return f"{owner}, you are near {payload.get('milestone_value', 'the next')} {payload.get('metric', 'milestone')}. Want a review push message?"
        if kind == "curious_ask_due":
            return f"{owner}, quick question: which service is most in demand this week? Reply one name and I will draft a post."
        if kind == "active_planning_intent":
            topic = str(payload.get("intent_topic", "this idea")).replace("_", " ")
            return f"{owner}, I can turn {topic} into a package, price anchor, and GBP post. Want the draft?"
        if kind == "supply_alert":
            batches = ", ".join(payload.get("affected_batches", [])[:2])
            return f"{owner}, supply alert for {payload.get('molecule', 'medicine')} batches {batches}. Want the affected-customer filter?"
        if kind == "category_seasonal":
            trends = ", ".join(payload.get("trends", [])[:3])
            return f"{owner}, seasonal demand shift: {trends}. Want a shelf and post checklist?"
        if kind == "gbp_unverified":
            uplift = self._format_pct(payload.get("estimated_uplift_pct"))
            return f"{owner}, your GBP is unverified; verified profiles can unlock about {uplift} more visibility here. Want the verification steps?"
        return f"{owner}, I found a specific signal for {merchant.get('identity', {}).get('name', 'your business')}. Want the draft?"
