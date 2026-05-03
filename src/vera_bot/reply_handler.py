from __future__ import annotations

from typing import Any, Dict, List, Optional

from .composer import Composer
from .context_store import ContextStore

AUTO_REPLY_SIGNALS = [
    "thank you for contacting",
    "our team will respond",
    "automated message",
    "automated assistant",
    "we will get back to you",
    "we will respond shortly",
    "auto reply",
    "auto-reply",
]

INTENT_COMMIT_PHRASES = [
    "let's do it",
    "lets do it",
    "let us do it",
    "go ahead",
    "ok do it",
    "okay do it",
    "sounds good",
    "proceed",
    "what's next",
    "what is next",
    "whats next",
    "yes",
    "yes please",
    "haan",
    "haan karo",
    "kar do",
    "chalo",
    "theek hai",
    "sure",
    "do it",
]

HOSTILE_SIGNALS = [
    "stop messaging",
    "not interested",
    "spam",
    "useless",
    "do not contact",
    "dont contact",
    "don't contact",
    "block",
    "remove me",
    "unsubscribe",
    "stop it",
]


class ReplyHandler:
    def __init__(self, store: ContextStore, composer: Composer) -> None:
        self.store = store
        self.composer = composer

    def handle_reply(self, reply: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = reply.get("conversation_id")
        message = reply.get("message", "")
        received_at = reply.get("received_at")
        from_role = reply.get("from_role", "merchant")
        request_merchant_id = reply.get("merchant_id")
        request_customer_id = reply.get("customer_id")

        if not conversation_id:
            return {"action": "end", "rationale": "Missing conversation_id."}

        if self.store.is_conversation_ended(conversation_id):
            return {"action": "end", "rationale": "Conversation already closed."}

        # --- DETECT PATTERNS FIRST (before meta check) ---
        conversation = self.store.get_conversation(conversation_id)
        is_auto_reply = self._detect_auto_reply(conversation, message)

        # Use the actual from_role instead of always "merchant"
        self.store.append_turn(conversation_id, from_role, message, received_at)

        # 1) Hostile detection — always runs
        if self._detect_hostile(message):
            body = "Sorry about that \u2014 I won\u2019t message again. If you ever want to restart, just send \u2018Hi Vera\u2019. \U0001f64f"
            if self.store.is_repeat(conversation_id, body):
                self.store.end_conversation(conversation_id)
                return {"action": "end", "rationale": "Merchant hostile; already apologized."}
            self.store.record_sent(conversation_id, body)
            self.store.end_conversation(conversation_id)
            return {
                "action": "send",
                "body": body,
                "cta": "none",
                "rationale": "Acknowledged opt-out gracefully and ended conversation.",
            }

        # 2) Auto-reply detection — always runs (merchant only)
        if from_role == "merchant" and is_auto_reply:
            count = self.store.bump_auto_reply_count(self._auto_reply_key(reply, conversation_id, message))
            if count == 1:
                return {
                    "action": "send",
                    "body": "Looks like an auto-reply. When the owner sees this, just reply YES to continue.",
                    "cta": "binary_yes_no",
                    "rationale": "Detected auto-reply; one gentle prompt for the owner.",
                }
            if count == 2:
                return {
                    "action": "wait",
                    "wait_seconds": 86400,
                    "rationale": "Auto-reply repeated; waiting 24h before retry.",
                }
            self.store.end_conversation(conversation_id)
            return {"action": "end", "rationale": "Auto-reply repeated 3+; ending conversation."}

        if from_role == "merchant":
            self.store.reset_auto_reply_count(conversation_id)

        # --- RESOLVE CONTEXT ---
        meta = self.store.get_conversation_meta(conversation_id)
        merchant_id = meta.merchant_id or request_merchant_id
        trigger_id = meta.trigger_id

        merchant = self.store.get_context("merchant", merchant_id) if merchant_id else None
        trigger = self.store.get_context("trigger", trigger_id) if trigger_id else None
        category = None
        if merchant:
            category = self.store.get_context("category", merchant.get("category_slug", ""))

        customer = None
        customer_id_resolved = meta.customer_id or request_customer_id
        if customer_id_resolved:
            customer = self.store.get_context("customer", customer_id_resolved)

        # --- CUSTOMER REPLY BRANCH ---
        # When from_role is "customer", we reply TO the customer (not the merchant)
        if from_role == "customer":
            return self._handle_customer_reply(
                conversation_id, message, merchant, trigger, category, customer, customer_id_resolved
            )

        # --- MERCHANT REPLY BRANCH ---
        # 3) Intent commit detection — runs even without full context
        if self._detect_intent_commit(message):
            if merchant and trigger and category:
                body = self._build_intent_commit_response(category, merchant, trigger, customer)
            else:
                owner = self._get_owner_name(merchant)
                body = f"Great {owner}, drafting the next step now. I'll send you the draft shortly \u2014 reply CONFIRM to proceed."
            if self.store.is_repeat(conversation_id, body):
                return {"action": "end", "rationale": "Avoiding repeated response."}
            self.store.record_sent(conversation_id, body)
            return {
                "action": "send",
                "body": body,
                "cta": "binary_confirm_cancel",
                "rationale": "Merchant confirmed intent; switching to action mode.",
            }

        # --- FULL LLM REPLY (merchant path, requires context) ---
        if not merchant or not trigger or not category:
            # Even without full context, try a generic helpful reply
            owner = self._get_owner_name(merchant)
            body = f"Thanks {owner}, I'll look into that. Give me a moment to check the details and get back to you."
            if self.store.is_repeat(conversation_id, body):
                return {"action": "end", "rationale": "Missing context and would repeat."}
            self.store.record_sent(conversation_id, body)
            return {
                "action": "send",
                "body": body,
                "cta": "open_ended",
                "rationale": "Partial context; generic helpful follow-up.",
            }

        conversation = self.store.get_conversation(conversation_id)
        result = self.composer.reply(category, merchant, trigger, customer, conversation, message)
        body = result.get("body", "")
        if not body or self.store.is_repeat(conversation_id, body):
            # Fallback if LLM returned empty or repeat
            owner = self._get_owner_name(merchant)
            body = f"Got it {owner}. Let me work on that and get back to you with specifics."
            if self.store.is_repeat(conversation_id, body):
                return {"action": "end", "rationale": "Avoiding repeated response."}

        self.store.record_sent(conversation_id, body)
        return {
            "action": "send",
            "body": body,
            "cta": result.get("cta", "open_ended"),
            "rationale": result.get("rationale", "Follow-up reply."),
        }

    def _handle_customer_reply(
        self,
        conversation_id: str,
        message: str,
        merchant: Optional[Dict[str, Any]],
        trigger: Optional[Dict[str, Any]],
        category: Optional[Dict[str, Any]],
        customer: Optional[Dict[str, Any]],
        customer_id: Optional[str],
    ) -> Dict[str, Any]:
        """Handle a reply from a customer — reply should address the customer, not the merchant."""
        cust_name = "there"
        if customer:
            cust_name = customer.get("identity", {}).get("name", "there")

        owner = self._get_owner_name(merchant)
        clinic_name = "the clinic"
        if merchant:
            clinic_name = merchant.get("identity", {}).get("name", "the clinic")

        # Try LLM reply with customer-specific prompt
        if merchant and trigger and category and self.composer.llm.is_configured():
            try:
                from .prompts import BASE_SYSTEM_PROMPT, format_context_summary
                facts = self.composer._collect_facts(category, merchant, trigger, customer)
                conversation = self.store.get_conversation(conversation_id)
                convo_text = __import__('json').dumps(conversation[-6:], ensure_ascii=False, indent=2)
                context_text = format_context_summary(category, merchant, trigger, customer, facts)
                user_prompt = (
                    f"You are replying TO THE CUSTOMER (named {cust_name}), on behalf of the merchant ({clinic_name}). "
                    f"send_as must be merchant_on_behalf. Address the customer by name ({cust_name}), NOT the merchant. "
                    "Return JSON with keys: body, cta, rationale, send_as.\n\n"
                    "Context summary:\n" + context_text + "\n\n"
                    "Conversation so far:\n" + convo_text + "\n\n"
                    "Last customer message:\n" + (message or "") + "\n\n"
                    "Return ONLY valid JSON."
                )
                raw = self.composer.llm.complete(BASE_SYSTEM_PROMPT, user_prompt)
                parsed = self.composer._extract_json(raw)
                if parsed and parsed.get("body"):
                    body = parsed["body"].strip()
                    cta = parsed.get("cta", "open_ended")
                    if cta not in {"binary_yes_no", "binary_confirm_cancel", "open_ended", "none", "multi_choice_slot"}:
                        cta = "open_ended"
                    if not self.store.is_repeat(conversation_id, body):
                        self.store.record_sent(conversation_id, body)
                        return {
                            "action": "send",
                            "body": body,
                            "cta": cta,
                            "send_as": "merchant_on_behalf",
                            "rationale": parsed.get("rationale", "Customer reply via LLM."),
                        }
            except Exception:
                pass

        # Fallback: deterministic customer-addressed reply
        body = f"Hi {cust_name}, thanks for your reply! We'll get that sorted for you at {clinic_name}. We'll confirm the details shortly."
        if self.store.is_repeat(conversation_id, body):
            return {"action": "end", "rationale": "Would repeat customer reply."}
        self.store.record_sent(conversation_id, body)
        return {
            "action": "send",
            "body": body,
            "cta": "open_ended",
            "send_as": "merchant_on_behalf",
            "rationale": "Customer reply — addressing customer on behalf of merchant.",
        }

    def _detect_auto_reply(self, conversation: List[Dict[str, Any]], message: str) -> bool:
        msg = (message or "").lower().strip()
        if any(sig in msg for sig in AUTO_REPLY_SIGNALS):
            return True
        previous = [
            (t.get("message", "").lower().strip())
            for t in conversation
            if t.get("from_role") == "merchant"
        ]
        return msg in previous

    def _auto_reply_key(self, reply: Dict[str, Any], conversation_id: str, message: str) -> str:
        merchant_id = reply.get("merchant_id") or "unknown_merchant"
        normalized = " ".join((message or "").lower().split())
        if merchant_id != "unknown_merchant" and normalized:
            return f"merchant_auto:{merchant_id}:{normalized}"
        return conversation_id

    def _detect_intent_commit(self, message: str) -> bool:
        msg = (message or "").lower().strip()
        # Remove punctuation for cleaner matching
        msg_clean = msg.replace(".", "").replace("!", "").replace("?", "").replace(",", "").strip()
        return any(phrase in msg_clean for phrase in INTENT_COMMIT_PHRASES)

    def _detect_hostile(self, message: str) -> bool:
        msg = (message or "").lower().strip()
        return any(sig in msg for sig in HOSTILE_SIGNALS)

    def _get_owner_name(self, merchant: Optional[Dict[str, Any]]) -> str:
        if not merchant:
            return "there"
        identity = merchant.get("identity", {})
        return identity.get("owner_first_name") or identity.get("name", "there")

    def _build_intent_commit_response(
        self,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
    ) -> str:
        owner = self._get_owner_name(merchant)
        kind = trigger.get("kind", "unknown")

        if kind == "research_digest":
            item = self._select_digest_item(category, trigger)
            if item and item.get("title"):
                return (
                    f"Great {owner}, drafting the abstract summary for '{item['title']}' now. "
                    "Reply CONFIRM and I'll send the draft message too."
                )
            return f"Great {owner}, drafting the abstract summary now. Reply CONFIRM to proceed."

        if kind == "recall_due" and customer:
            cust_name = customer.get("identity", {}).get("name", "the customer")
            return (
                f"Great {owner}, drafting the recall message for {cust_name} now. "
                "Reply CONFIRM to send."
            )

        if kind in {"perf_dip", "perf_spike", "seasonal_perf_dip"}:
            return (
                f"Great {owner}, drafting a post based on the latest performance data. "
                "Reply CONFIRM to proceed."
            )

        if kind in {"active_planning_intent", "curious_ask_due"}:
            topic = (trigger.get("payload", {}) or {}).get("intent_topic") or (trigger.get("payload", {}) or {}).get("ask_template") or "this idea"
            topic = str(topic).replace("_", " ")
            return (
                f"Great {owner}, turning {topic} into a ready draft now. "
                "Next I will prepare the post copy and offer angle; reply CONFIRM to finalize."
            )

        if kind in {"review_theme_emerged", "competitor_opened", "festival_upcoming", "ipl_match_today", "milestone_reached", "category_seasonal", "gbp_unverified"}:
            payload = trigger.get("payload", {}) or {}
            topic = payload.get("theme") or payload.get("competitor_name") or payload.get("festival") or payload.get("match") or payload.get("metric") or kind
            return (
                f"Great {owner}, drafting the next action for {str(topic).replace('_', ' ')} now. "
                "Reply CONFIRM and I will proceed with the final version."
            )

        if kind in {"supply_alert", "regulation_change", "cde_opportunity"}:
            item = self._select_digest_item(category, trigger)
            payload = trigger.get("payload", {}) or {}
            topic = payload.get("molecule") or payload.get("deadline_iso") or (item or {}).get("title") or "this update"
            return (
                f"Great {owner}, preparing the action checklist for {topic} now. "
                "Reply CONFIRM to proceed."
            )

        if kind in {"renewal_due", "winback_eligible"}:
            return (
                f"Great {owner}, drafting the renewal/comeback note now. "
                "Reply CONFIRM to proceed."
            )

        return f"Great {owner}, drafting the next step now. Reply CONFIRM to proceed."

    def _select_digest_item(self, category: Dict[str, Any], trigger: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        digest = category.get("digest", []) or []
        payload = trigger.get("payload", {}) or {}
        target_id = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
        if target_id:
            for item in digest:
                if item.get("id") == target_id:
                    return item
        return digest[0] if digest else None
