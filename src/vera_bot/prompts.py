from __future__ import annotations

from typing import Any, Dict, List, Optional

BASE_SYSTEM_PROMPT = (
    "You are Vera, magicpin's AI growth assistant for Indian local merchants. Write crisp "
    "WhatsApp-native messages that feel like a useful operator, not a newsletter.\n\n"
    "Hard rules:\n"
    "1) Use only facts present in the context. Never invent prices, dates, sources, slots, claims, "
    "patient outcomes, stock, or customer history.\n"
    "2) Start with the useful signal immediately. No preambles like 'I hope you are doing well'.\n"
    "3) Match category voice: dentists=clinical peer; salons=warm practical; restaurants=operator "
    "focused on demand/footfall; gyms=coach focused on momentum; pharmacies=trustworthy and precise.\n"
    "4) If merchant.languages includes 'hi' or customer language_pref includes 'hi', use natural "
    "Hindi-English code-mix in Roman script, but keep technical terms clear.\n"
    "5) Use one compulsion lever: loss aversion for dips/deadlines, curiosity for market signals, "
    "social proof for peer stats/reviews, or momentum for spikes/milestones.\n"
    "6) For research, compliance, supply, or news-like triggers, cite the named source in the body "
    "when available and include one concrete number/date if present.\n"
    "7) Cross-reference data! If the trigger relates to a specific customer segment (e.g. high-risk adults), look at the merchant's Customer aggregate to see exactly how many such customers they have, and mention that number.\n"
    "8) One primary CTA at the end. Prefer yes/no or confirm/cancel when an action is obvious.\n"
    "9) Do not include URLs. Respect taboo vocabulary. Keep most messages under 70 words.\n"
    "10) If customer context exists, send_as must be merchant_on_behalf; otherwise send_as must be vera.\n"
    "11) Output only valid JSON with no markdown or commentary.\n\n"
    "--- EXAMPLES OF 10/10 MESSAGES ---\n\n"
    "Example 1 (Research Digest for Dentists):\n"
    "Context: Dr. Meera. Trigger: JIDA Oct 2026 paper, 3-month fluoride recall cuts caries 38% in high-risk adults. Merchant has 124 high-risk adult patients.\n"
    "Message: \"Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look — JIDA Oct 2026 p.14. This likely affects your 124 high-risk adults. Want me to pull the abstract and flag patients due within 90 days?\"\n\n"
    "Example 2 (IPL Match for Restaurants):\n"
    "Context: Suresh, SK Pizza. Trigger: DC vs MI IPL match tonight at 7:30pm (Saturday). Merchant has BOGO pizza active.\n"
    "Message: \"Quick heads-up Suresh — DC vs MI at Arun Jaitley tonight, 7:30pm. Important: Saturday IPL matches usually shift -12% restaurant covers (people watch at home). Skip the match-night promo today; instead push your BOGO pizza (already active) as a delivery-only Saturday special. Want me to draft the Swiggy banner + an Insta story? Live in 10 min.\"\n\n"
    "Example 3 (Customer Winback for Gyms):\n"
    "Context: Karthik, PowerHouse Fitness. Trigger: Customer Rashmi lapsed 57 days ago, previous focus weight loss. Merchant has new Tue/Thu evening HIIT class.\n"
    "Message: \"Hi Rashmi 👋 Karthik from PowerHouse here. It's been about 8 weeks — happens to most members at some point, no judgment. We've added a Tue/Thu evening HIIT class that fits weight-loss goals well (45 min, 6:30pm). Want me to hold a free trial spot for you next Tue, 30 Apr? Reply YES — no commitment, no auto-charge.\"\n"
)

KIND_GUIDANCE = {
    "research_digest": "Cite the research source and one concrete number if present. Ask if the merchant wants the abstract or a draft message.",
    "regulation_change": "Lead with the authority/deadline and operational risk. Offer a checklist or SOP draft.",
    "cde_opportunity": "Mention credits/date/fee if present. Ask if the merchant wants a short registration note or team reminder.",
    "perf_dip": "Name the dip and anchor on a number. Reframe with a next step and a low-friction CTA.",
    "seasonal_perf_dip": "Name the dip as seasonal if marked expected, then suggest one acquisition action.",
    "perf_spike": "Celebrate the spike and suggest a simple follow-up to capitalize on momentum.",
    "renewal_due": "Use loss aversion around days remaining and plan continuity. Offer a concrete renewal/protection step.",
    "winback_eligible": "Mention the inactivity or expiry signal and one business consequence. Keep it low-pressure.",
    "recall_due": "Customer reminder with name, due reason, and a simple reply option. Do not invent slots.",
    "customer_lapsed_soft": "Warm winback without guilt. Mention prior relationship if available.",
    "customer_lapsed_hard": "No-shame winback with a simple trial or return option.",
    "appointment_tomorrow": "Reminder with date and a confirmation CTA.",
    "chronic_refill_due": "Precise, respectful refill reminder with medicine names and date if present.",
    "trial_followup": "Follow up on the trial and offer the next listed session option if present.",
    "wedding_package_followup": "Reference wedding/trial timing and suggest the next bridal prep step without pressure.",
    "review_theme_emerged": "Surface the theme and offer a response draft.",
    "competitor_opened": "Curiosity hook and a differentiation prompt. Do not invent competitor names.",
    "festival_upcoming": "Seasonal angle with a specific action related to the category.",
    "ipl_match_today": "Use the match, time, and local demand moment. Suggest a same-day offer or post.",
    "milestone_reached": "Celebrate and suggest the next small step.",
    "dormant_with_vera": "Low-pressure re-engagement with one useful new signal.",
    "curious_ask_due": "Ask one simple question and offer to draft a post.",
    "active_planning_intent": "Continue the merchant's active idea with a concrete draftable next step.",
    "supply_alert": "Use precise molecule/batch/manufacturer facts. Be careful and operational; offer a filtered customer/action list.",
    "category_seasonal": "Summarize the seasonal demand shift and recommend one shelf/content action.",
    "gbp_unverified": "Explain the visibility upside or risk of unverified profile and offer to guide the verification path.",
}


def build_prompt(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
    facts: List[str],
) -> str:
    kind = trigger.get("kind", "unknown")
    guidance = KIND_GUIDANCE.get(kind, "Compose a relevant message grounded in the trigger and merchant context.")
    context_text = format_context_summary(category, merchant, trigger, customer, facts)

    prompt = (
        "You will produce a JSON object with keys: body, cta, rationale, send_as.\n"
        "Valid cta values: binary_yes_no, binary_confirm_cancel, open_ended, none, multi_choice_slot.\n"
        "send_as must be vera or merchant_on_behalf.\n\n"
        "Guidance for this trigger kind: " + guidance + "\n\n"
        "Context summary (use only these facts):\n" + context_text + "\n\n"
        "Return ONLY valid JSON."
    )

    return prompt


def format_context_summary(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
    facts: List[str],
) -> str:
    lines: List[str] = []

    slug = category.get("slug")
    if slug:
        lines.append(f"Category: {slug}")

    voice = category.get("voice", {}) if isinstance(category.get("voice", {}), dict) else {}
    tone = voice.get("tone")
    taboos = _compact_list(voice.get("vocab_taboo") or voice.get("taboos") or [])
    allowed = _compact_list(voice.get("vocab_allowed") or [])
    if tone or taboos or allowed:
        lines.append(f"Voice: tone={tone or 'n/a'}; taboos={taboos or 'n/a'}; vocab_allowed={allowed or 'n/a'}")

    peer = category.get("peer_stats", {})
    if peer:
        lines.append(
            "Peer stats: avg_rating={avg_rating}, avg_reviews={avg_reviews}, avg_ctr={avg_ctr}, avg_calls_30d={avg_calls}".format(
                avg_rating=peer.get("avg_rating"),
                avg_reviews=peer.get("avg_review_count") or peer.get("avg_reviews"),
                avg_ctr=peer.get("avg_ctr"),
                avg_calls=peer.get("avg_calls_30d"),
            )
        )

    catalog_titles = _compact_list([o.get("title") for o in category.get("offer_catalog", []) if o.get("title")])
    if catalog_titles:
        lines.append(f"Offer patterns: {catalog_titles}")

    digest_item = _select_digest_item(category, trigger)
    if digest_item:
        parts = [digest_item.get("title"), digest_item.get("source")]
        trial_n = digest_item.get("trial_n")
        segment = digest_item.get("patient_segment")
        if trial_n is not None:
            parts.append(f"trial_n={trial_n}")
        if segment:
            parts.append(f"segment={segment}")
        lines.append("Digest item: " + " | ".join([p for p in parts if p]))

    identity = merchant.get("identity", {})
    merchant_name = identity.get("name")
    owner = identity.get("owner_first_name")
    locality = identity.get("locality")
    city = identity.get("city")
    languages = _compact_list(identity.get("languages") or [])
    if merchant_name or owner:
        lines.append(f"Merchant: name={merchant_name or 'n/a'}; owner={owner or 'n/a'}")
    if locality or city:
        lines.append(f"Location: {locality or 'n/a'}, {city or 'n/a'}")
    if languages:
        lines.append(f"Languages: {languages}")

    perf = merchant.get("performance", {})
    if perf:
        lines.append(
            "Performance: views={views}, calls={calls}, directions={directions}, ctr={ctr}".format(
                views=perf.get("views"),
                calls=perf.get("calls"),
                directions=perf.get("directions"),
                ctr=perf.get("ctr"),
            )
        )
        delta = perf.get("delta_7d", {})
        if delta:
            lines.append(
                "Delta 7d: views_pct={views_pct}, calls_pct={calls_pct}".format(
                    views_pct=delta.get("views_pct"),
                    calls_pct=delta.get("calls_pct"),
                )
            )

    offers = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active" and o.get("title")]
    offers_text = _compact_list(offers)
    if offers_text:
        lines.append(f"Active offers: {offers_text}")

    signals = _compact_list(merchant.get("signals") or [])
    if signals:
        lines.append(f"Signals: {signals}")

    aggregate = merchant.get("customer_aggregate", {})
    aggregate_text = _compact_payload(aggregate, limit=6) if isinstance(aggregate, dict) else ""
    if aggregate_text:
        lines.append(f"Customer aggregate: {aggregate_text}")

    review_themes = []
    for theme in merchant.get("review_themes", []) or []:
        if isinstance(theme, dict) and theme.get("theme"):
            bits = [str(theme.get("theme"))]
            if theme.get("sentiment"):
                bits.append(str(theme.get("sentiment")))
            if theme.get("occurrences_30d") is not None:
                bits.append(f"{theme.get('occurrences_30d')} in 30d")
            review_themes.append(" ".join(bits))
    review_text = _compact_list(review_themes, limit=3)
    if review_text:
        lines.append(f"Review themes: {review_text}")

    lines.append(
        "Trigger: kind={kind}, source={source}, urgency={urgency}, expires_at={expires_at}".format(
            kind=trigger.get("kind"),
            source=trigger.get("source"),
            urgency=trigger.get("urgency"),
            expires_at=trigger.get("expires_at"),
        )
    )
    payload_text = _compact_payload(trigger.get("payload", {}) or {})
    if payload_text:
        lines.append(f"Trigger payload: {payload_text}")

    if customer:
        cust_identity = customer.get("identity", {})
        cust_name = cust_identity.get("name")
        cust_lang = cust_identity.get("language_pref")
        cust_state = customer.get("state")
        pref = customer.get("preferences", {})
        pref_slots = pref.get("preferred_slots")
        if cust_name or cust_state:
            lines.append(f"Customer: name={cust_name or 'n/a'}; state={cust_state or 'n/a'}; lang={cust_lang or 'n/a'}")
        if pref_slots:
            lines.append(f"Customer preference: preferred_slots={pref_slots}")
        rel = customer.get("relationship", {}) if isinstance(customer.get("relationship", {}), dict) else {}
        rel_text = _compact_payload(
            {
                "last_visit": rel.get("last_visit"),
                "visits_total": rel.get("visits_total"),
                "services_received": rel.get("services_received"),
                "lifetime_value": rel.get("lifetime_value"),
                "favourite_dish": rel.get("favourite_dish"),
            },
            limit=5,
        )
        if rel_text:
            lines.append(f"Customer relationship: {rel_text}")

    if facts:
        facts_text = _compact_list(facts, limit=8)
        if facts_text:
            lines.append(f"Facts: {facts_text}")

    return "\n".join([f"- {line}" for line in lines if line])


def _compact_list(values: List[Any], limit: int = 4) -> str:
    cleaned = [str(v) for v in values if v]
    return ", ".join(cleaned[:limit])


def _compact_payload(payload: Dict[str, Any], limit: int = 6) -> str:
    parts = []
    for key in sorted(payload.keys()):
        val = payload.get(key)
        if isinstance(val, (str, int, float)):
            parts.append(f"{key}={val}")
        elif isinstance(val, dict) and "title" in val:
            parts.append(f"{key}.title={val.get('title')}")
        elif isinstance(val, list):
            list_text = _compact_list([_compact_list_value(v) for v in val], limit=3)
            if list_text:
                parts.append(f"{key}={list_text}")
        if len(parts) >= limit:
            break
    return ", ".join(parts)


def _compact_list_value(value: Any) -> str:
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in ("label", "title", "name", "iso", "id"):
            if value.get(key):
                return str(value.get(key))
    return ""


def _select_digest_item(category: Dict[str, Any], trigger: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    digest = category.get("digest", []) or []
    payload = trigger.get("payload", {}) or {}
    target_id = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
    if target_id:
        for item in digest:
            if item.get("id") == target_id:
                return item
    return digest[0] if digest else None
