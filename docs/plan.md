# 🎯 Vera AI Challenge — Battle Plan

> **Deadline:** 2 May 2026, 11:59 PM IST (~24 hours from now)
> **Goal:** Build the highest-scoring Vera bot and get selected for the core team.
> **This plan is designed to win, not just participate.**

---

## 1. What This Challenge Actually Is

magicpin's Vera talks to ~10,000 merchants/day over WhatsApp. They want to see if you can build a **smarter version**. You build an HTTP bot. Their judge calls your bot, pushes context, and scores your messages on 5 dimensions.

**The filter:** Most people will submit a generic bot that templates messages. The winning bot will demonstrate **product judgment** — knowing WHEN to send, WHAT to say, and HOW to say it for each specific merchant in each specific category.

---

## 2. Why "Rule-Based" Is the Wrong Approach

Both GPT and Sonnet suggested a rule-based approach as the "safe" option. This is a trap. Here's why:

| Dimension | Rule-based score ceiling | LLM-powered ceiling |
|---|---|---|
| Specificity | 6/10 (templates can inject numbers) | 10/10 (natural embedding of facts) |
| Category fit | 4/10 (one template per category = stiff) | 10/10 (adapts voice per category) |
| Merchant fit | 5/10 (slot-fill personalization) | 10/10 (contextual reasoning) |
| Trigger relevance | 6/10 (hardcoded per kind) | 10/10 (understands nuance) |
| Engagement compulsion | 3/10 (templates can't create curiosity) | 10/10 (crafts compelling hooks) |

The case studies prove this — every 9-10/10 score requires **natural language that feels like a peer talking**, not a template. A rule-based bot tops out at ~24/50. We need 40+/50.

**DECISION: We use LLM-powered composition with temperature=0 for determinism.** This is the only way to hit 40+/50.

---

## 3. Exactly How the Judge Scores (Source Code Analysis)

I read `judge_simulator.py` line by line. Here's what actually happens:

### Simulator Scenarios (what we MUST pass)
1. **`warmup`** — healthz + metadata + push 5 categories + 5 merchants → all must accept
2. **`phase2_short`** — push 3 triggers → tick → bot sends messages → LLM judge scores each
3. **`auto_reply_hell`** — judge sends same canned reply 4× → bot must detect and END by turn 3-4
4. **`intent_transition`** — merchant says "Ok lets do it. Whats next?" → bot must switch to ACTION (judge checks for: "done", "sending", "draft", "confirm", "proceed", "next")
5. **`hostile`** — "Stop messaging me. This is useless spam." → bot must END or apologize with ("sorry", "apolog", "won't")
6. **`full_evaluation`** — push ALL merchants + ALL triggers → tick in batches of 5 → score everything

### The LLM Scoring Prompt (exact fields from source)
The judge LLM sees:
- Category slug, voice tone, taboos
- Merchant name, owner name, locality, languages, performance (views/calls/ctr), signals, active offers
- Trigger kind, payload, urgency
- Customer identity (if present)
- Your bot's message body, CTA, and send_as

**Critical insight from the scoring prompt:** The judge checks if you used **real data from the context** vs fabricated data. Fabrication = `-2` penalty per instance.

### Dimension-by-Dimension Scoring Rules (from case studies)

| Dimension | What scores 9-10 | What scores 3-5 | What scores 0-2 |
|---|---|---|---|
| **Specificity** | "2,100-patient trial", "JIDA Oct 2026 p.14", "₹299", "22 of your chronic-Rx customers" | "improve your sales", "increase footfall" | No numbers, no sources |
| **Category fit** | Dentist=clinical/peer, Salon=warm/practical, Restaurant=operator, Gym=coach, Pharmacy=trustworthy | Generic professional tone | Promo voice for dentists ("AMAZING DEAL!") |
| **Merchant fit** | Owner first name, their actual CTR, their specific offers, their review themes | Generic "Hi there" | Wrong name, wrong city |
| **Trigger relevance** | "JIDA's Oct issue landed" (explicit trigger reference) | Implicit connection | No connection to why-now |
| **Engagement** | Loss aversion + specific CTA + low-friction ask | Multiple CTAs, vague asks | No CTA, no reason to reply |

---

## 4. Architecture — What We Build

```
vera-bot/
├── main.py                  # FastAPI server — 5 endpoints
├── context_store.py         # In-memory versioned context store
├── composer.py              # LLM-powered message composition engine
├── reply_handler.py         # Multi-turn conversation handling
├── prompts.py               # System prompt + per-trigger prompt templates
├── requirements.txt         # fastapi, uvicorn, httpx
├── Dockerfile               # Deployment container
├── README.md                # 1-page approach summary (required submission)
└── .env.example             # API key configuration
```

### 4.1 Context Store (`context_store.py`)

```python
class ContextStore:
    """In-memory versioned context store with O(1) lookup."""
    
    # Stores: {(scope, context_id): {"version": int, "payload": dict}}
    contexts: dict[tuple[str, str], dict]
    
    # Conversations: {conversation_id: [turns]}
    conversations: dict[str, list]
    
    # Suppression: {suppression_key: True} — prevent re-sends
    suppressed: set[str]
    
    # Sent bodies per conversation — anti-repetition
    sent_bodies: dict[str, set[str]]
```

**Key behaviors:**
- `push_context(scope, context_id, version, payload)` — idempotent; reject stale versions (409)
- `get_context(scope, context_id)` → payload or None
- `count_by_scope()` → `{"category": 5, "merchant": 50, ...}`
- `is_suppressed(key)` → bool
- `add_suppression(key)` — mark as sent
- `record_sent(conv_id, body)` — track for anti-repetition
- `is_repeat(conv_id, body)` → bool

### 4.2 Composer (`composer.py`) — The Brain

This is where we win or lose. The composer:

1. **Resolves all 4 contexts** from the store
2. **Selects the right prompt variant** based on trigger kind
3. **Calls the LLM** (temperature=0) with a meticulously crafted prompt
4. **Validates the output** (has CTA? has numbers? no taboo words? no fabrication?)
5. **Returns structured JSON** (body, cta, send_as, suppression_key, rationale)

```python
class Composer:
    def compose(self, trigger_id: str, store: ContextStore) -> dict:
        trigger = store.get_context("trigger", trigger_id)
        merchant = store.get_context("merchant", trigger["merchant_id"])
        category = store.get_context("category", merchant["category_slug"])
        customer = store.get_context("customer", trigger.get("customer_id"))
        
        # Build the rich prompt
        prompt = self._build_prompt(category, merchant, trigger, customer)
        
        # Call LLM with temperature=0
        result = self._call_llm(prompt)
        
        # Validate and fix
        result = self._validate(result, category, merchant, trigger)
        
        return result
```

### 4.3 Reply Handler (`reply_handler.py`) — Conversation Intelligence

Three critical detection patterns:

#### Auto-Reply Detection
```python
AUTO_REPLY_SIGNALS = [
    "thank you for contacting",
    "our team will respond",
    "automated message",
    "we will get back to you",
    "automated assistant",
]

def detect_auto_reply(message: str, conversation: list) -> bool:
    msg_lower = message.lower()
    # Signal 1: matches known auto-reply patterns
    if any(sig in msg_lower for sig in AUTO_REPLY_SIGNALS):
        return True
    # Signal 2: same message sent 2+ times in this conversation
    prev_msgs = [t["msg"] for t in conversation if t["from"] == "merchant"]
    if message in prev_msgs:
        return True
    return False
```

**Strategy on auto-reply:**
- Turn 1: Send one polite nudge ("Looks like an auto-reply — When the owner sees this, just reply 'Yes'")
- Turn 2: Wait 24h
- Turn 3+: End conversation

#### Intent Transition Detection
```python
INTENT_COMMIT_PHRASES = [
    "let's do it", "lets do it", "go ahead", "yes", "ok do it",
    "sounds good", "proceed", "what's next", "whats next",
    "haan karo", "kar do", "chalo", "theek hai",
]

def detect_intent_commit(message: str) -> bool:
    msg_lower = message.lower().strip()
    return any(phrase in msg_lower for phrase in INTENT_COMMIT_PHRASES)
```

**Strategy:** If committed → switch to ACTION mode. Never ask another qualifying question after "let's do it".

#### Hostile/Not-Interested Detection
```python
HOSTILE_SIGNALS = ["stop messaging", "not interested", "spam", "useless", 
                   "don't contact", "block", "remove me"]

def detect_hostile(message: str) -> bool:
    return any(sig in message.lower() for sig in HOSTILE_SIGNALS)
```

**Strategy:** Either END immediately, or send one graceful exit + END.

### 4.4 LLM Choice

**Primary: Google Gemini 2.0 Flash** (via free API key)
- Fast (under 3 seconds per call)
- Free tier: 15 RPM / 1M tokens/day — more than enough
- Quality is sufficient for well-prompted composition

**Fallback: DeepSeek Chat** (cheapest paid option at $0.14/M input tokens)
- If Gemini quota is exhausted

**Best Quality (if API key available): OpenAI GPT-4o-mini or Anthropic Claude Sonnet**
- Higher quality output but costs money

---

## 5. Prompt Engineering — The Competitive Edge

### 5.1 Base System Prompt

This prompt makes the LLM behave like a senior Vera product manager:

```
You are Vera, magicpin's AI merchant assistant. You compose WhatsApp messages
to merchants and their customers.

ABSOLUTE RULES:
1. ONLY use facts present in the context provided. Never fabricate data.
2. Use owner's first name (from identity.owner_first_name), not "Hi there".
3. Match the category voice: dentists=clinical peer, salons=warm practical, 
   restaurants=operator, gyms=coaching, pharmacies=trustworthy precise.
4. ONE primary CTA per message — binary YES/STOP for actions, open-ended for info.
5. No taboo words from category.voice.vocab_taboo.
6. If merchant.languages includes "hi", use natural Hindi-English code-mix.
7. No "I hope you're doing well" preambles. Get to the point.
8. Anchor on ONE verifiable fact (number, date, source citation).
9. End with the CTA, not in the middle.
10. Keep body concise for WhatsApp readability.
```

### 5.2 Per-Trigger-Kind Prompt Variants

Each trigger kind gets a specialized prompt that tells the LLM what pattern to follow:

| Trigger Kind | Prompt Emphasis | Example Output Shape |
|---|---|---|
| `research_digest` | Source citation, clinical anchor, curiosity hook | "JIDA's Oct issue… 2,100 patient trial… Want me to pull it?" |
| `perf_dip` | Reframe anxiety, show benchmarks, offer specific action | "Your calls -50% this week. Peer median is 12/mo. Here's what's different…" |
| `perf_spike` | Celebrate, explain why, leverage momentum | "Views +18% — likely your whitening post. Want to push another?" |
| `recall_due` | Patient name, service due, specific slots, price | "Hi Priya, Dr. Meera's clinic here 🦷 6-month cleaning due…" |
| `festival_upcoming` | Category-relevant angle, NOT generic "Diwali offer" | "Diwali in 4 days — salon bookings typically 3x. Bridal package not visible…" |
| `ipl_match_today` | Contrarian insight, use existing offers | "DC vs MI tonight — Saturday IPL = -12% dine-in. Push BOGO delivery instead" |
| `supply_alert` | Urgency, batch numbers, affected count, workflow offer | "Urgent: 2 atorvastatin batches recalled. 22 chronic-Rx customers affected…" |
| `competitor_opened` | Voyeur curiosity, differentiation data | "New dentist 1.3km away — Smile Studio, Cleaning @ ₹199. Your edge: 4.4★ vs 0 reviews" |
| `curious_ask_due` | Ask a question, offer reciprocity | "Quick check — what service most asked-for this week? I'll turn it into a Google post" |
| `winback_eligible` | Loss framing, show what they're missing | "38 days since expiry. Views dropped 30%. 24 potential customers joined your area…" |
| `review_theme_emerged` | Show the pattern, offer fix | "4 reviews mention late delivery. Quote: 'took 50 mins'. Want a response template?" |
| `milestone_reached` | Celebrate, social proof, next target | "You're at 145 reviews — 5 from 150! At that level, Google gives priority placement…" |
| `active_planning_intent` | Deliver the artifact, not another question | "Here's a starter version — you can edit: [concrete tiered plan]" |
| `customer_lapsed_hard` | No shame, previous goal, free trial | "It's been 8 weeks — happens to everyone. New HIIT class matches weight-loss goals…" |
| `chronic_refill_due` | Full molecule names, date, total + savings, delivery | "Sharma ji ki 3 monthly medicines khatam hongi 28 April ko. Same dose ready hai…" |
| `regulation_change` | Compliance urgency, deadline, specific action | "DCI revised radiograph dose limits. Deadline: Dec 15. E-speed film passes; D-speed doesn't…" |
| `cde_opportunity` | Event details, credits, fee, CTA to register | "IDA Delhi webinar on digital impressions — 2 CDE credits, free for members. May 2, 7pm…" |
| `seasonal_perf_dip` | Reframe as normal, save spend, focus retention | "Views -30% — normal April-June lull. Skip ad spend now, save for Sept when conversion is 2x…" |
| `gbp_unverified` | Uplift data, simple process, offer to do it | "Your GBP isn't verified — that alone could lift visibility 30%. Takes a postcard or phone call…" |
| `category_seasonal` | Shelf/inventory action, specific demand shifts | "Summer shift: ORS demand +40%, sunscreen +38%, cold-cough -60%. Time to restock shelves?" |
| `dormant_with_vera` | Low-key re-engagement, no pressure | "Been a while since we chatted. Quick update: [one relevant new signal]" |

### 5.3 Anti-Pattern Validation

Before returning any message, validate:

```python
def validate(body: str, category: dict) -> list[str]:
    issues = []
    
    # Check taboo words
    taboos = category.get("voice", {}).get("vocab_taboo", [])
    for taboo in taboos:
        if taboo.lower() in body.lower():
            issues.append(f"TABOO: '{taboo}' found in body")
    
    # Check for generic patterns the judge penalizes
    generics = ["increase your sales", "grow your business", "flat % off",
                 "I hope you're doing well", "amazing deal", "limited time"]
    for g in generics:
        if g.lower() in body.lower():
            issues.append(f"GENERIC: '{g}' found")
    
    # Check for URLs (penalty: -3 per URL in api-call-examples.md)
    if re.search(r'https?://', body):
        issues.append("URL found — Meta would reject")
    
    return issues
```

---

## 6. The 5 Endpoints — Implementation Spec

### `GET /v1/healthz`
```json
{
  "status": "ok",
  "uptime_seconds": 3600,
  "contexts_loaded": { "category": 5, "merchant": 50, "customer": 200, "trigger": 100 }
}
```
- Must return within **2 seconds**
- Must reflect actual context counts

### `GET /v1/metadata`
```json
{
  "team_name": "<your-name>",
  "team_members": ["<your-name>"],
  "model": "gemini-2.0-flash",
  "approach": "LLM-powered composer with per-trigger-kind prompt dispatch, auto-reply detection, and intent-transition routing. Context-grounded — every fact in the message is derived from pushed context.",
  "contact_email": "<your-email>",
  "version": "1.0.0",
  "submitted_at": "2026-05-02T00:00:00Z"
}
```

### `POST /v1/context`
- **Idempotent by (context_id, version)** — same version = 409 with current_version
- **Higher version replaces atomically**
- Store in memory with `{version, payload}` per key
- Return `{"accepted": true, "ack_id": "...", "stored_at": "..."}`

### `POST /v1/tick`
- Receives `{now, available_triggers}`
- For each trigger:
  1. Check if suppressed → skip
  2. Resolve merchant → resolve category → resolve customer (if customer-scoped)
  3. Call composer → validate output
  4. Add to actions list
- Cap at **20 actions** per tick (hard limit from testing brief)
- Must return within **10 seconds**
- Return `{"actions": [...]}`

### `POST /v1/reply`
- Receives merchant/customer reply
- Run detection pipeline: auto-reply → hostile → intent commit → normal
- For auto-reply: count per conversation → escalate (nudge → wait → end)
- For hostile: end or apologize+end
- For intent commit: switch to action mode
- For normal: call LLM with full conversation history for contextual reply
- Must return within **10 seconds**
- Return `{"action": "send"|"wait"|"end", "body": "...", "cta": "...", "rationale": "..."}`

---

## 7. Deployment Strategy

### Option A: Railway (Recommended — Fastest)
```bash
# 1. Push code to GitHub
# 2. Connect Railway to repo
# 3. Railway auto-detects Dockerfile, builds, deploys
# 4. Get URL: https://vera-bot-xxxx.up.railway.app
```
- Free tier: 500 hours/month, 512MB RAM
- HTTPS included
- Auto-deploy on push

### Option B: Render
```bash
# 1. Push to GitHub
# 2. Create Web Service on Render
# 3. Connect repo, set env vars (API key)
# 4. Get URL: https://vera-bot-xxxx.onrender.com
```
- Free tier available
- CAUTION: Spins down after 15 min idle (cold start ~30s — could fail healthz)
- Use Render's paid tier ($7/mo) to avoid cold starts

### Option C: ngrok (Backup — local testing exposed)
```bash
ngrok http 8080
# Get URL: https://abcd1234.ngrok-free.app
```
- Keep terminal open during evaluation
- Risky: if your PC sleeps, bot dies

> **Railway is the best choice.** Render free tier has cold-start issues that could fail the 3-consecutive-healthz check and get you disqualified.

---

## 8. Execution Timeline — Time-Boxed Phases

### Phase 1: Foundation (2 hours)
- [x] Read all docs ✅
- [ ] Run `generate_dataset.py` to create expanded dataset
- [ ] Set up project: `main.py`, `context_store.py`, `requirements.txt`
- [ ] Implement all 5 endpoints with skeleton logic
- [ ] Test locally: healthz, metadata, context push all working

### Phase 2: Composer Brain (3 hours)
- [ ] Write the base system prompt in `prompts.py`
- [ ] Write prompt variants for the top 8 trigger kinds
- [ ] Implement `composer.py` with LLM call + JSON parsing
- [ ] Wire composer into `/v1/tick`
- [ ] Test: push a category + merchant + trigger → tick → verify message quality

### Phase 3: Reply Intelligence (2 hours)
- [ ] Implement `reply_handler.py`
- [ ] Auto-reply detection (pattern matching + same-message tracking)
- [ ] Intent transition detection (commit phrase list)
- [ ] Hostile handling (signal detection + graceful exit)
- [ ] Normal reply handling (LLM with conversation history)
- [ ] Wire into `/v1/reply`

### Phase 4: Testing & Polishing (2 hours)
- [ ] Configure `judge_simulator.py` with our LLM API key
- [ ] Run `warmup` scenario → fix any issues
- [ ] Run `auto_reply_hell` scenario → verify detection
- [ ] Run `intent_transition` scenario → verify action switch
- [ ] Run `hostile` scenario → verify graceful exit
- [ ] Run `full_evaluation` → aim for 40+/50 average
- [ ] Fix any anti-patterns the judge flags

### Phase 5: Deploy & Submit (1 hour)
- [ ] Create Dockerfile
- [ ] Push to GitHub
- [ ] Deploy to Railway
- [ ] Verify all endpoints via curl from public URL
- [ ] Run `judge_simulator.py` against deployed URL
- [ ] Submit URL on magicpin portal
- [ ] Write `README.md`

**Total estimated time: ~10 hours** (with buffer for debugging)

---

## 9. What Makes THIS Plan Win

### vs. Generic Template Bots
Most submissions will slot-fill templates. Our LLM composer with per-trigger prompts will produce messages that feel like a thoughtful colleague, not a robot.

### vs. Naive LLM Bots
Many will dump all context into one giant prompt and hope. Our **per-trigger-kind prompt dispatch** ensures each message type gets optimized independently — research digests cite sources, perf dips reframe anxiety, recall reminders offer specific slots.

### vs. Bots That Ignore Post-Submission Injection
The judge injects NEW context mid-test. Our bot automatically uses the latest version of every context because we always read from the store at composition time. Bots that pre-compute or cache messages will fail here.

### vs. Bots That Don't Handle Replies
The auto-reply, intent-transition, and hostile scenarios are explicit test cases. Most submissions will fumble these. We handle all three with hardened detection.

### The Key Differentiators
1. **No fabrication** — every fact traces to pushed context
2. **Category voice matching** — clinical for dentists, warm for salons, operator for restaurants
3. **Owner first name** — always addressed personally (Dr. Meera, Suresh, Karthik)
4. **Hindi-English code-mix** — when merchant languages include "hi"
5. **Contrarian advice** — like Case Study 5 (don't push IPL promo on Saturday)
6. **Source citations** — for research/compliance triggers
7. **Anti-repetition** — tracked per conversation
8. **Suppression** — never re-send on same suppression key
9. **Conversation intelligence** — auto-reply detection, intent routing, graceful exits
10. **Fast** — every endpoint responds in <10 seconds

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM API quota exhaustion | Bot stops working mid-evaluation | Use Gemini free tier (1M tokens/day) + DeepSeek fallback |
| LLM timeout (>10s for tick) | Judge marks tick as failed | Set LLM timeout to 8s; return empty actions on timeout |
| Railway cold start | Healthz fails 3x → disqualified | Use Railway's always-on plan ($5/mo) |
| LLM outputs invalid JSON | Score 0 for that action | Wrap in try/except, regex-extract JSON, fallback to template |
| LLM fabricates data | -2 penalty per instance | Post-generation validation: check numbers in output appear in context |
| Bot crashes | Everything fails | Global exception handling + restart logic |

---

## 11. Decision Points — Need Your Input

Before I start coding, I need you to answer these:

1. **LLM API Key:** Which provider do you have access to?
   - Gemini (free — can get key in 2 min at https://aistudio.google.com/apikey)
   - OpenAI (GPT-4o-mini)
   - Anthropic (Claude Sonnet)
   - DeepSeek (cheapest paid)
   - Groq (fast + free tier)

2. **Deployment:** Do you have a Railway/Render account? Or should we use ngrok?

3. **Your name + email:** For the metadata endpoint and submission form.

4. **Start immediately?** Once you answer the above, I'll build the complete bot end-to-end.

---

## Appendix: Quick Reference — Scoring Cheat Sheet

```
EVERY MESSAGE MUST HAVE:
✅ Owner first name (Dr. Meera, not "Hi")
✅ One verifiable number from context
✅ Category-appropriate voice
✅ Explicit "why now" connection to trigger
✅ Single CTA at the end
✅ Rationale that matches the message

EVERY MESSAGE MUST NOT HAVE:
❌ URLs (Meta would reject, -3 penalty)
❌ Multiple CTAs
❌ Taboo words (guaranteed, cure, best in city)
❌ Fabricated data (-2 penalty per instance)
❌ Long preambles ("I hope you're doing well")
❌ Re-introduction after first message
❌ Same body sent twice (-2 per repeat)
❌ Generic offers ("Flat 30% off")
```
