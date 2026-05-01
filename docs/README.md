# Vera bot

Deterministic Vera bot for the magicpin AI Challenge. Implements the required HTTP endpoints and composes messages using an LLM with temperature 0. If no LLM is configured, the bot falls back to deterministic templates.

## Repo layout

- src/vera_bot/: application code
- docs/: challenge docs and notes
- dataset/: seed data

## Run locally

1) Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2) Set environment variables (see .env.example).

3) Start the server:

```bash
uvicorn vera_bot.main:app --app-dir src --host 0.0.0.0 --port 8080
```

## Endpoints

- GET /v1/healthz
- GET /v1/metadata
- POST /v1/context
- POST /v1/tick
- POST /v1/reply

## Configuration

Use environment variables:

- LLM_PROVIDER: openai or gemini
- LLM_API_KEY: API key for the provider
- LLM_MODEL: model name (optional)
- LLM_BASE_URL: base URL for OpenAI-compatible providers (optional)
- TEAM_NAME, TEAM_MEMBERS, CONTACT_EMAIL, BOT_VERSION, SUBMITTED_AT

## Notes

- The bot stores context in memory and is deterministic for the same inputs.
- Messages avoid URLs and taboo words from the category context.
- Auto-reply detection, hostile handling, and intent-commit routing are implemented.
