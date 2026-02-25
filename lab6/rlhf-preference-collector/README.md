# RLHF Preference Collector

A Streamlit app for collecting human preference labels for RLHF/DPO alignment training.

The app sends one prompt to a local Ollama model twice (different temperatures), randomizes display position (A/B), and stores the user preference with metadata.

## Why This Matters

Preference pairs (`prompt`, `chosen`, `rejected`) are the core supervised signal used in many RLHF and DPO pipelines. This app helps build that dataset with:

- Position-bias mitigation (randomized A/B assignment)
- No re-roll policy (prevents cherry-picking)
- Rich metadata for quality analysis

## Project Structure

- `app.py`: Streamlit UI and workflow/state management
- `llm.py`: Ollama wrapper (health checks + generation)
- `database.py`: Supabase integration + local JSONL fallback
- `export.py`: stats and JSONL export helpers
- `config.py`: environment configuration/constants
- `setup_supabase.sql`: SQL schema and indexes
- `.env.example`: required environment variables
- `requirements.txt`: Python dependencies

## Prerequisites

1. Python 3.10+
2. Install Ollama: [https://ollama.com](https://ollama.com)
3. Pull a model (default):

```bash
ollama pull llama3.2
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
cp .env.example .env
```

Required/default vars:

- `OLLAMA_HOST` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3.2`)
- `NUM_PREDICT` (default: `1024`)

Optional Supabase vars:

- `SUPABASE_URL`
- `SUPABASE_KEY`

3. Optional: Create Supabase table

- Open Supabase SQL editor and run `setup_supabase.sql`.

4. Run the app:

```bash
streamlit run app.py
```

## Supabase vs Offline Mode

- If Supabase is configured and reachable, records are written to `preference_data`.
- If Supabase is missing/unavailable, the app falls back to local `preference_data.jsonl`.
- This means the app works fully offline with just Ollama + local JSONL storage.

## Data Schema

Each record includes:

- `id` (UUID)
- `timestamp` (ISO 8601)
- `prompt`
- `response_a`, `response_b`
- `chosen`, `rejected` (both `null` for ties)
- `preference` (`a`, `b`, `tie`)
- `model`
- `generation_params` (JSON)
- `response_a_latency_ms`, `response_b_latency_ms`
- `session_id` (UUID)
- `position_mapping` (JSON)

## Exporting Data

Sidebar actions:

- `Export Training Data`: JSONL with only `{prompt, chosen, rejected}` and excludes ties (ready for DPO/RLHF training)
- `Export Full Data`: JSONL with all metadata

Data source for exports:

- Supabase (if connected)
- Local JSONL fallback otherwise

## DPO Training Notes

Use `training_data.jsonl` where each line is:

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

Typical next step:

1. Tokenize `prompt/chosen/rejected`
2. Train with a DPO implementation
3. Keep ties out of DPO objective (already excluded by export)
