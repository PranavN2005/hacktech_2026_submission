# Hacktech 2026 Submission

## Collaborators

- Pranav Nallaperumal
- Ahad Jiva

## Development

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install frontend packages

```bash
cd frontend
pnpm install
```

### Run tests

```bash
.venv/bin/python -m pytest tests/ -v
```

### Generate a fresh agent population

The structured three-stage pipeline (sample → map → narrate) lives in
`backend/persona_pipeline.py` and is driven by `generate_population.py`.
The LLM is **only** allowed to write the `bio` field; every numerical
parameter is computed deterministically from a Latin Hypercube sample
plus a Gaussian copula over six trait axes.

```bash
# Stage 1 + 2: pure math. Produces a fully valid agent population, no LLM.
.venv/bin/python generate_population.py sample --n 500 --seed 0 --out population.json

# Stage 3: fill in the bio strings via an LLM. Resumable, rate-limited.
.venv/bin/python generate_population.py bios --input population.json --model gpt-4o-mini

# Diagnostics: range checks + summary stats + histogram.
.venv/bin/python generate_population.py validate --input population.json

# Smoke test the engine adapter:
.venv/bin/python generate_population.py load-check --input population.json
```

`generate_population.py legacy ...` (or `python persona_gen.py --allow-legacy`)
re-enables the deprecated one-shot LLM persona generator. The legacy path
is gated behind `USE_LEGACY_PERSONA_GENERATION` and exists for short-term
comparison only.

### Start the server

```bash
.venv/bin/uvicorn backend.main:app --port 8000
```
```bash
cd frontend
pnpm dev
```

### Verify the API

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/init | head -c 400
curl -N "http://127.0.0.1:8000/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=3&interval=0"
```

The server resolves the agents file in this order:

1. `AGENTS_PATH` environment variable (explicit override).
2. `population.json` in the working directory (preferred — produced by `sample`).
3. `agents_500_pro_max.json` (legacy fallback).

```bash
AGENTS_PATH=my_population.json .venv/bin/uvicorn backend.main:app --port 8000
```

