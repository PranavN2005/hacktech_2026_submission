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
.venv/bin/python -m pytest tests/test_engine.py -v
```

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

By default the server loads `agents_500_pro_max.json` from the working directory.
Override with the `AGENTS_PATH` environment variable:

```bash
AGENTS_PATH=my_agents.json .venv/bin/uvicorn backend.main:app --port 8000
```

