# Kayba Hosted API

The Kayba hosted API lets you upload traces, generate insights, and pull optimised prompts without running ACE roles locally. The `ace-cloud` CLI wraps every API endpoint.

## Prerequisites

1. A Kayba API key (set `KAYBA_API_KEY` or pass `--api-key` to every command).
2. Install the `cloud` extra:

```bash
pip install ace-framework[cloud]
```

Or if you installed from source:

```bash
uv sync
```

## Authentication

Every command reads `KAYBA_API_KEY` from the environment. You can also pass it explicitly:

```bash
export KAYBA_API_KEY=your-key-here
ace-cloud cloud upload traces/
```

The default API endpoint is `https://use.kayba.ai/api`. Override it with `KAYBA_API_URL` or `--base-url`.

## CLI Reference

### Upload traces

```bash
# Single file
ace-cloud cloud upload trace.md

# Directory (recursive)
ace-cloud cloud upload traces/

# Pipe from stdin
cat trace.md | ace-cloud cloud upload -

# Force file type
ace-cloud cloud upload traces/ --type json
```

Files larger than 350k characters trigger a warning. Supported types: `md`, `json`, `txt` (auto-detected from extension).

### Generate insights

```bash
# From all uploaded traces
ace-cloud cloud insights generate --wait

# Specific traces
ace-cloud cloud insights generate --traces ID1 --traces ID2

# Custom model and epochs
ace-cloud cloud insights generate --model claude-opus-4-6 --epochs 3 --wait
```

Options:

| Flag | Description |
|------|-------------|
| `--traces ID` | Trace IDs to analyse (repeatable) |
| `--model` | `claude-sonnet-4-6` or `claude-opus-4-6` |
| `--epochs N` | Number of analysis epochs |
| `--reflector-mode` | `recursive` or `standard` |
| `--anthropic-key` | Anthropic API key for server-side LLM calls |
| `--wait` | Poll until the job completes |

### List and triage insights

```bash
# List all
ace-cloud cloud insights list

# Filter by status
ace-cloud cloud insights list --status pending

# JSON output
ace-cloud cloud insights list --json

# Accept specific insights
ace-cloud cloud insights triage --accept ID1 --accept ID2

# Accept all pending
ace-cloud cloud insights triage --accept-all

# Reject with a note
ace-cloud cloud insights triage --reject ID1 --note "Too vague"
```

### Generate and pull prompts

```bash
# Generate a prompt from accepted insights
ace-cloud cloud prompts generate

# Generate with a label and save to file
ace-cloud cloud prompts generate --label "v2-coding" -o prompt.md

# List prompt versions
ace-cloud cloud prompts list

# Pull latest prompt
ace-cloud cloud prompts pull

# Pull specific version
ace-cloud cloud prompts pull --id PROMPT_ID -o skillbook-prompt.md

# Pretty-print full JSON
ace-cloud cloud prompts pull --pretty
```

### Job status and materialisation

```bash
# Check job status
ace-cloud cloud status JOB_ID

# Poll until complete
ace-cloud cloud status JOB_ID --wait --interval 10

# Materialise results into the skillbook
ace-cloud cloud materialize JOB_ID
```

### Batch pre-processing

The `batch` command groups traces into batches before analysis. It works in two modes:

**Prepare mode** (default) — extracts trace metadata and prints a classification prompt:

```bash
ace-cloud cloud batch traces/
```

This writes a skeleton `batches.json` and prints a prompt to stdout. Pipe it to an LLM (e.g. Claude Code) to fill in the batch assignments.

**Apply mode** — validates and optionally uploads a batch plan:

```bash
# Validate only
ace-cloud cloud batch traces/ --apply batches.json

# Validate and upload each batch
ace-cloud cloud batch traces/ --apply batches.json --upload
```

Options:

| Flag | Description |
|------|-------------|
| `--prompt FILE` | Custom classification prompt template |
| `-o FILE` | Output batch plan file (default: `batches.json`) |
| `--apply FILE` | Apply an existing batch plan |
| `--upload` | Upload each batch (requires `--apply`) |
| `--min-batch-size N` | Minimum traces per batch (default: 10) |
| `--max-batch-size N` | Maximum traces per batch (default: 30) |

## End-to-end workflow

```bash
# 1. Upload traces
ace-cloud cloud upload traces/

# 2. Generate insights (waits for completion)
ace-cloud cloud insights generate --wait

# 3. Review insights
ace-cloud cloud insights list --status pending

# 4. Accept the good ones
ace-cloud cloud insights triage --accept-all

# 5. Generate a prompt
ace-cloud cloud prompts generate -o prompt.md

# 6. Use the prompt in your agent
cat prompt.md
```

## Python client

The `KaybaClient` class can be used directly in Python code:

```python
from ace.cli.client import KaybaClient

client = KaybaClient(api_key="your-key")

# Upload traces
result = client.upload_traces([
    {"filename": "trace.md", "content": "...", "fileType": "md"},
])

# Generate insights
job = client.generate_insights(model="claude-sonnet-4-6")

# Check status
status = client.get_job(job["jobId"])

# List and triage
insights = client.list_insights(status="pending")
client.triage_insight(insights["insights"][0]["id"], "accepted")

# Generate and pull prompts
client.generate_prompt()
prompts = client.list_prompts()
prompt = client.get_prompt(prompts[0]["id"])
```

## API endpoints

| Method | Path | Client method |
|--------|------|---------------|
| `POST` | `/traces` | `upload_traces()` |
| `POST` | `/insights/generate` | `generate_insights()` |
| `GET` | `/insights` | `list_insights()` |
| `PATCH` | `/insights/:id` | `triage_insight()` |
| `GET` | `/jobs/:id` | `get_job()` |
| `POST` | `/jobs/:id` | `materialize_job()` |
| `POST` | `/prompts/generate` | `generate_prompt()` |
| `GET` | `/prompts` | `list_prompts()` |
| `GET` | `/prompts/:id` | `get_prompt()` |

## Environment variables

| Variable | Description |
|----------|-------------|
| `KAYBA_API_KEY` | API key (required) |
| `KAYBA_API_URL` | Base URL (default: `https://use.kayba.ai/api`) |
| `ANTHROPIC_API_KEY` | Passed to server for LLM calls via `--anthropic-key` |
