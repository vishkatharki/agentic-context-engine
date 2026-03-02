# OpenClaw Integration

Make your [OpenClaw](https://docs.openclaw.ai) agent self-improving. ACE reads session transcripts, extracts what worked and what didn't, and feeds learned strategies back to the agent via a skillbook that it reads at the start of every session.

## How It Works

```
OpenClaw session ends  -->  transcript saved to ~/.openclaw/agents/main/sessions/*.jsonl
                                             |
                            ace-learn (session start or on-demand)
                                             |
                        LoadTracesStep --> OpenClawToTraceStep
                                             |
                            TraceAnalyser (Reflect -> Tag -> Update -> Apply)
                                             |
                         +------------------+------------------+
                         |                                     |
                  ace_skillbook.json                   ace_skillbook.md
                  (machine-readable)                   (human-readable)
                                                               |
                                          AGENTS.md tells agent to read skillbook
                                                               |
                                          Agent loads strategies into context
```

1. OpenClaw writes session transcripts to `~/.openclaw/agents/<id>/sessions/*.jsonl`
2. `ace-learn` runs at the start of the next session (or on-demand)
3. **LoadTracesStep** reads JSONL files into raw event lists
4. **OpenClawToTraceStep** converts events into structured traces
5. **TraceAnalyser** runs the learning pipeline (Reflect -> Tag -> Update -> Apply)
6. Updated skillbook is written to the workspace volume
7. The agent reads `ace_skillbook.md` into its context and applies relevant strategies

## Setup

There are two ways to set up ACE with OpenClaw:

- **Docker (recommended)** — bake ACE into the OpenClaw image. Zero runtime setup, the agent can trigger learning itself.
- **Host** — run ACE on the host machine via cron or manually. Simpler if you don't want to customize Docker.

---

## Docker Setup (Recommended)

This approach extends your OpenClaw Docker image with Python 3.12 and ACE pre-installed. The agent runs `ace-learn` at session start automatically.

### Prerequisites

- **OpenClaw** running via Docker (with `docker-compose.yml`)
- An **LLM API key** for the reflection model (Anthropic, OpenAI, AWS Bedrock, or any [LiteLLM-supported provider](https://docs.litellm.ai/docs/providers))

### 1. Get the Dockerfile

Copy `Dockerfile.ace` from the ACE repo into your OpenClaw directory:

```bash
# From the ACE repo
cp examples/openclaw/Dockerfile.ace /path/to/your/openclaw/
```

Or download it directly:

```bash
curl -o Dockerfile.ace https://raw.githubusercontent.com/Kayba-ai/agentic-context-engine/main/examples/openclaw/Dockerfile.ace
```

### 2. Build the image

From your OpenClaw directory:

```bash
# Build the base OpenClaw image first (if not already built)
docker build -t openclaw:base .

# Extend with ACE
docker build -t openclaw:local --build-arg OPENCLAW_IMAGE=openclaw:base -f Dockerfile.ace .
```

This adds ~200MB to the image and takes about 10 seconds. It installs:
- **uv** — Python package manager
- **Python 3.12** — via uv standalone builds (bookworm ships 3.11)
- **ACE framework** — cloned from GitHub with all dependencies
- **`ace-learn`** — wrapper command the agent calls

Then point your OpenClaw setup at the new image. In your `.env` file:

```bash
OPENCLAW_IMAGE=openclaw:local
```

### 3. Pass your API key through docker-compose

ACE needs an LLM API key to run reflection. Add the relevant key to your `docker-compose.yml` environment section (or your `.env` file):

```yaml
services:
  openclaw-gateway:
    environment:
      # ... existing keys ...
      # Add ONE of these depending on your provider:
      AWS_BEARER_TOKEN_BEDROCK: ${AWS_BEARER_TOKEN_BEDROCK}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
      LITELLM_API_KEY: ${LITELLM_API_KEY}
      # Optional: override the default reflection model
      ACE_MODEL: ${ACE_MODEL:-bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0}
```

The default model is `bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0`. Set `ACE_MODEL` in your `.env` to override (e.g., `ACE_MODEL=openrouter/anthropic/claude-sonnet-4-6`).

### 4. Add auto-learning to AGENTS.md

Add these sections to your `~/.openclaw/workspace/AGENTS.md`. This tells the agent to run learning at session start and read the skillbook.

First, add the skillbook to the "Every Session" checklist (or equivalent startup instructions):

```markdown
## Every Session

Before doing anything else:

1. Read `SOUL.md` — this is who you are
2. ...existing items...
3. Run `ace-learn` then read `skills/kayba-ace/ace_skillbook.md` — your learned strategies
```

Then add the detailed learning instructions:

```markdown
## Auto-Learning

At the start of each session, before responding to the user's first message,
run:

    ace-learn

After it completes, report the results to the user. Include:
- How many new sessions were processed
- How many new strategies were added (and their IDs)
- Total strategy count in the skillbook

If there were no new sessions, report that briefly.

If the command fails, tell the user that auto-learning failed and suggest they
check their API key configuration, then continue normally.
If `ace-learn` is not found, tell the user ACE is not installed and continue.

### On-demand learning

If the user asks to "learn from this session" (or similar), process only the
current session's transcript:

    ace-learn <path-to-current-session.jsonl>

Find the current session file in `~/.openclaw/agents/main/sessions/` (it will
be the most recently modified `.jsonl` file). Report what was learned.

Do **not** reprocess all sessions — only the current one.

## Learned Strategies

After running `ace-learn`, read the file `skills/kayba-ace/ace_skillbook.md`
into your context. This file contains strategies learned from past sessions.

**Important:** You must explicitly read the file — markdown links are not
auto-inlined. Use your file-reading tools to load the full content.

When a strategy is relevant to the current task:

1. Apply it.
2. Cite the strategy ID (e.g. `conversation_style-00003`) so the user can
   trace which learned behaviour influenced the response.
```

A complete snippet is available at [`examples/openclaw/AGENTS.md.snippet`](https://github.com/Kayba-ai/agentic-context-engine/blob/main/examples/openclaw/AGENTS.md.snippet).

### 5. Restart the gateway

```bash
docker compose down && docker compose up -d openclaw-gateway
```

### 6. Verify

Send a message to your agent (e.g., via Telegram). It should:

1. Run `ace-learn` and report what it found
2. Read the skillbook into its context
3. Respond to your message, citing strategy IDs when relevant

You can also test directly:

```bash
# Dry run — parses sessions without making LLM calls
docker run --rm -v ~/.openclaw:/home/node/.openclaw openclaw:local ace-learn --dry-run

# Full run
docker run --rm \
  -v ~/.openclaw:/home/node/.openclaw \
  -e AWS_BEARER_TOKEN_BEDROCK="$AWS_BEARER_TOKEN_BEDROCK" \
  openclaw:local ace-learn
```

---

## Host Setup

If you prefer running ACE on the host machine (outside Docker), this approach reads session files directly from disk.

### Prerequisites

- **Python 3.12+** (check with `python3 --version`)
- **[uv](https://docs.astral.sh/uv/)** package manager
- **OpenClaw** installed and running with at least one completed session
- An **LLM API key** for the reflection model

### 1. Install ACE

```bash
git clone https://github.com/Kayba-ai/agentic-context-engine.git
cd agentic-context-engine
uv sync
```

### 2. Set up as an OpenClaw skill

```bash
mkdir -p ~/.openclaw/workspace/skills/kayba-ace
cp examples/openclaw/learn_from_traces.py ~/.openclaw/workspace/skills/kayba-ace/
```

### 3. Configure your API key

```bash
# Option A: Anthropic direct
export ANTHROPIC_API_KEY="sk-ant-..."

# Option B: OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
export ACE_MODEL="openrouter/anthropic/claude-sonnet-4-6"

# Option C: AWS Bedrock (uses SDK auth, no key needed)
export ACE_MODEL="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"

# Option D: LiteLLM proxy
export LITELLM_API_KEY="your-key"
export LITELLM_API_BASE="https://your-proxy.example.com/v1"
export ACE_MODEL="anthropic/claude-sonnet-4-5"
```

You can also put these in `~/.openclaw/.env` or `~/.env` — the script loads both via `python-dotenv`.

### 4. Verify

```bash
cd /path/to/agentic-context-engine
uv run python ~/.openclaw/workspace/skills/kayba-ace/learn_from_traces.py --dry-run
```

### 5. Run learning

```bash
# One-off run
uv run python ~/.openclaw/workspace/skills/kayba-ace/learn_from_traces.py

# Process specific files
uv run python ~/.openclaw/workspace/skills/kayba-ace/learn_from_traces.py \
  ~/.openclaw/agents/main/sessions/f967d602.jsonl

# Reprocess everything
uv run python ~/.openclaw/workspace/skills/kayba-ace/learn_from_traces.py --reprocess
```

### 6. Automate with cron

```bash
crontab -e
```

Add:

```
*/30 * * * * cd /path/to/agentic-context-engine && uv run python ~/.openclaw/workspace/skills/kayba-ace/learn_from_traces.py >> /tmp/ace-openclaw.log 2>&1
```

### 7. Wire up AGENTS.md

Follow the same AGENTS.md instructions from [step 4 of the Docker setup](#4-add-auto-learning-to-agentsmd). For the host setup, the agent can't run `ace-learn` directly (it's not in the container), so replace the `ace-learn` command with instructions to reference the skillbook file only. Learning happens externally via cron.

---

## Making Learned Strategies Available to the Agent

The learning script writes two files to the skill directory:

| File | Description |
|---|---|
| `ace_skillbook.json` | Machine-readable skillbook (persists across runs) |
| `ace_skillbook.md` | Human-readable skillbook grouped by section |
| `ace_processed.txt` | Tracks which sessions have been processed |

The agent loads strategies by **reading `ace_skillbook.md` at session start**. This must be an explicit instruction in AGENTS.md — OpenClaw does not auto-inline linked files. The agent uses its file-reading tools to load the skillbook content into its context window.

Once loaded, the agent has all strategies available and can cite their IDs (e.g., `conversation_style-00003`) when applying them.

## Pipeline Steps

**LoadTracesStep** — Reads a JSONL file and parses each line into a list of event dicts.

**OpenClawToTraceStep** — Converts raw OpenClaw events into a structured trace:

```python
{
    "question": "User: ...\n\nUser: ...",   # All user messages
    "reasoning": "[thinking] ...\n[tool:read] ...\n[response] ...",
    "answer": "Last assistant response",
    "skill_ids": [],
    "feedback": "OpenClaw session: 3 user messages, model: ..., 14605 tokens",
    "ground_truth": None
}
```

**TraceAnalyser** — Runs the ACE learning tail:

1. **Reflect** — LLM analyzes the trace for patterns, errors, and effective strategies
2. **Tag** — Scores cited skills as helpful/harmful/neutral
3. **Update** — LLM decides skillbook mutations (ADD, UPDATE, REMOVE, CONSOLIDATE)
4. **Apply** — Commits changes to the in-memory skillbook

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `ACE_MODEL` | `bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0` | LLM for reflection and skill extraction |
| `OPENCLAW_AGENT_ID` | `main` | Agent ID for session discovery |
| `OPENCLAW_HOME` | `$HOME/.openclaw` | OpenClaw home directory (used by `ace-learn` only; do **not** set as a gateway env var — see [troubleshooting](#openclaw_home-double-nesting)) |
| `LITELLM_API_KEY` | - | API key (for non-Bedrock providers) |
| `LITELLM_API_BASE` | - | LiteLLM proxy base URL |
| `SPH_LITELLM_KEY` | - | Alternative API key variable |
| `AWS_BEARER_TOKEN_BEDROCK` | - | AWS Bedrock bearer token |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `OPENROUTER_API_KEY` | - | OpenRouter API key |

### CLI arguments

```
ace-learn [OPTIONS] [FILES...]

Options:
  --dry-run          Parse sessions but skip learning (no LLM calls)
  --reprocess        Ignore processed log, reprocess all sessions
  --agent AGENT_ID   OpenClaw agent ID (default: main)
  --output DIR       Output directory for skillbook files
  --opik             Enable Opik observability logging

Positional:
  FILES              Specific JSONL files to process (skips discovery)
```

## Example Skillbook Output

After processing a few sessions, `ace_skillbook.md` might contain:

```markdown
## aws_bedrock

### `aws_bedrock-00001`

Set API_KEY=None for Bedrock models to use AWS_BEARER_TOKEN_BEDROCK

**Justification:** Enhanced specificity about AWS_BEARER_TOKEN_BEDROCK requirement
**Evidence:** Pipeline succeeded after changing API_KEY logic to None for Bedrock models

*Tags: helpful=1, harmful=0, neutral=0*

## debugging

### `debugging-00005`

Test litellm calls directly before debugging ace_next pipeline

**Justification:** Systematic debugging approach that isolated authentication issues
**Evidence:** Direct litellm.completion() calls worked while ace_next failed

*Tags: helpful=1, harmful=0, neutral=0*
```

## Troubleshooting

### "Sessions directory not found"

The agent hasn't run yet, or `OPENCLAW_AGENT_ID` is wrong. Check:

```bash
ls ~/.openclaw/agents/
```

### "Nothing new to learn from"

All sessions have been processed. Use `--reprocess` to rerun, or wait for new sessions.

### `ace-learn` not found in Docker

Make sure you built with `Dockerfile.ace` and are using the correct image tag:

```bash
docker run --rm openclaw:local which ace-learn
```

### Import errors for `ace_next` (host setup)

Make sure you run from the ACE repo root with `uv run`:

```bash
cd /path/to/agentic-context-engine
uv run python ~/.openclaw/workspace/skills/kayba-ace/learn_from_traces.py
```

### JSON parsing errors with Haiku models

The deployed script includes a monkey-patch to strip control characters from LLM responses. If you see `Invalid control character` errors, make sure you're using the latest version of the script.

### API key errors in Docker

Make sure your LLM API key is passed through `docker-compose.yml`. Check with:

```bash
docker compose exec openclaw-gateway env | grep -E 'API_KEY|BEARER_TOKEN|ACE_MODEL'
```

### OPENCLAW_HOME double-nesting

If the gateway reads a bare-bones config (missing Telegram, models, credentials) or you see paths like `/home/node/.openclaw/.openclaw/`, the `OPENCLAW_HOME` environment variable is being set inside the container. OpenClaw derives its data directory from `$HOME/.openclaw/`; setting `OPENCLAW_HOME` explicitly causes it to nest a second `.openclaw/` inside the mount.

**Fix:** Make sure `OPENCLAW_HOME` is **not** set as a gateway environment variable. If you're using an older version of `Dockerfile.ace` that sets `ENV OPENCLAW_HOME`, either rebuild with the latest Dockerfile or unset it in your `docker-compose.yml`:

```yaml
environment:
  OPENCLAW_HOME:   # empty value unsets the Dockerfile ENV
```

`ace-learn` uses `OPENCLAW_HOME` internally with a safe fallback — this is fine and does not affect the gateway.

## What to Read Next

- [Integration Pattern](../guides/integration.md) — how the INJECT/EXECUTE/LEARN pattern works
- [The Skillbook](../concepts/skillbook.md) — how learned strategies are stored
- [ACE Design](../ACE_DESIGN.md) — architecture and step reference
