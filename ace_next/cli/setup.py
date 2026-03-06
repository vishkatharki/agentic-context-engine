"""``ace setup`` — interactive configuration wizard.

Guides the user through:
1. Enter a model name (any LiteLLM model string)
2. Validate the connection — if it fails, prompt for keys
3. Optionally assign different models per ACE role
4. Save .env (secrets) and ace.toml (model config)
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from ..providers.config import (
    ACEModelConfig,
    ModelConfig,
    find_config,
    load_config,
    load_dotenv,
    save_config,
    save_env_var,
)
from ..providers.registry import (
    PROVIDER_MODEL_EXAMPLES,
    _PROVIDER_KEY_ENV,
    get_missing_keys,
    get_provider,
    suggest_models,
    validate_connection,
)


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

BOLD = "\033[1m" if _IS_TTY else ""
DIM = "\033[2m" if _IS_TTY else ""
GREEN = "\033[32m" if _IS_TTY else ""
RED = "\033[31m" if _IS_TTY else ""
YELLOW = "\033[33m" if _IS_TTY else ""
CYAN = "\033[36m" if _IS_TTY else ""
RESET = "\033[0m" if _IS_TTY else ""


def _ok(msg: str) -> None:
    print(f"  {GREEN}v{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}x{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  {label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    return value or default


def _prompt_secret(label: str) -> str:
    try:
        value = getpass.getpass(f"  {label}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    return value


def _confirm(label: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        value = input(f"  {label} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if not value:
        return default
    return value in ("y", "yes")


# ---------------------------------------------------------------------------
# Model + key flow
# ---------------------------------------------------------------------------


def _detect_credential_source(provider: str) -> str | None:
    """Return which credential env var is set for *provider*."""
    env_vars = _PROVIDER_KEY_ENV.get(provider)
    if env_vars is None:
        return None
    if isinstance(env_vars, str):
        env_vars = [env_vars]
    found = [v for v in env_vars if os.environ.get(v)]
    if not found:
        return None
    return ", ".join(found)


def _validate_and_prompt_keys(
    model: str,
    provider: str,
    directory: Path,
) -> bool:
    """Try to validate *model*. If auth fails, prompt for missing keys and retry.

    Returns True on success. On non-auth failures (model not found, etc.)
    prints the error and returns False so the caller can re-prompt.
    """
    # First: just try it — handles bearer tokens, ~/.aws/credentials, etc.
    print(f"  Validating...", end="", flush=True)
    result = validate_connection(model)

    if result.success:
        print(
            f"\r  {GREEN}v{RESET} Connected! "
            f"({model} via {result.provider}, {result.latency_ms}ms)"
        )
        # Show which credential was used
        cred_source = _detect_credential_source(result.provider or provider)
        if cred_source:
            _info(f"Using {cred_source}")
        return True

    # Not an auth problem — show error and return
    if "Invalid API key" not in result.error and "connect" not in result.error.lower():
        print(f"\r  {RED}x{RESET} {result.error}                    ")
        if "not found" in result.error.lower():
            suggestions = suggest_models(model)
            if suggestions:
                _info("Did you mean one of these?")
                for s in suggestions:
                    _info(f"  - {s}")
        return False

    # Auth/connection failure — prompt for missing keys
    print(f"\r  {YELLOW}!{RESET} No credentials found for {provider}                    ")

    # Prefer our own mapping over LiteLLM's generic response, since
    # LiteLLM often returns wrong keys (e.g. bedrock_converse gets
    # generic bedrock keys instead of the bearer token alternative).
    our_keys = _PROVIDER_KEY_ENV.get(provider)
    if our_keys is not None:
        missing = [our_keys] if isinstance(our_keys, str) else list(our_keys)
    else:
        missing = get_missing_keys(model)
        if not missing:
            missing = [f"{provider.upper()}_API_KEY"]

    provided_keys: dict[str, str] = {}
    for env_var in missing:
        value = _prompt_secret(f"{env_var}")
        if value:
            provided_keys[env_var] = value
            os.environ[env_var] = value

    if not provided_keys:
        _fail("No credentials provided.")
        return False

    # Retry validation
    print(f"  Validating...", end="", flush=True)
    result = validate_connection(model)

    if result.success:
        print(
            f"\r  {GREEN}v{RESET} Connected! "
            f"({model} via {result.provider}, {result.latency_ms}ms)"
        )
        # Persist keys to .env
        for env_var, value in provided_keys.items():
            save_env_var(env_var, value, directory)
        _ok(f"Saved credentials to .env")
        return True
    else:
        print(f"\r  {RED}x{RESET} {result.error}                    ")
        # Roll back
        for env_var in provided_keys:
            os.environ.pop(env_var, None)
        return False


def _setup_model(
    role_label: str,
    directory: Path,
    *,
    default_model: str = "",
) -> str:
    """Prompt for model, validate connection. Return the validated model string.

    Loops until validation succeeds or the user quits (Ctrl-C).
    """
    while True:
        model = _prompt(f"{role_label} model", default=default_model)
        if not model:
            continue

        provider = get_provider(model)

        if provider == "unknown":
            _fail(f"Could not detect a provider for '{model}'.")
            _info("Use the format: provider/model-name (e.g. groq/llama-3.1-70b)")
            suggestions = suggest_models(model)
            if suggestions:
                _info("Did you mean one of these?")
                for s in suggestions[:5]:
                    _info(f"  - {s}")
            print()
            continue

        if _validate_and_prompt_keys(model, provider, directory):
            return model

        print()  # blank line before retry


# ---------------------------------------------------------------------------
# Main setup flow
# ---------------------------------------------------------------------------


def run_setup(directory: str | Path = ".") -> ACEModelConfig:
    """Run the interactive setup wizard. Returns the saved config."""
    directory = Path(directory).resolve()

    print()
    print(f"{BOLD}ACE Setup{RESET}")
    print()

    # Load existing .env if present
    load_dotenv()

    # Check for existing config
    existing = find_config(directory)
    if existing:
        try:
            old = load_config(existing.parent)
            _info(f"Found existing config: {existing}")
            _info(f"  Default model: {old.default.model}")
            for role in ("agent", "reflector", "skill_manager"):
                cfg = getattr(old, role)
                if cfg:
                    _info(f"  {role}: {cfg.model}")
            print()
            if not _confirm("Reconfigure?"):
                print()
                _ok("Keeping existing config.")
                return old
            print()
        except Exception:
            pass  # corrupted config — just reconfigure

    # Step 1: Default model
    print(f"{BOLD}Step 1: Choose your model{RESET}")
    print()
    _info("Examples: gpt-4o-mini, claude-sonnet-4-20250514, ollama/llama2")
    _info(f"Search models: {CYAN}ace models <query>{RESET}")
    print()

    default_model = _setup_model("Default", directory)
    print()

    # Step 2: Per-role assignment
    print(f"{BOLD}Step 2: Role assignment{RESET}")
    print()
    _info("ACE uses three roles. You can assign a different model to each,")
    _info("or use the same model for all (recommended to start).")
    print()

    use_same = _confirm("Use this model for all roles?")

    agent_cfg: ModelConfig | None = None
    reflector_cfg: ModelConfig | None = None
    skill_manager_cfg: ModelConfig | None = None

    if not use_same:
        print()
        _info("Press Enter to keep the default for any role.")
        print()

        for role_name, label in [
            ("agent", "Agent (executes tasks)"),
            ("reflector", "Reflector (analyses results)"),
            ("skill_manager", "Skill Manager (updates skillbook)"),
        ]:
            model = _prompt(label, default=default_model)
            if model != default_model:
                model = _setup_model(label, directory, default_model=model)
                if role_name == "agent":
                    agent_cfg = ModelConfig(model=model)
                elif role_name == "reflector":
                    reflector_cfg = ModelConfig(model=model)
                else:
                    skill_manager_cfg = ModelConfig(model=model)

    # Build and save config
    config = ACEModelConfig(
        default=ModelConfig(model=default_model),
        agent=agent_cfg,
        reflector=reflector_cfg,
        skill_manager=skill_manager_cfg,
    )

    config_path = save_config(config, directory)
    print()
    _ok(f"Saved model config to {config_path.name}")

    # Summary
    print()
    print(f"  {BOLD}Configuration summary:{RESET}")
    _info(f"  default:        {default_model}")
    for role in ("agent", "reflector", "skill_manager"):
        cfg = getattr(config, role)
        if cfg:
            _info(f"  {role + ':':<16}{cfg.model}")
    print()
    print(f"  {BOLD}Ready!{RESET} Use in code:")
    print()
    print(f"    {CYAN}from ace_next import ACELiteLLM{RESET}")
    print(f"    {CYAN}ace = ACELiteLLM.from_setup(){RESET}")
    print()

    return config


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``ace`` CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="ace",
        description="ACE Framework CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ace setup
    setup_parser = subparsers.add_parser("setup", help="Configure models and API keys")
    setup_parser.add_argument(
        "--dir",
        default=".",
        help="Directory to save config files (default: current directory)",
    )

    # ace models
    models_parser = subparsers.add_parser(
        "models", help="Search available models"
    )
    models_parser.add_argument("query", nargs="*", default=[], help="Search query (multiple terms = match all)")
    models_parser.add_argument("--provider", default=None, help="Filter by provider")
    models_parser.add_argument(
        "--limit", type=int, default=20, help="Max results (default: 20)"
    )

    # ace validate
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a model connection"
    )
    validate_parser.add_argument("model", help="Model name to validate")

    args = parser.parse_args()

    if args.command == "setup":
        run_setup(args.dir)
    elif args.command == "models":
        _cmd_models(" ".join(args.query), args.provider, args.limit)
    elif args.command == "validate":
        _cmd_validate(args.model)
    else:
        parser.print_help()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_models(query: str, provider: str | None, limit: int) -> None:
    """``ace models [query]`` — search available models."""
    from ..providers.registry import search_models

    load_dotenv()
    results, total = search_models(query=query, provider=provider, limit=limit)

    if not results:
        if query:
            print(f"No models matching '{query}'.")
        else:
            print("No models found.")
        print(f"Try: {CYAN}ace models gpt-4o{RESET} or {CYAN}ace models claude{RESET}")
        return

    print(
        f"{'Model':<45} {'Provider':<15} {'Input $/M':<10} {'Output $/M':<11} {'Key'}"
    )
    print("-" * 90)

    for m in results:
        in_cost = f"${m.input_cost_per_m:.2f}" if m.input_cost_per_m else "-"
        out_cost = f"${m.output_cost_per_m:.2f}" if m.output_cost_per_m else "-"
        key_status = f"{GREEN}v{RESET}" if m.key_found else f"{RED}x{RESET}"
        print(f"{m.model:<45} {m.provider:<15} {in_cost:<10} {out_cost:<11} {key_status}")

    if total > limit:
        print()
        print(
            f"{DIM}Showing {limit} of {total} models. "
            f"Narrow your search: {CYAN}ace models <query>{RESET}"
            f"{DIM} or use {CYAN}--limit {total}{RESET}"
        )


def _cmd_validate(model: str) -> None:
    """``ace validate <model>`` — test a model connection."""
    load_dotenv()

    print(f"Validating {model}...", end="", flush=True)
    result = validate_connection(model)

    if result.success:
        print(
            f"\r{GREEN}v{RESET} Connected! "
            f"({model} via {result.provider}, {result.latency_ms}ms)"
        )
    else:
        print(f"\r{RED}x{RESET} {result.error}")
        suggestions = suggest_models(model)
        if suggestions:
            print("Did you mean:")
            for s in suggestions:
                print(f"  - {s}")
        sys.exit(1)
