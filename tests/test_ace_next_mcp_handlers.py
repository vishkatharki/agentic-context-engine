import pytest
from unittest.mock import MagicMock, patch
from ace_next.integrations.mcp.config import MCPServerConfig
from ace_next.integrations.mcp.registry import SessionRegistry
from ace_next.integrations.mcp.handlers import MCPHandlers
from ace_next.integrations.mcp.models import (
    AskRequest, LearnSampleRequest, LearnFeedbackRequest,
    SkillbookGetRequest, SkillbookSaveRequest, SkillbookLoadRequest,
    SampleItem
)
from ace_next.integrations.mcp.errors import (
    ForbiddenInSafeModeError, SaveLoadDisabledError, ValidationError,
)

@pytest.fixture
def config():
    return MCPServerConfig(safe_mode=False)

@pytest.fixture
def registry(config):
    return SessionRegistry(config)

@pytest.fixture
def handlers(registry, config):
    return MCPHandlers(registry, config)


# ── ace.ask ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_ask(handlers):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.ask.return_value = "answer"
        runner.skillbook.skills.return_value = [1, 2, 3]
        mock_runner_cls.from_model.return_value = runner

        req = AskRequest(session_id="s1", question="q")
        resp = await handlers.handle_ask(req)

        assert resp.answer == "answer"
        assert resp.skill_count == 3
        # applied_skill_ids was removed from the response model
        assert "applied_skill_ids" not in resp.model_fields
        runner.ask.assert_called_once()


@pytest.mark.asyncio
async def test_handle_ask_enforces_prompt_limit(handlers):
    handlers.config.max_prompt_chars = 10
    req = AskRequest(session_id="s1", question="12345678901")
    with pytest.raises(ValidationError):
        await handlers.handle_ask(req)


# ── ace.skillbook.get ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_skillbook_get(handlers, registry):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        mock_skill = MagicMock()
        mock_skill.id = "k1"
        mock_skill.content = "cont"
        mock_skill.section = "test"
        mock_skill.helpful = 1
        mock_skill.harmful = 0
        mock_skill.neutral = 0
        runner.skillbook.skills.return_value = [mock_skill]
        runner.skillbook.stats.return_value = {"skills": 1}
        mock_runner_cls.from_model.return_value = runner

        await registry.get_or_create("s1")
        req = SkillbookGetRequest(session_id="s1")
        resp = await handlers.handle_skillbook_get(req)

        assert len(resp.skills) == 1
        assert resp.skills[0].id == "k1"
        assert resp.stats["skills"] == 1


@pytest.mark.asyncio
async def test_handle_skillbook_get_uses_skill_type(handlers, registry):
    """When skills are actual Skill dataclass instances, use direct access."""
    from ace_next.core.skillbook import Skill

    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        skill = Skill(id="s1", section="topic-a", content="do X")
        runner.skillbook.skills.return_value = [skill]
        runner.skillbook.stats.return_value = {"skills": 1}
        mock_runner_cls.from_model.return_value = runner

        await registry.get_or_create("s1")
        req = SkillbookGetRequest(session_id="s1")
        resp = await handlers.handle_skillbook_get(req)

        assert resp.skills[0].id == "s1"
        assert resp.skills[0].topic == "topic-a"
        assert resp.skills[0].content == "do X"
        assert resp.skills[0].helpful == 0
        assert resp.skills[0].harmful == 0
        assert resp.skills[0].neutral == 0


# ── ace.learn.sample ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_learn_sample_safe_mode(handlers):
    handlers.config.safe_mode = True
    req = LearnSampleRequest(session_id="s1", samples=[SampleItem(question="q")])
    with pytest.raises(ForbiddenInSafeModeError):
        await handlers.handle_learn_sample(req)


@pytest.mark.asyncio
async def test_handle_learn_sample_enforces_runtime_sample_limit(handlers):
    handlers.config.max_samples_per_call = 1
    req = LearnSampleRequest(
        session_id="s1",
        samples=[SampleItem(question="q1"), SampleItem(question="q2")],
    )
    with pytest.raises(ValidationError):
        await handlers.handle_learn_sample(req)


# ── ace.learn.feedback ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_learn_feedback_uses_trace_learning(handlers):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.skillbook.skills.side_effect = [["a"], ["a", "b"]]
        runner.learn_from_feedback.return_value = False
        runner.learn_from_traces.return_value = []
        mock_runner_cls.from_model.return_value = runner

        req = LearnFeedbackRequest(
            session_id="s1",
            question="q",
            answer="a",
            feedback="good",
        )
        resp = await handlers.handle_learn_feedback(req)

        assert resp.learned is True
        assert resp.new_skill_count == 1
        runner.learn_from_traces.assert_called_once()


@pytest.mark.asyncio
async def test_handle_learn_feedback_always_reports_learned_true(handlers):
    """learned=True on success even when no new skills are created."""
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.skillbook.skills.side_effect = [["a"], ["a"]]
        runner.learn_from_feedback.return_value = True
        mock_runner_cls.from_model.return_value = runner

        req = LearnFeedbackRequest(
            session_id="s1",
            question="q",
            answer="a",
            feedback="good",
        )
        resp = await handlers.handle_learn_feedback(req)

        assert resp.learned is True
        assert resp.new_skill_count == 0


@pytest.mark.asyncio
async def test_handle_learn_feedback_trace_uses_context_not_reasoning(handlers):
    """Fallback trace must map context to 'context', not 'reasoning'."""
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.skillbook.skills.side_effect = [[], []]
        runner.learn_from_feedback.return_value = False
        runner.learn_from_traces.return_value = []
        mock_runner_cls.from_model.return_value = runner

        req = LearnFeedbackRequest(
            session_id="s1",
            question="q",
            answer="a",
            feedback="good",
            context="some background",
        )
        await handlers.handle_learn_feedback(req)

        trace = runner.learn_from_traces.call_args[0][0][0]
        assert trace["context"] == "some background"
        assert "reasoning" not in trace


# ── ace.skillbook.save/load ──────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_save_safe_mode(handlers, registry):
    handlers.config.safe_mode = True
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM"):
        await registry.get_or_create("s1")
        req = SkillbookSaveRequest(session_id="s1", path="/tmp/some")
        with pytest.raises(ForbiddenInSafeModeError):
            await handlers.handle_skillbook_save(req)


@pytest.mark.asyncio
async def test_handle_load_safe_mode(handlers):
    handlers.config.safe_mode = True
    req = SkillbookLoadRequest(session_id="s1", path="/tmp/some")
    with pytest.raises(ForbiddenInSafeModeError):
        await handlers.handle_skillbook_load(req)


@pytest.mark.asyncio
async def test_handle_save_load_disabled(handlers, registry):
    """allow_save_load=false with safe_mode=false raises SaveLoadDisabledError."""
    handlers.config.safe_mode = False
    handlers.config.allow_save_load = False
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM"):
        await registry.get_or_create("s1")
        with pytest.raises(SaveLoadDisabledError):
            await handlers.handle_skillbook_save(
                SkillbookSaveRequest(session_id="s1", path="/tmp/f")
            )
        with pytest.raises(SaveLoadDisabledError):
            await handlers.handle_skillbook_load(
                SkillbookLoadRequest(session_id="s1", path="/tmp/f")
            )


@pytest.mark.asyncio
async def test_handle_skillbook_save_rejects_path_outside_root(handlers, registry):
    handlers.config.skillbook_root = "/tmp/ace-root"
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM"):
        await registry.get_or_create("s1")
        req = SkillbookSaveRequest(session_id="s1", path="/tmp/not-allowed/file.json")
        with pytest.raises(ValidationError):
            await handlers.handle_skillbook_save(req)


@pytest.mark.asyncio
async def test_handle_skillbook_load_rejects_path_outside_root(handlers):
    handlers.config.skillbook_root = "/tmp/ace-root"
    req = SkillbookLoadRequest(session_id="s1", path="/tmp/not-allowed/file.json")
    with pytest.raises(ValidationError):
        await handlers.handle_skillbook_load(req)
