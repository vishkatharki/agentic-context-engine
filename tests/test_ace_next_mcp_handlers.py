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
from ace_next.integrations.mcp.errors import ForbiddenInSafeModeError, ValidationError

@pytest.fixture
def config():
    return MCPServerConfig(safe_mode=False)

@pytest.fixture
def registry(config):
    return SessionRegistry(config)

@pytest.fixture
def handlers(registry, config):
    return MCPHandlers(registry, config)

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
        runner.ask.assert_called_once()

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
async def test_handle_learn_sample_safe_mode(handlers):
    handlers.config.safe_mode = True
    req = LearnSampleRequest(session_id="s1", samples=[SampleItem(question="q")])
    with pytest.raises(ForbiddenInSafeModeError):
        await handlers.handle_learn_sample(req)

@pytest.mark.asyncio
async def test_handle_save_load_safe_mode(handlers, registry):
    handlers.config.safe_mode = True
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM"):
        await registry.get_or_create("s1")
        req1 = SkillbookSaveRequest(session_id="s1", path="/tmp/some")
        with pytest.raises(ForbiddenInSafeModeError):
            await handlers.handle_skillbook_save(req1)
            
        req2 = SkillbookLoadRequest(session_id="s1", path="/tmp/some")
        with pytest.raises(ForbiddenInSafeModeError):
            await handlers.handle_skillbook_load(req2)


@pytest.mark.asyncio
async def test_handle_ask_enforces_prompt_limit(handlers):
    handlers.config.max_prompt_chars = 10
    req = AskRequest(session_id="s1", question="12345678901")
    with pytest.raises(ValidationError):
        await handlers.handle_ask(req)


@pytest.mark.asyncio
async def test_handle_learn_sample_enforces_runtime_sample_limit(handlers):
    handlers.config.max_samples_per_call = 1
    req = LearnSampleRequest(
        session_id="s1",
        samples=[SampleItem(question="q1"), SampleItem(question="q2")],
    )
    with pytest.raises(ValidationError):
        await handlers.handle_learn_sample(req)


@pytest.mark.asyncio
async def test_handle_learn_feedback_uses_trace_learning(handlers):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.skillbook.skills.side_effect = [["a"], ["a", "b"]]
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
