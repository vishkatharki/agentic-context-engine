import json
import mcp.types as types
from mcp.server import Server

from ace_next.integrations.mcp.handlers import MCPHandlers
from ace_next.integrations.mcp.models import (
    AskRequest, LearnSampleRequest, LearnFeedbackRequest,
    SkillbookGetRequest, SkillbookSaveRequest, SkillbookLoadRequest
)
from ace_next.integrations.mcp.errors import map_error_to_mcp

def register_tools(server: Server, handlers: MCPHandlers) -> None:
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="ace.ask",
                description="Ask a question and get a response from ACE.",
                inputSchema=AskRequest.model_json_schema()
            ),
            types.Tool(
                name="ace.learn.sample",
                description="Provide sample questions/answers for ACE to learn from.",
                inputSchema=LearnSampleRequest.model_json_schema()
            ),
            types.Tool(
                name="ace.learn.feedback",
                description="Provide feedback on an ACE answer.",
                inputSchema=LearnFeedbackRequest.model_json_schema()
            ),
            types.Tool(
                name="ace.skillbook.get",
                description="Get statistics and skills from the active skillbook.",
                inputSchema=SkillbookGetRequest.model_json_schema()
            ),
            types.Tool(
                name="ace.skillbook.save",
                description="Save the active skillbook to disk.",
                inputSchema=SkillbookSaveRequest.model_json_schema()
            ),
            types.Tool(
                name="ace.skillbook.load",
                description="Load a skillbook from disk into the session.",
                inputSchema=SkillbookLoadRequest.model_json_schema()
            )
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource] | types.CallToolResult:
        args = arguments or {}
        try:
            if name == "ace.ask":
                req = AskRequest(**args)
                resp = await handlers.handle_ask(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]
                
            elif name == "ace.learn.sample":
                req = LearnSampleRequest(**args)
                resp = await handlers.handle_learn_sample(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]
                
            elif name == "ace.learn.feedback":
                req = LearnFeedbackRequest(**args)
                resp = await handlers.handle_learn_feedback(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]
                
            elif name == "ace.skillbook.get":
                req = SkillbookGetRequest(**args)
                resp = await handlers.handle_skillbook_get(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]
                
            elif name == "ace.skillbook.save":
                req = SkillbookSaveRequest(**args)
                resp = await handlers.handle_skillbook_save(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]
                
            elif name == "ace.skillbook.load":
                req = SkillbookLoadRequest(**args)
                resp = await handlers.handle_skillbook_load(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]
                
            else:
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            mcp_err = map_error_to_mcp(e)
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=json.dumps(mcp_err))]
            )
