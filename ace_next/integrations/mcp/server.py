import sys
import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ace_next.integrations.mcp.config import MCPServerConfig
from ace_next.integrations.mcp.registry import SessionRegistry
from ace_next.integrations.mcp.handlers import MCPHandlers
from ace_next.integrations.mcp.adapters import register_tools

def create_server() -> Server:
    config = MCPServerConfig()
    
    # Configure logging based on config.log_level
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("ace_mcp_server")
    logger.info("Starting ACE MCP Server...")
    logger.info(f"Safe mode: {config.safe_mode}")
    logger.info(f"Default model: {config.default_model}")
    
    registry = SessionRegistry(config)
    handlers = MCPHandlers(registry, config)
    
    server = Server("ace-mcp-server")
    register_tools(server, handlers)
    
    return server

async def run_server() -> None:
    try:
        server = create_server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    except Exception as e:
        print(f"Failed to start ACE MCP Server: {e}", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    """CLI Entrypoint for ace-mcp."""
    asyncio.run(run_server())

if __name__ == "__main__":
    main()
