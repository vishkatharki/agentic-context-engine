import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any

from ace_next.runners import ACELiteLLM
from ace_next.integrations.mcp.config import MCPServerConfig
from ace_next.integrations.mcp.errors import SessionNotFoundError

@dataclass
class Session:
    session_id: str
    runner: ACELiteLLM
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class SessionRegistry:
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._sessions: Dict[str, Session] = {}
        self._registry_lock = asyncio.Lock()

    async def get_or_create(self, session_id: str, model: str | None = None, **runner_kwargs: Any) -> Session:
        """Get an existing session or create a new one, sweeping expired sessions first."""
        await self._sweep_expired()
        
        async with self._registry_lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_accessed = time.time()
                return session
            
            # Create new runner
            target_model = model or self.config.default_model
            runner = ACELiteLLM.from_model(target_model, **runner_kwargs)
            session = Session(session_id=session_id, runner=runner)
            self._sessions[session_id] = session
            return session

    async def get(self, session_id: str) -> Session:
        """Get an existing session. Raises SessionNotFoundError if not found."""
        await self._sweep_expired()
        
        async with self._registry_lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            
            session = self._sessions[session_id]
            session.last_accessed = time.time()
            return session

    async def delete(self, session_id: str) -> None:
        """Delete a session if it exists."""
        async with self._registry_lock:
            self._sessions.pop(session_id, None)

    async def _sweep_expired(self) -> None:
        """Remove sessions that have exceeded the TTL."""
        now = time.time()
        ttl = self.config.session_ttl_seconds
        
        async with self._registry_lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if now - session.last_accessed > ttl
            ]
            for sid in expired:
                del self._sessions[sid]
