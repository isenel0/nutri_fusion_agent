"""Abstract base contract for all meal-analysis agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from schemas import AgentResponse


class BaseAgent(ABC):
    """Defines the async processing interface every agent must implement."""

    @abstractmethod
    async def process(self, input_data: Any) -> AgentResponse:
        """Process agent-specific input and return standardized output."""
        raise NotImplementedError
