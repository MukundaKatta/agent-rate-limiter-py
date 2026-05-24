"""agent-rate-limiter: per-tool call-rate enforcement for LLM agents.

Quick start::

    from agent_rate_limiter import RateLimiter, RateLimitError

    # Allow at most 10 calls per 60 seconds
    limiter = RateLimiter(max_calls=10, window_seconds=60.0)

    def search(query: str) -> str:
        limiter.acquire("search")
        return do_search(query)

Decorator style::

    from agent_rate_limiter import rate_limited

    @rate_limited(max_calls=5, window_seconds=30.0)
    def lookup(key: str) -> str:
        ...

Registry for bulk config::

    from agent_rate_limiter import RateLimitRegistry

    reg = RateLimitRegistry()
    reg.add("search", max_calls=10, window_seconds=60.0)
    reg.add("summarize", max_calls=100, window_seconds=60.0)
    wrapped = reg.wrap_all(tool_dict)
"""

from .core import RateLimiter, RateLimitError, RateLimitRegistry, rate_limited

__all__ = ["RateLimitError", "RateLimitRegistry", "RateLimiter", "rate_limited"]
__version__ = "0.1.0"
