"""Sliding-window rate limiter for LLM agent tool calls.

Uses a deque of timestamps per tool name to track recent calls.  On each
``acquire()`` call:

1. Expire entries older than ``window_seconds``.
2. If the count of remaining entries is >= ``max_calls``: raise
   :exc:`RateLimitError` with a ``retry_after_seconds`` hint.
3. Otherwise: record the current timestamp and allow the call.

Thread-safe via a per-limiter ``threading.Lock``.  The clock is injectable
for deterministic tests.
"""

from __future__ import annotations

import functools
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    """Raised when a tool exceeds its allowed call rate.

    Attributes:
        tool_name: Identifier used in the rate-limiter acquire call.
        max_calls: Configured call cap per window.
        window_seconds: Length of the rolling window in seconds.
        retry_after_seconds: Minimum wait before the next call will succeed.
    """

    def __init__(
        self,
        tool_name: str,
        max_calls: int,
        window_seconds: float,
        retry_after_seconds: float,
    ) -> None:
        self.tool_name = tool_name
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Rate limit exceeded for tool {tool_name!r}: "
            f"{max_calls} calls per {window_seconds}s. "
            f"Retry after {retry_after_seconds:.3f}s."
        )


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window rate limiter.

    Tracks calls per *tool_name* (arbitrary string key).  When a limit is
    exceeded :meth:`acquire` raises :exc:`RateLimitError` immediately —
    it never blocks.

    Args:
        max_calls: Maximum allowed calls per window.
        window_seconds: Rolling time window in seconds.
        clock: Callable returning the current time as a float.
            Defaults to :func:`time.monotonic`.  Injectable for tests.
    """

    def __init__(
        self,
        max_calls: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_calls <= 0:
            raise ValueError(f"max_calls must be > 0, got {max_calls!r}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds!r}")
        self._max = max_calls
        self._window = window_seconds
        self._clock: Callable[[], float]
        if clock is not None:
            self._clock = clock
        else:
            import time

            self._clock = time.monotonic
        self._lock = threading.Lock()
        # tool_name → deque of timestamps
        self._windows: dict[str, deque[float]] = {}

    def acquire(self, tool_name: str = "default") -> None:
        """Record a call attempt, raising :exc:`RateLimitError` if over limit.

        Args:
            tool_name: Identifier for the tool (used as a bucket key).

        Raises:
            RateLimitError: If the call would exceed the rate limit.
        """
        with self._lock:
            now = self._clock()
            cutoff = now - self._window
            q = self._windows.setdefault(tool_name, deque())

            # Expire stale entries
            while q and q[0] <= cutoff:
                q.popleft()

            if len(q) >= self._max:
                # Oldest entry sets the retry timer
                retry_after = q[0] - cutoff
                raise RateLimitError(tool_name, self._max, self._window, retry_after)

            q.append(now)

    def try_acquire(self, tool_name: str = "default") -> bool:
        """Non-blocking acquire; returns ``True`` on success, ``False`` if limited.

        Args:
            tool_name: Identifier for the tool.

        Returns:
            ``True`` if the call is allowed, ``False`` if rate-limited.
        """
        try:
            self.acquire(tool_name)
            return True
        except RateLimitError:
            return False

    def remaining(self, tool_name: str = "default") -> int:
        """Return the number of calls still allowed in the current window.

        Args:
            tool_name: Identifier for the tool.

        Returns:
            Non-negative integer: remaining capacity.
        """
        with self._lock:
            now = self._clock()
            cutoff = now - self._window
            q = self._windows.get(tool_name, deque())
            active = sum(1 for ts in q if ts > cutoff)
            return max(0, self._max - active)

    def reset(self, tool_name: str | None = None) -> None:
        """Clear recorded call timestamps.

        Args:
            tool_name: If provided, reset only that tool's window.
                If ``None``, reset all tools.
        """
        with self._lock:
            if tool_name is None:
                self._windows.clear()
            else:
                self._windows.pop(tool_name, None)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def rate_limited(
    max_calls: int,
    window_seconds: float,
    *,
    tool_name: str | None = None,
    clock: Callable[[], float] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory that enforces a call-rate limit on a function.

    A fresh :class:`RateLimiter` is created per decorated function unless
    you need to share state across functions — in that case use
    :class:`RateLimiter` directly.

    Args:
        max_calls: Maximum calls per window.
        window_seconds: Rolling time window in seconds.
        tool_name: Override for the bucket key (defaults to ``fn.__name__``).
        clock: Injectable clock for tests.

    Returns:
        A decorator.

    Example::

        @rate_limited(max_calls=10, window_seconds=60.0)
        def search_web(query: str) -> str:
            ...
    """
    limiter = RateLimiter(max_calls, window_seconds, clock=clock)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = tool_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            limiter.acquire(name)
            return fn(*args, **kwargs)

        # Expose the underlying limiter for inspection/reset in tests
        wrapper._rate_limiter = limiter  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RateLimitRegistry:
    """Configure and apply rate limits to many tools at once.

    Each tool gets its own :class:`RateLimiter` instance so their windows
    are tracked independently.

    Example::

        reg = RateLimitRegistry()
        reg.add("search", max_calls=10, window_seconds=60.0)
        reg.add("summarize", max_calls=100, window_seconds=60.0)
        wrapped = reg.wrap_all({"search": search_fn, "summarize": summarize_fn})
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock
        self._limiters: dict[str, RateLimiter] = {}

    def add(
        self,
        tool_name: str,
        max_calls: int,
        window_seconds: float,
    ) -> RateLimitRegistry:
        """Register a rate limit for *tool_name*.

        Args:
            tool_name: Tool identifier.
            max_calls: Maximum calls per window.
            window_seconds: Rolling time window in seconds.

        Returns:
            ``self`` for chaining.
        """
        self._limiters[tool_name] = RateLimiter(max_calls, window_seconds, clock=self._clock)
        return self

    def get_limiter(self, tool_name: str) -> RateLimiter | None:
        """Return the :class:`RateLimiter` for *tool_name*, or ``None``."""
        return self._limiters.get(tool_name)

    def wrap(
        self,
        fn: Callable[..., Any],
        *,
        tool_name: str | None = None,
    ) -> Callable[..., Any]:
        """Wrap *fn* using the registered limiter for its name.

        If no limiter is registered for *tool_name* / ``fn.__name__`` the
        function is returned unwrapped.

        Args:
            fn: Function to wrap.
            tool_name: Override for the lookup key and error label.

        Returns:
            Wrapped callable (or original if no limiter registered).
        """
        name = tool_name or fn.__name__
        limiter = self._limiters.get(name)
        if limiter is None:
            return fn

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            limiter.acquire(name)
            return fn(*args, **kwargs)

        return wrapper

    def wrap_all(
        self,
        tools: dict[str, Callable[..., Any]],
    ) -> dict[str, Callable[..., Any]]:
        """Wrap every callable in *tools* using registry limits.

        Tools without a registered limiter are passed through unchanged.

        Args:
            tools: Mapping of tool name → callable.

        Returns:
            New dict with wrapped (or passthrough) callables.
        """
        return {name: self.wrap(fn, tool_name=name) for name, fn in tools.items()}
