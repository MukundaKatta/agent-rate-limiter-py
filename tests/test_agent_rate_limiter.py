"""Tests for agent-rate-limiter."""

from __future__ import annotations

import threading
import time

import pytest

from agent_rate_limiter import RateLimiter, RateLimitError, RateLimitRegistry, rate_limited

# ---------------------------------------------------------------------------
# Clock helper
# ---------------------------------------------------------------------------


def make_clock(start: float = 0.0):
    """Return (clock_fn, advance_fn) for deterministic tests."""
    _t = [start]

    def clock() -> float:
        return _t[0]

    def advance(seconds: float) -> None:
        _t[0] += seconds

    return clock, advance


# ---------------------------------------------------------------------------
# RateLimiter — basic
# ---------------------------------------------------------------------------


def test_allow_calls_under_limit():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=3, window_seconds=60.0, clock=clock)
    lim.acquire()
    lim.acquire()
    lim.acquire()  # 3rd call — should pass


def test_exceed_limit_raises():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=2, window_seconds=60.0, clock=clock)
    lim.acquire()
    lim.acquire()
    with pytest.raises(RateLimitError):
        lim.acquire()


def test_error_attributes():
    clock, _ = make_clock(start=100.0)
    lim = RateLimiter(max_calls=1, window_seconds=30.0, clock=clock)
    lim.acquire()
    with pytest.raises(RateLimitError) as exc_info:
        lim.acquire()
    err = exc_info.value
    assert err.tool_name == "default"
    assert err.max_calls == 1
    assert err.window_seconds == 30.0
    assert err.retry_after_seconds > 0


def test_error_message_contains_tool_name():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=1, window_seconds=10.0, clock=clock)
    lim.acquire("my_tool")
    with pytest.raises(RateLimitError, match="my_tool"):
        lim.acquire("my_tool")


def test_separate_buckets_per_tool():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=1, window_seconds=60.0, clock=clock)
    lim.acquire("tool_a")
    # tool_b is a separate bucket — should not be blocked
    lim.acquire("tool_b")


# ---------------------------------------------------------------------------
# RateLimiter — sliding window
# ---------------------------------------------------------------------------


def test_old_calls_expire():
    clock, advance = make_clock(start=0.0)
    lim = RateLimiter(max_calls=2, window_seconds=10.0, clock=clock)
    lim.acquire()
    lim.acquire()
    # Advance past the window
    advance(11.0)
    # Old calls expired — should succeed again
    lim.acquire()
    lim.acquire()


def test_partial_expiry():
    clock, advance = make_clock(start=0.0)
    lim = RateLimiter(max_calls=3, window_seconds=10.0, clock=clock)
    lim.acquire()  # t=0
    advance(5.0)
    lim.acquire()  # t=5
    lim.acquire()  # t=5 (3rd — at limit)
    with pytest.raises(RateLimitError):
        lim.acquire()  # still over limit at t=5
    advance(6.0)  # t=11 → first call (t=0) has expired
    lim.acquire()  # now 2 active (t=5, t=5) → under limit of 3


def test_retry_after_is_positive():
    clock, _ = make_clock(start=0.0)
    lim = RateLimiter(max_calls=1, window_seconds=30.0, clock=clock)
    lim.acquire()
    with pytest.raises(RateLimitError) as exc_info:
        lim.acquire()
    assert exc_info.value.retry_after_seconds > 0


def test_retry_after_decreases_with_time():
    clock, advance = make_clock(start=0.0)
    lim = RateLimiter(max_calls=1, window_seconds=30.0, clock=clock)
    lim.acquire()
    with pytest.raises(RateLimitError) as exc_info:
        lim.acquire()
    first_retry = exc_info.value.retry_after_seconds

    advance(5.0)
    with pytest.raises(RateLimitError) as exc_info2:
        lim.acquire()
    second_retry = exc_info2.value.retry_after_seconds

    assert second_retry < first_retry


# ---------------------------------------------------------------------------
# RateLimiter — try_acquire
# ---------------------------------------------------------------------------


def test_try_acquire_returns_true_when_allowed():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=5, window_seconds=60.0, clock=clock)
    assert lim.try_acquire() is True


def test_try_acquire_returns_false_when_limited():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=1, window_seconds=60.0, clock=clock)
    lim.try_acquire()
    assert lim.try_acquire() is False


def test_try_acquire_does_not_raise():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=1, window_seconds=60.0, clock=clock)
    lim.try_acquire()
    # Should not raise, even when limited
    result = lim.try_acquire()
    assert result is False


# ---------------------------------------------------------------------------
# RateLimiter — remaining
# ---------------------------------------------------------------------------


def test_remaining_full_window():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=5, window_seconds=60.0, clock=clock)
    assert lim.remaining() == 5


def test_remaining_after_calls():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=5, window_seconds=60.0, clock=clock)
    lim.acquire()
    lim.acquire()
    assert lim.remaining() == 3


def test_remaining_at_zero():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=2, window_seconds=60.0, clock=clock)
    lim.acquire()
    lim.acquire()
    assert lim.remaining() == 0


def test_remaining_recovers_after_expiry():
    clock, advance = make_clock(start=0.0)
    lim = RateLimiter(max_calls=2, window_seconds=10.0, clock=clock)
    lim.acquire()
    lim.acquire()
    assert lim.remaining() == 0
    advance(11.0)
    assert lim.remaining() == 2


def test_remaining_separate_buckets():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=3, window_seconds=60.0, clock=clock)
    lim.acquire("a")
    assert lim.remaining("a") == 2
    assert lim.remaining("b") == 3


# ---------------------------------------------------------------------------
# RateLimiter — reset
# ---------------------------------------------------------------------------


def test_reset_clears_single_tool():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=1, window_seconds=60.0, clock=clock)
    lim.acquire("tool_a")
    lim.reset("tool_a")
    lim.acquire("tool_a")  # should work again


def test_reset_all():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=1, window_seconds=60.0, clock=clock)
    lim.acquire("a")
    lim.acquire("b")
    lim.reset()
    lim.acquire("a")
    lim.acquire("b")


def test_reset_nonexistent_tool_is_noop():
    clock, _ = make_clock()
    lim = RateLimiter(max_calls=5, window_seconds=60.0, clock=clock)
    lim.reset("never_used")  # should not raise


# ---------------------------------------------------------------------------
# RateLimiter — validation
# ---------------------------------------------------------------------------


def test_zero_max_calls_raises():
    with pytest.raises(ValueError, match="max_calls must be > 0"):
        RateLimiter(max_calls=0, window_seconds=60.0)


def test_zero_window_seconds_raises():
    with pytest.raises(ValueError, match="window_seconds must be > 0"):
        RateLimiter(max_calls=5, window_seconds=0.0)


# ---------------------------------------------------------------------------
# RateLimiter — thread safety
# ---------------------------------------------------------------------------


def test_concurrent_calls_respect_limit():
    """Multiple threads racing — total successes should not exceed max_calls."""
    lim = RateLimiter(max_calls=5, window_seconds=60.0)
    successes = []
    errors = []

    def task():
        try:
            lim.acquire("shared")
            successes.append(1)
        except RateLimitError:
            errors.append(1)

    threads = [threading.Thread(target=task) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successes) == 5
    assert len(errors) == 5


# ---------------------------------------------------------------------------
# rate_limited decorator
# ---------------------------------------------------------------------------


def test_decorator_allows_calls_under_limit():
    clock, _ = make_clock()

    @rate_limited(max_calls=3, window_seconds=60.0, clock=clock)
    def fn(x: int) -> int:
        return x * 2

    assert fn(1) == 2
    assert fn(2) == 4
    assert fn(3) == 6


def test_decorator_raises_when_exceeded():
    clock, _ = make_clock()

    @rate_limited(max_calls=2, window_seconds=60.0, clock=clock)
    def fn() -> str:
        return "ok"

    fn()
    fn()
    with pytest.raises(RateLimitError):
        fn()


def test_decorator_custom_tool_name():
    clock, _ = make_clock()

    @rate_limited(max_calls=1, window_seconds=60.0, tool_name="custom", clock=clock)
    def fn() -> None:
        pass

    fn()
    with pytest.raises(RateLimitError) as exc_info:
        fn()
    assert exc_info.value.tool_name == "custom"


def test_decorator_preserves_metadata():
    @rate_limited(max_calls=5, window_seconds=60.0)
    def my_tool(x: int) -> int:
        """Docstring."""
        return x

    assert my_tool.__name__ == "my_tool"
    assert my_tool.__doc__ == "Docstring."


def test_decorator_exposes_limiter():
    @rate_limited(max_calls=5, window_seconds=60.0)
    def fn() -> None:
        pass

    assert isinstance(fn._rate_limiter, RateLimiter)


def test_decorator_propagates_exception():
    @rate_limited(max_calls=5, window_seconds=60.0)
    def fail() -> None:
        raise ValueError("inner error")

    with pytest.raises(ValueError, match="inner error"):
        fail()


# ---------------------------------------------------------------------------
# RateLimitRegistry
# ---------------------------------------------------------------------------


def test_registry_wrap_applies_limit():
    clock, _ = make_clock()

    def search(q: str) -> str:
        return q

    reg = RateLimitRegistry(clock=clock)
    reg.add("search", max_calls=1, window_seconds=60.0)
    wrapped = reg.wrap(search, tool_name="search")

    wrapped("hello")
    with pytest.raises(RateLimitError):
        wrapped("world")


def test_registry_wrap_passthrough_if_not_registered():
    def noop() -> str:
        return "ok"

    reg = RateLimitRegistry()
    wrapped = reg.wrap(noop)
    # No limiter registered — should pass through unbounded
    for _ in range(20):
        assert wrapped() == "ok"


def test_registry_wrap_all():
    clock, _ = make_clock()

    def a() -> int:
        return 1

    def b() -> int:
        return 2

    reg = RateLimitRegistry(clock=clock)
    reg.add("a", max_calls=1, window_seconds=60.0)
    wrapped = reg.wrap_all({"a": a, "b": b})

    assert wrapped["a"]() == 1
    with pytest.raises(RateLimitError):
        wrapped["a"]()

    # b has no limiter — unbounded
    for _ in range(5):
        assert wrapped["b"]() == 2


def test_registry_get_limiter_registered():
    reg = RateLimitRegistry()
    reg.add("tool", max_calls=5, window_seconds=30.0)
    assert isinstance(reg.get_limiter("tool"), RateLimiter)


def test_registry_get_limiter_not_registered():
    reg = RateLimitRegistry()
    assert reg.get_limiter("missing") is None


def test_registry_chaining():
    reg = RateLimitRegistry()
    result = reg.add("a", 5, 60.0).add("b", 10, 60.0)
    assert result is reg
    assert reg.get_limiter("a") is not None
    assert reg.get_limiter("b") is not None


def test_registry_separate_limiters_per_tool():
    clock, _ = make_clock()
    reg = RateLimitRegistry(clock=clock)
    reg.add("x", max_calls=1, window_seconds=60.0)
    reg.add("y", max_calls=1, window_seconds=60.0)

    wrapped_x = reg.wrap(lambda: "x", tool_name="x")
    wrapped_y = reg.wrap(lambda: "y", tool_name="y")

    wrapped_x()
    wrapped_y()

    with pytest.raises(RateLimitError):
        wrapped_x()
    with pytest.raises(RateLimitError):
        wrapped_y()


# ---------------------------------------------------------------------------
# RateLimitError
# ---------------------------------------------------------------------------


def test_error_is_exception():
    err = RateLimitError("fn", 5, 30.0, 15.0)
    assert isinstance(err, Exception)


def test_error_message_contains_details():
    err = RateLimitError("web_search", 10, 60.0, 12.5)
    msg = str(err)
    assert "web_search" in msg
    assert "10" in msg
    assert "60.0" in msg


def test_real_clock_integration():
    """Smoke test with real monotonic clock."""
    lim = RateLimiter(max_calls=3, window_seconds=1.0)
    lim.acquire("real")
    lim.acquire("real")
    lim.acquire("real")
    with pytest.raises(RateLimitError):
        lim.acquire("real")
    time.sleep(1.1)
    lim.acquire("real")  # window cleared
