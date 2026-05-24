# agent-rate-limiter

Per-tool call-rate enforcement for LLM agents. Sliding window, thread-safe, zero dependencies.

```python
from agent_rate_limiter import RateLimiter, RateLimitError

limiter = RateLimiter(max_calls=10, window_seconds=60.0)

def search(query: str) -> str:
    limiter.acquire("search")   # raises RateLimitError if over limit
    return do_search(query)
```

## Install

```bash
pip install agent-rate-limiter
```

## API

### `RateLimiter(max_calls, window_seconds, *, clock=None)`

```python
lim = RateLimiter(max_calls=10, window_seconds=60.0)
lim.acquire("tool_name")        # raises RateLimitError if over limit
lim.try_acquire("tool_name")    # returns bool, never raises
lim.remaining("tool_name")      # int: calls left in window
lim.reset("tool_name")          # clear one bucket
lim.reset()                     # clear all buckets
```

### `@rate_limited(max_calls, window_seconds, *, tool_name=None)`

Decorator. Works on regular `def` functions.

```python
@rate_limited(max_calls=5, window_seconds=30.0)
def lookup(key: str) -> str:
    ...
```

### `RateLimitRegistry`

```python
reg = RateLimitRegistry()
reg.add("search", max_calls=10, window_seconds=60.0)
reg.add("summarize", max_calls=100, window_seconds=60.0)
wrapped = reg.wrap_all({"search": search_fn, "summarize": summarize_fn})
```

Tools without a registered limit pass through unchanged.

### `RateLimitError`

Has `.tool_name`, `.max_calls`, `.window_seconds`, `.retry_after_seconds`.

## License

MIT
