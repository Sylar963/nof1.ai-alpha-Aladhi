# Skill: nof1-stack-and-testing

Use this before changing dependencies, touching tests, adding fixtures, or generating stack-specific code snippets for this repository.

## Source Of Truth

- Dependency compatibility: `requirements.txt`
- Test bootstrap: `tests/conftest.py`
- Project instructions: `AGENTS.md`, `CLAUDE.md`
- Runtime observed in this workspace:
  - Python `3.12.3`
  - `pytest 9.0.3`

Treat `requirements.txt` as the compatibility contract. The file uses lower bounds with major-version caps, so do not assume APIs from newer major releases.

## Stack Versions To Target

Use code and docs compatible with these ranges:

| Package | Version range |
|---|---|
| `hyperliquid-python-sdk` | `>=0.20.0,<0.21.0` |
| `python-dotenv` | `>=1.1.1,<2.0.0` |
| `web3` | `>=7.14.0,<8.0.0` |
| `aiohttp` | `>=3.13.1,<4.0.0` |
| `openai` | `>=2.5.0,<3.0.0` |
| `requests` | `>=2.32.5,<3.0.0` |
| `thalex` | `>=0.1.0` |
| `pyjwt[crypto]` | `>=2.8.0` |
| `nicegui` | `>=2.0.0` |
| `plotly` | `>=5.18.0` |
| `pandas` | `>=2.1.0` |
| `pywebview` | `>=5.0.0` |
| `sqlalchemy` | `>=2.0.0` |
| `pyinstaller` | `>=6.0.0` |
| `pytest` | `>=7.4.0` |
| `pytest-asyncio` | `>=0.21.0` |

## Practical Defaults

- Write Python compatible with `3.12`.
- Use `pytest` plus `pytest.mark.asyncio` for async tests.
- Assume NiceGUI desktop app patterns already in the repo; do not invent a web stack that is not present.
- There is no configured formatter, linter, or type checker in repo-level docs. Do not claim one exists unless you add it.

## Testing And Fixture Patterns

### What The Repo Actually Does

- `tests/conftest.py` is intentionally tiny and only adds the repo root to `sys.path`.
- Most fixtures are local to the test module, not centralized.
- Tests prefer lightweight in-memory fakes for adapters and pipelines when behavior spans multiple calls.
- Tests use `AsyncMock` and `MagicMock` for narrow seams and callback verification.
- Tests patch config and environment with `monkeypatch`, not manual global cleanup.
- File-backed state uses `tmp_path`; ephemeral databases use either `tmp_path` SQLite files or `sqlite:///:memory:`.

### Fixture Placement Rule

- Put a fixture in the same test module when it serves one file.
- Only move fixtures into `tests/conftest.py` when they are reused broadly across multiple modules.
- Prefer small, descriptive fixtures like `adapter`, `store`, `session`, `synthetic_chain`, `example_context`.

### Preferred Test Doubles

- For exchange or pipeline flows, create small fake classes with only the methods the test needs.
- Keep state on the fake instance so assertions can inspect recorded calls.
- For one-off async behavior, use `AsyncMock` instead of building a fake class.

Examples already in repo:

- End-to-end fake adapters: `tests/test_options_strategies.py`
- Builder/pipeline fakes: `tests/test_options_builder.py`, `tests/test_options_pipeline_integration.py`
- Async stub adapters via fixtures: `tests/test_thalex_greeks.py`, `tests/test_thalex_subscriptions.py`
- DB session fixture: `tests/test_options_trade_history.py`

### Monkeypatch Patterns

- Patch config dict entries with `monkeypatch.setitem(CONFIG, "key", value)`.
- Patch environment variables with `monkeypatch.setenv("NAME", value)`.
- Patch module or instance behavior with `monkeypatch.setattr(...)`.
- Prefer monkeypatch over direct mutation when the test needs automatic cleanup.

### Async Test Pattern

- Mark coroutine tests with `@pytest.mark.asyncio`.
- Keep async tests deterministic: fake network responses, fake clocks, or direct state injection.
- Avoid real network IO in tests.

### Data And Assertion Style

- Use fixed timestamps and dates where scenario meaning matters.
- Use inline payload dictionaries for decision-schema and API-shape tests.
- Use `pytest.approx(...)` for floats, prices, IVs, and greeks.
- Assert behavior, not implementation trivia: order side, routing, stored payload shape, visible state transitions.

### Database Test Pattern

- For ORM schema checks, use `create_engine("sqlite:///:memory:")`.
- For stateful stores, prefer `tmp_path / "name.db"` so the test exercises actual file-backed behavior.
- Open and close sessions inside fixtures with `yield` where needed.

## When Adding New Tests

- Mirror the surrounding file's style before introducing a new pattern.
- Reuse existing fake helpers from another test module if that keeps the test smaller and clearer; this repo already does that in `tests/test_options_pipeline_integration.py`.
- Add comments only when a test is encoding a non-obvious production constraint or a bug guard.
- Keep tests offline and deterministic.

## What To Avoid

- Do not add broad shared fixtures for convenience when a local fixture is enough.
- Do not introduce external test helpers or factory libraries unless the repo already uses them.
- Do not rely on live exchange credentials, live websockets, or live market data in tests.
- Do not use post-`pytest 9` assumptions that are not required here; keep usage conservative and compatible with the declared bounds.
