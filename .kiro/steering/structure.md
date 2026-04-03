# Project Structure

```
fin_agent/
├── main.py              # CLI entry point, argument parsing, chat loop, upgrade logic
├── config.py            # Config class: loads/saves .env and app_config.json, setup wizard
├── utils.py             # FinMarkdown (rich renderer), debug_print helper
├── backtest.py          # BacktestEngine: MA Cross, MACD, RSI, limit-up strategies
├── portfolio.py         # PortfolioManager: simulated holdings, P&L, persisted to portfolio.json
├── scheduler.py         # TaskScheduler: singleton, price alert polling, worker/background modes
├── notification.py      # NotificationManager: SMTP email sending (plain + HTML)
├── user_profile.py      # UserProfileManager: investment preferences, persisted to user_profile.json
│
├── agent/
│   └── core.py          # FinAgent: LLM chat loop, tool dispatch, streaming, session save/load
│
├── llm/
│   ├── base.py          # Abstract LLM base class
│   ├── factory.py       # LLMFactory: instantiates correct client from Config.LLM_PROVIDER
│   ├── deepseek_client.py   # DeepSeek-specific client
│   └── openai_client.py     # Generic OpenAI-compatible client (all other providers)
│
├── tools/
│   ├── tushare_tools.py     # All Tushare API wrappers + TOOLS_SCHEMA (OpenAI function definitions)
│   ├── technical_indicators.py  # MACD, RSI, KDJ, BOLL calculations; pattern detection
│   ├── portfolio_tools.py   # Tool wrappers + schema for portfolio operations
│   ├── scheduler_tools.py   # Tool wrappers + schema for price alert management
│   ├── profile_tools.py     # Tool wrappers + schema for user profile updates
│   └── local_report_tools.py    # Tool wrappers + schema for local PDF/CSV report reading
│
└── reports/             # Directory for user-placed local financial report files (PDF, CSV, Excel)
```

## Key Architectural Patterns

- Tool schema definition: Each `tools/` module exports a `*_TOOLS_SCHEMA` list (OpenAI function-calling format) and corresponding executor functions. `tushare_tools.py` aggregates all schemas into a single `TOOLS_SCHEMA` list and a single `execute_tool_call()` dispatcher used by `FinAgent`.

- Singleton scheduler: `TaskScheduler` uses `__new__` to enforce a single instance. In interactive mode it runs as a daemon thread; in `--worker` mode it runs blocking with a PID file heartbeat.

- Config is a class with only class-level attributes and `@classmethod` methods — it is not instantiated. `Config.load()` is called on module import.

- LLM streaming: `FinAgent.stream_chat()` is the canonical method, yielding typed event dicts (`content`, `thinking`, `tool_call`, `tool_result`, `answer`, `error`). The CLI `run()` method consumes this generator.

- All user data (portfolio, tasks, profile, sessions) is stored as JSON in the OS config directory, never in the project directory.

- Dates throughout the codebase use `YYYYMMDD` string format (Tushare convention).

- `total_mv` from Tushare is in units of 万元 (10k CNY). When filtering by 亿 (100M), multiply by `10000`.
