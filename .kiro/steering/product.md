# Fin-Agent Product Overview

Fin-Agent is a Chinese-market intelligent financial analysis assistant powered by LLMs and Tushare financial data. Users interact via natural language (primarily Chinese) to query stock prices, analyze financials, screen stocks, run strategy backtests, manage a simulated portfolio, and set price alerts.

## Key Capabilities
- Real-time and historical stock/index/ETF/futures/bond data via Tushare
- Macroeconomic data: GDP, CPI, M2, interest rates
- Technical analysis: MACD, RSI, KDJ, Bollinger Bands, pattern detection
- Smart stock screener via natural language conditions
- Strategy backtesting (MA Cross, MACD, RSI)
- Simulated portfolio management with P&L tracking
- Price alert monitoring with email notifications
- User preference memory (investment style, sector preferences)
- Multi-LLM support: DeepSeek (default), Moonshot, ZhipuAI, Qwen, Yi, SiliconFlow, OpenRouter, local models (Ollama/LM Studio)

## Distribution
Published as `fin-agent` on PyPI. Entry point: `fin-agent` CLI command. Also has a companion Electron desktop app (`fin-agent-desktop`).
