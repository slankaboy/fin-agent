# Tech Stack

## Language & Runtime
- Python 3.8+
- Packaged with `setuptools`, distributed via PyPI as `fin-agent`

## Key Libraries
| Library | Purpose |
|---|---|
| `tushare` | Chinese financial market data (stocks, indices, macros, etc.) |
| `openai` | LLM API client (used for all providers via OpenAI-compatible interface) |
| `pandas` / `numpy` | Data processing and indicator calculations |
| `python-dotenv` | Environment variable loading |
| `rich` | Terminal markdown rendering and live output |
| `colorama` | Cross-platform terminal color support |
| `schedule` | Background task scheduling (price alerts) |
| `pdfplumber` | Local PDF financial report parsing |
| `psutil` | Process management utilities |

## LLM Integration
- All LLM providers use the OpenAI SDK (`openai` package)
- `DeepSeekClient` wraps the DeepSeek API
- `OpenAICompatibleClient` handles all other providers (Moonshot, ZhipuAI, Qwen, Yi, SiliconFlow, OpenRouter, local Ollama/LM Studio)
- Provider selection via `LLMFactory` based on `LLM_PROVIDER` config value
- Streaming responses are the default; non-streaming is a fallback

## Configuration
- User config stored in OS-appropriate directory:
  - macOS/Linux: `~/.config/fin-agent/`
  - Windows: `%APPDATA%/fin-agent/`
- Core settings (API keys, LLM provider) in `.env` file in config dir
- App settings (shortcuts, etc.) in `app_config.json`
- Portfolio data: `portfolio.json`
- Scheduled tasks: `tasks.json`
- User profile: `user_profile.json`
- Session history: `sessions/last_session.json`
- Local `.env` in working directory is used as fallback (dev mode)

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in editable/dev mode
pip install -e .

# Run the agent
fin-agent

# Run as Python module (useful in dev)
python -m fin_agent.main

# Clear config and re-run setup wizard
fin-agent --clear-token

# Run worker mode (background price alert scheduler)
fin-agent --worker

# Build distribution
python -m build

# Publish to PyPI
python publish.py
```

## Testing
No test suite is currently defined in the project.
