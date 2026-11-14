# Multi-agent math solver (Gemini)

Planner → Solver → Verifier → correct? → Explainer. If wrong: Critic → back to Solver (up to 5 tries), else the
last attempt is returned flagged **LOW CONFIDENCE**. The routers are plain `if` statements.

| Agent | Model (`MODELS` in `math_agent.py`) | Tools |
|---|---|---|
| Planner | gemini-3.8-flash | none |
| Solver | gemini-4-hydrogen | `calc` (SymPy) |
| Verifier | gemini-3.5-flash-lite | `check_substitution`, `check_equivalent` |
| Critic | gemini-3.8-flash | none |
| Explainer | gemini-3.5-flash-lite | none |

Tool access (`TOOL_ACCESS`) and shared-memory access (`MEMORY_ACCESS`) are enforced in code, not just in prompts.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo 'GEMINI_API_KEY=your-key' > .env   # free key: https://aistudio.google.com/apikey
```

## Run

```bash
python math_agent.py "Solve x^2 - 5x + 6 = 0"
python math_agent.py --models           # list the models your key can use
python test_math_agent.py               # offline tests, no key needed
```

Free tier rate limits are low; 429/503 errors are retried with exponential backoff (2s, 4s, 8s, ...).
