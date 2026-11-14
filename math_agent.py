"""Multi-agent math solver on Gemini.

Planner -> Solver -> Verifier -> correct? -> Explainer
                                  no -> Critic -> tries < MAX_TRIES? -> back to Solver
"""
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from sympy import N, Symbol, simplify, solve
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

# Free-tier text models as of Nov 2026 (Pro models are paid-only). Run `python math_agent.py --models` to re-check.
MODELS = {
    "planner": "gemini-3.8-flash",
    "solver": "gemini-4-hydrogen",
    "verifier": "gemini-3.5-flash-lite",
    "critic": "gemini-3.8-flash",  # must differ from the solver's model
    "explainer": "gemini-3.5-flash-lite",
}
MAX_TRIES = 5
MAX_TOOL_ROUNDS = 12
MAX_RETRIES = 6
BACKOFF_BASE = 2  # seconds; waits 2, 4, 8, 16, 32

# ---------------------------------------------------------------- tools

BLOCKED = ("__", "import", "lambda", "exec", "eval", "open")
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _parse(s):
    # parse_expr calls eval(), so reject anything that could escape SymPy
    if any(b in s.lower() for b in BLOCKED):
        raise ValueError(f"blocked expression: {s!r}")
    return parse_expr(s, transformations=TRANSFORMS)


def _side_diff(equation):
    lhs, _, rhs = equation.partition("=")
    return _parse(lhs) - _parse(rhs or "0")


def calc(expression):
    if "=" in expression and not re.search(r"[<>!=]=", expression):
        return f"solutions: {solve(_side_diff(expression))}"
    r = _parse(expression)
    if getattr(r, "is_number", False):
        return str(r) if r.is_Integer or r.is_Float else f"{r} ≈ {N(r)}"
    return str(simplify(r)) if hasattr(r, "free_symbols") else str(r)


def check_substitution(equation, variable, value):
    residual = simplify(_side_diff(equation).subs(Symbol(variable), _parse(value)))
    holds = residual == 0 or (residual.is_number and abs(N(residual)) < 1e-9)
    return f"residual: {residual}, holds: {str(bool(holds)).lower()}"


def check_equivalent(expr_a, expr_b):
    diff = simplify(_parse(expr_a) - _parse(expr_b))
    return f"difference: {diff}, equivalent: {str(diff == 0).lower()}"


TOOLS = {"calc": calc, "check_substitution": check_substitution, "check_equivalent": check_equivalent}
TOOL_ACCESS = {
    "planner": set(),
    "solver": {"calc"},
    "verifier": {"check_substitution", "check_equivalent"},
    "critic": set(),
    "explainer": set(),
}


def _decl(name, description, **params):
    return types.FunctionDeclaration(name=name, description=description, parameters_json_schema={
        "type": "object",
        "properties": {p: {"type": "string", "description": d} for p, d in params.items()},
        "required": list(params),
    })


DECLS = {
    "calc": _decl("calc", "SymPy calculator. An equation like 'x^2 - 4 = 0' is solved. Also arithmetic, "
                  "simplify(...), solve(...), integrate(f, x), diff(f, x), factor(...), expand(...).",
                  expression="SymPy expression or equation, ^ allowed for powers"),
    "check_substitution": _decl("check_substitution", "Plug a value into an equation; reports residual and whether it holds.",
                                equation="equation, e.g. 'x^2 - 5x + 6 = 0'", variable="variable name, e.g. 'x'",
                                value="value to substitute, e.g. '3' or '1/2'"),
    "check_equivalent": _decl("check_equivalent", "Check whether two expressions are mathematically equal (simplify(a-b)==0).",
                              expr_a="first expression", expr_b="second expression"),
}


def dispatch(role, name, args):
    """Run a tool call, enforcing TOOL_ACCESS. Errors go back to the model as text."""
    if name not in TOOL_ACCESS[role]:
        return f"PERMISSION DENIED: {role} is not allowed to use {name}"
    try:
        return TOOLS[name](**args)
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"

# ---------------------------------------------------------------- shared memory


@dataclass
class Memory:
    problem: str
    plan: str = ""
    attempts: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)
    feedback: list = field(default_factory=list)
    tries: int = 0


MEMORY_ACCESS = {
    "planner": {"read": {"problem"}, "write": {"plan"}},
    "solver": {"read": {"problem", "plan", "attempts", "feedback"}, "write": {"attempts"}},
    "verifier": {"read": {"problem", "latest_attempt"}, "write": {"verdicts"}},
    "critic": {"read": {"problem", "latest_attempt", "verdicts"}, "write": {"feedback"}},
    "explainer": {"read": {"problem", "latest_attempt"}, "write": set()},
}


def view(mem, role):
    fields = {**vars(mem), "latest_attempt": mem.attempts[-1] if mem.attempts else ""}
    return {k: v for k, v in fields.items() if k in MEMORY_ACCESS[role]["read"]}


def write(mem, role, name, value):
    if name not in MEMORY_ACCESS[role]["write"]:
        raise PermissionError(f"{role} may not write {name}")
    if isinstance(getattr(mem, name), list):
        getattr(mem, name).append(value)
    else:
        setattr(mem, name, value)


def render(fields):
    out = []
    for k, v in fields.items():
        if isinstance(v, list):
            v = "\n\n".join(f"[{i}] {x}" for i, x in enumerate(v, 1)) or "(none)"
        out.append(f"## {k}\n{v}")
    return "\n\n".join(out)

# ---------------------------------------------------------------- agents

PROMPTS = {
    "planner": "You are the Planner. Break the math problem into short numbered steps for a solver to follow. "
               "Do NOT solve it and do NOT compute any numbers. Output only the numbered steps.",
    "solver": "You are the Solver. Work through the plan step by step and show your work. Use the calc tool for ALL "
              "arithmetic and algebra; never compute in your head. If there is feedback from earlier attempts, fix "
              "exactly what it points out. End with a single line: FINAL ANSWER: <answer>",
    "verifier": "You are the Verifier. Decide if the latest attempt's FINAL ANSWER is correct. Never trust the "
                "attempt's arithmetic; check independently with your tools. For equations, check_substitution every "
                "claimed solution into the ORIGINAL equation, and confirm none are missing (e.g. check_equivalent the "
                "factored form built from the roots against the original polynomial). For word problems, do NOT reuse "
                "the attempt's formula: set up your own equation in a variable straight from the problem text (e.g. "
                "60*(t+2) = 80*t) and check_substitution the answer into it. End your reply with JSON exactly like "
                '{"correct": true, "evidence": "..."}',
    "critic": "You are the Critic. The latest attempt was judged WRONG. Using the Verifier's evidence, find the FIRST "
              "incorrect step and give specific feedback on what is wrong and how to fix that step. Do NOT write a full "
              "solution or state the final answer.",
    "explainer": "You are the Explainer. Rewrite the verified solution cleanly for a student, in short numbered steps "
                 "and plain language. Do NOT change any numbers or the final answer, and do not do new calculations.",
}


def generate(client, model, contents, config):
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            if e.code not in (429, 503) or attempt == MAX_RETRIES - 1:
                raise
            wait = BACKOFF_BASE * 2 ** attempt
            print(f"  [rate limited ({e.code}) on {model}, retrying in {wait}s]")
            time.sleep(wait)


def run_agent(client, role, mem):
    allowed = TOOL_ACCESS[role]
    config = types.GenerateContentConfig(
        system_instruction=PROMPTS[role],
        tools=[types.Tool(function_declarations=[DECLS[n] for n in sorted(allowed)])] if allowed else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=render(view(mem, role)))])]
    for _ in range(MAX_TOOL_ROUNDS):
        resp = generate(client, MODELS[role], contents, config)
        if not resp.function_calls:
            return resp.text or ""
        contents.append(resp.candidates[0].content)  # keeps Gemini's thought signatures
        parts = []
        for fc in resp.function_calls:
            args = dict(fc.args or {})
            result = dispatch(role, fc.name, args)
            shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(f"  [{role.title()} -> {fc.name}] {shown} => {result}")
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))
    print(f"  [{role.title()}] hit {MAX_TOOL_ROUNDS} tool rounds, stopping")
    return "(tool budget exhausted, no answer)"


def final_answer(attempt):
    m = re.findall(r"FINAL ANSWER:\s*(.+)", attempt)
    return m[-1].strip() if m else "(no FINAL ANSWER line)"


def parse_verdict(text):
    for m in reversed(re.findall(r"\{[^{}]*\"correct\"[^{}]*\}", text)):
        try:
            v = json.loads(m)
            return {"correct": v.get("correct") is True, "evidence": str(v.get("evidence", ""))}
        except json.JSONDecodeError:
            pass
    return {"correct": False, "evidence": "verifier gave no parseable verdict: " + text[-300:]}


def make_client():
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY (env var or .env file). Get a free key at https://aistudio.google.com/apikey")
    return genai.Client(api_key=key)


def solve_problem(problem, client=None):
    client = client or make_client()
    mem = Memory(problem)
    print(f"\nPROBLEM: {problem}\n[Planner] planning...")
    write(mem, "planner", "plan", run_agent(client, "planner", mem))
    print(mem.plan.strip())
    while True:
        mem.tries += 1
        print(f"\n[Solver] attempt {mem.tries}...")
        write(mem, "solver", "attempts", run_agent(client, "solver", mem))
        print(f"[Solver] FINAL ANSWER: {final_answer(mem.attempts[-1])}")

        print("[Verifier] checking...")
        verdict = parse_verdict(run_agent(client, "verifier", mem))
        write(mem, "verifier", "verdicts", verdict)
        print(f"[Verifier] correct: {str(verdict['correct']).lower()} | {verdict['evidence']}")

        if verdict["correct"]:  # router 1
            print("[Explainer] writing explanation...")
            return {"answer": final_answer(mem.attempts[-1]), "confidence": "HIGH", "tries": mem.tries,
                    "explanation": run_agent(client, "explainer", mem)}
        if mem.tries >= MAX_TRIES:  # router 2 (checked before the Critic so we don't spend a call on unused feedback)
            print(f"[Router] {MAX_TRIES} tries used, giving up")
            return {"answer": final_answer(mem.attempts[-1]), "confidence": "LOW", "tries": mem.tries,
                    "explanation": mem.attempts[-1]}
        print("[Critic] reviewing...")
        write(mem, "critic", "feedback", run_agent(client, "critic", mem))
        print(f"[Critic] {mem.feedback[-1].strip()}")


def main():
    if sys.argv[1:] == ["--models"]:
        client = make_client()  # keep a reference: a garbage-collected Client closes its HTTP session mid-pagination
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                print(m.name)
        return
    problem = " ".join(sys.argv[1:]) or "Solve x^2 - 5x + 6 = 0"
    r = solve_problem(problem)
    flag = "" if r["confidence"] == "HIGH" else "  ** LOW CONFIDENCE: not verified **"
    print(f"\n{'=' * 60}\nANSWER: {r['answer']}   ({r['tries']} tries){flag}\n{'=' * 60}\n{r['explanation'].strip()}")


if __name__ == "__main__":
    main()
