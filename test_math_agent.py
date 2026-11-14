"""Offline tests: fake Gemini client, no network. Run: python test_math_agent.py (or pytest)."""
from google.genai import errors, types

import math_agent as ma

ma.BACKOFF_BASE = 0


def text(s):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=s)]))])


def call(name, **args):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]))])


class FakeClient:
    """Routes each request to handlers[role](contents) based on the system prompt."""

    def __init__(self, **handlers):
        self.handlers, self.calls = handlers, []
        self.models = self

    def generate_content(self, model, contents, config):
        role = next(r for r, p in ma.PROMPTS.items() if p == config.system_instruction)
        self.calls.append((role, list(contents)))
        return self.handlers[role](contents)


def last_text(contents):
    return " ".join(p.text or str(p.function_response.response) for p in contents[-1].parts)


def solver_with_tool(contents):
    # first turn: call calc; second turn (after tool result): answer
    if len(contents) == 1:
        return call("calc", expression="x^2 - 5x + 6 = 0")
    return text("roots via calc\nFINAL ANSWER: x = 2, x = 3")


def test_tools():
    assert ma.calc("x^2 - 4 = 0") == "solutions: [-2, 2]"
    assert ma.calc("integrate(x^2, x)") == "x**3/3"
    assert ma.calc("diff(sin(x), x)") == "cos(x)"
    assert ma.calc("2(3+4)") == "14"
    assert ma.check_substitution("x^2 - 5x + 6 = 0", "x", "3") == "residual: 0, holds: true"
    assert ma.check_substitution("x^2 - 5x + 6 = 0", "x", "4") == "residual: 2, holds: false"
    assert ma.check_equivalent("(x-2)(x-3)", "x^2 - 5x + 6").endswith("equivalent: true")
    assert ma.check_equivalent("(x-2)(x-4)", "x^2 - 5x + 6").endswith("equivalent: false")
    for bad in ["__import__('os').system('ls')", "(lambda: 1)()", "open('/etc/passwd')", "eval('1')"]:
        assert ma.dispatch("solver", "calc", {"expression": bad}).startswith("error: ValueError: blocked")


def test_tool_permissions():
    assert ma.dispatch("solver", "calc", {"expression": "1+1"}) == "2"
    for role, tool in [("planner", "calc"), ("solver", "check_substitution"), ("verifier", "calc"),
                       ("critic", "calc"), ("explainer", "check_equivalent")]:
        assert ma.dispatch(role, tool, {}).startswith("PERMISSION DENIED"), (role, tool)
    # a model that calls a tool it wasn't given gets rejected inside the real agent loop
    fake = FakeClient(verifier=lambda c: call("calc", expression="1+1") if len(c) == 1 else text(last_text(c)))
    out = ma.run_agent(fake, "verifier", ma.Memory("p", attempts=["a"]))
    assert "PERMISSION DENIED" in out


def test_memory_permissions():
    mem = ma.Memory("solve it", attempts=["a1", "a2"])
    assert ma.view(mem, "verifier") == {"problem": "solve it", "latest_attempt": "a2"}
    assert set(ma.view(mem, "planner")) == {"problem"}
    for role, name in [("verifier", "plan"), ("solver", "verdicts"), ("critic", "attempts"), ("explainer", "feedback"),
                       ("planner", "attempts")]:
        try:
            ma.write(mem, role, name, "x")
            raise AssertionError(f"{role} wrote {name}")
        except PermissionError:
            pass
    ma.write(mem, "critic", "feedback", "fix step 2")
    assert mem.feedback == ["fix step 2"]


def test_retry_loop_fixes_wrong_answer():
    verdicts = iter(['{"correct": false, "evidence": "x=4 gives residual 2"}', '{"correct": true, "evidence": "ok"}'])
    fake = FakeClient(
        planner=lambda c: text("1. factor\n2. solve"),
        solver=solver_with_tool,
        verifier=lambda c: text("checked. " + next(verdicts)),
        critic=lambda c: text("Step 2 is wrong: recheck the factor."),
        explainer=lambda c: text("Nice and clean."),
    )
    r = ma.solve_problem("Solve x^2 - 5x + 6 = 0", fake)
    assert r == {"answer": "x = 2, x = 3", "confidence": "HIGH", "tries": 2, "explanation": "Nice and clean."}
    solver_prompts = [last_text(c) for role, c in fake.calls if role == "solver" and len(c) == 1]
    assert "Step 2 is wrong" in solver_prompts[1]  # critic feedback reached the second attempt


def test_five_try_fallback():
    fake = FakeClient(
        planner=lambda c: text("1. guess"),
        solver=lambda c: text("FINAL ANSWER: 7"),
        verifier=lambda c: text("no JSON at all"),  # unparseable -> treated as incorrect
        critic=lambda c: text("wrong"),
    )
    r = ma.solve_problem("Solve x = 1", fake)
    assert (r["confidence"], r["tries"], r["answer"]) == ("LOW", 5, "7")
    roles = [role for role, _ in fake.calls]
    assert roles.count("solver") == 5 and roles.count("critic") == 4 and "explainer" not in roles


def test_rate_limit_backoff():
    failures = [errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})] * 2

    def planner(c):
        if failures:
            raise failures.pop()
        return text("1. step")

    fake = FakeClient(planner=planner)
    assert ma.run_agent(fake, "planner", ma.Memory("p")) == "1. step"
    assert len(fake.calls) == 3
    # non-retryable errors propagate immediately
    fake = FakeClient(planner=lambda c: (_ for _ in ()).throw(errors.ClientError(400, {"error": {"code": 400}})))
    try:
        ma.run_agent(fake, "planner", ma.Memory("p"))
        raise AssertionError("400 should not be retried")
    except errors.ClientError:
        assert len(fake.calls) == 1


def test_tool_round_cap():
    fake = FakeClient(solver=lambda c: call("calc", expression="1+1"))
    assert "exhausted" in ma.run_agent(fake, "solver", ma.Memory("p"))
    assert len(fake.calls) == ma.MAX_TOOL_ROUNDS


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
