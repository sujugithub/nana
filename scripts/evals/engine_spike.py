"""Identical deterministic tool workflow; run in each isolated candidate venv.

No network/model calls. This measures integration feasibility, not intelligence.
"""
import json
import sys
import time
from importlib.metadata import distributions

calls = []


def lookup(query: str) -> str:
    """Read a fixture fact by query."""
    calls.append(query)
    return "Nana runs on macOS."


start = time.monotonic()
engine = sys.argv[1]
if engine == "deep":
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class ScriptedModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ScriptedModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": "platform"}, "id": "fixture-1"}]),
        AIMessage(content="Nana runs on macOS."),
    ]))
    agent = create_deep_agent(model=model, tools=[lookup])
    result = agent.invoke({"messages": [{"role": "user", "content": "Look up Nana's platform."}]})
    answer = result["messages"][-1].content
elif engine == "pydantic":
    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import ModelResponse, ToolCallPart, TextPart
    from pydantic_ai_harness import ToolOutputLimits
    from pydantic_ai_harness.tool_output_limits import LocalFileStore

    def respond(messages, info):
        return ModelResponse([TextPart("Nana runs on macOS.")]) if calls else ModelResponse([
            ToolCallPart("lookup", {"query": "platform"}, tool_call_id="fixture-1")])

    agent = Agent(FunctionModel(respond), tools=[lookup], capabilities=[ToolOutputLimits(store=LocalFileStore())])
    answer = agent.run_sync("Look up Nana's platform.").output
elif engine == "nana":
    from pathlib import Path
    import tempfile
    from unittest.mock import patch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from workspace import store, runs, tools, contracts
    replies = iter(['{"tool":"lookup","arguments":{"query":"platform"}}',
                    '{"answer":"Nana runs on macOS."}'])
    def respond(*args):
        return {"answer": next(replies), "final_model": "fixture", "provider": "local",
                "route": "local", "tokens": {}, "estimated_cost_usd": 0, "mock": True}
    with tempfile.TemporaryDirectory() as root, patch.object(store, "ROOT", Path(root)), \
            patch.dict(tools.TOOLS, {"lookup": "Fixture lookup"}), \
            patch.dict(contracts.SCHEMAS, {"lookup": contracts.obj({"query": contracts.S}, ["query"])}), \
            patch.object(tools, "call", side_effect=lambda name, args, **kw: lookup(**args)), \
            patch.object(runs, "generate", side_effect=respond):
        run_id = store.run_create("agent", {"prompt": "Look up Nana's platform.", "tools": ["lookup"]})
        runs.worker("local", run_id, False)
        answer = store.run_get(run_id)["result"].get("answer")
else:
    raise SystemExit("Use deep, pydantic or nana")
assert calls == ["platform"] and answer == "Nana runs on macOS."
print(json.dumps({"engine": engine, "fixture": "lookup-platform", "mock": True,
                  "success": True, "tool_calls": len(calls), "elapsed_s": round(time.monotonic()-start, 3),
                  "installed_distributions": len(list(distributions()))}))
