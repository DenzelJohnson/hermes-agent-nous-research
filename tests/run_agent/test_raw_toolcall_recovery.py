"""Focused tests for raw plain-text tool call recovery in run_agent."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_raw_toolcall_prefix_content_is_recovered_and_executed():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False

    raw_toolcall = 'TOOLCALL>[{"name":"web_search","arguments":{"query":"results section"}}]'
    resp1 = _mock_response(content=raw_toolcall, finish_reason="stop", tool_calls=None)
    resp2 = _mock_response(content="Results loaded", finish_reason="stop")
    agent.client.chat.completions.create.side_effect = [resp1, resp2]

    with (
        patch("run_agent.handle_function_call", return_value="search result") as mock_handle_function_call,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("check the results")

    assert result["final_response"] == "Results loaded"
    assert result["api_calls"] == 2
    assert mock_handle_function_call.call_args.args[0] == "web_search"
    assert mock_handle_function_call.call_args.kwargs["tool_call_id"].startswith("call_")
