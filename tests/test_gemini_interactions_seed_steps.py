from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from lingtai.kernel.llm.interface import ToolResultBlock
from lingtai.llm.gemini.adapter import InteractionsChatSession


class _RecordingInteractions:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        if kwargs.get("stream"):
            return iter(())
        return SimpleNamespace(id="interaction-next", steps=[], usage=None)


def _session() -> tuple[InteractionsChatSession, _RecordingInteractions]:
    interactions = _RecordingInteractions()
    client = SimpleNamespace(interactions=interactions)
    session = InteractionsChatSession(
        client=client,
        model="gemini-test",
        config_kwargs={"system_instruction": "system"},
    )
    session._pending_seed_turns = [
        {
            "role": "model",
            "content": [
                {
                    "type": "function_call",
                    "id": "compact-1",
                    "name": "compact",
                    "arguments": {"_reason": "continue from the compact boundary"},
                }
            ],
        }
    ]
    return session, interactions


def _tool_result() -> list[ToolResultBlock]:
    return [
        ToolResultBlock(
            id="compact-1",
            name="compact",
            content={"status": "success"},
        )
    ]


def _assert_compact_seed_steps(request: dict) -> None:
    wire_input = request["input"]
    assert [step["type"] for step in wire_input] == [
        "model_output",
        "user_input",
    ]
    assert wire_input[0]["content"][0] == {
        "type": "function_call",
        "id": "compact-1",
        "name": "compact",
        "arguments": {"_reason": "continue from the compact boundary"},
    }
    result = wire_input[1]["content"][0]
    assert result["type"] == "function_result"
    assert result["call_id"] == "compact-1"
    assert result["name"] == "compact"


@pytest.mark.parametrize("stream", [False, True])
def test_pending_compact_seed_uses_interactions_steps(stream: bool) -> None:
    session, interactions = _session()

    if stream:
        session.send_stream(_tool_result())
    else:
        session.send(_tool_result())

    request = interactions.requests[-1]
    _assert_compact_seed_steps(request)

    # Client-side history keeps the established role/content representation;
    # only the Interactions wire input uses model_output/user_input steps.
    assert session.get_client_history()[0]["role"] == "model"

    # Exercise the installed SDK's exact pre-transport normalization without
    # calling interactions.create() or crossing the network boundary.
    from google.genai._gaos.google_genai import _normalize_create_body

    normalized = _normalize_create_body(copy.deepcopy(request))
    assert normalized["input"] == request["input"]
