import asyncio
import logging
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
)
from a2a.client.client import Consumer
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
    DataPart,
)


DEFAULT_TIMEOUT = 1200


def create_message(*, role: Role = Role.user, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id
    )

def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(str(part.root.data) if not isinstance(part.root.data, str) else part.root.data)
    return "\n".join(chunks)

def _unpack_task_event(task: Any, outputs: dict) -> None:
    """Populate outputs dict with context_id, status, merged response, and optional metadata from DataParts."""
    outputs["context_id"] = task.context_id
    outputs["status"] = task.status.state.value
    if task.status.message:
        outputs["response"] += merge_parts(task.status.message.parts)
    if not task.artifacts:
        return

    for artifact in task.artifacts:
        outputs["response"] += merge_parts(artifact.parts)
        for p in artifact.parts:
            if isinstance(p.root, DataPart) and isinstance(p.root.data, dict):
                outputs.setdefault("metadata", {}).update(p.root.data)


def _unpack_last_event(last_event: Any, outputs: dict) -> None:
    match last_event:
        case Message() as msg:
            outputs["context_id"] = msg.context_id
            outputs["response"] += merge_parts(msg.parts)
        case (task, _):
            _unpack_task_event(task, outputs)


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    consumer: Consumer | None = None,
) -> dict:
    """
    Send an A2A message to an agent endpoint and return output dictionary.

    Returns:
        dict: Containing 'response' (str), 'context_id' (str | None), optional 'status' (str),
              and optional 'metadata' (dict) populated from dictionary DataPart artifacts.
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        if consumer:
            await client.add_event_consumer(consumer)

        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None
        outputs = {
            "response": "",
            "context_id": None,
        }

        # if streaming == False, only one event is generated
        async for event in client.send_message(outbound_msg):
            last_event = event

        _unpack_last_event(last_event, outputs)
        return outputs
