import argparse
import contextlib
import uvicorn
import asyncio
import logging
import os
import re
import litellm
from dotenv import load_dotenv

load_dotenv()



from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
from debate_judge_common import DebateEval, debate_judge_agent_card

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adk_debate_judge")

# System prompt for the judge agent
judge_system_prompt = """
You are an experienced debate judge. You will be given the full transcript of a debate and a topic.
Your task is to evaluate the debate based on four key criteria: Emotional Appeal, Clarity of Argument and Reasoning, Logical Arrangement of Arguments, and Relevance to Debate Topic.

For each of the four subdimensions, provide a score from 0 to 1 (with 0 being the lowest and 1 being the highest) for both the **Pro (Affirmative)** side and the **Con (Negative)** side. Additionally, provide a brief analysis for both sides for each subdimension.

Please output the result in the specified JSON format. The JSON object must have the following structure, with keys "pro_debater", "con_debater", "winner", and "reason":

{
  "pro_debater": {
    "emotional_appeal": 0.0,
    "argument_clarity": 0.0,
    "argument_arrangement": 0.0,
    "relevance_to_topic": 0.0,
    "total_score": 0.0
  },
  "con_debater": {
    "emotional_appeal": 0.0,
    "argument_clarity": 0.0,
    "argument_arrangement": 0.0,
    "relevance_to_topic": 0.0,
    "total_score": 0.0
  },
  "winner": "pro_debater",
  "reason": "A brief explanation of why the winner was chosen."
}
"""

class DebateJudgeADK(GreenAgent):
    def __init__(self, model: str = "ollama/gpt-oss:20b-cloud"):
        self.model = model
        self._required_roles = ["pro_debater", "con_debater"]
        self._required_config_keys = ["topic", "num_rounds"]
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        try:
            int(request.config["num_rounds"])
        except Exception as e:
            return False, f"Can't parse num_rounds: {e}"
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting debate orchestration: {req}")

        try:
            debate = await self.orchestrate_debate(req.participants, req.config["topic"], int(req.config["num_rounds"]), updater)

            debate_text = ""
            for i, (pro, con) in enumerate(zip(debate["pro_debater"], debate["con_debater"]), start=1):
                debate_text += f"Pro Argument {i}: {pro}\n"
                debate_text += f"Con Argument {i}: {con}\n"

            await updater.update_status(TaskState.working, new_agent_text_message("Debate orchestration finished. Starting evaluation."))
            logger.info("Debate orchestration finished. Evaluating debate.")

            user_prompt = f"""
            Evaluate the debate on the topic: '{req.config["topic"]}'
            Debate transcript is as follows:
            {debate_text}
            Provide a JSON formatted response with scores and comments for each criterion for both debaters.
            """
            
            import litellm
            
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "system", "content": judge_system_prompt},
                          {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content.strip()
            
            # Remove <think> blocks entirely (common in DeepSeek/Qwen models)
            cleaned_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            
            # Try to extract from ```json ... ``` markdown block
            json_str = ""
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                # Fallback to greedy '{ ... }'
                json_start = cleaned_text.find('{')
                json_end = cleaned_text.rfind('}')
                if json_start != -1 and json_end != -1:
                    json_str = cleaned_text[json_start:json_end+1]
                else:
                    raise ValueError(f"No JSON object found in the LLM response. Response: \n{response_text}")
            
            try:
                debate_eval = DebateEval.model_validate_json(json_str)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON from LLM response. Error: {e}. Response: \n{response_text}") from e

            logger.info(f"Debate Evaluation:\n{debate_eval.model_dump_json()}")

            result = EvalResult(winner=debate_eval.winner, detail=debate_eval.model_dump())
            await updater.add_artifact(
                parts=[
                    TextPart(text=debate_eval.reason),
                    TextPart(text=result.model_dump_json()),
                ],
                name="Result",
            )

        finally:
            self._tool_provider.reset()

    async def orchestrate_debate(
        self,
        participants: dict[str, str],
        topic: str,
        num_rounds: int,
        updater: TaskUpdater,
    ) -> dict[str, list[str]]:
        debate: dict[str, list[str]] = {"pro_debater": [], "con_debater": []}

        async def turn(role: str, prompt: str) -> str:
            response = await self._tool_provider.talk_to_agent(prompt, str(participants[role]), new_conversation=False)
            logger.info(f"{role}: {response}")
            debate[role].append(response)
            await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {response}"))
            return response

        # Opening turns
        response = await turn("pro_debater", f"Debate Topic: {topic}. Present your opening argument.")
        response = await turn("con_debater", f"Debate Topic: {topic}. Present your opening argument. Your opponent opened with: {response}")

        # Remaining rounds
        for _ in range(num_rounds - 1):
            response = await turn("pro_debater", f"Your opponent said: {response}. Present your next argument.")
            response = await turn("con_debater", f"Your opponent said: {response}. Present your next argument.")

        return debate

async def main():
    parser = argparse.ArgumentParser(description="Run the A2A debate judge (ADK version).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--model", type=str, default=os.getenv("JUDGE_MODEL", "ollama/gpt-oss:20b-cloud"), help="LiteLLM model string to use for the judge")
    args = parser.parse_args()

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"
    
    agent = DebateJudgeADK(model=args.model)
    executor = GreenExecutor(agent)
    agent_card = debate_judge_agent_card("DebateJudgeADK", agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
    uvicorn_server = uvicorn.Server(uvicorn_config)
    await uvicorn_server.serve()

if __name__ == '__main__':
    asyncio.run(main())