import argparse
import json
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
from agentbeats.structured_output import call_structured
from debate_judge_common import DebateEval, debate_judge_agent_card
from data_generator import BarredDataGenerator
from agentbeats.replay import ReplayManager
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adk_debate_judge")

# System prompt for the judge agent
judge_system_prompt = """
<role>
You are a Senior Security Architect and Impartial Adjudicator. Your tone is clinical, technical, and immune to rhetorical flourishes.
</role>

<task>
Determine the absolute technical ground-truth verdict for a security predicate based on a code snippet and an adversarial debate. 
Success looks like a verdict that would hold up in a high-stakes security audit or a court of law.
</task>

<constraints>
- NEVER award a win based on rhetorical style.
- If a vulnerability requires a "misconfiguration" that is common (e.g. default credentials, outdated drivers), it is a VULNERABILITY.
- You MUST provide a technically deep 'mechanism' explanation.
- You MUST provide a clear 'counterfactual' (what would change the verdict).
- You MUST think through the technical arguments step-by-step in the `thinking_process` field before providing your final scores.
  1. Identify the core technical mechanism described in the code.
  2. Evaluate the Pro side's attack vector: Is it technically feasible?
  3. Evaluate the Con side's defense: Is it an invariant guarantee?
  4. Reach a final verdict based on technical risk.
</constraints>

<output_format>
Return your response ONLY as a valid JSON object conforming to the schema. Do not wrap the JSON in tags like <answer> or markdown formatting.
</output_format>
"""

class DebateJudgeADK(GreenAgent):
    def __init__(self, model: str = "ollama/gpt-oss:20b-cloud"):
        self.model = model
        self._required_roles = ["pro_debater", "con_debater"]
        self._required_config_keys = ["topic", "num_rounds"]
        self._tool_provider = ToolProvider()
        self.generator = BarredDataGenerator()
        self.replay_manager: Optional[ReplayManager] = None

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
        logger.info(f"Starting BARRED debate orchestration: {req}")
        
        # Initialize Replay Manager
        run_id = req.config.get("run_id", "default-run")
        seed = int(req.config.get("seed", 42))
        mode = req.config.get("mode", "record")
        cassette_path = req.config.get("cassette_path", f"artifacts/cassettes/{run_id}.json")
        self.replay_manager = ReplayManager.from_config(run_id, seed, cassette_path, mode)
        self.generator.replay_manager = self.replay_manager

        predicate = req.config.get("predicate", "The input block matches the target verdict.")
        target_verdict = req.config.get("target_verdict", "True")
        target_dimension = req.config.get("target_dimension", "General")
        max_refinements = int(req.config.get("max_refinements", 2))
        output_file = req.config.get("output_file", "training_corpus.jsonl")
        
        current_input_block = req.config.get("topic", "")
        
        try:
            last_judge_reason = ""
            for i in range(max_refinements + 1):
                await updater.update_status(TaskState.working, new_agent_text_message(f"Refinement Round {i+1}/{max_refinements + 1}"))
                
                # Step 1: Generate/Refine the sample
                if i == 0:
                    await updater.update_status(TaskState.working, new_agent_text_message("Generating initial boundary sample..."))
                    sample_data = await self.generator.generate_boundary_sample(current_input_block, predicate, target_dimension, target_verdict)
                else:
                    await updater.update_status(TaskState.working, new_agent_text_message(f"Refining sample based on judge feedback..."))
                    sample_data = await self.generator.refine_sample(current_input_block, predicate, target_dimension, target_verdict, sample_data.get("revised_input_block", ""), last_judge_reason)

                current_sample_block = sample_data.get("revised_input_block", current_input_block)
                
                # Step 2: Orchestrate Debate
                opposite_verdict = "False" if target_verdict == "True" else "True"
                pro_mission = f"PRO MISSION: Prove that the predicate '{predicate}' is {target_verdict} for the given code."
                con_mission = f"CON MISSION: Prove that the predicate '{predicate}' is {opposite_verdict} (i.e., disprove the target verdict)."
                
                debate = await self.orchestrate_debate(
                    req.participants, 
                    current_sample_block, 
                    int(req.config["num_rounds"]), 
                    updater,
                    pro_mission=pro_mission,
                    con_mission=con_mission,
                    predicate=predicate,
                    target_verdict=target_verdict
                )

                transcript = ""
                for j, (pro, con) in enumerate(zip(debate["pro_debater"], debate["con_debater"]), start=1):
                    transcript += f"pro_debater (Round {j}): {pro}\n"
                    transcript += f"con_debater (Round {j}): {con}\n"

                await updater.update_status(TaskState.working, new_agent_text_message("Debate finished. Judging..."))

                # Step 3: Judge
                judge_prompt = f"""
<context>
Predicate: {predicate}
Target Verdict: {target_verdict}
The side 'pro_debater' is the ADVOCATE and MUST prove the Target Verdict is {target_verdict}.
The side 'con_debater' is the SKEPTIC and MUST prove the verdict is {opposite_verdict}.

Code Snippet:
{current_sample_block}

Debate Transcript:
{transcript}
</context>
"""
                try:
                    judge_model = os.getenv("JUDGE_MODEL", self.model)
                    debate_eval = await call_structured(
                        replay_manager=self.replay_manager,
                        model=judge_model,
                        messages=[
                            {"role": "system", "content": judge_system_prompt},
                            {"role": "user", "content": judge_prompt},
                        ],
                        schema_name="debate_eval",
                        schema_model=DebateEval,
                        strict=True,
                        repair_on_fail=True,
                        repair_model=judge_model,
                    )
                    last_judge_reason = debate_eval.reason
                except Exception as e:
                    logger.error(f"Judge structured output failed: {e}")
                    last_judge_reason = f"Failed to parse judge response: {e}"
                    continue
                
                is_valid = debate_eval.winner == "pro_debater"
                
                if is_valid:
                    await updater.update_status(TaskState.working, new_agent_text_message("Consensus reached! Exporting sample."))
                    
                    # Persist RunRecord (A1)
                    record_path = req.config.get("record_path", f"artifacts/runs/{run_id}.json")
                    self.replay_manager.save_record(record_path)

                    export_data = {
                        "instruction": f"Analyze this input for the condition: {predicate}",
                        "input": current_sample_block,
                        "output": {
                            "verdict": "1" if target_verdict == "True" else "0",
                            "reasoning": debate_eval.reason,
                            "mechanism": debate_eval.mechanism,
                            "counterfactual": debate_eval.counterfactual,
                            "adjudication": {
                                "pro": debate_eval.pro_debater.critique,
                                "con": debate_eval.con_debater.critique
                            }
                        }
                    }
                    with open(output_file, "a") as f:
                        f.write(json.dumps(export_data) + "\n")
                    
                    await updater.add_artifact(
                        parts=[TextPart(text=f"Sample Accepted and saved to {output_file}"), TextPart(text=debate_eval.reason)],
                        name="Result",
                    )
                    return
                else:
                    logger.info(f"Refinement required. Judge reason: {last_judge_reason}")

            # Persist RunRecord (A1) - Failure path
            record_path = req.config.get("record_path", f"artifacts/runs/{run_id}.json")
            self.replay_manager.save_record(record_path)

            await updater.update_status(TaskState.working, new_agent_text_message("Failed to reach consensus after max refinements."))
            await updater.add_artifact(
                parts=[TextPart(text="Failed to reach consensus.")],
                name="Result",
            )

        finally:
            self._tool_provider.reset()


    async def orchestrate_debate(
        self,
        participants: dict[str, str],
        code: str,
        num_rounds: int,
        updater: TaskUpdater,
        pro_mission: str,
        con_mission: str,
        predicate: str,
        target_verdict: str
    ) -> dict[str, list[str]]:
        debate: dict[str, list[str]] = {"pro_debater": [], "con_debater": []}

        async def turn(role: str, prompt: str, new_conv: bool = False) -> str:
            # A2: Record/Replay Agent Turns via Cassette
            # Use a pseudo-model and params to unique-ify the turn
            pseudo_model = f"a2a/{role}"
            pseudo_params = {"new_conversation": new_conv, "agent_url": str(participants[role])}
            
            cached = self.replay_manager.cassette.get_response(pseudo_model, [{"role": "user", "content": prompt}], pseudo_params)
            if cached:
                logger.info(f"Replaying {role} turn from cassette.")
                response = cached
            else:
                if self.replay_manager.cassette.mode == "replay":
                    raise RuntimeError(f"Offline Replay Error: No cached response for agent {role}")
                
                response = await self._tool_provider.talk_to_agent(prompt, str(participants[role]), new_conversation=new_conv)
                self.replay_manager.cassette.save_response(pseudo_model, [{"role": "user", "content": prompt}], pseudo_params, response)
            
            logger.info(f"{role}: {response}")
            debate[role].append(response)
            await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {response}"))
            return response

        # Opening turns with high-intensity mission injection
        context = f"PREDICATE: {predicate}\nTARGET VERDICT: {target_verdict}\n\nCODE TO ANALYZE:\n{code}"
        pro_opening = f"### MISSION CRITICAL\n{pro_mission}\n\n{context}\n\nPresent your opening technical argument immediately. No preamble."
        pro_resp = await turn("pro_debater", pro_opening, new_conv=True)
        
        con_opening = f"### MISSION CRITICAL\n{con_mission}\n\n{context}\n\nOpponent's Opening: {pro_resp}\n\nPresent your counter-argument immediately. Be technical."
        con_resp = await turn("con_debater", con_opening, new_conv=True)

        # Remaining rounds
        for r in range(num_rounds - 1):
            pro_resp = await turn("pro_debater", f"Your opponent said: {con_resp}. Present your next argument.")
            con_resp = await turn("con_debater", f"Your opponent said: {pro_resp}. Present your next argument.")

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
