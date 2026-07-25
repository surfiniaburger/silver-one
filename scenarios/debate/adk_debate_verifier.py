import argparse
import json
import uvicorn
import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Part, TextPart

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
from agentbeats.structured_output import call_structured
from agentbeats.replay import ReplayManager, ReplayError
from debate_judge_common import VerifierReport, debate_judge_agent_card

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adk_debate_verifier")

# --- System Prompt ---

verifier_system_prompt = """
<role>
You are an Elite Predictive Security Auditor and Symbolic Execution Specialist. 
Your goal is to be a technical skeptic. You do not care about the "story" of the debate; you only care about the bit-level truth of the code.
</role>

<task>
Perform a "Counterfactual Audit" on a proposed security vulnerability. 
You must verify if the 'Mechanism' described is actually possible given the 'Code Snippet' and the 'Anchors' provided.
</task>

<thinking_process>
1. **Anchor Scan**: Verify that every anchor provided exists verbatim in the code.
2. **Variable Mapping**: Identify the key variables and their types.
3. **Data-Flow Trace**: Map the path from Input (Source) to the Vulnerable Operation (Sink).
4. **Invariant Check**: Identify which security invariant (e.g., "bounds check", "NULL check") is missing.
5. **Counterfactual Test**: If the anchors were patched or the input was changed, would the vulnerability still exist?
</thinking_process>

<constraints>
- You MUST fail the audit if the mechanism relies on "imaginary" code not present in the snippet.
- You MUST fail the audit if the anchors are "narrative" (comments) rather than "functional" (operations).
- You MUST be clinical and technically precise. Use terms like 'integer wrap-around', 'pointer aliasing', or 'race window'.
- If the mechanism is a hallucination (e.g. it claims an overflow on a 64-bit size_t that can't realistically overflow), you MUST call it out.
</constraints>

<output_format>
Return your response ONLY as a valid JSON object conforming to the VerifierReport schema.
DO NOT use markdown code blocks.
</output_format>
"""

class DebateVerifierADK(GreenAgent):
    def __init__(self, model: str = "ollama/deepseek-v3.1:671b-cloud"):
        self.model = model
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        if "code" not in request.config:
            return False, "Missing 'code' in config"
        if "mechanism" not in request.config:
            return False, "Missing 'mechanism' in config"
        if "anchors" not in request.config:
            return False, "Missing 'anchors' in config"
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting Predictive Verification: {req}")
        
        code = req.config["code"]
        mechanism = req.config["mechanism"]
        anchors = req.config["anchors"]
        predicate = req.config.get("predicate", "")
        run_id = str(req.config.get("run_id", "verifier-adhoc"))
        seed = int(req.config.get("seed", 42))
        mode = str(req.config.get("mode", "record"))
        cassette_path = str(req.config.get("cassette_path", f"artifacts/cassettes/{run_id}.json"))
        clock_now = req.config.get("clock_now")

        verifier_prompt = f"""
<context>
Code Snippet:
<Untrusted_Code>
{code}
</Untrusted_Code>

Proposed Predicate: {predicate}
Proposed Mechanism: {mechanism}
Proposed Anchors: {json.dumps(anchors)}
</context>

AUDIT TASK:
Does the Code Snippet actually support the Mechanism via these Anchors? 
Perform a bit-level data-flow trace to confirm or debunk the claim.
"""

        try:
            verifier_model = os.getenv("VERIFIER_MODEL", self.model)
            replay_manager = ReplayManager.from_config(
                run_id=run_id,
                seed=seed,
                cassette_path=cassette_path,
                mode=mode,
                model_config={"verifier": verifier_model},
                created_at=clock_now,
            )
            report = await call_structured(
                replay_manager=replay_manager,
                model=verifier_model,
                messages=[
                    {"role": "system", "content": verifier_system_prompt},
                    {"role": "user", "content": verifier_prompt},
                ],
                schema_name="verifier_report",
                schema_model=VerifierReport,
                strict=True,
                repair_on_fail=True,
                repair_model=verifier_model,
                stage="verifier_audit",
                options={"keep_alive": "24h"},
            )
            usage_summary = replay_manager.get_usage_summary()
            
            # Export the report as the result
            await updater.add_artifact(
                parts=[
                    TextPart(text=json.dumps(report.model_dump(), ensure_ascii=False)),
                    TextPart(text=json.dumps({"verifier_llm_usage": usage_summary}, ensure_ascii=False)),
                    TextPart(text=f"Verification Result: {'PASSED' if report.passes_audit else 'FAILED'}"),
                    TextPart(text=report.thinking_process),
                    TextPart(text=f"Logic Audit: {report.logic_error or 'None'}")
                ],
                name="VerifierReport",
            )
            
            # Return the structured data
            # Note: In the ADK framework, we can return JSON in the final response
            logger.info(f"Verification complete. Pass: {report.passes_audit}")
            
        except ReplayError:
            raise
        except Exception as e:
            logger.exception("Verifier failed during structured call.")
            raise RuntimeError(f"Verification error: {e}") from e

async def main():
    parser = argparse.ArgumentParser(description="Run the In-Varia Predictive Verifier Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9020, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--model", type=str, default=os.getenv("VERIFIER_MODEL", "ollama/deepseek-v3.1:671b-cloud"), help="Model to use for verification")
    args = parser.parse_args()

    agent = DebateVerifierADK(model=args.model)
    executor = GreenExecutor(agent)
    
    # Simple card for discovery
    scheme = os.getenv("SERVER_SCHEME", "http")
    agent_url = args.card_url or f"{scheme}://{args.host}:{args.port}/"  # NOSONAR
    agent_card = debate_judge_agent_card("DebateVerifierADK", agent_url)

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
