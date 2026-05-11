from pydantic import BaseModel, Field
from typing import Literal

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)


class DebaterScore(BaseModel):
    technical_accuracy: float
    logic_soundness: float
    evidence_strength: float
    total_score: float
    critique: str # GEPA: Adjudication of debater performance

class DebateEval(BaseModel):
    thinking_process: str # GEPA: Step-by-step thinking before concluding
    pro_debater: DebaterScore
    con_debater: DebaterScore
    winner: Literal["pro_debater", "con_debater"]
    reason: str # Short summary
    mechanism: str # GEPA: Technical mechanism of the bug
    counterfactual: str # GEPA: What would change the verdict?
    predicate: str # The claim being judged (must match the run predicate)
    anchors: list[str] = Field(min_length=2) # Concrete evidence hooks (must appear in code)
    verifier_report: str = "not_applicable"
    support_level: Literal["supported", "unsupported", "inconclusive"] = "supported"


def debate_judge_agent_card(agent_name: str, card_url: str) -> AgentCard:
    skill = AgentSkill(
        id='moderate_and_judge_debate',
        name='Orchestrates and judges debate',
        description='Orchestrate and judge a debate between two agents on a given topic.',
        tags=['debate'],
        examples=["""
{
  "participants": {
    "pro_debater": "https://pro-debater.example.com:443",
    "con_debater": "https://con-debater.example.org:8443"
  },
  "config": {
    "topic": "Should artificial intelligence be regulated?",
    "num_rounds": 3
  }
}
"""]
    )
    agent_card = AgentCard(
        name=agent_name,
        description='Orchestrate and judge a structured debate between pro and con agents on a given topic with multiple rounds of arguments.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card
