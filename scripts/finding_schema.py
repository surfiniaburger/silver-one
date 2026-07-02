from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Literal


class Evidence(BaseModel):
    location_type: str = Field(
        ...,
        description="Type of evidence location, e.g., 'code', 'compatibility', 'test', 'generic'."
    )
    path: Optional[str] = Field(
        None,
        description="Path identifier, e.g. file path, test name, or ref."
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible location details (e.g., line numbers, function/method names, API signatures)."
    )

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, v):
        if not v:
            return v
        path_str = str(v).strip()
        if path_str.startswith("/"):
            path_str = path_str.lstrip("/")
        return path_str


class EngineeringImpact(BaseModel):
    correctness: Literal["NONE", "LOW", "MEDIUM", "HIGH"] = Field("NONE")
    compatibility: Literal["NONE", "LOW", "MEDIUM", "HIGH"] = Field("NONE")
    security: Literal["NONE", "LOW", "MEDIUM", "HIGH"] = Field("NONE")
    maintainability: Literal["NONE", "LOW", "MEDIUM", "HIGH"] = Field("NONE")
    performance: Literal["NONE", "LOW", "MEDIUM", "HIGH"] = Field("NONE")


class EngineeringFinding(BaseModel):
    title: str = Field(..., description="A concise title summarizing the finding.")
    category: str = Field(
        ...,
        description="Engineering taxonomy category (e.g., Compatibility, Correctness, Null Safety)."
    )
    severity: Literal["INFO", "WARN", "BLOCK"] = Field(..., description="Gating severity level.")
    evidence: Evidence = Field(..., description="Concrete location context.")
    engineering_rationale: str = Field(..., description="Explanation of why this issue exists.")
    engineering_consequence: str = Field(
        ...,
        description="Consequences if this issue is ignored."
    )
    impact: EngineeringImpact = Field(
        default_factory=EngineeringImpact,
        description="Impact evaluation across code quality domains."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM confidence score from 0.0 to 1.0."
    )
    recommended_action: str = Field(..., description="Concrete steps to resolve the issue.")

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, v):
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 1.0

    @field_validator("engineering_rationale", "engineering_consequence", "recommended_action", mode="before")
    @classmethod
    def validate_non_empty_strings(cls, v):
        if not v or not str(v).strip():
            return "No details provided."
        return str(v).strip()
