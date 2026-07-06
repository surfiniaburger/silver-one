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

    @field_validator("location_type", mode="before")
    @classmethod
    def validate_location_type(cls, v):
        if v is None:
            raise ValueError("location_type cannot be None.")
        if not isinstance(v, str):
            raise ValueError("location_type must be a string.")
        return v.strip()

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("path must be a string.")
        path_str = v.strip()
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
        if v is None:
            raise ValueError("Confidence cannot be None.")
        if isinstance(v, bool):
            raise ValueError("Confidence cannot be a boolean value.")
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError("Confidence must be a numeric value.")
        
        if 0.0 <= val <= 1.0:
            return val
        elif 1.0 < val <= 100.0:
            return val / 100.0
        else:
            raise ValueError("Confidence score is out of bounds.")

    @field_validator("title", "category", "engineering_rationale", "engineering_consequence", "recommended_action", mode="before")
    @classmethod
    def validate_string_field(cls, v):
        if v is None:
            raise ValueError("Field cannot be None.")
        if not isinstance(v, str):
            raise ValueError("Field must be a string.")
        return v.strip()


class FieldValidation(BaseModel):
    field_name: str
    status: Literal["VALID", "NORMALIZED", "REPAIRED", "INVALID"]
    raw_value: Any
    repaired_value: Any
    repair_type: Optional[str] = None
    repair_reason: Optional[str] = None


class ValidationContext(BaseModel):
    repaired: bool = False
    normalized: bool = False
    fields: List[FieldValidation] = Field(default_factory=list)


class UnitReviewArtifact(BaseModel):
    file_path: str
    name: str
    review: Any  # CodeReviewBreakdown (any to avoid circular dependency)
    validation: Optional[ValidationContext] = None
    raw_response: str = Field(..., description="The original raw unvalidated LLM response string")


class CQIResult(BaseModel):
    valid: bool
    value: Optional[float] = None
    reason: Optional[str] = None
    error_code: Optional[Literal[
        "MISSING_REVIEW",
        "MISSING_DIMENSION",
        "MISSING_SCORE",
        "INVALID_SCORE",
    ]] = None


class CompatibilityCheckResult(BaseModel):
    state: Literal["PASS", "FAIL", "NOT_EXECUTED", "CHECK_FAILED"]
    compatible: bool
    score: float
    regressions: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
    details: List[str] = Field(default_factory=list)


class BaselineCheckResult(BaseModel):
    state: Literal["AVAILABLE", "FIRST_RUN", "BASELINE_MISSING", "BASELINE_CORRUPTED"]
    available: bool
    required: bool = False
    reason: Optional[str] = None
    details: List[str] = Field(default_factory=list)


def _record_dimension_score_validation(
    dim: str,
    raw_dim: Dict[str, Any],
    parsed_dim: Any,
    fields: List[FieldValidation],
) -> None:
    raw_score = raw_dim.get("score")
    parsed_score = getattr(parsed_dim, "score", None)
    if raw_score is None or parsed_score is None:
        return

    try:
        raw_score_float = float(raw_score)
    except (ValueError, TypeError):
        fields.append(FieldValidation(
            field_name=f"{dim}.score",
            status="INVALID",
            raw_value=raw_score,
            repaired_value=parsed_score,
            repair_reason="non-numeric value"
        ))
        return

    if 10.0 < raw_score_float <= 100.0:
        fields.append(FieldValidation(
            field_name=f"{dim}.score",
            status="REPAIRED",
            raw_value=raw_score,
            repaired_value=parsed_score,
            repair_type="scaled_percentage",
            repair_reason="score scale conversion"
        ))


def _record_dimension_rationale_validation(
    dim: str,
    raw_dim: Dict[str, Any],
    parsed_dim: Any,
    fields: List[FieldValidation],
) -> None:
    raw_rat = raw_dim.get("rationale")
    parsed_rat = getattr(parsed_dim, "rationale", "")
    if raw_rat is None:
        fields.append(FieldValidation(
            field_name=f"{dim}.rationale",
            status="NORMALIZED",
            raw_value=None,
            repaired_value=""
        ))
    elif str(raw_rat).strip() != str(raw_rat):
        fields.append(FieldValidation(
            field_name=f"{dim}.rationale",
            status="NORMALIZED",
            raw_value=raw_rat,
            repaired_value=parsed_rat
        ))


def _validate_dimensions(raw_json: Dict[str, Any], review: Any, fields: List[FieldValidation]) -> None:
    dimensions = ["readability", "maintainability", "correctness", "complexity", "security", "test_coverage"]
    for dim in dimensions:
        raw_dim = raw_json.get(dim)
        parsed_dim = getattr(review, dim, None)
        if not isinstance(raw_dim, dict) or parsed_dim is None:
            continue

        _record_dimension_score_validation(dim, raw_dim, parsed_dim, fields)
        _record_dimension_rationale_validation(dim, raw_dim, parsed_dim, fields)


def _validate_finding_strings(idx: int, raw_find: Dict[str, Any], parsed_find: Any, fields: List[FieldValidation]) -> None:
    # Rationale
    raw_rat = raw_find.get("engineering_rationale")
    parsed_rat = getattr(parsed_find, "engineering_rationale", "")
    if raw_rat is None or str(raw_rat).strip() != str(raw_rat):
        fields.append(FieldValidation(
            field_name=f"findings[{idx}].engineering_rationale",
            status="NORMALIZED",
            raw_value=raw_rat,
            repaired_value=parsed_rat
        ))
        
    # Consequence
    raw_conseq = raw_find.get("engineering_consequence")
    parsed_conseq = getattr(parsed_find, "engineering_consequence", "")
    if raw_conseq is None or str(raw_conseq).strip() != str(raw_conseq):
        fields.append(FieldValidation(
            field_name=f"findings[{idx}].engineering_consequence",
            status="NORMALIZED",
            raw_value=raw_conseq,
            repaired_value=parsed_conseq
        ))
        
    # Recommended action
    raw_recom = raw_find.get("recommended_action")
    parsed_recom = getattr(parsed_find, "recommended_action", "")
    if raw_recom is None or str(raw_recom).strip() != str(raw_recom):
        fields.append(FieldValidation(
            field_name=f"findings[{idx}].recommended_action",
            status="NORMALIZED",
            raw_value=raw_recom,
            repaired_value=parsed_recom
        ))


def _validate_finding_confidence(idx: int, raw_find: Dict[str, Any], parsed_find: Any, fields: List[FieldValidation]) -> None:
    raw_conf = raw_find.get("confidence")
    parsed_conf = getattr(parsed_find, "confidence", None)
    if raw_conf is not None and parsed_conf is not None:
        try:
            raw_conf_float = float(raw_conf)
            if 1.0 < raw_conf_float <= 100.0:
                fields.append(FieldValidation(
                    field_name=f"findings[{idx}].confidence",
                    status="REPAIRED",
                    raw_value=raw_conf,
                    repaired_value=parsed_conf,
                    repair_type="scaled_percentage",
                    repair_reason="confidence scale conversion"
                ))
        except (ValueError, TypeError):
            fields.append(FieldValidation(
                field_name=f"findings[{idx}].confidence",
                status="INVALID",
                raw_value=raw_conf,
                repaired_value=parsed_conf,
                repair_reason="non-numeric value"
            ))


def _validate_finding_path(idx: int, raw_find: Dict[str, Any], parsed_find: Any, fields: List[FieldValidation]) -> None:
    raw_ev = raw_find.get("evidence", {})
    if isinstance(raw_ev, dict):
        raw_path = raw_ev.get("path")
        parsed_path = getattr(getattr(parsed_find, "evidence", None), "path", None)
        if raw_path is not None:
            clean_path = str(raw_path).strip().lstrip("/")
            if clean_path != str(raw_path):
                fields.append(FieldValidation(
                    field_name=f"findings[{idx}].evidence.path",
                    status="NORMALIZED",
                    raw_value=raw_path,
                    repaired_value=parsed_path
                ))


def _validate_single_finding(idx: int, raw_find: Dict[str, Any], parsed_find: Any, fields: List[FieldValidation]) -> None:
    _validate_finding_strings(idx, raw_find, parsed_find, fields)
    _validate_finding_confidence(idx, raw_find, parsed_find, fields)
    _validate_finding_path(idx, raw_find, parsed_find, fields)


def _record_dropped_finding(
    idx: int,
    raw_find: Any,
    fields: List[FieldValidation],
    reason: str,
) -> None:
    fields.append(FieldValidation(
        field_name=f"findings[{idx}]",
        status="REPAIRED",
        raw_value=raw_find,
        repaired_value=None,
        repair_type="dropped_invalid_finding",
        repair_reason=reason
    ))


def _is_valid_engineering_finding(raw_find: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    try:
        EngineeringFinding.model_validate(raw_find)
    except Exception as exc:
        return False, str(exc)
    return True, None


def _validate_findings(raw_json: Dict[str, Any], review: Any, fields: List[FieldValidation]) -> None:
    raw_findings = raw_json.get("findings", [])
    parsed_findings = getattr(review, "findings", [])
    if not isinstance(raw_findings, list) or not isinstance(parsed_findings, list):
        return

    parsed_idx = 0
    for idx, raw_find in enumerate(raw_findings):
        if isinstance(raw_find, EngineeringFinding):
            parsed_idx += 1
            continue

        if not isinstance(raw_find, dict):
            _record_dropped_finding(idx, raw_find, fields, "finding is not an object")
            continue

        valid_finding, invalid_reason = _is_valid_engineering_finding(raw_find)
        if not valid_finding:
            _record_dropped_finding(idx, raw_find, fields, invalid_reason or "invalid finding")
            continue

        if parsed_idx < len(parsed_findings):
            _validate_single_finding(idx, raw_find, parsed_findings[parsed_idx], fields)
        parsed_idx += 1


def _validate_top_level_review_fields(raw_json: Dict[str, Any], review: Any, fields: List[FieldValidation]) -> None:
    parsed_summary = getattr(review, "summary", "")
    if "summary" not in raw_json:
        fields.append(FieldValidation(
            field_name="summary",
            status="REPAIRED",
            raw_value=None,
            repaired_value=parsed_summary,
            repair_type="missing_default",
            repair_reason="missing summary repaired to empty string"
        ))
    else:
        raw_summary = raw_json.get("summary")
        if raw_summary is not None and str(raw_summary).strip() != str(raw_summary):
            fields.append(FieldValidation(
                field_name="summary",
                status="NORMALIZED",
                raw_value=raw_summary,
                repaired_value=parsed_summary
            ))

    parsed_severity = getattr(review, "severity", None)
    if "severity" not in raw_json:
        fields.append(FieldValidation(
            field_name="severity",
            status="REPAIRED",
            raw_value=None,
            repaired_value=parsed_severity,
            repair_type="missing_default",
            repair_reason="missing severity repaired to OK"
        ))


def build_validation_context(raw_json: Dict[str, Any], review: Any) -> ValidationContext:
    fields = []
    _validate_top_level_review_fields(raw_json, review, fields)
    _validate_dimensions(raw_json, review, fields)
    _validate_findings(raw_json, review, fields)
    repaired = any(f.status == "REPAIRED" for f in fields)
    normalized = any(f.status == "NORMALIZED" for f in fields)
    return ValidationContext(repaired=repaired, normalized=normalized, fields=fields)
