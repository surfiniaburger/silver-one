"""
Graph-Topology Semantic Data-Flow Schema and Evaluator Module.
Implements the contract specified in RFC_GRAPH_DATAFLOW_PRE_FILTER.md.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


SUPPORTED_SINKS = {"MEMORY_WRITE", "POINTER_DEREF", "ARRAY_INDEX", "SYSTEM_CALL"}
VALID_SANITIZERS = {"BOUNDS_CHECK", "RANGE_VALIDATION", "NULL_CHECK", "COMMAND_SANITIZATION", "ALLOWLIST_CHECK"}


@dataclass(frozen=True)
class FlowSignature:
    source_id: str
    sink_id: str
    source_type: str
    sink_type: str
    flow_type: str
    sanitizer_type: Optional[str] = None
    guarded_target: Optional[str] = None
    invalid_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "sink_id": self.sink_id,
            "source_type": self.source_type,
            "sink_type": self.sink_type,
            "flow_type": self.flow_type,
            "sanitizer_type": self.sanitizer_type,
            "guarded_target": self.guarded_target,
            "invalid_at": self.invalid_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FlowSignature":
        return cls(
            source_id=data["source_id"],
            sink_id=data["sink_id"],
            source_type=data["source_type"],
            sink_type=data["sink_type"],
            flow_type=data["flow_type"],
            sanitizer_type=data.get("sanitizer_type"),
            guarded_target=data.get("guarded_target"),
            invalid_at=data.get("invalid_at"),
        )


@dataclass
class FlowGraphSnapshot:
    snapshot_id: str
    scenario_id: str
    version: int
    nodes: Dict[str, dict] = field(default_factory=dict)
    signatures: List[FlowSignature] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())
    is_complete: bool = True
    parse_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "scenario_id": self.scenario_id,
            "version": self.version,
            "nodes": self.nodes,
            "signatures": [sig.to_dict() for sig in self.signatures],
            "created_at": self.created_at,
            "is_complete": self.is_complete,
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FlowGraphSnapshot":
        return cls(
            snapshot_id=data["snapshot_id"],
            scenario_id=data["scenario_id"],
            version=data["version"],
            nodes=data.get("nodes", {}),
            signatures=[FlowSignature.from_dict(s) for s in data.get("signatures", [])],
            created_at=data.get("created_at", time.time()),
            is_complete=data.get("is_complete", True),
            parse_error=data.get("parse_error"),
        )


def is_sanitizer_valid_for_sink(sink_type: str, sanitizer_type: Optional[str]) -> bool:
    """Verifies that sanitizer proof matches specific sink requirements."""
    if not sanitizer_type:
        return False
    if sink_type in ("MEMORY_WRITE", "ARRAY_INDEX"):
        return sanitizer_type in ("BOUNDS_CHECK", "RANGE_VALIDATION")
    if sink_type == "POINTER_DEREF":
        return sanitizer_type == "NULL_CHECK"
    if sink_type == "SYSTEM_CALL":
        return sanitizer_type in ("COMMAND_SANITIZATION", "ALLOWLIST_CHECK")
    return False


def evaluate_graph_reachability(
    graph_snapshot: FlowGraphSnapshot,
    as_of: Optional[float] = None,
    risk_threshold: float = 0.10,
) -> float:
    """
    Computes deterministic risk score based on source-to-sink graph topology.
    Fails closed (returns 1.0 / High Risk) if evidence is incomplete or parse errors occur.
    
    - Risk Score >= risk_threshold (0.10): REJECT (Flagged High Risk).
    - Risk Score < risk_threshold (0.10): PASS (Guarded/Safe Flow score = 0.05).
    """
    # Fail closed on incomplete extraction or parse error
    if not graph_snapshot.is_complete or graph_snapshot.parse_error is not None:
        return 1.0

    eval_time = as_of if as_of is not None else graph_snapshot.created_at

    active_signatures = [
        sig for sig in graph_snapshot.signatures
        if sig.invalid_at is None or sig.invalid_at > eval_time
    ]

    # Check for unhandled or unknown sink types in active signatures (fail closed)
    for sig in active_signatures:
        if sig.sink_type not in SUPPORTED_SINKS:
            return 1.0

    for sig in active_signatures:
        if sig.source_type == "UNTRUSTED_INPUT" and sig.sink_type in SUPPORTED_SINKS:
            # Check if sink-specific sanitizer proof is present on the path
            if not is_sanitizer_valid_for_sink(sig.sink_type, sig.sanitizer_type):
                return 1.0  # Confirmed unsanitized reachability path -> High Risk (Reject)

    return 0.05  # All flows guarded or safe -> Low Risk (Pass)
