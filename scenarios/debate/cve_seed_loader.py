import os
import csv
import json
import hashlib
import re
import asyncio
import logging
import argparse
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Set, List, Dict, Optional, Tuple, Any, Iterator
from pathlib import Path
import sys
from pydantic import BaseModel

# Ensure project root and debate directory are in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
debate_dir = str(Path(__file__).resolve().parent)
if debate_dir not in sys.path:
    sys.path.insert(0, debate_dir)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from agentbeats.replay import ReplayManager
from agentbeats.structured_output import call_structured
from adk_debate_judge import _predicate_quality  # type: ignore
from offline_b_gate import _is_generic_anchor  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cve_seed_loader")

STOPWORDS: Set[str] = {
    'code', 'vulnerable', 'function', 'buffer', 'overflow', 'driver',
    'kernel', 'memory', 'leak', 'attack', 'access', 'read', 'write',
    'integer', 'heap', 'stack', 'arbitrary', 'denial', 'service',
    'condition', 'flaw', 'error', 'null', 'pointer', 'race'
}

FALLBACK_PREDICATE: str = "Vulnerability suspected."
DEFAULT_EXPANSION_BATCH_SIZE: int = 200


class GepaExplanation(BaseModel):
    """Structured response schema for GEPA vulnerability analysis."""
    predicate: str = FALLBACK_PREDICATE
    evidence_hooks: List[str] = []
    uncertainty: str = "Medium"
    proof_requirements: str = ""


def _matches_language(language: str, target_lang: str) -> bool:
    """Check if snippet language matches the target language filter."""
    if not target_lang:
        return True
    langs = re.split(r'[,/ \t]+', (language or "").strip().lower())
    return target_lang in langs


def _extract_spans_from_text(text: str) -> List[str]:
    """Extract code spans from markdown backticks or colon-separated segments."""
    code_spans = re.findall(r'`([^`]+)`', text)
    if not code_spans:
        parts = text.split(":")
        code_spans = [p.strip() for p in parts if p.strip()] if len(parts) > 1 else [text.strip()]
    return code_spans


def _iter_source_rows(csv_path: str) -> Iterator[Dict[str, str]]:
    """Lazily stream rows from a CSV file without materializing the whole file in memory."""
    import sys
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    if not os.path.exists(csv_path):
        return
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _get_matching_source_candidates(csv_path: str, target_lang: str) -> Iterator[Tuple[str, str, Any]]:
    """Stream candidate tuples (code, language, safety) that match the target language."""
    for row in _iter_source_rows(csv_path):
        lang = (row.get("language") or "").strip().lower()
        if not _matches_language(lang, target_lang):
            continue
        code = row.get("code", "")
        if code:
            yield (code, lang, row.get("safety"))


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Synchronously append a JSON record line to file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


@dataclass
class SeedExpansionConfig:
    """Configuration options for incremental seed expansion."""
    target_total: int = 500
    existing_seeds_path: Optional[str] = None
    output_path: str = "scenarios/debate/cve_seeds_500.jsonl"
    min_marginal_yield: float = 0.20
    window_size: int = 50
    max_calls: int = 1500
    max_tokens: int = 6000000
    target_lang: str = "c"
    telemetry_path: Optional[str] = None
    attempts_path: Optional[str] = None
    purge_invalid_existing: bool = True

    @classmethod
    def from_args(
        cls,
        config: Optional[Any] = None,
        **kwargs: Any
    ) -> "SeedExpansionConfig":
        """Resolve config instance from either a config object, int, or keyword arguments."""
        if config is None:
            return cls(**kwargs)
        if isinstance(config, SeedExpansionConfig):
            if kwargs:
                data = {**config.__dict__, **kwargs}
                return cls(**data)
            return config
        if isinstance(config, int):
            return cls(target_total=config, **kwargs)
        raise TypeError(f"Unsupported config type: {type(config).__name__}")


class SeedExpansionState:
    """Encapsulates tracking state, stop-rule evaluation, and telemetry for seed expansion."""

    def __init__(
        self,
        existing_seeds: List[Dict],
        accepted_seeds: List[Dict],
        window_size: int = 50,
        attempts_path: Optional[str] = None,
    ):
        self.existing_seeds = existing_seeds
        self.accepted_seeds = accepted_seeds
        self.total_calls: int = 0
        self.total_tokens_est: int = 0
        self.rejection_counts: Dict[str, int] = defaultdict(int)
        self.sliding_window: deque = deque(maxlen=window_size)
        self.attempts_log: List[Dict] = []
        self.attempts_path = attempts_path
        self.stop_reason: str = "source_exhausted"

    def record_attempt(
        self,
        predicate: str,
        code_len: int,
        is_valid: bool,
        reasons: List[str],
        anchors: List[str],
    ) -> None:
        """Log metadata for a single explanation and validation attempt, persisting to JSONL if enabled."""
        attempt_entry = {
            "attempt_idx": self.total_calls,
            "valid": is_valid,
            "reasons": reasons,
            "predicate": predicate,
            "anchors_count": len(anchors),
            "code_len": code_len,
        }
        self.attempts_log.append(attempt_entry)
        if self.attempts_path:
            _append_jsonl(self.attempts_path, attempt_entry)

    async def record_success(
        self,
        seed_record: Dict[str, Any],
        output_path: str,
        target_total: int,
    ) -> None:
        """Register an accepted seed, append asynchronously to file, and update window stats."""
        self.accepted_seeds.append(seed_record)
        self.sliding_window.append(1)
        await asyncio.to_thread(_append_jsonl, output_path, seed_record)
        logger.info(
            f"[Accepted {len(self.accepted_seeds)}/{target_total}] "
            f"Calls: {self.total_calls} | Tokens: ~{self.total_tokens_est}"
        )

    def record_rejection(self, reasons: List[str]) -> None:
        """Tally rejection reasons and update sliding window with failure."""
        self.sliding_window.append(0)
        for r in reasons:
            self.rejection_counts[r] += 1
        logger.info(f"[Rejected attempt {self.total_calls}] Reasons: {reasons}")

    def evaluate_stop_rules(self, config: SeedExpansionConfig) -> Optional[str]:
        """Check all stopping criteria and return stop reason string if triggered."""
        if len(self.accepted_seeds) >= config.target_total:
            logger.info(f"Target total of {config.target_total} clean seeds reached!")
            return "target_total_reached"

        if self.total_calls >= config.max_calls:
            logger.warning(f"Stop Rule Triggered: max_calls ({config.max_calls}) reached.")
            return "max_calls_budget_reached"

        if self.total_tokens_est >= config.max_tokens:
            logger.warning(f"Stop Rule Triggered: max_tokens ({config.max_tokens}) reached.")
            return "max_tokens_budget_reached"

        if len(self.sliding_window) >= config.window_size:
            current_yield = sum(self.sliding_window) / len(self.sliding_window)
            if current_yield < config.min_marginal_yield:
                logger.warning(
                    f"Stop Rule Triggered: marginal yield ({current_yield:.2%}) "
                    f"fell below threshold ({config.min_marginal_yield:.2%}) "
                    f"over last {config.window_size} attempts."
                )
                return "marginal_yield_decay_threshold_reached"

        return None

    def build_telemetry(self, config: SeedExpansionConfig) -> Dict[str, Any]:
        """Construct a telemetry summary dictionary."""
        final_yield = (sum(self.sliding_window) / len(self.sliding_window)) if self.sliding_window else 0.0
        return {
            "stop_reason": self.stop_reason,
            "target_total": config.target_total,
            "initial_verified_count": len(self.existing_seeds),
            "final_accepted_count": len(self.accepted_seeds),
            "new_accepted_count": len(self.accepted_seeds) - len(self.existing_seeds),
            "total_calls": self.total_calls,
            "total_tokens_est": self.total_tokens_est,
            "rejection_breakdown": dict(self.rejection_counts),
            "final_window_yield": final_yield,
            "output_path": config.output_path,
        }

    def save_telemetry(self, config: SeedExpansionConfig) -> None:
        """Persist telemetry summary to JSON if a path is configured."""
        if not config.telemetry_path:
            return
        telemetry = self.build_telemetry(config)
        os.makedirs(os.path.dirname(os.path.abspath(config.telemetry_path)), exist_ok=True)
        with open(config.telemetry_path, 'w', encoding='utf-8') as tf:
            json.dump(telemetry, tf, indent=2)
        logger.info(f"Telemetry saved to {config.telemetry_path}")


class CVESeedLoader:
    def __init__(
        self,
        eval_csv_path: str,
        source_csv_path: str,
        replay_manager: ReplayManager,
        explainer_model: str = None
    ):
        self.eval_csv_path = self._resolve_path(eval_csv_path)
        self.source_csv_path = self._resolve_path(source_csv_path)
        self.replay_manager = replay_manager
        self.explainer_model = explainer_model or os.getenv("GEPA_MODEL", "ollama/gpt-oss:120b-cloud")
        self.explain_timeout_s = float(os.getenv("GEPA_EXPLAIN_TIMEOUT_S", "120"))
        self.explain_retries = int(os.getenv("GEPA_EXPLAIN_RETRIES", "2"))
        self.max_concurrency = int(os.getenv("GEPA_EXPLAIN_CONCURRENCY", "5"))

        # Separate stores: eval exclusion (exact & norm) vs candidate seeds (exact, norm, & fuzzy shingles)
        self.eval_exact_hashes: Set[str] = set()
        self.eval_norm_hashes: Set[str] = set()
        self.used_exact_hashes: Set[str] = set()
        self.used_norm_hashes: Set[str] = set()
        self.used_shingles: List[Set[str]] = []

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Resolve path directly or relative to project_root if not found."""
        if not path:
            return path
        if os.path.exists(path):
            return path
        project_candidate = os.path.join(project_root, path)
        if os.path.exists(project_candidate):
            return project_candidate
        return path

    def _normalize_code(self, code: str) -> str:
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = "".join(code.split()).lower()
        return code

    def _get_shingles(self, code: str, k: int = 5) -> Set[str]:
        tokens = re.findall(r'\w+', code.lower())
        if len(tokens) < k:
            return {tuple(tokens)} if tokens else set()
        return {tuple(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}

    def _jaccard_similarity(self, s1: Set[str], s2: Set[str]) -> float:
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def register_code(self, code: str):
        """Register a candidate code snippet into exact, normalized, and fuzzy shingle dedup sets."""
        if not code:
            return
        self.used_exact_hashes.add(hashlib.sha256(code.encode()).hexdigest())
        norm = self._normalize_code(code)
        self.used_norm_hashes.add(hashlib.sha256(norm.encode()).hexdigest())
        self.used_shingles.append(self._get_shingles(norm))

    def load_eval_exclusion_set(self):
        """Load evaluation set hashes for exclusion without polluting fuzzy shingle index."""
        logger.info(f"Loading eval exclusion set from {self.eval_csv_path}...")
        if not os.path.exists(self.eval_csv_path):
            logger.warning(f"Eval CSV not found at {self.eval_csv_path}. Skipping exclusion.")
            return

        count = 0
        for row in _iter_source_rows(self.eval_csv_path):
            code = row.get("code", "")
            if not code:
                continue
            self.eval_exact_hashes.add(hashlib.sha256(code.encode()).hexdigest())
            norm = self._normalize_code(code)
            self.eval_norm_hashes.add(hashlib.sha256(norm.encode()).hexdigest())
            count += 1
        logger.info(f"Loaded {count} eval samples for exclusion.")

    def _process_seed_line(
        self,
        line: str,
        lineno: int,
        filter_valid: bool,
    ) -> Tuple[Optional[Dict], bool]:
        """Parse and validate a single seed JSON line. Returns (record, was_purged)."""
        try:
            data = json.loads(line)
            code = data.get("topic", "")
            pred = data.get("predicate", "")
            gepa = data.get("gepa_info", {})
            existing_anchors = data.get("anchors", [])

            if filter_valid:
                is_val, reasons, anchors = self.validate_predicate(
                    pred, code, gepa, existing_anchors=existing_anchors
                )
                if not is_val:
                    logger.info(f"[Purged invalid baseline seed #{lineno}] Reasons: {reasons}")
                    return None, True
                if not data.get("anchors"):
                    data["anchors"] = anchors

            if code:
                self.register_code(code)
            return data, False
        except Exception as e:
            logger.warning(f"Failed to parse line {lineno}: {e}")
            return None, False

    def load_existing_seeds(self, existing_seeds_path: str, filter_valid: bool = True) -> List[Dict]:
        """Load an existing JSONL seed file into dedup memory, purging invalid/placeholder seeds."""
        resolved = self._resolve_path(existing_seeds_path)
        logger.info(f"Loading existing seeds from {resolved} (filter_valid={filter_valid})...")
        if not os.path.exists(resolved):
            logger.warning(f"Existing seeds file not found at {resolved}.")
            return []

        records: List[Dict] = []
        purged = 0
        with open(resolved, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record, was_purged = self._process_seed_line(line, lineno, filter_valid)
                if was_purged:
                    purged += 1
                elif record is not None:
                    records.append(record)

        logger.info(f"Loaded {len(records)} verified existing seeds (purged {purged} invalid baseline seeds).")
        return records

    def is_duplicate(self, code: str) -> bool:
        """Check if code is duplicate against evaluation benchmarks or already registered candidate seeds."""
        h1 = hashlib.sha256(code.encode()).hexdigest()
        if h1 in self.eval_exact_hashes or h1 in self.used_exact_hashes:
            return True

        norm = self._normalize_code(code)
        h2 = hashlib.sha256(norm.encode()).hexdigest()
        if h2 in self.eval_norm_hashes or h2 in self.used_norm_hashes:
            return True

        s = self._get_shingles(norm)
        for existing_shingles in self.used_shingles:
            if self._jaccard_similarity(s, existing_shingles) > 0.85:
                return True

        return False

    @staticmethod
    def _is_valid_anchor_candidate(anchor: Any, code: str) -> bool:
        """Check if an anchor token is non-generic and present verbatim in code."""
        if not isinstance(anchor, str):
            return False
        clean = anchor.strip(" `\"'.,;:")
        return bool(clean and clean in code and not _is_generic_anchor(clean))

    def _collect_existing_anchors(
        self,
        anchors: List[str],
        existing_anchors: Optional[List[str]],
        code: str,
    ) -> None:
        """Incorporate pre-existing valid anchors."""
        if not (existing_anchors and isinstance(existing_anchors, list)):
            return
        for a in existing_anchors:
            if self._is_valid_anchor_candidate(a, code) and a not in anchors:
                anchors.append(a)

    def _collect_span_anchors(
        self,
        anchors: List[str],
        texts: List[Any],
        code: str,
    ) -> None:
        """Extract valid anchors from backtick spans or structured parts in text."""
        for text in texts:
            if not isinstance(text, str):
                continue
            for span in _extract_spans_from_text(text):
                clean_span = span.strip(" `\"'.,;:")
                if self._is_valid_anchor_candidate(clean_span, code) and clean_span not in anchors:
                    anchors.append(clean_span)

    def _collect_identifier_anchors(
        self,
        anchors: List[str],
        texts: List[Any],
        code: str,
    ) -> None:
        """Extract valid identifier symbols from evidence texts."""
        for text in texts:
            if not isinstance(text, str):
                continue
            for sym in re.findall(r'\b([a-zA-Z_]\w{2,})\b', text):
                if sym.lower() in STOPWORDS or _is_generic_anchor(sym):
                    continue
                if re.search(r'\b' + re.escape(sym) + r'\b', code) and sym not in anchors:
                    anchors.append(sym)

    def extract_anchors(
        self,
        gepa_info: Dict,
        code: str,
        existing_anchors: Optional[List[str]] = None
    ) -> List[str]:
        """Extract and verify non-generic code anchors present verbatim in the code snippet."""
        anchors: List[str] = []
        self._collect_existing_anchors(anchors, existing_anchors, code)

        predicate = gepa_info.get("predicate", "")
        evidence_texts = gepa_info.get("evidence_hooks", []) + [predicate]

        self._collect_span_anchors(anchors, evidence_texts, code)
        self._collect_identifier_anchors(anchors, evidence_texts, code)
        return anchors

    def validate_predicate(
        self,
        predicate: str,
        code: str,
        gepa_info: Dict,
        existing_anchors: Optional[List[str]] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """Validate untrusted predicate against quality floor and anchor grounding requirements."""
        quality = _predicate_quality(predicate, code)
        reasons = list(quality.get("reasons", []))

        anchors = self.extract_anchors(gepa_info, code, existing_anchors=existing_anchors)
        if len(anchors) < 2:
            reasons.append("anchors_too_few")

        is_valid = len(reasons) == 0
        return is_valid, reasons, anchors

    async def gepa_explain(self, code: str, language: str) -> Dict[str, Any]:
        """Generate structured vulnerability analysis via call_structured."""
        system_prompt = """
<role>You are a Senior Vulnerability Researcher (GEPA Explainer).</role>
<task>Analyze the provided code snippet and generate a specific, falsifiable technical predicate about its security status.</task>
<output_format>
Return a JSON object:
{
  "predicate": "The code is vulnerable to [SPECIFIC MECHANISM] in `[LOCATION/FUNCTION]`",
  "evidence_hooks": ["`exact_code_token_or_call_1`", "`exact_code_token_or_call_2`"],
  "uncertainty": "Low/Medium/High",
  "proof_requirements": "What evidence would definitively prove/refute this?"
}
</output_format>
"""
        user_prompt = f"Language: {language}\n\nCode:\n{code}"

        try:
            explanation: GepaExplanation = await call_structured(
                replay_manager=self.replay_manager,
                model=self.explainer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                schema_name="gepa_explanation",
                schema_model=GepaExplanation,
                stage="gepa_explanation",
            )
            return explanation.model_dump()
        except Exception as e:
            mode = getattr(getattr(self.replay_manager, "cassette", None), "mode", "record")
            if mode == "replay":
                logger.error(f"GEPA Explainer failed in replay mode: {e}")
                raise
            logger.warning(f"GEPA Explainer failed: {e}")
            return {
                "predicate": FALLBACK_PREDICATE,
                "evidence_hooks": [],
                "uncertainty": "High",
                "proof_requirements": f"Error: {e}",
            }

    async def gepa_explain_with_retry(self, code: str, language: str, item_idx: int, total: int) -> Dict:
        """Retry wrapper for GEPA explanation with exponential backoff."""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.explain_retries + 2):
            try:
                logger.info(f"[GEPA] item {item_idx}/{total} attempt {attempt}...")
                return await asyncio.wait_for(
                    self.gepa_explain(code, language),
                    timeout=self.explain_timeout_s,
                )
            except Exception as e:
                last_err = e
                mode = getattr(getattr(self.replay_manager, "cassette", None), "mode", "record")
                if mode == "replay":
                    raise
                logger.warning(
                    f"[GEPA] item {item_idx}/{total} attempt {attempt} failed: {e}"
                )
                if attempt <= self.explain_retries:
                    await asyncio.sleep(min(2 * attempt, 5))
        logger.error(
            f"[GEPA] item {item_idx}/{total} exhausted retries; using fallback. Last error: {last_err}"
        )
        return {
            "predicate": FALLBACK_PREDICATE,
            "evidence_hooks": [],
            "uncertainty": "High",
            "proof_requirements": f"GEPA timeout/retry failure: {last_err}",
        }

    def _sample_reservoir(self, n: int, target_lang: str) -> List[Tuple[str, str, Any]]:
        """Collect n candidates via reservoir sampling using lazy CSV streaming."""
        reservoir: List[Tuple[str, str, Any]] = []
        count = 0
        dummy_rejections: Dict[str, int] = defaultdict(int)
        for code, lang, safety in _get_matching_source_candidates(self.source_csv_path, target_lang):
            if not self._is_valid_expansion_snippet(code, dummy_rejections):
                continue

            count += 1
            if len(reservoir) < n:
                reservoir.append((code, lang, safety))
            else:
                j = self.replay_manager.rng.randint(0, count - 1)
                if j < n:
                    reservoir[j] = (code, lang, safety)
        return reservoir

    async def get_seeds(self, n: int, target_lang: str = "c") -> List[Dict]:
        """Deterministic Reservoir Sampling from CSV (Legacy Compatibility Path)."""
        logger.info(f"Scanning {self.source_csv_path} with Reservoir Sampling (n={n})...")

        reservoir = self._sample_reservoir(n, target_lang)
        logger.info(f"Selected {len(reservoir)} candidates. Running GEPA Explainer...")

        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))
        total = len(reservoir)

        async def explain_task(idx: int, cand: Tuple[str, str, Any]) -> Dict[str, Any]:
            async with semaphore:
                code, lang, safety = cand
                gepa_info = await self.gepa_explain_with_retry(code, lang, idx, total)
                logger.info(f"[GEPA] item {idx}/{total} complete")
                return {
                    "topic": code,
                    "predicate": gepa_info.get("predicate", FALLBACK_PREDICATE),
                    "gepa_info": gepa_info,
                    "language": lang,
                    "original_safety": safety
                }

        tasks = [explain_task(i + 1, c) for i, c in enumerate(reservoir)]
        return await asyncio.gather(*tasks)

    def _is_valid_expansion_snippet(self, code: str, rejection_counts: Dict[str, int]) -> bool:
        """Check snippet length limits and duplicate status with full telemetry breakdown."""
        if not code:
            rejection_counts["empty_snippet"] += 1
            return False
        if len(code) < 50 or len(code) > 12000:
            rejection_counts["snippet_length_out_of_range"] += 1
            return False
        if self.is_duplicate(code):
            rejection_counts["duplicate_snippet"] += 1
            return False
        return True

    async def _process_expansion_candidate(
        self,
        code: str,
        lang: str,
        safety: Any,
        config: SeedExpansionConfig,
        state: SeedExpansionState,
    ) -> None:
        """Process a single candidate through GEPA explanation, validation, and persistence."""
        state.total_calls += 1
        gepa_info = await self.gepa_explain_with_retry(
            code, lang, len(state.accepted_seeds) + 1, config.target_total
        )

        predicate = gepa_info.get("predicate", "")
        est_tokens = len(code) // 4 + len(predicate) // 4 + 200
        state.total_tokens_est += est_tokens

        is_valid, reasons, anchors = self.validate_predicate(predicate, code, gepa_info)
        state.record_attempt(predicate, len(code), is_valid, reasons, anchors)

        if is_valid:
            seed_record = {
                "topic": code,
                "predicate": predicate,
                "gepa_info": gepa_info,
                "language": lang,
                "original_safety": safety,
                "anchors": anchors,
            }
            self.register_code(code)
            await state.record_success(seed_record, config.output_path, config.target_total)
        else:
            state.record_rejection(reasons)

    async def _process_candidate_batch(
        self,
        batch: List[Tuple[str, str, Any]],
        config: SeedExpansionConfig,
        state: SeedExpansionState,
    ) -> bool:
        """Process a batch of candidates. Returns True if a stop rule was triggered."""
        for code, lang, safety in batch:
            if not self._is_valid_expansion_snippet(code, state.rejection_counts):
                continue

            await self._process_expansion_candidate(code, lang, safety, config, state)

            stop_reason = state.evaluate_stop_rules(config)
            if stop_reason:
                state.stop_reason = stop_reason
                return True
        return False

    async def _run_expansion_loop(
        self,
        config: SeedExpansionConfig,
        state: SeedExpansionState,
    ) -> None:
        """Stream candidates from source CSV lazily and process until target is met or stop rule triggers."""
        logger.info(f"Streaming candidates from {self.source_csv_path}...")
        candidate_iter = _get_matching_source_candidates(self.source_csv_path, config.target_lang)
        while True:
            batch = await asyncio.to_thread(_load_candidate_batch, candidate_iter, 50)
            if not batch:
                break
            should_stop = await self._process_candidate_batch(batch, config, state)
            if should_stop:
                break

    def _init_expansion_seeds(
        self,
        config: SeedExpansionConfig,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Load verified existing seeds and initialize output file."""
        existing_seeds: List[Dict] = []
        if config.existing_seeds_path:
            existing_seeds = self.load_existing_seeds(
                config.existing_seeds_path,
                filter_valid=config.purge_invalid_existing
            )

        accepted_seeds = list(existing_seeds)
        needed_new = max(0, config.target_total - len(accepted_seeds))
        logger.info(
            f"Seed Expansion: {len(accepted_seeds)} verified baseline seeds retained. "
            f"Target: {config.target_total} total (+{needed_new} remaining)."
        )

        os.makedirs(os.path.dirname(os.path.abspath(config.output_path)), exist_ok=True)
        with open(config.output_path, 'w', encoding='utf-8') as out_f:
            for s in accepted_seeds:
                out_f.write(json.dumps(s) + "\n")
            out_f.flush()

        return existing_seeds, accepted_seeds

    async def expand_seeds(
        self,
        config: Optional[SeedExpansionConfig] = None,
        **kwargs: Any
    ) -> List[Dict]:
        """Incremental seed expansion with untrusted predicate validation, purge of invalid baselines, and dual stop rules."""
        import sys
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

        cfg = SeedExpansionConfig.from_args(config, **kwargs)
        existing_seeds, accepted_seeds = self._init_expansion_seeds(cfg)

        if len(accepted_seeds) >= cfg.target_total:
            logger.info("Target total already reached with verified existing seeds.")
            return accepted_seeds

        state = SeedExpansionState(
            existing_seeds=existing_seeds,
            accepted_seeds=accepted_seeds,
            window_size=cfg.window_size,
            attempts_path=cfg.attempts_path,
        )
        try:
            await self._run_expansion_loop(cfg, state)
        except Exception:
            state.stop_reason = "expansion_error"
            raise
        finally:
            state.save_telemetry(cfg)

        logger.info(
            f"Expansion finished. Final clean count: {len(state.accepted_seeds)} seeds. "
            f"Reason: {state.stop_reason}."
        )
        return state.accepted_seeds


def _load_candidate_batch(
    cand_iter: Iterator[Tuple[str, str, Any]],
    batch_size: int = 50,
) -> List[Tuple[str, str, Any]]:
    """Fetch next chunk of candidates from iterator."""
    batch: List[Tuple[str, str, Any]] = []
    for _ in range(batch_size):
        try:
            batch.append(next(cand_iter))
        except StopIteration:
            break
    return batch


def _export_seeds_to_jsonl(output_path: str, seeds: List[Dict]) -> None:
    """Synchronously write seed records to JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for seed in seeds:
            f.write(json.dumps(seed) + "\n")


async def main():
    parser = argparse.ArgumentParser(description="CVE Seed Loader & Expander for BARRED (Deterministic)")
    parser.add_argument("--eval-csv", default="kaggle_notebooks/cve_decision_benchmark_v1.csv")
    parser.add_argument("--source-csv", default="kaggle_notebooks/CVEFixes.csv")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--lang", default="c")
    parser.add_argument("--output", default="scenarios/debate/cve_seeds.jsonl")
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["record", "replay"], default="record")
    parser.add_argument("--cassette", default="artifacts/cassettes/gepa_seeds.json")

    # Expansion mode arguments
    parser.add_argument("--existing-seeds", default="", help="Path to existing JSONL seeds to preserve as prefix and deduplicate against")
    parser.add_argument("--target-total", type=int, default=0, help="Target total seeds count (e.g. 500)")
    parser.add_argument("--explainer-model", default="", help="Override GEPA explainer model (default: ollama/gpt-oss:120b-cloud)")
    parser.add_argument("--min-marginal-yield", type=float, default=0.20, help="Minimum acceptance rate over rolling window before early halt")
    parser.add_argument("--window-size", type=int, default=50, help="Rolling window size for marginal yield calculation")
    parser.add_argument("--max-calls", type=int, default=1500, help="Maximum LLM API calls before halting")
    parser.add_argument("--max-tokens", type=int, default=6000000, help="Maximum token budget before halting")
    parser.add_argument("--telemetry-out", default="artifacts/telemetry/seed_expansion_summary.json", help="Path to write expansion telemetry JSON")
    parser.add_argument("--attempts-out", default="artifacts/attempts/seed_expansion_attempts.jsonl", help="Path to write append-only attempts JSONL")
    parser.add_argument("--no-purge-invalid", dest="purge_invalid", action="store_false", default=True, help="Disable purging invalid baseline seeds")
    args = parser.parse_args()

    replay_mgr = ReplayManager.from_config(args.run_id, args.seed, args.cassette, args.mode)
    loader = CVESeedLoader(
        eval_csv_path=args.eval_csv,
        source_csv_path=args.source_csv,
        replay_manager=replay_mgr,
        explainer_model=args.explainer_model or None,
    )
    loader.load_eval_exclusion_set()

    if args.existing_seeds or args.target_total > 0:
        target_total = args.target_total if args.target_total > 0 else (DEFAULT_EXPANSION_BATCH_SIZE + args.n)
        config = SeedExpansionConfig(
            target_total=target_total,
            existing_seeds_path=args.existing_seeds or None,
            output_path=args.output,
            min_marginal_yield=args.min_marginal_yield,
            window_size=args.window_size,
            max_calls=args.max_calls,
            max_tokens=args.max_tokens,
            target_lang=args.lang,
            telemetry_path=args.telemetry_out,
            attempts_path=args.attempts_out,
            purge_invalid_existing=args.purge_invalid,
        )
        await loader.expand_seeds(config)
    else:
        seeds = await loader.get_seeds(args.n, args.lang)
        await asyncio.to_thread(_export_seeds_to_jsonl, args.output, seeds)
        logger.info(f"Exported {len(seeds)} seeds to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
