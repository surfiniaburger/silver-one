import os
import csv
import json
import hashlib
import random
import re
import asyncio
import logging
import argparse
from typing import Set, List, Dict, Optional
from agentbeats.replay import ReplayManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cve_seed_loader")

class CVESeedLoader:
    def __init__(
        self, 
        eval_csv_path: str, 
        source_csv_path: str, 
        replay_manager: ReplayManager,
        explainer_model: str = None
    ):
        self.eval_csv_path = eval_csv_path
        self.source_csv_path = source_csv_path
        self.replay_manager = replay_manager
        self.explainer_model = explainer_model or os.getenv("GEPA_MODEL", "ollama/gpt-oss:120b-cloud ")
        self.explain_timeout_s = float(os.getenv("GEPA_EXPLAIN_TIMEOUT_S", "120"))
        self.explain_retries = int(os.getenv("GEPA_EXPLAIN_RETRIES", "2"))
        self.max_concurrency = int(os.getenv("GEPA_EXPLAIN_CONCURRENCY", "5"))
        self.used_exact_hashes: Set[str] = set()
        self.used_norm_hashes: Set[str] = set()
        self.used_shingles: List[Set[str]] = []
        
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

    def load_eval_exclusion_set(self):
        logger.info(f"Loading eval exclusion set from {self.eval_csv_path}...")
        if not os.path.exists(self.eval_csv_path):
            logger.warning("Eval CSV not found. Skipping exclusion.")
            return

        import sys
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with open(self.eval_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("code", "")
                if not code:
                    continue
                
                self.used_exact_hashes.add(hashlib.sha256(code.encode()).hexdigest())
                norm = self._normalize_code(code)
                self.used_norm_hashes.add(hashlib.sha256(norm.encode()).hexdigest())
                self.used_shingles.append(self._get_shingles(norm))
        logger.info(f"Loaded {len(self.used_exact_hashes)} eval samples for exclusion.")

    def is_duplicate(self, code: str) -> bool:
        h1 = hashlib.sha256(code.encode()).hexdigest()
        if h1 in self.used_exact_hashes:
            return True
        
        norm = self._normalize_code(code)
        h2 = hashlib.sha256(norm.encode()).hexdigest()
        if h2 in self.used_norm_hashes:
            return True
            
        s = self._get_shingles(norm)
        for existing_shingles in self.used_shingles:
            if self._jaccard_similarity(s, existing_shingles) > 0.85:
                return True
                
        return False

    async def gepa_explain(self, code: str, language: str) -> Dict:
        system_prompt = """
<role>You are a Senior Vulnerability Researcher (GEPA Explainer).</role>
<task>Analyze the provided code snippet and generate a specific, falsifiable technical predicate about its security status.</task>
<output_format>
Return a JSON object:
{
  "predicate": "The code is vulnerable to [SPECIFIC MECHANISM] in [LOCATION]",
  "evidence_hooks": ["hook1", "hook2"],
  "uncertainty": "Low/Medium/High",
  "proof_requirements": "What evidence would definitively prove/refute this?"
}
</output_format>
"""
        user_prompt = f"Language: {language}\n\nCode:\n{code}"

        try:
            response = await self.replay_manager.acompletion(
                model=self.explainer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
            )
            res_content = response.choices[0].message.content.strip()
            if "```" in res_content:
                res_content = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', res_content, flags=re.DOTALL).strip()
            
            return json.loads(res_content)
        except Exception as e:
            logger.error(f"GEPA Explainer failed: {e}")
            return {"predicate": "Vulnerability suspected.", "evidence_hooks": []}

    async def gepa_explain_with_retry(self, code: str, language: str, item_idx: int, total: int) -> Dict:
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
                logger.warning(
                    f"[GEPA] item {item_idx}/{total} attempt {attempt} failed: {e}"
                )
                if attempt <= self.explain_retries:
                    await asyncio.sleep(min(2 * attempt, 5))
        logger.error(
            f"[GEPA] item {item_idx}/{total} exhausted retries; using fallback. Last error: {last_err}"
        )
        return {
            "predicate": "Vulnerability suspected.",
            "evidence_hooks": [],
            "uncertainty": "High",
            "proof_requirements": f"GEPA timeout/retry failure: {last_err}",
        }

    async def get_seeds(self, n: int, target_lang: str = "c") -> List[Dict]:
        """Deterministic Reservoir Sampling from CSV."""
        reservoir = []
        count = 0
        logger.info(f"Scanning {self.source_csv_path} with Reservoir Sampling (n={n})...")
        
        import sys
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        
        with open(self.source_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lang = (row.get("language") or "").strip().lower()
                if target_lang:
                    langs = re.split(r'[,/ \t]+', lang)
                    if target_lang not in langs:
                        continue
                
                code = row.get("code", "")
                if not code or len(code) < 50 or self.is_duplicate(code):
                    continue
                
                count += 1
                if len(reservoir) < n:
                    reservoir.append((code, lang, row.get("safety")))
                else:
                    # Deterministic swap based on seeded RNG
                    j = self.replay_manager.rng.randint(0, count - 1)
                    if j < n:
                        reservoir[j] = (code, lang, row.get("safety"))

        logger.info(f"Selected {len(reservoir)} candidates. Running GEPA Explainer...")
        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))
        total = len(reservoir)

        async def explain_task(idx: int, cand):
            async with semaphore:
                code, lang, safety = cand
                gepa_info = await self.gepa_explain_with_retry(code, lang, idx, total)
                logger.info(f"[GEPA] item {idx}/{total} complete")
                return {
                    "topic": code,
                    "predicate": gepa_info.get("predicate", "Vulnerability suspected."),
                    "gepa_info": gepa_info,
                    "language": lang,
                    "original_safety": safety
                }

        tasks = [explain_task(i + 1, c) for i, c in enumerate(reservoir)]
        seeds = await asyncio.gather(*tasks)
        return seeds

async def main():
    parser = argparse.ArgumentParser(description="CVE Seed Loader for BARRED (Deterministic)")
    parser.add_argument("--eval-csv", default="kaggle_notebooks/cve_decision_benchmark_v1.csv")
    parser.add_argument("--source-csv", default="kaggle_notebooks/CVEFixes.csv")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--lang", default="c")
    parser.add_argument("--output", default="scenarios/debate/cve_seeds.jsonl")
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["record", "replay"], default="record")
    parser.add_argument("--cassette", default="artifacts/cassettes/gepa_seeds.json")
    args = parser.parse_args()

    replay_mgr = ReplayManager.from_config(args.run_id, args.seed, args.cassette, args.mode)
    loader = CVESeedLoader(args.eval_csv, args.source_csv, replay_mgr)
    loader.load_eval_exclusion_set()
    
    seeds = await loader.get_seeds(args.n, args.lang)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for seed in seeds:
            f.write(json.dumps(seed) + "\n")
            
    logger.info(f"Exported {len(seeds)} seeds to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
