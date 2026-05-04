import os
import csv
import json
import hashlib
import random
import re
import asyncio
import logging
import litellm
import argparse
from typing import Set, List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cve_seed_loader")

class CVESeedLoader:
    def __init__(
        self, 
        eval_csv_path: str, 
        source_csv_path: str, 
        explainer_model: str = None
    ):
        self.eval_csv_path = eval_csv_path
        self.source_csv_path = source_csv_path
        self.explainer_model = explainer_model or os.getenv("GEPA_MODEL", "gemini/gemini-2.0-flash-exp") # Default to a strong model
        self.used_exact_hashes: Set[str] = set()
        self.used_norm_hashes: Set[str] = set()
        self.used_shingles: List[Set[str]] = [] # For fuzzy check
        
    def _normalize_code(self, code: str) -> str:
        # Strip comments
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Strip whitespace and normalize case (optional, keeping case for C/C++)
        code = "".join(code.split()).lower()
        return code

    def _get_shingles(self, code: str, k: int = 5) -> Set[str]:
        # Simple k-gram shingles for fuzzy dedup
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

        # Increase CSV field limit for large code snippets
        import sys
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with open(self.eval_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("code", "")
                if not code:
                    continue
                
                # k1: Exact
                self.used_exact_hashes.add(hashlib.sha256(code.encode()).hexdigest())
                
                # k2: Normalized
                norm = self._normalize_code(code)
                self.used_norm_hashes.add(hashlib.sha256(norm.encode()).hexdigest())
                
                # k3: Shingles
                self.used_shingles.append(self._get_shingles(norm))
        logger.info(f"Loaded {len(self.used_exact_hashes)} eval samples for exclusion.")

    def is_duplicate(self, code: str) -> bool:
        # k1: Exact
        h1 = hashlib.sha256(code.encode()).hexdigest()
        if h1 in self.used_exact_hashes:
            return True
        
        # k2: Normalized
        norm = self._normalize_code(code)
        h2 = hashlib.sha256(norm.encode()).hexdigest()
        if h2 in self.used_norm_hashes:
            return True
            
        # k3: Fuzzy
        s = self._get_shingles(norm)
        for existing_shingles in self.used_shingles:
            if self._jaccard_similarity(s, existing_shingles) > 0.85:
                return True
                
        return False

    async def gepa_explain(self, code: str, language: str) -> Dict:
        """
        GEPA Explainer: Takes a raw vulnerable/safe snippet and proposes a specific technical predicate.
        """
        system_prompt = """
<role>You are a Senior Vulnerability Researcher (GEPA Explainer).</role>
<task>Analyze the provided code snippet and generate a specific, falsifiable technical predicate about its security status.</task>
<context>
The code is sourced from CVEFixes. It is likely either a vulnerable version or a fixed version of a real CVE.
Your goal is to provide a 'Proposed Predicate' that can be debated.
</context>
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
            response = await litellm.acompletion(
                model=self.explainer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            res_content = response.choices[0].message.content.strip()
            # Handle cases where model returns ```json ... ``` even with response_format
            if "```" in res_content:
                res_content = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', res_content, flags=re.DOTALL).strip()
            
            return json.loads(res_content)
        except Exception as e:
            logger.error(f"GEPA Explainer failed for code snippet: {e}")
            return {
                "predicate": "The code contains a security vulnerability related to memory safety or logic errors.",
                "evidence_hooks": [],
                "uncertainty": "High",
                "proof_requirements": "Manual audit required."
            }

    async def get_seeds(self, n: int, target_lang: str = "c") -> List[Dict]:
        candidates = []
        logger.info(f"Scanning {self.source_csv_path} for candidates (lang: {target_lang})...")
        
        # Increase CSV field limit for large code snippets
        import sys
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        
        with open(self.source_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(candidates) >= n:
                    break
                    
                lang = (row.get("language") or "").strip().lower()
                if target_lang:
                    langs = re.split(r'[,/ \t]+', lang)
                    if target_lang not in langs:
                        continue
                
                code = row.get("code", "")
                if not code or len(code) < 50:
                    continue
                    
                if self.is_duplicate(code):
                    continue
                
                candidates.append((code, lang, row.get("safety")))
                logger.info(f"Found candidate {len(candidates)}/{n}")

        logger.info(f"Running GEPA Explainer on {len(candidates)} candidates...")
        semaphore = asyncio.Semaphore(5) # Concurrency limit

        async def explain_task(cand):
            async with semaphore:
                code, lang, safety = cand
                gepa_info = await self.gepa_explain(code, lang)
                return {
                    "topic": code,
                    "predicate": gepa_info["predicate"],
                    "gepa_info": gepa_info,
                    "language": lang,
                    "original_safety": safety
                }

        tasks = [explain_task(c) for c in candidates]
        seeds = await asyncio.gather(*tasks)
        return seeds

async def main():
    parser = argparse.ArgumentParser(description="CVE Seed Loader for BARRED (GEPA-first flow)")
    parser.add_argument("--eval-csv", default="kaggle_notebooks/cve_decision_benchmark_v1.csv")
    parser.add_argument("--source-csv", default="kaggle_notebooks/CVEFixes.csv")
    parser.add_argument("--n", type=int, default=10, help="Number of seeds to generate")
    parser.add_argument("--lang", default="c", help="Target language (e.g., c, cpp, python)")
    parser.add_argument("--output", default="scenarios/debate/cve_seeds.jsonl")
    args = parser.parse_args()

    loader = CVESeedLoader(args.eval_csv, args.source_csv)
    loader.load_eval_exclusion_set()
    
    seeds = await loader.get_seeds(args.n, args.lang)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for seed in seeds:
            f.write(json.dumps(seed) + "\n")
            
    logger.info(f"Exported {len(seeds)} seeds to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
