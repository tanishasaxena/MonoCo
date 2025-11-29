import yaml
import json
import numpy as np
import asyncio
import functools

from openai import OpenAI


def _run_sync(func, *args, **kwargs):
    """Run sync SDK call in thread executor for async parallelism."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


class ConceptScorer:
    def __init__(self, config_path: str):
        # Load YAML
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.prompt = config["prompt"].strip()

        with open(config["heart_disease_present_concepts"], "r") as f:
            self.present = [x.strip() for x in f if x.strip()]
        with open(config["heart_disease_absent_concepts"], "r") as f:
            self.absent = [x.strip() for x in f if x.strip()]

        self.all_concepts = self.present + self.absent

        self.client = OpenAI(api_key=config["api_key"])
        self.model = "gpt-4.1-mini"

    # ---------------------------------------------------------
    # SINGLE EXAMPLE CALL — sent to async executor
    # ---------------------------------------------------------
    def _call_one(self, features: dict):
        feature_str = "\n".join(f"{k}: {v}" for k, v in features.items())
        concept_str = "\n".join(f"- {c}" for c in self.all_concepts)

        prompt = f"""
{self.prompt}

Patient data:
{feature_str}

Concepts to score (0-1):
{concept_str}

Return ONLY a JSON object.
""".strip()

        # THIS is the CORRECT format for openai 2.8.1
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        return json.loads(raw)

    # ---------------------------------------------------------
    # BATCH (TRUE PARALLELISM)
    # ---------------------------------------------------------
    async def _async_batch(self, dicts, concurrency=12):
        sem = asyncio.Semaphore(concurrency)

        async def wrap(feats):
            async with sem:
                return await _run_sync(self._call_one, feats)

        tasks = [wrap(d) for d in dicts]
        return await asyncio.gather(*tasks)

    def get_concept_scores_batch(self, dicts, concurrency=12):
        return asyncio.run(self._async_batch(dicts, concurrency))

    # ---------------------------------------------------------
    # YOUR ORIGINAL ACC — untouched
    # ---------------------------------------------------------
    def automatic_concept_correction(self, concept_scores: dict, label: int):
        corrected = []
        n_present = len(self.present)

        for i, (_, score) in enumerate(concept_scores.items()):
            clipped = float(np.clip(score, 0, 1))

            if label == 0 and i < n_present:
                corrected.append(0.0)
            elif label > 0 and i >= n_present:
                corrected.append(0.0)
            else:
                corrected.append(clipped)

        return corrected


# ---------------------------------------------------------
# Parameterized ACC (unchanged)
# ---------------------------------------------------------
import torch
import torch.nn as nn

class Parameterized_ACC(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.weight + self.bias
