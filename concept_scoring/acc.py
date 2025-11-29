import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import openai, inspect, sys
print("PYTHONPATH:", sys.path)
print("OPENAI MODULE:", openai)
print("OPENAI FILE:", inspect.getfile(openai))
print("DIR OF MODULE:", list(openai.__dict__.keys())[:40])

import pandas as pd
from utils import ConceptScorer

scorer = ConceptScorer("config.yaml")

FILE_PATH = "/home/iron/code/MonoCo/data/heart_disease/heart_disease_uci.csv"
OUT_PATH  = "/home/iron/code/MonoCo/data/heart_disease/heart_disease_uci_concepts.csv"

features = pd.read_csv(FILE_PATH)

rows = []
BATCH = 32

for start in range(0, len(features), BATCH):
    print(start)
    end = start + BATCH
    chunk = features.iloc[start:end]

    feature_dicts = []
    labels = []

    for _, row in chunk.iterrows():
        d = row.to_dict()
        lbl = d.pop("num", None)
        feature_dicts.append(d)
        labels.append(lbl)

    # one batched API call
    scored = scorer.get_concept_scores_batch(feature_dicts)
    print(scored, labels)
    for corrected_raw, lbl in zip(scored, labels):
        corrected = scorer.automatic_concept_correction(corrected_raw, lbl)
        merged = {f"concept_{j}": v for j, v in enumerate(corrected)}
        merged["label"] = lbl
        rows.append(merged)

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
