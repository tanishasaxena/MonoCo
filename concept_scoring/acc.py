import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import *
import pandas as pd
from data import UCIData
scorer = ConceptScorer("concept_scoring/config.yaml")


AUGMENT = False
# print(scorer.heart_disease_absent_concepts)
# print(scorer.heart_disease_present_concepts)
# print(scorer.prompt)
# to run, use the other functions from the scorer.
dataset = UCIData()
dataset.augment() if AUGMENT else None
features, generated_features = dataset.get_data()

# load df into features including labels
concept_scores = []
for i, row in features.iterrows():
    break
    row_dict = row.to_dict()
    label = row_dict["num"]
    row_dict.pop("num", None)
    scores = scorer.get_concept_scores(row_dict)
    corrected_scores = scorer.automatic_concept_correction(scores, label)
    concept_scores.append(corrected_scores)
    