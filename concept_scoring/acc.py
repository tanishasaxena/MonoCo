from utils import *
import pandas as pd
scorer = ConceptScorer("config.yaml")

# print(scorer.heart_disease_absent_concepts)
# print(scorer.heart_disease_present_concepts)
# print(scorer.prompt)
# to run, use the other functions from the scorer. 
features = None 
# load df into features including labels
concept_scores = []
for i, row in features.iterrows():
    row_dict = row.to_dict()
    label = row_dict["num"]
    row_dict.pop("num", None)
    scores = scorer.get_concept_scores(features)
    corrected_scores = scorer.automatic_concept_correction(scores, label)
    concept_scores.append(corrected_scores)
    