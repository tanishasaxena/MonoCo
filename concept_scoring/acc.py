from utils import *

scorer = ConceptScorer("config.yaml")

print(scorer.heart_disease_absent_concepts)
print(scorer.heart_disease_present_concepts)
print(scorer.prompt)
# to run, use the other functions from the scorer. 