from uci_data import UCIData
import pandas as pd
import os
import json
import hashlib
from openai import OpenAI

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

data_storage_file = "augmented_heart_disease_data.json"

# data stores id mapped to augmented feature for each feature vector
if os.path.exists(data_storage_file):
    with open(data_storage_file, "r") as f:
        data_storage = json.load(f)
else:
    print(f"{data_storage_file} not found, initializing empty storage.")
    data_storage = {}

def augment_features(features: pd.DataFrame, num_aug: int) -> pd.DataFrame:
    """
    Reform UCI Heart Disease feature vector into a richer feature representation.
    Uses caching to minimize API calls.
    """
    new_features = {}
    not_in_storage = {}
    # use row index as id instead of hashing because I don't like hashing ¯\_(ツ)_/¯
    for idx, feature_vector in enumerate(features):
        # only augment the first num_aug entries
        if idx >= num_aug:
            break
        
        id = str(idx)
        # Utilize storage
        if id in data_storage:
            new_features[id] = data_storage[id]

        else:
            prompt = f"""
            You are a data scientist optimizing features for a heart disease classifier.
            Given this raw feature vector: {feature_vector}

            Return a JSON object with:
            - The same base features, cleaned/scaled logically
            - New domain-aware engineered features
            - Numeric or boolean values only
            - No text explanation
            Just output valid JSON.
            """

            # response = client.chat.completions.create(
            #     model="gpt-5",  # or "gpt-4o" if needed
            #     messages=[{"role": "user", "content": prompt}],
            #     response_format={"type": "json_object"}
            # )

            # reformed = response.choices[0].message.parsed
            reformed = json.loads("{\"text_f\":\"test_v\"}")

            # store result
            new_features[id] = reformed
            # add to data storage iff it's new
            not_in_storage[id] = reformed

    # persist data storage
    with open(data_storage_file, "w") as f:
        print(f"Saving augmented data to {data_storage_file}")
        json.dump(not_in_storage, f, indent=2)

    return new_features