import yaml
import json
import openai

class ConceptScorer:
    def __init__(self, config_path: str):
        """
        Loads configuration and initializes the concept scorer.
        Expects YAML file with:
            heart_disease_present_concepts: path_to_present_concepts.txt
            heart_disease_absent_concepts: path_to_absent_concepts.txt
            prompt: instruction prompt string
        """
        # Load YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.prompt = config.get("prompt", "").strip()

        # Load concept files
        present_path = config.get("heart_disease_present_concepts")
        absent_path = config.get("heart_disease_absent_concepts")

        with open(present_path, 'r') as f:
            self.heart_disease_present_concepts = [line.strip() for line in f if line.strip()]
        with open(absent_path, 'r') as f:
            self.heart_disease_absent_concepts = [line.strip() for line in f if line.strip()]

        # Initialize OpenAI client
        openai.api_key = config.get("api_key", None)

    def get_concept_scores(self, features: dict):
        """
        Evaluates concept scores given patient features.

        Args:
            features (dict): mapping from feature_name -> value

        Returns:
            dict[str, float]: concept -> score in [0, 1]
        """
        all_concepts = self.heart_disease_present_concepts + self.heart_disease_absent_concepts
        feature_desc = "\n".join([f"{k}: {v}" for k, v in features.items()])
        concept_list = "\n".join([f"- {c}" for c in all_concepts])

        query = f"""
        {self.prompt}

        Patient data:
        {feature_desc}

        Concepts to evaluate (score 0 to 1, where 1 = strongly present, 0 = absent):
        {concept_list}

        Return a JSON object: {{ "concept": score, ... }}
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": query}],
                temperature=0.0
            )
            raw = response.choices[0].message.content.strip()
            scores = json.loads(raw)
        except Exception as e:
            print(f"Error obtaining or parsing concept scores: {e}")
            scores = {}

        return scores
    
    def automatic_concept_correction(self, concept_scores: dict, label: int):
        """
        Clips all concept scores to [0,1].
        If label == 0 (no disease): set all 'present' concepts to 0.
        If label > 0 (disease): set all 'absent' concepts to 0.
        """
        corrected = {}

        for concept, score in concept_scores.items():
            clipped = float(np.clip(score, 0, 1))

            if label == 0 and concept in self.heart_disease_present_concepts:
                corrected[concept] = 0.0
            elif label > 0 and concept in self.heart_disease_absent_concepts:
                corrected[concept] = 0.0
            else:
                corrected[concept] = clipped

        return corrected
    
import torch
import torch.nn as nn

class Parameterized_ACC(nn.Module):
    """
    Parameterized Automatic Concept Correction (ACC)
    A one-to-one linear mapping: each input concept has its own weight and bias.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim (int): number of input (and output) concepts
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies element-wise affine transformation:
            y_i = w_i * x_i + b_i
        Args:
            x (torch.Tensor): [batch_size, dim]
        Returns:
            torch.Tensor: [batch_size, dim]
        """
        return x * self.weight + self.bias
