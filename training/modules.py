import torch
from torch import nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model, GPT2TokenizerFast, RobertaModel
import torch.nn.functional as F
from utils import top_k_top_p_filtering
    
class TCBL(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, concept_dim=20):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.concept_head = nn.Linear(hidden_dim, concept_dim)

    def forward(self, x):
        h = self.backbone(x)
        concepts = self.concept_head(h)  # shape: (batch, num_concepts)
        return concepts # return bottleneck latents optionally
    

    # def __init__(self, in_dim, hidden_dim=128, num_concepts=20):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(in_dim, hidden_dim),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(hidden_dim),
    #         nn.Linear(hidden_dim, hidden_dim),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(hidden_dim),
    #         nn.Linear(hidden_dim, num_concepts)
    #     )

    # def forward(self, x):
    #     return self.net(x)