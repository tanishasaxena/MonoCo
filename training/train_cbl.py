import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, concatenate_datasets
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from modules import CBL
import time
from utils import elastic_net_penalty, mean_pooling

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data import UCIDataset

CBL_PATH = "model_checkpoints/cbl/"

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)

def build_loaders(curr_dataset, mode):
    return torch.utils.data.DataLoader(curr_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    dataset = UCIDataset()
    # regular = 920, generated = 100, total: 1020
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [340, 680])

    print("tokenizing...")
    lora_config = LoraConfig(r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                  "down_proj"], bias="none", task_type=TaskType.FEATURE_EXTRACTION)
    
    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token

    # TODO: load concept set from acc.py
    concept_set = None

    print("creating loader...")
    train_loader = build_loaders(train_dataset, mode="train")
    val_loader = build_loaders(val_dataset, mode="valid")

    print("preparing backbone...")
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM = get_peft_model(preLM, lora_config)
    preLM.print_trainable_parameters()
    lora_layers = filter(lambda p: p.requires_grad, preLM.parameters())
    opt_prelm = torch.optim.Adam(lora_layers, lr=5e-5)
    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    opt_cbl = torch.optim.Adam(cbl.parameters(), lr=5e-5)

    print("start training...")
    best_loss = float('inf')

    if not os.path.exists(CBL_PATH):
        os.makedirs(CBL_PATH)

    start = time.time()
    epochs = CFG.epoch[args.dataset]

    for e in range(epochs):
        print("Epoch ", e+1, ":")
        preLM.train()
        cbl.train()
        training_concept_loss = []
        training_word_loss = []
        training_reg_loss = []

        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
            word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            concepts, unsup, vocabs = cbl(features.float())
            concept_loss = torch.nn.CrossEntropyLoss()(concepts[:, :-1, :].reshape(-1, len(concept_set)), concept_label.reshape(-1))
            word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
            loss = concept_loss + word_loss
            reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
            loss += 1.0 * reg
            opt_prelm.zero_grad()
            opt_cbl.zero_grad()
            loss.backward()
            opt_prelm.step()
            opt_cbl.step()

            # Note: removed classifier steps

            _, unsup, _ = cbl(features.detach().float())
            opt_cbl.zero_grad()
            opt_cbl.step()

            print("batch", str(i), "concept loss:", concept_loss.detach().cpu().numpy(), "word loss:", word_loss.detach().cpu().numpy(), "reg loss:", reg.detach().cpu().numpy(), end="\r")
            training_concept_loss.append(concept_loss.detach().cpu().numpy())
            training_word_loss.append(word_loss.detach().cpu().numpy())
            training_reg_loss.append(reg.detach().cpu().numpy())
        
        avg_training_concept_loss = sum(training_concept_loss)/len(training_concept_loss)
        avg_training_word_loss = sum(training_word_loss) / len(training_word_loss)
        avg_training_reg_loss = sum(training_reg_loss)/len(training_reg_loss)
        print("training concept loss:", avg_training_concept_loss, "training word loss:", avg_training_word_loss, "training reg loss: ", avg_training_reg_loss)

        # Evaluation
        preLM.eval()
        cbl.eval()
        val_concept_loss = []
        val_word_loss = []
        val_reg_loss = []
        for i, batch in enumerate(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
            word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
            with torch.no_grad():
                features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                concepts, unsup, vocabs = cbl(features.float())
            concept_loss = torch.nn.CrossEntropyLoss()(concepts[:, :-1, :].reshape(-1, len(concept_set)), concept_label.reshape(-1))
            word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
            reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
            val_concept_loss.append(concept_loss.detach().cpu().numpy())
            val_word_loss.append(word_loss.detach().cpu().numpy())
            val_reg_loss.append(reg.detach().cpu().numpy())
        
        avg_val_concept_loss = sum(val_concept_loss) / len(val_concept_loss)
        avg_val_word_loss = sum(val_word_loss) / len(val_word_loss)
        avg_val_reg_loss = sum(val_reg_loss) / len(val_reg_loss)
        print("val concept loss:", avg_val_concept_loss, "val word loss:", avg_val_word_loss, "val reg loss: ", avg_val_reg_loss)

        avg_val_loss = avg_val_concept_loss + avg_val_word_loss
        if avg_val_loss < best_loss:
            print("save model")
            best_loss = avg_val_loss
            preLM.save_pretrained(CBL_PATH + "epoch_" + str(e + 1))
            torch.save(cbl.state_dict(), CBL_PATH + "epoch_" + str(e + 1) + ".pt")
    
    end = time.time()
    print("time of training CBM:", (end - start) / 3600, "hours")