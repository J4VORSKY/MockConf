import os
import json
import argparse
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from collections import defaultdict
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

import tools
from constants import DEVSET_FILE_NAMES, TESTSET_FILE_NAMES, LABEL_TO_ID, ID_TO_LABEL

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For deterministic DataLoader shuffling
    torch.use_deterministic_algorithms(True, warn_only=True)

# ======= Dataset =======

class ClassificationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ======= Model =======

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


# ======= Feature Engineering =======

def feature_engineering(features: np.ndarray) -> np.ndarray:
    engineered = []
    for feature in features:
        similarity_score, src_length, tgt_length = feature
        length_ratio = src_length / tgt_length if tgt_length != 0 else 0
        length_difference = src_length - tgt_length

        engineered.append([
            similarity_score,
            src_length,
            tgt_length,
            length_ratio,
            length_difference,
            similarity_score,  # normalized similarity again
        ])
    return np.array(engineered)


# ======= Data Loading =======

def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        features, labels = [], []
        for label, examples in data.items():
            for sim_score, src, tgt in examples:
                features.append([sim_score, len(src), len(tgt)])
                labels.append(LABEL_TO_ID[label])
        return np.array(features), np.array(labels)


# ======= Training =======

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


# ======= Evaluation =======

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    accuracy = val_correct / val_total
    return val_loss, accuracy

def calculate_similarity_scores_for_labels(alignments: dict, model, labels_path: str):
    labse_model = SentenceTransformer("LaBSE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for alignment_name, alignment in alignments.items():
        print(f"Processing {alignment_name}")
        src_tokens = alignment["textA"].split()
        tgt_tokens = alignment["textB"].split()

        for pair in alignment["alignedPairs"]:
            if pair.get("parent") is not None:
                continue

            source_indices = pair["pair"][0]
            target_indices = pair["pair"][1]

            if not source_indices or not target_indices:
                continue

            source_text = " ".join(
                [src_tokens[i] for i in source_indices if i < len(src_tokens)]
            )
            target_text = " ".join(
                [tgt_tokens[i] for i in target_indices if i < len(tgt_tokens)]
            )

            embeddings = labse_model.encode([source_text, target_text])
            cos_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))

            features = feature_engineering([[cos_sim, len(source_indices), len(target_indices)]])
            input_tensor = torch.tensor(features, dtype=torch.float32).to(device)

            with torch.no_grad():
                outputs = model(input_tensor[0])
                _, predicted_label = torch.max(outputs, dim=0)

                pair["label"] = ID_TO_LABEL[int(predicted_label)]
                pair["labelIndex"] = int(predicted_label)

        os.makedirs(labels_path, exist_ok=True)
        with open(
            f"{labels_path}/{alignment_name}.json", "w"
        ) as f:
            json.dump(alignment, f, indent=4, ensure_ascii=False)

# ======= Main =======

def main(args):
    set_seed(args.seed)

    # Load and engineer features
    raw_features, labels = load_data(args.data_path)
    features = feature_engineering(raw_features)
    dataset = ClassificationDataset(features, labels)

    # Split into training and validation sets
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model, loss, optimizer
    input_size = features.shape[1]
    model = SimpleClassifier(input_size, args.hidden_size, len(LABEL_TO_ID))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)

    # ======= Inference on BERTAlign alignments =======

    print("Loading BERTAlign subsegments data...")
    bertalign_subsegments_alignments = tools.read_alignment_files(args.bertalign_subsegments_path)

    model.eval()
    
    calculate_similarity_scores_for_labels(bertalign_subsegments_alignments, model, args.bertalign_subsegments_labels_path)

# ======= Argparse =======

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple classifier on similarity features")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data_path", type=str, default="./analysis/similarity_scores.json")
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--bertalign_subsegments_path", type=str, default="./outputs/bertalign-sentence-alignment-subsegments/")
    parser.add_argument("--bertalign_subsegments_labels_path", type=str, default="./outputs/bertalign-sentence-alignment-subsegments-labels-200/")

    args = parser.parse_args()
    main(args)