"""
EE136IU Capstone Mini-Project - Topic 1: Comprehensive Model Benchmark
Standalone PyTorch Architecture & Inference Loader for Tier 2 (3-Layer MLP)
Author: Hai Nguyen (EEEEIU22093)
"""

import os
import pickle
import torch
import torch.nn as nn


class ModulationMLP(nn.Module):
    """
    3-Layer Multilayer Perceptron (MLP) for Automatic Modulation Classification
    Input: 18 physical domain features (Cumulants + Spectral statistics)
    Hidden 1: Linear(18 -> 128) + BatchNorm1d + ReLU + Dropout(0.2)
    Hidden 2: Linear(128 -> 64) + BatchNorm1d + ReLU + Dropout(0.2)
    Output: Linear(64 -> 8) logits for 8 modulation classes
    Total Trainable Parameters: 11,592
    """
    def __init__(self, input_dim: int = 18, num_classes: int = 8):
        super(ModulationMLP, self).__init__()
        self.network = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Hidden Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output Layer
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Alias for backward compatibility so both `ModulationMLP` and `MLP` work seamlessly
MLP = ModulationMLP


def load_pretrained_mlp(base_dir: str = None, device: str = "cpu"):
    """
    Helper to load the trained MLP model, fitted StandardScaler, and class mapping
    in a single call without running any training code.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, "class_mapping.pkl"), "rb") as f:
        mapping = pickle.load(f)

    with open(os.path.join(base_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    feature_cols = mapping["features"]
    idx_to_class = mapping["idx_to_class"]
    class_to_idx = mapping["class_to_idx"]
    classes = sorted(class_to_idx.keys())

    model = ModulationMLP(input_dim=len(feature_cols), num_classes=len(classes))
    state_dict = torch.load(os.path.join(base_dir, "best_mlp.pth"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, scaler, mapping
