# cgmnet/models/knowledge.py
"""
Defines modules for integrating knowledge features (fingerprints and descriptors)
into the CGMNet model.
"""
import torch
import torch.nn as nn
from typing import List, Dict

# ============================ FIX STARTS HERE ============================
# FINGERPRINT_DIMENSIONS is defined in fingerprint.py, not chem_utils.py.
# We correct the import path here.
from cgmnet.utils.fingerprint import FINGERPRINT_DIMENSIONS
# ============================= FIX ENDS HERE =============================


class KnowledgeFusion(nn.Module):
    """
    Fuses multiple knowledge features (knodes) into a single representation.
    """
    def __init__(self, knode_names: List[str], d_model: int, feat_drop: float):
        super().__init__()
        self.knode_names = knode_names
        self.d_model = d_model

        # Create a projection layer for each knowledge type
        self.knode_projects = nn.ModuleDict()
        for name in self.knode_names:
            in_dim = FINGERPRINT_DIMENSIONS.get(name)
            if in_dim is None:
                raise ValueError(f"Fingerprint '{name}' is not defined in FINGERPRINT_DIMENSIONS.")
            self.knode_projects[name] = nn.Sequential(
                nn.Linear(in_dim, self.d_model),
                nn.ReLU(),
                nn.Dropout(feat_drop)
            )
        
        # A final layer to combine the projected features
        self.fusion_layer = nn.Linear(len(self.knode_names) * self.d_model, self.d_model)

    def forward(self, knodes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            knodes: A dictionary where keys are knode names (e.g., 'ecfp') and
                    values are the corresponding feature tensors of shape (N, dim).

        Returns:
            A fused knowledge tensor of shape (N, d_model).
        """
        projected_knodes = []
        for name in self.knode_names:
            if name in knodes:
                projected = self.knode_projects[name](knodes[name])
                projected_knodes.append(projected)

        if not projected_knodes:
            return None
        
        # Concatenate all projected features
        concatenated_features = torch.cat(projected_knodes, dim=-1)
        
        # Fuse them into a single vector
        fused_representation = self.fusion_layer(concatenated_features)
        return fused_representation


class KnowledgePooling(nn.Module):
    """
    A simple pooling layer to get a graph-level representation from node features.
    """
    def __init__(self, d_model: int, n_tasks: int, dropout: float):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.predictor = nn.Linear(d_model, n_tasks)

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (total_nodes, d_model).
            batch_idx: A tensor indicating which graph each node belongs to,
                       shape (total_nodes,).

        Returns:
            Graph-level predictions of shape (num_graphs, n_tasks).
        """
        # Average node features for each graph in the batch
        num_graphs = batch_idx.max().item() + 1
        graph_reps = torch.stack(
            [x[batch_idx == i].mean(dim=0) for i in range(num_graphs)]
        )
        
        pooled_reps = self.pool(graph_reps)
        return self.predictor(pooled_reps)
