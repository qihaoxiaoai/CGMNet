# cgmnet/models/utils.py
"""
Utility classes and functions for model components.
"""
import torch
import torch.nn as nn
from copy import deepcopy
import torch_scatter

# Kept your original MLP helper class
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, use_layer_norm: bool = False):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_dim if i == 0 else out_dim, out_dim))
            if use_layer_norm: self.layers.append(nn.LayerNorm(out_dim))
            self.layers.append(nn.ReLU())
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers: x = layer(x)
        return x

# Optimized your get_subgraph_readout function to be more concise
def get_subgraph_readout(
    atom_features: torch.Tensor,
    node_ids: torch.Tensor,
    macro_node_ids: torch.Tensor
) -> torch.Tensor:
    """
    Aggregates node features to get subgraph (fragment) features
    by averaging the features of atoms within each fragment.
    """
    # Use torch_scatter with reduce='mean' for a direct and efficient mean readout
    return torch_scatter.scatter(
        atom_features[node_ids],
        macro_node_ids,
        dim=0,
        reduce='mean'
    )

# Replaced with the complete and correct implementation of ModelWithEMA
class ModelWithEMA(nn.Module):
    """
    A wrapper class for a model that maintains an Exponential Moving Average
    of the model's weights for more stable evaluation.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay
        
        # Create the EMA model, initialized with the same weights
        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False # EMA model is not trained directly
        self.ema_model.eval()

    def _update(self, model_online, model_ema):
        """Performs a single EMA update step for parameters and buffers."""
        with torch.no_grad():
            # Update trainable parameters
            for ema_param, online_param in zip(model_ema.parameters(), model_online.parameters()):
                ema_param.data.mul_(self.decay).add_(online_param.data, alpha=1 - self.decay)
            # Update non-trainable buffers (e.g., for batch norm)
            for ema_buffer, online_buffer in zip(model_ema.buffers(), model_online.buffers()):
                ema_buffer.copy_(online_buffer)

    def update_ema(self):
        """Public method to be called after each training step."""
        self._update(self.model, self.ema_model)

    def forward(self, *args, **kwargs):
        """
        This is the crucial missing 'forward' method.
        It passes the call to the internal online model, which is being trained.
        """
        return self.model(*args, **kwargs)

# Kept your original model_n_params helper function
def model_n_params(model: nn.Module) -> int:
    """Calculates the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
