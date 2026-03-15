# configs/cgmnet_config.py
"""
Central configuration file for the CGMNet model and training settings.
"""
import argparse

# --- Default Hyperparameters for the "base" CGMNet model ---
CGMNET_BASE_CONFIG = {
    # Model Architecture
    "d_model": 768,
    "n_mol_layers": 12,
    "n_subg_layers": 2,
    "n_heads": 12,
    "n_ffn_dense_layers": 2,
    "path_max_length": 2,
    "vq": False,

    # Input/Output Dimensions
    "in_feats": 137,
    "edge_feats": 14,

    # Regularization
    "feat_drop": 0.1,
    "attn_drop": 0.1,

    # Pre-training Specific
    "mask_rate": 0.3,
    "no_hier_loss": False,
    
    # Fine-tuning Specific
    "finetune_dropout": 0.1,
    "ema_decay": 0.99,
}

def add_model_and_training_args(parser: argparse.ArgumentParser):
    """
    Adds CGMNet model and training arguments to a parser.
    """
    # --- Model Architecture Arguments ---
    group = parser.add_argument_group("Model Architecture")
    group.add_argument('--d_model', type=int, default=CGMNET_BASE_CONFIG["d_model"])
    group.add_argument('--n_mol_layers', type=int, default=CGMNET_BASE_CONFIG["n_mol_layers"])
    group.add_argument('--n_subg_layers', type=int, default=CGMNET_BASE_CONFIG["n_subg_layers"])
    group.add_argument('--n_heads', type=int, default=CGMNET_BASE_CONFIG["n_heads"])
    group.add_argument('--n_ffn_dense_layers', type=int, default=CGMNET_BASE_CONFIG["n_ffn_dense_layers"])
    group.add_argument('--path_max_length', type=int, default=CGMNET_BASE_CONFIG["path_max_length"])
    group.add_argument('--vq', action='store_true', default=CGMNET_BASE_CONFIG["vq"])
    
    # ============================ FIX STARTS HERE ============================
    # The --in_feats and --edge_feats arguments were missing.
    group = parser.add_argument_group("Input/Output Dimensions")
    group.add_argument('--in_feats', type=int, default=CGMNET_BASE_CONFIG["in_feats"],
                       help='Dimension of input atom features.')
    group.add_argument('--edge_feats', type=int, default=CGMNET_BASE_CONFIG["edge_feats"],
                       help='Dimension of input bond features.')
    # ============================= FIX ENDS HERE =============================
    
    # --- Regularization Arguments ---
    group = parser.add_argument_group("Regularization")
    group.add_argument('--feat_drop', type=float, default=CGMNET_BASE_CONFIG["feat_drop"])
    group.add_argument('--attn_drop', type=float, default=CGMNET_BASE_CONFIG["attn_drop"])

    # --- Pre-training Arguments ---
    group = parser.add_argument_group("Pre-training")
    group.add_argument("--mask_rate", type=float, default=CGMNET_BASE_CONFIG["mask_rate"])
    group.add_argument('--no_hier_loss', action='store_true', default=CGMNET_BASE_CONFIG["no_hier_loss"])

    # --- Fine-tuning Arguments ---
    group = parser.add_argument_group("Fine-tuning")
    group.add_argument("--finetune_dropout", type=float, default=CGMNET_BASE_CONFIG["finetune_dropout"])
    group.add_argument("--ema_decay", type=float, default=CGMNET_BASE_CONFIG["ema_decay"])

    return parser
