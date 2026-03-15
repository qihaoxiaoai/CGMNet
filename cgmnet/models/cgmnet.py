# cgmnet/models/cgmnet.py
import torch
import torch.nn as nn
import dgl

from cgmnet.models.knowledge import KnowledgeFusion
from cgmnet.models.layers import GINLayer, GraphTransformerLayer, PathAttentionBias
from cgmnet.utils.register import register_model
from cgmnet.models.utils import get_subgraph_readout


def debug_tensor(tensor: torch.Tensor, name: str):
    if not torch.isfinite(tensor).all():
        print(f"\n/!\\ DEBUG: Instability found in tensor '{name}'! /!\\")
        print(f"    Contains NaN: {torch.isnan(tensor).any()}")
        print(f"    Contains Inf: {torch.isinf(tensor).any()}\n")


@register_model("cgmnet")
class CGMNet(nn.Module):
    def __init__(self, args, n_tasks: int):
        super().__init__()
        self.args = args
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.n_tasks = n_tasks

        self.subgraph_encoder = nn.ModuleList(
            [
                GINLayer(d_model=self.d_model, in_feats=args.in_feats, feat_drop=args.feat_drop)
                if i == 0
                else GINLayer(d_model=self.d_model, in_feats=self.d_model, feat_drop=args.feat_drop)
                for i in range(args.n_subg_layers)
            ]
        )

        if args.knodes:
            self.knode_fusion = KnowledgeFusion(knode_names=args.knodes, d_model=self.d_model, feat_drop=args.feat_drop)
            self.knowledge_gru = nn.GRUCell(self.d_model, self.d_model)
        else:
            self.knode_fusion = None

        self.mol_transformer = nn.ModuleList(
            [
                GraphTransformerLayer(
                    d_model=self.d_model,
                    n_heads=args.n_heads,
                    feat_drop=args.feat_drop,
                    attn_drop=args.attn_drop,
                )
                for _ in range(args.n_mol_layers)
            ]
        )

        self.distance_embedding = nn.Embedding(128, self.n_heads)
        self.path_bias_encoder = PathAttentionBias(self.d_model)

        self.pretrain_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.n_tasks),
        )
        self.predictor = None

    def init_ft_predictor(self, n_tasks, dropout):
        self.predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, n_tasks),
        )
        device = next(self.parameters()).device
        self.predictor.to(device)

    # =========================
    # Frozen encoder API
    # =========================
    @torch.no_grad()
    def encode(self, data: dict, return_frag: bool = False):
        """
        Extract graph-level embedding (and optionally fragment embeddings)
        without requiring predictor initialization.

        This method does NOT change any existing behavior of forward(), pretrain, or finetune.

        Args:
            data: batched data dict (same as forward)
            return_frag: if True, return (graph_rep, h_frag)

        Returns:
            graph_rep: Tensor [B, d_model]
            optionally h_frag: Tensor [sum(num_frags), d_model]
        """
        was_training = self.training
        self.eval()

        atom_g, frag_g = data["atom_graph"], data["fragment_graph"]

        # Avoid persistent side-effects on graph objects (edata/ndata pollution)
        with frag_g.local_scope():
            h_subg = atom_g.ndata["feat"]
            for layer in self.subgraph_encoder:
                h_subg = layer(atom_g, h_subg)

            h_frag = get_subgraph_readout(h_subg, data["node_ids"], data["macro_node_ids"])

            if self.knode_fusion and data.get("knodes") and data["knodes"]:
                h_knode = self.knode_fusion(data["knodes"])
                num_frags_per_mol = frag_g.batch_num_nodes()
                h_knode_expanded = h_knode.repeat_interleave(num_frags_per_mol, dim=0)
                h_frag = self.knowledge_gru(h_knode_expanded, h_frag)

            if "path" in frag_g.edata and "paths_matrix" in frag_g.edata:
                distances = frag_g.edata["path"].clamp(min=0, max=127)
                dist_emb = self.distance_embedding(distances)
                frag_g.edata["distance"] = dist_emb.unsqueeze(-1)

                paths = frag_g.edata["paths_matrix"]
                path_bias = self.path_bias_encoder(paths, h_frag)
                frag_g.edata["path_bias"] = path_bias.unsqueeze(1)

            for layer in self.mol_transformer:
                h_frag = layer(frag_g, h_frag)

            frag_g.ndata["h"] = h_frag
            graph_rep = dgl.mean_nodes(frag_g, "h")

        if was_training:
            self.train()

        if return_frag:
            return graph_rep, h_frag
        return graph_rep

    def forward(self, data: dict, explain: bool = False):
        atom_g, frag_g = data["atom_graph"], data["fragment_graph"]

        # Avoid persistent side-effects on graph objects (edata/ndata pollution)
        with frag_g.local_scope():
            h_subg = atom_g.ndata["feat"]

            for layer in self.subgraph_encoder:
                h_subg = layer(atom_g, h_subg)

            h_frag = get_subgraph_readout(h_subg, data["node_ids"], data["macro_node_ids"])

            if self.knode_fusion and data.get("knodes") and data["knodes"]:
                h_knode = self.knode_fusion(data["knodes"])
                num_frags_per_mol = frag_g.batch_num_nodes()
                h_knode_expanded = h_knode.repeat_interleave(num_frags_per_mol, dim=0)
                h_frag = self.knowledge_gru(h_knode_expanded, h_frag)

            if "path" in frag_g.edata and "paths_matrix" in frag_g.edata:
                distances = frag_g.edata["path"].clamp(min=0, max=127)
                dist_emb = self.distance_embedding(distances)
                frag_g.edata["distance"] = dist_emb.unsqueeze(-1)

                paths = frag_g.edata["paths_matrix"]
                path_bias = self.path_bias_encoder(paths, h_frag)
                frag_g.edata["path_bias"] = path_bias.unsqueeze(1)

            for layer in self.mol_transformer:
                h_frag = layer(frag_g, h_frag)

            if self.training and self.predictor is None:
                masked_indices = data["masked_indices"]
                preds = self.pretrain_predictor(h_frag[masked_indices])
                return preds

            frag_g.ndata["h"] = h_frag
            graph_rep = dgl.mean_nodes(frag_g, "h")

            if self.predictor is None:
                raise ValueError("Predictor for fine-tuning has not been initialized. Call `init_ft_predictor` first.")

            preds = self.predictor(graph_rep)

            if explain:
                return preds, h_frag
            return preds

