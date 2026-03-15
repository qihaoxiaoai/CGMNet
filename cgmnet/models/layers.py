# cgmnet/models/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GINConv
from dgl.nn.functional import edge_softmax

from cgmnet.models.utils import MLP

class GINLayer(nn.Module):
    def __init__(self, in_feats: int, d_model: int, feat_drop: float = 0.0):
        super().__init__()
        self.mlp = MLP(in_dim=in_feats, out_dim=d_model, num_layers=2)
        self.feat_drop = nn.Dropout(feat_drop)
        self.graph_conv = GINConv(apply_func=self.mlp, aggregator_type='sum')

    def forward(self, g, h):
        return self.graph_conv(g, self.feat_drop(h))

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, feat_drop: float, attn_drop: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_linear = nn.Linear(d_model, d_model * 3, bias=True)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.feat_drop = nn.Dropout(feat_drop)
        
        self.pre_attn_layer_norm = nn.LayerNorm(d_model)
        self.pre_ffn_layer_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(feat_drop),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, g, h):
        h_norm = self.pre_attn_layer_norm(h)
        
        qkv = self.qkv_linear(h_norm).view(-1, 3, self.n_heads, self.d_head)
        q = qkv[:, 0, ...].permute(1, 0, 2)
        k = qkv[:, 1, ...].permute(1, 0, 2)
        v = qkv[:, 2, ...].permute(1, 0, 2)

        with g.local_scope():
            g.ndata['q'] = q.permute(1, 0, 2)
            g.ndata['k'] = k.permute(1, 0, 2)
            g.ndata['v'] = v.permute(1, 0, 2)

            g.apply_edges(fn.v_dot_u('q', 'k', 'score'))
            g.edata['score'] = g.edata['score'] / (self.d_head ** 0.5)
            
            if 'distance' in g.edata:
                g.edata['score'] = g.edata['score'] + g.edata['distance']
            
            if 'path_bias' in g.edata:
                g.edata['score'] = g.edata['score'] + g.edata['path_bias']

            g.edata['attn'] = edge_softmax(g, g.edata['score'])
            g.edata['attn'] = self.attn_drop(g.edata['attn'])
            
            g.update_all(fn.u_mul_e('v', 'attn', 'msg'), fn.sum('msg', 'h_attn'))
            h_attn = g.ndata['h_attn'].reshape(-1, self.d_model)

        h = h + self.feat_drop(self.out_linear(h_attn))
        
        h_norm = self.pre_ffn_layer_norm(h)
        h_ffn = self.ffn(h_norm)
        h = h + self.feat_drop(h_ffn)
        
        return h

class PathAttentionBias(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.path_transform = nn.Linear(d_model, 1)

    def forward(self, paths: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        path_mask = (paths != -1).float()
        valid_paths = paths.clone().masked_fill_(paths == -1, 0)
        
        path_features = F.embedding(valid_paths, node_features)
        path_features = path_features * path_mask.unsqueeze(-1)
        
        num_path_nodes = path_mask.sum(dim=1, keepdim=True).clamp(min=1)
        aggregated_path_features = path_features.sum(dim=1) / num_path_nodes
        
        path_bias = self.path_transform(aggregated_path_features)
        return path_bias
