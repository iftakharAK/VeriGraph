

import torch
from torch import nn

try:
    import dgl
    from dgl.nn import GATConv
    DGL_AVAILABLE = True
except Exception:
    DGL_AVAILABLE = False

class GNNReasoner(nn.Module):
    """
    Graph Attention Network over statement nodes.
    If DGL is unavailable, falls back to identity mapping.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, heads: int = 2, out_dim: int = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim
        self.is_noop = not DGL_AVAILABLE
        if not self.is_noop:
            self.gat1 = GATConv(in_dim, hidden_dim, num_heads=heads, feat_drop=0.1, attn_drop=0.1)
            self.gat2 = GATConv(hidden_dim*heads, self.out_dim, num_heads=1, feat_drop=0.1, attn_drop=0.1)

    def forward(self, node_feats, edges, device="cuda"):
        """
        node_feats: (N, H) tensor
        edges: list of (u,v) tuples
        """
        if self.is_noop:
            return node_feats  # passthrough

        N = node_feats.size(0)
        g = dgl.graph(edges, num_nodes=N, device=device)
        h = self.gat1(g, node_feats)
        h = torch.relu(h.flatten(1))
        h = self.gat2(g, h).squeeze(1)  # (N, out_dim)
        return h