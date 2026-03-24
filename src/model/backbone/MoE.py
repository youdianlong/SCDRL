import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FAConv, global_add_pool


class FAGCNExpertLite(nn.Module):
    def __init__(self, num_features, hidden, num_conv_layers, dropout, epsilon):
        super().__init__()
        self.dropout = dropout
        self.global_pool = global_add_pool

        self.input_proj = nn.Linear(num_features, hidden)
        self.convs = nn.ModuleList(
            [FAConv(hidden, epsilon, dropout) for _ in range(num_conv_layers)]
        )
        self.output_proj = nn.Linear(hidden, hidden)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, data):
        x = data.x if data.x is not None else data.feat
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        device = self.input_proj.weight.device
        x = x.to(device)
        edge_index = edge_index.to(device)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        else:
            batch = batch.to(device)

        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.input_proj(h))

        h0 = h
        for conv in self.convs:
            h = conv(h, h0, edge_index)

        h = self.output_proj(h)
        graph_emb = self.global_pool(h, batch)

        return graph_emb, h


class SimpleGate(nn.Module):
    def __init__(self, in_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_experts)

    def forward(self, graph_features):
        # graph_features: [B, num_experts, hidden]
        mean_feature = graph_features.mean(dim=1)   # [B, hidden]
        logits = self.fc(mean_feature)              # [B, num_experts]
        weights = F.softmax(logits, dim=-1)
        return weights


class FAGCNMoELite(nn.Module):
    def __init__(
        self,
        num_features,
        hidden,
        num_conv_layers,
        dropout,
        epsilon,
        num_experts=3,
    ):
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            FAGCNExpertLite(
                num_features=num_features,
                hidden=hidden,
                num_conv_layers=num_conv_layers,
                dropout=dropout,
                epsilon=epsilon,
            )
            for _ in range(num_experts)
        ])

        self.gate = SimpleGate(hidden, num_experts)

    def forward(self, data):
        graph_list = []
        node_list = []

        for expert in self.experts:
            g_emb, n_emb = expert(data)
            graph_list.append(g_emb.unsqueeze(1))   # [B, 1, H]
            node_list.append(n_emb.unsqueeze(1))    # [N, 1, H]

        expert_graphs = torch.cat(graph_list, dim=1)   # [B, E, H]
        expert_nodes = torch.cat(node_list, dim=1)     # [N, E, H]

        weights = self.gate(expert_graphs)             # [B, E]

        fused_graph = torch.sum(expert_graphs * weights.unsqueeze(-1), dim=1)  # [B, H]

        fused_nodes = expert_nodes.mean(dim=1)  # [N, H]

        return fused_graph, fused_nodes, weights


def get_model(
    num_features,
    hid_dim=128,
    num_conv_layers=2,
    dropout=0.2,
    epsilon=0.1,
    num_experts=3,
):
    return FAGCNMoELite(
        num_features=num_features,
        hidden=hid_dim,
        num_conv_layers=num_conv_layers,
        dropout=dropout,
        epsilon=epsilon,
        num_experts=num_experts,
    )