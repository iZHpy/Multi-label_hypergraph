import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_max
from collections import defaultdict


class WeightedHypergraphLayer(nn.Module):
    def __init__(self, feature_dim, aggregation_type='mean', node_update='simple', num_heads=4, dropout_rate=0.1):
        super(WeightedHypergraphLayer, self).__init__()
        self.feature_dim = feature_dim
        self.aggregation_type = aggregation_type
        self.node_update = node_update
        self.num_heads = num_heads

        self.mlp1 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim)
        )

        if node_update == 'simple':
            self.mlp3 = nn.Sequential(
                nn.Linear(feature_dim * 2 + 1, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, feature_dim)
            )
        elif node_update == 'gated':
            self.mlp3 = nn.Sequential(
                nn.Linear(feature_dim * 3 + 1, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, feature_dim)
            )
            self.gate = nn.Linear(feature_dim * 2, feature_dim)
        else:
            raise ValueError(f"Unknown node update type: {node_update}")

        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)

        if aggregation_type in ['simple_attention', 'soft_attention']:
            self.attention = nn.Linear(feature_dim, 1)

    def forward(self, node_features, edge_features, edge_index, edge_weight):
        num_nodes = node_features.size(0)
        num_edges = edge_features.size(0)

        # Equation (6): Update hyperedge embeddings
        node_to_edge_messages = self.mlp1(node_features)[edge_index[0]]

        if self.aggregation_type == 'sum':
            q = scatter_sum(node_to_edge_messages, edge_index[1], dim=0, dim_size=num_edges)
        elif self.aggregation_type == 'mean':
            q = scatter_mean(node_to_edge_messages, edge_index[1], dim=0, dim_size=num_edges)
        elif self.aggregation_type == 'max':
            q = scatter_max(node_to_edge_messages, edge_index[1], dim=0, dim_size=num_edges)[0]
        elif self.aggregation_type in ['simple_attention', 'soft_attention']:
            attention_weights = F.softmax(self.attention(node_to_edge_messages), dim=0)
            weighted_sum = scatter_sum(node_to_edge_messages * attention_weights, edge_index[1], dim=0,
                                       dim_size=num_edges)
            if self.aggregation_type == 'soft_attention':
                total_weight = scatter_sum(attention_weights, edge_index[1], dim=0, dim_size=num_edges)
                q = weighted_sum / (total_weight + 1e-8)
            else:
                q = weighted_sum
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")

        edge_features = self.layer_norm1(edge_features + q)  # Add layer normalization

        # Equation (7): Aggregate messages from hyperedges to nodes, with weights
        edge_to_node_messages = self.mlp2(
            torch.cat([node_features[edge_index[0]], edge_features[edge_index[1]]], dim=-1))

        edge_weight_expanded = edge_weight[edge_index[1]]
        weighted_messages = edge_to_node_messages * edge_weight_expanded.unsqueeze(1)
        p_tilde = scatter_sum(weighted_messages, edge_index[0], dim=0, dim_size=num_nodes)

        edge_count = scatter_sum(edge_weight_expanded, edge_index[0], dim=0, dim_size=num_nodes)

        # Equation (8): Update node embeddings
        if self.node_update == 'simple':
            node_features_update = self.mlp3(torch.cat([
                node_features,
                p_tilde,
                edge_count.unsqueeze(1)
            ], dim=-1))
        elif self.node_update == 'gated':
            gate_input = torch.cat([node_features, p_tilde], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            node_features_update = self.mlp3(torch.cat([
                node_features,
                p_tilde,
                gate * p_tilde,  # 加入门控机制
                edge_count.unsqueeze(1)
            ], dim=-1))
        else:
            raise ValueError(f"Unknown node_update type: {self.node_update}")

        node_features = self.layer_norm2(
            node_features + node_features_update)  # Add residual connection and layer normalization

        return node_features, edge_features


class WeightedHypergraphModel(nn.Module):
    def __init__(self, num_labels, feature_dim, num_layers, feature_aggregate='mean', node2hyperedge_aggregate='mean', node_update='simple', num_heads=4, dropout_rate=0.1):
        super(WeightedHypergraphModel, self).__init__()
        self.label_embedding = nn.Embedding(num_labels, feature_dim)
        self.layers = nn.ModuleList([WeightedHypergraphLayer(feature_dim, aggregation_type=node2hyperedge_aggregate, node_update=node_update,
                                        num_heads=num_heads,dropout_rate=dropout_rate) for _ in range(num_layers)])
        self.final_node_projection = nn.Linear(feature_dim, feature_dim)
        self.final_edge_projection = nn.Linear(feature_dim, feature_dim)

        self.feature_aggregate = feature_aggregate
        if feature_aggregate == 'simple_attention':
            self.simple_attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, 1, bias=False)
            )
        elif feature_aggregate == 'self_attention':
            self.self_attention = MultiHeadAttention(feature_dim, num_heads)

        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, hypergraph, batch_features, start_index, end_index):
        device = batch_features.device
        batch_size = end_index - start_index

        node_features = self.label_embedding(hypergraph.node_index.to(device))
        edge_features = torch.zeros(len(hypergraph.id_to_edge), node_features.size(1), device=device)

        hyperedge_features = defaultdict(list)
        for i, sample_id in enumerate(range(start_index, end_index)):
            edge_id = hypergraph.get_hyperedge_id(sample_id)
            hyperedge_features[edge_id].append(batch_features[i])

        for edge_id, features in hyperedge_features.items():
            if len(features) == 1:
                edge_features[edge_id] = features[0]
            else:
                stacked_features = torch.stack(features)
                if self.feature_aggregate == 'max':
                    edge_features[edge_id] = torch.max(stacked_features, dim=0)[0]
                elif self.feature_aggregate == 'mean':
                    edge_features[edge_id] = torch.mean(stacked_features, dim=0)
                elif self.feature_aggregate == 'simple_attention':
                    attention_weights = self.simple_attention(stacked_features).squeeze(-1)
                    attention_weights = F.softmax(attention_weights, dim=0)
                    edge_features[edge_id] = (stacked_features * attention_weights.unsqueeze(-1)).sum(dim=0)
                elif self.feature_aggregate == 'self_attention':
                    edge_features[edge_id] = self.self_attention(stacked_features.unsqueeze(0)).squeeze(0)
                else:
                    raise ValueError(f"Unknown feature aggregation method: {self.feature_aggregate}")

        edge_features = edge_features
        edge_index = hypergraph.edge_index.to(device)
        edge_weight = hypergraph.edge_weight.to(device)

        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, edge_index, edge_weight)

        node_features = self.layer_norm(self.final_node_projection(node_features))
        edge_features = self.layer_norm(self.final_edge_projection(edge_features))

        return node_features, edge_features


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert feature_dim % num_heads == 0, "Feature dimension must be divisible by number of heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.output = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        query = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.feature_dim)
        output = self.output(context)

        return output.mean(dim=1)