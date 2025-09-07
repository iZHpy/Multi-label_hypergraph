import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexDecoder(nn.Module):
    def __init__(self, feature_dim, num_labels, hidden_dim=512):
        super(ComplexDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.num_labels = num_labels

        # Attention mechanism
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)

        # Label correlation learning
        self.label_correlation = nn.Parameter(torch.randn(num_labels, num_labels))

        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, sample_feature, label_feature):
        batch_size = sample_feature.shape[0]

        # Attention mechanism
        query = self.query_proj(sample_feature)  # [batch_size, 1, hidden_dim]
        key = self.key_proj(label_feature).unsqueeze(0)  # [1, num_labels, hidden_dim]
        value = self.value_proj(label_feature).unsqueeze(0)  # [1, num_labels, hidden_dim]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_probs, value)  # [batch_size, 1, hidden_dim]

        # Combine sample feature with context vector
        combined_feature = torch.cat([sample_feature, context_vector],
                                     dim=-1)  # [batch_size, 1, hidden_dim*2]

        # Final prediction
        hidden = F.relu(self.fc1(combined_feature))
        logits = self.fc2(hidden).squeeze(-1)  # [batch_size, num_labels]

        # Apply label correlation
        corr_logits = torch.matmul(logits, self.label_correlation)
        final_logits = logits + corr_logits

        return final_logits



class SimpleDecoder(nn.Module):
    def __init__(self, feature_dim, num_labels, use_mlp=True):
        super(SimpleDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        self.use_mlp = use_mlp

        if use_mlp:
            self.sample_mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )
            self.label_mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )

    def forward(self, sample_feature, label_feature):
        if self.use_mlp:
            sample_residual = sample_feature
            label_residual = label_feature

            sample_feature = self.sample_mlp(sample_feature) + sample_residual
            label_feature = self.label_mlp(label_feature) + label_residual

        logits = torch.matmul(sample_feature, label_feature.t())
        return logits



