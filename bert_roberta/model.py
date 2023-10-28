import torch
import torch.nn as nn


class ClassificationHead(nn.Module):  # for roberta
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, classifier_dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:,0, :]  # take <s> token (equiv. to [CLS]). 原来：[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
