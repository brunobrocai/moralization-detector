import torch
from torch import nn


class FakeBERTOutput:
    """Fake BERT output class for testing purposes."""

    def __init__(self, logits):
        self.logits = logits


class FakeBERTModel(nn.Module):
    """Fake BERT model for testing purposes."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        # Generate fake logits, e.g., randomly or fixed for testing purposes
        attention_mask = attention_mask + 1
        batch_size = input_ids.size(0)
        logits = torch.randn(batch_size, self.num_classes)
        return FakeBERTOutput(logits)
