import torch
from torch import nn


class PrototypicalPart(nn.Module):
    def __init__(
            self,
            n_features: int,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hard_parts = False
        self.weights = nn.Parameter(torch.randn(n_features), requires_grad=True)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        if self.hard_parts:
            return x * (self.weights > 9).float()
        return x * self.weights
