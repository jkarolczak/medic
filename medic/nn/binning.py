import torch
from torch import nn as nn
from torch.nn import functional as f

from medic.nn import EPS


class OneHotBinning(nn.Module):
    def __init__(
            self,
            name: str,
            n_values: int,
    ) -> None:
        super().__init__()
        self.name = name
        self.n_values = n_values

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        try:
            return f.one_hot(x.long(), num_classes=self.n_values).float()
        except RuntimeError as re:
            raise RuntimeError(f"Error in {self.name}: {re}") from re


class FuzzyBinning(nn.Module):
    def __init__(
            self,
            name: str,
            n_bins: int = 5,
            sigma: float = 0.2,
            hard: bool = False
    ) -> None:
        super().__init__()
        self.name = name
        self._hard = hard
        self.num_bins = n_bins
        self.bin_centers = nn.Parameter(torch.normal(mean=torch.zeros(n_bins), std=torch.ones(n_bins)),
                                        requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True)

    @property
    def hard(self) -> bool:
        return self._hard

    @hard.setter
    def hard(self, value: bool) -> None:
        self._hard = value
        self.bin_centers.requires_grad = not value
        self.sigma.requires_grad = not value

    def hard_binning(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        distances = (x.unsqueeze(-1) - self.bin_centers) ** 2
        hard_bins = torch.argmin(distances, dim=-1)
        return f.one_hot(hard_bins, num_classes=self.num_bins).float()

    def soft_binning(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        self.bin_centers.data = torch.sort(self.bin_centers)[0]

        x = x.unsqueeze(-1)
        distances = (x - self.bin_centers) ** 2
        membership = torch.exp(-distances / (2 * self.sigma ** 2 + EPS))
        membership = f.softmax(membership, dim=-1) + EPS
        return membership

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        try:
            if self._hard:
                x = self.hard_binning(x)
            else:
                x = self.soft_binning(x)
            return x
        except RuntimeError as re:
            raise RuntimeError(f"Error in {self.name}") from re
