import torch
from torch import nn

from proto_intervals import TDefinitionList
from proto_intervals.nn.binning import (FuzzyBinning,
                                        OneHotBinning)


class MLPClassifier(nn.Module):
    def __init__(
            self,
            definitions: TDefinitionList,
            n_classes: int,
            hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self._hard_binning = False
        embedding_size = sum([item["n_bins"] if item["binning"] else item["n_values"] for item in definitions])
        self.binning_layers = nn.ModuleList([
            FuzzyBinning(definition["n_bins"])
            if definition["binning"]
            else OneHotBinning(definition["n_values"])
            for definition in definitions
        ])

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    @property
    def hard_binning(self) -> bool:
        return self._hard_binning

    @hard_binning.setter
    def hard_binning(self, value: bool) -> None:
        for bin_layer in self.binning_layers:
            bin_layer.hard = value
        self._hard_binning = value

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        binned_features = [bin_layer(x[:, i]) for i, bin_layer in enumerate(self.binning_layers)]
        flat_x = torch.cat(binned_features, dim=-1)

        return self.mlp(flat_x)
