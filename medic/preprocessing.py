import torch

from medic import TDefinitionList


class StandardScaler:
    def __init__(self, definitions: TDefinitionList) -> None:
        self.mean = None
        self.std = None
        self.continuous_indices = [i for i, item in enumerate(definitions) if item["binning"]]

    def fit(self, x: torch.Tensor) -> None:
        self.mean = x.mean(dim=0)[self.continuous_indices]
        self.std = x.std(dim=0)[self.continuous_indices]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x[:, self.continuous_indices] = (x[:, self.continuous_indices] - self.mean) / self.std
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        x[:, self.continuous_indices] = x[:, self.continuous_indices] * self.std + self.mean
        return x

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)
