import torch
from torch import nn

from medic import TDefinitionList
from medic.nn import EPS
from medic.nn.binning import FuzzyBinning, OneHotBinning


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int = 4,
    ) -> None:
        """Feature extractor module that transforms input embeddings (preferably binary) into abstract embeddings.

        Args:
            embedding_dim (int): dimensionality of input embeddings.
            hidden_dim (int, optional): dimensionality of abstract embeddings. Defaults to 4.

        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(x)


class Medic(nn.Module):
    def __init__(
            self,
            definitions: TDefinitionList,
            n_classes: int,
            n_patches: int = 64,
            n_prototypes: int = 20,
            hidden_dim: int = 6,
    ) -> None:
        """MEDIC uses prototypical parts to classify objects.

        Args:
            definitions (TDefinitionList): list of feature definitions.
            n_classes (int): number of classes.
            n_patches (int, optional): number of parts to split input into. Defaults to 64.
            n_prototypes (int, optional): number of prototypes. Defaults to 32.
            hidden_dim (int, optional): dimensionality of abstract embeddings. Defaults to 4.

        """
        super().__init__()
        self._hard_binning = False
        self.hard_parts = False
        self.n_patches = n_patches
        self.n_prototypes = n_prototypes
        self.embedding_dim = sum([item["n_bins"] if item["binning"] else item["n_values"] for item in definitions])
        self.hidden_dim = hidden_dim

        # stage 1: discretize input into bins (trainable thresholds)
        self.binning_layers = nn.ModuleList([
            FuzzyBinning(name=definition["name"], n_bins=definition["n_bins"])
            if definition["binning"]
            else OneHotBinning(name=definition["name"], n_values=definition["n_values"])
            for definition in definitions
        ])

        # stage 2: split input into parts (trainable masks)
        self._patches_extractor = nn.Parameter(torch.randn(self.n_patches, self.embedding_dim), requires_grad=True)
        nn.init.xavier_normal_(self._patches_extractor)

        # stage 3: generate features (abstract embeddings) from each part
        self._feature_extractor = FeatureExtractor(self.embedding_dim, hidden_dim)

        # stage 4: compute distances to prototypes (learnable vectors in embedding space)
        self.register_buffer("max_distance", torch.tensor(1.0))
        self._prototypical_parts_embeddings = nn.Parameter(torch.randn(n_prototypes, self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self._prototypical_parts_embeddings)
        self.prototypical_parts = None

        # stage 5: classify based on distances to prototypes (single linear layer)
        self._classification_head = nn.Linear(self.n_prototypes, n_classes)

    @property
    def l1_factor(self) -> torch.Tensor:
        """L1 regularization factor for the patches extractor to promote sparsity."""

        return torch.mean(torch.abs(self._patches_extractor))

    @property
    def diversity_factor(self) -> torch.Tensor:
        """Diversity factor for the prototypes to promote diversity."""

        prototypes = self._prototypical_parts_embeddings
        distances = torch.cdist(prototypes, prototypes, p=2)
        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
        dists_non_diag = distances[mask]
        return -dists_non_diag.mean()

    @property
    def hard_binning(self) -> bool:
        """Whether hard or soft binning is used."""

        return self._hard_binning

    @hard_binning.setter
    def hard_binning(self, value: bool) -> None:
        for bin_layer in self.binning_layers:
            bin_layer.hard = value
        self._hard_binning = value

    @property
    def frozen_prototypes(self) -> bool:
        """Whether the prototypes are frozen or not."""
        return not self._prototypical_parts_embeddings.requires_grad

    @frozen_prototypes.setter
    def frozen_prototypes(self, value: bool) -> None:
        self._feature_extractor.requires_grad = not value
        self._patches_extractor.requires_grad = not value
        self._prototypical_parts_embeddings.requires_grad = not value

    def forward_binary_features(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        binned_features = [bin_layer(x[:, i]) for i, bin_layer in enumerate(self.binning_layers)]
        binned_features = torch.cat(binned_features, dim=-1).unsqueeze(1)
        return binned_features

    def forward_parts(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        binned_features = self.forward_binary_features(x)
        parts = self._patches_extractor * binned_features
        if self.hard_parts:
            parts = parts * (parts > EPS).float()
        return parts

    def forward_features(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self._feature_extractor(self.forward_parts(x))

    def forward_distances(
            self,
            x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        distances = torch.cdist(features, self._prototypical_parts_embeddings, p=2)

        if self.frozen_prototypes:
            self.max_distance = torch.maximum(
                self.max_distance,
                distances.max().detach()
            )
        distances = distances / self.max_distance

        distances = torch.min(distances, dim=1)
        return distances

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        min_distances, _ = self.forward_distances(x)
        classification = self._classification_head(min_distances)
        return classification

    def set_real_prototypes(self, x: torch.Tensor) -> None:
        """Finds the parts of real objects that are closest to the prototypes and sets them as the new prototypes.

        Additionally, parts of real objects that were selected as prototypical parts are set in the model as property called
         `prototypical_parts`.

        Args:
            x (torch.Tensor): real objects to find prototypical parts from. Typically, the training data.

        """

        self.frozen_prototypes = True

        binned_features = [bin_layer(x[:, i]) for i, bin_layer in enumerate(self.binning_layers)]
        binned_features = torch.cat(binned_features, dim=-1).unsqueeze(1)

        parts = self._patches_extractor * binned_features
        if self.hard_parts:
            parts = parts * (parts > 0.5).float()

        features = self._feature_extractor(parts)
        distances = torch.cdist(features, self._prototypical_parts_embeddings, p=2)

        distances = torch.flatten(distances, 0, 1)
        nearest_ids = torch.argmin(distances, dim=0)
        self._prototypical_parts_embeddings.data = torch.flatten(features, 0, 1)[nearest_ids].data

        prototypical_parts = torch.flatten(parts, 0, 1)[nearest_ids].detach()
        setattr(self, "prototypical_parts", prototypical_parts)
