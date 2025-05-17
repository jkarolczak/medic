from typing import Iterable

import numpy as np
import torch

from proto_intervals import TDefinitionList
from proto_intervals.classifiers.protopnet_classifier import ProtoPNet
from proto_intervals.nn.binning import (FuzzyBinning,
                                        OneHotBinning)
from proto_intervals.preprocessing import StandardScaler


def _bins_to_str(
        bins: list[tuple[float, float]],
        feature_name: str | None = None
) -> list[str]:
    bins_str = [f"{'(' if i == 0 else '['}{start:.3f}, {end:.3f})" for i, (start, end) in enumerate(bins)]
    if feature_name is not None:
        bins_str = [f"{feature_name} in {bin_}" for bin_ in bins_str]
    return bins_str


def _values_to_str(
        values: Iterable[int | float],
        feature_name: str | None = None
) -> list[str]:
    values_str = [f"{value}" for value in values]
    if feature_name is not None:
        values_str = [f"{feature_name} = {value}" for value in values_str]
    return values_str


def _print_feature_bins(
        bins: list[tuple[float, float]],
        feature_name: str = None
) -> None:
    print(f"Feature: {feature_name}")
    bins_str = _bins_to_str(bins)
    print(f"\tBins: {', '.join(bins_str)}")


def _centers_to_bins(
        centers: np.ndarray
) -> list[tuple[float, float]]:
    limits = np.concat([
        [-np.inf],
        (centers[1:] + centers[:-1]) / 2,
        [np.inf]
    ])
    return list(zip(limits[:-1].tolist(), limits[1:].tolist()))


def _print_categorical(
        n_values: int,
        feature_name: str,
) -> None:
    print(f"Feature: {feature_name}")
    print(f"\tBins: {', '.join([f'[{i}]' for i in range(n_values)])}")


def _binning_layers_to_human_readable(
        model: ProtoPNet,
        scaler: StandardScaler,
        definitions: TDefinitionList
) -> list[str]:
    human_readable = []
    for i, (layer, definition) in enumerate(zip(model.binning_layers, definitions)):
        if isinstance(layer, FuzzyBinning):
            mean = scaler.mean[scaler.continuous_indices.index(i)]
            std = scaler.std[scaler.continuous_indices.index(i)]
            bin_centers = (layer.bin_centers * std + mean).detach().cpu().numpy()
            bins = _centers_to_bins(centers=bin_centers)
            human_readable.append(_bins_to_str(bins=bins, feature_name=definition["name"]))
        if isinstance(layer, OneHotBinning):
            human_readable.append(_values_to_str(values=range(definition["n_values"]), feature_name=definition["name"]))
    return human_readable


def binning_layers(
        model: ProtoPNet,
        scaler: StandardScaler,
        definitions: TDefinitionList,
) -> None:
    human_readable = _binning_layers_to_human_readable(model=model, scaler=scaler, definitions=definitions)
    for i, bins in enumerate(human_readable):
        print(f"Feature {i}: {bins}")


def _parts_to_human_readable(
        model: ProtoPNet,
        scaler: StandardScaler,
        definitions: TDefinitionList,
        threshold: float = 0.1,
        prototypical_parts: None | torch.Tensor = None
) -> list[str]:
    antecedents = []
    prototypical_parts = prototypical_parts if prototypical_parts is not None else model.prototypical_parts

    for i, (layer, definition) in enumerate(zip(model.binning_layers, definitions)):
        if isinstance(layer, FuzzyBinning):
            mean = scaler.mean[scaler.continuous_indices.index(i)]
            std = scaler.std[scaler.continuous_indices.index(i)]
            bin_centers = (layer.bin_centers * std + mean).detach().cpu().numpy()
            bins = _centers_to_bins(centers=bin_centers)
            for bin_ in bins:
                antecedents.append(
                    f"{definition['name']} in " +
                    "(" +
                    (f"{bin_[0]:.3f}" if isinstance(bin_[0], float) else str(bin_[0])) +
                    ", " +
                    (f"{bin_[1]:.3f}" if isinstance(bin_[1], float) else str(bin_[1]))
                    + ")"
                )
        if isinstance(layer, OneHotBinning):
            for j in range(definition["n_values"]):
                antecedents.append(f"{definition['name']} = {j}")

    human_readable = []
    for part in prototypical_parts:
        idcs = torch.where(part > threshold)[0]
        rule = []
        for i in idcs:
            rule.append(antecedents[i])
        human_readable.append(" AND ".join(rule))

    return human_readable


def prototypical_parts(
        model: ProtoPNet,
        scaler: StandardScaler,
        definitions: TDefinitionList,
        threshold: float = 0.1
) -> None:
    human_readable = _parts_to_human_readable(model=model, scaler=scaler, definitions=definitions, threshold=threshold)
    human_readable = sorted(human_readable, key=lambda x: len(x.split(" AND ")))
    for i, part in enumerate(human_readable):
        print(f"Proto-part {i}: {part}")


def predict_and_explain(
        model: ProtoPNet,
        x: torch.Tensor,
        scaler: StandardScaler,
        definitions: TDefinitionList,
        threshold: float = 0.1,
        top_k: int = 5
) -> None:
    model.eval()
    with torch.no_grad():
        x = x.unsqueeze(0)
        instance_bins = model.forward_binary_features(x).squeeze([0, 1])
        prediction = model(x).argmax(dim=1).item()

    print(f"Prediction: {prediction}")
    human_readable_model = _parts_to_human_readable(model=model, scaler=scaler, definitions=definitions, threshold=threshold)
    min_distances, argmin_distances = model.forward_distances(x)
    min_distances, argmin_distances = min_distances.squeeze(0), argmin_distances.squeeze(0)
    ranking = (min_distances, argmin_distances, human_readable_model)
    ranking = sorted(zip(*ranking), key=lambda x: x[0])
    bins = sum(_binning_layers_to_human_readable(model=model, scaler=scaler, definitions=definitions), [])
    instance_conditions = [bin_.replace("[", "(").replace("]", ")") for bin_, criterion in zip(bins, instance_bins) if
                           criterion.item()]
    print("Ranking of parts:")
    for i, (distance, idx, part) in enumerate(ranking[:top_k]):
        matching = [val for val in instance_conditions if val in part]
        print(
            f"Rank {i}:"
            f"\tSimilarity: {1 - distance:.3f}\t"
            f"Part (model) {i}: {part}\t"
            f"Matching conditions: {', '.join(matching)}\t"
        )
