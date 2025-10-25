from abc import ABC

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from ucimlrepo import fetch_ucirepo

from medic import TDefinitionList


def load_artificial(n_samples: int = 1000) -> tuple[tuple[Tensor, Tensor], TDefinitionList]:
    np.random.seed(42)

    hemoglobin = np.clip(np.random.normal(loc=14.0, scale=8, size=n_samples), 8, 18)
    glucose = np.clip(np.random.normal(loc=90, scale=100, size=n_samples), 50, 250)
    cholesterol = np.clip(np.random.normal(loc=200, scale=40, size=n_samples), 100, 300)
    dummy = np.random.randint(0, 2, size=n_samples)
    age = np.clip(np.random.uniform(50, 20, size=n_samples), 18, 90)

    x = np.vstack([hemoglobin, glucose, cholesterol, dummy, age]).T
    y = (((hemoglobin < 12) & (glucose > 126)) | (cholesterol > 240)).astype(int)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    definition = [
        {"name": "hemoglobin", "binning": True, "n_bins": 3},
        {"name": "glucose", "binning": True, "n_bins": 3},
        {"name": "cholesterol", "binning": True, "n_bins": 3},
        {"name": "dummy", "binning": False, "n_values": 2},
        {"name": "age", "binning": True, "n_bins": 4},
    ]

    return (x, y), definition


class IDataset(ABC):
    x = None
    y = None
    definitions = None

    def as_torch(self) -> tuple[tuple[Tensor, Tensor], TDefinitionList]:
        """Convert dataset to torch tensors."""
        x = torch.tensor(self.x.values.astype(np.float32), dtype=torch.float32)
        y = torch.tensor(self.y.values.astype(np.float32), dtype=torch.long)
        return (x, y), self.definitions

    def as_pandas(self) -> tuple[tuple[pd.DataFrame, pd.Series], TDefinitionList]:
        """Convert dataset to pandas DataFrames."""
        return (self.x, self.y), self.definitions

    def as_numpy(self) -> tuple[tuple[np.ndarray, np.ndarray], TDefinitionList]:
        """Return dataset as numpy arrays."""
        x = self.x.to_numpy().astype(np.float32)
        y = self.y.to_numpy().astype(np.int32)
        return (x, y), self.definitions


class Cirrhosis(IDataset):
    def __init__(self, path: str = "data/cirrhosis.csv") -> None:
        self.x, self.y = self.read_data(path)
        self.definitions = [
            {"name": "N_Days", "binning": True, "n_bins": 3},
            {"name": "Drug", "binning": False, "n_values": 2},
            {"name": "Ascites", "binning": False, "n_values": 2},
            {"name": "Hepatomegaly", "binning": False, "n_values": 2},
            {"name": "Spiders", "binning": False, "n_values": 2},
            {"name": "Edema", "binning": False, "n_values": 3},
            {"name": "Bilirubin", "binning": True, "n_bins": 3},
            {"name": "Cholesterol", "binning": True, "n_bins": 3},
            {"name": "Albumin", "binning": True, "n_bins": 3},
            {"name": "Copper", "binning": True, "n_bins": 3},
            {"name": "Alk_Phos", "binning": True, "n_bins": 3},
            {"name": "SGOT", "binning": True, "n_bins": 3},
            {"name": "Tryglicerides", "binning": True, "n_bins": 3},
            {"name": "Platelets", "binning": True, "n_bins": 3},
            {"name": "Prothrombin", "binning": True, "n_bins": 3}
        ]

    def read_data(self, path: str) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(path)
        df = df[~df["Drug"].isna()]
        df = df.fillna({
            "Cholesterol": df["Cholesterol"].mean(),
            "Copper": df["Copper"].mean(),
            "Tryglicerides": df["Tryglicerides"].mean(),
            "Platelets": df["Platelets"].mean()
        })
        with pd.option_context("future.no_silent_downcasting", True):
            y = df["Status"].replace({
                "D": 0,
                "CL": 1,
                "C": 2
            })
            y = y.astype(int)
        x = df.drop(columns=[
            "ID",
            "Status",
            "Age",
            "Sex"
        ])
        with pd.option_context("future.no_silent_downcasting", True):
            x = x.replace({
                "Drug": {"D-penicillamine": 0, "Placebo": 1},
                "Ascites": {"N": 0, "Y": 1},
                "Hepatomegaly": {"N": 0, "Y": 1},
                "Spiders": {"N": 0, "Y": 1},
                "Edema": {"N": 0, "Y": 1, "S": 2},
            })
        return x, y


class Diabetes(IDataset):
    def __init__(self, path: str = "data/diabetes.csv") -> None:
        self.x, self.y = self.read_data(path)
        self.definitions = [
            {"name": "Pregnancies", "binning": True, "n_bins": 3},
            {"name": "Glucose", "binning": True, "n_bins": 3},
            {"name": "BloodPressure", "binning": True, "n_bins": 3},
            {"name": "SkinThickness", "binning": True, "n_bins": 3},
            {"name": "Insulin", "binning": True, "n_bins": 3},
            {"name": "BMI", "binning": True, "n_bins": 3},
            {"name": "DiabetesPedigreeFunction", "binning": True, "n_bins": 3},
            {"name": "Age", "binning": True, "n_bins": 3},
        ]

    def read_data(self, path: str) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(path)
        with pd.option_context("future.no_silent_downcasting", True):
            y = df["Outcome"]
            y = y.astype(int)
        x = df.drop(columns=["Outcome"])
        return x, y


class CKD(IDataset):
    """Chronic Kidney Disease dataset."""

    def __init__(self) -> None:
        self.x, self.y = self.read_data()
        self.definitions = [
            # Numerical features
            {"name": "age", "binning": True, "n_bins": 3},
            {"name": "bp", "binning": True, "n_bins": 3},
            {"name": "sg", "binning": False, "n_values": 5},
            {"name": "al", "binning": False, "n_values": 6},
            {"name": "su", "binning": False, "n_values": 6},
            {"name": "rbc", "binning": False, "n_values": 2},
            {"name": "pc", "binning": False, "n_values": 2},
            {"name": "pcc", "binning": False, "n_values": 2},
            {"name": "ba", "binning": False, "n_values": 2},
            {"name": "bgr", "binning": True, "n_bins": 3},
            {"name": "bu", "binning": True, "n_bins": 3},
            {"name": "sc", "binning": True, "n_bins": 3},
            {"name": "sod", "binning": True, "n_bins": 3},
            {"name": "pot", "binning": True, "n_bins": 3},
            {"name": "hemo", "binning": True, "n_bins": 3},
            {"name": "pcv", "binning": True, "n_bins": 3},
            {"name": "wc", "binning": True, "n_bins": 3},
            {"name": "rc", "binning": True, "n_bins": 3},
            {"name": "htn", "binning": False, "n_values": 2},
            {"name": "dm", "binning": False, "n_values": 2},
            {"name": "cad", "binning": False, "n_values": 2},
            {"name": "appet", "binning": False, "n_values": 2},
            {"name": "pe", "binning": False, "n_values": 2},
            {"name": "ane", "binning": False, "n_values": 2},
        ]

    def read_data(self) -> tuple[pd.DataFrame, pd.Series]:
        dataset = fetch_ucirepo(id=336)
        x = dataset.data.features
        y = dataset.data.targets

        # drop rows with missing values
        nas = x.isna().any(axis=1)
        x = x[~nas]
        y = y[~nas]

        mappings = {
            "rbc": {"normal": 1, "abnormal": 0},
            "pc": {"normal": 1, "abnormal": 0},
            "pcc": {"present": 1, "notpresent": 0},
            "ba": {"present": 1, "notpresent": 0},
            "htn": {"yes": 1, "no": 0},
            "dm": {"yes": 1, "no": 0},
            "cad": {"yes": 1, "no": 0},
            "appet": {"good": 1, "poor": 0},
            "pe": {"yes": 1, "no": 0},
            "ane": {"yes": 1, "no": 0},
        }

        with pd.option_context("future.no_silent_downcasting", True):
            y = y.iloc[:, 0].map({"ckd": 1, "notckd": 0}).astype(int)

            for col, mapping in mappings.items():
                x[col] = x[col].map(mapping)
        return x, y


def get_dataset(dataset_name: str) -> IDataset:
    match dataset_name:
        case "cirrhosis":
            return Cirrhosis()
        case "diabetes":
            return Diabetes()
        case "ckd":
            return CKD()
        case _:
            raise ValueError(f"Dataset {dataset_name} not found.")
