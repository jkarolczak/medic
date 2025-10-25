import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from torch.utils.data import (DataLoader,
                              TensorDataset)
from wandb.sdk.wandb_run import Run

from medic import pretty_print
from medic.classifiers.protopnet_classifier import ProtoPNet
from medic.data import get_dataset
from medic.preprocessing import StandardScaler
from medic.utils import (train_model,
                         evaluate_model)


def experiment_protopnet_pretty(
        dataset: str = "cirrhosis",
        batch_size: int = 32,
        hidden_dim: int = 5,
        n_prototypes: int = 40,
        learning_rate: float = 0.01,
        penalty_l1: float = 0.01,
        penalty_diversity: float = 0.01,
) -> None:
    (x, y), definitions = get_dataset(dataset).as_torch()

    n_samples = x.shape[0]
    n_classes = len(np.unique(y))

    train_idcs = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
    test_idcs = np.array([i for i in range(n_samples) if i not in train_idcs])

    x_train, y_train = x[train_idcs], y[train_idcs]
    x_test, y_test = x[test_idcs], y[test_idcs]

    class_weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train.detach().cpu().numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    scaler = StandardScaler(definitions=definitions)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # --- Training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProtoPNet(definitions=definitions, n_classes=n_classes, n_prototypes=n_prototypes, hidden_dim=hidden_dim
                      ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Train & Evaluate ---
    print("\nTraining phase...")
    train_model(model=model, dataloader=train_loader, device=device, optimizer=optimizer, criterion=criterion, num_epochs=40,
                penalty_l1=penalty_l1, penalty_diversity=penalty_diversity)
    evaluate_model(model=model, dataloader=test_loader, device=device)

    # --- Fine-Tuning: Freeze Binning and Train Only Classifier ---
    print("\nFine-tuning phase (hard binning)...")

    model.hard_binning = True
    train_model(model=model, dataloader=train_loader, device=device, criterion=criterion, optimizer=optimizer, num_epochs=30)
    evaluate_model(model=model, dataloader=test_loader, device=device)

    print("\nFine-tuning phase (real prototypes)...")

    model.set_real_prototypes(x_train)
    model.hard_parts = True
    train_model(model=model, dataloader=train_loader, device=device, criterion=criterion, optimizer=optimizer, num_epochs=30)
    evaluate_model(model=model, dataloader=test_loader, device=device)

    print("\nBinning Layers:")
    pretty_print.binning_layers(model, scaler, definitions=definitions)

    i = 1
    pretty_print.predict_and_explain(model, x_test[i], scaler, definitions=definitions)


def experiment_protopnet_cv(
        dataset: str = "cirrhosis",
        batch_size: int = 64,
        learning_rate: float = 0.005,
        hidden_dim: int = 4,
        penalty_l1: float = 0.01,
        penalty_diversity: float = 0.02,
        n_prototypes: int = 32,
        n_folds: int = 5,
        run: Run | None = None,
        *args, **kwargs
) -> float:
    if run is None:
        run = wandb.init(project="proto-intervals", entity="jacek-karolczak")

    (x, y), definitions = get_dataset(dataset).as_torch()

    n_classes = len(np.unique(y))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold = 0
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        fold += 1

        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        class_weights = compute_class_weight("balanced", classes=np.arange(n_classes),
                                             y=y_train.detach().cpu().numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        scaler = StandardScaler(definitions=definitions)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

        model = ProtoPNet(definitions=definitions, n_classes=n_classes, n_prototypes=n_prototypes, n_patches=2 * n_prototypes,
                          hidden_dim=hidden_dim).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # --- Train & Evaluate ---
        train_model(model=model, dataloader=train_loader, device=device, optimizer=optimizer, criterion=criterion,
                    num_epochs=40, penalty_l1=penalty_l1, penalty_diversity=penalty_diversity, verbose=False)
        evaluate_model(model=model, dataloader=test_loader, device=device, log_wandb=True, log_prefix="1-fuzzy", log_run=run)

        # --- Fine-Tuning Phase 1 ---
        model.hard_binning = True
        train_model(model=model, dataloader=train_loader, device=device, criterion=criterion, optimizer=optimizer,
                    num_epochs=30, verbose=False)
        evaluate_model(model=model, dataloader=test_loader, device=device, log_wandb=True, log_prefix="2-hard-binning",
                       log_run=run)

        # --- Fine-Tuning Phase 2 ---
        model.set_real_prototypes(x_train)
        model.hard_parts = True
        train_model(model=model, dataloader=train_loader, device=device, criterion=criterion, optimizer=optimizer,
                    num_epochs=30, verbose=False)
        acc, _, _ = evaluate_model(model=model, dataloader=test_loader, device=device, log_wandb=True,
                                   log_run=run)
        accuracies.append(acc)
    return np.mean(accuracies).item()


def experiment_protopnet_optuna(
        dataset: str = "cirrhosis",
        n_trials: int = 100,
        n_folds: int = 5,
) -> None:
    def objective(trial):
        config = {
            "dataset": dataset,
            "model_type": "protopnet",
            "batch_size": trial.suggest_int("batch_size", 16, 256, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "hidden_dim": trial.suggest_int("hidden_dim", 2, 16),
            "n_prototypes": trial.suggest_int("n_prototypes", 4, 96, step=4),
            "n_folds": n_folds,
            "penalty_l1": 0.01,
            "penalty_diversity": 0.02,
        }

        run = wandb.init(
            project="proto-intervals",
            entity="jacek-karolczak",
            config=config,
            group="optuna-cv-search",
            reinit=True,
        )

        acc = experiment_protopnet_cv(
            **config,
            run=run
        )

        wandb.finish()
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
