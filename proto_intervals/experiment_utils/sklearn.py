from typing import Literal

import numpy as np
import optuna
import wandb
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (accuracy_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from wandb.apis.public import Run

from proto_intervals.data import get_dataset


def experiment_classifier(
        dataset: str = "cirrhosis",
        model_type: Literal["xgb", "rf", "mlp", "tree"] = "xgb",
        n_folds: int = 5,
        run: Run | None = None,
        kwargs: dict = None,
) -> float:
    if run is None:
        run = wandb.init(project="proto-intervals", entity="jacek-karolczak")

    (x, y), definitions = get_dataset(dataset).as_numpy()

    n_classes = len(np.unique(y))
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold = 0
    accuracies = []

    for train_index, test_index in kf.split(x, y):
        fold += 1

        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        match model_type:
            case "xgb":
                import xgboost as xgb

                model = xgb.XGBClassifier(
                    objective='multi:softmax',
                    num_class=n_classes,
                    random_state=42,
                    seed=42,
                    **kwargs
                )
            case "tree":
                from sklearn.tree import DecisionTreeClassifier

                model = DecisionTreeClassifier(**kwargs, random_state=42)
            case "rf":
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(**kwargs, random_state=42)
            case "mlp":
                from sklearn.neural_network import MLPClassifier

                model = MLPClassifier(**kwargs, random_state=42)
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        gmean = geometric_mean_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average="macro")

        run.log({f"valid/gmean": gmean, "valid/accuracy": acc, "valid/recall": recall})

        accuracies.append(acc)

    return np.mean(accuracies)


def experiment_classifier_optuna(
        dataset: str = "cirrhosis",
        model_type: Literal["xgb", "tree"] = "xgb",
        n_trials: int = 100,
        n_folds: int = 5,
) -> None:
    def objective(trial):
        match model_type:
            case "xgb":
                config = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
                    "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                }
            case "rf":
                config = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                }
            case "mlp":
                config = {
                    "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes",
                                                                    [(100,), (50, 50), (100, 50), (50, 25), (30, 30, 30)]),
                    "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
                    "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                    "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True),
                    "max_iter": trial.suggest_int("max_iter", 100, 1000),
                }
            case "tree":
                config = {
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
                    "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-5, 0.1, log=True),
                }
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")

        run = wandb.init(
            project="proto-intervals",
            entity="jacek-karolczak",
            config={
                "model_type": model_type,
                "dataset": dataset,
                "n_folds": n_folds,
                **config
            },
            group="optuna-cv-search",
        )

        acc = experiment_classifier(
            dataset=dataset,
            model_type=model_type,
            n_folds=n_folds,
            kwargs=config,
            run=run
        )

        wandb.finish()
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
