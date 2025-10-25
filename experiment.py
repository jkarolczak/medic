from typing import TypeAlias, Literal

import click
import numpy as np
import torch


ModelTType: TypeAlias = Literal["xgb", "rf", "mlp", "tree"]

@click.command()
@click.argument("experiment", type=click.Choice(["protopnet-pretty", "protopnet-cv", "protopnet-optuna",
                                                 "sklearn-optuna"]))
@click.option("--dataset", default="cirrhosis", type=click.Choice(["cirrhosis", "ckd", "diabetes"]), help="Dataset name.")
@click.option("--batch_size", default=32, help="Batch size.")
@click.option("--learning_rate", default=0.001, help="Learning rate.")
@click.option("--penalty_l1", default=0.01, help="L1 penalty.")
@click.option("--penalty_diversity", default=0.01, help="Diversity penalty.")
@click.option("--model_type", type=click.Choice(ModelTType.__args__))
def main(experiment: str, dataset: str, batch_size: int, learning_rate: float, penalty_l1: float, penalty_diversity: float,
         model_type: ModelTType, n_prototypes: int, hidden_dim: int) -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    match experiment:
        case "protopnet-pretty":
            from medic.experiment_utils.proto import experiment_protopnet_pretty

            experiment_protopnet_pretty(dataset=dataset, batch_size=batch_size, learning_rate=learning_rate,
                                        penalty_l1=penalty_l1, hidden_dim=hidden_dim, n_prototypes=n_prototypes,
                                        penalty_diversity=penalty_diversity)
        case "protopnet-cv":
            from medic.experiment_utils.proto import experiment_protopnet_cv

            experiment_protopnet_cv(dataset=dataset, batch_size=batch_size, learning_rate=learning_rate,
                                    penalty_l1=penalty_l1, penalty_diversity=penalty_diversity)
        case "protopnet-optuna":
            from medic.experiment_utils.proto import experiment_protopnet_optuna

            experiment_protopnet_optuna(dataset=dataset)
        case "sklearn-optuna":
            from medic.experiment_utils.sklearn import experiment_classifier_optuna

            experiment_classifier_optuna(dataset=dataset, model_type=model_type)


if __name__ == "__main__":
    main()
