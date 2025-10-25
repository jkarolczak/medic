import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from wandb.apis.public import Run


def train_model(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        annealing: bool = False,
        num_epochs: int = 20,
        device: torch.device = torch.device("cpu"),
        penalty_l1: float = 0.0,
        penalty_diversity: float = 0.0,
        verbose: bool = True,
) -> None:
    temperture = 1.0
    model.train()
    for epoch in range(num_epochs):
        if annealing:
            temperture = max(0.05, temperture * 0.9)
            for layer in model.binning_layers:
                layer.temperature = temperture
        else:
            for layer in model.binning_layers:
                layer.temperature = 1.0
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch) + penalty_l1 * model.l1_factor + penalty_diversity * model.diversity_factor
            if torch.isnan(loss):  # Detect NaN loss
                print("NaN detected! Stopping training.")
                return
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")


def evaluate_model(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device = torch.device("cpu"),
        log_wandb: bool = False,
        log_prefix: str = "",
        log_run: Run = None,
) -> tuple[float, float, float]:
    model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.cpu())
            ground_truth.append(y_batch.cpu())
    predictions = torch.cat(predictions)
    ground_truth = torch.cat(ground_truth)
    num_classes = (torch.max(ground_truth) + 1).item()
    if num_classes <= 2:
        task = "binary"
    else:
        task = "multiclass"
    acc = torchmetrics.classification.Accuracy(task=task, num_classes=num_classes)(predictions, ground_truth)
    recall = torchmetrics.classification.Recall(task=task, average="macro", num_classes=num_classes)(predictions, ground_truth)
    spec = torchmetrics.classification.Specificity(task=task, average="macro", num_classes=num_classes
                                                   )(predictions, ground_truth)
    gmean = torch.sqrt(recall * spec)
    if log_wandb:
        prefix = f"{log_prefix}/valid" if log_prefix else "valid"
        log_run.log({f"{prefix}/accuracy": acc, f"{prefix}/recall": recall, f"{prefix}/gmean": gmean})
    else:
        print(f"Test - Accuracy: {acc:.4f}; Recall: {recall:.4f}; G-Mean: {gmean:.4f}")

    return acc.cpu().item(), recall.cpu().item(), gmean.cpu().item()
