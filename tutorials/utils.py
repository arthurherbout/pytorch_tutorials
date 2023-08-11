import torch
from torch import nn
from torch.utils.data import DataLoader
import typing


def get_device_for_training() -> str:
    """
    This method finds the device to use for training.
    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def report_model_parameters(model: nn.Module, show_values: bool = False) -> str:
    """
    Describes the neural network model.
    Args:
        model
        show_values: whether to include tensor values in the report
    Returns:
        report
    """
    report = ""
    for name, param in model.named_parameters():
        layer_str = f"Layer: {name} | Size: {param.size()}"
        if show_values is True:
            layer_str += f" | Values: {param[:2]}"
        layer_str += "\n"
        report += layer_str

    return report


def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> float:
    """

    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :return:
    """
    # set the model to training mode
    model.train()
    model.to(device)

    # initialize the average loss over a single epoch
    avg_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        X.to(device)
        y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # update rule for the average loss over a single epoch
        avg_loss = (batch * avg_loss + loss.item()) / (batch + 1)

    return avg_loss


def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: str) -> typing.Dict[str, float]:
    """

    :param dataloader:
    :param model:
    :param loss_fn:
    :return:
    """
    # set the model to evaluation mode
    model.eval()
    model.to(device)

    # initialize variables
    avg_loss = 0
    avg_acc = 0

    # no need to compute gradients during test mode
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X.to(device)
            y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # update rule for loss and accuracy
            avg_loss = (batch * avg_loss + loss.item()) / (batch + 1)
            avg_acc = (batch * avg_acc + (pred.argmax(1) == y).type(torch.float).sum().item()) / (batch + 1)

    return {"loss": avg_loss,
            "accuracy": avg_acc}



