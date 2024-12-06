
import collections

import torch


def test(model, data_loader, criterion, device=None):
    # Setup eval and device
    device = device or torch.device("cpu")
    model.eval()
    predicted = []
    targets = []
    losses = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            # Append data
            targets.append(target)
            predicted.append(outputs)
            losses.append(loss)

    return predicted, targets, losses


def join_dicts(d, u):
    """
    Recursively updates a dictionary
    """
    result = d.copy()

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            result[k] = join_dicts(d.get(k, {}), v)
        else:
            result[k] = v
    return result
