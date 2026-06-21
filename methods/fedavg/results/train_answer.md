The problem is to train a single shared deep model across a large population of mobile devices while every device keeps its own data private. The constraints are severe: data is non-IID across devices, dataset sizes vary by orders of magnitude, the number of devices is huge, and communication is rare and slow. Raw examples can never leave the device. Because on-device compute is nearly free compared with the scarce resource of communication rounds, the real objective is to reach a target accuracy in as few rounds as possible, spending extra local computation to buy fewer rounds.

The natural baseline is distributed SGD: each selected client computes its local gradient, the server averages those gradients weighted by dataset size, and takes one step. This is correct, but it consumes exactly one gradient step per communication round. Training a modern network can require tens of thousands of steps, so thousands of communication rounds are needed, which is impractical when a device participates only a few times per day. One-shot averaging, where each client trains locally to convergence and the server averages only once, uses a single round but cannot remove the bias of each client's local non-IID problem and can produce a model no better than one client's local model.

The method is Federated Averaging, or FedAvg. The key observation is that one synchronous distributed SGD step can be rewritten. In the baseline, each client sends its local gradient and the server takes a weighted step. Algebraically, this is identical to each client taking one local gradient step from the current global model and the server averaging the resulting models with the same weights. Once clients are shipping models instead of gradients, nothing prevents each client from taking many local steps before shipping. FedAvg lets each selected client run several epochs of plain SGD on its own data, in minibatches, and only then return its updated model to the server. The server then replaces the global model with the sample-count-weighted average of the returned client models. This trades cheap local computation for the expensive communication rounds.

Two design choices make this work. First, the weights are proportional to each client's dataset size, because the global objective is itself the sample-weighted average of the per-client objectives. Equal weighting would over-represent tiny clients. When only a fraction of clients participate in a round, the average is normalized over the selected clients' total sample count, not the whole population, so the result remains a proper convex combination of the models actually received. Second, the server re-broadcasts the same global model at the start of every round. Averaging independently trained non-convex models is normally dangerous because the straight line between them can cross a ridge in the loss landscape. Because all clients in a round start from the same point and drift only modestly before the next average, they remain in the same basin, and averaging stays meaningful.

FedAvg has three knobs: the client fraction C controls parallelism and the global batch size; the number of local epochs E controls how much each client trains per round; and the local minibatch size B controls the granularity of the local updates. Setting E to one and B to infinity recovers the one-step-per-round baseline. The algorithm is most useful in the interior between that baseline and the one-shot extreme: enough local work to make each round count, but not so much that clients overfit their own data and drift out of agreement.

```python
import random
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader


def _client_sgd(model, loader, loss_fn, local_epochs, local_lr, device):
    """ClientUpdate: E epochs of plain SGD on a client's local data."""
    opt = torch.optim.SGD(model.parameters(), lr=local_lr)
    total_loss, total_n = 0.0, 0
    for _ in range(local_epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            outputs = model(inputs)
            if outputs.dim() == 3:
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            loss = loss_fn(outputs, targets)
            loss.backward()
            opt.step()
            total_loss += loss.item() * inputs.size(0)
            total_n += inputs.size(0)
    return total_loss / max(total_n, 1)


class Strategy:
    """FedAvg: local SGD + sample-count-weighted average of client models,
    re-anchored each round to the broadcast global model."""

    def __init__(self, global_model, args):
        self.args = args

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        avg_loss = _client_sgd(model, loader, loss_fn,
                               local_epochs, local_lr, device)
        local_state = OrderedDict(
            (key, value.detach().cpu().clone())
            for key, value in model.state_dict().items()
        )
        return local_state, len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        if not client_updates:
            return OrderedDict((k, v.detach().clone())
                               for k, v in global_state_dict.items())
        total_samples = sum(n for _, n, _ in client_updates)
        if total_samples <= 0:
            raise ValueError("FedAvg aggregation requires positive sample counts")
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not ref.is_floating_point():
                new_state[key] = client_updates[0][0][key].detach().clone()
                continue
            acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
            for st, n, _ in client_updates:
                acc += st[key].detach().cpu().float() * (n / total_samples)
            new_state[key] = acc.to(ref.dtype)
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        return random.sample(range(num_available),
                             min(num_to_select, num_available))
```
