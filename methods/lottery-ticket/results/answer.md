# The Lottery Ticket Hypothesis

Given a dense network `f(x; theta)` initialized at `theta_0 ~ D_theta`, dense training reaches minimum validation loss at iteration `j` with test accuracy `a`. The claim is that there exists a binary mask `m` such that the masked network `f(x; m * theta_0)`, trained in isolation with `m` fixed, reaches test accuracy `a' >= a` in `j' <= j` iterations while using far fewer parameters: `||m||_0 << |theta|`.

The method that exposes such a subnetwork is iterative magnitude pruning with reset to the original initialization. The pruning step matches the canonical implementation: rank only currently surviving weights, choose the rounded percentile cutoff, and zero every entry whose absolute value is at or below that cutoff while preserving the old mask elsewhere.

```python
import numpy as np

def prune_by_percent_once(percent, mask, final_weight):
    sorted_weights = np.sort(np.abs(final_weight[mask == 1]))
    cutoff_index = np.round(percent * sorted_weights.size).astype(int)
    cutoff = sorted_weights[cutoff_index]
    return np.where(np.abs(final_weight) <= cutoff, np.zeros(mask.shape), mask)

def prune_by_percent(percents, masks, final_weights):
    return {
        name: prune_by_percent_once(percent, masks[name], final_weights[name])
        for name, percent in percents.items()
    }

def find_ticket(train_fn, prune_per_layer, rounds):
    initial, final = train_fn(presets=None, masks=None)
    masks = {name: np.ones(weights.shape) for name, weights in initial.items()}

    for _ in range(rounds):
        masks = prune_by_percent(prune_per_layer, masks, final)
        _, final = train_fn(presets=initial, masks=masks)

    return masks, {name: masks[name] * initial[name] for name in masks}
```

Here `train_fn` follows the released experiment API: its first call records the initial and final weights, and later calls pass `presets=initial` with the current masks. The model applies each mask inside the layer computation. In the fully connected MNIST setup, the pruning dictionary is `{"layer0": 0.2, "layer1": 0.2, "layer2": 0.1}`, so the output layer is pruned at half the hidden-layer rate.

The required control keeps the same masks but discards the original initialization:

```python
def random_reinit_control(train_fn, masks):
    return train_fn(presets=None, masks=masks)
```

If the original-initialization subnetwork trains well but the same mask with a fresh draw trains slower or less accurately, the sparse architecture alone is not the explanation. The scientific point is that overparameterized training succeeds partly because the dense random initialization contains many candidate subnetworks; pruning after training reveals one whose original initial weights make it trainable.
