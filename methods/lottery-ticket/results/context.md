## Research Question

Modern neural networks are routinely larger than the functions they ultimately need. Post-training pruning shows this directly: after dense training has succeeded, one can remove most of the weights and recover essentially the same accuracy after additional tuning. That makes inference cheaper, and sometimes the smaller model generalizes better.

The open question is whether a sparse subnetwork of comparable size can be trained from scratch—without first training the dense model—to match the dense model's accuracy and training speed.

## Background

Pruning starts from a trained network and deletes parameters that appear unnecessary. Optimal Brain Damage uses second-derivative saliency to remove weights whose deletion should least increase the loss. Optimal Brain Surgeon refines that idea with inverse-Hessian information and compensating updates to the remaining weights. Magnitude pruning is the simpler practical version: train a dense network, remove low-absolute-value weights, and continue training the survivors.

Initialization is also part of the training problem, not just a bookkeeping detail. Dense networks use scaled random initializations so activations and gradients remain numerically reasonable across layers. Those schemes are designed for the full parameterization; an arbitrary sparse subnetwork has fewer paths and a different optimization geometry.

## Baselines

The main compression baseline is train-prune-fine-tune. It demonstrates that a sparse endpoint can represent the learned function, but it starts the sparse model from values already shaped by dense training.

A second baseline keeps the sparse pattern but draws fresh random weights before training. It tests whether the mask structure alone, without any information from the dense training run, is sufficient.

Random sparse subnetworks of comparable size are another baseline. They test whether sparsity at the same parameter count is enough, without using a mask selected by a trained dense model.

## Evaluation Settings

The measurements are standard supervised image-classification training runs. A dense model is trained from a random initialization; validation loss identifies the iteration at which early stopping would occur; test accuracy at that point measures performance. Sparse alternatives are compared by the same two quantities: how quickly they reach their best validation loss and what test accuracy they achieve there.

The relevant testbeds are small fully connected and convolutional image models where repeated train-prune-train experiments are feasible. Masks are unstructured binary tensors over individual weights. Sparsity is reported as the fraction of weights that remain active.

## Code Framework

The machinery already exists for the experiment: a model has weights, binary masks, and a training loop; a masked forward pass multiplies each weight tensor by its mask so removed connections stay inactive; a pruning routine can rank currently surviving weights by absolute value and update the mask.

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

def search_sparse_candidate(train_fn, prune_per_layer, rounds):
    initial, final = train_fn(presets=None, masks=None)
    masks = {name: np.ones(weights.shape) for name, weights in initial.items()}

    for _ in range(rounds):
        masks = prune_by_percent(prune_per_layer, masks, final)
        presets = choose_presets(initial, final, masks)
        _, final = train_fn(presets=presets, masks=masks)

    return masks
```
