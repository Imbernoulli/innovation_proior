The core problem is that large neural classifiers overfit, and the most visible symptom of that overfitting lives in the output distribution rather than in any particular weight or activation. Standard regularizers such as dropout, weight decay, and batch normalization all modify the internal representations, but the conditional distribution $p_\theta(y|x)$ that the model actually emits from its softmax is left almost untouched. That distribution is the right object to regularize: it has a natural, fixed scale independent of how the network underneath is parameterized, and its wrong-class ratios carry generalization information that should not be destroyed. What is needed is a simple, differentiable penalty computed directly from the softmax output that discourages the peaked, near-deterministic predictions that accompany overfitting.

Existing ideas fall short because none of them target the output distribution in the right adaptive way. Dropout and weight decay act on internals and empirically still drive the softmax toward 0/1 spikes. Label smoothing replaces the one-hot target with a mixture of the hard label and a uniform prior, which bounds confidence below one, but it applies the same uniform pressure to every class regardless of how confident the model currently is. That makes it a rigid mean-shift: it flattens the entire output distribution toward a fixed target and erases the task-specific "dark knowledge" ratios among incorrect classes. Distillation preserves those ratios, but it requires a teacher network or a carefully scheduled self-distillation setup. Virtual adversarial training is effective but adds multiple forward and backward passes per step, making it expensive to train and to grid-search. A penalty that is cheap, parameterization-invariant, and adaptive to the model's own confidence is missing.

The method is the confidence penalty, introduced by Pereyra et al. in 2017. It adds the negative entropy of the softmax output to the usual cross-entropy loss. For logits $z$ and softmax probabilities $p_i = \exp(z_i) / \sum_j \exp(z_j)$, the entropy is $H(p) = -\sum_i p_i \log p_i$, so the training objective becomes

$L(\theta) = -\sum \log p_\theta(y|x) - \beta H(p_\theta(y|x)).$

Because we minimize the loss, the negative entropy term rewards high entropy, which is equivalent to penalizing low entropy and therefore over-confidence. The scalar $\beta$ is the only hyperparameter; it trades off data fit against output humility and is selected on validation data while the rest of the training pipeline stays fixed.

The gradient of the entropy term with respect to logit $z_i$ is $\partial H / \partial z_i = p_i(-\log p_i - H)$. This is the deviation of class $i$'s surprisal from the mean surprisal, weighted by the model's own probability $p_i$ for that class. The consequence is that the update pushes hardest on the dominant, over-confident class and barely touches classes that already have near-zero mass. It flattens the distribution toward uniform, but it does not force every incorrect class toward the same fixed target value. This adaptive weighting preserves the relative ratios among the long-tail wrong classes much better than a uniform target would. In the membership-inference setting, this adaptivity is especially important: the penalty concentrates on the spiked member outputs that produce the leakage signal, rather than translating the entire confidence distribution uniformly.

The confidence penalty can also be understood as a KL-divergence penalty toward the uniform distribution $u$, but in the reverse direction from label smoothing. Label smoothing corresponds to the forward KL $D_{KL}(u \| p)$, where every class is weighted by the constant $1/K$. The confidence penalty corresponds to the reverse KL $D_{KL}(p \| u) = -H(p) + \log K$, where each class is weighted by the model's current $p_i$. That asymmetry is exactly why the confidence penalty focuses on the classes the model is currently confident about and avoids writing an explicit wrong-label target distribution into every training example. In practice the entropy form is used, so no prior over labels needs to be chosen, making it easy to drop into language modeling and other tasks with highly non-uniform label frequencies.

A subtlety is that in reinforcement learning the entropy bonus is kept on throughout training to encourage exploration, but in supervised learning we want fast convergence early and humility only near the end. The simplest response is to anneal $\beta$ upward over training. A sharper alternative is a hinge that activates only when the output entropy falls below a threshold $\Gamma$: $L = -\sum \log p(y) + \beta \max(0, \Gamma - H(p))$. When the output is already humble enough the penalty is zero and training proceeds as usual; when the output becomes too peaked the penalty switches on in proportion to the shortfall. The plain unscheduled form with a fixed $\beta$ is the simplest default and the one used in the standard scaffold.

The implementation is numerically stable and adds essentially no cost to each step. We compute `log_softmax` once in a single pass, recover the probabilities by exponentiating, and then form the entropy. This avoids the precision loss that comes from taking `log(softmax(x))` as two separate operations.

```python
import torch
import torch.nn.functional as F


def compute_regularization(model, inputs, outputs, targets, config):
    """Confidence penalty: penalize low-entropy (over-confident) softmax outputs.

    Adds -beta * H(p) to the cross-entropy loss, so minimizing the loss
    maximizes the entropy of the model's predictions.

    Args:
        model:   the nn.Module (unused by this output-distribution penalty).
        inputs:  input batch, shape [B, C, H, W].
        outputs: model logits, shape [B, num_classes].
        targets: ground-truth integer labels, shape [B].
        config:  dict with at least key "beta" (regularization strength).

    Returns:
        A scalar differentiable tensor added to the cross-entropy loss.
    """
    beta = float(config.get("beta", 0.1))

    # Stable log-probabilities in one pass.
    log_p = F.log_softmax(outputs, dim=-1)
    p = log_p.exp()

    # Entropy averaged over the batch: H(p) = - sum_i p_i log p_i.
    entropy = -(p * log_p).sum(dim=-1).mean()

    # Negative entropy => penalize confidence.
    return -beta * entropy
```
