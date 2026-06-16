# Knowledge Distillation

## Problem

A cumbersome model — a large heavily-regularized network or an ensemble — gives the best accuracy but is too expensive (latency, memory, compute) to deploy. Transfer the *function* it has learned into a single small, fast model with minimal accuracy loss.

## Key idea

The knowledge of a trained classifier is its learned mapping from inputs to a distribution over classes, not its specific weights. The most valuable, transferable part is the *relative* probabilities the model assigns to the **wrong** classes ("dark knowledge") — a similarity structure over classes that a hard one-hot label discards. Those probabilities are tiny, so an ordinary softmax buries them and they barely influence training. **Raise the softmax temperature** $T$ to soften the teacher's output and surface that structure, train the small (student) model at the same temperature to match these *soft targets*, and revert to $T=1$ at deployment.

## Method

Temperature-scaled softmax:
$$q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}.$$

Train the student with a weighted sum of a soft transfer objective and, when labels are available, an ordinary hard-label objective:

- **Soft loss** $L_{\text{soft}}$: $KL(p_T\|q_T)$, equivalently soft-target cross-entropy up to the teacher's fixed entropy, where $p_T = \mathrm{softmax}(v/T)$ and $q_T = \mathrm{softmax}(z/T)$.
- **Hard loss** $L_{\text{hard}}$: ordinary cross-entropy against the true labels at $T=1$, optionally added with a smaller weight.

$$L = w_{\text{soft}}\, T^2\, L_{\text{soft}} + w_{\text{hard}}\, L_{\text{hard}}.$$

The $T^2$ factor compensates for the high-temperature soft-target gradient scaling as $1/T^2$: with it the relative contribution of the soft and hard terms remains roughly stable while sweeping $T$.

**Matching logits is the high-$T$ limit.** Per transfer case, the soft cross-entropy gradient is
$$\frac{\partial C}{\partial z_i} = \frac{1}{T}(q_i - p_i).$$
For large $T$, using $\exp(x/T)\approx 1+x/T$ and zero-meaning each case's logits ($\sum_j z_j=\sum_j v_j=0$),
$$\frac{\partial C}{\partial z_i} \approx \frac{1}{N T^2}(z_i - v_i),$$
i.e. minimizing $\tfrac12\sum_i (z_i-v_i)^2$ up to a positive scalar — the squared logit-matching objective. At lower $T$ the approximation breaks for very negative logits, so distillation pays *less* attention to matching them; $T$ is a dial trading off whether to fit those nearly-unconstrained (possibly noisy) logits. For a very small student, intermediate $T$ can be the better capacity allocation because it stops spending equal effort on every extremely negative logit.

## Code

```python
import torch
import torch.nn.functional as F

def transfer_loss(small_logits, large_logits, hard_targets=None,
                  temperature=1.0, soft_weight=1.0, hard_weight=0.0):
    T = float(temperature)

    # Soft transfer: KL(p_T || q_T), same gradient as soft-target CE.
    with torch.no_grad():
        p = F.softmax(large_logits / T, dim=1)
    log_q = F.log_softmax(small_logits / T, dim=1)
    soft = F.kl_div(log_q, p, reduction="batchmean") * (T * T)

    loss = soft_weight * soft
    if hard_targets is not None and hard_weight:
        # Optional hard-label CE at T=1.
        hard = F.cross_entropy(small_logits, hard_targets)
        loss = loss + hard_weight * hard
    return loss

# Training (large model frozen; transfer set = train set or a separate set)
large_model.eval()
opt = torch.optim.SGD(small_model.parameters(), lr=0.1, momentum=0.9)
T, hard_weight = 20.0, 0.0                                # set hard_weight > 0 when labels should also pull
for x, y in transfer_loader:
    with torch.no_grad():
        large_logits = large_model(x)
    small_logits = small_model(x)
    loss = transfer_loss(small_logits, large_logits, y, temperature=T,
                         soft_weight=1.0, hard_weight=hard_weight)
    opt.zero_grad(); loss.backward(); opt.step()
# Deployment: small_model runs an ordinary T=1 softmax.
```
