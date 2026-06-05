# Knowledge Distillation

## Problem

A cumbersome model — a large heavily-regularized network or an ensemble — gives the best accuracy but is too expensive (latency, memory, compute) to deploy. Transfer the *function* it has learned into a single small, fast model with minimal accuracy loss.

## Key idea

The knowledge of a trained classifier is its learned mapping from inputs to a distribution over classes, not its specific weights. The most valuable, transferable part is the *relative* probabilities the model assigns to the **wrong** classes ("dark knowledge") — a similarity structure over classes that a hard one-hot label discards. Those probabilities are tiny, so an ordinary softmax buries them and they barely influence training. **Raise the softmax temperature** $T$ to soften the teacher's output and surface that structure, train the small (student) model at the same temperature to match these *soft targets*, and revert to $T=1$ at deployment.

## Method

Temperature-scaled softmax:
$$q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}.$$

Train the student with a weighted sum of two cross-entropies:

- **Soft loss** $L_{\text{soft}}$: cross-entropy between the teacher's soft targets $p = \mathrm{softmax}(v/T)$ and the student's $q = \mathrm{softmax}(z/T)$, both at temperature $T$.
- **Hard loss** $L_{\text{hard}}$: ordinary cross-entropy against the true labels at $T=1$, given a *small* weight.

$$L = \alpha\, T^2\, L_{\text{soft}} + (1-\alpha)\, L_{\text{hard}}.$$

The $T^2$ factor is required because the soft-target gradient scales as $1/T^2$: with it the relative contribution of the soft and hard terms is invariant to $T$ while sweeping it.

**Matching logits is the high-$T$ limit.** Per transfer case, the soft cross-entropy gradient is
$$\frac{\partial C}{\partial z_i} = \frac{1}{T}(q_i - p_i).$$
For large $T$, using $\exp(x/T)\approx 1+x/T$ and zero-meaning each case's logits ($\sum_j z_j=\sum_j v_j=0$),
$$\frac{\partial C}{\partial z_i} \approx \frac{1}{N T^2}(z_i - v_i),$$
i.e. minimizing $\tfrac12\sum_i (z_i-v_i)^2$ — the squared logit-matching objective. At lower $T$ the approximation breaks for very negative logits, so distillation pays *less* attention to matching them; $T$ is a dial trading off whether to fit those nearly-unconstrained (possibly noisy) logits. Smaller students prefer intermediate $T$.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, hard_targets, T, alpha):
    # Soft cross-entropy at temperature T between teacher and student.
    p     = F.softmax(teacher_logits / T, dim=1)        # soft targets
    log_q = F.log_softmax(student_logits / T, dim=1)
    soft  = -(p * log_q).sum(dim=1).mean() * (T * T)    # rescale by T^2
    # Hard cross-entropy against true labels at T=1.
    hard  = F.cross_entropy(student_logits, hard_targets)
    return alpha * soft + (1.0 - alpha) * hard          # small (1-alpha) works best

# Training (teacher frozen; transfer set = train set or a separate set)
teacher.eval()
opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9)
T, alpha = 8.0, 0.9                                      # larger student -> higher T
for x, y in transfer_loader:
    with torch.no_grad():
        tz = teacher(x)
    sz = student(x)
    loss = distillation_loss(sz, tz, y, T, alpha)
    opt.zero_grad(); loss.backward(); opt.step()
# Deployment: student runs an ordinary T=1 softmax.
```

Soft targets transfer generalization the student never sees directly (e.g. invariance to translations absent from the transfer set), and even let a student recognize a class entirely omitted from its transfer set, since that class leaks probability mass into the soft targets of visually similar classes.
