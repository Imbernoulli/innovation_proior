The problem is supervised learning of Boolean functions over binary inputs. The target is a disjunctive normal formula — a disjunction of conjunctions of literals — and the goal is a model that is simultaneously trainable by ordinary gradient descent and interpretable as a symbolic rule after training. Standard additive MLPs fail the second requirement because a threshold plus a learned bias can implement a conjunction only up to arbitrary scaling; once training finishes there is no clean way to read off which variables belong to which clause. The same real-valued weights might represent the same Boolean behavior with shifted or scaled parameters, so a human or downstream system must impose an external thresholding convention that is not part of the learned object. Decision trees and ensembles can represent DNF exactly, but they give up the compact parametric form and cannot be tuned end-to-end with backprop in the same way as a neural network. What is needed is a differentiable architecture whose parameters are themselves the membership decisions of a DNF.

The proposed method is deep-dnf, a neural logic network that implements a differentiable DNF. The idea is to abandon additive thresholds for Boolean operations and instead use product logic on soft truth values in [0,1]. Conjunction is multiplication, so a clause is satisfied only when every selected literal is true. Disjunction is the De Morgan dual, 1 minus the product of one minus the selected inputs, which is the noisy-OR form. Each neuron learns one scalar membership per input through a sharpened sigmoid or a soft categorical selector. For a conjunction, the factor for input i is 1 - m_i(1 - x_i): if m_i is near zero the variable is skipped and the factor contributes 1, and if m_i is near one the factor becomes x_i and the variable is required. For a disjunction, the contribution is m_i x_i, so the OR fires when any selected input is true. A DNF is built by composing a bank of conjunction neurons followed by one disjunction neuron, and the reverse composition gives a CNF.

Training proceeds with ordinary backpropagation and a cross-entropy loss. The gradient for a conjunction membership is nonzero only on examples where the corresponding input is 0, which is exactly when that literal would suppress an overly broad clause; the sign of the loss decides whether the membership should move up or down. The membership logits are initialized sparsely so most clauses start inactive, which avoids the gradient-collapse problem caused by long products of small factors. For numerical stability the long products are evaluated in log space as exp(sum log(factor)), and factors are clamped away from zero. For mixed-polarity rules the input vocabulary is augmented with negated atoms, and the same scalar selector can include either polarity. Because the membership parameters are soft during training but converge toward zero or one, the final formula is recovered by thresholding the memberships and printing the corresponding literals and clauses. Any literal whose removal does not change the loss can be pruned for a cleaner rule. The method is therefore a fully differentiable DNF learner that retains a readable symbolic description.

```python
import torch
from torch import nn
import numpy as np


class DeepDNF(nn.Module):
    """Differentiable DNF with soft literal selection and noisy-OR output."""

    def __init__(self, n_features: int, n_terms: int, term_width: int | None = None):
        super().__init__()
        self.n_features = n_features
        self.n_terms = n_terms
        self.term_width = term_width
        # Three-way choice per literal: positive / negative / skip
        self.literal_logits = nn.Parameter(torch.empty(n_terms, n_features, 3))
        # Gate for each candidate term in the final noisy-OR
        self.term_logits = nn.Parameter(torch.empty(n_terms))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # Sparse init: strongly prefer skipping a literal and disabling a term
            self.literal_logits[..., 0].fill_(-4.0)   # positive literal
            self.literal_logits[..., 1].fill_(-4.0)   # negative literal
            self.literal_logits[..., 2].fill_(4.0)    # skip literal
            self.literal_logits.add_(0.03 * torch.randn_like(self.literal_logits))
            self.term_logits.fill_(-3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.literal_logits, dim=-1)
        pos = probs[..., 0]
        neg = probs[..., 1]
        skip = probs[..., 2]
        # For each literal: selected-positive contributes x, selected-negative contributes 1-x,
        # skipped contributes 1.
        literal = (
            pos.unsqueeze(0) * x.unsqueeze(1)
            + neg.unsqueeze(0) * (1.0 - x.unsqueeze(1))
            + skip.unsqueeze(0)
        ).clamp(min=1e-6, max=1.0)
        # Conjunction = product over literals (log-domain for numerical stability)
        conj = torch.exp(torch.log(literal).sum(dim=-1)).clamp(max=1.0)
        # Noisy-OR disjunction over the gated terms
        term_prob = torch.sigmoid(self.term_logits).unsqueeze(0) * conj
        log_not = torch.log1p(-term_prob.clamp(max=1.0 - 1e-6)).sum(dim=-1)
        prob = (1.0 - torch.exp(log_not)).clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.log(prob) - torch.log1p(-prob)

    def literal_usage(self) -> torch.Tensor:
        probs = torch.softmax(self.literal_logits, dim=-1)
        return (probs[..., 0] + probs[..., 1]).sum(dim=-1)


def build_model(n_features: int, n_terms: int, seed: int = 0) -> DeepDNF:
    torch.manual_seed(seed)
    return DeepDNF(n_features, n_terms)


def fit_dnf(model: nn.Module, train_x: np.ndarray, train_y: np.ndarray,
            epochs: int = 30, batch_size: int = 512, lr: float = 2e-3,
            target_width: float | None = None) -> DeepDNF:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    tx = torch.from_numpy(train_x).float().to(device)
    ty = torch.from_numpy(train_y).float().to(device)
    n = tx.shape[0]
    if target_width is None:
        target_width = 1.0

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(tx[idx]).view(-1)
            loss = criterion(logits, ty[idx])
            usage = model.literal_usage()
            width_penalty = ((usage - target_width).clamp(min=0.0) ** 2).mean()
            gate_penalty = torch.sigmoid(model.term_logits).mean()
            total = loss + 1e-4 * width_penalty + 1e-4 * gate_penalty
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
    return model


def extract_formula(model: DeepDNF, threshold: float = 0.5) -> list[list[tuple[int, int]]]:
    """Return a list of clauses; each clause is a list of (variable, polarity) pairs."""
    with torch.no_grad():
        probs = torch.softmax(model.literal_logits, dim=-1)
        gates = torch.sigmoid(model.term_logits)
        active_terms = (gates >= threshold).nonzero(as_tuple=True)[0].tolist()
        clauses = []
        for t in active_terms:
            clause = []
            for v in range(model.n_features):
                p_pos = probs[t, v, 0].item()
                p_neg = probs[t, v, 1].item()
                if p_pos >= threshold:
                    clause.append((v, 1))
                elif p_neg >= threshold:
                    clause.append((v, 0))
            clauses.append(clause)
        return clauses
```
