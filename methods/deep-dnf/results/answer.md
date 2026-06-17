# Neural Logic Network — the differentiable DNF

## Problem

Learn an unknown Boolean function `f: {0,1}^n → {0,1}` from labelled examples `(x, y)` by
gradient descent, with two simultaneous requirements: (1) it must fit, including functions
additive MLPs handle poorly (skewed-distribution DNF, large XOR/parity); (2) the trained
parameters must be *directly decodable* into an explicit logical formula — a DNF: an OR of
ANDs of literals — so a human can read off exactly which variables, in which polarity, the
function uses.

## Key idea

Build the network from *multiplicative* logic layers with no bias terms, using the product
relaxation of Boolean algebra over `[0,1]`:

```
NOT x = 1 - x        x AND y = x·y        x OR y = 1 - (1-x)(1-y)
```

AND is a product (true iff all factors are 1, with no threshold to tune), OR is the De
Morgan dual (a noisy-OR). To learn *which* variables enter a clause, attach to each input a
per-variable **membership** decision rather than a softmax-over-inputs selector — an
independent include/exclude (here include-positive / include-negated / skip) gate, so the
clause size need not be known in advance and the choice can be read off after training. The
parameters then *are* the formula.

**Conjunction layer** (neuron `j`, membership `m_i^{(j)} ∈ [0,1]`):

```
y_conj^{(j)} = Π_i ( 1 - m_i^{(j)} (1 - x_i) ),   m_i = σ(c·w_i), c > 1
```

`m_i → 0` ⇒ factor → 1 (variable dropped); `m_i → 1` ⇒ factor → `x_i` (literal required).

**Disjunction layer** (De Morgan / noisy-OR):

```
y_disj^{(j)} = 1 - Π_i ( 1 - m_i^{(j)} x_i )
```

**DNF model:** `DNF(x) = DISJ( CONJ(x) )` — a bank of soft conjunctions OR'd together. (CNF
is `CONJ(DISJ(x))`; a CNF+DNF block is `1 - (1-CNF)(1-DNF)`.)

Implementation essentials, each load-bearing:
- **Log-domain products** `exp(Σ log(ε + factor))` — the long products underflow and vanish
  the gradient otherwise; valid because all factors lie in `[0,1]`.
- **Sparse initialization** — start most memberships near 0 (factors ≈ 1) and only a few near
  1; a dense half-on start crushes every gradient through the product of many sub-1 terms.
- **Three-way categorical literal** (pos / neg / skip via softmax) so terms can use either
  polarity without ever selecting `x_i` and `¬x_i` together.
- **Per-term gate** `σ(g_j)` so a term can be switched off; over-provision terms and let the
  gates prune.
- **Sparsity penalties** — one-sided literal-width penalty `(usage − w)_+^2` and a mean-gate
  penalty — to push toward short, few-term (crisp DNF) solutions.

Convergence intuition: `∂y_conj/∂m_i ∝ (x_i − 1)`, so memberships are driven by
counterexamples (`x_i = 0` while the clause should fire); with a small learning rate and
counterexamples present in each batch, a single layer converges.

## Final form (code)

```python
import torch
from torch import nn


class NeuralDNF(nn.Module):
    """Differentiable DNF: soft conjunctions (membership-gated products of literals)
    aggregated by a noisy-OR. Trained parameters decode directly into the formula:
    each term's literals from its per-variable pos/neg/skip categorical, each term's
    presence from its gate."""

    def __init__(self, n_features: int, n_terms: int):
        super().__init__()
        self.literal_logits = nn.Parameter(torch.empty(n_terms, n_features, 3))  # pos/neg/skip
        self.term_logits = nn.Parameter(torch.empty(n_terms))                    # term gates
        self.out_bias = nn.Parameter(torch.zeros(1))
        self.n_features, self.n_terms = n_features, n_terms
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.literal_logits[..., 0].fill_(-4.0)   # positive literal off
            self.literal_logits[..., 1].fill_(-4.0)   # negative literal off
            self.literal_logits[..., 2].fill_(4.0)    # skip on  -> sparse start
            self.literal_logits.add_(0.03 * torch.randn_like(self.literal_logits))
            self.term_logits.fill_(-3.0)
            self.out_bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.literal_logits, dim=-1)
        pos, neg, skip = probs[..., 0], probs[..., 1], probs[..., 2]
        literal = (
            pos.unsqueeze(0) * x.unsqueeze(1)
            + neg.unsqueeze(0) * (1.0 - x.unsqueeze(1))
            + skip.unsqueeze(0)
        ).clamp(min=1e-6, max=1.0)
        conj = torch.exp(torch.log(literal).sum(dim=-1)).clamp(max=1.0)      # AND, log domain
        term_prob = torch.sigmoid(self.term_logits).unsqueeze(0) * conj      # gate each term
        log_not = torch.log1p(-term_prob.clamp(max=1.0 - 1e-6)).sum(dim=-1)
        prob = (1.0 - torch.exp(log_not)).clamp(min=1e-6, max=1.0 - 1e-6)    # noisy-OR
        return torch.log(prob) - torch.log1p(-prob) + self.out_bias          # -> logit

    def literal_usage(self) -> torch.Tensor:
        probs = torch.softmax(self.literal_logits, dim=-1)
        return (probs[..., 0] + probs[..., 1]).sum(dim=-1)                    # expected #literals/term


def fit_and_predict(model, train_x, train_y, test_x, term_width, seed,
                    lr=2e-3, epochs=30, batch_size=512):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    n = train_x.shape[0]
    target_width = float(term_width)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(train_x[idx]).view(-1)
            loss = criterion(logits, train_y[idx])
            usage = model.literal_usage()
            width_penalty = ((usage - target_width).clamp(min=0.0) ** 2).mean()   # short terms
            gate_penalty = torch.sigmoid(model.term_logits).mean()                # few terms
            total = loss + 1e-4 * width_penalty + 1e-4 * gate_penalty
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return (model(test_x).view(-1) >= 0.0).long()
```

The number of conjunction neurons is over-provisioned (e.g. `n_terms = max(4·s, 32)` for an
`s`-term target); the term gates and sparsity penalties prune the surplus. After training,
the formula is read directly from the parameters: term `j`'s literals are the variables whose
`pos`/`neg` mass dominates `skip`, and a term is present when `σ(g_j)` is large.
