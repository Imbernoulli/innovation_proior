**Problem.** Learn a hidden DNF concept from uniform random examples and predict it on a held-out
uniform test set, across three families (mixed-polarity, monotone, sparse junta). The first rung tests
the strongest inductive-bias bet: put the exact target shape — an OR of ANDs of literals — into the
model itself, as a differentiable relaxed-DNF network.

**Key idea.** Build the network from multiplicative logic with no bias terms, over `[0,1]`:
`AND = product`, `OR = noisy-OR = 1 − Π(1 − ·)`. Each conjunction selects its literals through a
*three-way categorical* per variable (include-positive / include-negative / skip) so it can use either
polarity but never both, and the clause is read straight off the memberships. Over-provision terms
(`n_terms = max(4·s, 32)`), gate each with `σ(g_j)` so the noisy-OR can prune, and train with AdamW +
BCE plus a one-sided literal-width penalty `(usage − w)_+^2` and a mean-gate penalty toward crisp DNF.

**Why it could win — and why it might not.** The hypothesis class *is* relaxed DNF, so gradient descent
only has to find *which* conjunctions, and a data-driven warm start (mine high-precision width-`w`
candidate terms from the variable-mean gaps and initialize the memberships at them) puts it near a
plausible formula before training. The risk is the random non-monotone family: a weaker mean-gap signal,
a real three-way decision per variable, and a noisy-OR that *accumulates* precision errors across 32+
soft conjunctions — union of slightly-wrong terms over-predicts 1.

**Implementation essentials.** Log-domain products (`exp(Σ log(factor))`, factors clamped to `[1e-6,1]`)
or the long products underflow and the gradient vanishes; sparse init (skip logit high, include logits
low, gates low) so clauses start near constant-1; 30 epochs, batch 512, lr 2e-3, penalty weights 1e-4.

**What to watch.** Expect strong monotone/sparse, weak random; the geometric mean is dragged by the
weakest family. If even the favorable families fall short of a generic tabular learner, the
inductive-bias bet has failed and the tree-ensemble baselines should overtake it.

```python
# EDITABLE region of custom_strategy.py — step 1: differentiable DNF (deep_dnf)
def build_model(config: TaskConfig, seed: int):
    """Neural Logic Network with soft literal selection and noisy-OR output."""
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class NeuralDNF(nn.Module):
        def __init__(self, n_features: int, n_terms: int):
            super().__init__()
            self.literal_logits = nn.Parameter(torch.empty(n_terms, n_features, 3))
            self.term_logits = nn.Parameter(torch.empty(n_terms))
            self.out_bias = nn.Parameter(torch.zeros(1))
            self.n_features = n_features
            self.n_terms = n_terms
            self.reset_parameters()

        def reset_parameters(self) -> None:
            with torch.no_grad():
                self.literal_logits[..., 0].fill_(-4.0)  # positive literal
                self.literal_logits[..., 1].fill_(-4.0)  # negative literal
                self.literal_logits[..., 2].fill_(4.0)   # skip literal
                self.literal_logits.add_(0.03 * torch.randn_like(self.literal_logits))
                self.term_logits.fill_(-3.0)
                self.out_bias.zero_()

        def initialize_terms(self, terms: list[tuple[np.ndarray, np.ndarray]]) -> None:
            self.reset_parameters()
            with torch.no_grad():
                for term_index, (variables, polarities) in enumerate(terms[: self.n_terms]):
                    self.term_logits[term_index] = 2.5
                    for var, pol in zip(variables, polarities):
                        self.literal_logits[term_index, int(var), :].fill_(-5.0)
                        self.literal_logits[term_index, int(var), 2] = -5.0
                        self.literal_logits[term_index, int(var), 0 if int(pol) == 1 else 1] = 5.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            probs = torch.softmax(self.literal_logits, dim=-1)
            pos = probs[..., 0]
            neg = probs[..., 1]
            skip = probs[..., 2]
            literal = (
                pos.unsqueeze(0) * x.unsqueeze(1)
                + neg.unsqueeze(0) * (1.0 - x.unsqueeze(1))
                + skip.unsqueeze(0)
            ).clamp(min=1e-6, max=1.0)
            conj = torch.exp(torch.log(literal).sum(dim=-1)).clamp(max=1.0)
            term_prob = torch.sigmoid(self.term_logits).unsqueeze(0) * conj
            log_not = torch.log1p(-term_prob.clamp(max=1.0 - 1e-6)).sum(dim=-1)
            prob = (1.0 - torch.exp(log_not)).clamp(min=1e-6, max=1.0 - 1e-6)
            return torch.log(prob) - torch.log1p(-prob) + self.out_bias

        def literal_usage(self) -> torch.Tensor:
            probs = torch.softmax(self.literal_logits, dim=-1)
            return (probs[..., 0] + probs[..., 1]).sum(dim=-1)

    n_terms = max(4 * config.num_terms, 32)
    model = NeuralDNF(config.n_features, n_terms)
    return model, {"kind": "neural_dnf", "n_terms": n_terms}


def make_dataset(
    dnf: tuple[tuple[tuple[int, int], ...], ...],
    config: TaskConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform Boolean training set."""
    x = make_uniform_examples(config, config.train_size, seed)
    y = evaluate_dnf(x, dnf)
    return x, y


def _mine_candidate_terms(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: TaskConfig,
    max_terms: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Data-driven initialization for a soft DNF layer."""
    from itertools import combinations, product

    rng = np.random.default_rng(seed)
    x = train_x.astype(np.int8, copy=False)
    y = train_y.astype(bool, copy=False)
    if y.sum() == 0 or (~y).sum() == 0:
        return []

    pos_mean = x[y].mean(axis=0)
    neg_mean = x[~y].mean(axis=0)
    variable_score = np.abs(pos_mean - neg_mean)
    if config.monotone:
        top_m = min(config.n_features, 30)
    elif config.sparse_subset > 0:
        top_m = min(config.n_features, max(config.sparse_subset + 4, 16))
    else:
        top_m = min(config.n_features, 18)
    top_vars = np.argsort(variable_score)[-top_m:]

    combos = np.array(list(combinations(top_vars.tolist(), config.term_width)), dtype=np.int64)
    if combos.size == 0:
        return []
    if config.monotone:
        patterns = np.ones((1, config.term_width), dtype=np.int8)
    else:
        patterns = np.array(list(product([0, 1], repeat=config.term_width)), dtype=np.int8)
    vars_arr = np.repeat(combos, len(patterns), axis=0)
    pols_arr = np.tile(patterns, (len(combos), 1))

    max_candidates = 36000
    if len(vars_arr) > max_candidates:
        keep = rng.choice(len(vars_arr), size=max_candidates, replace=False)
        vars_arr = vars_arr[keep]
        pols_arr = pols_arr[keep]

    scores = np.empty(len(vars_arr), dtype=np.float64)
    chunk = 512
    pos_count = max(int(y.sum()), 1)
    for start in range(0, len(vars_arr), chunk):
        end = min(start + chunk, len(vars_arr))
        v = vars_arr[start:end]
        p = pols_arr[start:end]
        match = np.ones((x.shape[0], end - start), dtype=bool)
        for j in range(config.term_width):
            match &= x[:, v[:, j]] == p[:, j]
        covered = match.sum(axis=0)
        pos_covered = match[y].sum(axis=0)
        precision = pos_covered / np.maximum(covered, 1)
        recall = pos_covered / pos_count
        chunk_scores = precision + 0.25 * recall
        chunk_scores[covered < 4] = -1.0
        scores[start:end] = chunk_scores

    order = np.argsort(scores)[::-1]
    selected: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for idx in order:
        if scores[idx] <= 0:
            break
        term_key = tuple(sorted((int(v), int(p)) for v, p in zip(vars_arr[idx], pols_arr[idx])))
        if term_key in seen:
            continue
        seen.add(term_key)
        selected.append((vars_arr[idx].copy(), pols_arr[idx].copy()))
        if len(selected) >= max_terms:
            break
    return selected


def fit_and_predict(
    model,
    info: dict,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: TaskConfig,
    seed: int,
) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    terms = _mine_candidate_terms(train_x, train_y, config, info.get("n_terms", 32), seed)
    if hasattr(model, "initialize_terms"):
        model.initialize_terms(terms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    train_x_t = torch.from_numpy(train_x).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().to(device)
    test_x_t = torch.from_numpy(test_x).float().to(device)

    epochs = 30
    batch_size = 512
    n = train_x_t.shape[0]
    target_width = float(config.term_width)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(train_x_t[idx]).view(-1)
            loss = criterion(logits, train_y_t[idx])
            usage = model.literal_usage()
            width_penalty = ((usage - target_width).clamp(min=0.0) ** 2).mean()
            gate_penalty = torch.sigmoid(model.term_logits).mean()
            total = loss + 1e-4 * width_penalty + 1e-4 * gate_penalty
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                pred = (model(train_x_t).view(-1) >= 0.0).long()
                tr_acc = (pred == train_y_t.long()).float().mean().item()
            print(
                f"TRAIN_METRICS epoch={epoch + 1} loss={loss.item():.4f} "
                f"terms={len(terms)} train_acc={tr_acc:.4f}",
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        preds = (model(test_x_t).view(-1) >= 0.0).long().cpu().numpy()
    return preds
```
