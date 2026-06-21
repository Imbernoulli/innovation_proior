The hidden target is a DNF — an OR of ANDs of literals over $n$ Boolean variables — and I get to design the learner. The scaffold default is a generic ReLU MLP that asks gradient descent to rediscover from scratch that Boolean structure even exists. My first move is the opposite bet: put the answer's *shape* into the model itself. If I build a network whose hypothesis class **is** relaxed DNF — soft conjunctions OR'd together — then I am not asking the optimizer to discover that conjunctions matter, only to find *which* conjunctions. That inductive-bias argument is strong enough that it should be the floor everything else is measured against: if encoding the exact target structure does not win, that is itself the finding.

I propose a differentiable DNF learner — a noisy-OR of soft conjunctions, which I implement as the module `NeuralDNF`. The whole architecture falls out of how I relax the connectives. I extend truth values from $\{0,1\}$ to $[0,1]$ and pick smooth surrogates from the product family that agree on the corners: $\mathrm{NOT}\,x = 1-x$, $x\,\mathrm{AND}\,y = x\cdot y$, and by De Morgan $x\,\mathrm{OR}\,y = 1-(1-x)(1-y)$. The choice of *multiplication* for AND is load-bearing. An additive perceptron has to count how many inputs are on and compare against a learned bias threshold — and that bias is exactly the part of a trained net you cannot read back as logic, and the part the optimizer chases when input statistics shift. A product sidesteps it entirely: $x_1 x_2 x_3$ is $1$ iff every factor is $1$ and $0$ the instant any factor is $0$, with no threshold and no bias anywhere. So a conjunction neuron is a product, and the disjunction over terms is the De Morgan dual, a noisy-OR.

The real design problem is that each conjunction should AND over *some unknown subset* of the variables, so the selection has to be learned differentiably. Putting a softmax over the $n$ inputs is wrong twice: a softmax picks essentially one variable, so selecting $w$ literals would need $w$ separate softmaxes and a width committed up front, and a softmax forced to concentrate over a long input vector converges painfully when $n$ is large. The right primitive is a per-variable, *independent* include/exclude decision, not a competition among variables. Because the target may use negative literals too — the random and monotone families differ exactly on this — I attach to each variable $i$ of term $j$ a **three-way categorical** (include-positive / include-negative / skip) via a softmax over three logits. With probabilities $\mathrm{pos}$, $\mathrm{neg}$, $\mathrm{skip}$ summing to $1$, the per-variable factor is

$$F_{ji} = \mathrm{pos}_{ji}\cdot x_i + \mathrm{neg}_{ji}\cdot(1-x_i) + \mathrm{skip}_{ji},$$

which is the identity ($=1$) when the variable is skipped, equals $x_i$ when included positively, and $1-x_i$ when included negatively — and a term can never select both $x_i$ and $\lnot x_i$ at once. The soft conjunction is the product $O^{\text{conj}}_j = \prod_i F_{ji}$, and the clause can be read straight off the memberships. A sigmoid gate $\sigma(g_j)$ multiplies each term so the disjunction can switch a term off, and the noisy-OR combines them as $p = 1 - \prod_j\bigl(1 - \sigma(g_j)\,O^{\text{conj}}_j\bigr)$, finally emitted as the logit $\log p - \log(1-p)$.

Three implementation facts decide whether this trains at all. First, **log-domain products**: a width-$w$ clause is a product of $n$ factors each $\le 1$, and over $n=60$ these underflow and the gradient vanishes, so I compute $\exp\bigl(\sum_i \log F_{ji}\bigr)$ with factors clamped to $[10^{-6},1]$ — valid because every factor lies in $[0,1]$. Second, **sparse initialization**: if most memberships start near $1$, every clause is a product of many sub-one terms, the output is crushed toward $0$, and the gradient dies before training begins, so I set the skip logit high ($\approx 4$) and the include logits low ($\approx -4$) so clauses start near the constant-$1$ function, and I set the per-term gate logits low ($\approx -3$) so terms start mostly off. Third, **over-provisioning and sparsity penalties**: I allocate $n_{\text{terms}} = \max(4s, 32)$ terms, gate each so the noisy-OR can prune, and add a one-sided literal-width penalty $(\text{usage} - w)_+^2$ plus a mean-gate penalty, pushing toward short clauses and few active terms — a crisp DNF rather than a mush of half-on memberships.

The convergence story is visible in the gradient: differentiating one membership gives $\partial O^{\text{conj}}/\partial m_i \propto (x_i - 1)$, since the product over the other factors is non-negative. To *raise* a membership I need a counterexample — a point where $x_i = 0$ yet the clause should fire — telling me $x_i$ is not a required literal. With 20000 uniform examples each width-4 conjunction's relevant counterexamples are abundant, so a single layer converges *if* it starts in the right neighborhood. That caveat is why I add the one piece this substrate allows that a generic differentiable-DNF method would not: a **data-driven warm start**. Pure gradient descent is slow to discover which width-$w$ conjunctions matter, so before training I mine candidate terms directly from the labels — I score each variable by the gap $|\mathbb{E}[x_i\mid y=1] - \mathbb{E}[x_i\mid y=0]|$, keep the top variables (more for the monotone family, a tight window around the junta size for sparse), enumerate width-$w$ combinations and polarity patterns over them, and score each candidate conjunction by its precision plus a small recall term, discarding any firing on fewer than four examples. The top distinct terms initialize the memberships at saturated logits, so the network *starts* near a plausible DNF and gradient descent only refines it. I train with AdamW + BCE on the noisy-OR logits, 30 epochs at batch 512, the two penalties at weight $10^{-4}$, and threshold at logit $0$.

I expect this floor to be strong on monotone and sparse — clean mean-gap signal, simple polarity decisions, the warm start recovering most terms — and weak on the mixed-polarity random family, where the mean-gap signal is muddier, the three-way decision is real, and the noisy-OR is unforgiving: it takes the *union*, so every slightly-imprecise mined term adds false-positive mass and OR-ing 32+ of them over-predicts $1$. If the random number lands well below the other two, the diagnosis is the aggregation, not the gradient refinement — and the comparison the next rungs exist to make is against a model that simply *splits on variables*, one conjunction per root-to-leaf path, for free.

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
