The random forest came in at $0.9346$ on random, $0.8536$ on monotone, $0.9312$ on sparse — geometric mean $0.9057$, a clear gain over deep_dnf's $0.8532$. The random family jumped from $0.7605$ to $0.9346$ exactly as predicted: exact axis-aligned splits did not suffer the noisy-OR's union-of-errors, and mixed polarity cost the tree nothing. But the forest paid on monotone, which *dropped* to $0.8536$ — the worst number on the ladder so far. That is the $s=20$ effect: a wide target of 20 width-4 terms over 40 variables is a lot of conjunctions, and with only $\sqrt{40}\approx 6$ random variables searched per node, many trees never get the right variable at the right depth, so the union of paths under-covers the term set. The decorrelation that rescued random is hurting the wide monotone family, and even the forest's best number is well short of the near-perfect accuracy that 20000 examples of a width-4 concept ought to permit.

That gap sends me back to the model the differentiable approach abandoned in the first place: a plain feed-forward network trained by backpropagation — an **MLP**. The forest's limitation is structural: it bags *independent* trees with *axis-aligned* splits under a *fixed* $\sqrt{n}$-per-node feature rule, so on a wide DNF it cannot flexibly allocate capacity to cover all the terms. A fully-connected MLP has none of those constraints — every hidden unit sees every input, the units jointly learn a distributed re-coding, and with enough hidden units *some* recoding makes any Boolean target linearly separable at the output. The reason I did not start here is the legibility worry that drove deep_dnf: an MLP smears the learned function across thousands of real-valued weights with no procedure to read off which variables, which polarity. But this task does not score legibility — it scores held-out accuracy — and on that axis the MLP's lack of structural commitment is an asset: it is free to carve the input space however the data demand, with no $\sqrt{n}$-per-node bottleneck.

The bet rests on representability, so let me make it explicit. A single linear-threshold unit fires on one side of a hyperplane, so it can compute one DNF term directly: put weights $+1$ on the positive literals and $-1$ on the negated ones, and a bias at the right count threshold — for a width-4 positive term, weights $+1$ on those four variables and bias $-3.5$ makes the pre-activation positive only when all four are on. The output unit then OR's the term-units, firing if *any* hidden unit fires, which is again a single linear-threshold decision (sum the hidden activations, threshold at $\ge 1$). So a two-layer net with one hidden unit per DNF term represents the target *exactly*: the hidden layer holds the conjunctions, the output the disjunction. With 256 hidden units and at most 20 terms there is roughly a $12\times$ over-provisioning on the monotone family — ample slack for gradient descent to find *a* fitting configuration rather than needing the exact minimal one. The representational question is settled; the only open question is whether backpropagation *finds* such a configuration from random examples. And there is an honest subtlety: representability is not findability. A width-4 term is satisfied by only $1$ in $16$ uniform inputs, so each term's positive examples are a thin slice of every batch and the loss is dominated by the easy, frequently-firing structure — so whether 20 epochs concentrate enough gradient on the rarely-satisfied conjunctions before the loss flattens is exactly what makes the wide monotone family the interesting test.

The training recipe is the scaffold default, and each piece is load-bearing. The unit is ReLU, not a hard threshold: a step unit has zero gradient almost everywhere, so a gradient method has nothing to descend, whereas ReLU is piecewise-differentiable and lets a small weight change produce a measurable change in the loss. The objective is binary cross-entropy on the output logit, $\mathcal{L} = -\bigl[y\log\sigma(z) + (1-y)\log(1-\sigma(z))\bigr]$, whose gradient at the output is just the signed probability error $\sigma(z)-y$; backpropagation chains it through the network, each weight's sensitivity being the upstream sensitivity times the local ReLU slope, computed in one backward sweep that reuses the forward activations. The cross-entropy choice matters more than it looks here: squared error on a saturated sigmoid has a gradient that *vanishes* precisely when the model is confidently wrong, which would stall exactly on the rarely-satisfied terms I am most worried about, whereas cross-entropy keeps a strong gradient on confident mistakes so the optimizer keeps pushing on the hard terms. Two hidden layers of 256 give depth (compose features, not just one linear recoding) and width (slack to find a good configuration). AdamW adapts the per-parameter step and decouples weight decay ($10^{-4}$) as a clean L2 pull toward small weights — mild regularization against overfitting the 20000 points, which matters because, like the forest, the MLP sees a vanishing fraction of the cube. Twenty epochs at batch 256 and lr $10^{-3}$ is enough passes for the loss to settle without overtraining, and I threshold the output sigmoid at $0.5$.

This is the *default* scaffold fill — the cheapest edit on the ladder — and that is part of the finding: the generic neural baseline, with no DNF-specific machinery at all, is being asked to beat both the hand-shaped DNF net and the tree ensemble. I expect the MLP to clearly beat the forest on monotone ($0.8536$): the forest's failure was feature-subset under-coverage, and the MLP has no such bottleneck — allocating 20 of its 256 units to the 20 terms is trivially within reach. On random ($0.9346$) I expect flat-to-up: mixed polarity is as free for a hidden unit (negative weights) as for a tree split, and a narrow 10-term target fits nearly exactly. Sparse ($0.9312$) is the one place I might give ground: the MLP must *learn* that 48 of 60 variables are irrelevant rather than getting it for free the way a tree does, though weight decay should push those weights toward zero. If the MLP fails to beat the forest on *monotone* specifically — where its full-connectivity advantage should be largest — then my account of the forest's weakness is wrong; and if no family reaches near-perfect accuracy, the remaining lever is to stop fitting in one flat pass and start *boosting* the residual errors.

```python
# EDITABLE region of custom_strategy.py — step 3: MLP + AdamW + BCE (mlp)
def build_model(config: TaskConfig, seed: int):
    """2-hidden-layer MLP (256, 256) with ReLU."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    hidden = 256
    model = nn.Sequential(
        nn.Linear(config.n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    return model, {"kind": "mlp", "hidden": hidden}


def make_dataset(
    dnf: tuple[tuple[tuple[int, int], ...], ...],
    config: TaskConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform Boolean training set."""
    x = make_uniform_examples(config, config.train_size, seed)
    y = evaluate_dnf(x, dnf)
    return x, y


def fit_and_predict(
    model,
    info: dict,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: TaskConfig,
    seed: int,
) -> np.ndarray:
    """AdamW + BCE training, 20 epochs."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_x_t = torch.from_numpy(train_x).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().to(device)
    test_x_t = torch.from_numpy(test_x).float().to(device)

    epochs = 20
    batch_size = 256
    n = train_x_t.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(train_x_t[idx]).view(-1)
            loss = criterion(logits, train_y_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                tr_acc = (
                    ((torch.sigmoid(model(train_x_t)).view(-1) >= 0.5).long()
                     == train_y_t.long()).float().mean().item()
                )
            print(
                f"TRAIN_METRICS epoch={epoch + 1} loss={loss.item():.4f} train_acc={tr_acc:.4f}",
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(test_x_t)).view(-1) >= 0.5).long().cpu().numpy()
    return preds
```
