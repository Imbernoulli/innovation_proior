**Problem.** The random forest (geomean 0.9057) beat deep_dnf but has a structural ceiling on wide
targets — it dropped to 0.8536 on the 20-term monotone family because random `sqrt(n)`-per-node feature
subsets under-cover so many conjunctions. The fix is the generic learner deep_dnf abandoned: a
fully-connected MLP trained by backprop, which has no per-node feature bottleneck and can allocate
capacity to all terms at once.

**Key idea.** A two-layer net represents any DNF exactly — one hidden ReLU unit per term (weights ±1 on
its literals, bias at the count threshold) computes a conjunction; the output unit OR's the hidden
activations. With 256 hidden units and ≤ 20 terms there is ample capacity; backpropagation finds a
fitting configuration from random examples. Full connectivity means every hidden unit sees every input,
so wide DNF is no harder than narrow — exactly the forest's weak spot.

**Why it should beat rf.** Monotone (forest's 0.8536) should jump: no feature-subset under-coverage,
trivial to allocate 20 of 256 units to 20 terms. Random (forest's 0.9346) should be flat-to-up: mixed
polarity is free via negative weights, narrow target fits nearly exactly. Sparse (forest's 0.9312) is
the risk: the MLP must *learn* that 48 of 60 variables are irrelevant rather than getting it free; weight
decay helps but it may trail there.

**Hyperparameters.** 2 hidden layers of 256, ReLU; AdamW lr 1e-3, weight_decay 1e-4; BCEWithLogits; 20
epochs, batch 256; sigmoid-threshold 0.5. (This is the scaffold default fill — the cheapest edit on the
ladder, which is part of the finding.)

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
