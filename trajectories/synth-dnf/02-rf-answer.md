**Problem.** The differentiable DNF was the weakest rung (geomean 0.8532), sunk on the mixed-polarity
random family (0.7605) by a noisy-OR that accumulates precision errors across many soft conjunctions.
The fix is to abandon the relaxation and learn with a model that does *exact*, axis-aligned conjunctive
splits — a decision tree — averaged to kill its variance.

**Key idea.** A decision tree's root-to-leaf path of length `w` is exactly a width-`w` DNF term, so a
deep tree represents the target DNF with no relaxation. One deep tree is high-variance; average many.
The ensemble's variance floor is `ρσ²` (correlation between trees), so the decisive move beyond plain
bagging is to *decorrelate*: at each node search only a random subset of `sqrt(n)` variables for the
split, so no single strong variable dominates every tree's top splits, `ρ` drops, and the average's
variance drops with it. Grow each tree deep and unpruned (low bias) and rely entirely on the average.

**Why it should beat deep_dnf.** Exact axis-aligned splits do not suffer the union-of-errors blowup;
mixed polarity is free (a split tests `x_i = 0` or `x_i = 1` symmetrically), so the random family should
jump well above 0.76. Monotone should be roughly flat (deep_dnf's warm start genuinely helps there);
sparse is the wild card — with only 12 of 60 variables relevant, a random split-subset often contains no
relevant variable, wasting splits on noise.

**Hyperparameters.** `RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2,
n_jobs=-1, random_state=seed)`. 200 trees push the `(1−ρ)σ²/B` term below the `ρσ²` floor; full depth
for low bias; `min_samples_leaf=2` trims single-point (pure-noise) leaves on the sparse cube.

```python
# EDITABLE region of custom_strategy.py — step 2: Random Forest (rf)
def build_model(config: TaskConfig, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=int(seed),
    )
    return model, {"kind": "random_forest"}


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
    model.fit(train_x.astype(np.float32), train_y.astype(np.int32))
    train_acc = float((model.predict(train_x.astype(np.float32)) == train_y).mean())
    print(
        f"TRAIN_METRICS rf n_trees={len(model.estimators_)} train_acc={train_acc:.4f}",
        flush=True,
    )
    preds = model.predict(test_x.astype(np.float32)).astype(np.int64)
    return preds
```
