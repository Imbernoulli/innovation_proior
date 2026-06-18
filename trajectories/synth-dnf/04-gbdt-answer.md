**Problem.** The wide 20-term monotone family is the bottleneck — it defeated deep_dnf, the random forest
(0.8536), and the MLP (0.8461) alike, because every *flat* learner (bag-of-trees, one-shot gradient
descent) leaves the same residual structure unmodeled. The fix is a learner that builds the target
*sequentially*, each new component correcting the ensemble's current errors.

**Key idea.** Gradient boosting: an additive model of small regression trees, fit forward-stagewise. At
each round, compute the negative-gradient pseudo-responses of the binomial deviance at the current
model (`2y/(1+e^{2yF})` — large where confidently wrong), fit the next tree to them by least squares,
and choose the step by a 1-D line search on the true deviance. This keeps the tree's exact
conjunctive splits (one root-to-leaf path per DNF term — what beat the MLP on the random family) and
adds the residual error-correction the random forest and MLP both lack.

**Why it should win.** Monotone (standing worst, 0.8461) should finally crack: 20 terms fit one residual
at a time, a depth-5 tree isolates each term, 500 shrunk rounds with early stopping run long on the hard
family. Random and sparse should hold near their ceilings — exact splits handle mixed polarity for free
and never split on irrelevant variables.

**Hyperparameters (task-tuned).** `n_estimators=500`, `learning_rate=0.05` (shrinkage + many rounds);
`max_depth=max(4, term_width+1)=5` — depth tied to the conjunction width so each tree can isolate one
width-4 term while staying a weak learner; `subsample=0.9` (stochastic boosting); `min_samples_leaf=2`;
`n_iter_no_change=25`, `validation_fraction=0.1`, `tol=1e-5` (early stop — don't overfit the easy
families, keep boosting the hard one).

```python
# EDITABLE region of custom_strategy.py — step 4: Gradient-Boosted Decision Trees (gbdt)
def build_model(config: TaskConfig, seed: int):
    """sklearn GradientBoostingClassifier."""
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=max(4, config.term_width + 1),
        min_samples_leaf=2,
        subsample=0.9,
        n_iter_no_change=25,
        validation_fraction=0.1,
        tol=1e-5,
        random_state=int(seed),
    )
    return model, {"kind": "gbdt"}


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
        f"TRAIN_METRICS gbdt n_estimators={len(model.estimators_)} train_acc={train_acc:.4f}",
        flush=True,
    )
    preds = model.predict(test_x.astype(np.float32)).astype(np.int64)
    return preds
```
