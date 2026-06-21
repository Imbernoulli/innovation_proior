The differentiable DNF landed at $0.7605$ on the random family, $0.9088$ on monotone, $0.8986$ on sparse — a geometric mean of $0.8532$, and that is the weakest thing on the ladder. The model with the *exact target structure baked in* failed precisely where I worried: the noisy-OR over 32+ soft conjunctions takes the *union*, so every slightly-imprecise mined term adds false-positive mass and the mixed-polarity random number bleeds toward the base rate. The lesson is blunt: hand-shaping the hypothesis class into relaxed DNF, trained from uniform examples, does not beat a model that simply carves the input space by variable. So I stop relaxing the logic and use a model that does exact, axis-aligned conjunctive splits natively.

I propose a **random forest**. The structural fact that makes a tree the natural learner here is the very one the noisy-OR was fumbling: a decision tree splits on one variable at a time, so a root-to-leaf path of length $w$ is a conjunction of $w$ literals — *exactly* a DNF term. The set of leaves labelled $1$ is therefore a DNF, and a deep enough tree represents the target DNF *exactly*, with no relaxation and no threshold to tune. Mixed polarity is free, because a split on $x_i$ tests $x_i = 0$ versus $x_i = 1$ symmetrically. The catch is the one every tree has: a single unrestricted tree grown until its leaves are pure will memorize the 20000 training points perfectly but is a high-variance estimator. Small changes in the sample move the early splits, and since every later split is conditioned on the earlier ones, an early perturbation cascades into a wholly different tree. With 20000 uniform examples covering a $2^{30}$-to-$2^{60}$ Boolean cube extremely sparsely, the bottom leaves are supported by a handful of points each and their labels are essentially guesses. The diagnosis from deep_dnf carries over — I have a *variance* problem, not a representation problem — and the fix is the same shape as averaging many soft conjunctions, except I average many *exact* trees.

What makes the averaging work is precise, and it is worth writing out because it dictates every hyperparameter. Take a predictor that is the average of $B$ trees, each grown on a resampled version of the data. If the trees were independent with variance $\sigma^2$, the average would have variance $\sigma^2/B$, vanishing as I add trees. Real trees on bootstrap resamples of the same data are *not* independent — they share most of the data, so their predictions are positively correlated with some pairwise $\rho$, and the variance of the average is

$$\rho\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .$$

The second term goes to zero with more trees, but the first does not: the ensemble's variance floor is $\rho\sigma^2$, set by how correlated the trees are. So the game is twofold — keep growing trees, which drives the $(1-\rho)\sigma^2/B$ term down for free, and, more importantly, drive *down the correlation* $\rho$. Plain bagging — bootstrap aggregation — lowers $\rho$ only a little, because a few strong variables dominate the top splits of *every* tree: if one variable is the single most informative split, nearly every bootstrap tree puts it at the root and the trees look alike near the top, where it matters most.

The decisive move that turns bagging into a random forest is to decorrelate by *also* randomizing the splits. At each node, instead of searching all $n$ variables for the best split, restrict the search to a fresh random subset of $m \ll n$ variables — sklearn's default $\sqrt{n}$ for classification, about 5 of 30, 6 of 40, 8 of 60. Now no single strong variable can dominate every tree's root, because in many trees it simply is not among the candidates at the top node; different trees are forced to build out of different variables, their errors decorrelate, $\rho$ drops, and the variance floor $\rho\sigma^2$ drops with it. This matters on this task specifically: the target DNF has $s$ terms over $w$ variables each, so there are several genuinely informative variables, and feature subsampling spreads the trees across them so the ensemble sees the whole term structure rather than fixating on the few variables with the largest marginal signal. Crucially the variance reduction costs no bias the way pruning would — I grow each tree deep and unpruned so each individual tree is low-bias / high-variance, and rely entirely on the average to kill the variance. Deep, unpruned, decorrelated, averaged: that is the recipe.

In the scaffold this is an almost trivial fill, which is the point — I do not need a bespoke network, I need the right off-the-shelf estimator. `build_model` returns a `RandomForestClassifier` with `n_estimators=200` (enough trees to push the $(1-\rho)\sigma^2/B$ term well below the $\rho\sigma^2$ floor, so more buys little), `max_depth=None` (grow until pure — low bias, counting on the average for variance), and `min_samples_leaf=2` (a light touch against single-point leaves, which on this sparse cube are pure noise; requiring two points trims the most overfit twigs without imposing real bias). I expect a large gain on random over deep_dnf's $0.7605$ — the exact axis-aligned splits do not suffer the union-of-errors blowup. Monotone should be roughly flat or slightly down: a wide 20-term target over 40 variables is many conjunctions to cover, and with only $\sqrt{40}\approx 6$ variables searched per node some terms may be under-covered. Sparse is the wild card — with only 12 of 60 variables relevant a random subset of $\sim 8$ often contains *zero* relevant variables, wasting a split on noise — though irrelevant variables are at least never chosen as good splits. If the forest's *random* number fails to clear $0.76$, my whole diagnosis is wrong; otherwise this vindicates the switch from relaxed logic to exact splits.

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
