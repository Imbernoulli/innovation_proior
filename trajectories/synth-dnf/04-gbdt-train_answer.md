The MLP closed out the random family at $0.9989$ and held sparse at $0.9876$, lifting the geometric mean to $0.9415$ — the best on the ladder — but it dropped the wide monotone family to $0.8461$, below the random forest ($0.8536$) and below even deep_dnf's best showing on that family. That is the one number that has refused to move. The 20-term monotone DNF has now defeated, in turn, the differentiable DNF net, the bagged forest, and the fully-connected MLP, and the pattern across those three failures is what finally tells me what is wrong. It is not a feature-subset bottleneck — the MLP has full connectivity and still failed — it is that every learner I have tried fits the target in *one flat pass*: the forest averages independent trees that never see each other's mistakes, and the MLP minimizes one fixed loss in a fixed number of epochs. On a target with 20 overlapping conjunctions competing for a fixed budget of fitting effort, a handful of rarely-firing terms gets under-modeled and there is no mechanism to go back and put more attention on the still-wrong examples. So I need a learner that builds the target up *sequentially*, with each new component explicitly correcting the errors the ensemble has made so far.

I propose **Gradient-Boosted Decision Trees (gbdt)**. The model is additive, $F(x) = \sum_m \rho_m\, h(x; a_m)$, where each $h$ is a small regression tree, fit forward-stagewise: hold $F_{m-1}$ fixed and add exactly one tree per round. The reason to choose *gradient* boosting over the older loss-specific residual schemes is what makes it both general and stable here. For squared error the stage subproblem is just "fit the next tree to the residual $y - F_{m-1}$" — the classic residual loop — but for classification I want the binomial deviance $\log(1 + e^{-2yF})$, whose stage subproblem has no closed form. Friedman's (2001) move rescues it: treat the function values at the training points as free parameters and take the negative gradient of the loss with respect to them, giving a vector of *pseudo-responses*, one per point, defined for any differentiable loss. That negative gradient lives only at the data, so it cannot itself be a model; instead I fit the next tree to it by least squares — the tree most parallel to the negative gradient over the training set — and then pick the step size by a one-dimensional line search on the *true* deviance. Cheap least squares to find a generalizable descent direction, then an honest scalar step on the real loss.

For the binomial deviance the pseudo-response at each point works out to
$$ \tilde y_i = \frac{2 y_i}{1 + e^{\,2 y_i F_{m-1}(x_i)}}, $$
which is large where the current model is confidently wrong and near zero where it is already right. So each new tree is fit literally to *where the ensemble is still making mistakes*. This is exactly the error-correction the monotone family has been missing: a misclassified slice from one of the 20 terms produces a large pseudo-response, the next depth-5 tree carves that region out, and the deviance comes down term by term. The contrast with the two flat learners is the whole argument. The forest fits every tree to the *same* labels in parallel — if its random feature subsets cause three of the twenty terms to be under-modeled, no tree is ever told "these points are still wrong," and the misclassified slice just sits there, averaged over. The MLP minimizes one loss in a fixed number of passes — a rarely-satisfied term contributes a thin sliver of gradient that the frequently-firing structure drowns out, so the optimizer plateaus with those terms half-learned. Boosting breaks both modes by construction, because the pseudo-response *re-weights* attention onto the still-wrong points every round, turning the slice the forest averaged away into the largest signal driving the next tree. Crucially, boosting keeps the property that let trees beat the MLP on the random family in the first place: a decision tree splits on one variable at a time, so a root-to-leaf path of length $w$ is exactly a width-$w$ conjunction — one DNF term — and mixed polarity is handled for free.

Three choices in the recipe are tuned to *this* task. First, **tree depth tied to the target width**: each tree is grown to depth $\max(4, \texttt{term\_width}+1) = 5$ here. A width-4 term is a length-4 conjunction, so a depth-4-to-5 tree can isolate a single term per path with one level of slack — deep enough to represent a term exactly, shallow enough that each tree stays a *weak* learner and the boosting does the composition. The learner reads `config.term_width` to set this, adapting its weak-learner capacity to the announced conjunction width without ever inspecting the hidden term list. Second, **shrinkage with many rounds**: a small learning rate of $0.05$ scales down each tree's contribution so no single tree overshoots, and $\texttt{n\_estimators}=500$ gives enough small steps to drive the deviance down — many shrunk steps generalize better than a few large ones, trading compute for accuracy. Third, **stochastic subsampling and early stopping**: $\texttt{subsample}=0.9$ fits each tree on a random 90% of the data, which decorrelates the trees and regularizes, while $\texttt{n\_iter\_no\_change}=25$ with a 10% internal validation split and $\texttt{tol}=10^{-5}$ halts the boosting once the held-out deviance stops improving. That asymmetry is exactly what I want: on the easy families (random, sparse) it stops early and does not overfit, while on the hard monotone family it keeps boosting as long as residual structure is still being reduced. The starting model $F_0$ is the best constant (the base-rate logit), and the final prediction thresholds the additive logit at zero. The scaffold fill is clean: `build_model` returns the `sklearn` `GradientBoostingClassifier` with those settings, `make_dataset` is the default uniform sample, and `fit_and_predict` calls `.fit` and `.predict` and returns the 0/1 vector. My expectation is that this is the rung that finally cracks monotone out of the mid-$0.84$s while random and sparse hold at their ceilings — and if it does not substantially beat $0.8461$ there, then the whole "flat-fitting" account is wrong and the wide target would have to be intrinsically hard to learn from 20000 examples regardless of algorithm.

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
