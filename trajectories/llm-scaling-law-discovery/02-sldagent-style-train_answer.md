The kernel-ridge rung told me, in numbers, exactly what a model with no scaling-law prior buys off the training hull: nothing usable. Every family came back with a *negative* held-out $R^2$ — vocab $-0.567$, dataconstrained $-13.4$, and lrbsz a catastrophic $-413.7$ with `NMAE` $16.2$, meaning the held-out predictions were sixteen times the target's own spread away from truth. That is the pure-locality failure I expected: the RBF similarity decays to zero as a query moves larger and denser than the fit, so the prediction collapses with no power-law tail and no floor. Vocab — the most saturated family, closest to a clean additive power law — was the *only* near-miss, which sharpens the diagnosis: where the test region is nearly inside the hull the black box is merely bad, and where it is far outside it is hopeless. The missing ingredient is the *right asymptotic form*, not more flexibility, so I stop being model-free and impose the power-law-plus-floor structure that extrapolates by construction.

I propose a set of **discovered-style symbolic laws** — a compact expression per family with the Chinchilla bones $E+A/N^\alpha+\dots$, fit per group by nonlinear least squares — but with the *cross-axis interaction* terms the textbook additive law drops. The reason not to transcribe the additive law directly is exactly what the kernel-ridge numbers hint: a sum of monotone decaying power terms is monotone in each axis and decoupled across axes, so it literally cannot represent a *basin* (an interior optimum) or a *cross-axis coupling* — and lrbsz, the disaster, is precisely a basin in $(l,b)$ where the two axes interact, while vocab, the near-miss, is where the axes plausibly act independently. So the move is to keep the additive backbone where it works and add interaction structure where each family needs it.

For **vocab**, kernel ridge's near-miss says the additive backbone already explains most of the structure, so I keep it but suspect $V$ and $D$ are not independent: a larger vocabulary changes how much each training character teaches the model, so the *value of data* should depend on the *vocabulary*. I write the law as a single joint multiplicative power across all three axes plus an explicit $V\times D$ cross term plus a floor,

$$L = E + A\,N^{-a_1}V^{-a_2}D^{-a_3} + A_{vd}\,V^{-g_1}D^{-g_2},$$

where the multiplicative term is the geometric-mean reading of scaling (loss falls as a *product* of per-axis powers) and the cross term picks up whatever residual the product misses specifically on the $(V,D)$ pair. The floor $E$ is left *unconstrained* (not exponentiated) because the unigram-normalised target can be negative, so I refuse to force the additive constant positive; the scale and exponents are exponentiated so the fitter explores a well-conditioned positive region. And because the target can be negative I fit the residuals in the *linear* domain — a log residual would be undefined on a negative target.

For **lrbsz**, the family I most need to fix, the physics is a basin: hold $N,D$ fixed and sweep the learning rate $l$, and loss falls then rises around an optimum $l^\star$; the same for batch size $b$; and $l^\star$ and $b^\star$ are *coupled*. A sum of monotone power terms cannot bend down then up, which is why every additive law lands hugely negative here. The natural representation of a basin is a *quadratic*, and the natural coordinates are logarithmic because the optima scale multiplicatively, so I work in $\Delta_x=\log l-\log l^\star$ and $\Delta_y=\log b-\log b^\star$, the log-distances from the fitted optima, and write a correlated quadratic bowl

$$\text{penalty} = k\,(\Delta_x^2 + \Delta_y^2 + 2\rho\,\Delta_x\Delta_y).$$

The $\Delta_x^2,\Delta_y^2$ terms make it cost to be off the optimum in either axis; the $\rho\,\Delta_x\Delta_y$ cross term tilts the bowl so the ridge of low loss runs diagonally in $(\log l,\log b)$ — exactly the lr/bsz coupling. I keep $\rho\in(-1,1)$ via $\rho=\tanh(\cdot)$ so the quadratic stays a genuine bowl rather than a saddle. Onto this basin I add a Chinchilla base $E+A\,N^{-\alpha}+B\,D^{-\beta}$ for the part of loss driven by scale rather than optimizer settings. Because this target is strictly positive I fit in the *log* domain (the homoscedastic residual for a multiplicative quantity) and give it the most restarts of the three — the basin's center and curvature are the hardest parameters to pin from data.

For **dataconstrained**, the defining fact is that the total tokens $D$ can exceed the unique pool $U$ because data is repeated, and a re-read token is worth less than a fresh one. The additive $B\,D^{-\beta}$ treats every token as fresh — it predicts the eleventh epoch helps as much as the first — which is exactly the asymptotic error that wrecks extrapolation to the denser test points. So I replace $D$ with an *effective* token count discounted by a multiplicative repeat-efficiency factor: define the repetition ratio $D/U$ and the efficiency multiplier $1/(1+(D/U)/R)$, near $1$ when data is barely repeated and decaying smoothly as repetition grows, with $R$ a learned repeat-budget constant; the effective tokens are $D_{\text{eff}}=D\cdot\text{efficiency}$ and the law is $E+A\,N^{-\alpha}+B\,D_{\text{eff}}^{-\beta}$. The asymptotics read correctly: at $D\approx U$ (single epoch) the multiplier is $\approx1$ and $D_{\text{eff}}\approx D$, recovering the fresh-data term; as $D/U$ grows the multiplier shrinks, $D_{\text{eff}}$ saturates, and additional repeated tokens stop driving loss down. This is the discovered-style counterpart to the human effective-token saturation — same physics, a different functional shape ($1/(1+\text{ratio}/R)$ rather than an exponential), which is the spirit of this rung. The target is strictly positive, so I fit in the log domain.

The fitting machinery is shared and is the one piece the scaffold hands me. For each `group` I fit the family's form by the provided `least_squares` with a soft-$\ell_1$ robust loss (so a few large-residual runs do not dominate), choosing the linear or log residual per family as above, with multi-start initializations (vocab and dataconstrained $8$ restarts, lrbsz $10$) keeping the best restart by mean-squared error. Positive quantities are carried as exponentials of free parameters; signed quantities ($E$ on vocab, $\rho$ on lrbsz) are left free or squashed. After fitting every group I store a median-of-groups parameter vector as the fallback for any unseen group — one shared expression per family, coefficients per group, the contract the task asks for. I expect vocab and dataconstrained to swing from negative to the low-$0.9$s, because both have a near-additive backbone the symbolic form captures and the effective-token term finally lets the law extrapolate to the denser test points. Lrbsz is the open risk: the basin's center $(l^\star,b^\star)$ is a single fitted constant, so if the held-out points sit where the *optimum itself drifts* with $N$ and $D$, one center cannot track it — I expect an enormous improvement over $-413.7$ in absolute error but the $R^2$ to *stay negative*, which would tell the next rung that the lrbsz optimum must be made an explicit function of scale.

```python
def _safe_log_residuals(pred, y):
    pred = np.clip(np.asarray(pred, dtype=float), EPS, None)
    y = np.clip(np.asarray(y, dtype=float), EPS, None)
    return np.log(pred) - np.log(y)


def _linear_residuals(pred, y):
    return np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)


def _fit_generic(X, y, init_u, unpack_fn, predict_fn, n_restarts=6,
                 use_log=True):
    init_u = np.asarray(init_u, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
    candidates = [init_u]
    for scale in np.linspace(0.05, 0.45, max(n_restarts - 1, 0)):
        candidates.append(init_u + rng.normal(scale=scale, size=init_u.shape))
    best_u, best_score = init_u, float("inf")
    for u0 in candidates:
        def residuals(u):
            try:
                pred = predict_fn(X, unpack_fn(u))
                resid = (_safe_log_residuals(pred, y) if use_log
                         else _linear_residuals(pred, y))
                return np.nan_to_num(resid, nan=1e3, posinf=1e3, neginf=-1e3)
            except Exception:
                return np.full_like(y, 1e3, dtype=float)
        try:
            result = least_squares(residuals, u0, method="trf",
                                   loss="soft_l1", f_scale=0.05, max_nfev=5000)
            u_opt = result.x
        except Exception:
            u_opt = np.asarray(u0, dtype=float)
        pred = predict_fn(X, unpack_fn(u_opt))
        if use_log:
            score = float(np.mean(_safe_log_residuals(pred, y) ** 2))
        else:
            score = float(np.mean((np.asarray(pred, dtype=float) - y) ** 2))
        if np.isfinite(score) and score < best_score:
            best_score, best_u = score, u_opt
    return unpack_fn(best_u)


# -------- sld-vocab: multiplicative interaction on log scales --------

def _vocab_sldagent_predict(X, params):
    n = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)
    v = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
    d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
    E, A, a1, a2, a3, A_vd, g1, g2 = params
    # Cross term links vocab and data.
    cross = A_vd * np.power(v, -g1) * np.power(d, -g2)
    return E + A * np.power(n, -a1) * np.power(v, -a2) * np.power(d, -a3) + cross


def _fit_vocab_sldagent(X, y):
    y = np.asarray(y, dtype=float)
    def unpack(u):
        E = u[0]
        A = np.exp(u[1])
        a1, a2, a3 = np.exp(u[2]), np.exp(u[3]), np.exp(u[4])
        A_vd = np.exp(u[5])
        g1, g2 = np.exp(u[6]), np.exp(u[7])
        return np.array([E, A, a1, a2, a3, A_vd, g1, g2], dtype=float)
    init = np.array([
        float(np.median(y)),
        np.log(max(abs(np.std(y)), 0.1)),
        np.log(0.1), np.log(0.2), np.log(0.2),
        np.log(max(abs(np.std(y)), 0.05)),
        np.log(0.3), np.log(0.3),
    ])
    return _fit_generic(X, y, init, unpack, _vocab_sldagent_predict,
                        n_restarts=8, use_log=False)


# -------- sld-lrbsz: Chinchilla base + joint (lr, bsz) coupling --------

def _lrbsz_sldagent_predict(X, params):
    lr = np.clip(np.asarray(X[:, 0], dtype=float), 1e-8, None)
    bsz = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
    d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
    n = np.clip(np.asarray(X[:, 3], dtype=float), 1.0, None)
    E, A, alpha, B, beta, k, log_lr_star, log_bsz_star, rho = params
    base = E + A * np.power(n, -alpha) + B * np.power(d, -beta)
    dx = np.log(lr) - log_lr_star
    dy = np.log(bsz) - log_bsz_star
    # Correlated quadratic bowl around (lr*, bsz*) with coupling rho.
    penalty = k * (dx * dx + dy * dy + 2.0 * rho * dx * dy)
    return base + penalty


def _fit_lrbsz_sldagent(X, y):
    y = np.asarray(y, dtype=float)
    lr = np.clip(np.asarray(X[:, 0], dtype=float), 1e-8, None)
    bsz = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
    def unpack(u):
        E = u[0]
        A = np.exp(u[1]); alpha = np.exp(u[2])
        B = np.exp(u[3]); beta = np.exp(u[4])
        k = np.exp(u[5])
        log_lr_star = u[6]; log_bsz_star = u[7]
        rho = np.tanh(u[8])  # keep in (-1, 1)
        return np.array([E, A, alpha, B, beta, k,
                         log_lr_star, log_bsz_star, rho], dtype=float)
    init = np.array([
        float(max(y.min() * 0.9, 0.1)),
        np.log(max(y.max() - y.min(), 0.1)), np.log(0.3),
        np.log(max(y.max() - y.min(), 0.1)), np.log(0.3),
        np.log(0.05),
        float(np.log(np.median(lr))), float(np.log(np.median(bsz))),
        0.0,
    ])
    return _fit_generic(X, y, init, unpack, _lrbsz_sldagent_predict,
                        n_restarts=10, use_log=True)


# -------- sld-dataconstrained: multiplicative repeat-efficiency term ---

def _dconstrained_sldagent_predict(X, params):
    u = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)
    n = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
    d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
    E, A, alpha, B, beta, R = params
    ratio = np.clip(d / u, 0.0, 200.0)
    # Repeat-efficiency: multiplier decays smoothly with repetition.
    efficiency = 1.0 / (1.0 + ratio / np.maximum(R, 1e-3))
    d_eff = np.maximum(d * efficiency, 1.0)
    return E + A * np.power(n, -alpha) + B * np.power(d_eff, -beta)


def _fit_dconstrained_sldagent(X, y):
    y = np.asarray(y, dtype=float)
    def unpack(u):
        E = np.exp(u[0])
        A = np.exp(u[1]); alpha = np.exp(u[2])
        B = np.exp(u[3]); beta = np.exp(u[4])
        R = np.exp(u[5])
        return np.array([E, A, alpha, B, beta, R], dtype=float)
    init = np.array([
        np.log(max(y.min() * 0.9, 0.1)),
        np.log(max(y.max() - y.min(), 0.1)), np.log(0.35),
        np.log(max(y.max() - y.min(), 0.1)), np.log(0.35),
        np.log(5.0),
    ])
    return _fit_generic(X, y, init, unpack, _dconstrained_sldagent_predict,
                        n_restarts=8, use_log=True)


def _sldagent_fit_params(benchmark_name, X, y):
    if benchmark_name == "sld-vocab":
        return _fit_vocab_sldagent(X, y)
    if benchmark_name == "sld-lrbsz":
        return _fit_lrbsz_sldagent(X, y)
    if benchmark_name == "sld-dataconstrained":
        return _fit_dconstrained_sldagent(X, y)
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


def _sldagent_predict_params(benchmark_name, X, params):
    if benchmark_name == "sld-vocab":
        return _vocab_sldagent_predict(X, params)
    if benchmark_name == "sld-lrbsz":
        return _lrbsz_sldagent_predict(X, params)
    if benchmark_name == "sld-dataconstrained":
        return _dconstrained_sldagent_predict(X, params)
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


class ScalingLawModel:
    """SLDAgent-style symbolic baseline for the harder SLDBench subsets.

    Uses discovered-style symbolic forms with cross-axis interactions:
    - vocab: additive power law with extra V*D cross term
    - lrbsz: Chinchilla base + correlated (lr, bsz) quadratic bowl
    - dataconstrained: multiplicative repeat-efficiency factor on D_eff
    """

    def __init__(self, benchmark_name, numeric_names=None, categorical_names=None):
        self.benchmark_name = benchmark_name
        self.numeric_names = list(numeric_names or [])
        self.categorical_names = list(categorical_names or [])
        self.group_params_ = {}
        self.default_params_ = None

    def fit(self, X_num, X_cat, y):
        X_num = np.asarray(X_num, dtype=float)
        y = np.asarray(y, dtype=float)
        labels = group_labels(X_cat)
        fitted = []
        for group in sorted(set(labels.tolist())):
            mask = labels == group
            params = _sldagent_fit_params(self.benchmark_name,
                                          X_num[mask], y[mask])
            self.group_params_[group] = params
            fitted.append(params)
        self.default_params_ = np.median(np.stack(fitted, axis=0), axis=0)
        return self

    def predict(self, X_num, X_cat):
        X_num = np.asarray(X_num, dtype=float)
        labels = group_labels(X_cat)
        preds = np.zeros(len(labels), dtype=float)
        for group in sorted(set(labels.tolist())):
            mask = labels == group
            params = self.group_params_.get(group, self.default_params_)
            preds[mask] = _sldagent_predict_params(self.benchmark_name,
                                                   X_num[mask], params)
        return preds
```
