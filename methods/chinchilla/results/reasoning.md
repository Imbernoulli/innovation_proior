OK, let me think this through from scratch. I'm about to spend an enormous compute budget training one big language model, and the brutal fact is I get one shot. I know my budget in advance — so many accelerators for so long — which fixes a number of FLOPs $C$. What I have to decide before I start, and can't take back, is two things: how big to make the model, parameter count $N$, and how many tokens to train it on, $D$. The field's reflex is "make $N$ as big as you can afford." I want to ask whether that reflex is actually optimal, because if it's wrong I'm about to waste a fortune.

First, are $N$ and $D$ even independent choices? No — the budget couples them. Training cost is dominated by passing $D$ tokens through an $N$-parameter model, forward and backward. Each parameter does about one multiply-add per token, which is two FLOPs, so a forward pass over $D$ tokens is roughly $2ND$; the backward pass is about twice that; total roughly $6ND$. So to good approximation
$$C \approx 6ND.$$
That's the whole constraint. $C$ fixed means the product $ND$ is fixed: every parameter I add, I pay for by training on fewer tokens, and vice versa. So this isn't "pick $N$ as big as possible" — it's an allocation along a hyperbola $ND = C/6$, and I want the point on that hyperbola that minimizes the final loss:
$$N_{\text{opt}}(C),\ D_{\text{opt}}(C) = \operatorname*{argmin}_{N,D:\, 6ND=C} L(N,D).$$
I want the *functions* $N_{\text{opt}}(C)$ and $D_{\text{opt}}(C)$, not one answer — a recipe for any budget.

Now, what does the prevailing wisdom say the answer is? The existing scaling analysis concluded that when you get $10\times$ more compute you should make the model about $5.5\times$ bigger and train on only about $1.8\times$ more tokens — so $N$ scales like $C^{0.73}$ and $D$ like $C^{0.27}$, model size growing nearly three times as fast as data. That's why essentially everyone, at every size, trains on roughly 300B tokens: data is treated as nearly fixed, and compute goes almost entirely into parameters. Let me hold that as the claim to test: $a \approx 0.73$, $b\approx 0.27$ in $N_{\text{opt}}\propto C^a$, $D_{\text{opt}}\propto C^b$.

Something nags at me before I even run anything, and it's about how the loss-versus-tokens curve gets measured. When I train with a cosine learning-rate schedule, the rate is supposed to decay — by about $10\times$ — over the run and bottom out right as I stop. The end-of-training loss is low partly *because* the learning rate has annealed to near zero by then. Now suppose I want to know "what loss does an $N$-parameter model reach after $D$ tokens" for several values of $D$. The cheap way is to train one long run and read the loss off at each intermediate $D$. But at an intermediate point of a long cosine cycle, the learning rate is still high — it hasn't annealed for stopping *there* — so the loss I read is higher than what I'd get if I'd actually scheduled the run to end at $D$. The measured loss at intermediate $D$ is biased upward. How big is the effect? If the cosine cycle overshoots the actual stopping point by more than about a quarter, final performance is visibly worse. That's not a rounding error. So if the prevailing analysis estimated intermediate-$D$ losses from long runs without re-matching the schedule, it would systematically *understate* how much a model gains from more tokens — and that bias pushes the inferred optimum toward big models trained on few tokens. Which is exactly the conclusion I'm suspicious of. So rule one for me: every run's cosine cycle length is matched to its own token count. With that fixed, let me estimate the frontier — and I don't trust any single estimation method, so I'll build three independent ones and see whether they agree.

The most direct estimator: fix a family of model sizes, sweep how long I train each, and read the frontier off the envelope of the training curves. Take models from 70M up past 10B. For each size $N$, run it several times — four runs — with the learning rate decaying $10\times$ over token-horizons that span a wide range, a factor of 16, so I cover short and long training for each size. Each run gives me a loss-versus-FLOPs curve; I smooth it (a little Gaussian smoothing over steps) and interpolate, so I have a continuous map from FLOPs to loss for that run. Now overlay every run's curve and, at each FLOP value, ask which run is lowest. That lower envelope is the best loss achievable at each compute level, and the run that achieves it tells me the $(N,D)$ that did it. Sweep $C$ across many log-spaced values — say 1500 of them — read off the winning $N$ and its $D=C/(6N)$ at each, and I have empirical $(C, N_{\text{opt}}, D_{\text{opt}})$ points. Fit a power law in log-log space to each: $N_{\text{opt}}\propto C^a$, $D_{\text{opt}}\propto C^b$. When I do this the exponents come out at $a=0.50$ and $b=0.50$. Equal. Not $0.73/0.27$ — dead even. And a consistency check that makes me trust it: the points that win the envelope all sit in the last ~15% of their training, which is exactly what you'd expect if matching the cosine cycle to the horizon is the right thing — the winning runs are ones that annealed right at the end. So the schedule fix and the result are telling the same story.

That's one method, and it has a soft spot: it leans on smoothing and interpolating curves, and on the envelope being well-sampled. Let me attack the same question from a completely different angle, holding *compute* fixed instead of size. Pick a set of fixed FLOP budgets — nine of them, from about $6\times10^{18}$ up to $3\times10^{21}$. For each budget $C$, train a range of model sizes; the constraint $D=C/(6N)$ sets each one's token count automatically, and I match the cosine cycle to that. Now for a single $C$, plot final loss against $N$. What's the shape? At small $N$ the model is too small — it underfits, high loss. At large $N$, since $C$ is fixed, $D$ has been squeezed tiny, so the model is barely trained — also high loss. In between there's a sweet spot. So each iso-compute curve is a valley with a clear minimum, and that minimum is directly the optimal $N$ for that $C$ — no envelope, no interpolation across runs, I can *see* it. To pin the bottom precisely, fit a parabola to each iso-FLOP curve and take its vertex. Then, across the nine budgets, fit the power law of vertex-$N$ versus $C$ (and the corresponding $D$). This gives $a=0.49$, $b=0.51$. Again, essentially equal, and from a method that shares almost none of the first method's machinery. Two independent roads to the same place.

Now I want a third method that isn't just another way of reading off minima — I want a *model* of the loss surface itself, $\hat L(N,D)$, that I can fit once and then optimize analytically. What functional form should it have? Let me think about where loss comes from, decomposing the risk of the predictor I actually end up with. Predicting the next token means choosing a map $f$ from context to a distribution over tokens. There's a Bayes-optimal $f^\star$ that minimizes cross-entropy on the true data distribution — its loss $L(f^\star)$ is irreducible, the entropy of natural text; no model, however large, beats it. Now restrict to Transformers of size $N$: the best one in that class, $f_N$, can't quite reach $f^\star$ because the hypothesis space is finite-dimensional — that's a function-approximation gap $L(f_N)-L(f^\star)$ that depends on $N$ and shrinks as $N$ grows. And finally I don't even get $f_N$: I run a finite number of gradient steps over a single pass through $D$ tokens, landing at some $\bar f_{N,D}$ — a stochastic-approximation gap $L(\bar f_{N,D})-L(f_N)$ that depends on $D$ and shrinks as I train on more data. So the loss splits cleanly into three:
$$L(N,D) = \underbrace{L(f^\star)}_{\text{Bayes}} + \underbrace{\big(L(f_N)-L(f^\star)\big)}_{\text{approximation},\ f(N)} + \underbrace{\big(L(\bar f_{N,D})-L(f_N)\big)}_{\text{stochastic},\ g(D)}.$$
What forms do $f(N)$ and $g(D)$ take? The approximation gap for restricted function classes tends to fall as a power of the dimension — for two-layer networks it's expected to go like $N^{-1/2}$. The stochastic gap is early-stopping of a stochastic first-order method, whose convergence rate is lower-bounded by $D^{-1/2}$ and, importantly, is dimension-free — it depends on the optimization, not on $N$. Dimension-free is what lets me write it as a function of $D$ alone, decoupled from $N$. So both gaps are power laws, and the natural parametric form is
$$\hat L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$
with $E$ the irreducible Bayes risk, $A/N^\alpha$ the approximation term, $B/D^\beta$ the stochastic term. The $1/2$ exponents are an upper expectation; the actual $\alpha,\beta$ I'll fit, and I'd expect them at or below $1/2$.

Fitting it: I have many runs, each a triple $(N_i, D_i, L_i)$. I want $(A,B,E,\alpha,\beta)$ minimizing the discrepancy between $\hat L$ and the observed $L_i$. Two choices matter. First, fit in *log* space — match $\log\hat L$ to $\log L_i$ — because loss spans orders of magnitude and I care about relative error. Second, the residuals will have outliers, especially the low-compute runs which are noisier and which I care about least for extrapolating to huge budgets. A squared loss would let those outliers dominate the fit. So use the Huber loss, which is quadratic for small residuals and linear for large ones — robust, downweighting outliers. With a small $\delta=10^{-3}$ it treats the noisy low-FLOP points as outliers automatically rather than bending the whole surface to fit them. (A larger $\delta$ overfits the small-compute regime and predicts held-out large runs badly; smaller barely changes anything.) So
$$\min_{A,B,E,\alpha,\beta}\ \sum_i \operatorname{Huber}_\delta\!\Big(\log\hat L(N_i,D_i)-\log L_i\Big).$$
There's a numerical wrinkle: $\log\hat L = \log\big(E + A N^{-\alpha} + B D^{-\beta}\big)$ is a log of a sum, which is unstable to evaluate and differentiate directly. Rewrite each term as an exponential of a log: $A N^{-\alpha} = \exp(a - \alpha\log N)$ with $a=\log A$, and likewise $b=\log B$, $e=\log E$. Then
$$\log\hat L = \operatorname{LSE}\big(a-\alpha\log N,\ b-\beta\log D,\ e\big),$$
the log-sum-exp of three terms — numerically stable, smooth, and easy to autodiff. So I optimize over $(a,b,e,\alpha,\beta)$, then recover $A,B,E=\exp(a),\exp(b),\exp(e)$. The objective is non-convex, so I run L-BFGS from a grid of initializations and keep the best, checking the winner isn't on the grid boundary. Fitting this out gives
$$L(N,D) = 1.69 + \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.28}},$$
so $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$ — both exponents below $1/2$, as expected. For any large extrapolated point I should keep the unrounded optimizer output, roughly $E=e^{0.5267228}$, $A=e^{6.0073404}$, $B=e^{6.0179186}$, $\alpha=0.33917084$, $\beta=0.2849083$, because rounding the exponents moves the frontier noticeably.

Now the payoff of having a closed form: I can solve the allocation analytically instead of reading it off a grid. Minimize $\hat L = E + A N^{-\alpha} + B D^{-\beta}$ subject to $6ND=C$. Substitute $D=C/(6N)$ so it's one variable:
$$g(N) = E + A N^{-\alpha} + B\Big(\frac{6N}{C}\Big)^{\beta}.$$
The $E$ drops out under differentiation. As $N$ shrinks the first term blows up (model too small); as $N$ grows the third blows up (too few tokens). So there's an interior minimum — set $g'(N)=0$:
$$g'(N) = -\alpha A\, N^{-\alpha-1} + \beta B\Big(\frac{6}{C}\Big)^{\beta} N^{\beta-1} = 0.$$
Move the negative term over:
$$\alpha A\, N^{-\alpha-1} = \beta B\Big(\frac{6}{C}\Big)^{\beta} N^{\beta-1}.$$
Divide both sides by $N^{\beta-1}$ and by $\beta B (6/C)^\beta$:
$$N^{-\alpha-1-(\beta-1)} = N^{-(\alpha+\beta)} = \frac{\beta B}{\alpha A}\Big(\frac{6}{C}\Big)^{\beta} \quad\Longrightarrow\quad N^{\alpha+\beta} = \frac{\alpha A}{\beta B}\Big(\frac{C}{6}\Big)^{\beta}.$$
Take the $(\alpha+\beta)$-th root:
$$\boxed{\,N_{\text{opt}}(C) = \Big(\frac{\alpha A}{\beta B}\Big)^{\frac{1}{\alpha+\beta}} \Big(\frac{C}{6}\Big)^{\frac{\beta}{\alpha+\beta}} = G\,\Big(\frac{C}{6}\Big)^{a},\quad G=\Big(\frac{\alpha A}{\beta B}\Big)^{\frac{1}{\alpha+\beta}},\quad a=\frac{\beta}{\alpha+\beta}.}$$
And then $D$ falls straight out of the constraint, $D_{\text{opt}}=\dfrac{C/6}{N_{\text{opt}}} = G^{-1}\big(C/6\big)^{1-a}=G^{-1}\big(C/6\big)^{b}$ with $b=\dfrac{\alpha}{\alpha+\beta}$. Notice $a+b = \dfrac{\beta+\alpha}{\alpha+\beta}=1$ identically — whatever the exponents, the optimal $N$ and $D$ split the compute so their log-exponents sum to one. The near-balance is not automatic; it appears because the fitted $\alpha$ and $\beta$ are close.

Plug in the fit: $\alpha+\beta = 0.34+0.28 = 0.62$, so $a = \beta/(\alpha+\beta) = 0.28/0.62 = 0.46$ and $b = \alpha/(\alpha+\beta) = 0.34/0.62 = 0.54$. So this third, completely different method — a parametric surface fit and an analytic minimization — says $N_{\text{opt}}\propto C^{0.46}$, $D_{\text{opt}}\propto C^{0.54}$. (It tilts very slightly toward more tokens than the first two, because the Huber loss downweights the low-compute runs and the frontier has a touch of negative curvature, pulling $N_{\text{opt}}$ down a bit at large $C$.)

Line them up. Method one (training-curve envelope): $a=0.50,\ b=0.50$. Method two (iso-FLOP valleys): $a=0.49,\ b=0.51$. Method three (parametric fit, solved in closed form): $a=0.46,\ b=0.54$. Three independent methodologies, three different sets of runs and fits, and they all land on the same place: $N$ and $D$ should grow in *equal* proportion with compute, exponents clustered around one-half — flatly contradicting the prevailing $0.73/0.27$ that grows the model nearly three times as fast as the data. The agreement across methods is what makes me believe it; no single one is self-certifying, but their consensus is.

What does this say concretely? It says current large models are badly mis-allocated — far too big for the number of tokens they saw. Take a model trained at a large budget, on the order of $5.76\times10^{23}$ FLOPs, sitting at 280B parameters on ~300B tokens. The envelope estimate puts that budget near 67B parameters and about 1.5T tokens; the iso-FLOP estimate is in the same broad region; the parametric fit is more aggressive, around 40B parameters with substantially more tokens when I use the unrounded coefficients. The exact large-scale point differs, but the direction does not: the same compute should move from hundreds of billions of parameters toward tens of billions, with far more data. If I need one full-scale allocation under extrapolation uncertainty, I choose the larger end of that predicted range, around 70B parameters and roughly 1.4T tokens. A smaller model is not just predicted to be better at that budget — it's cheaper to *serve* afterward, since inference cost scales with $N$. So the allocation error has been costing on both ends.

Let me write the method as code, grounded in how the parametric fit and the closed-form solve are actually done. The loss model first — three terms, evaluated stably via log-sum-exp so it can be fit in log space:

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, huber


def log_loss_pred(theta, logN, logD):
    # theta = (a, b, e, alpha, beta) with A,B,E = exp(a),exp(b),exp(e).
    # log L_hat = LSE(a - alpha*logN, b - beta*logD, e)  ==  log(E + A/N^alpha + B/D^beta)
    a, b, e, alpha, beta = theta
    terms = np.stack([a - alpha * logN, b - beta * logD, np.full_like(logN, e)], axis=0)
    return logsumexp(terms, axis=0)


def parametric_loss(N, D, params):
    A, B, E, alpha, beta = params
    return E + A / N ** alpha + B / D ** beta            # L_hat(N, D)
```

The fit: minimize the Huber penalty between predicted and observed *log* loss, over the log-parameterization, from a grid of initializations (the objective is non-convex), then exponentiate back:

```python
def fit_parametric(runs, delta=1e-3):
    # runs: array of (N_i, D_i, L_i)
    runs = np.asarray(runs, dtype=float)
    N, D, L = runs[:, 0], runs[:, 1], runs[:, 2]
    logN, logD, logL = np.log(N), np.log(D), np.log(L)

    def objective(theta):
        r = log_loss_pred(theta, logN, logD) - logL
        return np.sum(huber(delta, r))                   # robust: downweights outliers

    best = None
    for alpha0 in [0., 0.5, 1.0, 1.5, 2.0]:              # grid of inits
        for beta0 in [0., 0.5, 1.0, 1.5, 2.0]:
            for e0 in [-1., -0.5, 0., 0.5, 1.]:
                for a0 in [0., 5., 10., 15., 20., 25.]:
                    for b0 in [0., 5., 10., 15., 20., 25.]:
                        res = minimize(objective, [a0, b0, e0, alpha0, beta0],
                                       method="L-BFGS-B")
                        if best is None or res.fun < best.fun:
                            best = res
    a, b, e, alpha, beta = best.x
    return (np.exp(a), np.exp(b), np.exp(e), alpha, beta)   # (A, B, E, alpha, beta)
```

The closed-form allocation — exactly the minimization I just did by hand:

```python
def optimal_allocation(C, params):
    A, B, E, alpha, beta = params
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    a = beta / (alpha + beta)            # N_opt exponent
    b = alpha / (alpha + beta)           # D_opt exponent ; a + b == 1
    N_opt = G * (C / 6.0) ** a
    D_opt = (C / 6.0) ** b / G           # == (C/6) / N_opt, so 6*N_opt*D_opt == C
    return N_opt, D_opt
```

And the empirical estimators read the optimum straight off runs and corroborate the parametric one. The training-curve envelope interpolates each already-smoothed curve at the same compute budget and keeps the lowest loss; the iso-FLOP valley fits a parabola in log-space and takes the vertex; the power-law fit turns the resulting optima into scaling exponents:

```python
def envelope_optimum(C, run_curves):
    # run_curves: iterable of {"N": scalar, "flops": array, "loss": smoothed array}
    best = None
    logC = np.log(C)
    for curve in run_curves:
        flops = np.asarray(curve["flops"], dtype=float)
        if C < flops[0] or C > flops[-1]:
            continue
        N = float(curve["N"])
        loss = float(np.interp(logC, np.log(flops), curve["loss"]))
        D = C / (6.0 * N)
        if best is None or loss < best[2]:
            best = (N, D, loss)
    if best is None:
        raise ValueError("no curve covers the requested compute budget")
    return best


def isoflop_optimum(runs_at_fixed_C):
    # at one compute budget: loss is a valley in log-N; vertex of the parabola is N_opt.
    runs_at_fixed_C = np.asarray(runs_at_fixed_C, dtype=float)
    logN = np.log(runs_at_fixed_C[:, 0])
    loss = runs_at_fixed_C[:, 2]
    c2, c1, c0 = np.polyfit(logN, loss, 2)               # loss ~ parabola in log N
    logN_star = -c1 / (2 * c2)
    return np.exp(logN_star)


def fit_power_law(Cs, values):
    # value ∝ C**exponent  ->  linear fit in log-log
    slope, intercept = np.polyfit(np.log(Cs), np.log(values), 1)
    return np.exp(intercept), slope                      # (coeff, exponent a or b)
```

The causal chain, start to end: a single fixed compute budget couples model size and data through $C\approx 6ND$, so the real decision is where on the hyperbola $ND=C/6$ to sit — and the prevailing answer (grow $N$ as $C^{0.73}$, data only $C^{0.27}$) rests on intermediate-horizon losses measured without matching the cosine schedule to the token count, which biases the frontier toward big-$N$/small-$D$. Matching the schedule to each run's horizon and then estimating the frontier three independent ways — the lower envelope of training curves ($a,b\approx 0.50,0.50$), iso-FLOP valleys ($0.49,0.51$), and a parametric surface $\hat L = E + A/N^\alpha + B/D^\beta$ minimized in closed form under $6ND=C$ to give $N_{\text{opt}}=G(C/6)^{\beta/(\alpha+\beta)}$, $D_{\text{opt}}=G^{-1}(C/6)^{\alpha/(\alpha+\beta)}$ with $\alpha=0.34,\beta=0.28$ yielding $(0.46,0.54)$ — all agree that $N$ and $D$ should scale in nearly equal proportion. The consequence is that the era's large models are several times too big for their token counts; re-allocating a large budget points to tens of billions of parameters and trillion-token training rather than hundreds of billions of parameters on about 300B tokens, which is both better at that compute and cheaper to serve.
