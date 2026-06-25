Let me start from what actually hurts when I tune these models. I have a validation loss `f(x)` over a configuration `x` — learning rate, batch size, dropout, number of layers, all mixed up: some continuous, some integer, some categorical. I want `x* = argmin f(x)`, and I can only see `f` through noise, `y = f(x) + eps`, because training is stochastic. The thing that has changed, the thing that makes my old habits fail, is that one evaluation of `f` now costs days. So the budget I can spend tuning is maybe a handful of full trainings, total. Under that constraint I have two desires that pull against each other: I want something *good* fast — strong anytime performance, because I can't afford to wait — and I want the *best* configuration if I let it run longer. And I want it to keep working when there are dozens of hyperparameters, when the space is mixed and categorical, when the objective is as noisy as deep RL, and when I throw thirty machines at it.

So what do I have on the table, and where exactly does each one stall? Random search I trust as a floor: uniform draws, embarrassingly parallel, robust to whatever shape `X` has. But it never learns. Every draw ignores everything I've already seen, so in a big space I'm just waiting to get lucky, and the rate of progress per evaluation never improves. That's the thing I most want to fix — guidance.

The line of work that gives guidance is Bayesian optimization. Keep a probabilistic model `p(f | D)` of the objective from the data `D = {(x_i, y_i)}` so far, and an acquisition function that says where to look next. The natural one is expected improvement over the incumbent `alpha = min y_i`: `a(x) = E[max(0, alpha - f(x))] = ∫ max(0, alpha - f(x)) dp(f | D)`. Pick `x` to maximize `a`, evaluate, refit, repeat. The usual surrogate is a Gaussian process, and on a low-dimensional continuous problem it really is the gold standard — smooth, calibrated uncertainty. But the moment I write down its costs against my desiderata it starts failing them one by one. Fitting a GP is cubic in the number of observations, `O(|D|^3)`. It degrades in high dimensions. Off-the-shelf kernels don't even apply to a space with categorical and conditional hyperparameters without me hand-building a kernel. And the results swing with hyperpriors I have to set carefully per problem. Worst of all, and this is true of *any* black-box BO regardless of surrogate: with almost no data the model is uninformative, so its first dozens of suggestions are no better than random. It starts cold. And here "starting cold for a few dozen evaluations" means burning a few dozen *full trainings* — exactly the budget I don't have. So BO buys me guidance and final quality but throws away anytime performance, and the GP in particular fails scalability, flexibility, robustness, and is computationally heavy. The GP is not the surrogate for me.

Now I notice there's a second, almost orthogonal idea in the field that attacks cost directly instead of guidance. For nearly every ML objective I can define a *cheap approximation* `tilde f(·, b)` parameterized by a budget `b`: train for fewer epochs, on less data, for fewer MCMC steps, for fewer RL seeds. At `b_max` it's the real `f`; below that it's cheaper and noisier. And the empirical fact that makes this useful is that a configuration's quality is often *partly* visible at small `b` — bad configurations tend to look bad early. So I shouldn't pay full budget for everything; I should pour resources into what looks promising cheaply and cut the rest. Successive halving is the clean version: take `n` configurations, evaluate all on a small budget, sort, keep the best `1/eta`, multiply the budget by `eta`, repeat to `b_max`. Survivors get exponentially more resource.

But successive halving has a wart that I have to stare at, because it's the thing the next idea is built to fix. It needs `n` as an input, and for a fixed total budget `B` the budget per configuration is roughly `B/n`. So do I want many configurations each run briefly — large `n`, aggressive early stopping — or a few configurations each run long — small `n`, conservative? It depends entirely on how fast configurations *differentiate*: if quality shows up early, large `n` is right; if configurations only separate at large budgets, large `n` murders the eventual winner before it's had a chance, and I want small `n`. And I don't know which regime I'm in — that depends on the unknown shape of the learning curves. This is the `n`-versus-`B/n` problem, and there's no a-priori right answer.

Hyperband's move is the right kind of move when you can't choose a parameter: hedge over it. Don't pick one `n`; run successive halving for a geometric ladder of `n` values, each in its own "bracket," so that whichever regime I'm actually in, *some* bracket has the right aggressiveness. Let me make sure I can reconstruct the bracket schedule from scratch, because I'm going to inherit it and I'd better understand why it's shaped the way it is. With a discount factor `eta` and budgets in `[b_min, b_max]`, the number of distinct budget rungs is set by how many times I can multiply `b_min` by `eta` before reaching `b_max`: `s_max = floor(log_eta(b_max / b_min))`. I index brackets by `s` from `s_max` down to `0`. Bracket `s` should start its configurations at budget `r = b_max · eta^{-s}` — so `s = s_max` starts at the smallest budget (most aggressive) and `s = 0` starts already at `b_max` (no early stopping at all — that bracket is just random search at full budget). Now, how many configurations `n` should bracket `s` get? I want every bracket to cost about the same total resource, so that hedging is fair. A bracket with `s+1` rungs spends, at each rung, about `n · r` (the number of live configurations times the budget each gets), and because halving keeps `(configs × budget)` roughly constant across rungs, the whole bracket costs about `(s+1) · n · r`. I want that to equal a common `B`. The clean choice the field uses is `B = (s_max + 1) · b_max`. Before integer rounding, that gives the ideal count `n_ideal = ((s_max+1)/(s+1)) · eta^s`, and then `(s+1) · n_ideal · r = (s+1) · (s_max+1)/(s+1) · eta^s · b_max · eta^{-s} = (s_max+1) · b_max = B`.

Let me not take that algebra on faith — let me actually instantiate the ladder and add up the rung costs, because the bracket sizes are going to be *integers*, and integer rounding could quietly break the balance the formula promises. Take `b_min = 1`, `b_max = 27`, `eta = 3`, so `s_max = floor(log_3 27) = 3` and the common target is `B = (3+1)·27 = 108`. The pre-rounding ideal counts come out clean: `s = 3` gives `n_ideal = 27` at `r = 1`, `s = 2` gives `12` at `r = 3`, `s = 1` gives `6` at `r = 9`, `s = 0` gives `4` at `r = 27`, and `(s+1)·n_ideal·r` is `108` in every one of the four cases — so the continuous accounting is exactly balanced. But now I round. The integer scheduler I'm going to inherit sets the leading count by `n0 = floor((s_max+1)/(s+1)) · eta^s` and then halves down the rungs. Summing `n·budget` over the rungs of each realized bracket I get costs of `108`, `81`, `108`, `108` for `s = 3, 2, 1, 0`. So three brackets land exactly on `108` but the `s = 2` bracket comes in at `81` — because `floor(4/3) = 1`, not `1.33`, so its leading count is `1·9 = 9` instead of the ideal `12`, and the bracket is a quarter cheaper. That's a useful thing to have learned by computing rather than trusting: the cost balance is exact only in the real-valued idealization, and the integer scheduler approximates it to within a rounding-sized wobble. It's close enough that the hedge is still fair, and there's nothing to fix here — but I'd have been wrong to claim "every bracket costs `B`" as if rounding were free. The `eta^s` is what makes the aggressive brackets numerous-and-cheap-per-config and the conservative ones few-and-expensive, and now I've seen the schedule actually balance on the page.

And Hyperband is genuinely strong. It inherits all of random search's robustness, scalability, flexibility, and parallelism, and the cheap fidelities give it the anytime performance BO lacked — at small-to-medium budgets it routinely beats both random search and black-box BO. So it's close to my desiderata. But there is one hole, and it's a glaring one once I name it: Hyperband draws the `n` configurations for every bracket *uniformly at random*. It never conditions those draws on anything it has learned. So as the budget grows and the cheap-fidelity advantage saturates, its edge over random search just... evaporates, and its final quality trails any method that actually learns where good configurations live. It has the speed and none of the guidance. BO had the guidance and none of the speed.

So I'm staring at two methods that are almost exactly complementary. One is fast and blind; the other is slow and sighted. The obvious thing to want is both. Let me be careful, though — "combine them" is a slogan, not a method. *How*? The thing Hyperband is good at is deciding *how much budget to give which configuration* — the bracket schedule I just rederived and checked. The thing it's bad at is *which configurations to put into the brackets*. And that "which configuration" question is precisely what a model-based acquisition answers. The division of labor is clean: keep Hyperband's resource-allocation skeleton — the brackets, the budgets, the halving — completely intact, and at the one point where it reaches for a uniform random draw, instead ask a model. That leaves untouched every desideratum Hyperband already satisfies (anytime, scalable, robust, flexible, parallel) and adds the only one it's missing (strong final performance), because final performance is exactly what blind sampling was costing me.

Now, *which* model? Not a GP — I already convicted it on scalability, mixed spaces, robustness, and cost, and there's a new reason it's especially wrong *here*: the whole point of the bandit skeleton is that it makes a *huge number of cheap, low-budget evaluations*. A cubic-in-`|D|` model fed that flood of observations would have its fitting cost blow up and dominate the very evaluations I made cheap. The model also has to be dead simple — I want few moving parts I can verify and reimplement — and it has to natively eat categorical and integer hyperparameters. That set of requirements points away from modeling `p(f | x)` at all.

The alternative I keep coming back to is to model the *inputs* split by performance instead of the outputs. Don't fit `f`; fit two densities over configuration space. Take a quantile `gamma` of the observed losses, set the threshold `alpha` so a `gamma` fraction of observations falls below it, and build `l(x) = p(x | y < alpha)` from the configurations that did well and `g(x) = p(x | y >= alpha)` from the ones that didn't. Each is just a kernel density estimator — a Parzen window. Why is this so much better for my situation? Because a KDE is *linear* in the number of points, not cubic; because density estimation in a mixed space is trivial — Gaussian kernels on continuous dims, a categorical kernel on categorical dims; and because it's about as simple as a statistical model gets. Every objection I had to the GP dissolves.

But I have to check that this density-ratio idea actually optimizes the right thing, or it's just a heuristic that happens to feel reasonable. The instinct is to propose high `l`, low `g` — "looks like the good ones, not like the bad ones." Is that *expected improvement*, or am I fooling myself? Let me grind through EI under this parameterization. EI is `EI(x) = ∫_{-∞}^{α} (α - y) p(y | x) dy`. Write `p(y | x) = p(x | y) p(y) / p(x)`. By construction the densities split exactly at `alpha`: `p(x | y) = l(x)` for `y < alpha` and `g(x)` for `y >= alpha`, and `gamma = p(y < alpha)`. So the marginal is `p(x) = ∫ p(x | y) p(y) dy = gamma·l(x) + (1-gamma)·g(x)`. The numerator only runs over `y < alpha`, where `p(x | y) = l(x)`, a constant in `y`, so it pulls out:

```
∫_{-∞}^{α} (α - y) p(x | y) p(y) dy = l(x) ∫_{-∞}^{α} (α - y) p(y) dy.
```

That trailing integral is some positive number that doesn't depend on `x` at all — call it `C`; concretely `∫_{-∞}^{α}(α-y)p(y)dy = α·gamma - ∫_{-∞}^{α} y·p(y) dy`, both `x`-free. So

```
EI(x) = l(x)·C / p(x) = C·l(x) / ( gamma·l(x) + (1-gamma)·g(x) ).
```

Divide top and bottom by `l(x)`:

```
EI(x) = C / ( gamma + (1 - gamma)·g(x)/l(x) )  ∝  ( gamma + (1-gamma)·g(x)/l(x) )^{-1},
```

a strictly decreasing function of the ratio `g(x)/l(x)`. The algebra hinges on two moves — pulling `l(x)` out of the numerator because `p(x|y)` is constant below `alpha`, and the marginal being a `gamma`-mixture — and I don't fully trust a derivation I've only done symbolically, so let me put numbers to it before I build a method on it. Pick a concrete generative model I can integrate by hand: let the loss marginal be `p(y) = N(0,1)`, take `gamma = 0.25` so `alpha = Φ^{-1}(0.25) ≈ -0.674`, and define the conditionals over a 1-D config `x` as `l(x) = N(x; 0.3, 0.1)` for the good half and `g(x) = N(x; 0.7, 0.2)` for the bad half. Then I compute two things independently: the *direct* `EI(x) = ∫_{-∞}^{α}(α-y)·p(x,y)/p(x) dy` by brute numerical integration of `p(x,y) = p(x|y)p(y)`, and the *closed form* `C/(gamma + (1-gamma)g/l)` with `C = ∫_{-∞}^{α}(α-y)p(y)dy ≈ 0.149`. Evaluating at `x = 0.1, 0.3, 0.5, 0.7, 0.9`:

```
x=0.1  EI:direct=0.53121  formula=0.53121   g/l=0.041
x=0.3  EI:direct=0.49594  formula=0.49594   g/l=0.068
x=0.5  EI:direct=0.07726  formula=0.07726   g/l=2.241
x=0.7  EI:direct=0.00013  formula=0.00013   g/l=1490.5
x=0.9  EI:direct=0.00000  formula=0.00000   g/l=2.0e7
```

The two columns agree to five decimals at every point, the marginal `p(x)` from the formula matched the numerically integrated marginal at each `x`, and sweeping `x` across `[0,1]` and sorting by `g/l`, `EI` is monotonically decreasing along the whole sweep. So the "looks good, not bad" instinct wasn't a heuristic — it really is EI, and `argmin g(x)/l(x) = argmax l(x)/g(x)` maximizes it. And notice what dropped out of the computation: `C`, the `y`-integral, never needed to be computed to *rank* configurations, and I never had to model `p(y)` or `f` at all. The model lives entirely in the input densities. That's the simplicity I wanted, and now I've checked the acquisition rather than asserted it.

Now I have to make this concrete and I immediately hit the first real wall: I have observations on *many different budgets*. Hyperband evaluates the same and different configurations at `b_min`, at `eta·b_min`, all the way to `b_max`. Which observations do I fit the KDEs on? If I lump them all together I'm mixing fidelities — a configuration that scored well at `b_min` might be junk at `b_max`, and my "good" density would be polluted by cheap-budget verdicts I shouldn't trust. So lumping is wrong. The honest thing is to keep observations *separated by budget*: a set `D_b` for each budget `b`, and a separate pair of KDEs per budget. Now, which budget's model do I actually *use* to propose the next configuration? My true objective is `f = tilde f(·, b_max)`. The largest budget's model is the one that speaks about the thing I actually care about. But early on I have almost no data at `b_max` — that's the whole reason I'm using cheap fidelities. So I can't always use `b_max`. The resolution: use the model for the *largest budget for which I have enough observations to fit a trustworthy density*. Early in a run that's a small budget; as evidence accrues on larger budgets, the model I rely on climbs the fidelity ladder on its own. And this has a lovely side effect — it lets me *overcome* a wrong conclusion drawn at a small budget, because eventually the higher-fidelity model, built on more reliable data, takes over the proposing. The low fidelities bootstrap me fast; the high fidelity corrects me in the end.

"Enough observations" — I have to pin that down, it's not vague. To fit a density in `d` dimensions I need at least `d+1` points; with `d` or fewer points the KDE is degenerate (you can't pin a `d`-dimensional density with fewer points than dimensions plus one). So set the minimum number of points to model, `N_min = d + 1`, where `d` is the number of hyperparameters. That's the bare floor and it makes the model available as early as honestly possible.

Next wall: I need *two* densities, good and bad, and both need enough points. If I follow the textbook recipe — pick a quantile `gamma` and call the bottom `gamma` fraction "good" — then early on, when I have say `N_min + 2` points total, the "good" fraction `gamma·N_b` is a tiny handful, far below `N_min`, and the good KDE is degenerate even though I have enough points overall. I'm being too literal about the quantile. Let me relax it: don't insist the good set be *exactly* the `gamma` quantile when data is scarce; insist instead that *both* densities have at least `N_min` points, and otherwise split by performance as close to the quantile as I can. So take

```
N_{b,l} = max( N_min, gamma · N_b )      # requested number of "good" points
N_{b,g} = max( N_min, N_b - N_{b,l} )    # requested number of "bad" points
```

the `N_{b,l}` best configurations for `l` and the `N_{b,g}` worse configurations for `g`. The `max(N_min, ·)` expresses the real constraint: neither fitted density is useful until it has enough points. I want to be sure this floor actually does the job, so let me walk the realized split as observations accumulate, for a concrete `d = 2` (so `N_min = 3`) with the top `15%` going to the good set. The leading count `n_good = max(3, ⌊0.15·N⌋)` is `3` for a long time; the bad set takes the rows *after* the good slice, and I'll fit only when both realized slices have more rows than dimensions (`> 2`):

```
N=3   good_rows=3  bad_rows=0  -> skip (bad degenerate)
N=4   good_rows=3  bad_rows=1  -> skip (bad degenerate)
N=5   good_rows=3  bad_rows=2  -> skip (bad degenerate)
N=6   good_rows=3  bad_rows=3  -> fit
N=7   good_rows=3  bad_rows=4  -> fit
```

So even though `N_min = d+1 = 3` is satisfied at `N = 3`, the *first actual fit* doesn't happen until `N = 6` — because the good slice greedily eats the first three rows and the bad slice is starved until there are three more. That is exactly the failure the `max(N_min, ·)` and the realized-shape guard are there to catch: the `N_min` floor on the count is necessary but *not sufficient*; the realized bad slice is what actually gates fitting, and I have to check `bad.shape[0] > bad.shape[1]` and `good.shape[0] > good.shape[1]` directly rather than trusting the requested counts. Good — walking it told me the guard belongs on the realized arrays, not the requested sizes. Concretely with a percentile `q` (say the top `15%`), I sort by loss, ask for at least `N_min` good points and at least `N_min` bad points, and when the realized integer slices are still too small in `d` dimensions I do not force the KDE; I keep sampling randomly until both fitted arrays are nondegenerate. That is better than pretending a density exists just because the quantile formula has a name.

Now the acquisition step itself. I've shown I want `argmin g(x)/l(x)`. How do I optimize that over `X`? I am *not* going to run a full optimizer over the ratio — that would be expensive, and worse, it would return the single global argmax every time, which is terrible for two reasons I care about. First, it's greedy: it would hammer the current best mode of `l` and stop exploring. Second, and this matters because I want parallelism, if I always return the exact optimizer, consecutive suggestions are nearly identical, and `k` workers would all evaluate almost the same configuration — no speedup. So I deliberately *don't* fully optimize EI. Instead I sample a modest number `N_s` of candidate configurations, score each by the ratio, and return the best of the sample. Not fully optimizing is a feature: it keeps consecutive suggestions diverse enough to parallelize near-linearly, which is one of my desiderata.

But sample the candidates from *where*? Uniformly over `X` would mostly waste the `N_s` draws in regions `l` says are bad. I want to draw candidates near the promising configurations — so draw them from `l` itself, the good density. Concretely: pick a random "good" datum, and jitter it with a Gaussian kernel to get a nearby candidate, for each continuous dimension; for categorical dimensions, keep the datum's category most of the time and occasionally flip to a random one. That concentrates my `N_s` candidates where EI is plausibly high.

Here I hit a subtle wall I almost missed. If I sample from `l` with its *fitted* bandwidth, I'm being too greedy late in the run. Think about what happens to the largest-budget model in the late stages: it gets *queried* constantly (every proposal uses it) but *updated* rarely (new high-budget evaluations trickle in slowly because they're expensive). So I keep drawing candidates from the same tight, stale `l`, clustering on a handful of good points and never probing their neighborhood widely enough to improve. The fix is to sample from a *widened* version of `l` — the same KDE but with every bandwidth multiplied by a factor `bw > 1`. Widening the kernel spreads my candidates out around the promising configurations instead of right on top of them, which restores exploration exactly when the model would otherwise stagnate, and as a bonus it makes the `N_s` candidates more diverse, which again helps parallel workers. I draw from `l'` (widened `l`) but I *score* with the un-widened ratio `l(x)/g(x)`, because the widening is only about *where to look*, not about *what counts as good*. A factor around `3` is the kind of widening that noticeably spreads the proposals without flinging them into junk regions. And one more density-collapse trap: if every good configuration happens to share the same value on some dimension, that dimension's bandwidth fits to essentially zero and the kernel becomes a spike that can never propose anything else there. So floor every bandwidth at a small `min_bandwidth` (like `1e-3`) to keep a sliver of diversity on every dimension.

Now I have a model-based proposer, but I've quietly endangered the one guarantee Hyperband had for free: random search converges *eventually* because it keeps sampling the whole space. If I always propose from a model, and the model is wrong — the low fidelities are misleading, say — I could starve a whole region forever and never converge. I don't want to trade Hyperband's robustness for guidance. So I keep a constant fraction `rho` of all proposals as *pure uniform random* draws, ignoring the model. This is cheap insurance, and I should put a number on "cheap" rather than wave at it. Suppose I run `m` full cycles through the `s_max + 1` brackets, i.e. `m·(s_max+1)` successive-halving runs, each costing about `(s_max+1)·b_max` in Hyperband accounting. A fraction `rho` of the configuration draws are random, so the random stream contributes about `rho·m·(s_max+1)` full-budget random evaluations; in the same *total* budget, pure random search at `b_max` would do `m·(s_max+1)·(s_max+1)·b_max / b_max` evaluations. The ratio is the worst-case slowdown if the model is actively harmful. Let me plug in `rho = 1/3`, `s_max = 3`, `b_max = 27`, `m = 10`: the random stream gives `(1/3)·10·4 = 13.33` full-budget random evals, total budget is `10·4·(4·27) = 4320`, pure random would do `4320/27 = 160` evals, and `160/13.33 = 12.0`. And `rho^{-1}·(s_max+1) = 3·4 = 12.0` exactly — so the worst-case factor is `rho^{-1}·(s_max+1)`, a constant independent of `m` and `b_max`. Read that as: even if the lower fidelities mislead the model into proposing nothing useful, I am at most a *constant* factor `12` slower than random search here, and I *still converge*, because the random fraction keeps covering the whole space. So `rho` buys global exploration and Hyperband's convergence guarantee at a bounded, constant cost. Something like `rho = 1/3` keeps a healthy stream of random draws without drowning out the guidance. I also fall back to random whenever no budget yet has a fitted, nondegenerate model — there is simply no model to query.

One more design choice I should be deliberate about rather than inheriting blindly. The original way to do these input densities models each hyperparameter with its *own* one-dimensional Parzen estimator — a hierarchy of 1-D densities, effectively a product of 1-D pdfs. That's simple but it assumes the dimensions are independent, which throws away interaction effects: if "high learning rate" is only good *together with* "small batch size," a product of independent 1-D densities can't see that joint structure. Since I'm already paying for a KDE, I'd rather fit a *single multivariate* KDE over the whole vector — a product of 1-D *kernels* (one per dimension, Gaussian for continuous, a categorical kernel for categorical), which is a different object from a product of 1-D *pdfs* and does represent the joint density, so it captures the interactions. For the bandwidths I'll use a cheap rule-of-thumb estimator (Scott's / normal-reference) rather than cross-validated maximum likelihood — preliminary reasoning says the expensive bandwidth search doesn't buy enough accuracy to justify the overhead, and overhead is precisely what I can't afford when the model is queried this often. The discount factor I keep at Hyperband's recommended `eta = 3`; both Hyperband and this method are insensitive to it in a reasonable range, and `eta = 3` keeps the top third at each halving, a sensible aggressiveness.

Let me also be clean about a robustness detail the moment I write code: a training run can crash or return a non-finite loss. Throwing those away biases the data. Better to assign them `+inf` loss and let them count as *bad* configurations — they correctly push the densities away from regions that break training.

So the whole method assembles like this. The bandit skeleton is Hyperband, unchanged: brackets indexed by `s`, geometric budgets `b_max·eta^{-s}`, ideal balanced counts rounded to integer bracket sizes, successive halving keeping the top `1/eta` and multiplying the budget by `eta`. The only thing I've changed is the single line where Hyperband sampled a random configuration: now, with probability `rho` (or if no budget yet has a fitted KDE pair) I still draw uniformly at random; otherwise I use the largest budget with a model, fit good/bad multivariate KDEs on its best/worse split as results arrive, draw `N_s` candidates from the bandwidth-widened good density, and return the candidate minimizing `g(x)/l(x)` — which I checked is the EI-maximizer. Per-budget observation sets are shared and accumulated across all the brackets and iterations, so the models get richer as the run goes on.

Let me write the proposer as real code, filling the `suggest` slot of the search harness. I'll separate the model-fitting, which happens whenever a result comes in, from the proposing, which happens on each `suggest`, and I'll keep configurations encoded into the unit hypercube the way the config space provides.

```python
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm


class BOHB:
    """Model-based Hyperband. Hyperband decides how much budget goes to how many
    configurations; the random sampling of those configurations is replaced by a
    TPE-style density-ratio model (good/bad multivariate KDEs), fit per budget on
    the best/worst split, with EI maximized as argmin g(x)/l(x)."""

    def __init__(self, space, eta=3, top_n_percent=15, num_samples=64,
                 random_fraction=1/3, bandwidth_factor=3, min_bandwidth=1e-3,
                 min_points_in_model=None, seed=42):
        self.space = space
        self.rng = np.random.RandomState(seed)
        self.eta = eta                                  # successive-halving discount
        self.top_n_percent = top_n_percent             # q: percentile that is "good"
        self.num_samples = num_samples                 # N_s candidates per proposal
        self.random_fraction = random_fraction         # rho: keeps HB's convergence guarantee
        self.bw_factor = bandwidth_factor              # widen l for exploration when sampling
        self.min_bandwidth = min_bandwidth             # floor a collapsed bandwidth
        # need at least d+1 points to fit a d-dim KDE
        d = space.dim
        self.min_points = (d + 1) if min_points_in_model is None else max(min_points_in_model, d + 1)
        # statsmodels uses 'c' for continuous and 'u' for unordered categorical;
        # vartypes stores 0 for continuous or the number of categorical choices.
        self.kde_vartypes = "".join('u' if p.type == 'categorical' else 'c' for p in space.params)
        self.vartypes = np.array([len(p.choices) if p.type == 'categorical' else 0
                                  for p in space.params], dtype=int)
        self.configs = {}        # budget -> list of encoded config vectors
        self.losses = {}         # budget -> list of losses (lower is better)
        self.kde_models = {}     # budget -> {'good': KDE, 'bad': KDE}

    # ---- model fitting: called for each finished evaluation -------------------
    def register(self, config, loss, budget):
        # worker returns a validation loss; crashed / non-finite runs count as bad
        loss = loss if np.isfinite(loss) else np.inf
        self.configs.setdefault(budget, []).append(self.space.to_array(config))
        self.losses.setdefault(budget, []).append(loss)

        # don't bother modeling a budget once a larger one already has a model
        if self.kde_models and max(self.kde_models) > budget:
            return
        if len(self.configs[budget]) <= self.min_points - 1:
            return

        X = np.array(self.configs[budget])
        y = np.array(self.losses[budget])
        # split: top q% best for l and worse configurations for g, requesting
        # at least min_points in each density before the shape checks below.
        n_good = max(self.min_points, (self.top_n_percent * X.shape[0]) // 100)
        n_bad = max(self.min_points, ((100 - self.top_n_percent) * X.shape[0]) // 100)
        order = np.argsort(y)
        good = X[order[:n_good]]                 # the N_{b,l} best  -> l(x)
        bad = X[order[n_good:n_good + n_bad]]    # worse configs after the good slice -> g(x)
        if good.shape[0] <= good.shape[1] or bad.shape[0] <= bad.shape[1]:
            return                               # still degenerate in d dims

        # single multivariate KDE (product of 1-D kernels) -> captures interactions
        good_kde = sm.nonparametric.KDEMultivariate(good, var_type=self.kde_vartypes, bw='normal_reference')
        bad_kde = sm.nonparametric.KDEMultivariate(bad, var_type=self.kde_vartypes, bw='normal_reference')
        # floor every bandwidth so a collapsed dimension can't become a spike
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)
        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        self.kde_models[budget] = {'good': good_kde, 'bad': bad_kde}

    # ---- proposing: called on each suggest -----------------------------------
    def sample_config(self):
        # rho fraction random, and random whenever no model exists yet
        if not self.kde_models or self.rng.rand() < self.random_fraction:
            return self.space.sample_uniform(self.rng)

        # use the model for the LARGEST budget that has one (highest trustworthy fidelity)
        budget = max(self.kde_models)
        kde_good = self.kde_models[budget]['good']
        kde_bad = self.kde_models[budget]['bad']
        l = kde_good.pdf
        g = kde_bad.pdf
        # EI(x) is monotone decreasing in g/l, so minimize this (floors avoid 0/0)
        ratio = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

        best_val, best_vec = np.inf, None
        for _ in range(self.num_samples):
            # draw a candidate near a random good datum, from the WIDENED good KDE
            datum = kde_good.data[self.rng.randint(len(kde_good.data))]
            vec = []
            for value, bw, vt in zip(datum, kde_good.bw, self.vartypes):
                bw = max(bw, self.min_bandwidth)
                if vt == 0:                                    # continuous: jitter, widened
                    bw = self.bw_factor * bw
                    a, b = (0 - value) / bw, (1 - value) / bw  # truncate to unit interval
                    vec.append(sps.truncnorm.rvs(a, b, loc=value, scale=bw, random_state=self.rng))
                else:                                          # categorical: keep, or flip
                    if self.rng.rand() < (1 - bw):
                        vec.append(int(value))
                    else:
                        vec.append(self.rng.randint(vt))
            val = ratio(vec)                                   # score with the un-widened ratio
            if val < best_val:
                best_val, best_vec = val, vec
        if best_vec is None:                                   # sampling failed -> fall back
            return self.space.sample_uniform(self.rng)
        return self.space.from_array(best_vec)

    # ---- Hyperband bracket schedule (unchanged from HB) ----------------------
    def bracket_plan(self, min_budget, max_budget):
        max_sh_iter = int(np.log(max_budget / min_budget) / np.log(self.eta)) + 1
        budgets = max_budget * self.eta ** (-np.arange(max_sh_iter - 1, -1, -1))
        brackets = []
        for s in range(max_sh_iter - 1, -1, -1):
            # HpBandSter-style integer count; the pre-rounding ideal balances bracket cost.
            n0 = int(np.floor(max_sh_iter / (s + 1)) * self.eta ** s)
            ns = [max(int(n0 * self.eta ** (-i)), 1) for i in range(s + 1)]
            brackets.append({'budgets': budgets[-(s + 1):], 'num_configs': ns})
        return brackets
```

Let me trace the causal chain once, to be sure it holds together. I started needing an optimizer that is both fast and good, in a mixed, noisy, high-dimensional space, under a budget of only a few full trainings. Black-box Bayesian optimization gave me guidance and final quality but started cold and, with a Gaussian process, broke scalability, mixed-space flexibility, robustness, and efficiency. Hyperband gave me speed by hedging the `n`-versus-`B/n` choice across brackets of successive halving on cheap fidelities, satisfying anytime, scalability, robustness, flexibility, and parallelism — but it sampled configurations blind, so its final quality decayed to random search's. The two are complementary, and the division of labor is that Hyperband is good at *allocating budget* and bad at *choosing configurations*, while a model is the reverse — so I keep Hyperband's bracket skeleton exactly (whose cost balance I checked numerically — `108, 81, 108, 108` across the four brackets, balanced up to integer rounding) and replace only its random configuration draw. For the model I rejected the GP and modeled the inputs split by performance, two KDEs `l` (good) and `g` (bad), because a KDE is linear-time, mixed-space-native, and simple; and I checked numerically — direct EI integration matching the closed form to five decimals — that proposing `argmin g(x)/l(x)` is exactly maximizing expected improvement, so it's principled, not a heuristic. I keep observations separated by budget and always model the largest budget that has a nondegenerate KDE pair, so cheap fidelities bootstrap me early and the high fidelity corrects me late; I need `N_min = d+1` points as the floor for each density, and walking the realized split showed the floor alone isn't enough — the bad slice is starved until `N = 2·N_min`, so I refuse to fit until both realized slices have more rows than dimensions. I optimize EI by drawing `N_s` candidates from a bandwidth-*widened* good density — widened because the high-budget model is queried often but updated rarely, so it would otherwise stagnate, and the partial optimization keeps proposals diverse for parallel workers — scoring each by the un-widened ratio, with a bandwidth floor against collapse. I keep a random fraction `rho` to preserve Hyperband's convergence guarantee at a bounded constant cost (the worst-case slowdown came out to exactly `rho^{-1}·(s_max+1) = 12` in my check), and I use a single multivariate KDE rather than a product of 1-D densities so the model sees hyperparameter interactions. The result drops into the standard ask/tell harness as a per-budget density-ratio proposer behind Hyperband's unchanged scheduler.
