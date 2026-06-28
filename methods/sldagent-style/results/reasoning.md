Let me start from what actually goes wrong when I try to characterize a new scaling regime. I have a pile of training runs — say a few thousand rows, each one a setting of some descriptors `x` (parameter count `N`, tokens `D`, maybe vocabulary `V`, learning rate `l`, batch size `b`, number of experts `E`, unique tokens `U`) paired with a measured loss `y`. I want a compact formula `f_theta(x) -> y_hat` that, fit on the cheap small-scale rows, predicts the expensive large-scale rows I have not run. The whole point is extrapolation: I will hold out the largest models and datasets and ask whether a law fit on everything else lands on them. And the way this gets done today is that a person sits down, guesses a form by analogy to Chinchilla — loss is an irreducible floor plus a couple of power-law terms, `L = E + A/N^a + B/D^b` — fits the coefficients with BFGS, eyeballs the residuals, and tweaks. It works when the regime is close to the base case. It stops working the moment the axes interact or the asymptotics matter, and it is bounded by what one human can hold in their head about three or four coupled variables. I want to take the human out of the loop. So the question is: out of a fixed table of `(x, y)` rows, how do I *automatically* produce a symbolic law that extrapolates?

The first instinct is the obvious one: this is symbolic regression. Search over expression trees built from `{+, -, *, /, pow, log, exp}` and a handful of input variables, score each tree by how well it fits, evolve the good ones. Koza's genetic programming does exactly this; Schmidt and Lipson distilled free-form natural laws from pendulum data this way; SINDy does it with a sparse regression over a fixed library of candidate terms; AI Feynman prunes the tree search with physics-inspired heuristics. So why not just point one of these at my table? I keep running into the same three walls. First, the space of expressions is combinatorial — even with a tiny operator set and four variables, the number of trees of modest depth explodes, and a blind tree search wastes almost all its budget on syntactically valid but physically absurd forms. Second, these methods carry essentially no domain prior: they do not know that loss should have an irreducible floor, that capacity and data terms decay as power laws, that a coefficient ought to keep a sensible sign. They are happy to fit a high-degree polynomial that nails the training region and then dives to minus infinity just outside it — which is poison for an extrapolation task. Third, and this one is specific to my problem and I should not gloss over it: my data does not come as one curve. It comes as *groups* — different corpora, different architectures, different model families — and the thing I want is a *single shared form* whose coefficients differ per group, not a separate formula per group. Classical symbolic regression has no real handle on "discover one expression that generalizes across dozens of contexts." It is built to rediscover a single known equation from synthetic data, which is closer to curve fitting than to the open-ended thing I am facing, where nobody — not even the experts who study the regime — knows the right answer.

So pure tree search is the wrong engine. But I notice the *shape* of my problem matches something else that has been working lately. Think about the recent string of results where an open-ended scientific or algorithmic problem got solved by *searching over programs* scored by an automatic, objective evaluator: faster matrix-multiplication algorithms, an improved diagonal-Ramsey bound, faster sorting routines. The common structure is striking. There is a clear, continuous, machine-computable objective. The optimum is unknown. And progress does not come from one brilliant proposal — it comes from *generations* of incremental improvement, each candidate standing on the shoulders of the ones before it. My problem has exactly that signature: the scientific target is extrapolation `R^2`, target one, computed directly from data with no learned reward model and no human judge; the best law is genuinely unknown; and a single shot, however clever, is not going to find it — I will need to keep pushing the boundary of the best solution so far. During search I can only score on the seen split, but the score is still a direct numerical proxy for fit rather than an LLM opinion. That points me at *evolutionary program search*, not a one-pass solver and not a blind tree search.

And the natural way to drive that search now is with a language model. FunSearch is the template: pair a pretrained LLM with an automated evaluator and an island-based evolutionary loop. You keep a database of scored, valid programs. Each step you sample a few high-scoring ones, drop them into a "best-shot" prompt — here are some good programs, write me a better one — let the LLM propose a candidate, execute it, score it, and if it runs and scores well, put it back in the database. The evaluator is doing real work here beyond ranking: it is a guard against the model's confabulation. The LLM will cheerfully assert that some formula is great; the evaluator refuses to take its word and only lets programs that *actually run and fit the data* survive and propagate. That is exactly the discipline I need, because I cannot trust an LLM's stated belief that a law extrapolates — I can only trust the measured fit. FunSearch found real cap-set constructions and bin-packing heuristics this way, so the engine is proven. I will adopt it.

But now I have to ask the question that decides whether this actually improves on the human workflow: *what, exactly, is the unit of evolution?* What is the gene? FunSearch's answer is a single function body sitting inside a fixed program skeleton — it evolves the *what*, the candidate solution, while the surrounding harness is hand-written and frozen. If I copy that literally, my gene is the symbolic form: a function `scaling_law_func(x, theta)` that, given inputs and a parameter vector, returns a prediction. The skeleton around it — including the routine that *fits* `theta` to the data — I would write once, by hand, presumably as a standard optimizer like BFGS, and leave fixed. That is the cheapest thing to build and it inherits a proven recipe, so I should take it seriously and only abandon it if I can show it actually breaks. Let me try it.

Picture the easy regimes first. On something near Chinchilla, where a simple additive power law is already most of the answer, evolving the form alone with a fixed BFGS fitter does fine — the form is benign, the loss surface in `theta` is well-behaved, BFGS finds the coefficients, the fit is good. So if I only looked at the easy tasks I would conclude FunSearch's recipe transfers cleanly and call it a day. The hard regime is learning rate and batch size: four inputs, no established full-loss law, and the candidate forms the LLM proposes there are genuinely awkward — products of powers across four axes, quadratic basins in log-learning-rate, interaction terms. My worry is that the moment the form gets awkward, the *fitting* gets awkward too, and a vanilla BFGS from a generic start can no longer recover the coefficients even when the form is exactly right. That is a worry, not a fact, so let me actually check it before I redesign anything around it.

I will set up the cleanest possible test of the worry. Take a form I *know* is correct — generate a synthetic dataset from a fixed log-quadratic surface in `(ln l, ln b)` with scale drift, on a realistic grid (`l` in `[2.4e-4, 2.2e-2]`, `b` in `[16, 2048]`, `D` spanning `1e9`–`1e11`, `N` spanning `1e7`–`1e9`, 400 rows, a whisper of noise). Because I generated the data *from* this form, the right answer exists and a good fitter must recover it. Now fit the *same* form two ways and read off the seen-split `R^2`. Fitter A is the naive FunSearch-style default: minimize mean-squared error directly with BFGS from a sensible intercept-aware start, with a few restarts. Fitter B treats the form as what it is in log-space — linear in its coefficients — and does a single ridge least-squares solve. Running it:

```
ridge least-squares (log-space)            R^2 = 0.989
naive direct-MSE BFGS (best of 8 restarts) R^2 = -81.4   (nit=1, did not converge)
```

That settles the worry, and harder than I expected. On the *identical* functional form, the closed-form solve recovers it almost perfectly while the naive optimizer comes out at `R^2 = -81` — worse than predicting the mean. And I can see exactly why once I look: the design features span `ln D ≈ 23` and `ln N ≈ 16`–`21`, so `exp` of any non-tiny coefficient on those columns overflows, the direct-MSE surface is a cliff with vast flat regions, and BFGS stalls at the first iteration. The *form was right the whole time.* If I had been running the frozen-fitter recipe inside the evolutionary loop, this candidate would have been handed back an `R^2` of −81, scored as garbage, and evicted — and I would never have learned that the problem was the fitter, not the form. So I am throwing out the frozen-skeleton design. It is not a stylistic preference; it just measurably discards correct hard-regime forms as false negatives.

Now let me name the thing precisely, because the fix follows from that measurement. A scaling law, as an object I have to *produce*, is not one thing. It is two: the symbolic form `Expression(x, theta)` that says what shape the law has, and the procedure `Optimization(x, y)` that says how to turn observed rows into fitted coefficients `theta`. Human practice has always treated the second as a fixed background utility — "just BFGS it" — and FunSearch inherits that assumption by freezing the skeleton. But the `R^2 = -81` versus `R^2 = 0.99` gap I just measured says the second is *coupled* to the first and is load-bearing exactly when the form is hard. A power law with quadratic interactions across four log-scaled axes is not something you robustly fit by dropping it into a generic optimizer from a generic start; it wants the right loss (log-space, robust to outliers), the right initialization, the right restarts, maybe a closed-form linear sub-solve. So the gene cannot be the form alone. The gene has to be the *pair* — `(Expression, Optimization)` — co-evolved together. When the LLM proposes a more aggressive form, it gets to propose a fitter tailored to that form in the same breath; when it changes the fitter, the form it is fitting comes along. That is the move the failed experiment forces: evolve the form *and* its parameter-fitting routine jointly, as a single candidate program, because a great form with a naive fitter is worthless and the two have to be designed against each other.

I want to be sure the gap I measured is the coupling and not an artifact of my synthetic setup, so let me also name the falsification I would run on the real system. If co-evolving the pair were just decoration, then evolving the form alone with a fixed good optimizer should do essentially as well on the real benchmarks too. The discriminating case is precisely the regime where fitting bites — learning-rate/batch-size. So the diagnostic after the system exists is: freeze the fitting routine, evolve only the expression, and compare it to the co-evolved pair on that regime. My synthetic check already shows the mechanism is real on a controlled instance; this diagnostic is what would confirm it survives contact with the messy data. Either way the design follows from the object itself: discovering a good symbolic expression is only half the battle, and tailoring the optimizer to that expression and that data is what keeps hard candidates from being discarded.

Good. The gene is the pair. Now I have to build the population and the loop around it, and here I do want to lean on what already exists rather than reinvent it, because the database and selection machinery is generic and well understood. The danger in any long evolutionary run is collapse: the population converges onto one local optimum, every candidate becomes a minor variation of the current champion, and the search stops exploring. The cure I reach for is quality-diversity, MAP-Elites: do not keep just the single best program; carve a feature space into a grid of cells and keep one elite — the best-fitness program — *per cell*, where the cells are defined by descriptors of the candidate. What descriptors? The bounded combined score has to be one of them, so high-quality programs survive. But score alone re-collapses the population, so I add axes that measure *what kind* of solution it is: its complexity (a simple two-term power law versus a sprawling interaction model) and its novelty/diversity relative to the rest. Bin each of these into ten bins and now the database is forced to retain a simple candidate *and* a complex candidate *and* a novel candidate even if one of them currently scores a bit lower, because they live in different cells and only compete locally. That keeps cheap parsimonious forms alive alongside elaborate ones, instead of letting a slightly-better-fitting monster evict them. On top of MAP-Elites I run five islands — five sub-populations — that evolve mostly in isolation and exchange top migrants every twenty-five generations. Islands let different basins of the formula space get explored independently; a form that would be out-competed and killed in the main population can develop on its own island until it is good enough to migrate. Migration shares progress without homogenizing everyone onto one idea. This is the right scaffolding because my problem is open-ended and premature convergence is the failure mode I most fear.

Now the selection step — which parent do I mutate next? I want to mostly exploit, because I do have a clear objective and I want to make progress on it, but I cannot exploit so hard that I stop exploring an open space. So split the draw three ways. Most of the time — seventy percent — sample a parent from the elite archive, the high scorers, and try to improve on something already good. A solid chunk — twenty percent — sample uniformly from the current island, so the search keeps poking at parts of the space the champions ignore. The remaining ten percent uses the fallback path in the database — random globally in the shared sampler, fitness-weighted inside the island-specific sampler — while the archive and explicit best-program tracking preserve elites separately. Seventy / twenty / ten gives me exploit, explore, and residual mixing without letting one sampling mode own the run. Alongside the chosen parent I pull a few "inspiration" programs from the same island and also place the island's top programs in the prompt history, so the LLM sees not just one program to mutate but a small spread of good ideas it can recombine. That is the best-shot prompting from FunSearch, generalized: concrete high-quality examples are far more useful to the model than an abstract instruction.

What goes in the prompt, beyond the programs? The LLM is being asked to write a better `(Expression, Optimization)` pair, so I give it what a human scientist would want: the experimental context (what the variables mean), the inspiration programs, and concrete *data statistics* — the ranges, means, and variances of each input and the target — so it can reason about scales and signs rather than guessing blind. (That synthetic experiment is also a reminder of why the stats matter: the only reason I could anticipate the overflow is that `ln D` runs to ~25, and the LLM should see those ranges for the same reason.) I also hand it the mathematical constraints that keep the result honest. Parameter discipline is the important one: some tasks get an explicit coefficient cap, while the learning-rate/batch-size task is allowed to choose the count but is still told to keep the law parameter-efficient. The reason is overfitting. With the test region held out and no separate validation split, the only thing standing between me and a law that memorizes the seen rows is parsimony, and a cap or strong parameter-efficiency instruction is the bluntest, most reliable form of it. I also forbid the form from using *input-dependent statistics* of the batch — no `median`, `min`, `max` of the incoming points inside `Expression`. That one is subtle but worth being strict about: if the law could peek at the min or max of the inputs it is predicting on, it would leak the test distribution into its own definition and look great on the seen rows while failing to extrapolate, because at deployment those statistics are different. The law has to be a pure function of `(x, theta)`. And I pin the function *signatures* — `scaling_law_func(data_points, params)` and `fit_scaling_law(data_points, loss_values)` — so that whatever the LLM writes can actually be loaded and called by the evaluator. The evolvable region of the file I mark off explicitly, between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`, so the LLM rewrites the law and the fitter but cannot touch the plumbing.

I need a seed to bootstrap from — the first program in an empty database. It should be the simplest thing that runs end to end: a generic additive power-law `Expression` (each input raised to a fitted exponent, scaled by a fitted coefficient, summed, plus a bias — a Chinchilla-flavored baseline that works for any number of inputs) and a naive `Optimization` that just minimizes mean-squared error with BFGS from a flat initialization. It will not be good, especially on the hard regimes — that is the point, and it is the very pairing my synthetic test showed is weak. It executes, it gets a fitness, it goes into the database, and evolution takes over from there. Then the loop is the standard one: seed the database; for a fixed budget of iterations — fifty keeps the run finite while giving the population multiple generations to accumulate improvements — sample a parent and inspirations, build the prompt, ask the LLM for a full rewrite of the evolvable block (full rewrite rather than a tiny diff, because the programs are short and I want the model free to restructure both the form and the fitter at once), execute the child, score it on the seen split, and insert it back with its fitness. Crucially the test split is never touched anywhere in this loop — fitness is computed only on seen data. At the end, return the highest-scoring program in the database as the discovered law. Evaluation is purely numerical fitness — no LLM-as-judge, no cascade of cheap-then-expensive checks — for objectivity, and I run several program evaluations in parallel with a per-program timeout and a couple of retries so a single crashed or slow candidate does not stall the run. A program that errors or times out simply gets the failure floor for its fitness and is effectively discarded.

What is the fitness, concretely? Per group, fit the candidate's parameters with the candidate's own `fit_scaling_law` on that group's seen rows, predict on seen rows, concatenate the errors, and score. The metric I actually optimize is built from the normalized mean-squared error: `NMSE = mean((y - y_hat)^2) / Var(y)`, and `R^2 = 1 - NMSE`, with a `combined_score = 1 / (1 + NMSE)` that maps a perfect fit (NMSE = 0) to one and a hopeless fit toward zero and is monotone in `R^2`, which keeps the fitness bounded and well-behaved for the database's binning. That is the signal the whole evolution climbs.

Now let me make the pair concrete on that hard learning-rate/batch-size regime, since it is where the whole co-evolution argument was forced. The inputs are `(l, b, D, N)` and I want `lm_loss`. Loss as a function of learning rate and batch size has a basin — there is an optimum and I pay for being off it in either direction — so the natural move is to work in log-coordinates, `x = ln l` and `y = ln b`, where the basin looks quadratic, and to model the *log* of the loss as a quadratic surface in `(x, y)` whose center and curvature drift with scale `(N, D)`. Write it out: collect everything constant in `(x, y)` into a scale-dependent intercept `C(N, D)` and let the rest be a full quadratic with the cross-axis couplings the additive human forms could not express,

```
log(loss) = C(N, D) + b3*y + b4*x + b5*x^2 + b6*y^2 + b7*x*y + b9*(ln N)*y + b10*(ln D)*x.
```

The `b5 x^2`, `b6 y^2` give the basin in each axis; `b7 x y` couples learning rate and batch size, which is exactly the interaction that hand-additive laws drop; `b9 (ln N) y` and `b10 (ln D) x` let the optimum drift with model and data scale. The fitter drops out of the algebra, and this is precisely the property that fitter A choked on and fitter B exploited in the synthetic test: this form is *linear in its coefficients*. Every term is a known function of the inputs times an unknown `b`. So the right `fit_scaling_law` is not BFGS at all — it is a single linear least-squares solve. Build the design matrix `Z` whose columns are the feature functions — a constant, `ln P`, `ln D`, `ln b`, `ln l`, `(ln l)^2`, `(ln b)^2`, `(ln l)(ln b)`, `(ln P)(ln D)`, `(ln P)(ln b)`, `(ln D)(ln l)` — regress `ln y` on `Z`, and solve the normal equations with a whisper of ridge for conditioning: `(Z^T Z + lambda I) beta = Z^T ln y` with `lambda` around `1e-6`. One solve, globally optimal for this form, no local minima, using *all* the rows. That is exactly the configuration that scored `R^2 = 0.99` where the naive fitter scored `-81` — the synthetic check was, in retrospect, a dry run of this exact form-and-fitter pair. Contrast it with the prior art for this regime — Step Law fit only the *optima*, `lr* = c N^a D^b` and `bsz* = d D^g`, from about seventeen best-performing configurations out of three thousand experiments, throwing the rest away. This law models the entire loss surface from all the data and is fit in closed form. The form (a log-quadratic with interactions) and the fitter (ridge least-squares, not BFGS) are designed for each other, and neither alone would have gotten here.

And the log-quadratic form hands me something the prior optima-only approach gave for free but mine has to *derive*, and that derivation is worth doing carefully because it is the practical payoff. Since `log(loss)` is quadratic in `(x, y)`, minimizing loss is minimizing that quadratic, and a quadratic has its minimum where its gradient vanishes. Take the partials and set them to zero:

```
d/dx: b4 + 2*b5*x + b7*y + b10*ln D = 0,
d/dy: b3 + b7*x + 2*b6*y + b9*ln N = 0.
```

This is a 2x2 linear system in `(x, y)`. Write it in matrix form with `H` the Hessian of the quadratic,

```
[ 2*b5   b7  ] [ x ]     [ b4 + b10*ln D ]
[  b7   2*b6 ] [ y ]  = -[ b3 +  b9*ln N ].
```

For this stationary point to be a genuine minimum I need `H` positive-definite, which for a 2x2 symmetric matrix means positive diagonal and positive determinant `Delta = det(H) = 4*b5*b6 - b7^2 > 0` — and that is something I can *check* against the fitted coefficients rather than assume; if the search returns a form whose fitted `H` is positive-definite, the basin is real and the minimizer unique. Solve the system by Cramer's rule. With `A = 2 b5`, `C = 2 b6`, `B = b7`, and right-hand side `(-(b4 + b10 ln D), -(b3 + b9 ln N))`, the determinant is `Delta = A C - B^2 = 4 b5 b6 - b7^2`, and

```
x* = [ -2*b6*(b4 + b10*ln D) + b7*(b3 + b9*ln N) ] / (4*b5*b6 - b7^2),
y* = [  b7*(b4 + b10*ln D) - 2*b5*(b3 + b9*ln N) ] / (4*b5*b6 - b7^2).
```

So the optimal log-learning-rate and log-batch-size come out *affine in `ln N` and `ln D`* — collect the constant, the `ln D` coefficient, and the `ln N` coefficient and I can read off `x* = (-C b4 + B b3)/Delta + (-C b10)/Delta * ln D + (B b9)/Delta * ln N` and likewise for `y*`. Exponentiate to get the optima in natural units, `lr* = exp(x*)` and `bsz* = exp(y*)`. I should not trust the algebra until I have put numbers through it, so take a fitted coefficient vector of this form: `b3 = 0.0595, b4 = 0.1906, b5 = 0.0098, b6 = 0.0073, b7 = -0.006, b9 = -0.0089, b10 = -0.0012`. Then `A = 0.0196, C = 0.0146, B = -0.006`. First the positive-definiteness check: `Delta = A C - B^2 = 0.0196*0.0146 - (-0.006)^2 = 0.00028616 - 0.000036 = 0.00025016 > 0`, and the diagonal `2 b5, 2 b6` is positive, so `H` is positive-definite — the basin is real and the stationary point is a true minimum, not a saddle. Now read off the affine coefficients: `x* = -12.5510 + 0.070035 ln D + 0.213463 ln N` and `y* = -9.23329 + 0.028782 ln D + 0.697314 ln N`. For a one-billion-parameter model on a hundred billion tokens — `N = 2^30 ≈ 1.0737e9`, `D = 1e11`, so `ln N ≈ 20.7944`, `ln D ≈ 25.3284` — this gives `x* = -12.5510 + 0.070035·25.3284 + 0.213463·20.7944 = -6.3383` and `y* = -9.23329 + 0.028782·25.3284 + 0.697314·20.7944 = 5.9959`, so `lr* = exp(-6.3383) ≈ 1.767e-3` and `bsz* = exp(5.9959) ≈ 401.8`. Both land in plausible ranges — a learning rate of ~1.8e-3 and a batch of ~400 for a 1B/100B run are not absurd — which is a reassuring smell test on top of the algebra checking out. Fitting the whole loss surface, not just the optima, gives an analytic hyperparameter recommendation with no extra sweep. That is the kind of object the optima-only prior law could not provide, because it never modeled the surface whose gradients I just used.

The same design pressure tells me what kinds of forms I want the search to keep alive on other regimes: not just high seen-split fit, but safer asymptotics and cleaner units. Take mixture-of-experts. The human law is an exponential of a log-bilinear form, `log L = a log N + b log E_hat + c (log N)(log E_hat) + d`, and I suspect its weakness is its asymptotics, so let me push it rather than assert it. If the linear terms are small relative to a positive interaction coefficient `c`, then as `N` and `E` scale up together the exponent is dominated by `c (log N)(log E)` and grows without bound. Putting numbers through it with `a=-0.1, b=-0.05, c=0.02, d=1` and scaling `(N, E)` from `(1e9, 32)` upward, the predicted loss climbs `1.2 -> 2.0 -> 4.2 -> 11.4 -> 40 -> 182` and keeps going — exactly the wrong tail behavior for a loss, which should be bounded below by an entropy floor and never diverge. (With larger negative linear terms the divergence is masked in the seen range, so it is not unconditional — but the form *permits* it, which is the problem.) A candidate like `L = t1 N^t2 / (1 + t3 E^t4) + t5 N^{0.6 t2} + t6` cannot do that. Read what each piece does: `t1 N^t2` is a parameter-driven term attenuated by `1/(1 + t3 E^t4)`, a multiplicative discount that captures diminishing returns from adding experts; `t5 N^{0.6 t2}` is a second parameter term following the same exponent family at a reduced rate; and `t6` is an additive irreducible floor. Now check the limits — and check them numerically rather than just reading them off. With `t1=12, t2=-0.34, t3=0.5, t4=0.8, t5=3, t6=1.85` and `N=1e9`, sweeping `E = 1, 8, 64, ..., 1e9` gives `L = 1.9007, 1.8966, 1.8945, ... -> 1.893764`, and the claimed limit `t5 N^{0.6 t2} + t6` evaluates to `1.893764` — they agree, so as `E -> infinity` the first term's denominator grows, that term vanishes, and `L` settles to a finite floor plus the reduced-rate term. And sweeping `N = 1e7 ... 1e15` at fixed `E=64` gives `L = 1.9653, 1.9215, 1.8945, 1.8777, ..., 1.8526`, decreasing monotonically toward `t6 = 1.85` because `t2 < 0` makes both `N^t2` and `N^{0.6 t2}` decay — a bounded limit, as it must be. So this form keeps a sensible tail by construction where the human one can run away. Or take supervised fine-tuning: the human "rectified" power law is `L = t2 + t0/(D^{t1} + t3)`, where the offset `t3` sits additively next to `D^{t1}`, so its units depend on the exponent `t1` and it has no clean interpretation. A cleaner parameterization is `L = t2 + t0/(1 + (D/t3)^{t1})`, with the offset folded into a dimensionless ratio `(D/t3)^{t1}`. Now `t3` carries the natural units of dataset size and reads directly as the characteristic data scale at which the curve transitions from steep improvement to saturation. These are the forms I want the evolutionary loop to be able to discover: not larger expression trees for their own sake, but expressions whose tails and units keep making sense after fitting.

Let me also be honest with myself about what could go wrong, because the design has a soft spot. Fitness is computed on the seen split with no held-out validation set inside the loop. So nothing in the mechanism *prevents* the search from finding a form that fits the seen rows by exploiting an artifact rather than a mechanism. The parameter budget or parameter-efficiency instruction and the ban on input-dependent features are my two defenses — parsimony, and forbidding the law from peeking at the distribution it is predicting on. The MAP-Elites complexity axis keeps simple candidates competitive instead of letting a slightly better seen-split fit automatically evict them, but it is still only a proxy for mechanism. "Self-verification" — the agent deciding on its own which of two equally-fitting laws is the more robust mechanism — remains a real gap, so I keep the parameter discipline and the feature ban as hard constraints, not soft preferences, and I do not pretend they close the gap entirely.

Now let me put the whole thing down as the code I would actually run, filling the slots the harness left open. The gene is the pair `(scaling_law_func, fit_scaling_law)` living inside the EVOLVE block; the seed is the naive power law plus BFGS; the loop reuses the quality-diversity database and the 70/20/10 sampling; the prompt carries inspirations, data stats, parameter discipline, and fixed signatures.

```python
import numpy as np
import uuid
from scipy.optimize import minimize
from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


# ---------- the SEED candidate program (the first gene in the database) ----------
# Everything between the markers is what the LLM is allowed to evolve: BOTH the
# symbolic form AND the routine that fits its parameters. The seed is a generic
# additive power law fit by naive BFGS, including multi-target handling.

SEED_PROGRAM = r'''
# EVOLVE-BLOCK-START
"""Seed: generic additive power law + naive BFGS, for any number of inputs."""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    X = np.atleast_2d(np.asarray(data_points))          # (N, F)
    Npts, F = X.shape
    params = np.asarray(params)
    if params.ndim == 1:
        params = params[None, :]
    T, P = params.shape
    coeffs    = params[:, :F]                            # one coeff per input
    exponents = params[:, F:2 * F]                       # one exponent per input
    bias      = params[:, -1]                            # irreducible floor
    pred = (coeffs[None, :, :] * (X[:, None, :] ** exponents[None, :, :])).sum(axis=2) \
           + bias[None, :]
    return pred[:, 0] if pred.shape[1] == 1 else pred

def fit_scaling_law(data_points, loss_values):
    X = np.atleast_2d(np.asarray(data_points))
    y = np.asarray(loss_values)
    Npts, F = X.shape
    P = 2 * F + 1
    y2d = y[:, None] if y.ndim == 1 else y
    T = y2d.shape[1]
    init = np.ones((T, P))
    def objective(flat_params):
        params = flat_params.reshape(T, P)
        pred = scaling_law_func(X, params)
        return np.mean((pred - y2d) ** 2)                   # naive MSE
    res = minimize(objective, init.ravel(), method="BFGS")
    params_opt = res.x.reshape(T, P) if res.success else init
    return params_opt[0] if T == 1 else params_opt
# EVOLVE-BLOCK-END
'''


# ---------- fitness: score a candidate on the SEEN split only ----------

def fitness_of(program_module, seen_data_by_group):
    """For each group: fit the candidate's params with its own fitter, predict,
    accumulate squared error and variance. Return combined_score = 1/(1+NMSE),
    monotone in R^2 and bounded in (0, 1]. The test split is never touched here."""
    sse, sst = 0.0, 0.0
    for _group, (X, y) in seen_data_by_group.items():
        try:
            theta = run_with_timeout(program_module.fit_scaling_law, (X, y))
            pred  = np.asarray(program_module.scaling_law_func(X, theta), dtype=float)
            if not np.all(np.isfinite(pred)):
                return 0.0                                  # failure floor
            sse += float(np.sum((y - pred) ** 2))
            sst += float(np.sum((y - np.mean(y)) ** 2))
        except Exception:
            return 0.0                                      # crash/timeout -> floor
    nmse = sse / sst if sst > 0 else np.inf
    return 1.0 / (1.0 + nmse)                               # R^2 = 1 - nmse


# ---------- prompt construction ----------

def build_prompt(parent_code, inspirations, task_ctx, data_stats, parameter_instruction):
    inspo = "\n\n".join(f"# inspiration (score={s:.4f})\n{c}" for c, s in inspirations)
    return f"""You evolve BOTH `scaling_law_func` (the symbolic law) AND
`fit_scaling_law` (the routine that fits its parameters) for: {task_ctx}.

{parameter_instruction}
Do NOT use input-dependent statistics (median/min/max of data_points) in
scaling_law_func -- that leaks the test distribution and breaks extrapolation.
Keep the signatures `scaling_law_func(data_points, params)` and
`fit_scaling_law(data_points, loss_values)`. Write all changes between
# EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END.

Data statistics (ranges, means, variances): {data_stats}

High-scoring programs to build on:
{inspo}

Current program to improve:
{parent_code}
    """


# ---------- the evolutionary loop (reused MAP-Elites + islands structure) ----------

def add_scored_program(db, code, score, parent_id=None):
    program = Program(
        id=str(uuid.uuid4()),
        code=code,
        parent_id=parent_id,
        metrics={"combined_score": float(score)},
    )
    db.add(program)
    return program

def discover(task, n_iterations=50, n_islands=5,
             parameter_instruction="Keep the law parameter-efficient; use a task-specific cap where configured."):
    seen = load_seen_data(task)                             # {group: (X, y)}
    db_config = DatabaseConfig(                             # MAP-Elites + islands
        population_size=100,
        archive_size=50,
        num_islands=n_islands,
        feature_dimensions=["combined_score", "complexity", "diversity"],
        feature_bins=10,
        exploitation_ratio=0.70,
        exploration_ratio=0.20,
        elite_selection_ratio=0.10,
        migration_interval=25,
        migration_rate=0.10,
    )
    db = ProgramDatabase(db_config)
    seed_mod = load_module(SEED_PROGRAM)
    seed_program = add_scored_program(db, SEED_PROGRAM, fitness_of(seed_mod, seen))

    for it in range(n_iterations):
        parent, inspirations = db.sample(num_inspirations=3)        # parent + best-shot
        inspiration_items = [(p.code, p.metrics.get("combined_score", 0.0)) for p in inspirations]
        prompt = build_prompt(parent.code, inspiration_items,
                              task.context, task.data_stats, parameter_instruction)
        child_code = llm_propose(prompt)                    # full rewrite of EVOLVE block
        try:
            child_mod = load_module(child_code)
            score = fitness_of(child_mod, seen)             # seen split only
        except Exception:
            score = 0.0
        add_scored_program(db, child_code, score, parent_id=parent.id)  # insert; test untouched

    return db.get_best_program()                            # highest-scoring law found


# ---------- one hard-regime candidate form: the lr&bsz log-quadratic ----------
# A form that is linear in its coefficients, paired with closed-form ridge least squares.

def scaling_law_func_lrbsz(data_points, params):
    X = np.asarray(data_points, dtype=float)               # cols: lr, bsz, D, N
    if np.any(X <= 0):
        raise ValueError("inputs must be positive for log transform")
    l_lr, l_b, l_D, l_P = (np.log(X[:, 0]), np.log(X[:, 1]),
                           np.log(X[:, 2]), np.log(X[:, 3]))
    Z = np.column_stack([                                   # design matrix (linear in beta)
        np.ones_like(l_lr), l_P, l_D, l_b, l_lr,
        l_lr ** 2, l_b ** 2, l_lr * l_b,                   # basin + lr/bsz coupling
        l_P * l_D, l_P * l_b, l_D * l_lr,                  # scale-dependent drift
    ])
    return np.exp(Z.dot(np.asarray(params, dtype=float)))  # predict in log-loss space

def fit_scaling_law_lrbsz(data_points, loss_values):
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float).ravel()
    l_lr, l_b, l_D, l_P = (np.log(X[:, 0]), np.log(X[:, 1]),
                           np.log(X[:, 2]), np.log(X[:, 3]))
    l_y = np.log(y)
    Z = np.column_stack([
        np.ones_like(l_lr), l_P, l_D, l_b, l_lr,
        l_lr ** 2, l_b ** 2, l_lr * l_b,
        l_P * l_D, l_P * l_b, l_D * l_lr,
    ])
    lam = 1e-6                                              # tiny ridge for conditioning
    A = Z.T.dot(Z) + lam * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T.dot(l_y))                # one global linear solve


def optimal_lr_bsz(beta, N, D):
    """Closed-form minimizer of the log-quadratic in (x=ln lr, y=ln bsz).
    beta indices follow the design matrix above: x=ln lr, y=ln bsz, with
    coefficients b4 (l_lr), b3 (l_b), b5 (l_lr^2), b6 (l_b^2), b7 (l_lr*l_b),
    b10 (l_D*l_lr), b9 (l_P*l_b). Solves H [x;y] = -rhs, H = [[2b5,b7],[b7,2b6]]."""
    b3, b4, b5, b6, b7 = beta[3], beta[4], beta[5], beta[6], beta[7]
    b9, b10 = beta[9], beta[10]
    Acoef, Bcoef, Ccoef = 2 * b5, b7, 2 * b6
    Delta = Acoef * Ccoef - Bcoef ** 2                     # det(H) > 0 => real minimum
    if Delta <= 0:
        raise ValueError("the fitted log-quadratic is not positive-definite in lr/bsz")
    rx, ry = b4 + b10 * np.log(D), b3 + b9 * np.log(N)
    x_star = (-Ccoef * rx + Bcoef * ry) / Delta            # Cramer's rule
    y_star = ( Bcoef * rx - Acoef * ry) / Delta
    return np.exp(x_star), np.exp(y_star)                  # lr*, bsz* as funcs of (N, D)
```

The causal chain is now tight. I start stuck: scaling laws are hand-crafted one regime at a time, slow and bounded by what a person can reason about across coupled variables, and the hard regimes expose the brittleness of fixed human forms. Pure symbolic regression is the wrong engine — combinatorial blow-up, no domain prior, no way to share one form across many groups. But the problem has the signature of open-ended program search against a clear computable objective with an unknown optimum, where progress is generational — so the engine is LLM-driven evolutionary program search, with an automatic evaluator guarding against the model's confabulation. The pivotal choice was the unit of evolution: FunSearch evolves only the symbolic form and freezes the fitter, but fitting the *same correct* log-quadratic form two ways gave `R^2 = -81` for the naive BFGS default versus `R^2 = 0.99` for a closed-form solve — a correct form scored as garbage purely because of the fitter — so the gene has to be the *pair*, the form and its parameter-fitting routine co-evolved. Around that gene I reuse the quality-diversity machinery — MAP-Elites over (combined score, complexity, diversity), five islands with periodic migration, and 70/20/10 parent sampling — plus a best-shot prompt carrying inspirations, data statistics, parameter discipline for parsimony, a ban on input-dependent features to stop test-distribution leakage, and pinned signatures so candidates stay runnable. Fitness is `1/(1+NMSE)` on the seen split only; the test split is never touched; the best program after fifty iterations is the discovered law. On the lr&bsz regime, the kind of program this makes possible is a log-quadratic form linear in its coefficients paired with a closed-form ridge least-squares fit that uses all the rows instead of only optima, and the same fitted form gives analytic `lr*` and `bsz*` (checked: positive-definite `Delta = 0.00025016`, `lr* ≈ 1.767e-3`, `bsz* ≈ 401.8` at 1B/100B) by solving a two-by-two system.
