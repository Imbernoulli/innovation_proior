The function is the whole task, and before I reach for anything nonlinear I want the floor — the
simplest principled fill of `run_causal_discovery` that already returns *directed* edges, not a
correlation blur. The floor matters because it tells me how much of the work is the *nonlinearity*
versus the *direction*: if a method that ignores nonlinearity entirely already does well, the
nonlinear machinery the later rungs add is wasted; if it falls over, I have measured exactly the gap
the rest of the ladder has to close. So I deliberately start with a method that breaks
forward/backward symmetry by the *other* identifiability route — non-Gaussian noise, not nonlinear
mechanisms — and is linear throughout. That is DirectLiNGAM, and it is the floor by construction
because the data here is generated nonlinearly while this method assumes linearity.

Let me write down what actually blocks me, because the floor still has to be principled. I have `X`
of shape `(n_samples, n_variables)`, no interventions, no time stamps, nothing telling me which
variable is upstream. The one object I can compute most cheaply from observational data is the
covariance, and that is exactly where the field's difficulty lives. Take two correlated variables
with joint covariance `Σ`. Model one, `x_2 = b x_1 + e_2`, and model two, `x_1 = b' x_2 + e_1`, each
fix their coefficient and residual variance by least squares — `b = ρσ_2/σ_1`,
`var(e_2)=σ_2²(1-ρ²)` forward, `b'=ρσ_1/σ_2`, `var(e_1)=σ_1²(1-ρ²)` backward — and both reconstruct
the *identical* `Σ`. The covariance cannot tell them apart, and this is not a defect of one
estimator: PC's conditional-independence tests and GES's DAG scores are also functions of the
covariance under Gaussianity, so they bottom out at the Markov equivalence class, which on two
variables contains both directions. Covariance is direction-blind; whatever breaks the tie must be
something it does not see.

That impossibility already prunes most of what I could put in the function. The scaffold default
returns zeros — an absence, not a floor. PC or GES return a CPDAG whose undirected edges I would have
to orient by a coin, and the scoring counts *directed* edges with a true positive needing the arrow
right, so a coin-flip orientation on a 20-node graph throws away exactly the thing being measured; a
CPDAG method is structurally incapable of being the *directed* floor. A correlation-threshold
skeleton with an arbitrary orientation rule is the "correlation blur" I do not want. That leaves the
two genuine escapes from the equivalence class: non-Gaussianity and nonlinearity. I am reserving
nonlinearity for the rest of the ladder, so the floor takes the non-Gaussian route — and there the
choice narrows to ICA-LiNGAM versus DirectLiNGAM.

First, why non-Gaussianity is an escape at all. What does the covariance throw away? Everything above
second order. A Gaussian is fully described by its mean and covariance, so for Gaussian data there
*is* nothing above second order — exactly why the Gaussian-linear case is hopeless. But suppose the
disturbances `e_i` are *non-Gaussian*. Now the data carries higher-order structure, and the asymmetry
between cause and effect can leave a fingerprint there. Solve the linear SEM for `x`:
`x = (I - B)^{-1} e = A e`, with `A = (I-B)^{-1}` permutable to lower-triangular with unit diagonal
because `B` is permutable to strictly lower-triangular. That is a linear mixture of mutually
independent non-Gaussian sources — the ICA model exactly — and Comon's theorem says the mixing is
recoverable up to permutation, scaling, and sign when at most one source is Gaussian. So
non-Gaussianity turns the unidentifiable covariance problem into an identifiable one. This is the
*other* escape from the equivalence class, parallel to the nonlinear one the rest of my ladder will
use, and it is why a linear method can still produce arrows at all.

Concretely, in the two-node pair `x_1 → x_2` with `x_1` skewed (the exponential noise here has a
positive third cumulant), the correct-direction residual of `x_2` on `x_1` is exactly the clean
`e_2` and does not inherit `x_1`'s skew, while the wrong-direction residual of `x_1` on `x_2` mixes
`e_1` and `e_2`, and cumulants add under linear mixing so its skew and kurtosis are a blend
differing at third and fourth order from either clean source. Both residuals are uncorrelated with
their regressor and variance-matched, so second-order statistics see none of it — but the *shape*,
the departure from Gaussian, is asymmetric between the two directions, and that asymmetry is what
`diff_MI` reads off. The entropy criterion is the cheapest scalar summary of the higher-order
fingerprint the covariance throws away.

Now the decision between the two non-Gaussian estimators, made concretely. The first LiNGAM estimator
ran ICA, undid its permutation/scaling indeterminacies, and read off `B`. But ICA's contrast is
non-convex — local minima, an init guess that can give an outright wrong answer, a step size and a
stopping rule with no principled setting — and worse, its permutation scoring is scale-dependent. That
last flaw is not cosmetic. The assignment of estimated components to variables is chosen to make the
recovered mixing as close to a permuted-triangular matrix as possible, and "close" is measured on the
raw scale, so if I standardize the variables first — a routine, almost reflexive preprocessing step —
the relative magnitudes the assignment keys on change, and at finite sample size the recovered order
can *flip*. An estimator whose answer reverses when I rescale an input is keying on the wrong thing,
and I cannot trust a floor that is contingent on whether I remembered to standardize. I want to keep
the non-Gaussianity, which is what makes the problem identifiable, but drop the ICA machinery.
DirectLiNGAM is precisely that: it gets the causal *order* directly, with no iterative parameter
search, and — as I will make sure below — it standardizes internally so the order is scale-invariant
by construction, which is the exact defect that sank ICA-LiNGAM.

The structural fact that makes the direct approach possible: in a DAG with no latent confounders,
acyclicity forces at least one variable to have no parents — a source, an exogenous variable. For an
exogenous `x_j` the model says `x_j = e_j`: it simply *equals* its own independent non-Gaussian
disturbance. If I could find that variable, I would know it sits first in the order; peel its effect
off all the others, and the remaining system is again a LiNGAM with one fewer variable, so I recurse.
The whole problem reduces to one question: how do I detect, from data, which variable is exogenous?
Suppose I tentatively treat `x_j` as the source and regress every other `x_i` on it by least squares,
forming the residual `r_i^{(j)} = x_i - (cov(x_i,x_j)/var(x_j)) x_j`. The clean characterization —
provable in both directions — is: `x_j` is exogenous if and only if `x_j` is independent of *every*
residual `r_i^{(j)}`. The forward direction is direct from `x = Ae`: when `x_j = e_j`, the
least-squares coefficient is exactly `a_{ij}`, the residual is the bundle of *other* sources, and
that bundle is independent of `e_j`. The converse is where the non-Gaussianity earns its keep: if
`x_j` has a parent, then by Darmois–Skitovitch two linear combinations sharing a non-Gaussian source
with nonzero weight in both cannot be independent, so some residual stays dependent on `x_j`. With
*Gaussian* `e_j` the converse fails — independent combinations may share a Gaussian source — and the
test gives false positives. That is the precise sense in which this method needs non-Gaussian noise,
and it is a warning flag I should carry forward: one of my three scenarios uses Gaussian noise, and
this exogeneity test has no teeth there.

There is one landmine I must step around in code. Least-squares regression *guarantees* the residual
is uncorrelated with the regressor — by construction, for *every* candidate, source or not — so
uncorrelatedness is useless as the exogeneity test. If I scored
candidates by residual correlation I would score them all identically at zero and learn nothing. I
need a measure of genuine independence that sees the higher-order dependence uncorrelatedness misses.
The canonical such measure is mutual information, and DirectLiNGAM's `pwling` variant computes it
cheaply from *one-dimensional* differential entropies via a likelihood-ratio between the two pairwise
directions: the joint-entropy terms cancel under a unit-determinant change of variables, leaving a
difference of regressor-plus-residual entropies. Each entropy is the maximum-entropy approximation of
a standardized variable — the Gaussian entropy minus two non-Gaussianity penalties, a log-cosh
(heavy-tail) term and an odd `u·exp(-u²/2)` (skew) term, with fixed constants `k1 ≈ 79.047`,
`k2 ≈ 7.4129`, `gamma ≈ 0.37457`.

The criterion only makes sense if this approximation is *calibrated to the Gaussian*: `gamma`
(`0.37457`) is `E[log cosh Z]` for a standardized Gaussian, so the first penalty vanishes for Gaussian
input and is positive otherwise, the skew term vanishes by odd symmetry, and `_entropy` of a
standardized Gaussian collapses to its exact differential entropy `(1 + log 2π)/2 = 1.4189` nats.
For fixed unit variance the Gaussian is the entropy maximizer, so a non-Gaussian standardized
variable scores *below* `1.4189` — a standardized Laplace at `1.315`, a standardized exponential at
`1.085`, more sharply under for the heavier-skewed exponential. That is the operational content of
the source test: an exogenous variable *is* one of these raw non-Gaussian sources, its entropy
depressed below Gaussian, whereas a residual bundling several independent sources is pushed *toward*
Gaussian by the central-limit tendency and has higher entropy, and
`diff_MI = (H(x_j)+H(r_{i|j})) - (H(x_i)+H(r_{j|i}))` reads that asymmetry. So `_search_causal_order`
aggregates for each candidate `i` the squared *unfavorable-sign* differences
`M(i) = sum_j min(0, diff_MI(i,j))²` and picks the variable with the least accumulated
counter-evidence: a genuine source meets no partner more source-like than itself, so its negative
contributions clip to zero and `M(i) = 0`, while a non-source meets at least one partner that squares
a negative into `M(i) > 0`. The squaring makes one clear contradiction decisive. This also says
exactly how nonlinearity will corrupt the test: on curved mechanisms the `diff_MI` signs are no
longer reliably fixed by the true order, so sources pick up spurious negatives and non-sources
sometimes escape them, and the `argmin` drifts — the order degrades gracefully into noise rather than
failing catastrophically at one step.

I standardize every variable and residual before scoring, so the order is scale-invariant — the exact
defect that sank ICA-LiNGAM — and reuse `_BaseLiNGAM._estimate_adjacency_matrix` to turn the recovered
order `K` into connection strengths. The cost bounds what I can afford: `_search_causal_order` is
`O(d²n)` per call and is called `d` times as the order is built, so `O(d³n)` overall — about
`1.6·10⁷` entropy-term evaluations at `d=20`, `n=2000`, cheap, with nothing to tune. The residual
peeling after each pick (`x_i ← x_i - β x_m`) keeps the reduced system a LiNGAM one dimension smaller,
so the recursion is exact under the linear model. One deliberate simplification against the fullest
version: I estimate the full triangular adjacency from the order with no sparse adaptive-lasso pruning
step, so I should expect a denser graph and lower precision than a lasso-pruned variant would give.
The full module — build `U`, repeatedly find the exogenous index, peel it out, then estimate the
adjacency from `K` — is in the answer, and its output obeys `B[i,j] != 0` means `j -> i`.

Now what this floor must do on *this* data, which is the entire point of running it first. The data is
generated by *nonlinear*
mechanisms. DirectLiNGAM assumes *linear* ones. The mismatch bites in two compounding ways. First,
the residual `r_i^{(j)} = x_i - β x_j` is a *linear* residual; if the true mechanism is
`x_i = f(x_j) + e_i` with `f` curved, the linear fit leaves a structured, `x_j`-dependent remainder
*even in the correct direction*, so the clean forward direction of the exogeneity lemma — residual
independent of the source — no longer holds exactly, and the very signal the test reads is muddied.
Second, the entropy criterion reads the non-Gaussianity of those residuals, but a nonlinear function
of a non-Gaussian input produces residuals whose higher-order structure no longer matches the
additive-source picture the lemma assumed, so the calibration I just verified (Gaussian at the top,
sources depressed below) is applied to quantities that are neither clean sources nor clean bundles.
Both effects push the recovered order toward noise. So I expect DirectLiNGAM to land arrows that are
*frequently reversed or spurious* — its F1 should be low across the board, and because it estimates a
fairly full triangular graph from a possibly-wrong order (no lasso to cut it back), its SHD should be
large and its precision poor. The mechanical prediction I can be sharp about: on a 20-node graph the
triangular fill from a recovered order can place up to `d(d-1)/2 ≈ 190` nonzero entries before the
`|B|>0.01` threshold, so even after that cut I expect a *predicted* edge count well above the true
one, with precision dragged down accordingly — the failure mode is over-drawing, not under-drawing.

The scenarios should split predictably, and the splits follow from what I have already established.
On SF20-GP the noise is exponential — the most strongly non-Gaussian of the families, scoring furthest
below the Gaussian entropy in my check, which is exactly the regime where source detection has the
most signal — but the GP mechanisms are smooth and very nonlinear, which hurts the linear residual; I
expect a middling-to-low F1 and a large SHD on the 20-node scale-free graph. On ER20-Gauss the noise
is *Gaussian*, which is the worst case for this method on two counts at once: the converse of the
exogeneity lemma fails for Gaussian sources (false positives in source detection, because
Darmois–Skitovitch gives no leverage), and the mechanisms are nonlinear; I expect the lowest or
near-lowest F1 here, with a very large SHD. On ER12-LowSample the graph is smaller (12 nodes) but `n`
is only 150, so every entropy and covariance estimate is noisy and the sample averages I rely on are
computed from few points; I expect low F1 driven mostly by variance across seeds, though the smaller
graph caps the absolute SHD — with 12 nodes the triangular fill has at most `d(d-1)/2 = 66` positions
to be wrong about, against `190` on the 20-node graphs, so even a badly wrong order cannot run the SHD
into the same range as the ER20 case regardless of how poorly the method does. I will not pretend to know the exact ranking among these three before I
see it — the exponential-noise help on SF20-GP and the Gaussian-noise hurt on ER20-Gauss could land
in a different order than I guess — but the *shape* I am committing to is low-F1, low-precision,
high-SHD everywhere; the table will tell me which scenario is worst.

So the expectation I hold the next rung to: a linear floor should sit well below any genuinely
nonlinear method on F1, failing by wrong/dense direction rather than missing edges. If the first rung
that models nonlinearity does not clearly beat this on F1 *and* shrink SHD, the leverage is not the
nonlinearity and I have misdiagnosed the task.
