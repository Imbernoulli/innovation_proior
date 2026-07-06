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
covariance, so I should ask what it can give me — and the answer is the reason the whole field is
hard. Take two correlated variables and make the two-node calculation completely explicit, because
the impossibility has to be *shown*, not asserted. Suppose the true joint covariance is `Σ` with
`var(x_1)=σ_1²`, `var(x_2)=σ_2²`, and `cov(x_1,x_2)=ρσ_1σ_2`. Model one says `x_1` causes `x_2`,
`x_2 = b x_1 + e_2`: least squares forces `b = cov/var(x_1) = ρσ_2/σ_1` and then
`var(e_2) = σ_2² - b·cov = σ_2²(1-ρ²)`. Model two reverses it, `x_1 = b' x_2 + e_1`, giving
`b' = ρσ_1/σ_2` and `var(e_1) = σ_1²(1-ρ²)`. Put numbers on it — `σ_1²=1`, `σ_2²=2.25`, `ρ=0.6`,
so `cov=0.9`. Forward: `b=0.9`, `var(e_2)=1.44`, and it rebuilds `var(x_2)=0.9²·1 + 1.44 = 2.25` and
`cov = b·var(x_1) = 0.9`. Backward: `b'=0.4`, `var(e_1)=0.64`, rebuilding `var(x_1)=0.4²·2.25+0.64=1`
and `cov = b'·var(x_2) = 0.9`. Both models reproduce the *identical* `Σ`. The covariance literally
cannot tell these apart, and this is not a defect of one estimator: PC tests conditional
independences, GES scores DAGs, but under Gaussianity both are functions of the covariance too, so
they bottom out at the Markov equivalence class — and on two variables that class contains both
directions. Covariance is direction-blind. If I only ever look at second-order statistics I am stuck
at the equivalence class, full stop. Whatever breaks the tie must be something the covariance does
not see.

That impossibility already prunes most of what I could put in the function, so let me actually walk
the shortlist rather than assume the answer. The scaffold default returns zeros — it measures
nothing, so it is not even a floor, just an absence. PC or GES would run, but by the two-node
argument above they return a CPDAG whose undirected edges I would have to orient by a coin, and the
harness scores *directed* edges with a true positive needing the arrow right, so a coin-flip
orientation on a 20-node graph throws away exactly the thing being measured; a CPDAG method is
structurally incapable of being the *directed* floor. A correlation-threshold skeleton with some
arbitrary orientation rule is the "correlation blur" I explicitly do not want — it has no principled
direction at all. That leaves the two genuine escapes from the equivalence class: non-Gaussianity and
nonlinearity. I am reserving nonlinearity for the rest of the ladder, so the floor should take the
non-Gaussian route — and there the choice narrows to ICA-LiNGAM versus DirectLiNGAM. So the shortlist
collapses to one real decision, and I have to make it on the merits.

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

It helps to see concretely where in the data the fingerprint lives, because "higher-order structure"
is otherwise a slogan. Go back to the two-node pair `x_1 → x_2`, `x_2 = b x_1 + e_2`, and suppose the
source `e_1` (so `x_1`) is skewed — say the exponential noise this benchmark uses, with a positive
third cumulant. In the *correct* direction, the residual of `x_2` regressed on `x_1` is exactly `e_2`,
whose distribution is the clean noise: its shape does not inherit `x_1`'s skew. In the *wrong*
direction, the residual of `x_1` regressed on `x_2` is a linear combination of `e_1` and `e_2` — it
mixes the two sources, and by the way cumulants add under linear mixing its skew and kurtosis are a
blend that differs, at third and fourth order, from either clean source. Second-order statistics see
none of this: both residuals are constructed to be uncorrelated with their regressor and can be
matched in variance. But the *shape* — the departure from Gaussian that the log-cosh and skew
penalties measure — is asymmetric between the two directions, and that asymmetry is precisely the
quantity `diff_MI` reads off. So the entropy criterion is not an arbitrary contrast; it is the
cheapest scalar summary of the higher-order fingerprint that the covariance throws away.

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

There is one landmine I must step around in code, and the task's edit handles it correctly. Least-
squares regression *guarantees* the residual is uncorrelated with the regressor — by construction, for
*every* candidate, source or not — so uncorrelatedness is useless as the exogeneity test. If I scored
candidates by residual correlation I would score them all identically at zero and learn nothing. I
need a measure of genuine independence that sees the higher-order dependence uncorrelatedness misses.
The canonical such measure is mutual information, and DirectLiNGAM's `pwling` variant computes it
cheaply from *one-dimensional* differential entropies via a likelihood-ratio between the two pairwise
directions: the joint-entropy terms cancel under a unit-determinant change of variables, leaving a
difference of regressor-plus-residual entropies. Each entropy is the maximum-entropy approximation of
a standardized variable — the Gaussian entropy minus two non-Gaussianity penalties, a log-cosh
(heavy-tail) term and an odd `u·exp(-u²/2)` (skew) term, with fixed constants `k1 ≈ 79.047`,
`k2 ≈ 7.4129`, `gamma ≈ 0.37457`.

I should not take those constants on faith; the whole criterion only makes sense if the approximation
is *calibrated to the Gaussian*, so let me check that it is. The constant `gamma` sits inside the
first penalty as `(mean(log cosh u) - gamma)²`, which means the penalty vanishes exactly when
`mean(log cosh u) = gamma`. If `gamma` is the value of `E[log cosh U]` for a standardized Gaussian,
then the first penalty is zero for Gaussian input and positive otherwise — a non-Gaussianity meter.
I evaluate `E[log cosh Z]` on a large standard-normal sample and get `0.3744`, which matches the
tabulated `0.37457` to the third digit; the skew term `E[Z·exp(-Z²/2)]` I get as `≈ -0.0002`, i.e.
zero by the odd symmetry of the integrand against the even Gaussian density. So both penalties vanish
for a Gaussian, and `_entropy` of a standardized Gaussian collapses to `(1 + log 2π)/2 = 1.4189` — the
exact differential entropy of `N(0,1)` in nats. That is the calibration I wanted: for fixed unit
variance the Gaussian is the entropy maximizer, so a *non*-Gaussian standardized variable must score
*below* `1.4189`, and the two penalties are precisely how much below. I confirm the direction on the
noise families this benchmark actually uses: a standardized Laplace scores `1.315` and a standardized
exponential `1.085`, both under the Gaussian `1.419`, and more sharply under for the heavier-skewed
exponential. This tells me something operational about the source test. An exogenous variable *is* one
of these raw non-Gaussian sources, so its entropy is depressed below Gaussian, whereas a residual that
bundles several independent sources is pushed *toward* Gaussian by the central-limit tendency and so
has higher entropy — and the pairwise difference `diff_MI = (H(x_j)+H(r_{i|j})) - (H(x_i)+H(r_{j|i}))`
is exactly reading that asymmetry. This is the code in the editable function: `_entropy`,
`_diff_mutual_info`, and `_search_causal_order` aggregate, for each candidate `i`, the squared
*unfavorable-sign* pairwise differences `M(i) = sum_j min(0, diff_MI(i,j))²`, and pick the variable
with the least accumulated counter-evidence — the one most consistent with being the source.

I want to be sure I read that aggregation correctly, because the sign convention is the whole test and
it is easy to get backwards, so let me trace it on the two-variable case. Fix a pair `(i,j)` and let
`diff_MI(i,j)` be positive when the "`i` upstream of `j`" reading is the more independent one — that
is, when regressing `i`'s effect out leaves a residual more independent of `i` than the reverse. If
`i` is the *true* source of the pair, then for that pair `diff_MI(i,j) > 0`; the unfavorable branch
`min(0, diff_MI(i,j))` clips to zero, contributing nothing to `M(i)`. Do this for every partner `j`
and a genuine source accumulates `M(i) = 0`: no partner produces evidence against it. A *non*-source
`i`, by contrast, will meet at least one partner `j` that is more source-like, for which
`diff_MI(i,j) < 0`, and that negative value survives the clip and is squared into `M(i) > 0`. So the
criterion selects `argmin_i M(i)` — the variable against which the pairwise comparisons raise the
least objection — and the squaring makes a single strong contradiction count more than several weak
ones, which is what I want from a source test: one partner that clearly should come first is decisive.
This also tells me exactly how the nonlinearity will corrupt the test. On curved mechanisms the
`diff_MI` signs are no longer reliably determined by the true order, so both true sources pick up
spurious negative contributions and non-sources sometimes escape them, and the `argmin` drifts — the
order degrades gracefully into noise rather than failing catastrophically at one step.

Note what the task's fill leans on and what it omits. It standardizes every variable and residual
before scoring (so the order is scale-invariant — the exact defect that sank ICA-LiNGAM), and it
reuses the library's `_BaseLiNGAM._estimate_adjacency_matrix` to turn the recovered order `K` into
the connection strengths. Let me account for the cost while I am at it, because it bounds what I can
afford at the floor: `_search_causal_order` scores each of up to `d` candidates against each of the
other `d`, and each pairwise score is two regressions and four entropies over `n` samples, so one
call is `O(d²n)`; it is called `d` times as the order is built, giving `O(d³n)` overall. At the
benchmark's `d=20`, `n=2000` that is on the order of `1.6·10⁷` entropy-term evaluations — cheap, no
tuning, no iteration count to guess. The residual peeling that follows each pick (`x_i ← x_i - β x_m`
for every remaining `i`) is what keeps the reduced system a LiNGAM one dimension smaller, so the
recursion is exact under the linear model. This last detail is the one place the task's edit diverges
from the most polished version of the method: the standalone derivation I keep as reference prunes
edges with a sparse adaptive lasso (BIC-selected), whereas the harness fill estimates the full
triangular adjacency from the order with no extra sparsity step — so I should expect a denser graph
and lower precision than the lasso-pruned variant would give.

So the floor fill is settled, and it is a faithful transcription of causal-learn's DirectLiNGAM core
into `run_causal_discovery`: build `U = arange(n)` (variables not yet placed), repeatedly call
`_search_causal_order` on the working data to find the exogenous index `m`, append it to the order
`K`, regress `m` out of every other working variable, and after the order is complete hand `K` to a
local `_BaseLiNGAM` to estimate the adjacency. The output already obeys the harness convention
`B[i,j] != 0` means `j -> i`. The full scaffold module is in the answer.

Now the part that actually matters for the ladder: reasoning about what this floor must do on *this*
data, because that is the entire point of running it first. The data is generated by *nonlinear*
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

The falsifiable expectation I will hold the next rung to: DirectLiNGAM is a *linear* floor, so its F1
across all three scenarios should sit well below what any genuinely nonlinear method can reach, and
its failure mode is *wrong/dense direction* (low precision, high SHD), not missing edges. If the next
rung — the first one that actually models nonlinearity — does not clearly beat this on F1 *and* shrink
SHD, then the nonlinearity is not where the leverage is and I have misdiagnosed the task. I am
starting here precisely so that gap is measured, not assumed.
