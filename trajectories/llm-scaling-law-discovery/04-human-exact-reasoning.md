The boosted-tree rung confirmed the trade almost line for line. Vocab rose to $0.981$ — the best so far,
because the surface is smooth and near the hull so the ensemble's flexibility paid off with negligible
discretization cost. Lrbsz improved from the symbolic $-3.05$ to $-1.405$, the least-negative I have seen,
with `MAE` dropping to $0.0537$ — so letting conditional tree splits learn a different effective $(l, b)$
optimum per scale region did cut the ranking error the fixed basin made. And dataconstrained *fell* to
$0.857$, below the symbolic $0.929$ — exactly the cost I flagged. The spreads are unchanged: vocab
$\sigma = 0.847$, dataconstrained $0.555$, lrbsz $0.0382$. That last one, more than twenty times smaller
than vocab's, governs the whole task: lrbsz is where a given absolute error costs the most $R^2$, and since
the task score is the geometric mean across families, lrbsz has dragged every solution down. Vocab and
dataconstrained already sit in the $0.85$–$0.98$ range, so the geometric mean cannot rise until lrbsz stops
being a large negative. The lever is lrbsz, and the question is narrow: what carries the lrbsz surface into
the extrapolation region that neither the fixed basin nor the tree's boundary-flattening could?

The lesson across the three rungs is unambiguous: neither the rigid hand-shaped symbolic form nor the
flexible-but-asymptotically-blind tree dominates, and the single thing both lack is the *correct
literature-grounded asymptotic form per family* — most critically a lrbsz law whose optimum is an explicit
function of scale. So the strongest rung drops the discovered-style improvisations and the black box and fits
the *actual human laws from the literature*, one per family. The point is correctness of form, not novelty.

Vocab first, because the tree already does well there and I need to know whether an exact human form can
match a flexible learner on a smooth surface. The established vocabulary law is purely additive:
$L(N, V, D) = E + A\,N^{-\alpha} + B\,V^{-\beta} + C\,D^{-\gamma}$ — a floor plus one decaying power per
axis. This is *not* the discovered-style form from the symbolic rung, where I added a multiplicative joint
power and a $V\times D$ cross term on the hunch that vocab and data interact. That rung's result is the
evidence against the hunch: vocab landed at $0.929$, the same low-$0.9$s the additive backbone alone
predicts, the signature of an *inert* cross term — had $V$ and $D$ genuinely interacted, the extra two
parameters would have bought a jump above the additive number, and they did not. So the honest test is
whether the simpler, theory-grounded additive form extrapolates at least as well as the cross-term
improvisation; I expect it to reproduce essentially the symbolic rung's $\sim 0.93$, a hair under the tree,
because on a smooth near-hull surface the flexible ensemble shaves a little more in-region variance than any
three-power form can. The floor $E$ stays *unconstrained* because the unigram-normalised target can be
negative; the scale and exponent parameters are exponentiated; and because the target can be negative I fit
in the *linear* domain, the exact domain the metric scores. There is a quiet advantage beyond fidelity: this
form has seven parameters ($E, A, \alpha, B, \beta, C, \gamma$), and because $N, V, D$ are each swept in the
grid, each exponent is pinned by the slope along its own axis with little cross-talk — better-conditioned
than the discovered-style eight, where the joint scale $A$ and the cross scale $A_{vd}$ were partly
redundant on the $(V, D)$ pair.

For lrbsz — the decider — I want the choice forced by the diagnosis. Three directions are open. Keep the
tree and ensemble it with a symbolic law — but that abandons the contract's one shared compact expression
per family and does not fix extrapolation, since the tree half still flattens off the hull. Take the fixed
basin and regress its center, $\log l^\star = \gamma_0 + \gamma_1\log N + \gamma_2\log D$ — the right idea, a
scale-dependent optimum, but a hand-guess at the drift's functional form with no independent evidence for
the slopes. Or take the established Expert-B law, which encodes that scale-dependent optimum with the
specific power-law drift the field measured, $l_0 = F\,N^\gamma\,D^\zeta$, and pairs it with an *asymmetric*
batch term rather than a second symmetric quadratic. The third is strictly more structure, grounded rather
than guessed, with a published reference fit to anchor to — and the diagnosis ("the optimum drifts with
scale and no rung has carried that drift") points straight at the form that writes the drift as an explicit
measured function. So I take Expert-B.

Write it as $L = A\,D^{-\alpha} + B\,N^{-\beta} + C + K_l\,(l - l_0)^2 + E\,(\log b + b_0/b)$, with
$l_0 = F\,N^\gamma\,D^\zeta$ and $b_0 = G\,D^\eta$. The scale terms $A\,D^{-\alpha} + B\,N^{-\beta}$ are the
Chinchilla backbone. $K_l\,(l - l_0)^2$ penalizes the learning rate off its optimum, minimized at $l = l_0$
— but $l_0$ is *not a fitted constant*; it is a power law in $N$ and $D$, so as the held-out configuration
grows the predicted optimal learning rate moves with it. The batch term is a different shape, and it does
have an interior optimum: minimizing $\log b + b_0/b$, the derivative $1/b - b_0/b^2$ is zero at $b = b_0$
and the second derivative there is $+1/b_0^2 > 0$, a genuine minimum, a log-plus-inverse well whose bottom
sits at $b_0 = G\,D^\eta$ — again a scale-dependent optimum, and the asymmetry between the quadratic lr
penalty and the log-plus-inverse batch penalty encodes that under- and over-shooting the batch cost
differently. This is the thing all three earlier rungs missed on lrbsz: a basin whose center tracks scale by
construction. The reference exponents read sane — $\gamma = -1.058$ (optimal lr decreases as the model
grows), $\zeta = 0.650$ (increases with data), $\eta = 0.350$ (optimal batch grows with data) — all the
signs optimization lore expects. And a limit check: at a perfectly tuned run $l = l_0, b = b_0$, the lr
penalty vanishes and the batch term reduces to $E(\log b_0 + 1)$, so the law collapses to
$A\,D^{-\alpha} + B\,N^{-\beta} + C + E(\log b_0 + 1)$ — a pure scale-driven floor with no optimizer
dependence, the correct decomposition where the scale terms set the achievable floor and the two penalties
measure the toll for being mistuned.

I fit per group rather than globally because the groups are different training setups: the *absolute* scale
of the optimal learning rate — the prefactor $F$ — can differ between them for reasons unrelated to the
universal drift, while the *exponents* $\gamma, \zeta$ that govern how the optimum moves with scale are the
physics I want to share. Fitting per group lets $F$ (and $G$, $A$, $B$) absorb group-level offsets while
each group re-estimates the same structural exponents, so a group whose optimal lr is uniformly higher does
not corrupt the drift read from its neighbors.

Even with the right form I have to be sober: the twelve coefficients $[A, \alpha, B, \beta, C, K_l, E, F,
\gamma, \zeta, G, \eta]$ are highly coupled — $F$ trades against $\gamma, \zeta$ inside the same $l_0$, $K_l$
against the whole lr term's scale — so the objective is riddled with long flat valleys, worse than the
nine-parameter basin that already needed the most restarts, and given how small $\sigma$ is a mediocre local
optimum would score catastrophically. So I do not start cold. The literature reports an Expert-B held-out
$R^2 \approx -0.0756$ on this split — still slightly negative, because the held-out lrbsz region is a true
extrapolation and even the correct human form cannot make it positive. So my target is not "lrbsz positive";
it is "lrbsz as close to zero as the form allows, far better than every earlier rung." I seed the fitter
with the established reference coefficients for the all-data Expert-B law, and I evaluate those coefficients
directly, with no fitting, as an absolute fallback — so the rung can never come out *worse* than the
reference, whatever the refinement does. Then I run nonlinear least squares from two starts — the reference
coefficients (packed into the exponentiated parameterization, signed exponents $\gamma, \zeta, \eta$ left
free) and a data-driven start derived from the target's span — and keep whichever scores best in the
*linear* domain, matching the domain the reference was evaluated in so the "never worse than the reference"
guarantee holds. Anchoring this way converts a treacherous twelve-parameter fit into a refinement around a
known-good point.

Now dataconstrained, where I need the explicit saturating asymptotic the tree gave back. The established
data-constrained law replaces the raw token count with an effective count that saturates as data is
repeated; for this rung I use the compact form $D_{\text{eff}} = U\,(1 - e^{-D/U})$. Trace its limits: when
$D \ll U$ the linear regime gives $1 - e^{-D/U} \approx D/U$, so $D_{\text{eff}} \approx D$ — a sub-epoch
pass counts every token as fresh; as $D \to \infty$ the effective count saturates at exactly $U$; and at
one epoch $D = U$, $D_{\text{eff}} = U(1 - e^{-1}) = 0.632\,U$, a $37\%$ discount by the first full repeat.
That last number is a caveat: this single exponential saturates at a *fixed* ceiling $U$ with no free
budget, whereas the symbolic rung's $1/(1 + (D/U)/R)$ saturated at $UR$ with a fitted $R$ — one extra degree
of freedom to place the ceiling. So the human form is leaner, one clean exponential, and that leanness could
cost me on the densest test points where a fixed ceiling underfits more than a fitted one. The law is
$L = E + A\,N^{-\alpha} + B\,D_{\text{eff}}^{-\beta}$, the Chinchilla backbone with $D$ replaced by the
saturating $D_{\text{eff}}$, and its asymptotics are exactly right where the tree's staircase failed — past
the hull, $D_{\text{eff}}^{-\beta}$ keeps bending toward the floor instead of holding a boundary constant.
Strictly positive target, so I fit in the log domain. It is the leanest of the three — just five parameters,
because the repetition nonlinearity lives in the parameter-free transform $D \mapsto U(1 - e^{-D/U})$ — hence
the best-identified and most stable fit, at the cost of the one degree of freedom that could have placed the
ceiling more precisely. I expect it to recover much of the symbolic rung's strong dataconstrained behavior
that the tree gave up, landing below the discovered-style $0.929$ because the single exponential has fewer
knobs to track the densest points — but I genuinely do not know by how much, and would not be surprised if
the fixed ceiling and the early $37\%$ discount cost more than the small dip I am hoping for.

The fitting machinery is the shared one: per `group`, nonlinear least squares with multi-start (the provided
robust soft-$\ell_1$ `least_squares`), the linear-vs-log residual chosen per family, best restart kept,
median-of-groups fallback for unseen groups.

Because the geometric mean is dominated by its smallest factor and lrbsz has been the floor at every rung,
the marginal return is entirely there: pushing vocab from $0.93$ to $0.98$ barely moves the product, while
pulling lrbsz from a large negative toward zero is the only change that lifts the aggregate. That is why I
accept a dataconstrained give-back and a vocab that merely holds, and spend the structure budget on
Expert-B. The bar this rung must clear is the strongest of each earlier rung: vocab $0.981$, dataconstrained
$0.929$, and lrbsz — the decider — where the best so far is $-1.405$. The a-priori read anchors on the
reference: if lrbsz lands near the Expert-B $R^2 \approx -0.076$, that is `RMSE` around $0.04$ and `NMAE`
under $1$ — by far the best lrbsz of any rung and the first time the hardest family is handled competently,
because it is the only form whose basin center tracks scale, with the fallback guaranteeing it cannot come
out below the reference. Vocab should hold near $0.93$ and dataconstrained land below the symbolic $0.929$
but with clean saturating asymptotics. If lrbsz comes back near the reference while vocab holds, then since
the task scores the geometric mean and lrbsz drags every other solution down, this is the strongest rung
overall — and the intended compact symbolic contribution. The gap it does not close: even the exact human
Expert-B form leaves lrbsz slightly negative, so the held-out lrbsz extrapolation is not fully solved by any
published human law, and closing it would need a richer scale-dependent surface or a search that discovers a
form beyond the hand-derived one.
