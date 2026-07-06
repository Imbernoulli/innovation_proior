The boosted-tree rung confirmed the trade I expected, almost line for line. Vocab rose to $0.981$ — the
best vocab number so far, because the surface there is smooth and near the hull, so the ensemble's
flexibility paid off and its discretization cost was negligible. Lrbsz improved from the symbolic $-3.05$ to
$-1.405$, the least-negative lrbsz $R^2$ I have seen, with `MAE` dropping further to $0.0537$ — so letting
conditional tree splits learn a different effective $(l, b)$ optimum per scale region did cut the ranking
error the fixed basin made. And dataconstrained *fell* to $0.857$, below the symbolic $0.929$ — exactly the
cost I flagged. Before I decide the next form, the $\sigma$ tool once more: from this table $\sigma =
\mathrm{RMSE}/\sqrt{1 - R^2}$ gives vocab $0.115/\sqrt{1 - 0.981} = 0.847$, lrbsz $0.0592/\sqrt{1 + 1.405} =
0.0382$, dataconstrained $0.210/\sqrt{1 - 0.857} = 0.555$ — the same three values I have now recovered from
all three rungs, so the split's per-family spreads are pinned and I can read every prediction below in their
units. The lrbsz $\sigma = 0.0382$ is the one that governs the whole task: it is more than twenty times
smaller than vocab's and fifteen times smaller than dataconstrained's, so lrbsz is the family where a given
absolute error costs the most $R^2$, and — because the task score is the geometric mean across families —
it is the family that has dragged every solution down. Vocab and dataconstrained are already in the
$0.85$–$0.98$ range; the geometric mean cannot rise until lrbsz stops being a large negative. So the lever
that matters is lrbsz, and the question for this rung is narrow: what carries the lrbsz surface into the
extrapolation region that neither the fixed basin nor the tree's boundary-flattening could?

The lesson across the three rungs is now unambiguous. Neither the rigid hand-shaped symbolic form nor the
flexible-but-asymptotically-blind tree dominates: the tree wins where flexibility matters (vocab, and the
lrbsz ranking), the symbolic law wins where the explicit asymptotic form matters (dataconstrained), and the
single thing both lack is the *correct literature-grounded asymptotic form per family* — most critically, a
lrbsz law whose optimum is an explicit function of scale, the very thing the symbolic rung's fixed basin and
the tree's boundary-flattening both fail to carry. So the strongest rung is the one I have been circling:
drop the discovered-style improvisations and the black box, and fit the *actual human laws from the
literature*, one per family, each with the asymptotic structure the field established. The point is not
novelty; it is correctness of form. Let me derive each family's law from its established source, in
discovery order, and pin down where each fixes a measured failure of the rungs before it.

Vocab first, because the tree already does well there ($0.981$) and I need to know whether an exact human
form can match or beat a flexible learner on a smooth surface. The established vocabulary law is purely
additive: $L(N, V, D) = E + A\,N^{-\alpha} + B\,V^{-\beta} + C\,D^{-\gamma}$ on the unigram-normalised loss —
a floor plus one decaying power term per axis. This is *not* the discovered-style form from the symbolic
rung: there I added a multiplicative joint power and a $V\times D$ cross term on a hunch that vocab and data
interact. That rung's result is the evidence I now read against the hunch — vocab landed at $0.929$, the
same low-$0.9$s the additive backbone alone predicts, which is the signature of an *inert* cross term: had
$V$ and $D$ genuinely interacted, the extra two parameters would have bought a jump above the additive
number, and they did not. The human law says the axes contribute additively and independently, and my own
symbolic rung's number is consistent with that. So the honest test is whether the simpler, theory-grounded
additive form extrapolates at least as well as the cross-term improvisation. I expect it to land at
essentially the symbolic rung's $0.929$ — reconstructing, $R^2 = 1 - (\mathrm{RMSE}/0.847)^2$, so matching
$0.929$ needs `RMSE` near $0.226$, which the three-power form should hit — a hair under the tree's $0.981$,
because on a smooth near-hull surface the flexible ensemble can shave a little more in-region variance than
any three-power form can. The floor $E$ stays *unconstrained* — not exponentiated — because the
unigram-normalised target can be negative and the additive constant must be free to absorb the sign; the
scale and exponent parameters are exponentiated to keep the fit well-conditioned, and because the target can
be negative I fit the residuals in the *linear* domain, not log — which is also the exact domain the metric
scores. There is a quiet advantage in this simpler form beyond fidelity to the established law: it has seven
parameters ($E, A, \alpha, B, \beta, C, \gamma$), one per axis plus the shared floor and scales, and because
$N$, $V$, and $D$ are each swept in the grid, each exponent is pinned by the slope along its own axis with
little cross-talk. The discovered-style form I used before carried eight, with the joint scale $A$ and the
cross scale $A_{vd}$ partly redundant on the $(V, D)$ pair — a coupling that makes the fit slower to
identify. So the additive human form is not just theory-cleaner, it is better-conditioned, which is part of
why I expect it to reproduce the symbolic rung's vocab number rather than fall short of it despite dropping
the cross term.

Before I write the lrbsz form down I should be explicit about the candidates, because this is the decider
family and I want the choice forced by the diagnosis rather than picked by taste. Three directions are open.
I could keep the tree and ensemble it with a symbolic law — but that abandons the task's contract, which
asks for one shared compact symbolic expression per family, and it does not fix the extrapolation, since the
tree half still flattens off the hull. I could take the symbolic rung's fixed basin and simply regress its
center, writing $\log l^\star = \gamma_0 + \gamma_1\log N + \gamma_2\log D$ — which is the right *idea*, a
scale-dependent optimum, but it is my hand-guess at the drift's functional form bolted onto a symmetric
quadratic, and I have no independent evidence for the guessed slopes. Or I take the established Expert-B law,
which encodes exactly that scale-dependent optimum — $l_0 = F\,N^\gamma\,D^\zeta$ — but with the specific
power-law drift the field measured, and pairs it with an *asymmetric* batch term rather than a second
symmetric quadratic, because the batch dependence is not symmetric. The third is strictly more structure than
the second, grounded rather than guessed, and it is the one with a published reference fit I can anchor to.
The diagnosis — "the optimum drifts with scale and no rung has carried that drift" — points straight at the
form that writes the drift as an explicit, measured function, so I take the Expert-B law.

Now lrbsz, the family that has defeated every rung, and the one where the human form carries the decisive
structure. The established Expert-B law is hierarchical and additive in its scale terms but — crucially — it
makes the optimizer-setting optima *explicit functions of scale*. Write it as $L = A\,D^{-\alpha} +
B\,N^{-\beta} + C + K_l\,(l - l_0)^2 + E\,(\log b + b_0/b)$, where $l_0 = F\,N^\gamma\,D^\zeta$ and $b_0 =
G\,D^\eta$. Read what each piece does and verify it is the right shape. The data and parameter terms
$A\,D^{-\alpha} + B\,N^{-\beta}$ are the Chinchilla scale backbone. $K_l\,(l - l_0)^2$ is a quadratic
penalty for the learning rate being off its optimum, and minimizing it over $l$ gives $l = l_0$ — but $l_0 =
F\,N^\gamma\,D^\zeta$ is *not a single fitted constant*; it is a power law in the model size $N$ and the data
$D$, so as the held-out configuration grows the predicted optimal learning rate *moves with it*. The
batch-size term is a different shape, and I should confirm it actually has an interior optimum: minimizing
$\log b + b_0/b$ over $b$, the derivative is $1/b - b_0/b^2$, zero at $b = b_0$, and the second derivative
$-1/b^2 + 2b_0/b^3$ evaluated at $b_0$ is $+1/b_0^2 > 0$, a genuine minimum. So the batch term is a
logarithmic-plus-inverse well whose bottom sits at $b = b_0 = G\,D^\eta$, again a scale-dependent optimum,
and the asymmetry between the quadratic lr penalty and this log-plus-inverse batch penalty encodes that
under- and over-shooting the batch size cost differently. This is structurally the thing all three earlier
rungs were missing on lrbsz: a basin whose center tracks scale by construction. I can even read the physics
off the reference exponents to check they are sane — $\gamma = -1.058$ means the optimal learning rate
*decreases* as the model grows (bigger models want smaller steps), $\zeta = 0.650$ means it *increases* with
data, and $\eta = 0.350$ means the optimal batch *grows* with data. All three signs are what optimization
lore says they should be, which is a reassuring sign the form is not just flexible but correct.

Two more checks convince me the form is the right one to fit per group. First, a limit: at a perfectly tuned
run, $l = l_0$ and $b = b_0$, the lr penalty $K_l(l - l_0)^2$ is zero and the batch term reduces to
$E(\log b_0 + 1)$, so the whole law collapses to $A\,D^{-\alpha} + B\,N^{-\beta} + C + E(\log b_0 + 1)$ — a
pure scale-driven floor with no optimizer dependence left, exactly the loss a well-tuned model of that size
and data budget should pay. That is the correct decomposition: the scale terms set the achievable floor, and
the two penalties measure the toll for being mistuned, vanishing when tuning is optimal. Second, why fit the
coefficients per group rather than globally: the groups here are different training setups, and the *absolute*
scale of the optimal learning rate — the prefactor $F$ in $l_0 = F\,N^\gamma\,D^\zeta$ — can differ between
them for reasons that have nothing to do with the universal drift, while the *exponents* $\gamma, \zeta$ that
govern how the optimum moves with scale are the physics I actually want to share. Fitting per group lets $F$
(and $G$, and the scale prefactors $A, B$) absorb group-level offsets while each group re-estimates the same
structural exponents, so a group whose optimal lr is uniformly higher does not corrupt the drift I read from
its neighbors. The shared expression is the contract; the per-group coefficients are what make it fit real,
heterogeneous runs.

Even with the right form, I have to be sober about the fit, because lrbsz is genuinely hard and the twelve
coefficients $[A, \alpha, B, \beta, C, K_l, E, F, \gamma, \zeta, G, \eta]$ are highly coupled — a change in
$F$ can be absorbed by $\gamma$ and $\zeta$ inside the same $l_0$, and $K_l$ trades against the whole lr
term's scale, so the objective is riddled with long flat valleys, far worse than the nine-parameter basin of
the symbolic rung that already needed the most restarts. A cold multi-start into twelve coupled parameters
would land wherever the first descent happens to stop, and given how small $\sigma$ is, a mediocre local
optimum would score catastrophically. The literature reports an Expert-B held-out $R^2$ of about $-0.0756$
on this split — *still slightly negative*, because the held-out lrbsz region is a true extrapolation and even
the correct human form cannot make it positive. So my target is not "lrbsz positive"; it is "lrbsz as close
to zero as the literature form allows, far better than every earlier rung." To get there robustly I do not
start the fitter cold. I seed it with the *established reference coefficients* for the all-data Expert-B law,
which already achieve $R^2 \approx -0.0756$, and I evaluate those coefficients directly, with no fitting, as
an absolute fallback — so the rung can never come out *worse* than the established reference, whatever the
refinement does. Then I run nonlinear least squares from two starts — the reference coefficients (packed
into the fitter's exponentiated parameterization, with the positive quantities log-transformed and the
signed exponents $\gamma, \zeta, \eta$ left free) and a data-driven start derived from the target's span —
and keep whichever scores best in the *linear* domain. I score by raw mean-squared error rather than log
here because this target is positive but delicate and the reference was evaluated that way, so matching the
scoring domain to the reference is what lets "never worse than $-0.0756$" actually hold. Anchoring on the
reference is the load-bearing trick: it converts a treacherous twelve-parameter fit into a refinement around
a known-good point, so the rung lands at the literature number instead of wherever a cold fit falls. The
two-start design is deliberate and minimal: near a known-good point I want a *refinement*, not a broad
search that could wander off into a flat valley, so I give the fitter the reference start and a single
data-driven start as insurance against this particular split having shifted the optimum away from the
reference's all-data fit — and above both sits the un-fitted reference evaluation itself, so the worst case
is bounded at $-0.0756$ by construction. That bound is worth stating plainly: the fallback is not a
convenience, it is what guarantees the decider family cannot regress below the established number no matter
how the refinement behaves on this split.

Now dataconstrained, where I need the explicit saturating asymptotic the tree gave back. The established
data-constrained law replaces the raw token count with an *effective* count that saturates as data is
repeated. The full published treatment models the diminishing value of both repeated tokens and excess
parameters with two geometric-decay constants — but for this rung I use the *compact* effective-token form
the family actually needs, $D_{\text{eff}} = U\,(1 - e^{-D/U})$, and I want to trace its limits before I
trust it. When $D \ll U$ the exponential's linear regime gives $1 - e^{-D/U} \approx D/U$, so $D_{\text{eff}}
\approx D$ — a single sub-epoch pass counts every token as fresh, correct. As $D \to \infty$ the effective
count saturates at exactly $U$: no matter how many times I loop, a fixed pool yields effective signal bounded
by its size. And at exactly one epoch $D = U$, $D_{\text{eff}} = U(1 - e^{-1}) = 0.632\,U$ — the form already
applies a $37\%$ discount by the first full repeat. That last number is a caveat I should hold onto: this
single-exponential saturates at a *fixed* ceiling $U$ with no free budget, whereas the symbolic rung's
$1/(1 + (D/U)/R)$ factor saturated at $U R$ with a fitted $R$, one extra degree of freedom to place the
ceiling. So the human form is deliberately leaner — one clean exponential, no repeat-budget parameter — and
that leanness could cost me on the densest test points, where a fixed ceiling underfits more than a fitted
one. The law is $L = E + A\,N^{-\alpha} + B\,D_{\text{eff}}^{-\beta}$, the Chinchilla backbone with $D$
replaced by the saturating $D_{\text{eff}}$, and its asymptotics are exactly right where the tree's staircase
failed — past the hull, $D_{\text{eff}}^{-\beta}$ keeps bending toward the floor as the effective tokens
saturate, instead of the tree holding a boundary constant. This target is strictly positive so I fit it in
the *log* domain, where — by the same first-order argument as the earlier rungs — minimizing the log
residual on a tight positive cluster is a well-conditioned proxy for the linear error the metric scores. And
this form is the leanest of the three: just five parameters ($E, A, \alpha, B, \beta$), because the entire
repetition nonlinearity lives in the *parameter-free* transform $D \mapsto U(1 - e^{-D/U})$ — there is no
repeat-budget constant to fit, unlike the symbolic rung's $R$. So it is the best-identified law of the
three, which is the flip side of its leanness: fewer knobs to pin means a more stable fit, at the cost of
the one extra degree of freedom that could have placed the saturation ceiling more precisely. I expect this to recover much of the symbolic rung's strong dataconstrained behavior that
the tree gave up, landing below the discovered-style $0.929$ because the simplified single exponential has
fewer degrees of freedom to track the densest points — but I genuinely do not know by how much, and I would
not be surprised if the fixed ceiling and the early $37\%$ discount cost more than the small dip I am hoping
for; the $D_{\text{eff}}$ residuals on the densest test points will tell me, and I am accepting some
in-region fit for a cleaner, more clearly-extrapolating form.

The fitting machinery is shared and is the one piece the loop provides: per `group`, fit the family's form by
nonlinear least squares with multi-start initializations (the provided robust soft-$\ell_1$
`least_squares`), the linear-vs-log residual chosen per family as above, keep the best restart, and store a
median-of-groups fallback for unseen groups — one shared expression per family, coefficients per group, the
contract the task asks for. The full scaffold module is in the answer.

It is worth being concrete about why lrbsz is the lever and not just "a weak family," because the aggregation
makes it so. The task score is the geometric mean across the three families, and a geometric mean is
dominated by its smallest factor: multiplying the smallest term by two moves the product far more than the
same change to the largest. Across the earlier rungs the ordering of family scores has been stable — vocab
strong, dataconstrained middling-to-strong, lrbsz the floor — so the geometric mean has been pinned near its
lrbsz factor the whole way. That means the marginal return on effort is entirely on lrbsz: pushing vocab from
$0.93$ to $0.98$ barely moves the product, while pulling lrbsz from a large negative up toward zero is the
only change that can lift the aggregate. This is why I am willing to *accept* a dataconstrained give-back and
a vocab that merely holds, and spend the structure budget on the Expert-B law: the geometric mean rewards
raising the minimum, and lrbsz is the minimum by a wide margin.

So this rung's whole bet is that the *correct literature form per family* dominates both the rigid
improvisation and the flexible black box, family by family. The bar it must clear is the strongest of each
earlier rung: vocab $0.981$ (the tree), dataconstrained $0.929$ (the symbolic form), and lrbsz — the decider
— where the best so far is the tree's $R^2 = -1.405$ with `MAE` $0.0537$. Turning the reference and the
locked $\sigma$ into falsifiable numbers: if the Expert-B law lands at its reported $R^2 \approx -0.0756$,
then `RMSE` $= 0.0382\sqrt{1.0756} = 0.0396$, and at the $\sim 0.85$ ratio of `MAE` to `RMSE` these tight
lrbsz targets have shown across the earlier rungs, `MAE` $\approx 0.0337$ with `NMAE` $= 0.0337/0.0382 =
0.88$. So I predict lrbsz roughly $-0.05$ in $R^2$, `MAE` below $0.034$, and `NMAE` below $0.9$ — by far the
best lrbsz of any rung, and the first time the hardest family is handled competently, because it is the only
form whose basin center tracks scale. Vocab should hold near $0.929$ (the additive human form, a touch under
the tree because it cannot shave the last in-region variance), and dataconstrained should land in the
mid-range, below the symbolic $0.929$ but with clean saturating asymptotics. If lrbsz comes back near the
reference while vocab holds, then since the task scores the geometric mean across families and lrbsz is the
family that drags every other solution down, this rung is the strongest overall — it handles the hard family
competently *and* it is the intended symbolic contribution, a compact law per benchmark rather than a black
box. The remaining gap it does not close — and the honest open question for anything stronger — is that even
the exact human Expert-B form leaves lrbsz slightly negative, so the held-out lrbsz extrapolation is not
fully solved by any published human law; closing it would require either a richer scale-dependent surface or
a search that discovers a form beyond the hand-derived one.
