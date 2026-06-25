The adaptive-KL rung learned everywhere, exactly as I expected, but it left precisely the weakness I was
worried about, and the numbers say where. On HalfCheetah it landed a mean of 1676.6 but with a huge
spread — 1194.9 / 2695.0 / 1139.9 across the three seeds, a 2.4× gap between the best and worst seed —
so the dense-reward environment is *learnable* under the penalty servo but wildly inconsistent, which is
the seed-to-seed instability I predicted from the reactive, after-the-fact $\beta$ adjustment: one seed
got lucky with its KL overshoots and ran to 2695, two paid for the overshoots and stalled near 1150.
Swimmer came in at 101.4 (81.6 / 113.0 / 109.5) — again the seed-42 value sits far below the other two,
the same one-seed-drags-it-down signature. InvertedDoublePendulum was the steadiest at 6877.1 (6764.7 /
7497.4 / 6369.2). So the diagnosis is sharp: the soft KL penalty does not *forbid* a bad move, it
*prices* one, and on a noisy minibatch the optimizer will sometimes pay the price and take a step it
should not, which is why the seed variance is the dominant failure rather than a flat ceiling. Because
the task scores by geometric mean across the three environments, that HalfCheetah spread and the
weak-seed Swimmer number are exactly what will hold this rung down. The question for the next rung is
whether I can get a policy update that does not depend on the importance ratio at all — because the
ratio is the thing that fans out and the KL penalty is only a soft attempt to rein it back in.

Let me attack from a completely different direction, because the whole penalty-vs-clip family is fighting
the same enemy — the importance ratio $r_t$ — and maybe the move is to stop using a ratio surrogate
entirely. Step back to what I actually want from a policy update: make the actions that turned out
better (high advantage) more likely, and the ones that turned out worse less likely, without moving so
far that the data I am training on stops describing the policy. The penalty form does this through
$r_t\hat A_t$ minus a KL term. But there is an older idea that frames policy improvement as plain
*supervised regression*: regress the policy directly onto the actions the agent took, with each action
weighted by how good it was. No ratio, no clipping, no KL — just a weighted maximum-likelihood fit. The
weight is an exponentiated advantage, $w_t = \exp(\hat A_t/\beta)$, and the policy loss is the negative
weighted log-likelihood $-\hat{\mathbb E}_t[\,w_t\,\log\pi_\theta(a_t|s_t)\,]$. This is the
advantage-weighted-regression lineage, and it is appealing here precisely because the failure I just
diagnosed was *about the ratio*: if there is no ratio in the loss, there is no ratio to fan out, and the
reactive KL servo I was unhappy with disappears entirely.

Let me convince myself the exponential weight is the right shape and not just a convenient knob, because
this is the crux of why AWR is a principled improvement step and not a lateral move. Frame the update as
a constrained problem: find the new policy that maximizes expected advantage subject to staying KL-close
to the data-generating policy — the *same* trust-region intuition as before, but solved in closed form
instead of penalized. The solution to "maximize $\mathbb E_\pi[A]$ subject to
$\mathrm{KL}[\pi\,\|\,\pi_{old}]\le\epsilon$" is the exponentially-tilted policy
$\pi^*(a|s)\propto\pi_{old}(a|s)\exp(A(s,a)/\beta)$, where $\beta$ is the Lagrange multiplier for the KL
constraint. I cannot represent that tilted distribution directly, but I can *project* it back onto my
Gaussian policy class by minimizing the KL from $\pi^*$ to $\pi_\theta$ — and that projection is exactly
weighted maximum likelihood, $\min_\theta -\mathbb E_{s,a\sim\pi_{old}}[\exp(A/\beta)\log\pi_\theta(a|s)]$.
So the exponential advantage weight is not a heuristic; it is the closed-form trust-region solution
expressed as a regression target. This is the trust region I wanted at rung 1, but achieved by
*construction* — the weights bake the "stay close" in — rather than by a soft penalty the optimizer can
trade away. That is the conceptual reason to expect it to be steadier on the dense-reward environment
where the penalty form swung between 1150 and 2695.

Now I have to be very careful, because the same-named idea has a canonical realization that is *not* what
this task's harness implements, and if I import that story I will write the wrong method. The original
advantage-weighted-regression algorithm is *off-policy*: it keeps a large replay buffer, recomputes
TD($\lambda$) returns over stored paths to get advantages, fits a separate critic and a separate actor
with their own momentum optimizers and step counts, normalizes the advantage before exponentiating, uses
a temperature near $1.0$, clips the weights at $20$, and adds an action-bound penalty. The whole point
there is *reusing old data* across many iterations, which is why the regression framing matters — a
ratio surrogate degrades badly on stale data, but a weighted regression onto whatever actions are in the
buffer does not. None of that off-policy machinery exists in this scaffold. The loop here is strictly
on-policy: it collects 2048 fresh transitions, computes *GAE* advantages (not buffer TD($\lambda$)) in
the frozen reverse scan, hands me a *single* shared 2×64 actor-critic, and asks me to fill one
`compute_losses` that Adam steps on for ten epochs over that one fresh batch. There is no replay buffer
to point AWR's regression at, no separate solvers, no second network. So the faithful realization here is
*on-policy AWR*: the advantage-weighted regression *objective* dropped into the PPO loop's plumbing,
using the loop's GAE advantages and its single optimizer. I am keeping the idea — supervised weighted
regression instead of a ratio surrogate — and discarding the off-policy apparatus the idea was born with,
because the harness does not expose it.

That harness shape forces three concrete choices, and each one is a real departure from the canonical
version that I want stated plainly rather than glossed. First, the temperature. The canonical default is
$\beta\approx1.0$, but that is calibrated for *raw, separately-normalized* advantages over a replay
buffer. The loop here has already normalized `mb_advantages` to roughly unit scale per minibatch
(`norm_adv=True`), so unit-scale advantages divided by $\beta=1.0$ give weights $\exp(\pm1)$ that barely
separate good actions from bad — the regression would be nearly uniform and learn almost nothing. To get
real selectivity from unit-scale normalized advantages I need a *small* temperature; I use
$\beta=0.05$, which turns a $+1$-sigma advantage into a weight $\approx e^{20}$ before clipping and a
$-1$-sigma advantage into $\approx e^{-20}\approx0$, i.e. it sharply concentrates the regression on the
better-than-average actions. This is the single most important deviation from the canonical recipe and it
is *because* the loop pre-normalizes the advantages — the temperature has to be read against the scale of
the input it actually receives, not the scale the original method assumed.

Second, the weight clip and stabilization. With $\beta=0.05$ the exponential weights have an enormous
dynamic range, and a single outlier advantage would produce a weight that dominates the entire minibatch
gradient — the AWR analogue of the ratio blowing up. So I clamp the weights at `_awr_max_weight = 20.0`
(the canonical clip value, which carries over cleanly), and then I do something the off-policy version
does not need: I *self-normalize* the clipped weights to have mean one across the minibatch,
`weights = weights / (weights.sum() + 1e-8) * weights.numel()`. The reason is specific to this loop.
The off-policy version runs the actor for a fixed number of gradient steps with a momentum optimizer at
its own step size; the absolute weight scale just rolls into that step size. Here the regression loss
shares Adam and the global gradient-norm clip with the value loss, and Adam's update is sensitive to the
*scale* of the gradient relative to its running second moment, so a minibatch whose weights happen to be
mostly tiny (every action below average) would produce a vanishing policy gradient and waste the step,
while a minibatch with one huge surviving weight would saturate the gradient-norm clip. Renormalizing
the weights to mean one keeps the effective regression step size constant from minibatch to minibatch,
which is the same robustness the loop's per-minibatch advantage normalization buys on the input side —
this is the on-policy AWR's way of being stable inside a shared-optimizer K-epoch loop, and it is
machinery the canonical buffer-based version simply does not have.

Third, what computes the weights and what carries the gradient. The advantage weights are a *target*,
not a path: the regression should push $\log\pi_\theta$ up on high-weight actions, but the weights
themselves must not receive gradient, or the optimizer could cheat by reshaping the advantage estimate.
So the entire weight computation — the exp, the clamp, the renormalization — sits under
`torch.no_grad()`, and only the `newlogprob` carries gradient into the policy loss
$-\hat{\mathbb E}[w\,\log\pi_\theta]$. The value head gets the same plain MSE toward the GAE returns that
rung 1 used — I am still introducing no clipping discipline, and the regression framing does not change
the critic's job. The full module is in the answer.

It is worth noting what this rung deliberately leaves out relative to the canonical method, so the
reasoning lands the harness's implementation: there is no separate replay buffer or
off-policy reuse (the loop is on-policy), no separate critic/actor optimizers or step-count scheduling
(one shared Adam), no advantage *re*-normalization before the exp beyond what the loop already did, no
action-bound or L2 auxiliary losses, and the value targets are GAE returns rather than buffer
TD($\lambda$). What survives, and what makes this AWR, is the core: replace the importance-ratio
surrogate with an exponentiated-advantage-weighted supervised regression onto the taken actions.

Let me close on falsifiable expectations against the rung-1 numbers, because that is what the next rung
will read. My central bet is *reliability through construction*: because the trust region is baked into
the regression weights rather than enforced by a soft, reactive penalty, I expect the seed-to-seed
swings that plagued ppo-penalty to shrink on at least one environment, and I expect AWR to *win* on the
environments where the advantage signal is clean and the better-than-average actions are well separated —
HalfCheetah most of all, where I expect to clear the penalty rung's 1676.6 mean. On
InvertedDoublePendulum, where the achievable return is large and the dynamics reward decisive
exploitation of high-advantage actions, the sharp $\beta=0.05$ concentration should also help, so I
expect to beat the penalty rung's 6877.1 there too. But I am genuinely worried about the cost of the
sharp temperature on the *low-signal* environment. Swimmer has long-horizon credit assignment and noisy
advantages; a temperature that concentrates the regression so aggressively on the top-weighted actions
will, on noisy advantages, concentrate on *noise* — it will confidently regress toward whichever actions
happened to get a high (possibly spurious) advantage on that minibatch, and with no ratio and no KL to
pull it back, nothing damps that. So my concrete, falsifiable prediction is asymmetric: AWR beats the
penalty rung on HalfCheetah and InvertedDoublePendulum but is *worse* on Swimmer than the penalty rung's
101.4, and possibly the worst Swimmer of any rung. Because the task aggregates by geometric mean, a
Swimmer collapse would cap AWR's overall score even if it wins the other two outright — which would be
the exact signature that the next move is not "abandon the ratio" but "keep a ratio surrogate and make
its trust region *hard and built into the loss* so it is reliable on every environment at once."
