# Synthesis V4 — PPO (in-frame; composed-from for results_v4)

This notes file is the single source the three V4 deliverables are transcribed from.
It carries (A) the load-bearing-ancestor write-ups, (B) the central first-principles
object + the precise difficulty, (C) the design-decision → why table (with rejected
alternatives + their failure modes), and (D) the V4-specific framing constraints so
the deliverables don't drift.

Target = PPO (clipped surrogate). In-frame: NEVER name PPO/"clip"/"surrogate"/
"trust-region-via-clip" in context.md; never reference the source paper as an artifact
in any file. Prior-art ancestors (vanilla-PG/CPI/TRPO/A3C/GAE/Adam/DQN/CEM) cite freely.

------------------------------------------------------------------------------------
## A. Load-bearing ancestors (the lineage) — substantive write-ups

1. **Score-function / likelihood-ratio policy gradient (Williams 1992; Sutton et al.
   2000).** ∇_θ η = E_t[∇_θ log π_θ(a_t|s_t) · Ψ_t]. Lowest-variance natural Ψ is the
   advantage A=Q−V; a state-only baseline V integrates out (E_a[∇log π]=∇_θ 1 = 0) so
   it's unbiased but kills the common-mode variance. With autodiff you write a scalar
   surrogate L^PG = Ê_t[log π_θ Â] whose gradient = ĝ. GAP: one gradient step per
   sample (data-inefficient); step size treacherous — too large => policy collapses,
   and a collapsed policy doesn't recover. Multi-epoch ascent on L^PG is exactly what
   is NOT justified (surrogate only valid at current θ).

2. **Conservative Policy Iteration / surrogate bound (Kakade & Langford 2002).** Exact
   identity: η(π) − η(π_old) = E_{τ∼π}[Σ_t γ^t A_{π_old}(s_t,a_t)] — advantage of OLD
   policy under NEW policy's own trajectory dist. Proof = advantage telescopes:
   A=E[r+γV(s')−V(s)]; Σγ^t of it along τ∼π collapses the value chain leaving
   Σγ^t r_t − V(s_0); take E over π => η(π) − E[V(s_0)] = η(π) − η(π_old). Unusable
   directly (E over π's own states, unknown). Swap to ρ_old:
   L_{π_old}(π) = η(π_old) + E_{s∼ρ_old, a∼π}[A_{π_old}(s,a)]. Matches η to first order
   at π=π_old; error of the swap BOUNDED by a TV/KL distance × constant => genuine
   LOWER BOUND on improvement while the policy stays close. SEED IDEA: closeness is the
   precise condition that makes the surrogate honest, not a heuristic rail. Source of
   the "CPI" name.

3. **TRPO (Schulman et al. 2015b).** max_θ Ê_t[r_t Â] s.t. Ê_t[KL[π_old,π_θ]] ≤ δ,
   r_t = π_θ/π_old. Solves over ~1e6 params: linearize objective (grad g), quadratic
   approx of KL (Hessian = Fisher F), step = natural-gradient F⁻¹g scaled to δ boundary,
   θ = θ_old + √(2δ/(gᵀF⁻¹g))·F⁻¹g. F never formed — conjugate gradient on Fisher-vector
   products (couple backprops each) + backtracking line search to enforce true KL +
   verify improvement. Reliable on continuous control. GAPS (sharp): (a) heavy 2nd-order
   machinery (CG, FVP, line search); (b) ~one constrained step per batch — no natural
   K-epoch cheap minibatch SGD => leaves data efficiency on the table; (c) architecture-
   hostile — KL defined on policy output dist, so sharing params with the value head, or
   dropout/noise/aux heads, smears the very distribution the Fisher measures. TRPO ALSO
   notes the cleaner PENALTY form max Ê_t[r_t Â − β KL] (plain first-order SGD), but the
   β the bound hands you uses MAX KL over states => β enormous => steps useless; a single
   fixed hand-chosen β won't hold across tasks or within a run (advantage scale + KL
   sensitivity drift).

4. **A3C / A2C (Mnih et al. 2016).** Many actor-learners, each on its own env copy,
   accumulate grads for a shared net. Fixed rollout length T << episode, n-step
   bootstrapped advantage Â_t = −V(s_t)+r_t+γr_{t+1}+...+γ^{T−t}V(s_T). Two reusable
   ingredients: (i) a single net SHARING params between policy head + value head (so the
   value loss folds into the objective); (ii) an ENTROPY bonus S[π_θ](s) to discourage
   premature deterministic collapse. A2C = synchronous GPU-friendly variant (wait for all
   N workers, batch NT, one sync update). GAP: still one grad step per batch, no trust
   region — with param sharing that single step can be destructive. Parallel short
   rollouts also DECORRELATE data (successive states in one trajectory are correlated =>
   inflated grad variance; stitching N independent streams breaks it — the on-policy
   analog of DQN's replay buffer).

5. **GAE (Schulman et al. 2015a).** δ_t = r_t + γV(s_{t+1}) − V(s_t) (one-step adv est);
   Â_t^{(γ,λ)} = Σ_{l≥0} (γλ)^l δ_{t+l}. λ = bias/variance knob: λ→0 => δ_t (low var,
   biased by V); λ→1 telescopes to Σγ^l r_{t+l} − V(s_t) = Monte-Carlo advantage (unbiased,
   high var). λ≈0.95 with γ=0.99 best. Truncated to T-rollout => A3C n-step at λ=1, so a
   strict generalization. One reverse scan: Â_t = δ_t + γλ(1−done)Â_{t+1}. THE advantage
   estimator a new method reuses.

6. **DQN (Mnih et al. 2015).** Q(s,a;θ) via net, replay + target net, TD min toward
   r+γ max_a' Q(s',a';θ⁻). Cracked Atari from pixels. GAP: built around discrete argmax
   => no clean continuous control (MuJoCo); brittle/poorly-understood (reward-scale /
   hyperparam sensitivity).

7. **Importance sampling.** E_{a∼π}[A] = E_{a∼π_old}[(π/π_old)A]. r_t = π_θ/π_old, r=1 at
   start. DANGER lives here: as π_θ drifts, ratios fan out, variance grows, a few big-r
   samples dominate; worse, the optimizer SEEKS big r on +adv samples (cheapest way to
   raise the surrogate — push one number up, not find a better action). The ratio is the
   distance signal AND the danger signal at once.

8. **Adam (Kingma & Ba 2014); CEM (Szita & Lőrincz 2006); adaptive-stepsize PG.** Adam =
   the first-order optimizer the whole premise rests on (eps=1e-5 not 1e-8). CEM =
   gradient-free black-box over params, scales poorly with param count. Adaptive-PG =
   Adam stepsize rescaled by realized KL each batch; still one step per batch. Extra
   yardsticks.

------------------------------------------------------------------------------------
## B. Central object + the precise difficulty

OBJECT: maximize η(π_θ) = E[Σ_t γ^t r_t], stochastic policy.
WHY HARD (the exact tension): reusing one on-policy batch for several gradient steps =
the data efficiency we want, but it is also what makes naive multi-epoch PG ascent blow
up — the surrogate is valid only while π_θ ≈ π_old, multi-epoch ascent walks the policy
out of that region where the surrogate decouples from η and the update turns destructive.
TRPO controls the drift but only with 2nd-order machinery that resists parallelism, param
sharing, stochastic units, and cheap minibatch reuse.
SHARPENED QUESTION: recover TRPO's reliable drift-controlled behavior with ONLY first-
order optimization + ordinary minibatch SGD — cheap, general, a few lines on top of
vanilla PG.

CHAIN OF APPROXIMATIONS (theory → practice):
η-difference identity (KL02) → swap ρ_π → ρ_old (surrogate honest while close) →
importance sampling to reuse π_old data => L^CPI = Ê[r_t Â] → leash needed → KL trust
region (TRPO, but heavy) → penalty form (β won't sit still) → FLIP: refuse credit for
leaving a ratio band → clip(r,1−ε,1+ε)·Â → plain clip freezes overshoots → take min of
clipped & unclipped => L^CLIP (pessimistic lower bound) → multi-epoch minibatch Adam now
safe → fold in GAE + value head + entropy + hygiene.

SPECIAL CASE FALL-OUT: at θ_old every r=1, clip inactive => L^CLIP = L^CPI = ordinary PG,
so we never broke the gradient we started from; the brake only engages once θ moves.

------------------------------------------------------------------------------------
## C. Design-decision → WHY table (rejected alternatives + failure modes)

| Decision | Why this | Rejected alternative & its failure |
|---|---|---|
| Optimize advantage Â, not raw return | raw Σγ^t r_t huge/all-positive/high-var => common-mode noise drowns signal | weight by raw return: every logprob pushed up, useful signal lost |
| State-only baseline V(s) | integrates out (unbiased), slashes variance | no baseline: variance scales with absolute return |
| Write scalar surrogate, autodiff it | grad of L^PG = ĝ; let framework differentiate | hand-code gradient: brittle, no autodiff reuse |
| Importance ratio r_t = π_θ/π_old | reuse π_old data for "a∼π" expectation; r=1 at start = exactly PG | — (it's the only bridge) |
| Leash on r needed | optimizer seeks big-r on +adv (cheap surrogate inflation) => blowup | unleashed L^CPI: same disaster as multi-epoch L^PG |
| Reject KL hard-constraint (TRPO) for the new method | want first-order + K-epoch minibatch + param sharing | TRPO: CG/FVP/line-search heavy, 1 step/batch, arch-hostile |
| Reject FIXED β penalty | right β depends on adv scale + KL sensitivity, both drift across tasks & within a run | fixed β: reasonable early => tiny late (or vice-versa); the bound's β uses max-KL => enormous => useless steps |
| Clip ratio to [1−ε,1+ε] inside objective | flatten => gradient dies once r leaves band; trust region as a flat spot in the loss; ε unit-free => 1 value robust across tasks where β isn't | KL penalty: needs a coefficient that won't sit still |
| min(rÂ, clip(r)Â) — keep the pessimistic term | plain clip only flattens, doesn't push back => FREEZES an overshoot (Â<0 already at r=3: clipped term flat, gradient 0, can't correct). min restores corrective gradient: ignore a ratio move only when ignoring makes the objective LOOK BETTER, always include when worse => pessimistic lower bound on L^CPI | plain clip (no min): freezes wrong-direction overshoots |
| ε = 0.2 | ±20% per-update ratio band; big enough for real K-epoch progress, small enough realized KL stays ~0.01–0.02 (bound tight) | ε=0.05: brake bites instantly, microscopic steps (TRPO-tiny-δ), wastes reuse. ε=0.5/1.0: band too wide, clip rarely engages => near-unclipped L^CPI, blowup returns |
| Keep adaptive-KL-penalty as a baseline variant | servo β onto d_targ (d<d_targ/1.5 => β/=2; d>d_targ·1.5 => β*=2); self-tunes => works where fixed β didn't; init β barely matters | — kept as baseline to beat; carries KL in loss + outer loop, expect clip simpler & ≥ as good |
| 1.5 / 2 in the β servo | heuristic, forgiving — β self-corrects in a few iters | — |
| Parallel N actors × fixed T steps | decorrelate data (break within-trajectory correlation), pool NT/batch, scalable | single long trajectory: correlated samples, inflated variance |
| Truncated GAE for Â | one λ knob trades bias/var; λ=1 => A3C n-step (strict generalization); reverse scan w/ (1−done) mask | n-step only: no bias/var knob. MC: high var. δ_t only: biased by V |
| Fit V via squared error, target Â+V_old | V is the baseline that made GAE low-var; consistent w/ computed GAE | MC returns directly: lose the variance reduction λ<1 buys |
| SHARE net or separate; fold value loss in | objective is just a diff loss => sharing is now FREE (the thing TRPO's Fisher made awkward); c_1 weights value term | TRPO sharing: smears the Fisher / KL the constraint needs |
| Clip the value loss too (max of unclipped & V_old±ε-clipped sq err) | value head on same drifting net, same K epochs => analogous overshoot; symmetry w/ policy clip, never let clip REDUCE the loss | unclipped V over K epochs: V can lurch far from V_old |
| Entropy bonus c_2 S[π_θ] | sustain exploration, slow premature deterministic collapse (A3C) | none on categorical (Atari) => premature collapse; c_2=0 fine on MuJoCo (Gaussian log-std is its own exploration knob) |
| Orthogonal init, hidden gain √2 | preserves activation/grad variance through tanh over 2 hidden layers | bad gain: signal shrinks/blows before learning |
| Policy-head gain 0.01 | initial action dist near-uniform/centered => start exploring, not born over-confident in an arbitrary action | large gain: over-confident init, stuck in bad deterministic basin |
| Value-head gain 1.0 | plain regression head, no reason to shrink | — |
| Adam eps = 1e-5 (not 1e-8) | RL grads get small; too-tiny denom floor lets step blow up when 2nd-moment ≈ 0 | 1e-8: ill-conditioned updates at small grads |
| Linear LR anneal → 0 | big steps early (any reasonable dir helps), small late (don't jitter out of a good policy) | constant LR: late jitter |
| Per-minibatch adv normalization (−mean/÷std) | keeps effective step scale constant across tasks/training regardless of reward/value scale; done where the loss is computed | unnormalized: step scale swings with reward scale |
| Global grad-norm clip 0.5 | cheap last line vs an occasional spiky minibatch grad; caps length not direction | none: one bad minibatch step destructive |
| State-INDEPENDENT log-std (one vector) | σ(s) head could silently drive exploration→0 in specific states early; one global σ keeps exploration honest, 1 fewer thing to destabilize | σ(s): per-state exploration collapse |

------------------------------------------------------------------------------------
## D. V4 framing constraints (so the files don't drift from spec)

- **context.md** five headers: Research question / Background / Baselines / Evaluation
  settings / Code framework. NO "PPO"/"clip"/"surrogate"/"trust-region-via-clip"; NO
  source-paper artifact; NO outcomes; pre-method only.
- **CODE FRAMEWORK = MINIMAL PRE-METHOD SCAFFOLD.** Presuppose NOTHING about the policy-
  improvement rule. Scaffold = bare on-policy PG harness ONLY:
    * Gym vectorized env + rollout collection into preallocated buffers
    * `class Policy: # TODO` (incl. a value head)
    * a GAE advantage estimator (prior art — may be named "GAE")
    * a generic `def update(rollouts): # TODO: the policy-improvement objective we'll design`
    * an Adam optimizer
    * the rollout → advantage → minibatch training loop
  NO "reference implementation"/"official repo" wording. NO method names (no clip/
  surrogate/PPO/trust-region-via-clip). The final code FILLS IN update() and corresponds
  piece-for-piece to this scaffold. (Write final code first, then hollow update()/Policy.)
- **reasoning.md**: continuous first-person, ZERO real markdown headers, present tense,
  all derivations inline (CPI surrogate + telescope proof, IS ratio, clip + min CASE
  ANALYSIS for Â>0 and Â<0, why plain clip freezes overshoots, GAE telescoping λ=0/λ=1).
  DEAD ENDS lived: multi-epoch L^PG diverges; fixed β won't sit still. AHA: the min as a
  pessimistic lower bound; clip = trust region as a flat spot. INSIGHT-BEFORE-METHOD at
  every step (motivation first, formula drops out). 2.4 revision pass.
- **answer.md**: opens with the method (PPO), faithful CleanRL code, no citation header,
  no arXiv links in code.
- Code grounded in CleanRL ppo_continuous_action.py (verified: max(pg_loss1,pg_loss2),
  clipped value loss, GAE reverse scan, orthogonal init gains, Adam eps 1e-5, anneal,
  per-mb adv norm, grad clip 0.5, state-indep logstd, T=2048/K=10/mb=64/γ.99/λ.95/ε.2).
