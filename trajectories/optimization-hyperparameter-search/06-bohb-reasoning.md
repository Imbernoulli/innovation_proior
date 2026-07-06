Hyperband closed the gap exactly where I expected and left exactly the residual I expected. Its convergence
AUC recovered above TPE's on every benchmark, and the size of each recovery is itself the story: NN 0.9497
against TPE's 0.8306 is a jump of 0.119, SVM 0.9539 against 0.8763 is 0.078, XGBoost 0.9603 against 0.9329 is
0.027 — the recovery is largest exactly on the NN, the benchmark TPE was weakest on, which is the multi-
fidelity hedge front-loading the curve that TPE's 25%-of-budget warm-up had flattened. Line it up against
DEHB too and a quieter result appears: Hyperband's deterministic hedge actually *edged* DEHB's DE-driven
allocation on two of three — NN 0.9497 versus DEHB's 0.9351 and XGBoost 0.9603 versus 0.9492 — and only lost
to it on SVM (0.9539 versus 0.9813), where DEHB's data-dependent promotion happened to keep firing cheap
faithful 2-fold looks. And the `total_evals` came back high and *dead stable across seeds*: 105 on XGBoost, 95
on SVM and the NN, every seed identical, the deterministic-bracket tell that distinguishes it from DEHB's
data-dependent swing (DEHB's SVM ran 65 to 354 evals across seeds).

But on final best it did *not* clearly beat TPE, and the three numbers are worth reading as a set because
together they are the whole motivation for this rung. XGBoost −0.3912 against TPE's −0.3921 is a wash, if
anything a hair worse. SVM 0.9778 against TPE's 0.9795 is worse by 0.0018. NN −3053 against TPE's −3048 is
worse by about 5. So on *every* benchmark Hyperband's final quality lost, narrowly, to TPE's — despite
Hyperband spending far more evaluations and winning convergence handily. That is the cap I predicted, stated in
numbers: Hyperband allocates budget brilliantly but samples every configuration uniformly at random, so it
never *aims* — its quality is bounded by random sampling, and against TPE's model-aimed configs it wins only
on the anytime curve, not on the configs themselves. The whole ladder now points at one move. The TPE rung had
the model and no allocation; the Hyperband rung has the allocation and no model. They were traded off against
each other for two rungs running. The final rung stops trading and *compounds* them: keep Hyperband's exact
bracket schedule, but replace its random configuration sampling with TPE's `l(x)/g(x)` model, so the multi-
fidelity allocation and the model-guided selection work together.

Let me derive why this composition is the right one and not just a hopeful stapling, because the two parts
have to actually reinforce. Every HPO method makes two decisions that are logically separate: *which*
configuration to evaluate, and *how much* resource to spend on it before judging. The ladder so far has
attacked these one at a time. Random search aimed at neither. CMA-ES aimed the which (a learned distribution)
but fixed the how-much at full and could not amortize the model — and paid for it with that 0.2241 SVM AUC
seed. DEHB aimed the how-much (multi-fidelity) and learned the which model-free, inheriting the cheap-rank
risk that cratered an NN seed to −3086. TPE aimed the which (the `l/g` model) and fixed the how-much, buying
the best final scores on the board (XGBoost −0.3921, SVM 0.9795) at the cost of anytime convergence. Hyperband
aimed the how-much (the bracket hedge) and left the which to random sampling, buying the convergence back but
capping final quality below TPE. Laid out this way, the two axes have never both been handled by the same
rung, and the two strongest single-axis rungs — TPE on which, Hyperband on how-much — are precisely the two I
have in hand. Hyperband's strength is on the *evaluation* axis: it spends little on hopeless configs and
reserves full-fidelity for survivors, and it hedges the configs-versus-fidelity dilemma so it is robust to a
misleading cheap fidelity. Its weakness is on the *selection* axis: random sampling. TPE's strength is exactly
the selection axis: a probabilistic model of where good configs live that aims each draw at high `l(x)/g(x)`.
Its weakness was the evaluation axis: single fidelity, a warm-up tax, no anytime climb. The weaknesses are
*disjoint* and each method's strength covers the other's weakness — so the composition is not arbitrary, it is
the unique way to get a method that is strong on both axes at once. Hyperband still *decides how much resource
each config gets and when to kill it*; TPE *decides which configs to propose in the first place*. Replace the
line that says "sample uniformly" with "sample from the model" and nothing else about the bracket machinery
has to change.

It is worth pinning down just how surgical that replacement is on this substrate, because it is what makes the
attribution clean. The bracket schedule uses `s_max = min(3, floor(log_η(total_budget)))`, and since
`log₃ 50 = 3.56` and `log₃ 40 = 3.36` both floor to 3, `s_max = 3` on every benchmark — which is *the same*
`s_max` Hyperband got from its own `min(4, floor(...))`, because the floor(3) binds below both caps at these
budgets. So the two methods run the *identical* bracket schedule: the same four brackets opening 27, 18, 6, 4
configs at fidelities 0.1, 0.111, 0.333, 1.0, the same halving cascades, the same deterministic evaluation
count. The only thing that differs between Hyperband and this rung is the single line that chooses each
config — random there, model-guided here. That is the cleanest isolation the ladder affords: whatever final-
quality gain appears cannot come from the allocation, because the allocation is byte-for-byte Hyperband's; it
can only come from aiming the configs. The tighter `min(3, ...)` cap is insurance for larger budgets (a
budget past 81 would let Hyperband spawn a fifth bracket while this rung would not, on the reasoning that each
bracket now consumes model-guided samples), but at 40 and 50 it is a no-op, which is exactly what I want for a
controlled comparison.

Now the model has to be fed by the *multi-fidelity* history, which is the one genuinely new design question
the composition raises and where the implementation makes its choices. In plain TPE every observation was a
full-fidelity score; here most observations are cheap low-fidelity scores, and they are noisier. The
implementation's choice is pragmatic: maintain a single pooled set of all completed trials — `(encoded_vec,
score, fidelity)` — and fit one `l/g` model over the *whole pool* regardless of fidelity, rather than the
purist's per-fidelity models. The arithmetic forces this. Of the ~95 evaluations a 40-budget run spends, only
about nine ever reach full fidelity — one full-fidelity survivor from the aggressive bracket, two from the
next, two from the next, four from the safe bracket — and they arrive *late*, after the cascades complete. So
a purist per-fidelity model fitted only at fidelity 1.0 would have essentially zero data for most of the run
and could not turn on until the very end; at γ = 0.15 it would need its good set drawn from a handful of
full-fidelity points, which is `int(0.15·9) = 1` — a single-point density, useless. Pooling across fidelities
is what lets the model turn on at all within the budget: after just `n_startup = 8` pooled trials — reached
almost immediately, since the brackets queue 55 configs on the first pass — there is a fittable pool, and it
grows to dozens fast. The cost of pooling is that the model treats a cheap noisy score the same as an
expensive clean one, which is a real approximation, but it is the only one that yields a usable model this
early.

The two model hyperparameters shift from TPE's for reasons the pooling makes concrete. `n_startup = 8` is
lower than TPE's 10 because the brackets generate observations so fast that eight pooled trials arrive in the
first handful of `suggest` calls, so there is no reason to hold the model off longer. And `γ = 0.15` is *more
aggressive* than TPE's 0.25 — a smaller "good" set — precisely because the pooled multi-fidelity history is
large: at a pool of 95, γ = 0.15 gives `int(14.25) = 14` good points, plenty to fit a density, where TPE's
40-observation single-fidelity history could not have afforded so tight an elite (0.15·40 = 6, versus the 10
that 0.25 gave it). A tighter elite is both affordable here and more *exploitative*, which is what I want once
the bracket schedule is handling exploration on its own — the allocation hedges, so the model is free to aim
hard. The KDEs use the same single-global-bandwidth simplification as the TPE rung (`bw = max(0.05,
std·bw_factor + ε)` with `bw_factor = 1.0`), and EI is again optimized by sampling `n_candidates = 24` and
taking the argmax of `log l − log g`.

The compounding is easiest to see through the selection pressure the model puts on the *cheap* rungs, which is
where Hyperband was most blind. The aggressive bracket opens 27 configs at fidelity 0.1. In Hyperband those 27
are a uniform random flood; here, once the 8-trial warm-up is met, each of the 27 is the argmax of `log l −
log g` over 24 fresh draws under a model fitted on the growing pool — so the flood entering the cheapest rung
is already a best-of-24 selection toward the good region rather than best-of-one. That changes what survives
the first halving: Hyperband keeps the top 9 of 27 *random* configs, while this rung keeps the top 9 of 27
*aimed* configs, so the population climbing to 150 iterations, 450, and full fidelity is drawn from a
distribution already concentrated where `l/g` is high. This is the mechanism behind expecting the convergence
curve to climb at least as fast as Hyperband's and the full-fidelity survivors to be genuinely better: the
model does not wait for full fidelity to help, it improves the *input* to every rung of the ladder, cheap
rungs included, which is exactly the front-loading Hyperband's random sampling left on the table. And because
the pool feeding the model is the multi-fidelity history, it reaches a fittable size within the first dozen
evaluations — long before a single-fidelity model would have had anything to say — so the aiming turns on early
in the run when the AUC integral is most sensitive to it.

It is worth being precise about why the pooled model still composes correctly with the bracket schedule
despite mixing fidelities, because that is the subtle joint where the staple could fail. The model's job is
only to *rank* candidate configs by `l(x)/g(x)`; it does not need calibrated scores, only a roughly correct
notion of which region the good configs occupy. Cheap low-fidelity scores are biased estimators of the
terminal score, but on most of the space they are *order-correlated* with it — a config that scores in the top
γ at a third of the trees is, more often than not, genuinely in a good region — so pooling them sharpens
`l(x)` toward the right region faster than waiting for scarce full-fidelity data would. The places this breaks
are exactly the configs whose cheap-vs-expensive ranking inverts, and those are rare enough on XGBoost and SVM
that the pooled model is a clear net win there; the NN is the known exception — its 50-iteration scores rank
configs differently from its 500-iteration truth — which is why I keep flagging it. And crucially, even when
the model is mildly misled, Hyperband's *allocation* is the backstop: a config the model over-rates still has
to survive the successive-halving rungs at increasing fidelity before it consumes a full evaluation, so a
model error that the cheap fidelity would have masked gets caught by the schedule. The model aims; the
schedule vetoes. That mutual correction is the real reason the composition is more than the sum of its parts,
and it is exactly what a pure-model rung (no veto, TPE) and a pure-schedule rung (no aim, Hyperband) each
lacked.

Let me trace the two-way correction on a concrete case to be sure it is real and not a slogan. Suppose the
pooled model, fooled by a cheap NN score, over-rates a region — say a large-`learning_rate_init` config that
looks strong at 50 iterations but diverges by 500. In pure TPE that config would be proposed and then
evaluated at full fidelity, burning a whole cost unit on a loser. Here it enters through an aggressive
bracket at fidelity 0.1 (50 iterations), where it does look good, so it survives the first halving to 150
iterations — and there, if the divergence has started to show, it is culled before it ever reaches the
full-fidelity rung, costing only 0.1 + a share of 0.3 rather than a full 1.0. Conversely, suppose the model
correctly aims at a slow-but-good config: the schedule still makes it prove itself up the fidelity ladder, but
because the model aimed it, the *pool of configs entering the cheap rungs is better on average* than
Hyperband's random flood, so more of the survivors reaching full fidelity are genuinely good. The aim
improves the input to the schedule; the schedule protects the budget from the aim's mistakes. Neither the
warm-up-taxed TPE nor the blind-sampling Hyperband had both halves of that loop.

The bracket schedule is otherwise Hyperband's exactly. On the first call the brackets are built by drawing
each bracket's configs from `_sample_from_model` — which falls back to uniform until the 8-trial warm-up is
met, then samples model-guided — and queued at their initial fidelities. As trials return, each is appended to
the pooled trial set (so the model keeps improving), matched back to its bracket by exact `config ==
last.config` and fidelity proximity, and when a bracket's rung completes, the successive-halving advance runs
inline: sort, keep top `len//η`, raise the fidelity, re-queue the survivors. When the queue empties, instead
of Hyperband's random full-fidelity fallback, BOHB generates a *model-guided* full-fidelity config — so even
the fallback is aimed, which matters because, as the cost arithmetic showed, roughly a fifth to a quarter of
the run is those fallback draws, and having the model aim them rather than sample randomly is free extra
selection pressure on the very evaluations that fill out the budget. The distilled module is in the answer.

One implementation detail falls out favorably and is worth noting because it is where DEHB had to work harder.
The model proposes configs by drawing uniform candidates and selecting the argmax of `log l − log g`; the
selected candidate is a genuine, unperturbed config from the space, which the harness then clips and returns.
So the exact-equality match `cfg == last.config` that Hyperband used still works here — the model does not
*mutate* a config the way DE did, it *chooses among* clean draws — and I do not need DEHB's `np.allclose` on
encoded vectors to reconcile a returned config with a queued one. The pooled trial set does store the
*encoded* vector for the KDE, but bracket bookkeeping stays on the exact config identity, so the queue model
of Hyperband transfers wholesale. That the composition inherits Hyperband's clean matching rather than DEHB's
tolerance-based one is a small sign the parts really do fit: I am adding a selection model to an allocation
skeleton without disturbing how the skeleton tracks its own trials.

Before I trust the whole construction, let me check it degenerates into its own parts at the extremes, because
a composition that does not contain its ingredients as limits is probably mis-assembled. Shrink the budget so
that `floor(log_η budget) = 0` and `s_max = 0`: the schedule collapses to a single bracket of a few configs
straight at fidelity 1.0, there is no cheap rung to triage and no promotion, and every config is drawn from
the model at full fidelity — which is exactly TPE (with an 8-trial warm-up instead of 10). So at a tiny budget
BOHB *is* the model-only rung, the correct floor. Now grow the budget: `s_max` rises, more and cheaper rungs
appear, the pool of observations swells so the per-fidelity data thins its excuse away, and the method becomes
model-guided search running inside a deep multi-fidelity hedge — the full intended object. And hold the model
off entirely (imagine `n_startup` larger than the whole budget): `_sample_from_model` never leaves its uniform
fallback, and BOHB reduces to plain Hyperband, byte for byte. So the construction sits exactly on the line
between TPE and Hyperband, containing each as a boundary case and adding value only in the interior where both
a model and a fidelity ladder are affordable — which is precisely the regime these 40–50-budget benchmarks
live in. That the same object collapses to the two strongest prior rungs at its two extremes is the strongest
evidence I have, short of running it, that the composition is coherent rather than a hopeful staple.

Where should this land against the whole ladder? It should be the first rung that is strong on *both* axes
simultaneously. On convergence AUC it should at least *match* Hyperband — because the schedule is byte-for-byte
identical — and plausibly edge it on the benchmarks where aiming the early cheap configs helps the curve climb
faster, since now the cheap brackets are not flooding random configs but model-guided ones, so the survivors
that reach high fidelity are better from the start. On final best it should finally *beat* the pure-Hyperband
random-sampling cap: I expect XGBoost to improve past Hyperband's −0.3912 toward the best the ladder has
touched — Hyperband and DEHB both already hit −0.3885 on their best seed, so a model that aims the full-
fidelity survivors should pull the *mean* toward that −0.388/−0.389 band — and SVM to reach TPE's model-driven
0.9795 rather than Hyperband's random 0.9778. The NN is the benchmark I am still watching. It is the one where
the cheap fidelity ranks configs poorly, and pooling cheap and expensive scores into one model means a
misleading cheap score can pull the model's elite toward a region that is good at 50 iterations but not 500 —
so I expect a real risk that a single NN seed's convergence AUC dips sharply, a bracket whose model was misled
by cheap scores, even while the NN *final best* improves. On that final best, the reachable target is set by
what the ladder has already shown is possible: DEHB touched −2998 on its best NN seed, so a model-aimed BOHB
should be able to pull the NN mean below Hyperband's −3053 and TPE's −3048, into the low −3000s or better,
without my needing to claim a precise figure the run has not yet produced.

So the falsifiable expectations against the prior numbers are: `total_evals` stable across seeds and equal to
Hyperband's (105/95/95 — the schedule is identical, so this must hold or something is wrong); convergence AUC
at least matching Hyperband's (XGBoost ~0.96, SVM ~0.95, NN ~0.95) with the NN possibly showing a single
low-variance-breaking seed if the pooled model is misled by cheap scores; and — the decisive one — final best
scores that *break the Hyperband cap*, with XGBoost the clearest case (model-aimed full-fidelity survivors
pushing the mean past −0.391 toward the −0.3885 the best seeds already reached) and SVM matching TPE's 0.9795
and the NN improving over −3053. If I see final best improve over Hyperband *while* convergence AUC stays at
Hyperband's level — strong on selection and allocation at once — that is the composition working as designed,
and it is why this is the strongest rung: it is the only method on the ladder that is not weak on either axis.
The one honest caveat the numbers should expose is the pooled-fidelity model's vulnerability on the NN, where
a single seed's convergence AUC may dip sharply — the cheap-rank-correlation problem re-entering through the
model rather than through triage — which is the residual a method modeling each fidelity separately, or
learning the cross-fidelity correlation, would be the next thing to attack beyond this ladder. But that
residual is narrow and named; everything else about the profile — matched convergence, broken final-best cap,
seed-stable cost — is the two-axis composition finally paying off, and it is the reason the ladder ends here:
no ingredient on the board is being traded against another anymore.
