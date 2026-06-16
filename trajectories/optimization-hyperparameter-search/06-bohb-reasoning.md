Hyperband closed the gap exactly where I expected and left exactly the residual I expected. Its convergence
AUC recovered above TPE's on every benchmark — NN 0.9497 (TPE 0.8306), SVM 0.9539 (TPE 0.8763), XGBoost
0.9603 — confirming the multi-fidelity hedge front-loads the curve that TPE's warm-up flattened. And the
`total_evals` came back high and *dead stable across seeds*: 105 on XGBoost, 95 on SVM and the NN, every
seed identical, the deterministic-bracket tell that distinguishes it from DEHB's data-dependent swing (DEHB's
SVM ran 65 to 354 evals across seeds). But on final best it did *not* clearly beat TPE: XGBoost −0.3912
(TPE −0.3921, a wash), SVM 0.9778 (TPE 0.9795, if anything slightly worse), NN −3053 (TPE −3048, slightly
worse). That is the cap I predicted, stated in numbers: Hyperband allocates budget brilliantly but samples
every configuration uniformly at random, so it never *aims* — its quality is bounded by random sampling, and
against TPE's model-aimed configs it wins only on convergence, not on the configs themselves. The whole
ladder now points at one move. The TPE rung had the model and no allocation; the Hyperband rung has the
allocation and no model. They were traded off against each other for two rungs running. The final rung stops
trading and *compounds* them: keep Hyperband's exact bracket schedule, but replace its random configuration
sampling with TPE's `l(x)/g(x)` model, so the multi-fidelity allocation and the model-guided selection work
together.

Let me derive why this composition is the right one and not just a hopeful stapling, because the two parts
have to actually reinforce. Every HPO method makes two decisions that are logically separate: *which*
configuration to evaluate, and *how much* resource to spend on it before judging. The ladder so far has
attacked these one at a time. Random search aimed at neither. CMA-ES aimed the which (a learned
distribution) but fixed the how-much at full and could not amortize the model. DEHB aimed the how-much
(multi-fidelity) and learned the which model-free, inheriting the cheap-rank risk. TPE aimed the which (the
`l/g` model) and fixed the how-much. Hyperband aimed the how-much (the bracket hedge) and left the which to
random sampling. Laid out this way, the two axes have never both been handled by the same rung, and the two
strongest single-axis rungs — TPE on which, Hyperband on how-much — are precisely the two I have in hand.
Hyperband's strength is on the *evaluation* axis: it spends little on hopeless configs and reserves
full-fidelity for survivors, and it hedges the configs-versus-fidelity dilemma so it is robust to a
misleading cheap fidelity. Its weakness is on the *selection* axis: random sampling. TPE's strength is
exactly the selection axis: a probabilistic model of where good configs live that aims each draw at high
`l(x)/g(x)`. Its weakness was the evaluation axis: single fidelity, a warm-up tax, no anytime climb. The
weaknesses are *disjoint* and each method's strength covers the other's weakness — so the composition is not
arbitrary, it is the unique way to get a method that is strong on both axes at once. Hyperband still
*decides how much resource each config gets and when to kill it*; TPE *decides which configs to propose in
the first place*. Replace the line that says "sample uniformly" with "sample from the model" and nothing else
about the bracket machinery has to change.

Now the model has to be fed by the *multi-fidelity* history, which is the one genuinely new design question
the composition raises and where the implementation makes its choices. In plain TPE every observation was a
full-fidelity score; here most observations are cheap low-fidelity scores, and they are noisier. The
implementation's choice is pragmatic: maintain a single pooled set of all completed trials — `(encoded_vec,
score, fidelity)` — and fit one `l/g` model over the *whole pool* regardless of fidelity, rather than the
purist's per-fidelity models. Under a 40–50-cost budget there are simply not enough observations at the
highest fidelity to fit a model there early, so pooling across fidelities is what lets the model turn on at
all within the budget; the cost is that the model treats a cheap noisy score the same as an expensive clean
one, which is a real approximation but the only one that yields a usable model this early. The model warms up
after `n_startup = 8` pooled trials (lower than TPE's 10, because the budget is tighter and the brackets
generate observations fast), then splits at `γ = 0.15` — *more aggressive* than TPE's 0.25, a smaller "good"
set, because with the pooled multi-fidelity history there are more observations and a tighter elite is both
affordable and more exploitative, which is what I want once allocation is handling exploration. The KDEs use
the same single-global-bandwidth simplification as the TPE rung (`bw = max(0.05, std·bw_factor + ε)` with
`bw_factor = 1.0`), and EI is again optimized by sampling `n_candidates = 24` and taking the argmax of
`log l − log g`.

It is worth being precise about why the pooled model still composes correctly with the bracket schedule
despite mixing fidelities, because that is the subtle joint where the staple could fail. The model's job is
only to *rank* candidate configs by `l(x)/g(x)`; it does not need calibrated scores, only a roughly correct
notion of which region the good configs occupy. Cheap low-fidelity scores are biased estimators of the
terminal score, but on most of the space they are *order-correlated* with it — a config that scores in the
top γ at a third of the trees is, more often than not, genuinely in a good region — so pooling them sharpens
`l(x)` toward the right region faster than waiting for scarce full-fidelity data would. The places this
breaks are exactly the configs whose cheap-vs-expensive ranking inverts, and those are rare enough on
XGBoost and SVM that the pooled model is a clear net win there; the NN is the known exception, which is why I
keep flagging it. And crucially, even when the model is mildly misled, Hyperband's *allocation* is the
backstop: a config the model over-rates still has to survive the successive-halving rungs at increasing
fidelity before it consumes a full evaluation, so a model error that the cheap fidelity would have masked
gets caught by the schedule. The model aims; the schedule vetoes. That mutual correction is the real reason
the composition is more than the sum of its parts.

The bracket schedule is Hyperband's, with one tightening for the model: `s_max = min(3, floor(log_η(total)))`
caps brackets at 4 rather than Hyperband's 5, because each bracket now consumes model-guided samples and a
tighter budget should not spread itself across more brackets than the model can inform. On the first call the
brackets are built by drawing each bracket's configs from `_sample_from_model` — which falls back to uniform
until the 8-trial warm-up is met, then samples model-guided — and queued at their initial fidelities. As
trials return, each is appended to the pooled trial set (so the model keeps improving), matched back to its
bracket by exact `config == last.config` and fidelity proximity, and when a bracket's rung completes, the
successive-halving advance runs inline: sort, keep top `len//η`, raise the fidelity, re-queue the survivors.
When the queue empties, instead of Hyperband's random full-fidelity fallback, BOHB generates a *model-guided*
full-fidelity config — so even the fallback is aimed. The distilled module is in the answer.

Where should this land against the whole ladder? It should be the first rung that is strong on *both* axes
simultaneously. On convergence AUC it should at least match Hyperband (the schedule is identical) and
plausibly beat it on the benchmarks where aiming the early cheap configs helps the curve climb faster —
because now the cheap brackets are not flooding random configs but model-guided ones, so the survivors that
reach high fidelity are better from the start. On final best it should finally *beat* the pure-Hyperband
random-sampling cap: I expect XGBoost to improve past Hyperband's −0.3912 toward the best of the ladder,
because the model aims the full-fidelity survivors, and SVM/NN to at least match the strongest band. The
benchmark I am still watching is the NN: it is the one where the cheap fidelity ranks configs poorly, and
pooling cheap and expensive scores into one model means a misleading cheap score can pull the model's elite
toward a region that is good at 50 iterations but not 500 — so I expect a real risk of a single NN seed
cratering on convergence AUC (a bracket whose model was misled by cheap scores), even while the NN final best
improves.

So the falsifiable expectations against the prior numbers are: `total_evals` stable across seeds like
Hyperband's (105/95/95 — same deterministic schedule); convergence AUC at least matching Hyperband's
(XGBoost ~0.96, SVM ~0.95) and the NN possibly lower-variance-breaking on one seed if the pooled model is
misled by cheap scores; and — the decisive one — final best scores that *break the Hyperband cap*, with
XGBoost the clearest case (model-aimed full-fidelity survivors pushing past −0.39 toward the ladder's best,
around −0.389) and SVM/NN at or above the strongest band (SVM ~0.9795, NN improving toward −3015). If I see
final best improve over Hyperband *while* convergence AUC stays at Hyperband's level — strong on selection and
allocation at once — that is the composition working as designed, and it is why BOHB is the strongest rung:
it is the only method on the ladder that is not weak on either axis. The one honest caveat the numbers should
expose is the pooled-fidelity model's vulnerability on the NN, where a single seed's convergence AUC may dip
sharply (the cheap-rank-correlation problem re-entering through the model rather than through triage) — the
residual that a method modeling each fidelity separately, or learning the cross-fidelity correlation, would
be the next thing to attack beyond this ladder.
