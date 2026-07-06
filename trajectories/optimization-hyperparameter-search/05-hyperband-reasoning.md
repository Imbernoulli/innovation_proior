TPE landed exactly on the profile I predicted, and the contrast with DEHB is the whole point. `total_evals`
came back pinned to the budget — 50 on XGBoost, 40 on SVM and the NN, a flat count with no seed-to-seed
variance — confirming the model-only rung does no triage and spends exactly one cost unit per look. The final
quality recovered and then some: on SVM TPE's best reached 0.9795, which beats not only DEHB's triaged 0.9661
but even random search's 0.9778; on XGBoost it reached −0.3921, the best final score of any rung so far
(random −0.3939, DEHB −0.4016); and the NN best came back to −3048.1 with its worst seed at −3063.6 instead
of DEHB's −3086, so removing the low-fidelity mis-promotion fixed the slip exactly as the diagnosis said. A
model spending every full-price evaluation on an aimed config is the reliable way to buy final quality.

But the convergence AUC distinctly *lost* to DEHB, and the size of the loss is itself informative. SVM 0.8763
against DEHB's 0.9813 is a gap of 0.105; the NN 0.8306 against 0.9351 is a gap of 0.104; but XGBoost 0.9329
against 0.9492 is a gap of only 0.016. Why is the XGBoost gap six times smaller than the other two? Read it
off the warm-up arithmetic. TPE spends its first 10 configs on a random warm-up before the model turns on,
and that is 10 of 40 — a full 25% of the budget — on SVM and the NN, but only 10 of 50 — 20% — on XGBoost.
During that warm-up the best-so-far curve climbs no faster than random search's, so a fixed slab of the AUC
integral is ceded before the model makes a single aimed suggestion, and the slab is largest exactly on the
two 40-budget benchmarks where the gap to DEHB is largest. The XGBoost gap is small because its budget is
bigger and its warm-up fraction smaller, so the model gets more of the run to front-load the curve. The
lesson is now sharp and it is symmetric: a good *model* fixes final quality and reliability; cheap
*multi-fidelity* fixes anytime convergence; neither has both. DEHB had the multi-fidelity half but bolted a
model-free DE learner onto it and inherited the low-fidelity-correlation risk that cratered one NN seed and
swung its SVM `total_evals` from 65 to 354 across seeds. The question this rung isolates is the multi-fidelity
skeleton *on its own*, done right — without DE, without a model — so I can see what principled resource
allocation buys by itself and have a clean, *deterministic* substrate to graft a model onto next.

Before committing to that, let me be honest about the alternatives on the table. One: simply add multi-
fidelity to TPE right now — feed the model with cheap evaluations and let it aim. That is the eventual
combination and I will build to it, but doing it now confounds two changes at once; if the combined method
wins I will not know whether the gain came from the allocation skeleton or the model living inside it, and the
whole point of a ladder is to attribute each gain to one ingredient. Two: just re-run DEHB and tune it — but
DEHB's damage came from *two* coupled sources, its DE learner's data-dependent promotion (the 65-to-354 swing)
and its low-fidelity mis-promotion, and I cannot cleanly separate the allocation skeleton from the learner
inside DEHB. Three, which is the disciplined move: isolate the pure allocation skeleton with *random*
sampling — no DE, no model — so the only thing on trial is whether principled resource allocation, by itself,
recovers the anytime convergence TPE lost. That isolation is what makes the next rung's gain attributable
purely to adding a model.

Strip the problem to the evaluation side. The waste that random search and TPE both pay is that every
configuration is trained to full resource before I look at it, so with budget B I see only B configurations.
But I *know* most randomly drawn configs declare themselves hopeless after a sliver of resource — a
learning rate an order of magnitude too high diverges in the first few NN iterations, a too-shallow tree
ensemble plateaus after a fraction of its trees. Random search pays full price for that information when a
sliver would do. The lever is not *which* config (TPE's lever) but *how much* I spend on each before I throw
it out. The clean framing is a non-stochastic best-arm-identification bandit: a config is an arm, training it
one more unit is a pull, and the partial loss ℓ_{i,j} converges to a terminal value ν_i at an *unknown
rate*. I want the arm with the best ν_i, but I cannot invoke i.i.d. concentration — the loss sequence is an
arbitrary converging sequence, not noise around a mean.

Successive halving is the algorithm that exploits this without knowing the rate: evaluate N arms at the
lowest resource, keep the top 1/η, multiply the resource by η, repeat. Survivors get exponentially more
resource; the full evaluation is paid only for the final handful. The reason it works against an arbitrary
converging sequence is an envelope argument worth spelling out, because it is what makes the method
principled rather than a heuristic. Define γ(j) as the smallest non-increasing function bounding how far any
partial loss at resource j can sit from its terminal value — it exists because the limits exist. Two arms
with terminal gap Δ have envelopes that stop overlapping once 2γ(j) ≤ Δ, i.e. at resource τ = γ⁻¹(Δ/2); at
or beyond τ the ordering of the *partial* losses is guaranteed to match the ordering of the *terminal*
losses. So a config separates from the best once the resource clears its gap-dependent threshold — small when
curves converge fast or the gap is large, large when curves crawl or the gap is tiny. SH climbs the resource
ladder exactly so that, rung by rung, more configs cross their separation threshold and the surviving
ordering converges to the terminal one without ever needing to know γ. Let me make the envelope concrete
against this harness so it is not just symbols. On the NN, resource is `max_iter` and the cheapest rung is 50
iterations; a config whose terminal advantage Δ is large — say a well-scaled learning rate against a divergent
one — has its envelopes separate at a small τ, so 50 iterations already ranks them correctly and triage is
safe. But a config whose terminal edge is *slow to appear* — a small learning rate that needs 400 iterations
to overtake a faster-but-worse rival — has a large τ that 50 iterations does not clear, so at the cheapest
rung its partial loss still sits above the rival's and SH throws it out. That is exactly the DEHB-NN failure
(seed 42, −3086) read through the envelope: not bad luck, but a config whose separation threshold exceeded the
cheapest rung's resource.

But SH has one input it cannot set for itself: N. For a fixed budget, many configs each run cheaply (large N)
is right *only if* the cheap resource ranks configs like the expensive one; if the cheap fidelity is
misleading, large N discards at low resource exactly the config that would have won at full. That is the
configurations-versus-fidelity dilemma, and it is the *same* failure I watched bite DEHB's NN. Hyperband
refuses to guess N and *hedges* across the whole spectrum: run several SH instances ("brackets"), each
starting at a different number-of-configs-versus-initial-fidelity tradeoff. The most aggressive bracket throws
a flood of configs at the cheapest fidelity (great if cheap predicts expensive); the least aggressive bracket
is essentially random search at full fidelity (the safe fallback when cheap is misleading). By spanning the
brackets, Hyperband covers the entire dilemma and is at most about (number of brackets) times slower than the
best fixed choice — a guarantee no single SH setting has. The reason this is the right shape of hedge, and not
just "try several settings," is that the brackets are *complementary*: each one is the optimal allocation
under a different assumption about how predictive the cheap fidelity is, and since I do not know which
assumption holds, paying a logarithmic-factor overhead to run all of them buys robustness to being wrong about
the one thing SH most needs to assume. On a benchmark where the cheap fidelity is faithful — SVM, whose cheap
rungs are 2-fold CV that ranks like 5-fold — the aggressive bracket carries the run and the safe bracket is
cheap insurance; on a benchmark where it lies — the NN's 50 versus 500 iterations — the safe bracket protects
the eventual winner that the aggressive bracket would have killed. That is exactly the DEHB-NN failure mode
turned into a hedge instead of a gamble. Critically, every config is still sampled *uniformly at random*:
Hyperband is purely an allocation method, it never uses one evaluation's outcome to decide where to look next.
That is exactly its limitation and exactly why it is the right rung here — it measures the multi-fidelity
skeleton in isolation, with random sampling, so the gap between it and the next rung will be attributable
purely to adding a model.

Now the implementation the scaffold fills in, derived against the harness, and here the bracket schedule is
worth computing out because it is what makes this rung deterministic. With η = 3, `s_max = min(4,
floor(log_η(total_budget)))`, and since `log₃ 50 = 3.56` and `log₃ 40 = 3.36` both floor to 3, every
benchmark gets `s_max = 3` — the cap at 4 is insurance that does not bind at these budgets. For each bracket s
from 3 down to 0 the number of configs is `n = min(ceil((s_max+1)/(s+1))·η^s, total_budget)` and the initial
fidelity is `r = max(1/η^s, 0.1)`. Compute the four brackets: s = 3 gives `ceil(4/4)·27 = 27` configs at
fidelity 0.1; s = 2 gives `ceil(4/3)·9 = 2·9 = 18` at 0.111; s = 1 gives `ceil(4/2)·3 = 2·3 = 6` at 0.333;
s = 0 gives `ceil(4/1)·1 = 4` at 1.0. So the run opens with 27 + 18 + 6 + 4 = 55 queued configurations
spanning the full spectrum — 27 cheap flood configs at one end, 4 full-fidelity random-search configs at the
other — before a single halving fires, and every one of those counts is fixed by the budget alone with no
dependence on the data or the seed. Each bracket then advances: when its current rung is fully scored, sort by
score, keep the top `len//η` (so 27 → 9 → 3 → 1, 18 → 6 → 2, 6 → 2), multiply the fidelity by η capped at
1.0, and re-queue the survivors. The `suggest` loop is queue-driven: it pops the next queued (config,
fidelity), and when a trial returns it matches it back to its bracket by exact `config == last.config` and
`|fidelity − last.budget| < 0.05`, fills in the score, and advances the bracket when its rung completes. When
the queue empties, fall back to a random full-fidelity draw. The exact-equality match is a real constraint of
this harness: it works because the configs are reproducibly generated and clipped, so the returned config
compares equal to the queued one; a method that perturbed configs would break this matching, which is why
Hyperband (random configs, never perturbed) fits the queue model cleanly where DEHB needed `np.allclose` on
encoded vectors. The distilled module is in the answer.

Let me verify the schedule degenerates sensibly and that the halving preserves at least one survivor, because
an off-by-one in `len//η` could silently drop a bracket to zero. The `n_keep = max(1, len // η)` floor
guarantees a bracket never empties mid-cascade: the 6-config bracket goes `6 → max(1, 2) = 2 → max(1, 0) = 1`
and stops at a single full-fidelity survivor, and even the 4-config full-fidelity bracket, which starts at
fidelity 1.0, simply never advances (its fidelity is already 1.0) and evaluates its 4 configs as plain random
search — the correct degenerate limit of a bracket with no room to triage. Push the budget down toward `s_max
= 0` and the whole schedule collapses to that single bracket: Hyperband becomes random search, which is the
right floor for a method that only ever *allocates*. Good — the schedule is well-formed, and its total count
of evaluations is fixed the instant the budget is known, which is the property I most want to see confirmed.

It helps to trace what resource the survivors of one bracket actually climb, because the fidelity ladder maps
onto each benchmark's cost knob differently and that is where the hedge earns its keep. Take the aggressive
bracket s = 3, whose fidelities across the cascade are 0.1, 0.3, 0.9, 1.0. On the NN these become `max(50,
int(500·fid))` iterations — 50, 150, 450, 500 — so a config that survives all three halvings is trained at 50
iterations first (cheap, noisy), then 150, then 450, then a full 500, and the resource it earns grows only as
it keeps winning. On SVM the same fidelities become `max(2, int(5·fid))` folds — 2, 2, 4, 5 — so the two
cheapest rungs are both 2-fold CV and only the survivors reach 4- and 5-fold, which is why SVM's cheap triage
is faithful: 2-fold ranks configs much like 5-fold on a clean binary problem. On XGBoost the fidelity scales
`n_estimators` (floor 10), so survivors climb from a stub ensemble toward the full one. The point is that the
*same* bracket schedule adapts its meaning to each benchmark's fidelity contract, and the safe bracket s = 0
— four configs straight at full resource — is the insurance that holds when, as on the NN, the cheapest rung's
resource is too little to rank correctly.

Now the cost accounting, because it is what makes `total_evals` both large and deterministic. The opening 55
configs cost `27·0.1 + 18·0.111 + 6·0.333 + 4·1.0 = 2.7 + 2.0 + 2.0 + 4.0 = 10.7` cost units — so on the
40-budget benchmarks the opening flood already spends about a quarter of the budget while buying 55 looks, the
multi-fidelity lever in one line. Running every bracket's halving cascade to completion adds another ~12 cost
units (the survivors at 0.3, 0.9, 1.0 and so on), for roughly 23 cost units and 78 total evaluations to
exhaust the schedule. But 23 is well under the 40 or 50 budget, so the loop does not stop there: with the
queue empty it falls to random full-fidelity draws, each costing a full unit, until the cost clears the
budget. That means the 40-budget benchmarks absorb roughly another seventeen full-fidelity fallback draws and
XGBoost's larger budget roughly twenty-seven — pushing the total count into the nineties on SVM and the NN and
past a hundred on XGBoost. Every term in that sum is fixed by the budget, so the count is not just inflated but
*identical across seeds* — which is the concrete, checkable difference from DEHB, whose promotion cascade read
the data and swung its SVM count from 65 to 354.

One choice I have been treating as given deserves its own justification: why η = 3. The halving rate sets both
how many configs each rung keeps (1/η) and how much the resource multiplies (×η), and the theory that the
envelope argument sits inside says the total work to identify the best arm is minimized near η = e ≈ 2.72; 3
is the smallest integer at that optimum, and it has the added virtue on this substrate that ×3 on the fidelity
matches the `1/η^s` ladder cleanly (0.1 → 0.3 → 0.9 → 1.0). A larger η — say 5 — would keep too few survivors
per rung (27 → 5 → 1), collapsing the population before the resource is high enough to rank it reliably; a
smaller η — 2 — would climb the resource too slowly, spending many rungs before the survivors reach full
fidelity and eating the budget on intermediate looks. η = 3 is the balance, and it is the same rate DEHB used,
so holding it fixed keeps this rung a clean isolation of *allocation* rather than a re-tuning of the ladder.
To see the collapse concretely: at η = 5 the aggressive bracket would open 27 configs and keep only
`27 // 5 = 5` after the first halving and then `5 // 5 = 1`, so it reaches a single survivor in two rungs
having spent almost the entire flood at the noisiest fidelity — the population is gone before the resource is
high enough to trust the ranking. At η = 2 the same 27 configs would step 27 → 13 → 6 → 3 → 1, five rungs, and
the resource would only reach full fidelity after four intermediate looks per survivor, spending the budget on
half-trained models. Three keeps the survivor count healthy at each rung (27 → 9 → 3 → 1) while tripling the
resource each step, which is exactly the schedule the cost accounting above showed lands the total count where
it does — another reason the choice is forced rather than free.

Where should this land against TPE and DEHB? Hyperband should recover the *anytime* strength TPE lost, because
the aggressive brackets surface a decent config from the cheap fidelity fast — so I expect its convergence AUC
to beat TPE's clearly (TPE NN 0.8306, SVM 0.8763) and to be competitive with DEHB's (NN 0.9351, SVM 0.9813).
Its `total_evals` should climb well past the budget like DEHB's — the 55 opening configs plus the cheap rungs
stretch the cost far beyond 40 or 50 — but, and this is the sharp distinction, it should be *identical across
seeds*, because the bracket schedule is deterministic given the budget and nothing in it reads the data. That
is the exact opposite of DEHB's seed-to-seed swing (65 to 354 on SVM) that came from DE's data-dependent
promotion, and it is a virtue in its own right: reproducible cost. On final best, Hyperband should match the
competitive band but I expect it to *not* clearly beat TPE, because it samples randomly — it allocates budget
brilliantly but never aims, so against TPE's model-aimed configs (SVM 0.9795, XGBoost −0.3921) it should be
roughly even on final quality and win only on convergence. And the NN remains the risk benchmark: Hyperband
triages at low fidelity too, so where the envelope threshold τ exceeds the cheapest rung's 50 iterations, an
aggressive bracket can still kill the eventual winner — though the safe full-fidelity bracket should keep the
mean from cratering the way DEHB's single seed did.

I should also expect the same convergence-AUC-above-1.0 artifact DEHB showed, and for the same mechanical
reason, though in a milder and more predictable form. The AUC integrates the min-max-normalized best-so-far
curve, and Hyperband, like DEHB, floods the run with cheap noisy low-fidelity scores that can drag the
normalization floor very low; when the good configs then surface, the normalized curve can overshoot and the
integral can read above 1.0. So I would not be surprised to see a Hyperband seed cross 1.0 on the NN or
XGBoost, exactly where DEHB did (DEHB's NN seeds 1.0035, 1.0079, one XGBoost seed 1.0031) — but because the
bracket schedule fixes *how many* cheap scores land and when, the overshoot should be less seed-to-seed
erratic than DEHB's, tracking the deterministic count rather than DE's data-dependent promotion. If instead
the AUC stays cleanly below 1.0 everywhere, that would tell me the fallback full-fidelity draws that fill out
the budget are dominating the tail of the curve and steadying the normalization — either way the number is
interpretable through the same lens.

So the falsifiable expectations against the prior numbers: convergence AUC should rise back above TPE's on
every benchmark (especially the NN, where TPE's 0.8306 was weakest) and sit near DEHB's, because the multi-
fidelity hedge front-loads the curve; `total_evals` should be well above the budget *and dead stable across
seeds* — the deterministic-bracket tell, distinguishing it cleanly from DEHB's variance — because the 55
opening configs and every halving are fixed by the budget alone; and final best should track the competitive
band without clearly beating TPE, because random sampling allocates but does not aim. If I see exactly that —
AUC recovered and stable, total_evals inflated but seed-invariant, final best even with TPE — the residual
weakness is named precisely: Hyperband's quality is capped by random sampling. It has the perfect skeleton and
the wrong sampler. The final rung is then forced: keep this bracket schedule, but replace the random sampling
with the model from the TPE rung, so the multi-fidelity *allocation* and the model-guided *selection* compound
instead of being traded off.
