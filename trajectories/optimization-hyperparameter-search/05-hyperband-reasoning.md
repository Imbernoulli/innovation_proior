TPE landed exactly on the profile I predicted, and the contrast with DEHB is the whole point. `total_evals`
came back pinned to the budget — 50 on XGBoost, 40 on SVM and the NN — confirming the model-only rung does no
triage. The NN final best *recovered*: −3048.1 mean, with seed 42 at −3063.6 instead of DEHB's −3086, so
removing the low-fidelity mis-promotion fixed the slip exactly as the diagnosis said. But the convergence AUC
distinctly *lost* to DEHB: SVM 0.8763 (DEHB 0.9813), NN 0.8306 (DEHB 0.9351), XGBoost 0.9329 (DEHB 0.9492).
The cause is structural and visible in the numbers: TPE spends 10 of its 40–50 evaluations on a random
warm-up before the model turns on, and every evaluation thereafter is full price, so the best-so-far curve
cannot climb early the way cheap multi-fidelity triage made DEHB's climb. So the two rungs are
complementary, and the lesson is sharp: a good *model* fixes final quality and reliability; cheap
*multi-fidelity* fixes anytime convergence; neither has both. DEHB had the multi-fidelity half but bolted a
model-free DE learner onto it and inherited the low-fidelity-correlation risk. The question this rung
isolates is the multi-fidelity skeleton *on its own*, done right — without DE, without a model — so I can see
what principled resource allocation buys by itself and have a clean substrate to graft a model onto next.

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
ordering converges to the terminal one without ever needing to know γ. But SH has one input it cannot set for
itself: N. For a fixed budget, many configs each run cheaply (large N) is right *only if* the cheap resource
ranks configs like the expensive one; if the cheap fidelity is misleading, large N discards at low resource
exactly the config that would have won at full. That is the configurations-versus-fidelity dilemma, and it is
the *same* failure I watched bite DEHB's NN (−3086): aggressive low-fidelity triage where the cheap rank was
wrong.

Hyperband refuses to guess N and *hedges* across the whole spectrum: run several SH instances ("brackets"),
each starting at a different number-of-configs-versus-initial-fidelity tradeoff. The most aggressive bracket
throws a flood of configs at the cheapest fidelity (great if cheap predicts expensive); the least aggressive
bracket is essentially random search at full fidelity (the safe fallback when cheap is misleading). By
spanning the brackets, Hyperband covers the entire dilemma and is at most about (number of brackets) times
slower than the best fixed choice — a guarantee no single SH setting has. The reason this is the right shape
of hedge, and not just "try several settings," is that the brackets are *complementary*: each one is the
optimal allocation under a different assumption about how predictive the cheap fidelity is, and since I do not
know which assumption holds, paying a logarithmic-factor overhead to run all of them buys robustness to being
wrong about the one thing SH most needs to assume. On a benchmark where the cheap fidelity is faithful, the
aggressive bracket carries the run and the safe bracket is cheap insurance; on a benchmark where it lies, the
safe bracket protects the eventual winner that the aggressive bracket would have killed. That is exactly the
DEHB-NN failure mode (−3086) turned into a hedge instead of a gamble. Critically, every config is still
sampled *uniformly at random*: Hyperband is purely an allocation method, it never uses one evaluation's
outcome to decide where to look next. That is exactly its limitation and exactly why it is the right rung
here — it measures the multi-fidelity skeleton in isolation, with random sampling, so the gap between it and
the next rung will be attributable purely to adding a model.

Now the implementation the scaffold fills in, derived against the harness. With η = 3, compute `s_max =
min(4, floor(log_η(total_budget)))` — the cap at 4 keeps the number of brackets bounded so a small budget
does not spawn more brackets than it can fund. For each bracket s from s_max down to 0, the number of configs
is `n = min(ceil((s_max+1)/(s+1))·η^s, total_budget)` and the initial fidelity is `r = max(1/η^s, 0.1)` —
the 0.1 floor is the loop's clip, so the cheapest rung is fidelity 0.1, not arbitrarily small. Each bracket
samples its n configs uniformly and queues them all at fidelity r. The `suggest` loop is queue-driven: it
pops the next queued (config, fidelity), and when a trial returns it matches it back to its bracket by exact
`config == last.config` and `|fidelity − last.budget| < 0.05`, fills in the score, and when a bracket's
current rung is fully scored, *advances* it — sort by score, keep the top `len//η`, multiply the fidelity by
η (capped at 1.0), and re-queue the survivors at the higher fidelity. When the queue empties, fall back to a
random full-fidelity draw. The exact-equality match is a real constraint of this harness: it works because
the configs are reproducibly generated and clipped, so the returned config compares equal to the queued one;
a method that perturbed configs would break this matching, which is why Hyperband (random configs, never
perturbed) fits the queue model cleanly where DEHB needed `np.allclose` on encoded vectors. The distilled
module is in the answer.

Where should this land against TPE and DEHB? Hyperband should recover the *anytime* strength TPE lost,
because the aggressive brackets surface a decent config from the cheap fidelity fast — so I expect its
convergence AUC to beat TPE's clearly (TPE NN 0.8306, SVM 0.8763) and to be competitive with DEHB's. Its
`total_evals` should climb well past the budget like DEHB's, but more predictably — the bracket schedule is
deterministic given the budget, so I expect a fixed `total_evals` per benchmark across seeds (around 105 on
XGBoost, 95 on SVM and NN), unlike DEHB's seed-to-seed swing (65 to 354 on SVM) that came from DE's
data-dependent promotion. That predictability is itself a virtue. On final best, Hyperband should match the
competitive band but I expect it to *not* clearly beat TPE, because it samples randomly — it allocates budget
brilliantly but never aims, so against TPE's model-aimed configs it should be roughly even on final quality
and win only on convergence. And the NN remains the risk benchmark: Hyperband triages at low fidelity too, so
if the NN's cheap rank is misleading, an aggressive bracket can still kill the eventual winner — I expect the
NN final best around DEHB/random-search level, not a clear improvement.

So the falsifiable expectations against the prior numbers: convergence AUC should rise back above TPE's on
every benchmark (especially NN, where TPE's 0.8306 was weak) and sit near DEHB's, because the multi-fidelity
hedge front-loads the curve; `total_evals` should be high and *stable across seeds* (the deterministic
bracket tell, distinguishing it from DEHB's variance); and final best should track the competitive band
without clearly beating TPE, because random sampling allocates but does not aim. If I see exactly that — AUC
recovered and stable, total_evals fixed and inflated, final best even with TPE — the residual weakness is
named precisely: Hyperband's quality is capped by random sampling. It has the perfect skeleton and the wrong
sampler. The final rung is forced: keep this bracket schedule, but replace the random sampling with the model
from the TPE rung, so the multi-fidelity *allocation* and the model-guided *selection* compound instead of
being traded off.
