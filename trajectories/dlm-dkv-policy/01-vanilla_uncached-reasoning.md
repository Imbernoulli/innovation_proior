The cache policy is the whole point, but it sits on top of a generation procedure, and with no caching
that procedure is the floor — so the first thing to pin down is running LLaDA's denoising rollout
correctly, at full cost, and understanding exactly why the obvious speedup is illegal. Everything I build
later is a controlled relaxation of this floor, and the reason the floor is the *correct* reference and not
merely the lazy one is precisely what makes caching hard.

Start from what the model is. LLaDA is a masked-diffusion language model: a bidirectional transformer
trained to predict masked tokens, defining a generative distribution through a forward token-masking
process and a learned reverse unmasking process. The forward process masks each token independently with
probability `t` as `t` runs from 0 (clean) to 1 (all-masked); the reverse process, run from a fully masked
response, repeatedly predicts the clean token at every masked position and commits some. Two facts decide
the cache story. First, the predictor is **time-independent**: a masked position's clean value depends
only on the unmasked context around it, and that context is clean data, so the network needs no timestep
input and is a plain bidirectional transformer. Second, and load-bearing: the attention is
**bidirectional** — every position attends to every other, masked ones included, which is what lets the
model use right-context and decode in any order.

Now the question the task turns on: can I bolt on the autoregressive KV cache? In causal attention token
`i` attends only to tokens `≤ i`, so its key and value depend only on already-fixed tokens; once position
`i` is generated, `K_i` and `V_i` never change, and generation appends at the end. Across steps the prefix
is invariant, so I cache it once and reuse with zero error. Both pillars — fixed states, fixed append
order — are false for a diffusion LM. Because attention is bidirectional, a token's key and value are a
function of the *whole* sequence, so the moment I unmask a new token anywhere, the context of every other
token changes and the KV of tokens I committed earlier shift. And there is no append order: the transfer
rule commits whichever masked positions are most confident, anywhere in the response, so I cannot even
pre-decide whose states to refresh. The exact prefix cache is not inconvenient here; it is mathematically
inapplicable.

The smallest example with any bidirectional structure makes the failure concrete. Take `[x0, x1, [MASK],
[MASK]]`, one head. Causal: position 1's attention row is a distribution over `{0,1}`; fill positions 2 and
3 later and the row is unchanged — the mask forbids looking right — so its key, value, read-out, and every
downstream feature are frozen and the cache is exact forever. Drop the causal mask, as LLaDA does. Position
1's row is now over `{0,1,2,3}` and it reads out `Σ_j a_{1j} V_j`. While 2 and 3 are `[MASK]`, `V_2, V_3`
are value projections of the mask *embedding* — a generic near-constant vector. The instant position 2
commits to a real token, `V_2` swings and `a_{1,2}` re-weights, so position 1's read-out moves, which
shifts its hidden state, which shifts its own key and value at the next layer, and so on up the stack. A
token I may already have committed at position 1 has a representation that keeps changing every time a
neighbor resolves. That is the wall: revealing anyone perturbs everyone, and the perturbation propagates
through depth — caused by bidirectionality itself, since restoring the causal mask zeroes every `a_{1,j>1}`
and collapses the argument back to the exact append-only cache.

There is a genuine redundancy hiding in here, and it is the seed of every later policy. The predictor is
time-independent and a revealed token never un-reveals, so across a rollout the network is fed a sequence
that changes only a little each step. On MATH the schedule is `gen_length=256`, `num_steps=256`, so the
rollout commits on the order of one token per step: there are only about `num_steps` distinct global
inputs, and at any single position the *local* conditioning changes far fewer times than that, because only
edits inside its attention neighborhood matter. If the decoding front is spatially local at a scale of
order ten positions, a given position's neighborhood churns for maybe ten to twenty of the 256 steps and is
quiescent the rest — so something like ninety percent of the per-position, per-step feature recomputations
reproduce a value already computed. That ninety percent is the entire prize the later policies chase. But
it is licensed *only* where a position's conditioning has not changed, and in a bidirectional model I
cannot certify that cheaply — any distant commit can reach in through attention and move a feature I thought
settled. A stale cached value can be silently wrong, and because committed tokens are frozen for the rest
of the rollout, a wrong commit from a stale feature is unrecoverable. The redundancy is real; the license
to exploit it is what I do not yet have.

So the tempting shortcuts should each be a conclusion I am forced past, not a reflex. The exact AR prefix
cache is already dead on correctness. The seductive one is caching the *prompt* only: prompt tokens are
never masked and never change their ids, so surely their KV is frozen. Walk two layers. A prompt position's
layer-one key is a function of its own fixed embedding — frozen, good. But its layer-one value read-out is
`Σ_j a V_j` over the whole sequence including the response, and as the response fills those `V_j` and the
attention onto them change, so the prompt position's layer-two hidden state moves, and from there up its
key and value are no longer frozen. The drift is second-order — the prompt is a large coherent block whose
self-attention dominates, so the response perturbs it only weakly — but "weak" is not "zero," I have no
cheap certificate for how weak, and on a gated metric a slow accumulation of prompt-feature error is
exactly what could nudge a borderline answer under threshold. So prompt-only caching is not *wrong* the way
the AR cache is; it is *unlicensed at the floor*, where I am buying correctness outright. I note it and
defer it — clearly the first door to open once I am willing to trade. A segment-and-clock cache (split
prompt/response, refresh each on its own interval) has the same objection sharper: a segment is not a unit
of KV dynamics, so one interval must simultaneously refresh settled tokens and reuse moving ones. Only the
exact rollout carries a correctness guarantee I can state, so at the floor that is what I run.

Now write the rollout in the harness's vocabulary, because the policy I edit is a *plan* the fixed harness
executes, not the generation code. The harness lays the response as mask tokens, runs the bidirectional
forward, owns the per-layer cache buffers, and commits tokens; `DLMRefreshPolicy` hands it dicts. The block
schedule takes the workload's own `gen_length`, `block_length`, `num_steps` unchanged and opens no warm
forward (no cache to warm). The query plan returns `query_scope = "full_sequence"` — no subset I am
entitled to skip. The cache-refresh plan is the heart of the refusal: `use_feature_cache = False`, both
refresh intervals 1, `row_selector = "none"`, `kv_update = "full_refresh"`, no layer reset. No attention
probes (`need_attention_weights = False`) — the floor needs no importance or drift signal because it reuses
nothing. Transfer keeps the standard LLaDA commit: `mode = "low_confidence"` over the current block, the
harness default count, force at least one. `after_step` returns the cache state untouched. Every dict value
is the most expensive legal choice.

Two of these I keep native deliberately, because they are quality-reference decisions, not cache decisions.
The block schedule stays semi-autoregressive — LLaDA partitions `gen_length` into `block_length=32` blocks
decoded left to right (8 blocks on MATH, 16 on HumanEval, 2 on ARC) — because the block structure bounds
where confidence-based transfer may commit, and quietly collapsing it would make the recorded `final_score`
no longer the model's native accuracy, useless as the ceiling later policies are measured against. Likewise
the transfer rule stays exactly `low_confidence / current_block / force_one`: a threshold, a different
scope, or a larger count are policy levers that change what gets committed and therefore the accuracy, and
at the floor I am buying native accuracy exactly. So the floor is "cache nothing *and* decode exactly as
the model was sampled."

That floor's cost has arithmetic under it. Per step the harness runs one full forward over `N = prompt +
gen_length` through the 32-layer stack, cost scaling like `L·(N² d_k·heads + N d²)`, over `num_steps`
steps. Over a MATH rollout the generated region alone sees `256 × 256 = 65,536` position-forwards, of which
only 256 are the "first time this position's neighborhood changed" forwards an oracle cache would keep — the
other ~99.6% recompute a feature whose conditioning did not move. That is a wildly loose bound, but even
capturing a quarter of it would swing `reuse_ratio` from 0 well above 0.7, so the headroom is enormous and
the floor sits at the far bad end. HumanEval's `512 × 512` makes the waste four times larger, ARC's `64 ×
64` the smallest — the same ordering as throughput, so ARC should decode fastest in tokens/s and HumanEval
slowest, pure schedule arithmetic independent of any measured number.

The score confirms why the floor is the weakest policy. Each workload applies accuracy as a near-lossless
soft gate, then rewards reuse (weight 0.75) and throughput (0.25), geometric-mean across the three. The
control reuses *nothing*, so the dominant term is 0 everywhere; it does the most work per step, so
throughput is the field's slowest and that term is near 0 too; the gate passes comfortably because the
rollout is exact — but passing only unlocks the two efficiency terms I just argued are near zero, and a
geometric mean of near-zero factors is the lowest score any policy here can produce. Weakest not because it
is wrong — it is the only exactly-correct policy — but because the metric pays for reuse and throughput and
the exact rollout spends both.

So the step-1 edit is trivial: leave `DLMRefreshPolicy` at the default (the full policy is in the answer).
Its value is three references. The *quality ceiling*: its per-workload `final_score` is native LLaDA
accuracy, and any policy below it has paid quality for speed. The *throughput floor*: the slowest decode,
what the throughput term normalizes against. And the *reuse floor* at exactly 0, so the efficiency headroom
is the whole interval up to 1. The shape should be identical across workloads — reuse exactly 0, refresh 1,
throughput at the bottom, ordered ARC fastest and HumanEval slowest — with the three accuracies clearing
their gates, ARC by the least since its gate at 84 sits right against native accuracy. If any workload came
in *under* its gate the ladder premise (that there is quality to trade) would be wrong, and MATH or lm-eval
is where I would see it first. The diagnosis is already pointed at step 2: not a quality problem but an
efficiency one the floor refuses to address — turn the feature cache on and start reusing the positions
that are quasi-static across steps, keeping the exact accuracy.
