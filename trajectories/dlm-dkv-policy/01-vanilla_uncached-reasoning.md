The cache policy is the whole point, but it sits on top of a generation procedure, and with no caching
that procedure is the floor — so the thing to pin down first is just running LLaDA's denoising rollout
correctly, at full cost, and understanding exactly why the obvious speedup is illegal. Everything I build
later is a controlled relaxation of this floor, so I want the floor to be unambiguous: a full
bidirectional forward over the entire sequence at every step, reusing nothing. Let me derive why that is
the *correct* reference and not merely the lazy one, because the reason is precisely what makes caching
hard.

Start from what the model is. LLaDA is a masked-diffusion language model: a bidirectional transformer
trained to predict masked tokens, defining a generative distribution through a forward token-masking
process and a learned reverse unmasking process. The forward process masks each token independently with
probability `t` as `t` runs from 0 (clean) to 1 (all-masked); the reverse process, run from a fully
masked response, repeatedly predicts the clean token at every masked position and commits some of them.
The two facts that decide the cache story both come straight out of this construction. First, the
predictor is **time-independent**: a masked position's clean value depends only on the unmasked context
around it, and that context is literally clean data, so the network needs no timestep input and is just
a plain bidirectional transformer with no causal mask. Second, and this is the load-bearing one, the
attention is **bidirectional** — every position attends to every other position, masked ones included,
which is exactly what lets the model use right-context and decode in any order.

Now ask the question the whole task turns on: can I bolt on the autoregressive KV cache? In a causal
transformer the cache is free, and I should say precisely why, because the reason is what fails here. In
causal attention token `i` attends only to tokens `≤ i`, so its key and value depend only on already-fixed
tokens; once position `i` is generated, `K_i` and `V_i` never change again, and generation appends new
tokens at the end. Across decoding steps the prefix is literally invariant, so I cache it once and reuse
it with zero error. Both pillars — fixed states and fixed append order — are false for a diffusion LM.
Because attention is bidirectional, a token's key and value are a function of the *whole* sequence, so the
moment I unmask a new token anywhere, the context of every other token changes, and the keys and values
of tokens I committed earlier shift too. And there is no append order: the transfer rule commits whichever
masked positions are most confident, anywhere in the response, so I cannot even pre-decide whose states to
refresh. The exact prefix cache is not inconvenient here; it is mathematically inapplicable.

Let me make that concrete with the smallest example that has any bidirectional structure at all, because I
want to *see* the failure, not just assert it. Take a four-position sequence `[x0, x1, [MASK], [MASK]]`,
one attention head. In the causal world, the attention row for position 1 is a distribution over `{0,1}`
and depends on `x0,x1` only; if I later fill positions 2 and 3, position 1's row is unchanged by
construction — the mask forbids it from looking right — so its key, value, attention read-out, and every
downstream feature are frozen, and the cache for position 1 is exact forever. Now drop the causal mask, as
LLaDA does. Position 1's attention row is now a distribution over `{0,1,2,3}`, and the value it reads out is
`Σ_j a_{1j} V_j`. While positions 2 and 3 are `[MASK]`, `V_2` and `V_3` are the value projections of the
mask *embedding* — a generic, near-constant vector — and `a_{1,2}, a_{1,3}` are whatever attention the model
puts on unresolved slots. The instant position 2 gets committed to a real token, `V_2` swings from the mask
embedding to that token's value, and `a_{1,2}` re-weights, so position 1's read-out `Σ_j a_{1j} V_j` moves,
which shifts position 1's hidden state, which shifts *its* own key and value at the next layer, and so on up
the stack. So a token I may have already committed at position 1 has a representation that keeps changing
every time a neighbor resolves. That is the whole wall in one line: revealing anyone perturbs everyone, and
the perturbation propagates through depth. The degenerate check falls out for free — restore the causal
mask and every `a_{1,j>1}` is forced to zero, the read-out loses all dependence on the right context, and
the argument collapses back into the exact append-only AR cache. So the failure is *caused by* bidirectionality,
and nothing weaker than that is responsible.

There *is* a genuine redundancy hiding in here, and I should name it now because it is the seed of every
later rung, but I also have to be honest that the floor captures none of it. The predictor is time-independent
and a revealed token never un-reveals, so across a whole rollout the network is fed a sequence that changes
only a little each step. Count it: on MATH the schedule is `gen_length=256`, `num_steps=256`, so the rollout
commits on the order of one token per step, and the global input to the network is edited in one place at a
time. There are therefore only about `num_steps` distinct global inputs over the entire rollout, and at any
single position the *local* conditioning — the thing that actually controls its feature — changes far fewer
times than that, because only edits inside its attention neighborhood matter to it. If the decoding front is
spatially local at a scale of order ten positions, a given position's neighborhood is genuinely churning for
only maybe ten to twenty of the 256 steps and is quiescent the rest; that means something like ninety percent
of the per-position, per-step feature recomputations reproduce a value that was already computed. That ninety
percent is the entire prize the ladder is chasing. But it is licensed *only* where a position's conditioning
state has not changed, and I just showed that in a bidirectional model I cannot certify that cheaply — any
distant commit can, in principle, reach in through attention and move a feature I thought was settled. A stale
cached value can therefore be silently wrong, and because committed tokens are frozen for the rest of the
rollout, a wrong commit caused by a stale feature is unrecoverable. The redundancy is real; the license to
exploit it is what I do not yet have.

So before I settle on the floor let me actually walk the tempting shortcuts, because "cache nothing" should
be a conclusion I am forced into, not a reflex. Four things are on the table. The first is the exact AR
prefix cache; I have already killed it — no causal prefix, no append order — so it is out on correctness, not
economy. The second is the seductive one: cache the *prompt* only. The prompt tokens are never masked and
never change their token ids, so surely their keys and values are frozen and I can compute them once and reuse
them for all 256 steps, which on a long prompt is most of the sequence. Walk it two steps. A prompt position's
key at layer one is a function of its own embedding, which is fixed, so the layer-one key really is frozen —
good. But its layer-one *value read-out* is `Σ_j a V_j` over the whole sequence including the response region,
and as the response fills in, those `V_j` and the attention onto them change, so the prompt position's hidden
state at layer two moves, and from layer two upward its key and value are no longer frozen. The drift is
second-order — the prompt is a large, coherent block of real tokens whose self-attention dominates, so the
response perturbs it only weakly — but "weak" is not "zero," and I have no cheap certificate for how weak, and
on a gated metric a slow accumulation of prompt-feature error is exactly the kind of thing that could nudge a
borderline answer under threshold. So prompt-only caching is not *wrong* the way the AR cache is wrong; it is
*unlicensed at the floor*, where I have decided to buy correctness outright. I note it and defer it — it is
clearly the first door to open once I am willing to trade. The third option is a segment-and-clock cache:
split into prompt and response, refresh each on its own interval. Same objection, sharper: a segment is not a
unit of KV dynamics, so a single interval per segment must simultaneously refresh tokens that already settled
(wasted compute) and reuse tokens that are actively moving (lost quality), and again I have no floor-level
certificate. The fourth option is the exact rollout, and it is the only one of the four that carries a
correctness guarantee I can actually state. At the floor I want that guarantee, so the floor is the exact
rollout, and the other three become the design space the ladder walks into once it starts paying quality for
speed.

So let me write the rollout the floor actually runs, in the scaffold's vocabulary, because the policy I
edit is a *plan* the fixed harness executes, not the generation code itself. The harness already lays the
response region as mask tokens after the prompt, runs the bidirectional forward, owns the per-layer cache
buffers, and commits tokens; my `DLMRefreshPolicy` only hands it dicts of decisions. So "uncached control"
becomes a specific set of dict values, and each one is a deliberate refusal of a shortcut. The block
schedule takes the workload's own `gen_length`, `block_length`, and `num_steps` unchanged — for MATH-500
that is 256 generated tokens in blocks of 32 over 256 steps — and opens no warm forward, because there is
no cache to warm. The query plan returns `query_scope = "full_sequence"`: every step forwards every
position, because with no cache there is no subset I am entitled to skip. The cache-refresh plan is the
heart of the refusal — `use_feature_cache = False`, both the prompt and generation refresh intervals set
to 1 (recompute every step), `row_selector = "none"`, `kv_update = "full_refresh"`, no layer reset. No
attention probes are requested (`need_attention_weights = False`), because the floor needs no importance
or drift signal to decide what to reuse — it reuses nothing. The token-transfer plan keeps the standard
LLaDA commit rule: `mode = "low_confidence"` over the current block, transferring the harness-provided
default count per step, forcing at least one. And `after_step` returns the cache state untouched, because
there is no rollout state to carry between steps when nothing is cached. Every one of these dict values is
the *most expensive* legal choice, which is exactly what makes this the floor.

Two of those choices deserve a second look, because they are places I could have "helpfully" deviated and I
want to be sure I did not. The first is the block schedule. LLaDA's native sampler is semi-autoregressive: it
partitions the `gen_length` response into blocks of `block_length=32` and decodes them left to right, so MATH
runs 8 blocks, HumanEval 16 blocks, ARC 2 blocks, with `num_steps` steps spread across them. It would be
tempting to collapse the blocks — decode the whole response as one undifferentiated diffusion region — since
the floor already forwards the full sequence anyway and blocking looks like an autoregressive vestige. I keep
the native blocking untouched, and the reason is that the block structure is part of the *quality* reference,
not the cache policy: LLaDA uses it to keep the confidence-based transfer from committing far-flung tokens
prematurely (a block bounds where the sampler is allowed to commit this step), and if I quietly changed it the
`final_score` I record would no longer be the model's native accuracy, so it would be useless as the ceiling
every later rung is measured against. The second is the transfer rule. `mode = "low_confidence"` with
`scope = "current_block"` and `force_one = True` is the exact commitment rule: predict all masked positions,
keep the most confident within the active block, re-mask the rest, and guarantee at least one commit per step
so the rollout cannot stall. I decline every alternative here — a confidence *threshold*, a different scope,
a larger per-step count — for the same reason: those are policy levers that change what gets committed and
therefore change the accuracy, and at the floor I am buying the native accuracy exactly. So the floor is not
just "cache nothing"; it is "cache nothing *and* decode exactly as the model was sampled," which is what lets
its three recorded numbers stand as clean references.

Let me account for what that floor actually costs, so the "slowest of the field" claim has arithmetic under
it. Per denoising step the harness runs one full forward over `N = prompt + gen_length` positions through the
whole 32-layer stack; the cost per step scales like `L · (N² d_k · heads + N · d² )` — the quadratic
attention term plus the feed-forward term — and there are `num_steps` such steps. For MATH that is 256 full
forwards over a few hundred positions; for HumanEval `gen_length=512` and `num_steps=512`, so both `N` and the
step count are larger, and the total work grows super-linearly; for ARC `gen_length=64`, `num_steps=64`, a
much smaller `N` and far fewer steps. Nothing in the floor amortizes any of this: the redundancy I counted
above — the ninety percent of recomputations that reproduce a known value — is paid in full every step. Put a
number on the waste itself. Over a MATH rollout the floor performs `256 × 256 = 65,536` position-forwards in
the generated region alone (steps times generated positions), and of those only 256 are the "first time this
position's neighborhood actually changed" forwards that a perfect oracle cache would keep — the other ~99.6%
are, in principle, recomputing a feature whose conditioning did not move. That is a wildly loose bound (an
oracle is not achievable, because I cannot certify non-change cheaply), but even a policy that captured a
quarter of it would swing `reuse_ratio` from 0 to well above 0.7, which tells me the headroom is enormous and
the floor sits at the far bad end of it. HumanEval's `512 × 512 = 262,144` generated position-forwards make the
absolute waste four times larger again, and ARC's `64 × 64 = 4,096` make it the smallest — which is the same
ordering as the throughput prediction, from the other side of the ledger. So the
throughput ordering is forced by `N` and step count together: ARC should decode fastest in tokens per second
because its forward is smallest and it takes the fewest steps, and the two long workloads should be slowest,
with HumanEval slowest of all since it has both the longest sequence and the most steps. That is a prediction
I can check against the `tokens_per_s` column, and it does not depend on any measured number — it is pure
schedule arithmetic.

Let me also be sure each decline is *correct* and not merely conservative in the limit, because the temptation
is real. Consider the degenerate schedule `num_steps = 1`: the model commits every masked position in a single
forward, there is no second step, and the caching question is vacuous — there is nothing to reuse *from*. The
caching problem exists *only* because `num_steps > 1`, and its size grows with the ratio of steps to
tokens-committed-per-step, which is exactly the redundancy factor I estimated. This confirms the floor is not
leaving free money on the table through laziness: with one token per step over 256 steps the redundancy is
large, and the floor's refusal to touch it is a deliberate purchase of exactness, not an oversight. The
conservative, exactly-correct thing is therefore to run a full bidirectional forward over the entire current
sequence at every single denoising step, recomputing everything: no prefix KV cache (there is no causal
prefix, and committed tokens are re-attended to as new tokens reveal), no feature cache, no skipped positions.
This is the plain reference rollout, and it is the only policy on the whole ladder that I can claim is
*exactly* the model's intended behavior. Everything above it trades a bounded amount of that exactness for
compute saved.

One more thing about why this is the right *starting* point in this task specifically, given how the score
is built. The score is an efficiency score: each workload first applies the final task accuracy as a
near-lossless soft quality gate (`math ≥ 35`, `humaneval ≥ 40`, `lm-eval ≥ 84`), and then — once the gate is
satisfied — rewards cache reuse (the dominant term, weight 0.75) and decode throughput (weight 0.25,
normalized against the visible baseline envelope), geometric-mean across the three workloads. Read that
against the uncached control, term by term. The reuse term: by construction the control reuses *nothing*, so
`reuse_ratio` is 0 on every workload, and a term weighted 0.75 that is zero pins each workload's efficiency
factor low no matter what else happens. The throughput term: the control does the most work per step, so its
`tokens_per_s` is the bottom of the field, and normalized against a baseline envelope that includes faster
policies it contributes near zero too. The gate: the control passes comfortably, precisely because it is the
exact rollout and its accuracy is the model's native accuracy, so nothing is multiplied down for quality — but
passing the gate only *unlocks* the efficiency terms, and those are the two I just argued are near zero. Push
it through the geometric mean: if every workload's efficiency factor is near zero, the cube-root product is
near zero, and this is by construction the lowest score the ladder can produce. The floor is the weakest
policy not because it is wrong — it is the only exactly-correct one — but because the metric pays for reuse and
throughput, and the exact rollout deliberately spends both.

So at step 1 my edit is the trivial one: leave `DLMRefreshPolicy` at the scaffold default. Full-sequence
queries, feature cache off, refresh-everything-every-step, no probes, low-confidence transfer, identity
`after_step` (the full policy is in the answer). The reason to run it at all is that it establishes three
references every later rung is measured against. It establishes the *quality ceiling*: since it is the exact
rollout, its `final_score` per workload is the native LLaDA accuracy, and any cache policy that drops below it
has paid quality for speed — the gap to this number is the quality cost I am trying to keep near zero, and the
gates sit some margin below it. It establishes the *throughput floor*: the slowest decode of the field, the
number the throughput term normalizes improvements against. And it establishes the *reuse floor* at exactly 0,
which means the efficiency headroom on this task is the entire interval from 0 up to 1 — there is a lot to win,
because the floor wins none of it.

Here, then, is what I expect and what would falsify it. The shape should be identical across all three
workloads: `reuse_ratio` exactly 0 (there is no other value it can take with the feature cache off and full
refresh every step — a nonzero reuse here would mean the harness is caching something I did not ask it to, and
I would have to go read the substrate), `refresh_ratio` exactly 1, throughput at the bottom of whatever the
field will be, ordered ARC fastest and HumanEval slowest by the schedule arithmetic above. The three
`final_score`s should land at the native accuracy and clear their gates — I expect MATH and HumanEval to clear
by a comfortable margin and ARC, whose gate at 84 is set right up against strong native accuracy, to clear by
the least. If instead any workload comes in *under* its gate, that is not a caching failure at all — it would
mean the exact rollout itself does not reach the gate, and the entire ladder premise (that there is quality
headroom to trade) would be wrong; I do not expect that, but the MATH and lm-eval scores are where I would see
it first. Whatever the precise accuracies, the diagnosis is already pointed at the next step: I do not have a
quality problem, I have an *efficiency* problem that the floor refuses to address at all. The fix is not to
change what the model computes but to stop recomputing what has not changed — to open the first door I deferred,
turn the feature cache on, and begin reusing the state of the positions that are quasi-static across steps,
while keeping the exact rollout's accuracy. That is the first real cache policy, and the number it has to beat
is this one: reuse 0, throughput at the bottom, quality at the native ceiling.
