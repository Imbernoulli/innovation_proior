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
refresh. The exact prefix cache is not inconvenient here; it is mathematically inapplicable. That is the
wall the entire ladder above this floor is built to climb around, and the honest floor is the one that
does not pretend the wall isn't there.

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
there is no rollout state to carry between steps when nothing is cached.

Let me make sure each of these declines is *correct*, not just conservative, because the temptation to
cache is real and the diffusion structure even hints that caching is possible. The hint is exactly the
two facts above: the predictor is time-independent, and a revealed token never changes, so across a whole
rollout there are at most `L` distinct conditioning states the network ever sees — most steps only flip a
handful of masked positions and leave the rest of the input identical to the previous step, and the
network's output on an unchanged input is identical. That is a genuine redundancy, and it is what every
later rung will exploit. But it is licensed *only* when the conditioning state of a position has not
changed, and in a bidirectional model the moment any masked token reveals, it changes the context for
*every other* position — there is no causal prefix whose keys and values are provably frozen. An early
position's correct prediction genuinely depends on tokens that get revealed later, so a stale cached value
can be silently wrong, and committed tokens are frozen for the rest of the rollout, which means a wrong
commit caused by a stale feature is unrecoverable. The conservative, exactly-correct thing is therefore to
run a full bidirectional forward over the entire current sequence at every single denoising step,
recomputing everything: no prefix KV cache (there is no causal prefix, and committed tokens are
re-attended to as new tokens reveal), no feature cache, no skipped positions. This is the plain reference
rollout, and it is the only policy on the whole ladder that I can claim is *exactly* the model's intended
behavior. Everything above it trades a bounded amount of that exactness for compute saved.

One more thing about why this is the right *starting* point in this task specifically, given how the score
is built. The score is an efficiency score: each workload first applies the final task accuracy as a
near-lossless soft quality gate, and then — once the gate is satisfied — rewards cache reuse (the dominant
term, weight 0.75) and decode throughput (weight 0.25, normalized against the visible baseline envelope),
geometric-mean across the three workloads. Read that against the uncached control. By construction the
control reuses *nothing*: `reuse_ratio` is 0 on every workload, so the dominant efficiency term is 0 on
every workload. Its throughput is whatever a no-cache forward gives — the slowest of the field, since it
does the most work per step — so the throughput term is near 0 too. The quality gate it passes
comfortably, precisely because it is the exact rollout and its accuracy is the model's native accuracy. So
the floor's profile is sharp and predictable: it should *pass quality everywhere and earn essentially zero
efficiency credit everywhere*, which under a geometric mean of near-zero per-workload values is the lowest
score on the ladder. This is the weakest policy not because it is wrong — it is the only exactly-correct
one — but because the metric pays for reuse and throughput, and the exact rollout deliberately spends both.

So at step 1 my edit is the trivial one: leave `DLMRefreshPolicy` at the scaffold default. Full-sequence
queries, feature cache off, refresh-everything-every-step, no probes, low-confidence transfer, identity
`after_step` (the full policy is in the answer). Now let me reason about what this floor must do, because
that is the entire reason to run it. It establishes three things I will need as references for every later
rung. It establishes the *quality ceiling*: since it is the exact rollout, its `final_score` per workload
is the native LLaDA accuracy, and any cache policy that drops below it has paid quality for speed — the gap
to this number is the quality cost I am trying to keep near zero. It establishes the *throughput floor*:
the slowest decode of the field, the number the throughput term normalizes improvements against. And it
establishes the *reuse floor* at exactly 0, which means the efficiency headroom on this task is the entire
interval from 0 up to 1 — there is a lot to win, because the floor wins none of it.

I expect the three workloads to differ mainly in absolute accuracy and in how much sequence there is to
recompute, not in the *shape* of the result, which should be the same everywhere: quality gate passed,
reuse zero, throughput at the bottom. MATH-500 and HumanEval are longer generations (256 and 512 tokens),
so the per-step full forward is heaviest there and the throughput should be lowest; ARC-Challenge is short
(64 tokens), so its full forward is cheapest and its throughput highest of the three, but still
cache-free. Whatever the precise accuracies land at, the diagnosis is already pointed at the next step: I do
not have a quality problem, I have an *efficiency* problem that the floor refuses to address at all. The
fix is not to change what the model computes but to stop recomputing what has not changed — to turn the
feature cache on and begin reusing the state of the positions that are quasi-static across steps, while
keeping the exact rollout's accuracy. That is the first real cache policy, and the number it has to beat is
this one: reuse 0, throughput at the bottom, quality at the native ceiling.
