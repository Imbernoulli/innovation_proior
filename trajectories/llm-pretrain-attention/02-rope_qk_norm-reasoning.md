The QK-Norm run came back almost exactly where I expected, and the number that matters is the one it
*didn't* move. Validation loss landed at 2.2885, WikiText-2 43.65, LAMBADA 69.99, downstream ARC-Easy
55.64, HellaSwag 33.41, PIQA 63.17, WinoGrande 51.30. Convert it into a picture before reading it as
success or failure. Cross-entropy 2.2885 is an in-domain perplexity of `exp(2.2885) = 9.86` — on FineWeb the
model picks the next token as if from about ten options. But WikiText-2 at 43.65 and LAMBADA at 69.99 are
seven times that; some is domain shift, but LAMBADA being the *worst* of the three is the tell, because it
is built so the final word is only predictable from a long-range dependency to earlier in the passage — a
perplexity of 70 there says the model is barely resolving those links. The downstream numbers agree against
their chance baselines: ARC-Easy and HellaSwag are four-way (25%), PIQA and WinoGrande two-way (50%).
ARC-Easy 55.64 and PIQA 63.17 sit comfortably above chance, HellaSwag 33.41 is eight points over, and
WinoGrande 51.30 is *at* chance — and WinoGrande is pure pronoun resolution, binding a word to a referent
several tokens back, a purely relational dependency. The same failure LAMBADA reports, seen through another
lens: whatever is about the *relation between positions* rather than the content at a position, this model
is not doing.

That is the floor, for the reason I argued before running it: RMSNorm on q and k stabilizes the logit
*scale* and touches nothing about *position*. Order still arrives through the learned absolute `wpe` table,
so every dependency — a verb agreeing with a subject three tokens back, a closing quote matching an opening
one forty tokens back, the WinoGrande pronoun — is reconstructed out of absolute slot signals. The 2.2885 is
what "stability fixed, position still absolute" costs, and it points straight at the seam I deliberately
left alone.

Let me be precise about what is wrong with the position signal, because I want the *right* relative scheme,
not just any change. Self-attention is order-blind by construction: with q, k, v linear in the token
embeddings the whole computation is permutation-equivariant — shuffle the tokens and the outputs shuffle
identically. Order must be injected by hand, and the only quantity deciding which token attends to which is
the logit `q_m^T k_n`. The default injects position additively and absolutely — `x_i <- x_i + wpe[i]` before
the stack — so `q_m = W_q(x_m + p_m)`, `k_n = W_k(x_n + p_n)`, and expanding the logit gives four terms:
pure content `x_m^T W_q^T W_k x_n`, plus three each carrying *absolute* `p_m` or `p_n`. The logit depends on
*where m and n sit in the buffer*, not on the offset `m - n` — and the offset is what language relations are
about. The `wpe` cross term `p_m^T W_q^T W_k p_n` could in principle encode `m - n`, but I am forcing the
network to learn indirectly, through the `wpe` cross-products, something I could hand it directly and
exactly. That indirection is the bottleneck the 2.2885 pays for.

So I want the logit to depend on `x_m`, `x_n`, and `m - n` only. Two genuinely different routes. The first
patches the additive expansion — replace the absolute `p_n` with a learned relative table indexed by
`m - n`, or a distance-bucketed scalar bias — the whole relative-attention family. Every version ends with a
learned table or bias living *inside* the pairwise logit, meaning new parameters, a clip or bucket for far
distances, and a table the optimizer must fill: a per-offset vector in `head_dim` over a 1024-token context
is up to `(2·1024 - 1)·64 ≈ 131k` numbers before any sharing, and even the bucketed scalar-bias variant adds
a few hundred parameters per head per layer plus bucket boundaries as new hyperparameters — all on the
frozen budget, all something the optimizer learns from scratch that could interact with the drift I just
stabilized. The second route *demands* the relative property as a constraint and solves for the injection,
hoping the solution is parameter-free. Try that, since if it closes it is strictly cleaner. Require
`<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m - n)`, with the boundary `f(x, 0) = W x` so it reduces to
ordinary attention at position 0.

Solve in `d = 2`, identifying `R^2` with the complex plane and using `<a, b> = Re[a b*]`. Write `f` in polar
form. The magnitude equation against the boundary forces the magnitude position-independent — I take the
stable, norm-preserving branch, since I do not want position to amplify one side and shrink the other. The
phase equation forces the phase *arithmetic* in position: the same extra angle `m·theta` added to query and
key on top of each vector's own angle. The solution is a rotation, `f_q(x_m, m) = R(m theta) W_q x_m`, and
the inner product is `<R(m theta) a, R(n theta) b> = a^T R(m theta)^T R(n theta) b`. Since `R(α)^T = R(-α)`
and `R(-α)R(β) = R(β-α)`, this is `(W_q x_m)^T R((n-m)theta)(W_k x_n)` — the absolute indices `m, n` appear
*only* through the single rotation `R((n-m)theta)`. Exactly the demand, and I bolted nothing on: the
relative property fell out of the constraint, with no table and no learned parameter.

Put a number on it so the algebra is not fooling me. Take `d = 2`, `theta = 1`, raw query and key both the
unit vector `(1, 0)`. Query at `m = 5`, key at `n = 2`, offset 3: `f_q = R(5)·(1,0) = (cos5, sin5)`,
`f_k = (cos2, sin2)`, dot `= cos5·cos2 + sin5·sin2 = cos(5-2) = cos3 ≈ -0.99`. Slide both positions up by 95
— `m = 100`, `n = 97`, same offset — and the dot is `cos(100-97) = cos3 ≈ -0.99`, identical. The logit did
not move when I shifted both absolute positions equally; it tracks only the offset, the invariance the
absolute `wpe` could never guarantee.

Lift to the real head dimension by chopping it into `d/2` independent 2-planes and rotating each at its own
frequency. The inner product is a sum over planes, each relative-only by the 2D argument, and a sum of
relative-only is relative-only. Stack into block-diagonal `R_m`, the i-th 2×2 block a rotation by
`m·theta_i`; block-diagonal rotations still compose by adding angles, so `R_m^T R_n = R_{n-m}` and
`q_m^T k_n = x_m^T W_q^T R_{n-m} W_k x_n` — the offset sits in a single rotation between the content
projections, no learned table, no clip, no bias bucket. Reuse the geometric spectrum
`theta_i = 10000^{-2(i-1)/d}`. With `d = head_dim = 64` the exponent is `(i-1)/32`: plane 1 has
`theta_1 = 1`, a period of `2π ≈ 6.3` tokens — the fast plane resolving local offsets; plane 32 has period
about 47,000 tokens, a slow ramp carrying coarse near-absolute position; and the plane whose period equals
the context length sits at `i ≈ 18.7`. So the decay is *graded*, not a cliff: as `|m - n|` grows the
per-plane phases spread across frequencies and the summed positional contribution loses coherence — close
tokens see all 32 planes coherently, distant tokens a shrinking slow subset. That is the right shape for a
language prior: local dependencies get full positional resolution, long-range ones a coarser but non-zero
signal. And each `R` is orthogonal, so applying it can never blow up or collapse the representation as it
propagates through 24 layers.

Now the edit surface. Position is no longer additive, so I set `self.use_pos_emb = False`; the loop gates
the absolute table on exactly this flag — `getattr(self.transformer.h[0].attn, 'use_pos_emb', True)` in
`GPT.forward`, keyed on layer 0's attention — so flipping it skips the `wpe` add entirely. This is
mandatory here, not chosen: leaving `wpe` on while also rotating q and k would stack two position schemes,
an absolute-additive one and a relative-rotational one, and keep paying the absolute handicap I am removing.
And I realize the rotation elementwise rather than build the sparse block-diagonal matrix: precompute
`inv_freq = 1/(10000^{(arange(0, head_dim, 2)/head_dim)})` as a buffer, per forward form
`freqs = outer(arange(T), inv_freq)`, then `cos` and `sin`, and apply the *split-half* rotation —
`x1 = x[..., :d]`, `x2 = x[..., d:]` with `d = head_dim/2`, `y1 = x1·cos - x2·sin`, `y2 = x1·sin + x2·cos`,
concatenate `[y1, y2]`.

The split-half choice deserves a check, since the interleaved `(2i, 2i+1)` pairing is the other obvious
layout and mismatching them silently breaks the relative property. Split-half pairs coordinate `i` with
`i + d` as the two legs of one 2-plane; interleaved pairs `2i` with `2i+1`. Both partition the 64
coordinates into 32 planes rotated by matched frequencies, related by a fixed permutation of the head
dimension. Since `W_q, W_k` are learned, the model can absorb any fixed permutation into them at no cost, so
the two layouts are equally expressive. What is *not* free is consistency: `R_m^T R_n = R_{n-m}` is
per-plane and only holds when q and k share the pairing and frequency assignment — rotate q split-half and k
interleaved and the dot product sums mismatched planes, and the offset no longer factors out. So I pick one
layout, split-half, and apply it identically to q and k, never to v: v carries the content averaged into the
output, and position belongs in the logit.

Here is what makes this step 2 and not a fresh start: I keep the QK-Norm I just validated. The 2.2885 run
proved the score-stability fix is real and free — no parameters, a clean run — and it is *orthogonal* to
position: RMSNorm strips q/k magnitude, RoPE rotates q/k direction. So I stack them. The order looks like it
might matter, so check whether it does. RMSNorm is a pure rescale `N(x) = sqrt(d)·x/||x||`; RoPE is an
orthogonal rotation `R` with `||Rx|| = ||x||`, so `N(Rx) = sqrt(d)·Rx/||Rx|| = sqrt(d)·Rx/||x|| = R(Nx)` —
the two commute exactly, and the realized q, k are identical whichever order I write. I write
RMSNorm-then-RoPE, `q = _apply_rope(F.rms_norm(q, (q.size(-1),)), T)`, as the natural pipeline — normalize
direction, then place it in position-space — but nothing downstream depends on the choice. Everything else —
the fused SDPA path, the causal mask, the output projection — is as in step 1. So this step is "step 1 plus
relative position": two `F.rms_norm` and two rotations on the per-head q and k, `use_pos_emb` flipped to
False, the `wpe` table dropped. The full module is in the answer.

I should own the degrees of freedom the edit does not expose. RoPE here is the *frozen* sinusoidal schedule
at base 10000 — I am not learning the frequencies (they barely move from this initialization anyway) and not
tuning the base for the 1024-token context. That last is a real choice I should own: the spectrum spends its
slowest planes on periods of tens of thousands of tokens, far past the 1024 I ever see, so a smaller base
would pack more of the frequency ladder into the window I actually use and arguably give finer mid-range
resolution. I decline it — lowering the base is an untested hyperparameter search on a frozen budget, base
10000 is the well-validated default that transfers across scales, and retuning it here would fold a second,
confounded change into a step whose one job is to measure what relative position buys. And the QK-Norm half
is still the parameter-free RMSNorm from step 1, so the realized logit is `sqrt(d_k)·cos(angle)` modulated
by the relative rotation, sharpness pinned at `sqrt(d_k)` rather than tuned — the same omission I already
accepted.

Now the expectations against the 2.2885 floor. The thesis is that absolute-additive position is the
bottleneck the QK-Norm run was paying for and RoPE removes it, so this should clear 2.2885 by a
*representational* margin — several hundredths of a nat, not the thousandths a scaling tweak buys. The
perplexities should follow, with LAMBADA the most sensitive: its last-word prediction hinges on exactly the
long-range dependency a relative encoding is built to serve, so 69.99 should drop more in relative terms
than WikiText-2's 43.65, which leans on local structure absolute position already handled passably.
Downstream should rise with the LM gain, and if the mechanism is right WinoGrande — the near-chance pronoun
task — is where I would most hope to see movement, though I hold that loosely since a score at chance moves
for noisy reasons. The risk: the two changes are stacked, so if the combined run does *not* beat the floor,
the position story is wrong and stability was the real bottleneck — but I do not believe that, because the
four-term expansion is a mechanical handicap and the constraint-solve provably removes it. The subtler
question this sets up is that I am stacking RoPE *on top of* RMSNorm, and once position is relative the
RMSNorm half — its fixed `sqrt(d_k)` sharpness ceiling — may contribute little or even slightly cost me,
since RoPE is itself norm-preserving and already damps the position-direction drift QK-Norm was guarding.
If the combined run lands good but not clearly better than plain RoPE would, that is the tell to strip back
to plain RoPE and see whether the cosine ceiling was quietly costing the deliberate sharpening.
