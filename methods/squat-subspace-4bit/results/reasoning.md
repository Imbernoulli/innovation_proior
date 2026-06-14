Let me start from what actually goes wrong when I quantize the KV cache and the answers drift. The cache is huge — it grows with sequence length, batch, and depth, and for long generations it's the memory and the bandwidth bottleneck — so I want to store keys and values in two or four bits instead of sixteen. The tuning-free way everyone does this is to take a group of cached numbers, find their min and max, and round each onto a `b`-bit grid: `qtz(x) = round((x − m)/Δ)` with zero-point `m = min` and scale `Δ = (max − min)/(2^b − 1)`, dequantize as `deq(x̄) = x̄·Δ + m`. Cheap, streaming-friendly, no calibration. And the criterion everyone optimizes, implicitly, is reconstruction error: pick the grouping (per channel for keys, per token for values, as the outlier structure dictates) so that `‖k − deq(k^qtz)‖₂` is as small as possible, group by group. Keys per channel because a few channels carry persistent large magnitudes and per-channel grouping confines those outliers to their own group; values per token because they have no such fixed-channel pattern. Fine. So why does a long reasoning chain still fall apart at low bits?

Here is the thing that bugs me. The cached key `k_i` is never read by anything *as* `k_i`. It is only ever consumed inside attention, where the new token's query `q_n` dots it: the score is `q_n·k_iᵀ/√d`, those scores go through a softmax, and the softmax weights average the values. So the quantity I should care about is not how close `deq(k_i^qtz)` is to `k_i` in raw Euclidean distance — it's how close the *attention output* is to what it would be in full precision. Reconstruction error and attention error are not the same thing. Two quantized keys with the *identical* `‖k − deq‖₂` can disturb the next token's score by wildly different amounts: if the residual `r_i = k_i − deq(k_i^qtz)` happens to point in the direction of `q_n`, the score shifts by `q_n·r_iᵀ`, which can be large; if `r_i` is perpendicular to `q_n`, the score doesn't move at all. The compression criterion is completely blind to the *direction* of the residual, and direction is exactly what attention cares about. That's the wall the per-element criterion keeps hitting: it spends bits making `k` numerically faithful in every direction equally, when the model only needs it faithful in the directions the queries actually probe.

So let me try to make that intuition exact — bound the attention-output error in terms of the residuals, and see what it tells me to preserve. Write the pre-softmax score vector `w = [q_n·k_iᵀ/√d]_{i=1..n}` and `w^deq = [q_n·k_i^{deqᵀ}/√d]_i`, where `k_i^deq = deq(k_i^qtz)`. Stack the values into `V` (rows `v_i`) and `V^deq` (rows `v_i^deq`). The full-precision attention output is `softmax(w)·V`, the quantized one is `softmax(w^deq)·V^deq`. I want `‖softmax(w)V − softmax(w^deq)V^deq‖₂`. Add and subtract the cross term `softmax(w)V^deq`:

  `softmax(w)V − softmax(w^deq)V^deq = softmax(w)(V − V^deq) + (softmax(w) − softmax(w^deq))V^deq`,

and the triangle inequality splits it into a value-error piece and a score-error piece. The first piece: `softmax(w)` is a probability vector (nonnegative, sums to one), so `softmax(w)(V − V^deq)` is a convex combination of the rows `v_i − v_i^deq`, and its norm is at most `Σ_i ‖v_i − v_i^deq‖₂`. Good — that piece is governed entirely by value reconstruction error, and it says: quantize values to keep `‖v_i − deq(v_i^qtz)‖₂` small. That justifies the standard per-token value quantization; nothing exotic needed there.

The second piece is the interesting one. `(softmax(w) − softmax(w^deq))V^deq` is at most `‖softmax(w) − softmax(w^deq)‖₂ · ‖V^deq‖_F`. Now I need how much the softmax moves when its logits move. The softmax map is Lipschitz — its Jacobian is `diag(p) − ppᵀ` whose operator norm is bounded by `1/2` (the variance of a distribution on `[0,1]` indicator coordinates can't exceed `1/4`, and the bound `1/2` on the map is the clean constant) — so `‖softmax(w) − softmax(w^deq)‖₂ ≤ (1/2)‖w − w^deq‖₂`. And `‖w − w^deq‖₂` is, componentwise, `q_n·(k_i − k_i^deq)ᵀ/√d`, so

  `‖w − w^deq‖₂ = (1/√d)·sqrt( Σ_i (q_n·r_iᵀ)² ) ≤ (1/√d) Σ_i |q_n·r_iᵀ|`,

bounding the `ℓ₂` norm of the vector by its `ℓ₁` norm. Putting the pieces together,

  `‖Attention(orig) − Attention(quant)‖₂ ≤ [ ‖V^deq‖_F / (2√d) ]·Σ_i |q_n·(k_i − deq(k_i^qtz))ᵀ| + Σ_i ‖v_i − deq(v_i^qtz)‖₂.`

There it is in black and white, and it tells me precisely what to spend bits on. The output is preserved when two things hold: (i) the value residuals are small — handle with per-token value quantization, settled — and (ii) the *inner products* `q_n·(k_i − k_i^deq)ᵀ` are small, with the scale set by the size of the dequantized values. So the key residual `r_i` should be as orthogonal as possible to the future query `q_n`. Not `‖r_i‖` small. The projection of `r_i` onto the query. That's a different objective from every compression-based method, and it's the lever.

But I immediately hit the obvious problem: at the moment I quantize token `i`'s key, the future queries `q_n` for `n > i` don't exist yet — they're produced by tokens I haven't generated. I can't make `r_i` orthogonal to a vector I don't have. So either this whole idea is dead on arrival, or there's some structure in the queries I can exploit ahead of time. Let me poke at the queries and see whether they're predictable.

The trained queries give me a way out: they don't fill the `d`-dimensional space — they cluster in a low-dimensional subspace. If I measure how far normalized queries sit from the span of the top singular directions, the deviation drops off fast: a few-dozen-dimensional span already captures a whole sequence's queries to within about twenty percent, way below `d`. That's a fact about how the trained model uses its queries, and it changes the game. I don't need the *individual* future query `q_n`; I need the *subspace the future queries will live in*. And the next observation makes it actionable: the subspace spanned by the *prompt's* queries — which I have, right after prefill, before I decode a single token — essentially coincides with the subspace spanned by prompt-plus-response queries. The queries the model will generate stay in the same neighborhood the prompt's queries already mapped out. (It's robust enough that even a different sequence from the same task spans nearly the same subspace; only crossing to an unrelated dataset degrades it.) So I *can* estimate the relevant query directions before decoding: do SVD on the prompt queries once, keep the top `r` directions, and treat that as a stand-in for "the directions future queries will probe."

Let me write that subspace concretely. After prefill I have `Q = X_0 W^Q` for the prompt. Take its SVD, `Q = U Σ Vᵀ`. The right singular vectors `v_1,…,v_r` (rows of `V`) are the directions in `d`-space that the queries actually use, ordered by how much energy `σ_1 ≥ σ_2 ≥ …` the queries put along each. I'll define the subspace matrix `Q̂ = diag(σ_1,…,σ_r)·V[:r] ∈ R^{r×d}` — the top-`r` right singular vectors *scaled by their singular values*. Why scale them, rather than just take the orthonormal `V[:r]`? Because I don't want to protect all `r` directions equally; I want to protect the directions the queries actually load onto most. The penalty I'm about to build will involve `Q̂ᵀQ̂ = Σ_{j≤r} σ_j² v_j v_jᵀ`, so a direction with a big singular value gets a quadratically bigger weight. The geometry of "which directions matter to attention" is exactly encoded in the singular values, so I carry them. For grouped-query attention, several query heads share one KV head, so before the SVD I concatenate the query tensors of the heads that share a KV head — the subspace has to cover all the queries that will ever dot this head's keys.

With those directions in hand, the key quantization goal becomes honest. For a key `k`, find its `b`-bit representation `k^qtz` minimizing the reconstruction error while keeping the residual orthogonal to `Q̂`:

  `min_{k^qtz} ‖k − deq(k^qtz)‖₂ ` subject to ` Q̂(k − deq(k^qtz)) = 0`, `k^qtz` a vector of `b`-bit ints.

Two things make this hard. The orthogonality is over a subspace I've now estimated, good. But the constraint is a set of linear equations on a residual that must also land on the integer quantization grid — that's a combinatorial, integer-constrained problem, generally intractable to solve exactly. And a hard `Q̂·r = 0` is too rigid: the grid may simply not contain a nearby point whose residual kills all protected projections. So let me relax. Instead of a hard constraint, fold the orthogonality into the objective as a penalty with a knob `λ`:

  `min ‖k − deq(k^qtz)‖₂² + λ‖Q̂(k − deq(k^qtz))‖₂².`

`λ = 0` recovers pure reconstruction — the compression baseline — and `λ → ∞` drives the residual into the orthogonal complement of `Q̂`, enforcing the constraint in the limit. `λ` continuously trades fidelity of the key against orthogonality to the query subspace. That's the right shape. But it's still got the integer grid inside it. How do I actually quantize the coordinates while respecting this penalty?

Let me think about quantizing one coordinate at a time and *compensating*. I can quantize the coordinates of `k` in a fixed order, and after I round a coordinate, I'm still free to *adjust the not-yet-quantized coordinates* to push the total residual back toward orthogonality with `Q̂`. The already-quantized coordinates are locked on the grid; the future coordinates are still continuous, so they're my degrees of freedom to cancel the damage. Let me set up iteration `t`. Let `k̂_t` be the running (de)quantized vector, `k̂_0 = k`. At step `t`, the first `t−1` coordinates are frozen at their quantized values, I round coordinate `t` to the grid, and I update coordinates `t+1,…,d` to minimize the penalized objective. Writing `δ = k̂_t − k̂_{t−1}`, the per-step problem is

  `min_δ δᵀδ + λ δᵀ Q̂ᵀQ̂ δ = δᵀ P δ`,  with  `P = I + λ Q̂ᵀQ̂`,

subject to: the first `t−1` coordinates of `δ` are zero (frozen), and coordinate `t` of `δ` is pinned to the rounding error `e_t = deq(qtz([k̂_{t−1}]_t)) − [k̂_{t−1}]_t`. `P` is `I` plus a PSD matrix, so it's positive definite and invertible — good, the quadratic has a unique minimizer. The constraints are linear: stack the fixed coordinates as `T δ = b`, where `T = [I, 0]` selects the first `t` coordinates and `b = [0,…,0,e_t]ᵀ` is zero on the frozen coordinates and equal to the rounding error on coordinate `t`.

This is a quadratic program with linear equality constraints — solve it with a Lagrangian. `L(δ, μ) = δᵀP δ − μᵀ(Tδ − b)`. Stationarity: `∂L/∂δ = 2Pδ − Tᵀμ = 0`, so `δ = (1/2) P⁻¹ Tᵀ μ`. Feasibility: `Tδ = b`, so `(1/2) T P⁻¹ Tᵀ μ = b`, giving `μ = 2 (T P⁻¹ Tᵀ)⁻¹ b` (assuming `T P⁻¹ Tᵀ` invertible). Substitute back:

  `δ = P⁻¹ Tᵀ (T P⁻¹ Tᵀ)⁻¹ b.`

Now I need to read off what this does to the *future* coordinates concretely, so let me write `P_inv = P⁻¹` in blocks aligned with the split "first `t` coordinates / remaining `d−t`":

  `P_inv = [[A_t, B_tᵀ], [B_t, C_t]]`,  `A_t ∈ R^{t×t}` the top-left block.

With `T = [I, 0]` (the `t×d` selector), `P⁻¹ Tᵀ = [A_t; B_t]` (the first `t` columns of `P_inv`), and `T P⁻¹ Tᵀ = A_t`. So

  `P⁻¹ Tᵀ (T P⁻¹ Tᵀ)⁻¹ = [A_t; B_t] A_t⁻¹ = [I; B_t A_t⁻¹].`

And `b = [0; e_t]` where the nonzero entry `e_t` is on coordinate `t` (the last of the first-`t` block). Multiplying, the first `t` coordinates of `δ` come out as `b` itself — coordinate `t` carries the rounding error, the earlier ones stay zero, exactly the frozen/pinned constraint — and the *remaining* `d−t` coordinates get

  `δ_{remaining} = B_t A_t⁻¹ b = B_t h_t · e_t`,  where `h_t` is the last column of `A_t⁻¹`

(picking the column that multiplies the single nonzero entry of `b`). So the update rule for the not-yet-quantized coordinates is

  `coordinates after t ← coordinates after t + (deq(qtz([k̂_{t−1}]_t)) − [k̂_{t−1}]_t)·B_t h_t.`

The rounding error on coordinate `t` is spread into the future coordinates through `B_t h_t`, in exactly the direction that minimizes the penalized residual — i.e., that pushes the total residual back toward the orthogonal complement of the query subspace. Each rounding mistake gets compensated by the freedom still left downstream.

Wait. Stare at this for a second. `min δᵀ H δ` with a quadratic metric `H`, quantize a coordinate, then update the *remaining* coordinates by a rounding-error vector times inverse-metric blocks to cancel the induced error — I've seen this exact structure. This is the Optimal Brain Surgeon / OBQ error-feedback update. In weight quantization (Hassibi et al.; Frantar & Alistarh's Optimal Brain Compression; GPTQ), you quantize a weight to minimize the layer *output* error `‖WX − ŴX‖²`, whose Hessian is `H = 2XXᵀ`, and the compensation on the free coordinates can be written as `−(w_q − quant(w_q))/[H⁻¹]_{qq}·(H⁻¹)_{F,q}`, equivalently `(quant(w_q) − w_q)/[H⁻¹]_{qq}·(H⁻¹)_{F,q}`; the block form is the same sign convention with `w_Q − quant(w_Q)` negated. My update is the *same algebra*, with one substitution: I've replaced the data Hessian `2XXᵀ` with `P = I + λ Q̂ᵀQ̂`. That's the whole conceptual content in one line — instead of measuring "error" under the data covariance the way weight quantization does, I measure it under the metric `I + λ Q̂ᵀQ̂`, which charges residual components along the query subspace by `(1 + λσ²)` and leaves the orthogonal complement at unit weight. So the OBS machinery, repurposed with a *query-subspace metric* instead of a *data Hessian*, falls right out of my attention-output bound. And it tells me the residual-orthogonality objective and the error-compensating update are the same object: minimizing the residual under `I + λ Q̂ᵀQ̂` *is* the OBS-style compensation under that metric.

That recognition also rescues me from a problem I'd otherwise have. OBS in its pure form picks the *next coordinate to quantize greedily* — the one with the least induced error — and the set of remaining coordinates differs per row, so you'd re-invert per row, which is hopeless to do on-the-fly. But GPTQ's observation is that on large matrices, quantizing in a *fixed* order (just `1, 2, 3, …`) is nearly as good as the greedy order, because a coordinate quantized late, with few remaining coordinates to compensate, balances against the slightly larger error you'd have saved by handling it earlier. If I quantize in fixed coordinate order, then `A_t` and `A_t⁻¹` depend only on `P = I + λ Q̂ᵀQ̂` — not on the data, not on which key — so they're the *same for every key, every token, every sample in the batch*. I compute them once after prefill and reuse them for the whole decode. That's what makes this run on-the-fly with negligible per-step cost; without the fixed-order insight it'd be a non-starter.

Now the efficiency, because the obvious implementation is too slow. At each step `t` I need `A_t⁻¹`, the inverse of the top-left `t×t` block of `P_inv`. Inverting from scratch each step is `Σ_t t³ = O(d⁴)` — way too much for the decode loop. But the `A_t` are nested: `A_t` is the top-left block of `A_{t+1}` (both are top-left blocks of the same `P_inv`). So I should be able to get `A_t⁻¹` from `A_{t+1}⁻¹` by a cheap downdate rather than a fresh inverse. This is again exactly the OBC trick — removing a coordinate from an inverse by one Gaussian-elimination step. Let me derive it with the block-matrix inverse. Write the `(t+1)×(t+1)` inverse in blocks splitting off the *last* coordinate:

  `A_{t+1}⁻¹ = [[M, Nᵀ], [N, O]]`,  `M ∈ R^{t×t}`, `O` the bottom-right scalar (or `g×g` block).

I want `A_t⁻¹`, the inverse of the top-left `t×t` block `A_t` of `A_{t+1}`. The block-inverse / Schur identity says: if `A_{t+1} = [[A_t, *],[*, *]]` and its inverse is `[[M, Nᵀ],[N, O]]`, then `A_t⁻¹ = M − Nᵀ O⁻¹ N`. (This is the corollary of the standard formula `M⁻¹ = [[(M/D)⁻¹, …],[…, …]]`: the top-left block of the inverse equals `(A_t)⁻¹ + (A_t)⁻¹ B (M/A)⁻¹ C (A_t)⁻¹` and back-substituting the off-diagonal blocks `Q = −A_t⁻¹B(M/A)⁻¹`, `R = −(M/A)⁻¹C A_t⁻¹`, `S = (M/A)⁻¹` gives `A_t⁻¹ = M − Q S⁻¹ R = M − Nᵀ O⁻¹ N`.) For the element-by-element case `g = 1`, `O` is the bottom-right scalar `ā` and `N = a` is the bottom row, so `A_t⁻¹ = Ā_{t+1} − (1/ā)·aᵀa` — exactly the one-step Gaussian elimination that removes the trailing coordinate's influence. I start at `A_T⁻¹ = P = I + λ Q̂ᵀQ̂` itself (where `T = d/g`, since `A_T` is all of `P_inv`, so `A_T⁻¹ = P_inv⁻¹ = P`) and recurse downward `A_{T−1}⁻¹, …, A_1⁻¹`. Each downdate is `O(t²)`, so the whole sweep is `O(d³)` instead of `O(d⁴)`. And if `r ≪ d`, the initial inverse `(I + λQ̂ᵀQ̂)⁻¹` can be had in `O(r d²)` by Woodbury, but `O(d³)` once per prefill per head is already cheap.

One more efficiency lever I should take: don't quantize coordinate-by-coordinate at all. Quantize a *block* of `g` coordinates per iteration. Everything above generalizes — `b` carries a length-`g` rounding-error vector `d_t` on the block, `T = [I, 0]` selects the first `tg` coordinates, `A_t ∈ R^{tg×tg}`, `H_t` is the last `g` columns of `A_t⁻¹`, `B_t` is the bottom-left `(d−tg)×tg` block of `P_inv`, and the remaining-coordinate update becomes `coordinates after block t ← coordinates after block t + B_t H_t d_t`. With a head dimension of `128` and `g = 64`, that's `T = 2` blocks — only *one* compensation update — so the whole key step is: quantize block 1, compute its rounding error `d_t`, push `B_1 H_1 d_t` into block 2, quantize block 2, done. Per-token values stay on the plain min-max path. The block downdate uses the same Schur identity with `O` now a `g×g` block, inverted with a tiny dampening `tol·I` for numerical safety.

Let me also settle the knobs from the structure, not by fiat. `λ`: the penalty weight. Because `Q̂` carries the *scaled* singular vectors, `Q̂ᵀQ̂ = Σ σ_j² v_j v_jᵀ` already has large entries (leading query singular values are big), so `λ Q̂ᵀQ̂` becomes dominant at quite small `λ`; push `λ` too high and the orthogonality term swamps reconstruction and the keys are pulled badly off, so a small `λ ≈ 10⁻³` is the right scale for orthogonality without wrecking fidelity. `r`, the subspace dimension: bigger `r` captures more query directions but costs more and risks over-constraining the residual (too many directions to be orthogonal to, not enough freedom in the complement); for short prompts a handful of directions suffice, while longer, richer prompts want a larger `r` — a few dozen — to cover the wider span their queries explore. `g`, coordinates quantized per iteration: sets `T = d/g` iterations and hence how fine-grained the compensation is; bigger `g` is faster (fewer updates) at the cost of coarser error feedback. `G`, the quantization group for scale/zero-point, and `R`, the residual buffer size, I inherit from the streaming compression setup — the freshest keys are kept in full precision until the buffer flushes, the rest are quantized in groups of `G`. The same algorithm runs at any `b`; here I target 4-bit.

Let me also be honest about what the subspace estimate costs me. I'm making the residual orthogonal to the *prompt's* query subspace, betting it covers the *future* queries. The bet is backed by the deviation curves — prompt-only ≈ prompt+response — but it's an approximation; a future query with a component outside `Q̂` is unprotected. That's exactly what `λ` and `r` hedge: `r` wide enough to cover the span the future queries will probably use, `λ` strong enough to protect those directions but not so strong it deforms the key. And sharing `Q̂` (hence `A_t⁻¹`, `P_inv`) across the batch — computing them from the first sample only — is justified by the same low-dimensional-subspace fact: queries from different sequences of the same task land in nearly the same subspace, so one set of matrices serves the batch and saves the memory of per-sample inverses.

The streaming-quantizer slots now have concrete contents: a prefill hook that builds `Q̂` by SVD on the prompt queries, the `A_t⁻¹` recursion built once from `P = I + λ Q̂ᵀQ̂`, the key quantizer that quantizes block by block and compensates the remainder with `B_t H_t d_t`, and the per-token value quantizer with the residual window.

```python
import math
import torch

FP_BITS = 16  # full-precision KV reference


class AdaptiveKVQuantizer:
    """Subspace-orthogonal K/V quantization. Build a query subspace from the prompt
    during prefill; when quantizing keys, push each block's rounding error into the
    not-yet-quantized coordinates so the residual stays orthogonal to that subspace."""

    def __init__(self):
        self.bits = 4
        self.group_size = 32          # scale/zero-point group (per-channel keys, per-token values)
        self.residual_length = 32     # most-recent tokens kept FP16
        self.subspace_dim = 60        # r: top-r query directions
        self.squat_lambda = 0.001     # λ: reconstruction vs orthogonality
        self.quant_group_size = 64    # g: coordinates quantized per iteration (T = d/g)
        self.shared_svd = True        # reuse the subspace across the batch
        self.query_subspaces = {}     # layer_id -> Q-hat

    def reset_request(self, request_meta, budget_state):
        self.query_subspaces = {}

    def needs_prefill_qkv_observer(self) -> bool:
        return True                   # we must see the prompt's queries before decode

    def query_observation_position(self) -> str:
        return "post_rope"

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        # Build Q-hat = diag(sigma[:r]) * V[:r] from the prompt queries (Obs 2).
        if query_states is None:
            return None
        batch, query_heads, _, head_dim = query_states.shape
        kv_heads = int(attention_meta.get("kv_heads", query_heads))
        if query_heads % kv_heads != 0:               # GQA: concat query heads sharing a KV head
            kv_heads = query_heads
        matrix = query_states.reshape(batch, kv_heads, -1, head_dim).float()
        rank = min(int(self.subspace_dim), matrix.shape[-2], matrix.shape[-1])
        if rank <= 0:
            return None
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        # scale the top-r right singular vectors by their singular values
        scaled_vh = torch.diag_embed(singular_values[:, :, :rank]).matmul(vh[:, :, :rank, :])
        self.query_subspaces[layer_id] = (scaled_vh[0:1] if self.shared_svd else scaled_vh).detach()
        return None

    def _residual_keep_length(self, seq_len: int) -> int:
        residual_length = max(0, min(seq_len, int(self.residual_length)))
        return seq_len % residual_length if residual_length else 0

    def _minmax_last_dim(self, data, group_size, bits):
        # asymmetric group-wise min-max quantize+dequantize along the last dim
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, 2 ** int(bits) - 1)
        trailing = data.shape[-1]
        group_size = trailing if int(group_size) <= 0 else int(group_size)
        padded = math.ceil(trailing / group_size) * group_size
        work = torch.nn.functional.pad(data, (0, padded - trailing)) if padded != trailing else data
        grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
        gmin = grouped.amin(dim=-1, keepdim=True)
        gmax = grouped.amax(dim=-1, keepdim=True)
        scale = (gmax - gmin).clamp(min=1e-5) / max_int          # Delta = (max-min)/(2^b-1)
        q = torch.round((grouped - gmin) / scale).clamp(0, max_int)
        return q.mul(scale).add(gmin).reshape(*work.shape[:-1], padded)[..., :trailing]

    def _generate_At_inv(self, query_subspace, tol: float = 1e-7):
        # A_T^{-1} = P = I + lambda * Q-hat^T Q-hat, then downdate A_t^{-1} from A_{t+1}^{-1}.
        batch, heads, _, head_dim = query_subspace.shape
        q_group = head_dim if int(self.quant_group_size) <= 0 else int(self.quant_group_size)
        groups = math.ceil(head_dim / q_group)                   # T = d/g
        matrices = [None] * groups
        eye = torch.eye(head_dim, device=query_subspace.device, dtype=torch.float32)
        A_t = eye.expand(batch, heads, head_dim, head_dim) + float(self.squat_lambda) * \
            query_subspace.float().transpose(-1, -2).matmul(query_subspace.float())
        matrices[groups - 1] = A_t                               # A_T^{-1} = P (since A_T = P_inv)
        for group_idx in range(groups - 1, 0, -1):
            current_dim = group_idx * q_group
            width = min(q_group, A_t.shape[-1] - current_dim)
            # split A_{t+1}^{-1} = [[M, N^T], [N, O]] about the trailing g-block
            M_t1 = A_t[:, :, :current_dim, :current_dim]
            N_t1 = A_t[:, :, current_dim:current_dim + width, :current_dim]
            O_t1 = A_t[:, :, current_dim:current_dim + width, current_dim:current_dim + width]
            local_eye = torch.eye(width, device=query_subspace.device, dtype=torch.float32)
            O_inv = torch.inverse(O_t1 + tol * local_eye.expand(batch, heads, width, width))
            A_t = M_t1 - N_t1.transpose(-1, -2).matmul(O_inv.matmul(N_t1))   # Schur: A_t^{-1}=M - N^T O^{-1} N
            matrices[group_idx - 1] = A_t[:, :, :, -q_group:]    # H_t = last g columns of A_t^{-1}
        return matrices

    def _squat_quantize_keys(self, key_states, query_subspace):
        # Per-channel key quantization: quantize block i, compensate the rest with B_t H_t d.
        batch, heads, _, head_dim = key_states.shape
        query_subspace = query_subspace.to(device=key_states.device)
        if query_subspace.shape[0] == 1 and batch > 1:           # shared subspace -> broadcast
            query_subspace = query_subspace.expand(batch, -1, -1, -1)
        if query_subspace.shape[1] != heads or query_subspace.shape[-1] != head_dim:
            raise ValueError("query subspace shape does not match the key tensor")
        matrices = self._generate_At_inv(query_subspace)         # A_1^{-1},...,A_{T-1}^{-1}, A_T^{-1}
        P_inv = torch.inverse(matrices[-1])                      # P_inv = (I + lambda Q^T Q)^{-1}
        work = key_states.float().clone()
        q_group = head_dim if int(self.quant_group_size) <= 0 else int(self.quant_group_size)
        groups = math.ceil(head_dim / q_group)
        for group_idx in range(groups):
            start = group_idx * q_group
            end = min(head_dim, start + q_group)
            chunk = work[:, :, :, start:end]
            # per-channel quantization of this block (group along the token axis)
            dequant = self._minmax_last_dim(chunk.transpose(2, 3).contiguous(),
                                            self.group_size, self.bits).transpose(2, 3)
            if group_idx < groups - 1:
                d_vec = (dequant - chunk).float()               # rounding error of this block
                next_start = end
                H_t = matrices[group_idx]                        # last g cols of A_t^{-1}
                B_t = P_inv[:, :, next_start:, :next_start]      # bottom-left block of P_inv
                # push the error into the not-yet-quantized coordinates: + B_t H_t d
                update = d_vec.matmul(H_t.transpose(-2, -1)).matmul(B_t.transpose(-2, -1))
                work[:, :, :, next_start:] = work[:, :, :, next_start:] + update
            work[:, :, :, start:end] = dequant
        return work

    def _quantize_with_residual(self, tensor, quant_fn):
        # keep the most-recent `residual` tokens FP16, quantize the rest
        work = tensor.float().clone()
        _, _, seq_len, _ = work.shape
        residual = self._residual_keep_length(seq_len)
        quant_end = seq_len - residual
        if quant_end <= 0:
            return work.to(tensor.dtype), FP_BITS
        work[:, :, :quant_end, :] = quant_fn(work[:, :, :quant_end, :])
        avg_bits = (quant_end * self.bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id, key_states, cache_meta):
        query_subspace = self.query_subspaces.get(layer_id)
        if query_subspace is None:
            raise RuntimeError("key quantization needs the prefill query observer")
        return self._quantize_with_residual(
            key_states, lambda data: self._squat_quantize_keys(data, query_subspace))

    def quantize_value(self, layer_id, value_states, cache_meta):
        # values: plain per-token min-max (Theorem term (i)); no subspace needed
        return self._quantize_with_residual(
            value_states, lambda data: self._minmax_last_dim(data, self.group_size, self.bits))

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta) -> float:
        residual = self._residual_keep_length(seq_len)
        quant_tokens = max(0, seq_len - residual)
        return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
```

So the causal chain. I started from the fact that a quantized key is only ever read through `q_n·k_iᵀ`, so what I should preserve is the attention output, not the key's Euclidean norm. Bounding the attention-output error split it into a value-reconstruction term — handled by per-token value quantization — and a key term that depends on the *inner product* of the key residual with future queries, telling me to make each key residual orthogonal to the queries rather than small. Future queries are unknown at quantization time, but queries empirically live in a low-dimensional subspace that the prompt's queries already span, so I estimate that subspace by SVD on the prompt and protect its top directions, scaled by singular values so the directions the queries use most are protected most. Turning "residual orthogonal to `Q̂`" into a soft penalty gave the metric `P = I + λ Q̂ᵀQ̂`, and minimizing the residual under that metric coordinate-block by coordinate-block — quantize a block, then push its rounding error into the not-yet-quantized coordinates — turned out to be the Optimal Brain Surgeon error-feedback update with the data Hessian replaced by the query-subspace metric. Quantizing in fixed order (the GPTQ insight) makes the inverse matrices `A_t⁻¹`, `P_inv` identical across keys, tokens, and batch — they depend only on `P = I + λ Q̂ᵀQ̂`, not on the data — so each key step rebuilds the same tables from the prefill subspace; the OBC Schur-complement downdate computes the nested inverses `A_t⁻¹` in `O(d³)` instead of `O(d⁴)`; and with `g = 64` on a `128`-dim head it's two blocks and one compensation step. Values go through plain per-token min-max, the most-recent residual window of tokens stays FP16, and the knobs `λ`, `r`, `g`, `G`, `R` follow from the scaled-singular-vector penalty, the span the future queries occupy, and the streaming compression setup — all running on-the-fly with no calibration and no fine-tuning.
