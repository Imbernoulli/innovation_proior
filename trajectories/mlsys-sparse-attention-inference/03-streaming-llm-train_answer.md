BigBird landed where the NIAH-vs-QA split predicted, and the way it failed is the lesson. NIAH collapsed from the oracle's $1.0$ to $0.2$ — chance — because a content-blind mask covers a mid-haystack needle only by luck of window, global, or fixed-random placement. The QA tasks held up as distributed evidence should (Qasper F1 $0.0871$ against the oracle's $0.1406$, MultiFieldQA-EN $0.2298$ against $0.3447$), but the density readings are the real tell: $0.2601$, $0.2421$, $0.2693$ — two of three *above* $0.25$ and the third right at the $0.25 + 0.02$ ceiling. So BigBird did not fail for lack of budget; it spent its full budget, and then some, and still cratered NIAH and underperformed on Qasper. A meaningful share of that budget went to **random** blocks — long-range expander edges chosen blind to the question — and at inference, with a fixed sample and no query signal, a random far block is overwhelmingly likely to be irrelevant, so that budget was wasted on noise *and* starved the parts that reliably carry signal. The right move is not adaptivity yet; it is to fix the static floor first and cut the random gamble that bought nothing.

I propose **StreamingLLM** — a few attention sinks plus a recent sliding window — and the two ingredients are exactly the two BigBird diluted with random. The first is the **recent local window**: in trained models the keys immediately preceding the query carry the bulk of the attention mass, and next-token prediction leans hardest on local context. The second is the **attention sinks** at the very start, and there is a real mechanism here, not a heuristic. Softmax forces the attention weights to sum to one — there is no "attend to nothing" option — so on a query whose context holds nothing it strongly needs, the model must still deposit a full unit of mass somewhere, and it learns to dump that surplus on a fixed, always-reachable set of positions. Under causal masking, the only positions visible to *every* later query are the first few tokens, so those become the universal dumping ground. They are valuable not for their content but for the softmax denominator they hold: drop them and every remaining weight is renormalized into a shape the model never saw, and quality falls off a cliff. So a few sink tokens plus a recent window is the minimal static pattern that respects how trained attention actually distributes its mass. I use `num_sinks = 4`, because models with no single consistent start token spread the sink role across several initial positions and one or two do not fully restore the distribution.

This rung is also a deliberate control. Sink+window and the adaptive method I will eventually want are not competitors — the adaptive method still needs a guaranteed local keep and still benefits from anchors — so whatever I learn about sizing the local budget carries forward, and there is falsification value I do not get by skipping it: if a *correctly sized* sink+window, spending its whole budget on local context and anchors with nothing on random, still leaves NIAH at chance, then I have isolated the NIAH failure to *staticness per se* rather than to BigBird's particular budget split. I should be precise about what the sinks do *not* buy, because it bounds my NIAH expectation: the sinks sit at the *start*, so they cover the needle only if it was planted in the first few tokens, a vanishing chance in an 8K haystack. The sinks fix *stability*, not *retrieval*.

What makes this *this task's* method and not the canonical streaming formulation is substantial. The canonical StreamingLLM is a **KV-cache eviction** scheme: it keeps a rolling cache, evicts the middle, and re-indexes positions *within the cache* so the rotary positions stay contiguous and in-distribution, requiring a position-shift adapter that rotates cached keys by cache-position at each decode step. None of that exists here. The harness runs `use_cache=False`; every forward is a full parallel pass over the entire prefix, the same module replays at every generation step, and the positions are the model's own RoPE, applied before I see $q, k, v$. So in this setting StreamingLLM is not a cache policy at all — it is a **static sink+window mask** over the full $(N, N)$ causal matrix, and only the *attention pattern* transfers, not the constant-memory/latency story. I therefore size the window from the density contract directly. Where BigBird's overshoot teaches me to be exact rather than conservative-by-fudge-factor, I want the measured density to land *at* the budget, because every token of window I can afford is local context recovered. Once the query index exceeds $\text{num\_sinks} + W$, each row keeps exactly $\text{num\_sinks} + W$ keys — the sinks plus the last-$W$ window, no overlap — so the total mask sum is about $N \cdot (\text{num\_sinks} + W)$ and the density over $N(N{+}1)/2$ is
$$\rho = \frac{2(\text{num\_sinks} + W)}{N + 1}.$$
Setting $\rho$ equal to the budget and solving gives
$$W \approx \mathrm{round}\!\left(\frac{\rho\,(N+1)}{2}\right) - \text{num\_sinks},$$
clamped to at least 1. This is the correction that matters: the naive sizing that took $\text{avg\_row} = \rho (N{+}1)/2$ as a *row-count* conflated the row-relative window with the column-relative density and over-shot — the same overshoot BigBird showed — while deriving $W$ straight from $\text{mask\_sum}/\text{denom}$ lands it on budget. (The non-causal branch is the symmetric $|i - j| \le W$ with $W \approx (\mathrm{round}(\rho N) - \text{num\_sinks} - 1)/2$, but `is_causal=True` always here, so the causal formula is the live one.)

The implementation is the same masked-softmax form as the previous rung: build the $(N, N)$ boolean keep-mask — $(i - j) \ge 0 \,\&\, (i - j) < W$ for the row-relative last-$W$ window, OR $j < \text{num\_sinks}$ for the sink columns, AND the causal lower triangle — then compute $QK^\top \cdot \text{scale}$ in float32 for stable masking and softmax, `masked_fill` the dropped entries with $-\infty$, softmax, `nan_to_num` any empty row, multiply by $V$, cast back. I report `last_density` as the *measured* fraction of the realized mask over $N(N{+}1)/2$, not the formula, so the contract gets the true kept fraction. Because the window is solved to hit budget exactly, I expect all three densities to sit just *under* $0.25$ — already a falsifiable improvement over BigBird, whose two-of-three densities sat above it. On QA I expect to reclaim the budget BigBird wasted on random and turn it into local window where the evidence and the language-modeling signal live, pulling Qasper F1 up from $0.0871$ toward the $0.10$ mark and holding MultiFieldQA in its band. On NIAH I am honest: a static sink+window still cannot reach a mid-haystack needle, so I expect it to stay stuck around $0.2$. If NIAH stays at chance while the QA numbers rise and the densities drop under budget, that is the clean diagnosis — the static floor is now correct, and the only remaining gap to the oracle's NIAH $1.0$ is the one thing no fixed mask can buy: a selection that depends on what the query is asking. That gap hands the next rung its job.

```python
# EDITABLE region of custom_sparse_attn.py — rung 3: StreamingLLM (sink + sliding window)
class SparseAttention(nn.Module):
    """StreamingLLM-style sink + sliding window attention.

    Causal mode (LLM): row-relative last-W window (``i-W+1 <= j <= i``) plus
    the first ``num_sinks`` columns.
    Non-causal mode (ViT/DiT): symmetric window (``|i-j| <= W``) plus the
    first ``num_sinks`` columns.
    """

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget
        # 4 attention sinks: enough to recover almost all the dense-attention
        # quality on streaming inputs.
        self.num_sinks = 4
        self.last_density = None

    def _build_mask(self, N, device, is_causal):
        idx = torch.arange(N, device=device)
        if is_causal:
            # Solve density = 2*(num_sinks+W)/(N+1) = budget for W.
            W = max(1, int(round(self.density_budget * (N + 1) / 2.0)) - self.num_sinks)
            di = idx[:, None] - idx[None, :]
            local = (di >= 0) & (di < W)
        else:
            # Solve density = (num_sinks + 2W + 1)/N = budget for W.
            W = max(1, (int(round(self.density_budget * float(N))) - self.num_sinks - 1) // 2)
            di = idx[:, None] - idx[None, :]
            local = di.abs() <= W
        sinks = (idx[None, :] < min(self.num_sinks, N))
        mask = local | sinks
        if is_causal:
            mask = mask & (idx[:, None] >= idx[None, :])
        return mask

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        scale = scale if scale is not None else 1.0 / math.sqrt(D)

        mask = self._build_mask(N, q.device, is_causal)  # (N, N)
        denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
        self.last_density = float(mask.sum().item()) / max(denom, 1.0)

        # Broadcast (N,N) mask across (B,H).
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v.float())
        return out.to(q.dtype)
```
