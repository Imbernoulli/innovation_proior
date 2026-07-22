OK, let me think this through from scratch. I have a sequence model with no recurrence and no convolution in it — just self-attention sub-layers and per-position feed-forward sub-layers stacked up. And I have a problem that I keep tripping over when I look at what this stack actually computes: it has no idea what order the tokens are in.

Let me make sure that's really true before I spend effort fixing it, because if it isn't true I'm wasting my time. Take the self-attention output at one position. It's `softmax(q·k_j / √d) ` weights times the values `v_j`, summed over all positions `j`. The query `q` is a linear function of the token at my position; each key `k_j` and value `v_j` is a linear function of the token at position `j`. Nowhere in `q·k_j` does the index `j` appear — only the *contents* of the two token vectors. So if I take the whole input and permute the tokens — move position 3 to position 7 and vice versa — every `k_j, v_j` just gets relabeled, the set `{k_j, v_j}` is identical, the softmax is over the same set of scores, and the weighted sum is exactly the same number. Permute the input, the output permutes the same way with identical values. It's permutation-equivariant. And the feed-forward sub-layer acts on each position on its own, so it can't see neighbors at all. Stack these forever and the model still reads its input as a *bag* of token vectors. "the cat sat" and "sat the cat" are the same object to it. So yes — order is genuinely invisible, and I have to put it back in by hand.

So what does "putting order back in" even mean operationally? Somewhere, the number that says "this token is at position `t`" has to enter the computation, and it has to enter in a form the existing machinery can consume. The existing machinery is: linear layers, dot products, softmax. All of it eats `d_model`-dimensional vectors. The cleanest thing I can do, the thing that doesn't force me to redesign attention, is to manufacture, for each position `t`, a vector `p_t ∈ ℝ^d_model` that somehow *is* the position, and fuse it with the token embedding at the bottom of the stack. Then the very first linear projections (the ones producing `q, k, v`) get to look at a vector that already knows where it sits, and order is suddenly available to everything above.

Two ways to fuse a position vector with a token vector: concatenate them, or add them. Let me actually think about which, because I'd otherwise just default to one without reason. If I concatenate, the input becomes `[embed(x_t); p_t]` of width `d + d_pos`, and every downstream weight matrix grows to that width. What do I buy for that? The first linear layer applied to a concatenation `[a; b]` is `W[a;b] = W_a a + W_b b` — it's *exactly* the same as applying one matrix to `a`, another to `b`, and summing. So concatenation is just "add, but give the position its own private input projection and pay for extra width at every layer above." If I instead simply *add* `embed(x_t) + p_t` and keep the width at `d`, the first projection `W(embed + p) = W·embed + W·p` still gets a separate linear readout of the position (`W·p`), and I've widened nothing. The model can carve out a subspace of the `d` dimensions to hold position and learn to read it out — it doesn't need dedicated channels. So: add, not concat. Same as the learned-table approach already does, and at the same width-`d` cost. Good — addition it is.

Now the real question: what is `p_t`? The obvious first try is the dumbest one. Let the position vector just carry the integer: `p_t = t` broadcast, or stuck in one channel. Stare at this for a second. `t` is unbounded. If I train on sequences up to length 20 and then run at length 40, the model has seen position values 0…20 and is now being fed 21…40 — numbers in a range it never trained on, and large-magnitude inputs that push the downstream linear layers into regimes they were never calibrated for. The activations blow up. Dead end, and it's exactly the failure I want to avoid since the whole point is to behave on sequences longer than training.

Patch attempt: normalize it. `p_t = t / L` where `L` is the sequence length, so it lives in `[0, 1]` and never blows up. But now stare at what a *step* means. In a length-10 sequence, going from position 3 to position 4 is a jump of `1/10 = 0.1`. In a length-100 sequence the same one-token step is `1/100 = 0.01`. So "the token one step back" corresponds to a different numeric delta depending on how long the sentence happens to be. The model can't learn a single rule for "look one position back," because one position back isn't a fixed quantity anymore. The encoding has to mean the same thing across sequences of different lengths, and `t/L` doesn't. Dead end.

So I want something bounded — like the normalized version — but where a fixed positional *step* always corresponds to the same change in the code, regardless of where in the sequence I am or how long the sequence is. Bounded plus shift-consistent. What's bounded and has a built-in notion of "constant step"? A periodic function. `sin(ω t)` lives in `[-1, 1]` for every `t`, defined for all real `t`, and a step of `Δt` always advances the phase by `ω Δt` no matter where I am. That's promising. But one sinusoid alone is hopeless: it aliases. `sin(ω t)` comes back to the same value every period `2π/ω`, so positions a period apart are literally identical, and worse, a single scalar can't possibly distinguish a few hundred positions — there isn't enough information in one number. So one frequency is out.

The fix has to be many frequencies at once. Here's the picture I keep coming back to: how does a plain binary counter give every integer a rich code with bounded (0/1) digits? The lowest bit flips every step, the next bit every two steps, the next every four — each bit oscillates at half the rate of the one below it, and the *combination* of all the bits separates an enormous range while each individual bit stays in `{0,1}`. I want the continuous analogue of that. Replace each bit with a sinusoid, and let the frequencies decrease geometrically as I go up the dimensions — fast oscillation in the first dimensions (the "low bits," fine-grained local position), slow oscillation in the later dimensions (the "high bits," coarse global position). A whole vector of sinusoids at geometrically spaced frequencies: bounded in `[-1,1]` everywhere, defined for every real `t`, and practically separated over the intended range by having clocks at many scales. That's the bounded-and-discriminative part solved.

Let me pin the frequencies. I'll index the dimension pairs by `i = 0, 1, …, d/2 − 1`, and set the angular frequency of pair `i` to `ω_i = 10000^{−2i/d}`. So `ω_0 = 1` (the fastest, wavelength `2π`) and the last pair has frequency close to `10000^{-1}` (the slowest, wavelength close to `10000·2π`). Why geometric, and why 10000? Geometric because then each successive dimension multiplies the wavelength by a constant factor, so the dimensions tile the scale axis evenly in log-space — I get roughly equal resolution at every scale from local to global, instead of bunching all my frequencies near one band. And 10000 sets the slow end of the wavelength range: the slowest clock changes only gradually across ordinary sequence lengths, while the fast clocks keep local positions distinct.

Now the part that actually decides the design — the relative-position requirement. I said I want "one token back" to mean the same thing everywhere. Let me see whether a vector of sinusoids gives me that, and let me be careful, because I think the naive single-sinusoid-per-dimension layout almost works but is missing something. Take one frequency `ω` and ask: is there a fixed transformation, the same at every absolute position `t`, that maps the code at `t` to the code at `t + k`? If I only stored `sin(ω t)`, then I'd need to get `sin(ω(t+k))` from it. Expand: `sin(ω(t+k)) = sin(ωt)cos(ωk) + cos(ωt)sin(ωk)`. The problem jumps out — the right side needs `cos(ωt)`, which I threw away. From `sin(ωt)` alone I can't recover the shift, because at a given value of `sin(ωt)` I don't know whether the phase is rising or falling; I've lost half the state of the oscillator. So a single sinusoid per dimension can't support a clean shift map.

But that immediately tells me the cure: keep *both* `sin(ωt)` and `cos(ωt)` for each frequency — store the full phase as a point on the unit circle, the pair `(sin ωt, cos ωt)` at angle `ωt`. Now redo the shift. Using the angle-addition formulas,

  sin(ω(t+k)) = sin(ωt)cos(ωk) + cos(ωt)sin(ωk),
  cos(ω(t+k)) = cos(ωt)cos(ωk) − sin(ωt)sin(ωk).

Write that as a matrix acting on the pair:

  ⎡ sin(ω(t+k)) ⎤   ⎡  cos(ωk)   sin(ωk) ⎤ ⎡ sin(ωt) ⎤
  ⎢            ⎥ = ⎢                    ⎥ ⎢        ⎥.
  ⎣ cos(ω(t+k)) ⎦   ⎣ −sin(ωk)   cos(ωk) ⎦ ⎣ cos(ωt) ⎦

Look at that 2×2 matrix. It depends only on `k` — the offset — and on the fixed frequency `ω`. It does **not** depend on `t`. In this stored `(sin, cos)` coordinate order, it is the orthogonal rotation that advances the phase by `ωk`; if I had written the coordinates as `(cos, sin)`, the signs would be arranged differently. So the map "advance the position by `k`" is, for each frequency, a fixed rotation `R(ω_i k)`, and stacking the rotations for all the frequency pairs into a block-diagonal matrix `M_k = diag(R(ω_0 k), R(ω_1 k), …)`, I get `p_{t+k} = M_k p_t`, a single linear transformation that is the *same* at every absolute position `t` and depends only on the relative offset `k`. For "k steps back" I use the same matrix with offset `-k`. That's the property I was after, made precise: the code for a fixed relative shift is a fixed linear function of the current code, identical everywhere in the sequence. *That* is why the pairing of sine with cosine per frequency, and why I pair them rather than just listing sines: without the cosine partner the rotation has nothing to act on.

So the layout writes itself: interleave, even dimensions hold the sines, odd dimensions hold the cosines, one frequency per pair:

  PE(t, 2i)   = sin(t / 10000^{2i/d}),
  PE(t, 2i+1) = cos(t / 10000^{2i/d}),

with `i = 0 … d/2 − 1`. Each dimension is a sinusoid; consecutive even/odd dimensions share a frequency so they form the (sin, cos) phase pair the rotation argument needs.

And now the fifth requirement, extrapolation, I almost get for nothing — let me make sure I actually get it and I'm not fooling myself. The learned table had a hard wall: it stores a separate vector for indices `1…L` and has literally no entry for `L+1`, so beyond training length there's nothing to look up. Here, `sin` and `cos` are defined for every real `t`. Position `L+1` isn't a missing slot — it's just the next point further along the same curves the model has been seeing the whole time, and the rotation relation `p_{t+k} = M_k p_t` holds for those positions too, with the same `M_k`. So whatever fixed-offset logic the model learned at small positions is, structurally, still valid at large ones. It's not a guarantee the learned weights generalize, but the *representation* doesn't break the way a finite table does.

One more thing I have to get right or the whole signal is poorly balanced, and it's easy to miss. I'm adding `p_t` to the token embedding `embed(x_t)`. The sinusoids are `O(1)` — components in `[-1, 1]`, with each sine/cosine pair contributing squared norm 1, so the positional vector has norm about `√(d/2)`. What's the typical magnitude of an embedding component? With the usual small embedding initialization, the raw lookup has much smaller per-component scale than the fixed sinusoid. So I scale the embedding up before adding: multiply `embed(x_t)` by `√d_model`. That is an embedding-layer scale, not a change to the position formula. Its job is to put token content and fixed position on comparable numerical scales at the point where I sum them, so neither side is washed out.

Let me also sanity-check that I'm not overcomplicating the precompute. The frequencies `10000^{−2i/d}` for `i = 0 … d/2−1` are best computed in log space rather than calling `pow` repeatedly: `10000^{−2i/d} = exp((2i) · (−ln 10000 / d))`. So I build `div_term = exp(arange(0, d, 2) · (−ln 10000 / d))`, which is the vector of `ω_i`, then outer-product it with the position indices to fill the table. The whole encoding is fixed — no learnable parameters, no gradient — so I compute it into a buffer and add the slice for the current sequence length. If the run needs positions beyond the buffer, I must extend or recompute the same formula; clamping all later positions to the final buffer row would silently destroy the encoding by making distinct positions identical.

Let me write the concrete thing the model actually runs. A fixed sinusoidal table holds the phase code; the embedding module multiplies token lookups by `√d_model`; the positional module adds the table slice and applies dropout to the sum; then the result enters the order-blind decoder stack — which now isn't order-blind, because every position has been stamped with its phase code before the first attention layer touches it.

```python
import math
import torch
import torch.nn as nn


class ScaledEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lut(tokens) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal absolute positional encoding, added before dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("sinusoidal positional encoding requires even d_model")
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = self._build(max_len, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("pe", pe.unsqueeze(0))

    def _build(self, length: int, device, dtype) -> torch.Tensor:
        pe = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T <= self.pe.size(1):
            pe = self.pe[:, :T].to(device=x.device, dtype=x.dtype)
        else:
            pe = self._build(T, device=x.device, dtype=x.dtype).unsqueeze(0)
        return self.dropout(x + pe.detach())


class SeqModel(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers, max_len, dropout=0.1):
        super().__init__()
        self.tok = ScaledEmbedding(d_model, vocab)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.lut.weight

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.pos(self.tok(tokens))
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=tokens.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal)
        return self.head(self.ln_f(x))
```

The chain, end to end: the attention/feed-forward stack is permutation-equivariant, so order is invisible and must be added as a `d_model` vector → addition beats concatenation because a linear layer reads a projection off the added position at no extra width → a raw index is unbounded and a length-normalized index makes a fixed step mean different things, so I need a bounded, shift-consistent code → one bounded sinusoid aliases, so I use many geometrically spaced frequencies, a continuous counter with clocks at many scales → pairing `sin` with `cos` per frequency makes a position shift a fixed `t`-independent rotation, so relative offsets become a single linear map identical everywhere → because `sin`/`cos` are defined for all `t`, positions past the training length are not missing table entries → and the embedding is scaled by `√d_model` so the learned content and fixed position live on comparable scales at the sum.
