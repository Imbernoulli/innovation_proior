The model I am working with is built entirely out of self-attention and per-position feed-forward layers, with no recurrence and no convolution anywhere in it, and that choice buys parallel throughput at the cost of one structural defect that has to be repaired before the thing can do sequence transduction at all: it has no notion of order. A self-attention output at a position is $\mathrm{softmax}(q\cdot k_j/\sqrt{d})$ weights against the values $v_j$, summed over all $j$, and the score $q\cdot k_j$ depends only on the *contents* of the query and key vectors, never on the index $j$. Permute the input tokens and every $k_j, v_j$ is merely relabeled, the set is identical, the softmax runs over the same scores, and the weighted sum comes out unchanged — the layer is permutation-equivariant, and the feed-forward sub-layer acts on each position alone and cannot see neighbors either. Stacked forever, this reads its input as a bag of token vectors: "the cat sat" and "sat the cat" are the same object. So order must be injected by hand, as a fixed-width $d_{\text{model}}$ object that the existing linear-and-softmax machinery can consume, and it has to satisfy several constraints at once — bounded magnitude as the index grows so long sequences do not push activations outside the trained range, a distinct code per position, relative offsets that are easy to use and mean the same thing everywhere, and definedness beyond the longest training sequence so the model can run on longer inputs. The options on the table fail one or another of these. Recurrence supplies order for free but is exactly the sequential bottleneck I am trying to eliminate. A learned absolute-position table $p_1,\dots,p_L$ added to the embeddings is the right template — order as an addable width-$d$ vector — but it has a hard wall at $L$ (position $L+1$ has no entry, so it cannot extrapolate) and each $p_t$ is learned independently, so no relationship between the code at $t$ and at $t+k$ is guaranteed by construction. A raw integer index is unbounded and blows up the downstream linear layers at test-time lengths; normalizing it to $t/L$ stays bounded but makes a one-token step worth $0.1$ in a length-10 sentence and $0.01$ in a length-100 one, so "one position back" stops meaning a fixed quantity. And a single bounded periodic counter $\sin(\omega t)$ aliases — it returns to the same value every period, and one scalar cannot identify a position among hundreds.

What I propose is Sinusoidal Absolute Positional Encoding: a fixed, non-learned vector that encodes position $t$ as paired sinusoids at geometrically spaced frequencies, added to the token embedding at the bottom of the stack. The two questions are how to fuse the position vector with the token vector, and what that vector should be. On fusion, I add rather than concatenate, and the reason is that concatenation buys nothing real: the first linear layer over a concatenation $[e; p]$ is $W[e;p] = W_a e + W_b p$, which is exactly applying one matrix to $e$, another to $p$, and summing — so concatenation is just "add, but give position its own private projection and pay for extra width at every layer above." If I simply add $\mathrm{embed}(x_t) + p_t$ at width $d$, the very first projection still gets a separate linear readout $W\cdot p$ of the position, and I have widened nothing; the model can carve out a subspace of the $d$ dimensions to hold position and read it back out. So addition, at the same width-$d$ cost as the learned table. For the vector itself I want something bounded like the normalized index but where a fixed positional step always advances the code by the same amount regardless of where I am or how long the sequence is — bounded plus shift-consistent. A periodic function gives exactly that: $\sin(\omega t)$ lives in $[-1,1]$ for every real $t$ and a step of $\Delta t$ always advances the phase by $\omega\,\Delta t$. One frequency aliases, so I use many at once, in direct analogy to a binary counter where the lowest bit flips every step, the next every two, the next every four — each bit oscillates at half the rate below it, and the combination separates an enormous range while every bit stays bounded. Replacing each bit with a sinusoid and letting the frequencies decrease geometrically up the dimensions gives fast clocks (fine local position) in the early dimensions and slow clocks (coarse global position) in the later ones. Concretely I index the pairs by $i = 0,\dots,d_{\text{model}}/2 - 1$ and set the angular frequency of pair $i$ to $\omega_i = 10000^{-2i/d_{\text{model}}}$, so the wavelengths run as a geometric progression from $2\pi$ (fastest, $i=0$) to roughly $10000\cdot 2\pi$ (slowest). Geometric spacing is the choice because then each dimension multiplies the wavelength by a constant factor, tiling the scale axis evenly in log-space and giving roughly equal resolution from local to global rather than bunching frequencies in one band; the constant $10000$ fixes the slow end of that range.

$$
PE_{(t,\,2i)} = \sin\!\big(t \,/\, 10000^{2i/d_{\text{model}}}\big), \qquad
PE_{(t,\,2i+1)} = \cos\!\big(t \,/\, 10000^{2i/d_{\text{model}}}\big).
$$

The load-bearing design choice — the one that actually decides the layout — is pairing a sine with a cosine at each frequency rather than just listing sines, and it is forced by the relative-position requirement. Ask whether there is a fixed transformation, identical at every absolute $t$, that maps the code at $t$ to the code at $t+k$. With only $\sin(\omega t)$ stored, I would need $\sin(\omega(t+k)) = \sin(\omega t)\cos(\omega k) + \cos(\omega t)\sin(\omega k)$, and the right side requires $\cos(\omega t)$, which I discarded — from $\sin(\omega t)$ alone the phase is ambiguous (I cannot tell whether it is rising or falling), so half the oscillator state is lost and no clean shift map exists. Keep both, store the full phase as the pair $(\sin\omega t, \cos\omega t)$ on the unit circle, and the angle-addition formulas give

$$
\begin{bmatrix} \sin \omega(t+k) \\ \cos \omega(t+k) \end{bmatrix}
=
\begin{bmatrix} \cos \omega k & \sin \omega k \\ -\sin \omega k & \cos \omega k \end{bmatrix}
\begin{bmatrix} \sin \omega t \\ \cos \omega t \end{bmatrix}.
$$

That $2\times 2$ matrix in this $(\sin,\cos)$ coordinate order depends only on the offset $k$ and the fixed frequency $\omega$, not on $t$: it is the orthogonal rotation that advances the phase by $\omega k$. Stacking the per-frequency rotations block-diagonally gives $PE_{t+k} = M_k\, PE_t$ with $M_k$ independent of absolute position, and "$k$ positions back" is the same matrix at offset $-k$. So a fixed relative shift is a single linear function of the current code, identical everywhere in the sequence — which is precisely why the cosine partner must be there, since without it the rotation has nothing to act on. The fifth requirement, extrapolation, then comes almost for free: $\sin$ and $\cos$ are defined for every real $t$, so position $L+1$ is not a missing table slot but simply the next point on the same curves the model has been seeing all along, and the rotation relation holds there with the same $M_k$. There is one more thing to get right or the signal is poorly balanced: each sine/cosine pair contributes squared norm $1$, so $p_t$ has norm about $\sqrt{d/2}$ and its components are $O(1)$, while a freshly initialized embedding lookup has much smaller per-component scale. So I scale the token embedding up by $\sqrt{d_{\text{model}}}$ before the addition — an embedding-layer scale, not a change to the position formula — to put learned content and fixed position on comparable numerical magnitudes at the point where they are summed, so neither side washes the other out. The frequencies are computed in log space, $10000^{-2i/d_{\text{model}}} = \exp\!\big(2i\cdot(-\ln 10000/d_{\text{model}})\big)$, the whole encoding is fixed with no learnable parameters or gradient and stored as a buffer; if a run needs positions past the buffer I recompute the same formula rather than clamping later positions to the last row, which would silently make distinct positions identical.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lut(tokens) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Fixed sinusoidal absolute positional encoding, added before dropout."

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("sinusoidal positional encoding requires even d_model")
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = self._build(max_len, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("pe", pe.unsqueeze(0))     # [1, max_len, d_model]

    def _build(self, length: int, device, dtype) -> torch.Tensor:
        pe = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)     # omega_i = 10000^{-2i/d_model}
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims: sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims:  cos partner
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T <= self.pe.size(1):
            pe = self.pe[:, :T].to(device=x.device, dtype=x.dtype)
        else:
            pe = self._build(T, device=x.device, dtype=x.dtype).unsqueeze(0)
        return self.dropout(x + pe.detach())


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class SeqModel(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers, max_len, dropout=0.1):
        super().__init__()
        self.tok = ScaledEmbedding(d_model, vocab)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.lut.weight          # tied embeddings

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.pos(self.tok(tokens))                  # scaled embeddings + fixed PE
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=tokens.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal)
        return self.head(self.ln_f(x))
```
