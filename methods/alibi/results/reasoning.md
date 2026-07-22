Let me start from the thing that actually annoys me in practice. I train a transformer language model on input subsequences of length $L$, and that single number sets the cost of everything: attention is quadratic in $L$, so doubling the training length roughly quadruples the attention compute and memory per step. Longer inputs are unambiguously good for the model — every prediction gets more left context, the loss drops — but they are punishingly expensive to train on. So the obvious dream: train on short sequences, then at inference just hand the model longer ones. Use a small $L$ during training and a big $L_\text{valid}$ at test time. If that worked, I'd get long-context quality at short-context training cost.

Does it work? The recurrent models from before transformers were trained short and run long as a matter of course, and nobody thought twice. Vaswani et al. even floated the hope that the architecture "may extrapolate to sequence lengths longer than the ones encountered during training." So let me just check it. Take a standard transformer LM, train it at $L=512$, and evaluate it on the validation set at $512+k$ tokens, sweeping $k$ from a handful up to thousands. Plot perplexity against $k$.

What I see is discouraging. For very small $k$ — the first dozen or two tokens of extra context — perplexity improves a little, which makes sense: a few more context tokens help. Then it flattens almost immediately, and then it *climbs*. By the time I'm a few hundred tokens past $L$, the extra context is hurting rather than helping. The model doesn't gently plateau; it breaks when you give it more tokens than it trained on. So the casual assumption is false for this architecture as built. The dream of "train short, test long" is unusable unless I understand why.

So why? Let me think about what's actually different between a position inside $[0,L)$ and a position past $L$. The transformer layers themselves don't care about length — the feedforward sublayer, the softmax, the embedding lookup, even the attention parameters, none of them have a size baked to $L$; they all map $n$ vectors to $n$ vectors for any $n$. The only place length enters is the thing I bolt on to tell the model about order: the position signal added to the token embeddings at the bottom. During training the model only ever sees that signal at positions $0$ through $L-1$. It learns, in its weights, how to interpret *those particular* signals. At position $L+100$, whatever the signal is, it's something the model never saw — a new vector, or a new combination of sinusoid phases, or (for learned embeddings) literally undefined. It's out of distribution. The model has no learned response there, so it does something arbitrary, and the prediction falls apart.

That reframes the whole problem. The failure isn't the transformer; it's the *position representation*. And I can test that claim cleanly: hold the architecture, the data, the random seed, the number of epochs — everything — fixed, and change *only* how position is injected. If extrapolation behavior changes, the position method is the culprit.

Let me walk the position methods I have on the table and ask, for each, what happens past $L$.

Learned absolute embeddings first, because they're the most popular in big models. I keep a trainable vector for each position $0..L-1$ and add it to the token there. Inside the training range this is great. But ask it for position $L$: there is no vector. None was ever created. The method has literally nothing to say about any position beyond $L$. So its extrapolation isn't "weak," it's *undefined* — zero. Dead end, and an instructive one: any scheme that *stores* a per-position thing can't extrapolate, because the store ends at $L$.

Sinusoidal embeddings next, and these are more interesting because they *should* be able to extrapolate. The signal at position $p$ is a fixed vector of sines and cosines of $p$ at a geometric ladder of frequencies — no parameters, and crucially it's a *function* of $p$, defined for every $p$, including $p>L$. So unlike the learned table, there's a value to hand the model at position $L+100$. The hope was always that this is why sinusoids would generalize. But my plot says otherwise: trained at $L$, it improves for maybe twenty extra tokens and then degrades just like before. Why, if the signal is defined everywhere? Because "defined" isn't "in distribution." The model learned to interpret the specific vectors that occur for $p<L$; past $L$ the sinusoids keep going into phase combinations the model never trained against. The value exists, but the model's learned response to it doesn't. Same out-of-distribution failure, just dressed up. So a non-learned, everywhere-defined function isn't sufficient on its own — it also has to be the *kind* of function whose behavior past $L$ is a natural continuation of what the model already learned, not a march into novel territory.

So a workable method has to continue cleanly past $L$, not just be defined there. The methods that do better than sinusoidal might show what that takes.

Rotary. Instead of adding a vector at the bottom, it rotates the query and key vectors by an angle proportional to their position, at every attention layer. A query at $i$ and a key at $j$ then meet through a relative rotation of $i-j$, so it's effectively a *relative* scheme. Two things jump out. It injects position at *every* layer, not just the first. And it touches only the queries and keys — never the values. Since the attention output is a weighted average of value vectors, and the values carry no position, the output of each attention sublayer has no explicit absolute-position component baked into it. When I run it past $L$, it does extrapolate better than sinusoidal — improving for maybe a hundred-plus extra tokens before degrading. So "relative, injected everywhere, kept out of the values" correlates with better extrapolation. But it's slower and heavier than sinusoidal because of the per-layer rotations. Better, not efficient.

The relative-bias method (the T5 variant) is the most telling. It adds *nothing* to the token embeddings. Instead, after I compute each query–key dot-product score, I add a learned scalar that depends only on the relative distance between the query and the key — same bias function reused across the network, one set per head. The distances are bucketed: each small distance gets its own learned bias, and all distances past some cutoff collapse into shared far-away buckets. That bucketing is suggestive for extrapolation — a distance I never saw at training falls into a bucket I already learned, so at least it's not undefined. And like rotary, position enters only through the query–key comparison, never the values. This one extrapolates best of the lot — perplexity keeps improving for several hundred extra tokens. So changing only the position method, with the transformer held byte-for-byte fixed, moves the breakage from a few dozen tokens to several hundred. That's the cleanest evidence I have that the ceiling lives in the position mechanism and not in the transformer: I changed nothing else and extrapolation moved.

But there's a sting. The relative-bias method is at least about twice as slow as sinusoidal in my setup, and it adds learned parameters and some memory. Think about what that does to my original plan. The whole point of "train short, extrapolate long" was to *save*: train at $L=512$ and then run at 1024, instead of paying to train at 1024. But if the position method that lets me extrapolate is itself 2× slower per step, then training the slow method at 512 costs about the same as training the cheap method at 1024 — I've handed back the savings at the door. Extrapolation that comes bundled with a 2× tax isn't extrapolation I can bank.

So now the goal is sharp and three-sided. I want a position method that (1) actually extrapolates to much longer inputs, like the relative-bias method does; (2) costs no more than sinusoidal — the cheapest thing I have; and (3) ideally has *no learned parameters at all*, so I don't overfit it to the training lengths and so it transfers across model sizes and datasets without retuning. Notice (3) is itself a hint about why things break: every parameter the position mechanism learns is a parameter it can tune to the specific positions $[0,L)$, i.e. a way to memorize the training-length regime and then fall off a cliff outside it. The learned absolute table is the extreme case. Fewer learned position parameters might mean less to overfit and a cleaner continuation past $L$.

Let me collect the clues into a wish. From the relative-bias and rotary successes: I want a *relative* method (only $i-j$ matters, so there's no absolute coordinate to run off the end of), injected at every layer through the query–key scores, and kept out of the values. From the parameter argument: I want it *non-learned*, a fixed function, so there's nothing to overfit to $[0,L)$. From the sinusoidal failure: the function's behavior past $L$ must be a smooth, natural *continuation* of its behavior inside $L$, not a wander into new shapes.

What is the simplest object that satisfies all of that? It acts on the query–key score for query $i$ and key $j$; it depends only on the relative distance $i-j$; it has no parameters; and it continues smoothly forever. The relative-bias method already showed me that *adding a scalar to the score before softmax* is enough to encode position. So I don't need the scalar to be *learned* — I just need it to be a fixed function of $i-j$. The minimal such function that's monotone and continues forever is a straight line in the distance. Let me try that and see whether it actually has the continuation property I'm after.

So for zero-indexed query position $i$ attending to key position $j$ (with $0\le j\le i$ under the causal mask), I leave the dot product alone and add a bias

$$\text{softmax}\!\big(\mathbf q_i \mathbf K_{\le i}^\top + m\cdot[-i,\,-(i-1),\,\dots,\,-2,\,-1,\,0]\big),$$

where the bracketed vector is just the negative distances $-(i-j)$ as $j$ runs from $0$ to $i$ — i.e. the key right under the query gets $0$, the one before it $-1$, two back $-2$, and so on — and $m$ is a fixed positive scalar I'll pin down in a moment. No position is added to the embeddings anywhere; the values are untouched; the only change is this one additive term on the scores, at every layer. It's relative by construction — the bias for the pair $(i,j)$ is $-m(i-j)$, a function of the gap alone.

Now I want to check this against the failure I'm trying to escape, and not just by hand-waving "it's defined everywhere." The real worry was that *defined* didn't mean *in distribution* for sinusoids. So let me make the in-distribution claim precise and actually test it. Softmax is invariant to adding a constant to every entry of a row: $\text{softmax}(s + c\mathbf 1) = \text{softmax}(s)$, because the constant cancels in the ratio $e^{s_j+c}/\sum_k e^{s_k+c}$. So only *differences* of scores within a row matter. For my bias, the difference between two keys $j$ and $j'$ in the same query row is $-m(i-j) - (-m(i-j')) = -m(j'-j)$ — the absolute position $i$ has dropped out entirely. Whatever query row I'm on, near the start of training or 3000 tokens into an inference document, the *shape* the bias contributes to the row is identical: a fixed slope $-m$ per step of distance. At a never-seen distance $d=3000$, the bias $-3000m$ is just a point further down the very same line $-md$ the model rode for every small $d$ in training. There's no new vector and no new phase; it's the forced continuation of a line, and the only thing the model has to interpret is the same per-step slope it always saw. That's the property the table and the sinusoids lacked, and now I can see *why* it holds rather than hoping it does: it's the slope, not the absolute level, that the model conditions on, and the slope never changes.

There's a bonus falling out of this I should pin down numerically, because I keep asserting "distant keys get suppressed" and I want to know how fast. With the dot products held flat for a moment, the unnormalized weight on a key at distance $d$ is $e^{-md}$, so stepping one token further back multiplies the weight by $e^{-m}$. Take $m=\tfrac12$: $e^{-1/2}\approx 0.607$, so each step back keeps about 61% of the previous weight — a clean geometric decay. Let me tabulate it: $d=0$ gives weight $1$; $d=2$ gives $e^{-1}\approx0.37$; $d=10$ gives $e^{-5}\approx 6.7\times10^{-3}$; $d=100$ gives $e^{-50}\approx 1.9\times10^{-22}$; and at $d=512$ — a full training length back — the weight is $e^{-256}\approx 6.6\times10^{-112}$. That last number is the whole point made concrete: when I extrapolate and hand the model far-away tokens it never trained against, those tokens don't arrive as a disruptive novelty, they arrive already crushed to numerically zero weight. Extra distance can't blow anything up; it can only add terms that are exponentially suppressed before they reach the softmax's normalizer. So the method has a built-in *recency bias* — each head, through its own $m$, effectively attends within a soft window — and this is a second, independent reason it survives past $L$: the far context the model never saw is exactly the context the bias mutes.

I should be honest about the limit of that argument, though. The suppression is only "crushing" once $md$ is large. For the *nearest* extra tokens — say a head with a gentle slope, where the new tokens are only a few steps further back than ones it saw — the down-weighting is mild, and the local distribution does shift a little when I lengthen the input. That's fine; mild is not breakage, and the heads with small $m$ are the ones meant to look broadly anyway. The claim I can stand behind is the strong-suppression-at-large-distance one, which is what protects against the catastrophic far tokens; the near tokens were never the danger.

Could the penalty be something other than linear? I could imagine $-m\,d^2$, or $-m\log d$, or the bucketed-step shape the relative-bias method uses. The quadratic over-punishes and would crush context too fast; the log barely separates far distances; the buckets reintroduce a cutoff and learned parameters — the very thing I'm trying to avoid. The line is the minimal choice that's monotone, parameter-free, and continues cleanly. Start with the simplest thing that could work, and only complicate it if it fails. Let me keep the line.

Now the one knob: the slope $m$, per head. First, should $m$ be learned? Tempting — let the model pick its own windows. But learning $m$ means tuning it on the training lengths, which is exactly how I'd overfit the position mechanism to $[0,L)$ and reintroduce the cliff. And indeed when I let the slopes be trainable, the extrapolation comes out weak (and it costs a few percent in training speed for the privilege). So fix the slopes before training. Parameter-free, as I wanted.

If $m$ is fixed, what values? One slope for all heads can't be right — then every head has the same effective window, and I lose the obvious value of having some heads look locally and some look broadly. So give each head its own $m$: a head with large $m$ has a sharp recency bias (a short window); a head with small $m$ has a gentle penalty (a long, near-uniform window). I want a *spread* of windows across the heads.

How to space them? Let me reason about what a slope does. A near-zero slope makes the penalty almost flat — that head attends nearly uniformly over the whole context, very long-range. A slope near $1$ makes the penalty steep — strongly local. So I want slopes spread across $(0,1)$. Spread how — linearly? If I space them linearly, say $0.9, 0.8, \dots, 0.1$, I get lots of heads with steep, local windows and very few with the small slopes that give the long-range behavior. But the long-range heads are precisely the scarce, valuable resource — and the difference between slope $0.9$ and $0.8$ is tiny (both are very local), while the gap between $0.02$ and $0.01$ is huge (one head reaches twice as far as the other). So I want the slopes *denser as they approach $0$* — fine resolution among the long-range heads, coarse among the short-range ones. That's a geometric spacing, not a linear one: each slope a fixed fraction of the previous.

A geometric sequence in $(0,1)$ with a common ratio $r<1$ does exactly this — the terms pile up near $0$. So for $n$ heads let the slopes be $m_h = r^h$ for $h=1,\dots,n$, geometric with ratio $r$. I just need to fix $r$. A clean, memorable choice: for the common case of $8$ heads, let the slopes run $\tfrac12, \tfrac14, \tfrac18, \dots, \tfrac1{256}$ — that is, $m_h=(1/2)^h$, starting at $1/2$ with ratio $1/2$, the last head having a barely-there slope of $1/256$ (very long range) and the first a firmly local $1/2$. Eight heads, eight nicely spread windows, dense toward the small-slope end.

Now generalize off $8$. I want the *shape* of that spread to be invariant to the head count — whether I have $16$ heads or $8$, the slopes should cover the same $(0,1)$ territory with the same near-$0$ density, just sampled more or less finely. The $8$-head choice has start $=$ ratio $= 2^{-1} = 2^{-8/8}$. So for the power-of-two head counts I actually use, I read it as: start and ratio both equal $2^{-8/n}$. Then

$$m_h = \big(2^{-8/n}\big)^{h} = 2^{-8h/n}, \qquad h=1,\dots,n.$$

Let me not just assert the invariance — let me actually compute both schedules and look. For $n=8$: $2^{-8/8}=1/2$, and the eight slopes come out $\tfrac12, \tfrac14, \tfrac18, \tfrac1{16}, \tfrac1{32}, \tfrac1{64}, \tfrac1{128}, \tfrac1{256}$ — the inverses are exactly $2,4,8,16,32,64,128,256$, so this is the clean halving ladder I wanted. For $n=16$: start and ratio are $2^{-8/16}=2^{-1/2}=1/\sqrt2\approx0.7071$, and the sixteen slopes come out $0.7071,\,0.5,\,0.3536,\,0.25,\,0.1768,\,0.125,\,\dots,\,0.0055,\,0.0039$. Now compare the two lists position by position. The $8$-head slopes $\tfrac12,\tfrac14,\tfrac18,\dots$ appear in the $16$-head list at exactly the even slots ($0.5$ is there, $0.25$ is there, $0.125$ is there, down to $0.0039=1/256$), and the *new* odd-slot entries are $0.7071, 0.3536, 0.1768, \dots$. Are those new entries the geometric means I claimed? The geometric mean of $0.5$ and $0.25$ is $\sqrt{0.125}=0.3536$ — yes, that's the odd-slot value sitting between them; the mean of $0.25$ and $0.125$ is $\sqrt{0.03125}=0.1768$ — yes again. So doubling the heads literally inserts the geometric midpoint between each consecutive pair of the old slopes and keeps the endpoints almost fixed (the only stretch is the new top entry $0.7071$ above the old $0.5$). That's the invariance I wanted, and now I've checked it holds rather than just hoping the formula does the right thing: the same spread, refined, not a different spread.

In code the power-of-two start is written as $2^{-2^{-(\log_2 n - 3)}}$, which looks opaque until I simplify the exponent: $-2^{-(\log_2 n-3)} = -2^{3-\log_2 n} = -8/n$. So it's the same $2^{-8/n}$, just expressed so it's easy to compute from $n$. For a head count that isn't a power of two, the pure geometric rule no longer gives the same clean interpolation structure, so the fallback is exact and separate: take the slopes of the nearest lower power of two, then append every other slope from the next power-of-two set until I have enough heads. That preserves the "in $(0,1)$, dense near $0$" shape without pretending the non-power-of-two case is a single geometric sequence.

Is the exact schedule load-bearing, or just a reasonable point in a good region? A short manual sweep over a handful of slope sets says the latter: the method is robust to the choice as long as the slopes live in $(0,1)$ and crowd toward $0$. Even drawing slopes randomly from an exponential distribution works in some cases (with higher variance). So I don't need to re-tune this per dataset or per model size — I fix it once, like the sinusoidal frequency ladder was fixed once and reused everywhere. That robustness is reassuring: it means the power-of-two geometric rule and its explicit fallback are not fragile magic numbers; they are a convenient way to hit the right region.

One more detail on the bias itself. The original score has a $1/\sqrt{d_k}$ scaling on the dot product to keep the softmax in a sane regime. Do I scale the bias too? No — the bias isn't a dot product of $d_k$ random-ish terms whose variance grows with $d_k$; it's a deliberate fixed penalty in score units. Scaling it by $1/\sqrt{d_k}$ would just rescale my slopes by a constant, which I can absorb into $m$ anyway. So leave the bias unscaled; it's added raw to the (already-scaled) scores.

Now, implementation — and here's where the efficiency promise has to actually cash out. In a causal LM I already add a mask to the query–key scores before softmax: an $L\times L$ matrix that is $0$ on and below the diagonal and $-\infty$ strictly above it, so query $i$ can't see keys to its right. My bias is *also* just an additive term on those same scores. So I don't add a new operation anywhere — I fold the linear bias into the mask. Build, per head, the matrix whose entry $(i,j)$ is $-m(i-j)$ for $j\le i$, leave $-\infty$ above the diagonal, and add that to the scores exactly where the plain causal mask used to go. Zero extra runtime: the addition was already happening.

There's a small efficiency nicety in how to build it, and because it relies on the softmax-shift argument I want to actually verify it rather than trust the algebra. Naively I'd construct the full per-$(i,j)$ difference matrix. But for an allowed key $j\le i$, the finite bias I want is

$$-m(i-j)=mj-mi.$$

For a fixed query row $i$, the $-mi$ part is a constant added to every allowed key in that row, and softmax kills row constants — so I should be able to skip materializing it. Concretely: build the base row pattern $m[0,1,\dots,L-1]$ for *every* query row (the same row broadcast down), and let the causal $-\infty$ triangle hide the future keys. On row $i$ the visible part is $m[0,1,\dots,i]$, whereas the bias I actually wanted on that row is $[-mi,-m(i-1),\dots,-m,0]$; the two differ by the constant $mi$ on every visible entry. The sign even looks flipped, which is exactly the kind of thing I'd get wrong if I only reasoned about it, so let me trace a tiny case. Take one head, $m=\tfrac12$, $L=3$, and pretend the dot products are all zero so the attention weights are purely the softmax of the bias. The "true" bias matrix (rows = query $i$, $-\infty$ above the diagonal) is

$$\begin{bmatrix} 0 & -\infty & -\infty \\ -0.5 & 0 & -\infty \\ -1.0 & -0.5 & 0 \end{bmatrix},$$

and the broadcast construction $m[0,1,2]$ down all rows, then the causal triangle, is

$$\begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0.5 & -\infty \\ 0 & 0.5 & 1.0 \end{bmatrix}.$$

These are visibly different matrices — the second isn't even monotone the right way per row. But take the softmax of each row. Row $2$ of the true matrix: $\text{softmax}(-1,-0.5,0)=(0.186,\,0.307,\,0.506)$. Row $2$ of the constructed matrix: $\text{softmax}(0,0.5,1.0)=(0.186,\,0.307,\,0.506)$ — the same triple, because the two rows differ only by the additive constant $1.0=mi$, which cancels. I checked every row this way and the two softmax outputs agree to the last digit (max difference $0$). So the broadcast construction is exact, not approximate, and it's a single outer product of the slope vector with `arange` — no per-position loop and no per-$(i,j)$ difference matrix.

The only cost at all is memory. The plain causal mask is $L\times L$ and shared across heads. My biased mask differs per head (different $m$), so it's $n\times L\times L$. That's a slightly larger buffer — on the order of tens of megabytes in these models — and that's the entire price. No parameters, no extra matmuls, a negligible memory bump. Compare that to the relative-bias method's 2× slowdown, and this is exactly the efficient extrapolation I was after.

At this point the code path is small: a function that produces the per-head slopes, and the construction that folds the linear penalty into the causal mask.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_slopes(n_heads):
    # For power-of-two head counts, m_h = 2^{-8h/n}, h = 1..n: a geometric
    # sequence in (0,1) with start = ratio = 2^{-8/n}. Fixed before training.
    def slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))   # = 2^{-8/n}
        ratio = start
        return [start * ratio ** i for i in range(n)]  # start^{i+1} = 2^{-8(i+1)/n}

    if math.log2(n_heads).is_integer():
        return slopes_power_of_2(n_heads)
    # For other head counts, keep the lower power-of-2 set and append every other
    # slope from the next power-of-2 set.
    closest = 2 ** math.floor(math.log2(n_heads))
    return (slopes_power_of_2(closest)
            + get_slopes(2 * closest)[0::2][: n_heads - closest])


def build_attn_mask(seq_len, n_heads, device=None):
    # Put the static linear score bias in the additive attention mask.
    slopes = torch.tensor(get_slopes(n_heads), device=device)          # (n_heads,)

    # Per-head pattern: m_h * [0, 1, ..., L-1], broadcast to all rows.
    # On query row i this equals the desired -m_h(i-j) bias plus the row constant m_h*i.
    bias = slopes[:, None, None] * torch.arange(seq_len, device=device)[None, None, :]
    bias = bias.expand(n_heads, seq_len, seq_len)                      # (n_heads, L, L)

    # Standard causal mask: -inf strictly above the diagonal, 0 on/below.
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), 1)

    # Future positions remain -inf; allowed positions carry the linear relative bias.
    return causal[None] + bias                                        # (n_heads, L, L)


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

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)    # bias is not scaled
        scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class LMModel(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.tok(tokens)  # no added position embedding
        mask = build_attn_mask(T, self.layers[0].attn.n_heads, device=x.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(self.ln_f(x))


def lm_loss(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```
