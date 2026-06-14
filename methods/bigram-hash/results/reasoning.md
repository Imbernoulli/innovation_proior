Let me start from what actually hurts. I have a neural model that eats text, and the very first thing it does is turn each discrete token into a dense vector, because everything downstream is a continuous loss that needs gradients. The default way to do that is a lookup table: one trainable row per token, `E` of shape `K × d`, and the token `w` becomes `E[id(w)]`. Clean, exact, trains by backprop. But the cost is `K · d`, and `K` is the killer. For a real corpus the vocabulary runs to hundreds of thousands; the word2vec release carries three million entries at `d = 300`, which is close to a billion parameters sitting in the embedding alone. The embedding layer becomes the bulk of the model, and most of that bulk is dead weight, because of Zipf: a handful of tokens make up most of the text and the long tail of rare tokens each get their own `d`-vector that almost never receives a gradient. I'm paying full price for rows that barely move.

And it gets categorically worse the moment I want to embed more than single tokens. Bag-of-words throws away word order; adding `n`-gram features — at minimum bigrams, the pair of (previous token, current token) — puts cheap local order back, and that's known to buy a few points of accuracy. But the number of distinct bigrams is the square of the vocabulary. For a 50k vocabulary that's two and a half billion ordered pairs. A literal bigram embedding table is not large, it's unstorable. So whatever I do, the bigram feature forces me off the "one row per feature" model entirely. That's the real constraint, and it's the one I should design around: I need a representation whose parameter cost is bounded and roughly *independent* of how big the true feature space is, even when that space is `K^2` or `K^n`.

What tools do I have for "shrink a huge index space into a bounded one"? The obvious one is feature hashing — the hashing trick. Fix a hash function `h: {1,…,n} → {1,…,B}` and define the hashed map by summing every original coordinate that lands in a bucket, `phi_i(x) = sum_{j : h(j)=i} x_j`. For my case it specializes nicely: give the table `B` rows, and a feature `w` (a token, or a bigram) gets the row `E[h(w)]`. No dictionary — I compute the hash on the fly, so I never have to enumerate the feature space, which is exactly what I need for the `K^2` bigram space and for online settings where the vocabulary isn't even known in advance. Memory is `B · d` no matter how many distinct features exist; I just pick `B`. So `B` is now a memory knob I control instead of a number the data forces on me. This solves the storability problem outright.

But it buys that with collisions, and I should be honest about how bad they are rather than wave at "it works in practice." If a hash has range `K` and `K` is far below the number of distinct features, many distinct features hash to the same bucket and get *the same row* — the model literally cannot tell them apart. Let me put numbers on it with the birthday calculation, because I want to know exactly how the hash range trades against collisions. Take one feature `w_0`, hashed to some bucket. Any other feature avoids that bucket with probability `(K-1)/K = 1 - 1/K`. The `|T|-1` other features are independent, so the probability `w_0` collides with *nobody* is `(1 - 1/K)^{|T|-1}`, and therefore the collision probability is

```
p_col = 1 - (1 - 1/K)^{|T|-1}.
```

For large `K`, `(1 - 1/K)^{|T|-1} ≈ exp(-(|T|-1)/K) ≈ exp(-|T|/K)`, so `p_col ≈ 1 - exp(-|T|/K)`, and the expected number of features that are in *some* collision is `C_tot = |T| · p_col`. Now feel what that means for the bigram case. Even being generous and saying only `|T| = 10^8` distinct bigrams actually occur, with a million hash slots I get `p_col ≈ 1 - exp(-10^8/10^6) ≈ 1`: essentially every feature collides with something. To push collisions down I'd have to push the hash range up toward `|T|`, which throws away the entire memory win. So a single hash is a genuine wall: bounded memory and low collisions are in direct tension, and one hash function can't have both.

There's a subtler problem hiding in the *unsigned* sum, too, and it's worth pinning down because the fix for it will matter later. When several features collide in bucket `i`, the unsigned `phi_i = sum_{j:h(j)=i} x_j` adds their contributions all in the same direction. So a bucket that happens to catch several active features is systematically inflated — the hashed representation is *biased*, not just noisy. I need the collision terms to have no preferred sign, while the true feature's own contribution stays unchanged. The cheapest way to do that is to attach an independent random sign to each original feature, `xi: {1,…,n} → {±1}`, and use the signed map

```
phi_i^{(h,xi)}(x) = sum_{j : h(j)=i} xi(j) x_j .
```

Now the colliding contributions get random `±1` signs, so in expectation they should cancel rather than pile up, but I need to check that the real signal survives. The right object is the inner product a downstream layer consumes. Compute `⟨x, x'⟩_phi = sum_i phi_i(x) phi_i(x')`. Expand one bucket: `phi_i(x) phi_i(x') = sum_{j,l : h(j)=h(l)=i} xi(j) xi(l) x_j x_l'`. Sum over `i` and the constraint becomes `h(j) = h(l)`. Split into the diagonal `j = l` and the off-diagonal `j ≠ l`:

```
⟨x, x'⟩_phi = sum_j xi(j)^2 x_j x_j'  +  sum_{j ≠ l, h(j)=h(l)} xi(j) xi(l) x_j x_l'.
```

The first sum has `xi(j)^2 = 1` always, so it's exactly `sum_j x_j x_j' = ⟨x, x'⟩`, the true inner product. The second sum is the collision error, and there `j ≠ l` so `xi(j)` and `xi(l)` are independent `±1` with mean zero, giving `E[xi(j) xi(l)] = 0` — every off-diagonal term vanishes in expectation. So `E[⟨x, x'⟩_phi] = ⟨x, x'⟩`: the signed hash kernel is **unbiased**. The variance of that error sum scales like `1/B` (more buckets, fewer colliding pairs, smaller spread), so on unit vectors the distortion is `O(1/B)` and shrinks as I add buckets. That's a real, quantitative statement: the number of buckets enters only as a `1/B` variance term, which is the formal reason hashing into a much smaller space is usable at all. Good — the sign trick converts "collisions bias the representation" into "collisions add zero-mean `O(1/B)` noise," and noise I can live with where bias I can't.

But the sign trick fixes the *bias* of collisions, not the *fact* of them. Two genuinely different, genuinely important features still share one bucket and get one (now unbiased but shared) vector, and the model still can't separate them. And here's the part that really stings: the bucket assignment is a fixed hash with a discrete codomain. There is no gradient that can nudge an important feature out of a bad collision — `h` isn't differentiable, the assignment is frozen. I could try to *learn a better hash*, one where important features don't collide, but I can't optimize a function into `{1,…,B}` by gradient descent; the codomain is discrete and the loss is flat in the hash's parameters almost everywhere. That route is dead. So with a single fixed hash I am stuck: I either eat collisions among important features, or I grow `B` back toward the full feature count and lose the memory win. Wall.

Let me back up and look at what I actually want. I don't need to *learn the hash*. I need the *effect* of a learned hash — important features kept distinct, unimportant ones allowed to share — without ever differentiating through a discrete assignment. The first idea is the collision math itself. A single hash with range `B` collides badly when `|T| ≫ B`. But what if I use not one hash but `k` independent hash functions `h_1, …, h_k`, each into `{1,…,B}`, and let a feature be described by the *tuple* of buckets it lands in, `(h_1(w), …, h_k(w))`? Two features truly collide only if they agree on *all* `k` coordinates. The combination of `k` independent hashes into `B` buckets behaves like a single hash into a range of size `B^k`, because the joint bucket `(h_1(w), …, h_k(w))` takes `B^k` possible values. Re-run the birthday number with `B^k` in place of `B`: with `|T| = 10^8`, `B = 10^6`, going from `k = 1` to `k = 2` takes the per-feature collision probability from `1 - exp(-10^8 / 10^6) ≈ 1` down to `1 - exp(-10^8 / 10^{12}) ≈ 10^{-4}`. The total-collision rate falls off a cliff, and I paid for it not in `B^k` parameters but in a constant factor: `k` lookups per feature into the same shared pool of `B` rows. That's the asymmetry I was missing — `k` independent small hashes give me the collision resistance of an astronomically larger table at a cost that's only linear in `k`.

But "describe a feature by `k` buckets" is only useful if I can turn those `k` rows into one `d`-vector, and ideally in a way that recovers a per-feature trainable degree of freedom — that's the thing the fixed hash threw away. So pull `k` component vectors from the shared pool, `H_1(w), …, H_k(w)`, where each `H_i(w)` is the row of `E` selected by hash `h_i`, and combine them as a weighted sum:

```
ê_w = sum_{i=1}^{k} p_w^i  H_i(w),     p_w = (p_w^1, …, p_w^k) ∈ R^k.
```

Now stare at the `p_w`. These are `k` scalars *per feature*, trainable, and they're continuous — there's nothing discrete to differentiate through. They are the "importance" of each of the `k` hash choices for feature `w`, and they give me back, by gradient descent, exactly the lever the fixed hash denied me. If feature `w` and feature `w'` collide on hash `h_1` (so `H_1(w) = H_1(w')`) but not on the others, the model can still separate them: it learns different weights and uses the *other* component vectors to pull `ê_w` and `ê_{w'}` apart. Each feature now lives in the `k`-dimensional subspace spanned by its `k` component vectors, and `p_w` is a learnable point in that subspace — a feature is no longer one frozen row, it's a trainable combination of `k` shared rows. The collision that survives all `k` hashes is rare (the `B^k` birthday number), and the collisions on any single hash are repairable by the weights. That's the construction: `k` hashes for collision *resistance*, learnable per-feature weights for collision *repair*.

Let me get the bookkeeping exact, because I want to know the parameter cost. The component pool is a trainable matrix `E` of size `B × d`. The hash functions decompose as `H_i(w) = E[ D_{2,i}(D_1(w)) ]`: first a token-to-id step `D_1: T → {1,…,K}` (a dictionary if I'm happy to build one, or just another hash if the feature space is too big or dynamic to enumerate), then the `i`-th id-to-bucket hash `D_{2,i}: {1,…,K} → {1,…,B}`. The per-feature weights `p_w` are rows of a trainable matrix `P` of size `K × k`, indexed by the same id, `w → P[D_1(w)]`. So the total trainable count is

```
B · d  +  K · k,
```

against `K · d` for the standard table. Since I can take `k` small — `k = 2` turns out to be plenty, by the `B^2` collision math — the `K · k` term is tiny, and the win is choosing `B ≪ K` (a rule of thumb of `K > 10 · B` works) so that `B · d` is a fraction of `K · d`. Going from a bigram model to a general `n`-gram model, where a standard table would have to add `(K_n - K_2) · d` parameters as the bucket count grows, here costs only `k` extra weights per new id and a fixed `B · d` pool — orders of magnitude less. The combinatorial blowup that made the literal bigram table impossible is gone: I picked `B`, and the feature space can be as large as it likes.

And there's a regularization story falling out of this that I didn't put in by hand, which is usually a sign the construction is right. With a standard table I'd start from the full `K · d` parameter space and *push* unneeded parameters toward zero with an `L1`/`L2` penalty. Here the capacity is set by `B` up front — parameters the model doesn't need were never allocated. The weights `p_w` do something similar at the feature level: for an unimportant feature the model can drive `p_w → 0`, which both mutes that feature's contribution and *removes it from the effective collision set*, because a feature with zero weight no longer corrupts whatever bucket it shares. So the birthday calculation that matters is no longer over all `|T|` features but over the important subset `T_imp ⊂ T` whose weights stay nonzero, with an expected collision count on the order of `|T_imp| · (1 - exp(-|T_imp|/B))` for a single bucket hash, and smaller again when I use the `B^k` joint range. The importance weights are an implicit, learned feature selection — the effect of "learn a hash that avoids important collisions," obtained without ever learning a hash.

Now let me check this is actually a *generalization* and not just a third option, by recovering the things I came from as special cases. Set `k = 1` and freeze every weight to `p_w^1 = 1`. Then `ê_w = H_1(w) = E[h_1(w)]` — that's exactly the hashing trick, one hash, one row, no weights. Push further: take `B = |T|` and let `h_1` be the identity (a perfect dictionary with no collisions). Then every feature has its own row and `ê_w = E[id(w)]` — that's the standard embedding table. So both baselines sit inside this family: the hashing trick at `k=1` with frozen unit weights, the standard table additionally at `B = |T|` with the identity hash. I haven't invented a competitor to feature hashing and standard embeddings; I've found the parameterized object that has both as corners, and the interesting interior is small `k` with `B ≪ K`. That's the test passing.

One refinement I notice while writing the combination down: the vector of weights `p_w` is itself `k` numbers that describe the feature, so I can optionally *concatenate* it onto the output, `e_w = ê_w ⊕ p_w`, giving the downstream model direct access to the importance pattern at the cost of `k` extra dimensions. And the computational overhead of the whole thing is negligible: building `ê_w` is a `(1 × k)` times `(k × d)` multiply per feature, plus `k` lookups instead of one; for small `k` that's nothing next to the rest of the model, and since I skip dictionary construction and post-hoc pruning entirely, end to end it can even be a touch faster.

So that's the representation in its general form. Now let me specialize it to the case that motivated all of this — bigram features in a causal language model — because that's where it has to actually run, and the specialization tightens several of the free choices into concrete values. The setting: a GPT-style decoder with a learned token table `wte`, learned absolute positions `wpe`, and the input embedding tied to the output projection. A plain token embedding gives the *same* vector for a token no matter what preceded it; I want to inject the cheapest piece of left context, the (previous, current) token pair, straight into the representation. The feature space here is ordered bigrams, size `K^2 ≈ 2.5 × 10^9` for `K ≈ 50k` — exactly the unstorable case — so this is the hashing trick's home turf.

Pick the pieces. The component pool is a bigram embedding table; how big? This is the birthday tradeoff again. The number of *distinct* bigrams that actually occur in a corpus is far below `K^2` and is itself heavily Zipfian — a relatively small set of frequent bigrams carries most of the mass — so I don't need `B` near `K^2`, I need `B` large enough that the *frequent* bigrams rarely collide with each other. A table of `B = 5K` rows — five times the unigram vocabulary — is the working choice: it's a small constant multiple of the token table I already have, and with the occurring-bigram count well below `B` for the frequent ones, meaningful collisions are rare precisely because of the token distribution. So `bigram_vocab_size = 5 · vocab_size`, and I'll fold the "two-step `D_1` then `D_2`" into a single hash, since for a `K^2` space building a dictionary is pointless.

The hash itself has to be cheap and GPU-friendly — it runs on every position of every batch — so no modular-exponentiation niceties, just integer arithmetic the tensor engine likes. Take the current token id `c` and the previous token id `p`, both as int32. If I just XOR `c ^ p` directly, low-magnitude ids would collide in structured ways (XOR of two small numbers is small, so adjacent bigrams pile into low buckets). So first spread each id across the int32 range by multiplying by a large odd constant — different constants for the two positions so the pair is order-sensitive (`(c,p)` must hash differently from `(p,c)`) — then XOR, then reduce into the bucket range by a modulus:

```
index(c, p) = ( r1 · c  XOR  r2 · p )  mod  (B - 1),
```

with `r1, r2` large fixed integers (`r1 = 36313`, `r2 = 27191`). Multiplying by large constants smears the ids so the XOR mixes high bits, the modulus lands it in `[0, B-1)`. Position 0 of a sequence has no previous token, so I send it to a single reserved bucket, the index `B - 1` that the modulus never produces, so the "no-context" case gets its own clean slot instead of being faked from whatever id happened to precede the start.

Now, how do I combine and inject the bigram signal in the fixed language-model harness? The full construction would store a `K × k` matrix of per-feature weights and combine several component rows for each feature. This harness is narrower: it gives me one embedding module, the ordinary token and position stream, and a hook that can add an extra tensor before each Transformer block. So the practical specialization is to use one hashed bigram row per position and learn *where in depth* to trust it. I zero-initialize the bigram table and gate its injection with a small learnable scalar per layer. This is not the full per-feature `p_w` matrix; it is the harness-sized analogue of an importance weight, collapsed to the one exposed degree of freedom — how much bigram signal to mix in at each layer. Zero-init plus a small initial scalar (`0.1`) is the crucial safety property: at the start the bigram contribution is identically zero, so the augmentation cannot hurt the already-well-tuned token+position baseline; training then grows the table and the per-layer gates only where gradients ask for that signal. I keep the output projection tied to `wte` untouched — the bigram signal is an additive input/value-side intervention, so it stays a pure embedding-level change and doesn't disturb the tied head.

The signed-hash derivation still tells me something important about collision error: if I add an independent `±1` sign for each bigram, shared buckets become unbiased in expectation, with the diagonal giving the true inner product and the off-diagonal collision terms cancelling. If I can afford a second sign lookup, I multiply the looked-up bigram row by that sign pattern before injection. In the smaller harness I deliberately keep the simpler unsigned table; the collision-control budget is the `5 · vocab_size` table, the Zipfian active-bigram distribution, zero initialization, and the learned per-layer gates. The resulting module has no sign table by construction.

Let me write the embedding module so it drops into the fixed harness — same `forward(idx) → (B,T,d)`, same tied-head and position-count hooks, plus the existing per-layer injection hook:

```python
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token + position embedding, augmented with a hashed bigram embedding.

    The bigram (previous, current) token pair is mapped through the hashing trick
    into a table B = 5 * vocab_size rows; its looked-up vector is injected, gated by
    a learnable per-layer scalar, into every layer. The
    bigram table is zero-initialised so the augmentation starts as a no-op.
    """

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # learned token table
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # learned absolute positions
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer

        # component pool for the bigram feature: 5x vocab buckets to keep
        # frequent-bigram collisions rare (birthday math on the occurring bigrams)
        self.bigram_vocab_size = config.vocab_size * 5
        self.bigram_embed = nn.Embedding(self.bigram_vocab_size, config.n_embd)
        nn.init.zeros_(self.bigram_embed.weight)                    # start as a no-op residual

        # per-layer gate on the bigram injection; small init so the model
        # grows the bigram signal only where it helps
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_layer,), 0.1))
        self._cached_bigram = None

    def _bigram_hash(self, idx):
        # hash the (previous, current) token pair into [0, B-1):
        #   index = (r1*curr XOR r2*prev) mod (B-1)
        # large multipliers spread the ids across int32 before the XOR mixes them;
        # position 0 has no previous token -> reserved bucket B-1.
        rand_int_1 = 36313
        rand_int_2 = 27191
        mod = self.bigram_vocab_size - 1
        x = idx.to(torch.int32)
        out = torch.zeros_like(x)
        out[:, 0] = mod                                            # reserved "no previous token" slot
        out[:, 1:] = torch.bitwise_xor(
            rand_int_1 * x[:, 1:],                                 # r1 * current token
            rand_int_2 * x[:, :-1]                                 # r2 * previous token
        ) % mod
        return out.long()

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)                                    # (B, T, d)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)                                    # (T, d)
        # look up the hashed bigram vector once; injected per layer via get_value_embed
        self._cached_bigram = self.bigram_embed(self._bigram_hash(idx))   # (B, T, d)
        return self.drop(tok_emb + pos_emb)

    def get_value_embed(self, layer_idx):
        # inject the bigram signal at every layer, gated by its learnable scalar
        if self._cached_bigram is None or layer_idx >= self.n_layer:
            return None
        return self.bigram_lambdas[layer_idx] * self._cached_bigram

    def get_lm_head_weight(self):
        return self.wte.weight            # keep the tied output projection untouched

    def get_num_pos_params(self):
        return self.wpe.weight.numel()    # positions excluded from the parameter count
```

So the chain closes. I started stuck with a standard embedding table whose `K · d` cost grows with the vocabulary, wastes most of its rows on rare tokens by Zipf, and flatly cannot hold an `n`-gram feature space of size `K^n`. Feature hashing bounded the memory to `B · d` and removed the dictionary, but a single fixed hash forces collisions among important features and can't be trained out of them because the hash codomain is discrete. The sign hash fixed the *bias* of collisions — the signed kernel is unbiased with `O(1/B)` variance — but not the fact that distinct important features share a row. The move that broke the wall was to stop trying to learn a discrete hash and instead use `k` independent hashes for collision resistance (the `B^k` birthday cliff) plus a continuous per-feature weight vector `p_w` for collision repair, giving `ê_w = sum_i p_w^i H_i(w)` at a cost of `B · d + K · k` with both the hashing trick (`k=1`, unit weights) and the standard table (`B=|T|`, identity hash) as special cases, and an implicit feature-selection/regularization effect from the weights. Specialized to bigrams in a causal LM, the unstorable `K^2` pair space becomes a `B = 5K` hashed table, the pair is hashed by a cheap order-sensitive `(r1·curr XOR r2·prev) mod (B-1)` with a reserved no-context slot, and the full per-feature importance matrix becomes one learnable gate per layer over a zero-initialized table. The augmentation starts as a no-op, grows only where gradients ask for it, and drops in as a `TokenEmbedding` that leaves the tied output head alone.
