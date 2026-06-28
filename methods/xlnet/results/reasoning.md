OK, let me think this through from scratch. I have a mountain of unlabeled text and a handful of labeled examples per downstream task, and the recipe everyone agrees on is: pretrain a big network self-supervised on the raw text, then finetune. The only thing I actually get to choose is the *objective* I pretrain with. So I want to stare hard at the two objectives that work and figure out, precisely, what each one gets wrong — because the fix has to live in the gap between them.

Start with the autoregressive language model, the oldest and cleanest one. Language modeling is just density estimation for text: I want p(x) for a sequence x = (x_1, …, x_T). The chain rule gives me that exactly, with no assumptions — the only freedom is the order in which I unroll it. The conventional choice is the forward order,

  p(x) = Π_{t=1}^T p(x_t | x_{<t}),

and I train by maximizing the log-likelihood,

  max_θ Σ_{t=1}^T log p_θ(x_t | x_{<t}),

where each conditional is a softmax over the vocabulary against a context vector h_θ(x_{1:t−1}) produced by the network:

  p_θ(x_t | x_{<t}) = exp(h_θ(x_{1:t−1})ᵀ e(x_t)) / Σ_{x′} exp(h_θ(x_{1:t−1})ᵀ e(x′)).

What's good about this is everything that makes density estimation honest. It's the product rule, which holds universally — no independence assumption anywhere. There are no artificial tokens stuffed into the input; the model only ever sees real text. So pretraining inputs look exactly like finetuning inputs, no mismatch. And because it's a genuine language model, every improvement in language modeling — deeper nets, longer context — drops straight in.

But here's the wall: h_θ(x_{1:t−1}) only ever sees the *left* context. The representation of a token is built from what came before it, never after. And the downstream tasks I care about — does this span answer this question, does sentence A entail sentence B — need a representation of a token that depends on *both* sides. Reading "Thom Yorke is the singer of Radiohead" and answering "who is the singer of Radiohead," the representation of "Thom Yorke" has to depend on "Radiohead," which is to its right. A forward AR model structurally cannot build that. People have tried to patch this — train a backward LM too and concatenate the two directions' features, the way ELMo does — but those two models are trained independently and glued together only at the very top. There's no deep, joint reasoning over both sides; it's two one-eyed models taped together.

Now the other objective, the denoising autoencoder, the one that's currently winning on understanding tasks. The move is: corrupt the input by replacing about 15% of the tokens with a special [MASK] symbol, call the corrupted sequence x̂, and train the model to reconstruct the masked tokens x̄ from x̂:

  max_θ log p_θ(x̄ | x̂) ≈ Σ_{t=1}^T m_t · log p_θ(x_t | x̂),

with m_t = 1 marking masked positions and each conditional again a softmax, now against a hidden state H_θ(x̂)_t that — and this is the whole point — is allowed to attend to *both* sides of position t, because reconstruction isn't density estimation so there's no causal constraint. That bidirectionality is exactly why it beats AR pretraining on understanding.

But stare at that "≈" sign. It's not cosmetic. The true object is the joint p(x̄ | x̂) over *all* masked tokens at once. What the objective actually optimizes is the *product* Σ_t m_t log p(x_t | x̂), i.e. each masked token reconstructed on its own. So this objective bakes in an independence assumption: given the unmasked context, the masked tokens are treated as mutually independent. Take [New, York, is, a, city] and mask both "New" and "York." This objective trains p(New | is, a, city) and p(York | is, a, city), and that's it. It never once trains p(York | New) — the dependency *between* the two things it's predicting is structurally invisible to the loss. And multi-token names, agreement, all the high-order long-range stuff in language, live exactly in those between-target dependencies.

Two more problems with the denoising route. The [MASK] symbol is all over pretraining and never appears in any real downstream sentence, so the network spends its pretraining adapting to inputs it'll never see again — a pretrain–finetune discrepancy. You can try to soften it by sometimes replacing [MASK] with the real token, but that has to stay rare, because if the model frequently just sees the true token in the slot it's supposed to predict, the objective goes trivial — it copies. And it's not a language model at all, so the river of language-modeling progress doesn't flow into it.

So let me lay the two side by side. AR: honest product rule, no independence assumption, no fake symbols, density-estimation framework — but unidirectional. Denoising AE: bidirectional — but independence assumption, fake [MASK] symbols, and outside density estimation. I want a single objective with the *left* column of virtues and the bidirectionality from the right. Is there one?

The unidirectionality of AR is tied to one specific choice: the *forward* order x_1, x_2, …, x_T. Position t conditions on x_{<t} only because I decided to unroll the chain rule left to right. But the chain rule doesn't care about order — any permutation of the T positions gives an equally valid factorization of the same joint p(x). So what if I don't commit to one order, and instead keep AR's exact product form while averaging the log-likelihood over a *random factorization order*? Let Z_T be the set of all T! permutations of (1, …, T). For a permutation z, write z_t for its t-th element and z_{<t} for its first t−1 elements:

  max_θ  E_{z ∼ Z_T} [ Σ_{t=1}^T log p_θ(x_{z_t} | x_{z_{<t}}) ].

Every term here is still a single autoregressive conditional — product rule, no independence assumption, no [MASK], pure density estimation. I'm only changing *which* order the chain rule runs in, sharing one set of parameters θ across all orders.

Before I get excited, I should check the thing I'm actually buying with this, because the whole argument rests on it: does averaging over orders really let a position condition on tokens from *both* of its sides? Let me just enumerate. Take T = 3 with the sentence (New, York, city) at positions (1, 2, 3), and look at the target "York" at position 2. Under each of the 3! = 6 orders, position 2 conditions on exactly the positions that *precede* it in that order. Its left neighbor is position 1 (New), its right neighbor is position 3 (city):

  order (1,2,3): York's prefix = {1}        → sees left
  order (1,3,2): York's prefix = {1,3}      → sees both
  order (2,1,3): York's prefix = {}         → sees neither
  order (2,3,1): York's prefix = {}         → sees neither
  order (3,1,2): York's prefix = {1,3}      → sees both
  order (3,2,1): York's prefix = {3}        → sees right

Counting: York conditions on its left neighbor (position 1) in 3 of the 6 orders, and on its right neighbor (position 3) in 3 of the 6 — exactly symmetric. So across the order distribution, position 2's context is drawn from both sides equally, not just the left. That is the bidirectionality, recovered without breaking the AR framework: no single conditional is bidirectional, but one shared θ is trained to handle whichever side the order happens to expose. And the between-target dependency that the denoising AE drops falls out of the same count: in 3 of the 6 orders position 1 (New) precedes position 2 (York), so the term log p(York | New, …) is literally in the sum for half the orders — that's the New→York dependency, trained directly.

I have to be careful about one thing immediately, or I'll wreck the model. "Permutation" here must mean permuting the *factorization order*, not shuffling the actual tokens around in the sequence. If I physically scrambled the input, I'd destroy word order, and worse, at finetuning time the model only ever sees text in natural order — I'd reintroduce a pretrain–finetune mismatch through the back door. So the sequence stays in its natural order, the positional encodings stay attached to the original positions, and the "permutation" is realized purely as an *attention mask*: under order z, when I go to predict the token at the t-th slot of the order, I let it attend only to the positions z_{<t} that precede it in z. The Transformer doesn't care — which positions a query may attend to is entirely the mask's business, decoupled from the physical layout. This is also exactly why this has to be a Transformer and not a recurrent net: an RNN's context is welded to the left-to-right sequence order, so it can't cheaply realize an arbitrary factorization order, whereas attention masking can.

Let me feel out whether this is the same thing as the orderless density estimators — MADE, orderless NADE — that already train over many orders. Mechanically, yes, the "share one model across sampled orders via masking" trick is theirs, and it's reassuring that a single shared model *can* be trained that way. But their motivation is the opposite of mine: they want an *orderless* density estimator, an order-agnostic inductive bias, and they're position-unaware — for an MLP that slides toward treating the context as a bag of words. I emphatically do *not* want orderless. Word order is the whole game in language; "dog bites man" ≠ "man bites dog." I want to permute the *factorization order* while staying fully *order-aware* about positions. That's why I'm keeping positional encodings on the original positions and only masking. Same trick, opposite goal.

Good. Now let me actually try to *implement* the conditional p_θ(x_{z_t} | x_{z_{<t}}) with a standard Transformer.

The standard thing: run the masked Transformer over the context, get a hidden state, call it h_θ(x_{z_{<t}}), and softmax against the embeddings:

  p_θ(X_{z_t} = x | x_{z_{<t}}) = exp(e(x)ᵀ h_θ(x_{z_{<t}})) / Σ_{x′} exp(e(x′)ᵀ h_θ(x_{z_{<t}})).

Look at what h_θ depends on. It's a function of the context x_{z_{<t}}, and nothing else. In particular it does *not* depend on z_t — on *which position I'm about to predict.* That's fine in a fixed forward LM, because there z_t is always "the next position, t," determined by the prefix. But here the order is permuted, so the same prefix can be followed by different next-positions.

Let me make this concrete and see if it breaks. Take two permutations z^(1) and z^(2) that share the same prefix but predict different next positions:

  z_{<t}^{(1)} = z_{<t}^{(2)} = z_{<t},   but   z_t^{(1)} = i ≠ j = z_t^{(2)}.

Substitute each into the formula. In both cases the context is x_{z_{<t}}, so h_θ(x_{z_{<t}}) is the *same vector*, so:

  p_θ(X_i = x | x_{z_{<t}}) = exp(e(x)ᵀ h(x_{z_{<t}})) / Σ_{x′} … = p_θ(X_j = x | x_{z_{<t}}).

The model predicts the *identical distribution* for position i and position j. Let me put actual numbers on it so I'm sure this is a real failure and not a notational mirage. Suppose the shared context produces a hidden vector h = (0.7, −0.4), and I have a three-word vocabulary with embeddings e(w₁) = (1, 0), e(w₂) = (0, 1), e(w₃) = (0.5, 0.5). The logits are e(w)ᵀh = (0.7, −0.4, 0.15), and softmax gives the distribution (0.524, 0.174, 0.302). That is the prediction for target position i. But for target position j the context is the same, so h is the same vector, so the logits and the softmax are *bit-for-bit the same*: (0.524, 0.174, 0.302) again. Two different positions in the sentence, forced to receive the identical predicted distribution — yet the true word at position i and the true word at position j generally differ, so at most one of these can be right. The parameterization literally cannot represent the target. This isn't a subtle inefficiency — the standard softmax is *position-blind* about its target, and under permuted orders that blindness is fatal. (In the degenerate extreme of an empty prefix, every position collapses to the same marginal — a bag-of-words.) The fix has to make the representation aware of *which position it's predicting*.

So I replace h_θ(x_{z_{<t}}) with a new kind of representation that *also takes the target position z_t as an input*:

  p_θ(X_{z_t} = x | x_{z_{<t}}) = exp(e(x)ᵀ g_θ(x_{z_{<t}}, z_t)) / Σ_{x′} exp(e(x′)ᵀ g_θ(x_{z_{<t}}, z_t)).

Now positions i and j get different g's because z_t differs, so the distributions can differ. Good — but now I have to actually *build* g_θ(x_{z_{<t}}, z_t), and that's where the next wall is.

The natural idea: "stand at" the target position z_t and let that position issue a query that gathers information from the context x_{z_{<t}} through attention. So g is the output of an attention operation whose query is keyed by the position z_t and whose keys/values come from the context tokens. Fine. But now I have to separate two jobs that a standard Transformer normally collapses into one hidden state.

To predict the token x_{z_t}, the representation g_θ(x_{z_{<t}}, z_t) must use the *position* z_t but must absolutely *not* use the *content* x_{z_t} itself. If g could see the token sitting at z_t, then predicting that very token from a representation that already contains it is trivial — the objective collapses, the model just reads off the answer.

But this *same* position z_t will, for later steps in the order, serve as *context* for predicting some other token x_{z_j} with j > t. And to be useful context, the representation at z_t had *better* encode the content x_{z_t}; otherwise the later token is being predicted from a context that has forgotten what word is actually at z_t.

So the representation at a single position needs, simultaneously, to *exclude* its own content when it is the thing being predicted and *include* its own content when it is serving as context for something else. One vector cannot do both. In a standard Transformer there's exactly one hidden state per position, so a single stream cannot satisfy the two uses.

The way out is to stop insisting on one representation per position and keep two. The content representation is h_{z_t} = h_θ(x_{z_{≤t}}): it attends to the context *and to z_t's own content*, exactly like a normal Transformer hidden state, so it can serve as context for later predictions. The ≤ t matters because it includes itself. The query representation is g_{z_t} = g_θ(x_{z_{<t}}, z_t): it attends to the context x_{z_{<t}} and to the position z_t, but not to the content x_{z_t}, so it is the one I feed into the softmax to predict x_{z_t}. The < t matters because it excludes itself.

Two streams, sharing all parameters, updated in parallel. Let me write the per-layer update, schematically:

  g_{z_t}^{(m)} ← Attention(Q = g_{z_t}^{(m−1)}, KV = h_{z_{<t}}^{(m−1)})    — query stream: uses z_t but cannot see x_{z_t}
  h_{z_t}^{(m)} ← Attention(Q = h_{z_t}^{(m−1)}, KV = h_{z_{≤t}}^{(m−1)})    — content stream: uses both z_t and x_{z_t}

The keys and values of *both* streams come from the *content* representations h — that's what carries actual token information. The streams differ in two ways only: which query does the asking (g vs h), and the attention range (z_{<t}, strictly-before, for the query stream so it never includes its own content; z_{≤t}, up-to-and-including, for the content stream). That's the whole mechanism, and it exactly resolves the contradiction: g_{z_t} pulls content from the *earlier* positions and from its own *position* index but never its own *token*, while h_{z_t} is a perfectly ordinary self-attention state that does include itself and is therefore available as context downstream.

Initialization. The content stream is just word embeddings, h_i^{(0)} = e(x_i), like any Transformer. The query stream can't be seeded with the word embedding (that would leak the content), so I seed it with a single shared trainable vector, g_i^{(0)} = w — a generic "I am a slot to be predicted" vector, identical for every position. Where does the position information come from, then, if every g starts from the same w? From the relative attention itself — the positional encodings enter through attention, so g_{z_t} acquires its target-position identity by attending *from* position z_t. The seed doesn't need to carry it.

There's a question I need to settle before going further: what does this two-stream apparatus cost me at finetuning time? Downstream tasks feed natural text and want one representation per token; if my pretrained model can only run in this strange two-headed mode, I've bought bidirectionality at the price of a new pretrain–finetune gap, which is exactly the disease I'm trying to cure. So let me actually trace the content-stream update and ask whether it reduces to something a vanilla Transformer would produce.

Write the content update out as a function of its inputs: h_{z_t}^{(m)} is computed from a query projected off h_{z_t}^{(m−1)}, keys and values projected off the content states h_{z_{≤t}}^{(m−1)}, the relative-position keys, and the attention mask — then a residual, a LayerNorm, and the FFN. Now compare term by term to a single-stream relative-attention layer: query from the hidden state, keys/values from the hidden states it may attend to, relative-position term, residual, LayerNorm, FFN. These are the *same computation*. The only place the permutation objective shows up in the content stream is the entry of the attention mask that decides the visible set z_{≤t}. So the difference between "content stream" and "ordinary Transformer layer" is one argument — the mask — and nothing in the parameter set or the wiring. At finetuning there is no sampled permutation: I hand the layer the ordinary task mask (full bidirectional, or whatever the head wants) and it *is* an ordinary Transformer. The query stream contributes no parameters of its own except the single seed vector w and reuses the content stream's projection weights, so dropping it instantiates nothing new and removes nothing the finetuning model needs. So the two-stream machinery exists only to make the pretraining objective well-posed; once the mask is the trivial one, it collapses to a standard backbone, and finetuning sees no artifact the pretraining introduced. That was the property I most needed to hold, and it holds by construction rather than by hope.

And to actually predict, I take the *last-layer query* representation g_{z_t}^{(M)} and plug it into the softmax:

  p_θ(X_{z_t} = x | x_{z_{<t}}) = exp(e(x)ᵀ g_{z_t}^{(M)}) / Σ_{x′} exp(e(x′)ᵀ g_{z_t}^{(M)}).

Let me go back to the tiny example that broke and check this actually undoes the position-blindness, rather than just hoping it does. Before, both target positions saw the same context vector h = (0.7, −0.4) and so got the identical distribution (0.524, 0.174, 0.302). Now each target builds g by attending *from its own position*, so the relative-position term added into the attention differs between the two targets. Model that minimally: g for target position i picks up a position contribution (0.3, 0), g for target position j picks up (0, 0.3), on top of the shared context. Then

  g_i = (0.7, −0.4) + (0.3, 0) = (1.0, −0.4),   logits e(w)ᵀg_i = (1.0, −0.4, 0.3),  softmax (0.574, 0.141, 0.285),
  g_j = (0.7, −0.4) + (0, 0.3) = (0.7, −0.1),   logits e(w)ᵀg_j = (0.7, −0.1, 0.3),  softmax (0.472, 0.212, 0.316).

The two distributions are now genuinely different — (0.574, 0.141, 0.285) vs (0.472, 0.212, 0.316) — so the parameterization can finally assign different predictions to position i and position j. Whatever the position contributions actually end up being after training, the point is they are no longer forced to coincide: making g depend on z_t is exactly what restores the model's ability to tell the two targets apart.

Let me push on optimization now, because I suspect the full objective is going to be brutal to train. In the full permutation objective every position in the order gets predicted, including the very first one, which is predicted from *nothing*, and the second, from one token, and so on — a lot of predictions conditioned on almost no context. With one shared θ asked to be sensible under all T! orders, that's a punishingly high-variance, slow-converging problem; I'd expect the loss to crawl. Let me ease it. The predictions that are actually informative — that have enough context to be learnable — are the *late* ones in the order, where z_{<t} is large. So instead of predicting every position, I'll only predict the *last* few in the sampled order. Split z at a cutpoint c into a non-target prefix z_{≤c} and a target tail z_{>c}, and only predict the tail:

  max_θ  E_{z ∼ Z_T} [ log p_θ(x_{z_{>c}} | x_{z_{≤c}}) ]  =  E_{z ∼ Z_T} [ Σ_{t=c+1}^{|z|} log p_θ(x_{z_t} | x_{z_{<t}}) ].

The tail is exactly the set of tokens with the longest available context under the current order, so I'm keeping the informative predictions and dropping the starved ones. Pick c so that roughly 1/K of the tokens are predicted — about a sixth in practice. And there's a nice efficiency dividend: for the non-target tokens z_{≤c} I never call the softmax, so I only need their content stream to serve as context. If I compact the computation to the selected prediction slots, I save query-stream compute and memory.

Worth pausing on how this lines up with the denoising AE, because superficially "only predict a subset" sounds like masking. Both end up doing partial prediction. For the AE it's a *necessity* — if it masked everything there'd be no context left to reconstruct from. For me it's an *optimization* choice — I could predict everything in principle. But the deep difference survives: even restricted to the same target set, when "New" precedes "York" in my order, I train log p(York | New, is, a, city), which contains the New→York dependency the AE's product factorization throws away. Formally: let T be the targets and N = x \ T the non-targets. The AE optimizes Σ_{x∈T} log p(x | N) — every target conditioned on the non-targets only. I optimize Σ_{x∈T} log p(x | N ∪ T_{<x}), where T_{<x} is the set of targets that precede x in the sampled order — every target conditioned on the non-targets *and* on the earlier targets. So for any target–context dependency (x, U): if U ⊆ N, both of us capture it; but if U reaches into the targets, U ⊆ N ∪ T_{<x} with U ∩ T_{<x} ≠ ∅, then only I capture it. Whenever the target set contains such earlier-target dependencies, I get a strictly denser training signal from the same targets. (And against a plain forward AR model the same lens shows its limit: it only ever covers (x, U) with U among the tokens *before x in the original order*; anything to x's right, like (New, {York}), it can never cover, while I cover it in expectation over orders.)

Now I want to fold in the recurrence and relative-position machinery from the state-of-the-art AR language model, because my objective is in the AR framework and I'd be foolish not to inherit longer context for free.

Relative positional encoding first, because it is the cleanest way to make this design reusable across segments. I've already committed to keeping tokens in natural order and realizing the permutation purely through attention masking, with positional encodings tied to the *original* positions. If I used absolute position embeddings added to the inputs, the recurrence below would break (position-1 of segment two would collide with position-1 of segment one), and more fundamentally the "order" the model perceives should come from the *mask*, not from baked-in absolute coordinates. The relative scheme gives me exactly that: the attention logit between a query at actual position i and a key at actual position j depends on position only through the relative distance i−j. Concretely the logit decomposes into a content term and a relative-position term, each with its own global bias:

  A_{ij} = (q_i + u)ᵀ k_j  +  (q_i + v)ᵀ W_R r_{i−j},

where q_i, k_j are the content query/key, r_{i−j} the sinusoidal relative-distance embedding, W_R its projection, and u, v two learnable global bias vectors. The first term is content-to-content; the second, content-to-relative-position. This is applied in *both* streams, with distances computed from the actual sequence indices i−j — or z_t−z_s when I am writing the same pair in factorization-order notation — never from the ranks t−s in the sampled order. That keeps the true geometry visible regardless of factorization order. (This is, again, what distinguishes me from the orderless density estimators: I am order-aware through these encodings; they are not.)

Now the segment recurrence. Take a long sequence and split it into two segments, x̃ = s_{1:T} and x = s_{T+1:2T}, with their own permutations z̃ and z. Process the first segment under z̃ and cache its per-layer *content* representations h̃^{(m)}. Then for the current segment, the content-stream update simply lets each query also attend to the cached memory, concatenated along the sequence dimension:

  h_{z_t}^{(m)} ← Attention(Q = h_{z_t}^{(m−1)}, KV = [ h̃^{(m−1)}, h_{z_{≤t}}^{(m−1)} ]),

and the query stream the same way with the cache prepended. The gradient does not flow into the cache; it is read-only memory.

I should pin down whether reusing this cache is actually safe, because there's a subtle hazard: the first segment was processed under its *own* random order z̃, and if the current segment's computation secretly depended on z̃, I'd have to re-process the first segment every time its order changed, which would defeat the whole point of caching. So let me check exactly which quantities the current-segment update reads out of the cache. It reads two things from h̃^{(m−1)}: the cached vectors themselves (used as keys and values), and the relative distance i−j between a current query at actual position i and a cached key at actual position j. The relative distance is a function of the actual sequence indices only — i and j are physical positions, fixed by where the tokens sit, not by any order. And the cached *vector* at position j: does it carry a fingerprint of z̃? Trace it back. h̃_j is itself the content state of position j in the previous segment, and I just established that the content update is ordinary self-attention whose only z̃-dependence is through *the previous segment's mask*, which has already been consumed — it determined which positions j attended to when h̃_j was computed, and that computation is finished and frozen. Once h̃_j is a concrete vector sitting in the cache, nothing downstream re-reads which order produced it; the current segment treats h̃ as plain content vectors at known actual positions. So the current-segment update is a function of (h̃, the current order z, actual positions) and *not* of z̃. That means I can cache h̃ once and reuse it under any order of the current segment, and across training the model is in effect learning to consume a memory that has been averaged over the previous segment's orders — which is what I want, and it costs nothing extra to get.

Two segments at finetuning need handling too. Following the standard two-sentence format I feed [CLS, A, SEP, B, SEP] and run permutation language modeling over the concatenation, reusing memory only within the same context. I don't add an absolute segment embedding to each position the way the denoising AE does; that's inconsistent with my whole relative philosophy and it can't extend past two segments. Instead I encode segments *relatively*: for a query at i attending to a key at j, I ask only whether i and j are in the *same* segment, using a learnable vector s_+ if so and s_− if not, and add a term a_{ij} = (q_i + b)ᵀ s_{ij} to the logit (b a learnable per-head bias). Only the same-segment-or-not relation matters, not which specific segment — that's the relative-encoding spirit, it generalizes better, and it lets me finetune on tasks with more than two segments. The token-level objective already couples the two segments through attention, so I do not add a separate next-sentence-prediction loss.

Let me write down the full per-layer computation with the real Transformer-XL machinery — multi-head relative attention, residual, layer norm, position-wise FFN — for both streams. Initialize h_t = e(x_t), g_t = w, and let h̃^{(m)} be the cached content memory from the previous segment. For layer m = 1, …, M and every t:

  ĥ_{z_t}^{(m)} = LayerNorm( h_{z_t}^{(m−1)} + RelAttn( h_{z_t}^{(m−1)}, [ h̃^{(m−1)}, h_{z_{≤t}}^{(m−1)} ] ) )
  h_{z_t}^{(m)} = LayerNorm( ĥ_{z_t}^{(m)} + PosFF( ĥ_{z_t}^{(m)} ) )
  ĝ_{z_t}^{(m)} = LayerNorm( g_{z_t}^{(m−1)} + RelAttn( g_{z_t}^{(m−1)}, [ h̃^{(m−1)}, h_{z_{<t}}^{(m−1)} ] ) )
  g_{z_t}^{(m)} = LayerNorm( ĝ_{z_t}^{(m)} + PosFF( ĝ_{z_t}^{(m)} ) )

The only difference between the two RelAttn calls is the key/value range: z_{≤t} for content (includes self), z_{<t} for query (excludes self). The prediction reads off the top query state:

  p_θ(X_{z_t} = x | x_{z_{<t}}) = exp(e(x)ᵀ g_{z_t}^{(M)}) / Σ_{x′} exp(e(x′)ᵀ g_{z_t}^{(M)}).

Now let me get to honest code, because there's a real implementation question hiding in "attend to z_{<t}": I am *not* going to recompute the attention from scratch for every position t under every order. Instead I realize the whole permutation as a single static attention mask over the natural-order sequence, and give the two streams two slightly different masks.

Assign each position its rank in the sampled order. Position i may attend to position j iff j precedes-or-equals i in the order for the content stream, or strictly precedes i for the query stream — equivalently, iff rank(j) ≤ rank(i) or rank(j) < rank(i). The content stream is allowed to see itself; the query stream is the content mask with the diagonal removed. In the code below, `perm_mask[i,j]=1` means i may *not* attend to j; I build it from a comparison of order-ranks, and the content mask is the same thing with the self-position re-allowed. With compact target mapping, only the selected prediction slots are materialized in the query stream; either way, `target_mask` is what gates the loss. The relative attention itself is the Transformer-XL `ac + bd (+ ef)` decomposition — content term plus relative-position term plus the relative-segment term. The loss is the tied-embedding cross-entropy of the target tokens against the top query states, averaged over the predicted positions.

```python
import tensorflow as tf

def embedding_lookup(x, n_token, d_embed, initializer):
    table = tf.get_variable('lookup_table', [n_token, d_embed], initializer=initializer)
    return tf.nn.embedding_lookup(table, x), table


def head_projection(h, d_model, n_head, d_head, initializer, name):
    w = tf.get_variable('{}/kernel'.format(name), [d_model, n_head, d_head],
                        dtype=h.dtype, initializer=initializer)
    return tf.einsum('ibh,hnd->ibnd', h, w)


def post_attention(h, attn_vec, d_model, n_head, d_head, dropout,
                   is_training, initializer, residual=True):
    proj_o = tf.get_variable('o/kernel', [d_model, n_head, d_head],
                             dtype=h.dtype, initializer=initializer)
    attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, proj_o)
    attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
    base = attn_out + h if residual else attn_out
    return tf.contrib.layers.layer_norm(base, begin_norm_axis=-1, scope='LayerNorm')


def rel_shift(x, klen=-1):
    s = tf.shape(x)
    x = tf.reshape(x, [s[1], s[0], s[2], s[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [s[0], s[1] - 1, s[2], s[3]])
    return tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])


def rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                  r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt, is_training, scale):
    ac = tf.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)        # content-to-content (A_ij term 1)
    bd = tf.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)        # content-to-rel-position (term 2)
    bd = rel_shift(bd, klen=tf.shape(ac)[1])                             # align relative offsets
    if seg_mat is None:                                                 # relative-segment term
        ef = 0
    else:
        ef = tf.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)   # (q_i + b)^T s_{ij}
        ef = tf.einsum('ijbs,ibns->ijbn', seg_mat, ef)                   # pick s_+ / s_- by same-segment
    attn_score = (ac + bd + ef) * scale                                 # scale = 1/sqrt(d_head)
    if attn_mask is not None:
        attn_score = attn_score - 1e30 * attn_mask                      # disallowed -> -inf
    attn_prob = tf.layers.dropout(tf.nn.softmax(attn_score, 1), dropatt, training=is_training)
    return tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)


def positionwise_ffn(inp, d_model, d_inner, dropout, kernel_initializer,
                     activation, is_training, reuse=None):
    with tf.variable_scope('ff', reuse=reuse):
        output = tf.layers.dense(inp, d_inner, activation=activation,
                                 kernel_initializer=kernel_initializer, name='layer_1')
        output = tf.layers.dropout(output, dropout, training=is_training, name='drop_1')
        output = tf.layers.dense(output, d_model,
                                 kernel_initializer=kernel_initializer, name='layer_2')
        output = tf.layers.dropout(output, dropout, training=is_training, name='drop_2')
        return tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1, scope='LayerNorm')


def cache_mem(curr_out, prev_mem, mem_len, reuse_len=None):
    if mem_len is None or mem_len == 0:
        return None
    if reuse_len is not None and reuse_len > 0:
        curr_out = curr_out[:reuse_len]
    if prev_mem is None:
        new_mem = curr_out[-mem_len:]
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]
    return tf.stop_gradient(new_mem)


def build_pretraining_inputs(inputs, targets, is_selected, perm_size, seq_len,
                             sep_id, cls_id, num_predict=None):
    index = tf.range(seq_len, dtype=tf.int64)
    index = tf.transpose(tf.reshape(index, [-1, perm_size]))
    index = tf.random_shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])

    non_func = tf.logical_not(tf.logical_or(tf.equal(inputs, sep_id), tf.equal(inputs, cls_id)))
    non_target = tf.logical_and(tf.logical_not(is_selected), non_func)
    target_or_func = tf.logical_not(non_target)

    rev_index = tf.where(non_target, -tf.ones([seq_len], tf.int64), index)
    target_tokens = tf.logical_and(target_or_func, non_func)
    target_mask = tf.cast(target_tokens, tf.float32)

    self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)
    perm_mask = tf.logical_and(self_rev_index[:, None] <= rev_index[None, :], target_or_func)
    perm_mask = tf.cast(perm_mask, tf.float32)

    new_targets = tf.concat([inputs[0:1], targets[:-1]], axis=0)
    features = {
        'perm_mask': perm_mask,
        'target': new_targets,
        'target_mask': target_mask,
        'input_k': inputs,
        'input_q': target_mask,
    }

    if num_predict is not None:
        indices = tf.boolean_mask(tf.range(seq_len, dtype=tf.int64), tf.cast(target_mask, tf.bool))
        actual = tf.shape(indices)[0]
        pad_len = num_predict - actual
        target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
        target_mapping = tf.concat([target_mapping, tf.zeros([pad_len, seq_len])], axis=0)
        mapped_target = tf.boolean_mask(new_targets, tf.cast(target_mask, tf.bool))
        mapped_target = tf.concat([mapped_target, tf.zeros([pad_len], dtype=mapped_target.dtype)], axis=0)
        mapped_mask = tf.concat([tf.ones([actual], tf.float32), tf.zeros([pad_len], tf.float32)], axis=0)
        features.update({'target_mapping': target_mapping, 'target': mapped_target, 'target_mask': mapped_mask})
    return features


def objective_attention_layer(h, g, r, mems, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed,
                              attn_mask_h, attn_mask_g, target_mapping,
                              d_model, n_head, d_head, dropout, dropatt,
                              is_training, kernel_initializer, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope, reuse=False):
        cat = tf.concat([mems, h], 0) if mems is not None and mems.shape.ndims > 1 else h
        k_head_h = head_projection(cat, d_model, n_head, d_head, kernel_initializer, 'k')
        v_head_h = head_projection(cat, d_model, n_head, d_head, kernel_initializer, 'v')
        k_head_r = head_projection(r, d_model, n_head, d_head, kernel_initializer, 'r')

        q_head_h = head_projection(h, d_model, n_head, d_head, kernel_initializer, 'q')
        attn_vec_h = rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                                   r_w_bias, r_r_bias, r_s_bias, attn_mask_h, dropatt, is_training, scale)
        output_h = post_attention(h, attn_vec_h, d_model, n_head, d_head, dropout, is_training, kernel_initializer)

    with tf.variable_scope(scope, reuse=True):
        q_head_g = head_projection(g, d_model, n_head, d_head, kernel_initializer, 'q')
        if target_mapping is not None:
            q_head_g = tf.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
            attn_vec_g = rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                                       r_w_bias, r_r_bias, r_s_bias, attn_mask_g, dropatt, is_training, scale)
            attn_vec_g = tf.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
        else:
            attn_vec_g = rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                                       r_w_bias, r_r_bias, r_s_bias, attn_mask_g, dropatt, is_training, scale)
        output_g = post_attention(g, attn_vec_g, d_model, n_head, d_head, dropout, is_training, kernel_initializer)
    return output_h, output_g


def pretraining_loss(hidden, target, target_mask, lookup_table, n_token, d_model,
                     initializer, use_tpu=False):
    softmax_w = lookup_table
    softmax_b = tf.get_variable('bias', [n_token], dtype=hidden.dtype,
                                initializer=tf.zeros_initializer())
    logits = tf.einsum('ibd,nd->ibn', hidden, softmax_w) + softmax_b
    if use_tpu:
        one_hot_target = tf.one_hot(target, n_token, dtype=logits.dtype)
        loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * one_hot_target, -1)
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
    return tf.reduce_sum(loss * target_mask) / tf.reduce_sum(target_mask)
```

Let me trace the whole chain once more to make sure it holds together. I started from the pain: AR pretraining is honest but one-eyed; denoising AE is two-eyed but lies about independence, smuggles in [MASK], and leaves the language-modeling framework. I kept AR's product form and made it bidirectional by averaging the log-likelihood over random factorization orders — no independence assumption, no fake symbols, still density estimation. Realizing the order as an attention mask (not a token shuffle) kept the natural sequence and the positional encodings intact, so finetuning sees no new artifact, and it is what makes a masked self-attention backbone the natural fit. Then a naive softmax turned out position-blind — two different next-positions sharing a prefix got identical predictions — which forced a target-position-aware representation g(x_{z_{<t}}, z_t). Building g exposed a contradiction: a position's representation must hide its own content to be predicted, yet expose it to serve as later context — irreconcilable in one stream, so two streams, a content stream (sees itself, becomes the ordinary finetuning model) and a query stream (sees position not content, feeds the softmax). Partial prediction tamed the optimization by predicting only the long-context tail, and skipping the query stream elsewhere paid for itself. Folding in Transformer-XL — relative positions that preserve actual geometry under mask-only permutation, plus segment recurrence whose memory is reusable because those positions are relative — extended the context, and a relative segment encoding handled multi-segment inputs while supporting more than two segments. The code is just that: the relative `ac + bd + ef` attention, two streams sharing content-derived keys/values but differing in query and mask, the permutation compiled into a static mask, and a cross-entropy over the predicted tail.
