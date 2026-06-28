Let me start from what scaling actually buys, because the standard story — "bigger is better, keep stacking parameters and data" — hides a conflation I want to pull apart. When I scale a Transformer LM from 100M to 100B parameters, performance climbs predictably. But that one number is doing two unrelated jobs. Part of the gain is *more computation* — more layers, more attention, more nonlinear mixing per token, genuine reasoning capacity. The other part is *more memorization* — the bigger network simply stores more facts and phrasings from the training set in its weights, and we know large LMs memorize hard enough to regurgitate training data verbatim. Those two are fused in the parameter count, and serving the giant model pays the full compute cost on every token regardless of which job a given prediction needs. So the question that won't leave me alone: can I decouple them? Can I get the memorization benefit from somewhere cheap — an external database I look things up in — and spend parameters only on computation?

If that works, it's a semi-parametric model: a modest network for reasoning, plus direct access to a huge text store for knowledge. The knowledge becomes inspectable (I can read what was retrieved) and updatable (swap the database) instead of frozen in weights. And the win is only real if two things hold. First, the database has to be allowed to get *enormous* — trillions of tokens — because that's the regime where it could substitute for tens of billions of parameters; existing retrieval-LM work caps out at a few billion tokens, so this is uncharted. Second, and this is the binding constraint on the architecture, incorporating retrieved text must cost time *linear* in the amount retrieved. If the cost is quadratic in retrieved tokens, then the more I retrieve the worse my compute blows up, and I've just moved the expense rather than removing it.

Let me survey what retrieval-augmented LMs already do, because each one fails the two constraints in an instructive way, and the failures point at the design.

One family is the frozen-retriever interpolation methods — continuous cache, and kNN-LM. They retrieve, at inference, *tokens* whose stored activation looks like the current activation, build a distribution from those retrieved tokens, and *interpolate* it with the LM's own next-token distribution. kNN-LM scaled the store to all of Wikipedia and improved Wikitext103. The appealing part: the retriever is frozen, so it scales, and you can bolt it onto any model with no retraining. But the fatal part for me is that the network *never reasons over the retrieved text*. It only mixes two output probability vectors at the very end. The retrieved passage can't influence the model's internal computation, can't be combined with the context, can't be reasoned about — it's a probability nudge. And it works at the granularity of individual tokens, which is storage-heavy and misses the structure of a passage. SPALM tries to patch this with a gating network over the retrieved data, but most of the network is still untouched by retrieval. So this family scales but can't reason.

The other family is the end-to-end-trained dense retrievers — DPR trains query and key BERT encoders contrastively to align a question with its answer passages; REALM trains the retriever directly against the LM cross-entropy; RAG and FiD put an encoder-decoder reader on top of DPR and dominate QA. These *do* let the model reason over retrieved text. But REALM's end-to-end training has to search the database during training and *periodically re-embed and re-index the entire corpus* as the retriever's parameters drift — which is exactly what caps it at a few billion tokens. You cannot re-index a trillion-token database every few hundred steps. And the whole DPR/RAG/FiD/REALM line is built around *QA*: one question, one retrieval on the prompt, a short answer. None of them is designed to model *arbitrary long text* — a 2000-token sequence whose beginning might need one set of documents and whose end needs a completely different set. They retrieve once and read; I need to retrieve repeatedly, for different parts of the same sequence.

So the two failure modes pull in opposite directions: frozen retrieval scales but can't reason; trainable retrieval reasons but can't scale. The two are not actually in conflict once I notice *where* each constraint bites. Scale fails in the trainable family because *training the retriever* forces re-indexing — but reasoning is a property of what happens *after* retrieval, not of the retriever itself. So I can split the two concerns at the retrieval boundary: keep the retriever *frozen* — use a pre-trained BERT to embed text into keys, never train it — so the database can be precomputed once and grown to trillions of tokens, and put all the *trainable* machinery *downstream* of retrieval, where the network reasons over whatever was retrieved. The retrieval stays fixed and cheap; the reasoning lives entirely in the parametric model attending over retrieved content. I don't yet know this buys the full kNN-LM-style scaling without the REALM-style reasoning loss — that's a claim about the architecture I haven't built — but the failure analysis says the two constraints touch different parts of the pipeline, so there's no a-priori reason they can't both be satisfied. Let me follow that and see if it closes.

Granularity is the next pressure point. kNN-LM retrieves per token. For a trillion-token database that's a trillion keys and a trillion lookups — storage and compute explode by a huge linear factor. And per-token retrieval throws away the passage structure I want to reason over. So retrieve at the level of contiguous *chunks* of tokens. Split each training sequence (say n=2048 tokens) into l chunks of size m (say m=64, so l=32). For each chunk, retrieve its k nearest neighbours from the database. Now the number of retrievals per sequence is l, not n — a factor-of-m reduction in lookups — and each retrieved unit is a coherent span I can encode and reason about. The chunk size m=64 is a balance: small enough that a chunk is topically coherent (so its single BERT embedding is a meaningful query), large enough that l stays modest.

What exactly is stored in the database, and how do I query it? The database is a key-value memory. For each chunk N of the corpus, the *value* I store is two contiguous spans: the neighbour chunk N itself, and F, its *continuation* in the original document — the text that came right after N. I store [N, F]. The *key* is the frozen BERT embedding of N, mean-pooled over time, BERT(N). To retrieve for a query chunk C, I embed it the same way, BERT(C), and find approximate nearest neighbours by L2 distance d(C, N) = ‖BERT(C) − BERT(N)‖². The result Ret(C) is the k values ([N¹,F¹],…,[Nᵏ,Fᵏ]). Why store the continuation F and not just the matched chunk N? Because the matched chunk is what's *similar* to my current chunk, but what I actually want to predict is what comes *next* — and N's continuation F is a concrete example of "text that followed something like my current chunk." The matched chunk still gives alignment and lexical evidence, while the continuation gives the prediction signal, and keeping both only fixes the value length at two chunks. I'll set N and F each to length 64, so Ret(C) has shape k × r with r=128.

Two practical things. The retrieval has to be fast: an approximate-nearest-neighbour index (SCaNN) gives top-k in O(log T) for a T-element database, so even a 2-trillion-token store answers in ~10ms, amortized over a 64-token chunk — negligible. And on-the-fly retrieval during training is too slow to keep up with the GPUs/TPUs, but because the embedder is *frozen*, the neighbours of every chunk never change during training, so I precompute all nearest neighbours once and save them alongside the data. The frozen retriever isn't just a scaling trick — it's what makes precomputation valid.

I should write down the objective so I keep myself honest about causality. The retrieval-enhanced log-likelihood is

  L(X | θ, D) = Σ_{u=1}^{l} Σ_{i=1}^{m} ℓ_θ( x_{(u-1)m+i} | (x_j)_{j < (u-1)m+i}, (Ret(C_{u'}))_{u' < u} ).

Read that carefully, because the condition on retrieval is *u' < u*, strictly previous chunks. The i-th token of chunk u may depend on all earlier tokens (ordinary autoregression) and on the neighbours retrieved for *earlier* chunks only; for the first chunk, that collection of earlier retrievals is empty. Why this exact condition? Here's the trap I have to avoid. The neighbours of chunk C_u are retrieved using C_u itself, and they include continuations F that are "what tends to follow text like C_u." If I let the tokens *inside* C_u attend to Ret(C_u), I'm leaking information about C_u's own content and its likely continuation into the prediction of C_u's tokens — the model could cheat, and worse, sampling would be inconsistent because I'd need C_u's neighbours before I've generated C_u. So a token in chunk u may only use neighbours from chunks strictly before it. Ret(C_u) is computed after C_u is known and is first useful for predicting C_{u+1}. That keeps the whole thing autoregressive and samplable: to generate chunk C_{u+1}, I retrieve on the already-generated C_u, and condition. The added cost of retrieval during sampling is linear in the number of chunks, dwarfed by the usual quadratic token-sampling cost.

Once the neighbours exist, I have to fold them into a decoder. The natural differentiable tool is encoder-decoder cross-attention. So: a bidirectional Transformer *encoder* processes the retrieved neighbours into encoded representations E, and the *decoder* — the main LM — attends over E through cross-attention, interleaved with its usual self-attention and feed-forward. Putting cross-attention in every decoder layer would pay the retrieval path at every depth; I want periodic access points and let causal self-attention carry the injected information forward. I'll interleave a retrieval block every few layers — one every 3 blocks starting from layer 6 — among standard LM blocks. So a retrieval block is Retro(H, E) = FFW(CCA(Attn(H), E)) and a plain block is LM(H) = FFW(Attn(H)), where CCA is the new operator I have to define. As long as Attn is causal, FFW is positionwise, and CCA respects causality, any stack of these followed by a readout head defines exactly the autoregressive likelihood above.

One subtlety in the encoder that's easy to miss: I want the neighbour encoding to be *aware of the chunk that retrieved it*, so the encoder isn't blindly summarizing a passage but emphasizing what's relevant to my current context. So the neighbour encoder is *conditioned on H_u*, the decoder's activations for chunk C_u, via cross-attention inside the encoder. That makes the retrieval representation differentiably modulated by the retrieving chunk — the gradient flows from the LM loss back through the encoder's use of H_u, even though the *retriever* (which neighbours were selected) is frozen. All neighbours for all chunks are encoded in parallel into E ∈ ℝ^{l×k×r×d'}, with E_u the encoded neighbours for chunk u.

The place causality can break is chunked cross-attention, CCA. I have to wire "chunk u's tokens attend to E_{u-1}" in a way that (a) is causal, (b) costs time linear in retrieved data, and (c) handles the alignment correctly. Let me get the alignment right by thinking about *which decoder token is first allowed to see a chunk's neighbours*. Chunk C_u occupies token positions (u-1)m+1 … um. Its neighbours Ret(C_u) were retrieved using all of C_u, so they encode information about the *whole* of C_u — including its last token. Now, the decoder activation at a position is what *predicts the next* token; I keep tripping over this when I phrase it as "which token sees the neighbours," so let me phrase it as "which *prediction* may use them." Predicting token um — the last token of C_u — must not use Ret(C_u), because Ret(C_u) was computed from C_u including that very token; that's leakage. The first prediction that may legitimately use Ret(C_u) is the *next* token, position um+1, the first token of chunk C_{u+1}; and it stays legitimate for all of C_{u+1}.

That tells me the cross-attention input has to be shifted by one relative to the retrieval chunk. The *activation* at position um (the last token of C_u) is the one that produces the prediction for position um+1, so that activation is exactly the one I want to attend over E_u — even though "position um" is inside C_u. The output of that cross-attention is consumed one step later, predicting a C_{u+1} token, which is allowed. So I define *attending chunks* H_u⁺ = the activations at positions um, um+1, …, um+m−1 — the last token of C_u plus the first m−1 tokens of C_{u+1} — and have H_u⁺ cross-attend over E_u. Formally H_u⁺ = (h_{um+i−1})_{i∈[1,m]}, u from 1 to l−1.

I do not trust this off-by-one from prose alone — it is exactly the kind of boundary I get wrong — so before going further I'll trace the actual index plumbing on a toy case and read off which prediction touches which neighbour set. I defer that to the implementation check below; if the trace shows any prediction using its own chunk's neighbours, the shift is wrong and I'll have to redo it.

With that alignment fixed, CCA computes, per attending chunk, the cross-attention between H_u⁺ and E_u, and concatenates the results back across time with padding. For u from 1 to l−1,

  CCA(H, E)_{um+i−1} = CrossAttention(h_{um+i−1}, E_u),  for token i∈[1,m].

The first m−1 tokens of the whole sequence (positions 1…m−1, inside C₁) can't attend to any previous chunk's neighbours — there are none — so CCA is the identity there: CCA(H,E)_j = h_j for j ∈ [1, m−1]. And the very last token, position lm, attends to the last retrieval set E_l as the boundary case whose activation will predict the next token after the sequence window. Inside each chunk's cross-attention, I merge the neighbour and time dimensions of E_u (so k neighbours of r tokens become one attended sequence of length k·r) and attend over all of them at once — the decoder token attends across time and across neighbours simultaneously.

The basic cross-attention operator, stripped to one head, is: with parameter matrices K, Q ∈ ℝ^{d×c} and V ∈ ℝ^{d×d}, for a query h ∈ ℝ^d attending over Y ∈ ℝ^{T×d},

  CrossAttention(h, Y) = softmax(Y K Qᵀ h) Y V,

softmax over the T attended positions; in practice multi-head. There's a real alignment prior between a data chunk and its neighbours — they're roughly positionally aligned, both "starting at the same place" — so I use *relative* positional encodings in the cross-attention rather than absolute ones: the distance between data token i of C_u⁺ and retrieval token i′ of Ret(C_u)^j is d(i,i′) = i − i′ + l − 1 (and symmetrically d_enc(i′,i) = i′ − i in the encoder's conditioning attention), turned into positional logits added to the content logits. This lets the model learn "attend to the neighbour position aligned with my position," which is sensible given how the chunks line up.

Before I trace the indices, let me check the proposed design against the two constraints I started with, since if it fails the cost constraint the alignment won't matter. *Linear cost*: each decoder token attends only to its single preceding chunk's k·r retrieved tokens — a fixed amount, independent of sequence length — so the total cross-attention work is the number of query-key pairs n · (k·r). Let me put numbers on it to be sure "linear" really means what I want. With n=2048, m=64 (so l=32), k=2 neighbours of r=128 tokens each, the per-token cross-attention fan-out is k·r = 256, and the total is n·k·r = 2048·256 = 524,288 query-key pairs. Compare full causal self-attention, n² = 2048² = 4,194,304 — so the retrieval path adds only ~12.5% on top of self-attention here, and crucially, if I double the sequence to n=4096 the cross-attention pairs double (1,048,576) while self-attention *quadruples* (16.8M). The per-token fan-out k·r stays fixed at 256 regardless of n; that is the definition of linear, and it means I can raise k and grow the eval database without the cross-attention cost following n upward. *Causality without losing long-range retrieval*: each CCA attends only to E_{u-1}, the immediately preceding chunk's neighbours — but the *self-attention* layers propagate information forward across chunks, so the activations of a token in chunk u can depend on *all* previous chunks' neighbours Ret(C_{u'})_{u'<u}, even though no single cross-attention touches more than one chunk's neighbours. The dependency on all prior retrievals rides along the self-attention path, while I pay cross-attention cost for only one chunk's worth — that is what reconciles "linear cost" with "condition on everything retrieved so far."

Now the deferred check, because the padding is fiddly and it's where I promised to verify the off-by-one. The update is: take decoder activations of shape (n, d); drop the first m−1 positions; pad the tail by m−1; reshape into (l, m, d) — these are the attending chunks H_u⁺. Cross-attend each against its k neighbours of length r. Reshape back to (n, d), pad the front with m−1 zeros, truncate to n. Let me run this index plumbing on a toy so small I can read every position, n=6 with m=2 (so l=3): chunks C₁={1,2}, C₂={3,4}, C₃={5,6} in 1-indexed positions, and Ret(C_u) = E_u are the neighbours of chunk u. I tag each decoder position with its index and push it through drop→pad→reshape.

Dropping position 1 and tail-padding gives the rows of attending chunks (positions, 1-indexed): row for E₁ = [2, 3], row for E₂ = [4, 5], row for E₃ = [6, —pad—]. So the activation at position 2 (the *last* token of C₁) cross-attends over E₁; position 3 (first token of C₂) also attends E₁; positions 4,5 attend E₂; position 6 attends E₃; and position 1 was dropped, so after front-padding it gets the identity (no cross-attention update). Now the real question — does any *prediction* use its own chunk's neighbours? Activation at position p predicts token p+1, so I check, for each p, the chunk of E it attends versus the chunk of token p+1:

  pos 1 → identity, predicts a C₁ token using no neighbours — fine, the first m−1=1 positions must be identity.
  pos 2 → attends E₁, predicts token 3 ∈ C₂; chunk(E)=1 < 2 ✓
  pos 3 → attends E₁, predicts token 4 ∈ C₂; 1 < 2 ✓
  pos 4 → attends E₂, predicts token 5 ∈ C₃; 2 < 3 ✓
  pos 5 → attends E₂, predicts token 6 ∈ C₃; 2 < 3 ✓
  pos 6 → attends E₃, predicts token 7 ∈ C₄ (next window); 3 < 4 ✓

Every row has chunk(attended E) strictly less than the chunk of the token being predicted — no prediction ever uses its own chunk's neighbours, which is the leakage condition u′<u in the objective. And it confirms the boundary I was nervous about: position 2, the last token of C₁, *is* the first position to attend E₁, and that's legitimate precisely because its output predicts a C₂ token. The shift is right. The front-padding by m−1 leaves the first m−1 positions as the identity, matching the empty-previous-neighbour case for chunk C₁ in the likelihood. The off-by-one held up under the trace.

The rest of the model is a standard decoder-only Transformer with two minimal modernizations: RMSNorm instead of LayerNorm, and relative position encodings (Transformer-XL style) instead of absolute. The neighbour encoder is small and shared across all neighbours and chunks — d′=896, a couple of layers — adding only ~19M parameters; the relative extra cost shrinks as the base model grows (about +30% params at 132M but only +8% at 7B). Cross-attention sits in the decoder at, e.g., layers 6, 9, 12 for the smallest model, plus once in the encoder for the query-conditioning.

A frozen retriever has a hazard I should head off: a neighbour could come from the *same document* as the training sequence, in which case Ret(C_u) might literally contain C_{u+1} — a direct leak that breaks causality and lets the model copy the answer. So I filter out neighbours that originate from the same document as X. And because this whole setup retrieves directly from the training data, evaluation is treacherous: a retrieval LM can exploit leakage by copy-pasting a training chunk to "predict" a leaked evaluation chunk. So I need leakage-aware evaluation — for each evaluation chunk, find its closest training neighbours, measure the longest common token substring as a fraction r(C) ∈ [0,1] of how much of the chunk was already seen, and report bits-per-byte *filtered* to chunks with overlap below a threshold α. That separates genuine generalization from memorized copying, and it's the honest way to claim the gain isn't just leakage.

Finally, this retrieval pathway is general — it can inject *any* external source, not just the pretraining corpus. For open-domain QA I can fine-tune the pretrained retrieval model and feed it DPR's retrieved Wikipedia passages as the neighbours: format the input as "question: {q} \n answer: {a}", left-pad so "answer:" lands at the end of the first 64-token chunk (aligning the answer with the first retrieving position), and give the model the top-20 DPR passages through chunked cross-attention. The same architecture that does language modeling now does QA by swapping what's in the database.

The code lands in the same order: the vectorized chunked cross-attention update, the neighbour encoder conditioned on the retrieving chunk, and the interleaved decoder.

```python
import jax
import jax.numpy as jnp

Q = jnp.zeros((d, d))
K = jnp.zeros((d, d))
V = jnp.zeros((d, d))

def cross_attention(chunk, neighbour):
    queries = chunk @ Q
    keys = neighbour @ K
    logits = queries @ keys.T
    values = neighbour @ V
    return logits, values

def multi_neighbour_cross_attention(chunk, neighbours):
    logits, values = jnp.vectorize(cross_attention,
        signature="(m,d),(r,d)->(m,r),(r,d)")(chunk, neighbours)
    logits += relative_positional_encodings(m, r)[None, :, :]
    logits = jnp.moveaxis(logits, 0, -1).reshape((m, r * k))
    values = jnp.moveaxis(values, 0, 1).reshape((r * k, d))
    return jax.nn.softmax(logits) @ values

def chunked_cross_attention_update(observation, neighbours):
    attending_chunks = jnp.pad(observation[m - 1:],
                               ((0, m - 1), (0, 0)),
                               mode="constant").reshape(l, m, d)
    chunked_output = jnp.vectorize(multi_neighbour_cross_attention,
        signature="(m,d),(k,r,d)->(m,d)")(attending_chunks, neighbours)
    update = jnp.pad(chunked_output.reshape(n, d),
                     ((m - 1, 0), (0, 0)),
                     mode="constant")[:n]
    return update

def split_chunks(H):
    return H.reshape(l, m, H.shape[-1])

def encode_neighbours(retrieved_tokens, H):
    H_chunks = split_chunks(H)
    E = embed_enc(retrieved_tokens)
    for p_enc in range(1, L_enc + 1):
        E = bidirectional_attn(E)
        if p_enc in P_enc:
            E = encoder_cross_attn(E, H_chunks)
        E = encoder_ffw(E)
    return E

def forward(X, retrieved_tokens):
    H = embed(X)
    E = None
    for p in range(1, L + 1):
        H = causal_attn(H)
        if p == min(P):
            E = encode_neighbours(retrieved_tokens, H)
        if p in P:
            H = H + chunked_cross_attention_update(H, E)
        H = ffw(H)
    return readout(H)
```

The causal chain: parameter scaling fuses computation with memorization, so offload memorization to a frozen, precomputable, trillion-token retrieval database while keeping a trainable network to *reason* over what's retrieved — unlike kNN-LM, which only interpolates output probabilities, and unlike REALM/RAG, which must re-index and so can't scale; retrieve at the chunk level (about 64 tokens) to cut lookups by a linear factor and store each neighbour with its continuation; condition each chunk only on the neighbours of strictly-earlier chunks, which preserves autoregressivity and samplability; fuse via chunked cross-attention shifted by one token so the last token of a chunk is the first to see that chunk's neighbours, costing time linear in retrieved data while self-attention silently propagates dependence on all prior retrievals; and evaluate with leakage-filtered bits-per-byte so copying and general use of retrieved information are measured separately.
