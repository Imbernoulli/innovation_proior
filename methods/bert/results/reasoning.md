Let me start from what actually hurts. I have effectively unlimited unlabeled text and almost no labels for any individual task — entailment, paraphrase, named entities, question answering all share the same starvation. The recipe that's emerged is: learn something general from the raw text, then transfer it. And the thing everyone learns from raw text is a language model, because it's the one objective that needs no labels at all: predict the next word, maximize Σₜ log p(xₜ | x₁,…,x_{t−1}). Train that on a big corpus and the hidden state at position t becomes a contextual summary of everything up to t. That's the magic — a word's vector now depends on its sentence, not just its dictionary entry, which fixed embeddings like word2vec and GloVe could never do.

But sit with that objective for a second, because I think the objective is quietly dictating something I don't want. "Predict the next word" only makes sense if, when predicting word t, the model hasn't already seen word t. So the hidden state for position t is allowed to depend on x_{<t} and nothing to its right. The representation it hands me is left-context-only by construction. GPT does exactly this with a Transformer: it takes the all-to-all self-attention of the Transformer and clamps a causal mask onto it so position t can't attend to anything after t, trains the left-to-right LM, then fine-tunes the whole network on each task with one tiny head. Clean, and it's the strongest thing going on the sentence-level benchmarks. ELMo tries to recover the right side a different way: train a forward LSTM LM and, separately, a backward LSTM LM, then concatenate the two directions' hidden states and feed them as frozen features into a task-specific model.

Now think about what I actually need a representation to be good at. Take extractive question answering: I'm given a passage and I have to point at the span that answers the question. The representation of a candidate answer word is only worth as much as the context it encodes, and the disambiguating words sit on *both* sides of it. A left-to-right-only vector for that word has literally never looked right. That's not a small inefficiency; for token-level tasks it's the wrong information. So GPT's representations are handicapped exactly where I care most. And ELMo? It looks bidirectional, but stare at it: the forward LM's layer-ℓ unit at position t is a function of the left only; the backward LM's is a function of the right only; they never mix until I concatenate them at the end. No single hidden unit at any internal layer is ever jointly conditioned on both sides. It's bidirectional the way taping a left photo next to a right photo is a panorama — the seam is the whole representation. Plus the features are frozen, so I'm still hand-building an architecture per task.

So here's what I want, stated as a wish: a deep Transformer where, in *every* layer, each token attends to all the other tokens on both sides, and I fine-tune the whole thing. The Transformer is already happy to do this — self-attention is all-to-all by default; directionality isn't in the architecture, it's something the causal mask *adds*. So just... drop the mask? Let every position attend everywhere, and train the LM?

Let me try it and watch it break. I have a multi-layer encoder, no mask. Representation of position i at layer ℓ is a function of all positions at layer ℓ−1 — including position i itself. If I train the hidden state at position i to recover token i, the first unmasked layer already has a direct path from token i's input embedding into the state that is supposed to guess it. If I keep the next-token convention instead and train position i−1 to predict token i, position i−1 can attend directly to position i in that same first unmasked layer. And even if I tried to remove only that direct edge, a deeper unmasked stack would route the identity through neighboring positions that have seen it and then hand it back on the next layer. The target is sitting in the input, and the network has a path to copy it to the spot where it's being asked to guess it. Each word can see itself, directly or indirectly depending on the convention. The "prediction" becomes a lookup, the loss goes to zero, and the model learns nothing about context. That's why the causal mask exists in the first place — it's not a stylistic choice, it's the firewall that stops the target from entering any state used to predict it. And the firewall is exactly the thing that kills bidirectionality. Unidirectionality and the next-word objective are welded together: you can't keep the objective and unweld them.

So the obstacle isn't the architecture, it's the *objective*. "Predict the next token, from a representation of that token's context" is self-contradictory the moment the context is allowed to include the token. The fix has to break that coupling — I need the prediction targets and the bidirectional-context positions to be *different things*, so that a target is never inside the context that's used to predict it.

How do I make a target not be in its own context? Remove it from the input. Don't ask "what's the next word given everything before it"; ask "I've hidden this word — what was it, given everything around it?" If the token is physically gone from the input, then no matter how all-to-all the attention is, there is no path for it to leak into the representation at its own position, because it isn't anywhere in the sequence to leak. And every layer can be fully bidirectional, because the thing I'm protecting against — self-leakage of the target — is handled by *absence*, not by masking the attention. The leakage problem just evaporates. I corrupt the input by hiding a random subset of tokens, run the full bidirectional encoder on the corrupted sequence, and predict only the hidden tokens from their surviving neighbors. This is exactly the old Cloze task (Taylor 1953) — fill in the blank — repurposed as the pre-training objective. Call it a masked language model. So the bidirectional objective didn't *allow* the masked formulation, it *forced* it: masking the input is the only way I've found to have full bidirectional attention and a non-trivial token-prediction target at the same time.

Now I have knobs to set, and each one wants a reason. What fraction do I hide? If I mask too little — say 1% — then per sequence I only get a loss signal from a handful of positions, so I'm paying for a full forward pass and learning almost nothing from it; it'll converge painfully slowly. If I mask too much — say half — then the surrounding context I'm supposed to reconstruct from has been gutted; for many blanks there genuinely isn't enough left to infer the word, so I'm asking the impossible and the signal is noise. There's a sweet spot where most of the sentence survives as context but enough positions are queried to make each pass worthwhile. 15% sits in that band — roughly one in seven tokens hidden, context mostly intact. And I only compute the loss on those masked positions, not on the whole sequence. That's a real difference from a denoising autoencoder, which would reconstruct the *entire* input including the parts it could already see; I don't want to spend capacity re-copying visible tokens, I only want the gradient from the genuinely-hidden ones.

I should be honest that this 15% has a cost I just created. A standard left-to-right LM gets a prediction — and therefore a gradient — at essentially every non-initial token position. I get it at 15%. So per step I'm extracting maybe a seventh of the token-prediction signal, and I should expect to need more steps to converge. That's a real tax. But it's a tax on *training time*, not on the thing I'm trying to make possible: a representation whose internal layers are genuinely conditioned on both sides. I'll pay it.

Now a subtler problem, and it took me a second to see it. To mask a token I have to put *something* in its slot — a special `[MASK]` symbol. During pre-training, then, 15% of positions are this `[MASK]` token. But when I fine-tune on a real task, there are no blanks — the input is ordinary, complete text, and `[MASK]` never appears. So I've trained the encoder on an input distribution it will never see again. Every masked position at pre-training time carries this `[MASK]` marker, the model learns "ah, a `[MASK]` here means: time to predict," and at fine-tuning time that cue is simply absent. That's a train/test mismatch baked right into the objective.

I can't stop masking — masking is the whole trick. But I can stop the `[MASK]` symbol from being a perfect, reliable tattoo on every position I'm going to query. The idea: when I pick the 15% of positions to predict, don't always put `[MASK]` there. Most of the time, yes — replace with `[MASK]`. But some of the time, replace the chosen token with a *random* word instead, and some of the time leave the original token sitting there untouched, and still ask the model to predict it. Now walk through what the encoder experiences. A `[MASK]` position is obviously a query, but not every query looks like `[MASK]`: some look like plausible ordinary tokens and some look like wrong ordinary tokens. So the marker is no longer a complete description of where the loss will land. The model can no longer make all of its useful contextual work live only at `[MASK]` positions; the non-mask prediction cases push gradients through representations that look like normal fine-tuning inputs. That's strictly more than I asked for and exactly what I want a general representation to be.

The bulk has to stay `[MASK]`, because that's what actually creates blanks to predict — call it 80%. Of the remaining 20%, I want two distinct medicines. One is the "leave it unchanged" case: this is what keeps the model honest about observed words. If I *never* left the true token in place, the model would learn that the token sitting at a to-be-predicted position is always either a mask or a lie, so it should ignore the input token there entirely and lean only on neighbors — which biases every representation away from the word actually present. By occasionally leaving the real word and demanding the model still "predict" it, I pull the representation back toward the actual observed token, accepting that those few selected positions can copy the visible word because the distribution-mismatch fix is worth that small leakage. The other is the "random word" case: this is what stops the model from blindly trusting the input token — it forces it to use context to *check* whether the present token even makes sense. I want both, balanced, so split the leftover evenly: 10% random, 10% unchanged. And a sanity check on the random case — it's only 10% of 15%, so 1.5% of all tokens get corrupted into nonsense. That's small enough that it won't meaningfully damage the model's grip on the language; it's a gentle regularizer, not a wrecking ball. 80/10/10 it is. (I'd want to confirm later that this matters more for some uses than others — I suspect if anyone ever froze these features instead of fine-tuning, the `[MASK]`-only version would hurt them badly, because a frozen model can't paper over the mismatch by adjusting its weights, whereas fine-tuning can. The mix is the safer bet either way.)

Good — that's a bidirectional, deep, fine-tunable representation, pre-trained from raw text. But now I look at the tasks I actually have to serve and notice the objective is missing something. MLM is a *token-level* signal: it teaches the model about words in context. Yet a huge fraction of what I care about is about the relationship between two *sentences*. Natural language inference: does sentence B follow from sentence A? Question answering: does this passage contain the answer to this question? Paraphrase: do these two sentences mean the same thing? A language model — masked or not — never explicitly models the relationship between two segments; it only ever models a stream of tokens. So a representation trained purely on MLM has no particular reason to encode "do these two pieces of text belong together."

I want a second self-supervised task — still zero labels — that forces the model to reason about the join between two segments. The cheapest possible version: take two text spans A and B; half the time let B be the span that genuinely follows A in the corpus, half the time replace B with a random span from somewhere else; train the model to tell the two cases apart. "Is B the actual next segment, or not?" — a binary next-sentence-prediction task. It costs nothing to generate: every document hands me real A→B pairs for free, and the negatives are just random draws. And it directly injects the inter-segment signal that QA and NLI need. (There's prior work ranking the true next sentence against distractors — Jernite et al. 2017, Logeswaran & Lee 2018 — but they transfer only a sentence embedding; I'm going to transfer *all* the parameters, so the whole network learns to carry this.)

For this to work the model has to (a) ingest two segments at once and (b) emit a single fixed vector that summarizes the pair so I can do binary classification on it. Both push me toward a specific input format. I'll pack both segments into one token sequence so that self-attention runs over the concatenation — and notice that's a bonus: self-attention over A-concatenated-with-B is automatically full cross-attention between A and B at every layer, which is exactly what pair tasks like QA want, and it folds the old "encode each separately, then cross-attend" pattern into a single stage. To make the binary decision I need a designated readout slot, so prepend a special classification token — `[CLS]` — whose final hidden state I'll treat as the aggregate sequence representation and feed to the next-sentence classifier (and later to any sentence-level task head). And the model has to know where A ends and B begins, in two senses: I drop a separator token `[SEP]` between them, and — because attention itself is position-agnostic about *which segment* a token is in — I add a learned segment embedding, one vector for "this token is in A," another for "this token is in B," summed into every token. So the input vector at each position becomes the sum of three things: the token embedding (what the word is), the segment embedding (which of the two texts it's in), and a position embedding (where it sits). I'll use a learned position table rather than a fixed sinusoidal one — it's simpler and a hard 512-token cap is fine for these tasks. This one input format — `[CLS] A [SEP] B [SEP]` with summed token+segment+position embeddings — handles everything: a pair task fills both segments, a single-sentence task just leaves B empty (a text-∅ pair), so fine-tuning on any task becomes "swap the inputs and the output head, tune all the parameters."

A few remaining choices, each with a reason. Tokenization: if I use whole words I get an enormous vocabulary and I'm helpless against any word I didn't see in training. Subword (WordPiece) units with a ~30k vocabulary fix both — open vocabulary by composition, and a small fixed softmax. The architecture itself I'm not reinventing: it's the Transformer encoder, multi-head self-attention with the 1/√d_k scaling (so the dot products don't grow with dimension and saturate the softmax), feed-forward inner width 4× the hidden size, residual + layer norm — these are settled Transformer defaults and I have no reason to fight them. I'll use the GELU nonlinearity in the feed-forward rather than ReLU; it's a smoother gate that the recent Transformer LMs favor. And for the MLM output head, I don't want a fresh full-vocabulary projection matrix — that's a vocab×hidden slab of parameters duplicating information the input embedding table already holds. The map from a hidden vector back to a word is the inverse problem of the map from a word to its embedding, so I tie the output projection to the input embedding matrix (with a per-token output bias). That saves the parameters and regularizes. I will run one small non-linear transform — a dense layer plus GELU plus layer norm — on the masked positions before that tied projection, to give the model a chance to reshape the contextual vector into "vocabulary-prediction space," since the same hidden vector is also being asked to serve general representation duty.

For the data: I need long, genuinely contiguous text for both objectives — MLM wants real context and NSP wants real A→B adjacency — so a document-level corpus (books, encyclopedia text) is essential; a sentence-shuffled corpus would have no true "next sentence" and no long spans. I sample two spans summing to ≤512 tokens, choose 15% of token positions after subword tokenization, apply the 80-10-10 replacement rule to those chosen positions, and form the NSP label by the 50/50 real-vs-random draw. The total pre-training loss is just the two objectives added: the mean cross-entropy over selected MLM prediction positions plus the mean next-sentence binary cross-entropy. Optimize with Adam, learning rate warmed up then linearly decayed, modest weight decay and dropout. One efficiency note worth acting on: attention cost is quadratic in sequence length, so most positions' worth of learning happens just as well at a short length — I'll train the large majority of steps at length 128 and only the last stretch at the full 512, which is mainly needed to actually learn the high position embeddings. To keep the comparison against the left-to-right fine-tuning baseline honest, I'll size one configuration identically to it, so the *only* difference is bidirectional-vs-causal attention plus the two pre-training tasks — that isolates whether bidirectionality is really what's paying off.

Let me write it as code, mirroring how I'd actually build it. First the encoder and the input — three summed embeddings, a stack of ordinary bidirectional Transformer layers with no causal mask (only a padding mask), a pooled `[CLS]` readout:

```python
import torch, torch.nn as nn, torch.nn.functional as F

VOCAB, MAXLEN, H, L, A = 30000, 512, 768, 12, 12
FFN = 4 * H

class EncoderLayer(nn.Module):                      # the inherited Transformer block
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(H, A, dropout=0.1, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(H), nn.LayerNorm(H)
        self.ff = nn.Sequential(nn.Linear(H, FFN), nn.GELU(), nn.Linear(FFN, H))
        self.drop = nn.Dropout(0.1)
    def forward(self, x, key_padding_mask):
        # NO causal mask -> every position attends both ways: this IS the bidirectionality
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.drop(a))
        return self.ln2(x + self.drop(self.ff(x)))

class InputEmbedding(nn.Module):                    # token + segment + position, summed
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, H, padding_idx=0)
        self.seg = nn.Embedding(2, H)               # segment A vs B -> encodes the pair
        self.pos = nn.Embedding(MAXLEN, H)          # learned positions, not sinusoidal
        self.ln, self.drop = nn.LayerNorm(H), nn.Dropout(0.1)
    def forward(self, ids, seg_ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        e = self.tok(ids) + self.seg(seg_ids) + self.pos(pos)
        return self.drop(self.ln(e))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = InputEmbedding()
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(L))
        self.pooler = nn.Linear(H, H)               # [CLS] readout for pair/sentence tasks
    def forward(self, ids, seg_ids, pad_mask):
        x = self.embed(ids, seg_ids)
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)
        pooled = torch.tanh(self.pooler(x[:, 0]))   # final hidden of [CLS]
        return x, pooled                            # per-token states, pooled vector
```

Then the two heads and the joint pre-training loss. The MLM head reshapes the contextual states, projects through the *tied* embedding matrix, and scores cross-entropy only on the selected prediction slots; the NSP head is a binary classifier on the pooled `[CLS]`:

```python
class MaskedLMHead(nn.Module):
    def __init__(self, tok_embed):
        super().__init__()
        self.transform = nn.Linear(H, H)            # reshape into "predict-the-word" space
        self.act, self.ln = nn.GELU(), nn.LayerNorm(H)
        self.decoder = nn.Linear(H, VOCAB, bias=True)
        self.decoder.weight = tok_embed.weight      # tie output weights to input embeddings
    def forward(self, seq):
        h = self.ln(self.act(self.transform(seq)))
        return self.decoder(h)                       # logits over vocab at every position

class NextSentenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Linear(H, 2)                   # IsNext vs NotNext
    def forward(self, pooled):
        return self.cls(pooled)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mlm = MaskedLMHead(self.encoder.embed.tok)
        self.nsp = NextSentenceHead()
    def forward(self, ids, seg_ids, pad_mask):
        seq, pooled = self.encoder(ids, seg_ids, pad_mask)
        return self.mlm(seq), self.nsp(pooled)

def pretrain_loss(model, ids, seg_ids, pad_mask, mlm_labels, nsp_labels):
    mlm_logits, nsp_logits = model(ids, seg_ids, pad_mask)
    # mlm_labels has the true id at the 15% chosen positions and -100 elsewhere,
    # so cross-entropy contributes ONLY at the selected prediction positions
    mlm = F.cross_entropy(mlm_logits.reshape(-1, VOCAB), mlm_labels.reshape(-1),
                          ignore_index=-100)
    nsp = F.cross_entropy(nsp_logits, nsp_labels.reshape(-1))  # binary, on the [CLS] vector
    return mlm + nsp                                 # the joint objective is just the sum
```

And the data generator that enacts the 15% / 80-10-10 rule and the 50/50 next-sentence draw — this is where the whole leakage fix and the mismatch fix actually live:

```python
MASK_ID, MASK_PROB = 103, 0.15
def make_example(span_a, span_b_true, random_span, vocab_size, cls=101, sep=102):
    if torch.rand(1).item() < 0.5:                  # 50% real next, 50% random
        span_b, nsp = span_b_true, 0                # 0 = IsNext
    else:
        span_b, nsp = random_span, 1                # 1 = NotNext
    ids   = [cls] + span_a + [sep] + span_b + [sep]
    seg   = [0]*(len(span_a)+2) + [1]*(len(span_b)+1)   # which segment each token is in
    labels = [-100]*len(ids)
    candidates = [i for i, tok in enumerate(ids) if tok not in (cls, sep)]
    num_to_predict = min(len(candidates), max(1, int(round(len(ids) * MASK_PROB))))
    for j in torch.randperm(len(candidates)).tolist()[:num_to_predict]:
        i = candidates[j]
        labels[i] = ids[i]                          # remember the true token as target
        r = torch.rand(1).item()
        if   r < 0.8: ids[i] = MASK_ID                              # 80% -> [MASK]
        elif r < 0.9: ids[i] = torch.randint(0, vocab_size, (1,)).item()    # 10% random
        # else 10%: leave the real token in place, still predict it
    return ids, seg, labels, nsp
```

The causal chain, start to end: I want one representation per token that is conditioned on both sides at every layer, because that's what token- and pair-level understanding needs; the standard next-word objective can't give it, because dropping the causal mask lets each token see itself and the objective collapses; so I choose 15% of positions as token-prediction targets, remove most of those targets from the input, and compute loss only on the chosen positions; I split the chosen positions 80/10/10 so the never-seen-at-fine-tuning `[MASK]` symbol stops being a complete cue and the model must represent ordinary-looking tokens too; I add a next-sentence task on a pooled `[CLS]` slot so the representation also encodes inter-segment relationships that pure token modeling misses; I pack pairs into one sequence with token+segment+position embeddings so a single bidirectional Transformer encoder, fine-tuned end-to-end with one swapped head, serves every task.
