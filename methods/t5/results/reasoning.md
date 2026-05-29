OK, let me think this through from scratch. The thing that's bugging me isn't a single broken model — it's that the whole field of transfer learning for NLP has gotten *good* and *messy* at the same time. Pre-train a big network on a pile of unlabeled text, fine-tune on your task, win. Everyone agrees that works. But every new result changes three or four things at once. One paper swaps the objective from next-token prediction to masked reconstruction, and also swaps the architecture from a decoder to an encoder, and also swaps the corpus from Wikipedia to web crawl, and also uses a different fine-tuning recipe, and reports a number that beats last month's number. So which of those moves actually did the work? Nobody can say. I can't even line two of these methods up next to each other, because they don't share an interface — BERT hands me a classification head, GPT hands me a left-to-right decoder, a span model hands me a start/end pointer, a translation model hands me a full seq2seq decoder. You can't hold "everything else fixed" when "everything else" includes the shape of the output.

So before I can ask any clean scientific question — does objective X beat objective Y? does architecture A beat architecture B? — I need to remove the one thing that makes the comparison impossible. The culprit is the output interface. As long as classification needs a softmax over a fixed label set, span extraction needs a pointer into the input, tagging needs per-token logits, and generation needs a decoder, the loss and the architecture are welded to the task. I can never swap the architecture or the objective freely, because swapping it breaks the head, and the head is task-specific.

Stare at that for a second. What if there's no head at all? What if every task — every single one — maps a string of text to a string of text? Classification: don't emit a class index through a softmax-over-3-labels, just *emit the word* "entailment". Question answering: emit the answer text. Summarization, translation: emit the target text, that's already text-to-text. Even the awkward ones bend: STS-B is a regression, predict a similarity from 1 to 5 — fine, I notice the human annotations are almost all in increments of 0.2, so round to the nearest 0.2 and emit the *number as a literal string*, "3.8", and at test time parse it back; that quietly turns a regression into a 21-way classification, but I never had to special-case it because it's just text out. The only thing I need to tell the model is *which* task it's doing, and even that can be text: stick a short prefix on the front of the input — "translate English to German: ", "stsb sentence1: ... sentence2: ...", "summarize: ". The prefix is basically a hyperparameter; I'd bet the exact wording barely matters.

And the moment I do that, look what falls out. There is one loss for everything: teacher-forced maximum likelihood, plain cross-entropy over the output token sequence. The *same* loss for the unsupervised pre-training and for every downstream task. There is one decoding procedure: autoregressive generation, greedy at test time. Multi-task training is now trivial — it's just mixing examples from different datasets into the same batch, because they all look identical to the model. There are no per-task modules to bolt on or tear off. And — this is the part I actually wanted — objective, architecture, corpus, and fine-tuning recipe are now *independent knobs*. I can vary any one of them and leave the other three untouched, because none of them touches the interface anymore. The unification isn't a trick to win a benchmark; it's the thing that makes a controlled study even possible. That's the whole reason to do it.

I should be honest that other people have reached for unification before, and see exactly where they stopped short, because that tells me what to keep. McCann and colleagues cast everything as question answering — a single Q/A format for ten tasks. Good instinct, but they *require* all tasks to be trained simultaneously and they impose an explicit question-answer structure. I don't want to be forced into joint training, and I don't want a rigid format; a short text prefix and the freedom to fine-tune one task at a time is lighter. Radford and colleagues cast tasks as language modeling and probe them zero-shot — prime the model with a document and "TL;DR:" and let it continue into a summary. Lovely, but it's a decoder-only model evaluated without fine-tuning, and I'll argue in a moment that pure left-to-right is the wrong tool for conditioning on a context. And Keskar and colleagues unify tasks as span extraction — append the candidate answers to the input and point at the right span. That's clever for multiple-choice, but it dies on generative tasks: you cannot enumerate every possible German sentence, so translation and summarization don't fit. Text-to-text swallows extractive *and* generative in one interface. So I'll take the spirit of all three and go further: literally text in, text out, no enumeration, no mandatory multi-task, no fixed Q/A schema.

Now I have a clean testbed. Time to actually choose the pieces, and now I can choose them by experiment instead of by fashion. First question: what's the model's shape? The Transformer self-attention block is my Lego brick either way — each output position is a weighted average over positions, weights from query-key dot products, multiple heads. The structural choice is really a choice of *attention mask*. A fully-visible mask lets every position see every other (bidirectional). A causal mask forbids looking right, j > i gets weight zero — that's a left-to-right language model. And you can mix them: fully-visible over a prefix, causal after it.

So the candidates are: (1) the original encoder-decoder — bidirectional encoder over the input, causal decoder over the output, decoder cross-attends into the encoder; (2) a single decoder-only stack, causal throughout, with input and output concatenated, like GPT; (3) a single "prefix LM" stack, fully-visible over the input prefix, causal over the target, like the unified-LM line.

Which one, and why? Let me reason about what the task actually demands before I pick. In every text-to-text task the model gets a context — the input — and must produce an output that depends on *all* of that context. Take translation: the German word I emit can depend on any English word, including ones to its right. If I use a fully causal mask, the representation of input token i is only allowed to depend on tokens up to i. So by the time the model is generating output, the thing it attends back to is a representation of the input that never got to look rightward — it's needlessly hobbled. This is the same complaint people made years ago about using a *unidirectional* RNN encoder in seq2seq, and it was right then too. So I want the input processed *bidirectionally*. That immediately knocks out the pure decoder-only LM as the natural choice, and it favors either the encoder-decoder or the prefix LM, both of which see the input fully.

But I can't just assert it; I want to compare them fairly, and "fairly" is subtle. Suppose a BERT-base-sized stack has L layers and P parameters. An encoder-decoder with L encoder layers plus L decoder layers has about 2P parameters. So is it "twice as big" as a decoder-only LM with L layers? In parameters, yes. But in *compute*? No. The encoder's L layers only ever run over the input sequence, and the decoder's L layers only ever run over the output sequence. A decoder-only LM with L layers runs all L layers over the *concatenation* of input and output. So the L+L encoder-decoder does roughly the same number of FLOPs as the L-layer decoder-only model, even though it has twice the parameters. I actually checked the step times — an L-layer LM and an L+L encoder-decoder clock in nearly identical, which confirms the FLOP equivalence. (The encoder-decoder attention adds maybe 10% of parameters and there's a quadratic-in-length attention term, so this is approximate, but close enough that I'll treat an L+L encoder-decoder as having the parameter count of a 2L-layer LM and the compute of an L-layer LM.)

That means I *cannot* match an encoder-decoder to a decoder-only model on both parameters and FLOPs at once — pick one to hold fixed. I'll hold FLOPs fixed, M, and sweep the parameter budget so nobody gets an unfair compute advantage. The lineup: encoder-decoder (2P params, M FLOPs); the same but with parameters *shared* between encoder and decoder (P params, M FLOPs); a half-depth encoder-decoder, L/2 each (P params, M/2 FLOPs); a decoder-only LM (P, M); and a prefix LM, same stack but fully-visible over the input (P, M).

Run that sweep with the denoising objective and the result is clean: the encoder-decoder wins on every task. The shared-parameter encoder-decoder — same compute, half the parameters — comes in essentially tied, which is a genuinely nice surprise and lines up with the observation that you can share parameters across Transformer blocks without losing much (the cross-layer-sharing idea). Halving the depth instead, to buy back the parameters, *hurts* — so it's depth that matters, not raw parameter count, and sharing is the cheap way to keep depth. And the shared encoder-decoder beats the prefix LM, which tells me something I wouldn't have guessed: even though the prefix LM also sees the input bidirectionally, the *explicit cross-attention* of an encoder-decoder is itself worth something beyond bidirectionality. So: encoder-decoder, original-Transformer shape, possibly with shared parameters if I'm parameter-constrained. The same sweep also re-confirms denoising beats a plain LM objective across the board — but let me earn that properly rather than just citing it.

Now the objective, which is really the heart, because that's the channel through which all the "general knowledge" gets into the model. What's on the table? The old way is a causal language model — predict the next token. It's historic and it fits a decoder-only model naturally. The newer way is denoising / masked reconstruction — corrupt the input, predict what was corrupted, using bidirectional context. People keep finding denoising transfers better, but I want to see it in *my* controlled setup, not take it on faith. So compare three genuinely different recipes through the text-to-text interface: a prefix-LM objective (split a span of text, encode the first half, predict the second), a BERT-style denoising objective (corrupt 15% of tokens, predict the original), and a deshuffling objective (shuffle the tokens, predict the original order). The denoising one wins; the prefix-LM objective is close on translation but behind elsewhere; deshuffling is clearly worst. Reconstructing a shuffled sentence is apparently just a weaker signal than reconstructing masked content. Good — denoising it is. Now I get to *redesign* the denoising objective for my setting instead of inheriting BERT's exactly, because BERT's was built for an encoder that predicts at masked positions, and I have a generative decoder.

Let me start from BERT's masked LM and pull on each piece. BERT corrupts 15% of tokens; most selected tokens become a `[MASK]` token, some become random tokens, and in the original recipe a small slice is left unchanged; the encoder predicts the originals at those positions. The first thing that doesn't port cleanly: I don't have an encoder making per-position predictions; I have to *generate a target sequence*. The naive port is: feed the corrupted sequence in, and make the model generate the *entire original sequence* as the target. That works — but the target is as long as the input, which means the decoder is doing self-attention over a long sequence for every example, and that's slow. So already I have a pressure: make the target *short*.

Pull on the next piece: the random-token and unchanged-token heuristics. Why do those even exist in BERT? They're there because BERT's encoder often sees `[MASK]` during pre-training but never at fine-tuning, so the inputs are perturbed to soften that mismatch. But in my setup the target is *generated*, not read off an encoder position, so that particular mismatch isn't really my problem. Let me just test dropping those heuristics — pure masking, predict the original. No loss in quality. Drop them. (That variant is essentially the MASS-style masking, so I'm in known territory.) One knob simpler.

Now attack the long-target problem head-on, because making targets short is the cheapest speedup available and the quality differences among denoising variants are turning out to be tiny. The waste is obvious: if I reconstruct the *full* original text, the decoder spends most of its effort copying the uncorrupted majority of tokens, which carries no learning signal. So predict *only the corrupted tokens*. Two ways to do that. Option A: replace each *consecutive run* of corrupted tokens with a single unique sentinel token — `<X>`, `<Y>`, `<Z>`, new IDs added to the vocabulary that don't correspond to any real word piece — and make the target be the concatenation of the dropped spans, each prefixed by the sentinel that stands in for it, ending with a final sentinel to mark the end. So "Thank you for inviting me to your party last week ." with "for inviting" and "last" corrupted becomes input "Thank you `<X>` me to your party `<Y>` week ." and target "`<X>` for inviting `<Y>` last `<Z>`". Notice that because consecutive corrupted tokens collapse into a *single* sentinel, the input gets shorter too, not just the target. Option B: just delete the corrupted tokens from the input entirely and ask the model to reconstruct them in order — input "Thank you me to your party week .", target "for inviting last".

Compare A and B. They're about equal in quality. B (drop tokens) actually nudges GLUE up, traced to a jump on CoLA — which makes sense the moment I think about it: CoLA is grammatical-acceptability judgment, and a model trained to notice *that tokens are missing* is being trained on something close to detecting un-grammaticality. But B is worse on SuperGLUE, and A's sentinels give the model an explicit anchor for *where* each span was. I'll take A — replace corrupted spans with sentinels, predict only the spans — as the baseline. Both A and B shorten sequences and speed up training; that's the real win, and quality is a wash.

Next knob: the corruption rate. 15% is just BERT's number; my framework is different, so check it. Sweep 10%, 15%, 25%, 50%. Barely matters — until 50%, which degrades GLUE and SQuAD and also lengthens the targets (more corrupted tokens to emit). So keep 15%, both because it's as good as anything and because it keeps targets short. The pattern is becoming a theme: among denoising variants, downstream quality is remarkably flat, so the tiebreaker should be *computational cost*, i.e. target length.

One more, and it interacts with the sentinel trick. I've been deciding to corrupt each token independently, i.i.d. But i.i.d. corruption rarely produces long consecutive runs — so the "collapse a run into one sentinel" shortening doesn't fire very often, and my sequences aren't as short as they could be. If I instead deliberately corrupt *contiguous spans*, every span is one sentinel and the sequences get genuinely shorter. There's prior reason to like this beyond speed — span-level masking was found to help over token-level. I can parametrize a span objective by the corruption rate plus the number of spans, which fixes the average span length: 500 tokens at 15% with 25 spans means 75 corrupted tokens, average span length 3. Sweep average lengths 2, 3, 5, 10. Length 10 is slightly worse; length 3 is slightly but consistently better than i.i.d. on most non-translation tasks *and* gives a speedup from shorter sequences. So land on: corrupt contiguous spans, 15% of tokens, average span length 3, replace each span with a unique sentinel, predict only the spans. That's the objective.

Step back: the loudest finding from all of this is the *negative* one. The big gap is denoising-versus-LM; once I'm inside the family of denoising objectives, the variants barely differ in quality. Which is liberating — it means I should pick the denoising variant by how cheap it is to train (short targets), not chase a magic objective. The objective's whole job is "reconstruct corrupted text," and the exact flavor is second-order.

Now the architectural fine print of the Transformer itself, because a few standard pieces are worth re-deriving rather than copying. Take normalization. Standard LayerNorm re-centers (subtract the mean) and re-scales (divide by std) and applies a learned gain *and* bias. Do I need the re-centering? The observation that's been floating around is that the mean-subtraction contributes little; what's doing the stabilizing work is the *scale* normalization. So drop the mean-subtraction and the bias entirely: normalize by the root-mean-square only, `x / sqrt(mean(x^2) + eps)`, times a learned per-feature scale. Cheaper, fewer parameters, and just as stable in practice. While I'm at it, where do I put the norm? The original puts it *after* the sublayer, inside the residual sum, which makes deep stacks finicky to train. Put it on the *input* of each sublayer and leave the residual path un-normalized — pre-norm — so the identity signal flows clean through the residual and training is more stable at depth. So each block is: normalize the input, run the sublayer (attention or feed-forward), dropout, add back the residual. And I'll strip biases from the dense and attention projections too — they cost parameters and buy nothing here.

The feed-forward sublayer: project up to d_ff, ReLU, project back down. How wide is d_ff? Four times d_model is the standard ratio and I'll keep it (3072 for d_model = 768). The intuition: attention only *mixes* information across positions; the position-wise MLP is where the model actually does most of its per-token computation and stores most of its learned transformations, so it wants to be the widest part of the block. Too narrow and you bottleneck that capacity; 4× is the well-worn sweet spot.

Attention: multiple heads, 12 of them, each with key/value dimension 64, so the concatenated inner dimension is 12 × 64 = 768 = d_model. Multiple heads so the model can attend to several places, in several subspaces, at once. Now there's a standard detail I want to question: the 1/√d_k scaling before the softmax. The reason it's there is that a dot product of two d_k-dimensional vectors with unit-ish entries has magnitude growing like √d_k, which would push the softmax into saturation; dividing by √d_k keeps the logits at a sane scale. But I don't actually have to do it in the forward pass — I can get the same effect by *initializing the query projection smaller*. If I set the query weights' init std to scale like (d_model · d_kv)^(−1/2) instead of d_model^(−1/2), that extra d_kv^(−1/2) factor is exactly the √d_k I was going to divide by, just baked into the weights at birth. The keys and values init at d_model^(−1/2), the output projection at (n_heads · d_kv)^(−1/2). Net result: the pre-softmax logits come out the right scale anyway, and I've removed an operation from the hot inner loop. Folding the scale into initialization, not the forward pass.

Position. Self-attention is a set operation — permute the inputs and the outputs just permute; it has no idea about order. So I must inject position somehow. The original adds absolute position signals (sinusoids or learned vectors) to the token embeddings. The trouble with absolute is it ties everything to absolute indices and generalizes badly to lengths you didn't train on. What actually matters for language is *relative* offset — "the word three to my left" is meaningful regardless of where in the sequence I am. The relative-position idea learns a signal per offset (key position minus query position) and injects it into attention. But the full version learns a whole *vector* per offset and adds it into the key/query computation, which is heavy. I can radically simplify: since the position signal ultimately just nudges the attention *logit* between a query and a key, let it be a single learned **scalar** per offset, added directly to that logit, with a different scalar per head. That's the lightest thing that could possibly encode "how far apart are these two tokens."

Now, do I learn a separate scalar for every possible offset? That's a lot, and most of them are far-apart pairs where the exact distance hardly matters. So bucket the offsets. Use, say, 32 total buckets. Half of them are *exact* — one bucket each for small offsets, where fine resolution matters ("adjacent" really is different from "two away"). The other half grow *logarithmically*: as the offset gets larger, lump wider and wider ranges into one bucket, up to a max distance of 128, and everything at or beyond 128 falls into a single catch-all bucket. The logic: nearby positions need sharp distinctions, distant positions only need a coarse sense of "far," and a log schedule plus a catch-all is exactly that — and it lets the model *generalize to longer sequences than it trained on*, because a never-before-seen huge offset just lands in the same far bucket as other huge offsets. For the encoder, attention is bidirectional, so split the buckets by sign — keep separate buckets for "key is to the left" versus "to the right." For the causal decoder, only non-positive offsets are visible after masking, so the full bucket set can be used for those past-or-current positions. Any single layer is blind to differences beyond 128, but that's fine: stack the layers and they compose local position information into sensitivity to longer ranges. And I'll *share these position parameters across all layers* — relative-position relationships are the same kind of thing at every depth, so reusing them costs almost nothing in parameters (it's just 32 × n_heads scalars per stack) while still letting each head learn its own positional preference. The relative bias gets computed once and threaded into every layer's attention logits.

A couple of remaining choices that I'll make for scale and pragmatism. Tie the embeddings: the input embedding matrix, the decoder's input embedding, and the final output softmax projection are one shared matrix — saves a big chunk of parameters and acts as a regularizer. Optimizer: I'm going to push this to very large sizes, and Adam-style optimizers keep a full second-moment estimate per parameter, which at 11 billion parameters is a lot of extra memory; Adafactor factorizes that second moment and gets most of the benefit at a fraction of the memory, so use Adafactor. Learning-rate schedule: I'll be varying the *number* of training steps across experiments, so I can't use a schedule that needs to know the total ahead of time. Inverse-square-root works without that knowledge — hold a constant rate of 0.01 through 10⁴ warmup steps, then decay like 1/√step. (A triangular schedule did a hair better but needs the total step count, so I'll trade the hair for genericity.) Vocabulary: SentencePiece word-pieces, 32k of them. There's a subtlety — I'll fine-tune on translation into German, French, and Romanian, but I only pre-train on English, so the *fixed* vocabulary has to already contain those languages' pieces or the model literally can't emit them. So train the tokenizer on a 10:1:1:1 mix of English, German, French, Romanian. And add the sentinel tokens as extra IDs. One vocabulary shared across input and output.

Last, the data, because the objective is only as good as what it reconstructs. I want a large, clean, *diverse*, and ideally public English corpus, and there isn't a good standard one — corpora get introduced inside method papers, rarely released, rarely compared, often narrow (just news, just one license). Common Crawl gives me scale for free — about 20 TB of web text a month — but most of it is garbage: menus, boilerplate, "enable JavaScript" notices, lorem-ipsum placeholder, source code, error pages, duplicated passages, offensive text. So I clean it with blunt heuristics that each kill a specific failure mode: keep only lines that end in real terminal punctuation (filters fragments and menus); drop pages with fewer than three sentences and lines under five words; drop any page containing a word from a bad-words list; drop lines/pages with "JavaScript", "lorem ipsum", or a curly brace (the brace is a cheap, high-precision code detector — it's everywhere in programming languages and almost never in prose); strip Wikipedia-style citation markers; drop boilerplate policy lines ("terms of use", "privacy policy", ...); deduplicate by throwing away repeated three-sentence spans; and keep only text that a language detector calls English with probability ≥ 0.99. That yields roughly 750 GB of clean English — the Colossal Clean Crawled Corpus. For the controlled study I'll pre-train on about 2³⁵ ≈ 34 billion tokens, never repeating any data — far less than what some others used, but a deliberately reasonable budget so the whole sweep is affordable, and never repeating matters because I've separately seen that a corpus small enough to be repeated many times degrades downstream performance. Big and diverse beats small and repeated; that's why I want C4 to be huge.

So the whole thing now hangs together as one causal chain. The pain was a fragmented field with no controlled testbed; the fix is to cast every task as text-to-text, which dissolves the task-specific head and gives me one loss, one decoder, and orthogonal knobs; with that testbed I find an encoder-decoder (bidirectional input + cross-attention, FLOP-matched) beats decoder-only and prefix-LM; a denoising objective beats LM, and within denoising the variants tie on quality so I pick the cheapest — span corruption, 15%, mean length 3, sentinels, predict only the spans; I simplify the Transformer with RMSNorm, pre-norm, no biases, relative scalar position biases bucketed log-spaced and shared across layers, and the 1/√d_k scaling folded into initialization; and I feed it a large, clean, diverse English web corpus. Let me write the core of it as code, mirroring how this actually gets built.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# RMSNorm: normalize by root-mean-square only -- no mean-subtraction, no bias.
# The re-centering term buys little; dropping it is cheaper and just as stable.
class T5LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))   # learned scale only
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)  # RMS, no mean
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.type_as(self.weight)


# Position-wise feed-forward: up to d_ff (= 4 * d_model), ReLU, back down.
# Attention only mixes positions; this MLP is where most per-token compute lives,
# so it is the widest part of the block. No biases.
class T5LayerFF(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.layer_norm(x)                 # pre-norm: normalize the *input*
        h = self.wo(self.dropout(F.relu(self.wi(h))))
        return x + self.dropout(h)             # residual added outside the norm


class T5Attention(nn.Module):
    def __init__(self, d_model, d_kv, n_heads, is_decoder,
                 has_relative_bias=False, num_buckets=32, max_distance=128):
        super().__init__()
        self.d_kv, self.n_heads = d_kv, n_heads
        self.inner = d_kv * n_heads
        self.is_decoder = is_decoder
        self.num_buckets, self.max_distance = num_buckets, max_distance
        # No biases. No 1/sqrt(d_k) in the forward pass -- it is folded into the
        # initialization (q-proj init std ~ (d_model * d_kv)^-0.5), so logits come
        # out correctly scaled without an explicit divide.
        self.q = nn.Linear(d_model, self.inner, bias=False)
        self.k = nn.Linear(d_model, self.inner, bias=False)
        self.v = nn.Linear(d_model, self.inner, bias=False)
        self.o = nn.Linear(self.inner, d_model, bias=False)
        self._reset_parameters(d_model)
        # Relative position as a single learned SCALAR per (offset-bucket, head),
        # added straight onto the attention logit. Learned only in the first layer
        # of the stack and then shared across all layers.
        self.has_relative_bias = has_relative_bias
        if has_relative_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def _reset_parameters(self, d_model):
        nn.init.normal_(self.q.weight, mean=0.0,
                        std=(d_model * self.d_kv) ** -0.5)
        nn.init.normal_(self.k.weight, mean=0.0, std=d_model ** -0.5)
        nn.init.normal_(self.v.weight, mean=0.0, std=d_model ** -0.5)
        nn.init.normal_(self.o.weight, mean=0.0, std=self.inner ** -0.5)

    @staticmethod
    def _bucket(rel_pos, bidirectional, num_buckets, max_distance):
        # Map an offset (key_pos - query_pos) to a bucket. Half the buckets are
        # exact for small offsets (fine resolution near the query); the other half
        # grow logarithmically up to max_distance; everything beyond collapses to
        # one catch-all bucket -> graceful generalization to longer sequences.
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (rel_pos > 0).long() * num_buckets   # separate sign for L/R
            rel_pos = rel_pos.abs()
        else:
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))  # causal: <=0 only
        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact
        large = max_exact + (
            torch.log(rel_pos.float() / max_exact)
            / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        large = torch.min(large, torch.full_like(large, num_buckets - 1))
        return ret + torch.where(is_small, rel_pos, large)

    def compute_bias(self, q_len, k_len, device):
        ctx = torch.arange(q_len, device=device)[:, None]
        mem = torch.arange(k_len, device=device)[None, :]
        buckets = self._bucket(mem - ctx, bidirectional=(not self.is_decoder),
                               num_buckets=self.num_buckets,
                               max_distance=self.max_distance)
        # (q_len, k_len, n_heads) -> (1, n_heads, q_len, k_len)
        return self.relative_attention_bias(buckets).permute([2, 0, 1]).unsqueeze(0)

    def forward(self, hidden, kv=None, mask=None, position_bias=None):
        B, qlen = hidden.shape[:2]
        kv_in = hidden if kv is None else kv
        klen = kv_in.shape[1]
        shape = lambda t: t.view(B, -1, self.n_heads, self.d_kv).transpose(1, 2)
        q, k, v = shape(self.q(hidden)), shape(self.k(kv_in)), shape(self.v(kv_in))
        scores = torch.matmul(q, k.transpose(3, 2))   # NB: no scaling factor
        if position_bias is None:
            if self.has_relative_bias:
                position_bias = self.compute_bias(qlen, klen, hidden.device)
            else:
                position_bias = torch.zeros((1, self.n_heads, qlen, klen),
                                            device=hidden.device, dtype=scores.dtype)
            if mask is not None:
                position_bias = position_bias + mask    # causal/padding mask folded in
        scores = scores + position_bias
        attn = F.softmax(scores.float(), dim=-1).type_as(scores)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.inner)
        return self.o(out), position_bias               # bias threaded to next layer


class T5LayerSelfAttention(nn.Module):
    def __init__(self, d_model, d_kv, n_heads, is_decoder, has_relative_bias, dropout):
        super().__init__()
        self.attn = T5Attention(d_model, d_kv, n_heads, is_decoder, has_relative_bias)
        self.layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, position_bias=None):
        y, position_bias = self.attn(self.layer_norm(x), mask=mask,
                                     position_bias=position_bias)
        return x + self.dropout(y), position_bias


class T5LayerCrossAttention(nn.Module):
    def __init__(self, d_model, d_kv, n_heads, dropout):
        super().__init__()
        # Cross-attention has no relative bias -- positions across two different
        # sequences are not comparable.
        self.attn = T5Attention(d_model, d_kv, n_heads, is_decoder=True,
                                has_relative_bias=False)
        self.layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv, mask=None, position_bias=None):
        y, position_bias = self.attn(self.layer_norm(x), kv=kv, mask=mask,
                                     position_bias=position_bias)
        return x + self.dropout(y), position_bias


class T5Block(nn.Module):
    # Encoder block = [self-attn, FF]. Decoder block = [self-attn, cross-attn, FF].
    def __init__(self, d_model, d_ff, d_kv, n_heads, is_decoder,
                 has_relative_bias, dropout):
        super().__init__()
        self.is_decoder = is_decoder
        self.self_attn = T5LayerSelfAttention(d_model, d_kv, n_heads, is_decoder,
                                              has_relative_bias, dropout)
        if is_decoder:
            self.cross_attn = T5LayerCrossAttention(d_model, d_kv, n_heads, dropout)
        self.ff = T5LayerFF(d_model, d_ff, dropout)

    def forward(self, x, mask=None, position_bias=None,
                enc_hidden=None, enc_mask=None, enc_position_bias=None):
        x, position_bias = self.self_attn(x, mask=mask, position_bias=position_bias)
        if self.is_decoder and enc_hidden is not None:
            x, enc_position_bias = self.cross_attn(x, enc_hidden, mask=enc_mask,
                                                   position_bias=enc_position_bias)
        x = self.ff(x)
        return x, position_bias, enc_position_bias


class T5Stack(nn.Module):
    def __init__(self, embed, n_layers, d_model, d_ff, d_kv, n_heads,
                 is_decoder, dropout):
        super().__init__()
        self.is_decoder = is_decoder
        self.embed = embed
        # Only the FIRST block carries the relative-bias parameters; the computed
        # bias is then passed down and reused by every subsequent block (shared
        # across layers, distinct per head within a layer).
        self.blocks = nn.ModuleList([
            T5Block(d_model, d_ff, d_kv, n_heads, is_decoder,
                    has_relative_bias=(i == 0), dropout=dropout)
            for i in range(n_layers)
        ])
        self.final_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, mask=None, enc_hidden=None, enc_mask=None):
        x = self.dropout(self.embed(input_ids))
        position_bias = None
        enc_position_bias = None
        for block in self.blocks:
            x, position_bias, enc_position_bias = block(
                x, mask=mask, position_bias=position_bias,
                enc_hidden=enc_hidden, enc_mask=enc_mask,
                enc_position_bias=enc_position_bias)
        return self.dropout(self.final_norm(x))


class T5ForConditionalGeneration(nn.Module):
    def __init__(self, vocab_size, d_model=768, d_ff=3072, d_kv=64,
                 n_heads=12, n_layers=12, dropout=0.1, pad_id=0, decoder_start_id=0):
        super().__init__()
        # One shared embedding for encoder input, decoder input, and (tied) output.
        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder = T5Stack(self.shared, n_layers, d_model, d_ff, d_kv,
                               n_heads, is_decoder=False, dropout=dropout)
        self.decoder = T5Stack(self.shared, n_layers, d_model, d_ff, d_kv,
                               n_heads, is_decoder=True, dropout=dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight          # weight tying
        self.d_model = d_model
        self.pad_id, self.decoder_start_id = pad_id, decoder_start_id

    def _shift_right(self, labels):
        # Teacher forcing: decoder input is the target shifted right by one, with a
        # start token prepended.
        shifted = labels.new_zeros(labels.shape)
        shifted[..., 1:] = labels[..., :-1].clone()
        shifted[..., 0] = self.decoder_start_id
        shifted[shifted == -100] = self.pad_id
        return shifted

    @staticmethod
    def _causal_mask(seq_len, device):
        m = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return (1.0 - m) * torch.finfo(torch.float32).min   # 0 keep, -inf forbid

    def forward(self, input_ids, attention_mask=None, labels=None):
        enc_mask = None
        if attention_mask is not None:                      # fully-visible padding mask
            enc_mask = (1.0 - attention_mask[:, None, None, :].float()) \
                * torch.finfo(torch.float32).min
        enc_hidden = self.encoder(input_ids, mask=enc_mask)

        if labels is None:
            raise ValueError("labels are required for teacher-forced training")
        dec_input = self._shift_right(labels)
        dec_self_mask = self._causal_mask(dec_input.size(1), dec_input.device)
        dec_hidden = self.decoder(dec_input, mask=dec_self_mask,
                                  enc_hidden=enc_hidden, enc_mask=enc_mask)

        # Tied output projection. (Mesh-TF scales by d_model**-0.5 here.)
        logits = self.lm_head(dec_hidden * (self.d_model ** -0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               labels.view(-1), ignore_index=-100)
        return loss, logits
```

And the span-corruption pre-training transform and the text-to-text task interface, which are what feed this model:

```python
def span_corrupt(token_ids, sentinels, noise_density=0.15, mean_span_len=3.0):
    """Corrupt ~15% of tokens in contiguous spans; replace each span with one
    unique sentinel; the target is the dropped spans, each prefixed by its
    sentinel, ending in a final sentinel. Both input and target come out short."""
    n = len(token_ids)
    n_noise = max(1, round(n * noise_density))
    n_spans = max(1, round(n_noise / mean_span_len))
    # mark a noise mask with n_noise corrupted tokens grouped into n_spans spans
    mask = _random_spans_noise_mask(n, n_noise, n_spans)

    inp, tgt, s = [], [], 0
    i = 0
    while i < n:
        if mask[i]:                                  # start of a corrupted span
            sent = sentinels[s]; s += 1
            inp.append(sent); tgt.append(sent)
            while i < n and mask[i]:                 # collapse the whole span
                tgt.append(token_ids[i]); i += 1
        else:
            inp.append(token_ids[i]); i += 1
    tgt.append(sentinels[s])                         # final sentinel ends the target
    return inp, tgt


def to_text_to_text(task, example):
    """Every task -> (input_text, target_text). A short prefix selects the task;
    classification emits the label word, regression emits a rounded number string,
    generation emits the target text directly."""
    if task == "mnli":
        return (f"mnli premise: {example['premise']} hypothesis: {example['hypothesis']}",
                example["label_text"])               # "entailment"/"neutral"/"contradiction"
    if task == "stsb":
        return (f"stsb sentence1: {example['s1']} sentence2: {example['s2']}",
                f"{round(example['score'] * 5) / 5:.1f}")   # nearest 0.2, as a string
    if task == "summarize":
        return (f"summarize: {example['article']}", example["highlights"])
    if task == "translate_en_de":
        return (f"translate English to German: {example['en']}", example["de"])
    if task == "squad":
        return (f"question: {example['question']} context: {example['context']}",
                example["answer"])
    raise ValueError(task)
```

Tracing the chain one more time, tight: a fragmented field with no controlled testbed → cast every task as text-to-text so the head disappears and one loss / one decoder / orthogonal knobs remain → with that testbed, FLOP-matched comparison says encoder-decoder (bidirectional input plus cross-attention) wins over decoder-only and prefix-LM → denoising beats LM, and within denoising the variants tie on quality so pick the cheapest, which lands on span corruption at 15% with mean length 3, sentinels, predicting only the corrupted spans → simplify the Transformer with RMSNorm, pre-norm, no biases, relative scalar position biases bucketed log-spaced and shared across layers, and the 1/√d_k scaling folded into initialization → train it with Adafactor and an inverse-sqrt schedule on a large, clean, diverse English web corpus. Every piece earned its place by removing a specific failure of the thing before it.
