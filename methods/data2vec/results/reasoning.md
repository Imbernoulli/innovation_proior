The thing I keep tripping over is that self-supervised learning is, at the level of slogans, the same everywhere — "hide part of the input, predict it from the rest" — and yet the actual code for text, speech, and images shares almost nothing. Each modality has invented its own *target*. BERT predicts a sub-word id. wav2vec 2.0 has to first *learn a codebook* of discrete speech units, because audio has no natural vocabulary, and then predict those. HuBERT does worse than that: it runs a whole iterative loop, k-means-clustering its own layer activations into pseudo-labels, predicting them, then re-clustering. Vision is the same zoo — BEiT predicts discrete visual tokens from an offline DALL-E VAE trained on extra data; MAE regresses raw pixels; DINO and BYOL chase augmentation invariance. Every one of these is a contraption built to manufacture a prediction target. If the high-level idea is genuinely modality-agnostic, the target machinery shouldn't have to be. Can I write *one* objective that works identically for all three?

Let me name what these targets have in common, because that's the thing I want to get rid of. Two properties. First, they're *modality-specific*: a word id, a learned speech unit, a visual token — each only makes sense in its own modality, and producing it requires bespoke machinery. Second, and this is the subtler one, they're *local*. A sub-word id describes one position. A visual token describes one patch. A pixel is one pixel. The target at position t carries information about position t and essentially nothing else. Even the discrete-token framing has a related flaw: a given token id gets a single fixed embedding that has to serve as the target for *every* occurrence of that token, in every sentence — the target ignores context entirely.

So what would a target be that needs no modality-specific manufacturing? The most general thing I have access to, in any modality, is the model's own internal representation. The Transformer, after I feed it the input, produces a hidden vector at every position. That vector exists in every modality — patches, audio frames, tokens all become hidden states. What if the prediction target is simply *the model's own representation of the input*? Then I'm not predicting words or pixels or codes; I'm predicting latent vectors, and "latent vector at position t" is defined the same way regardless of modality.

But predicting my own representation has an obvious failure: if the student predicts the representation that the *same network* produces, and I let gradients flow into the target, the network can make the task trivial by collapsing all representations to a constant — predict zero, target zero, loss zero, nothing learned. So the target has to come from a *frozen-ish* copy of the network that the loss can't directly game. This is exactly the mean-teacher / momentum-encoder trick: keep a teacher whose weights are an exponential moving average of the student's, `Δ ← τΔ + (1−τ)θ`, detach it, and let the student chase it. BYOL and DINO already do precisely this — regress an EMA teacher's latent representation. So the latent-target idea isn't new; the question is what task to wrap around it.

Here's where I want to be careful, because BYOL and DINO made a specific choice that I think is the wrong one for my goal. In those methods, the teacher and the student see *different augmentations* of the same image, and the student is asked to match the teacher's representation. The task that defines is "be invariant to the augmentation" — squash out the difference between two crops. That's a fine objective for image classification, but it's not a *fill-in-the-blank* task and it doesn't obviously port to text or speech, where the natural corruption is masking, not color-jitter. And invariance-to-augmentation throws away within-sample structure: it's pushing two views to agree on a global summary, not learning the internal relationships among parts.

The unification I want is to keep the latent EMA target but swap the task to *masked prediction*, which is the one task all three modalities already use. So: feed the *student* a **masked** version of the input. Feed the *teacher* — the EMA copy — the **unmasked, full** input. Ask the student, at each masked position, to predict the teacher's representation of that position *as computed from the complete input*. Now the student has to infer, from context, what the full-input representation at a hidden spot would be. That's BERT's fill-in-the-blank spirit, but the blank is filled with a continuous latent vector instead of a discrete id, and it's identical across modalities (the only modality-specific pieces are the feature encoder up front and the masking pattern).

Let me check one design decision that I almost got wrong: should the teacher see the masked or the unmasked input? My instinct says masked, to keep teacher and student symmetric. But think about what the target *should* contain. I want the target at a masked position to be informative — it should encode what's actually there, drawing on the whole sample. If I also masked the teacher's input, the teacher's representation at that position would itself be a guess from partial context, a weaker target. Giving the teacher the full input means its representation at position t is a genuinely *contextualized* vector: self-attention has let it look at the entire input, so the target carries information from the whole sample, not just position t. That directly fixes the "local target" complaint — the thing the student predicts is now rich, context-dependent, and example-specific. (In line with this: masking the teacher's input degrades accuracy.) So the teacher gets the unmasked input.

And notice what this buys me on the "fixed vocabulary" complaint too. The target is no longer drawn from a closed set. There's no codebook, no visual-token vocabulary, no list of sub-words. The set of possible targets is the continuous space of the teacher's representations, and the target for a given token *depends on its context* — the same word in two different sentences gets two different targets, because the teacher's contextualized vector differs. It's an open, adaptive vocabulary that the model effectively defines as it goes. The quantizer in wav2vec 2.0, the iterative clustering in HuBERT, the offline dVAE in BEiT — all of that machinery just evaporates; I never needed discrete units, only the network's own contextualized latents.

Now, which representation, exactly? A Transformer has L blocks. The obvious choice, the one BYOL and DINO make, is the *top* layer — the final representation. Let me question it. I recall a diagnostic from self-supervised speech models: the top layer isn't the best layer for downstream tasks; the *middle* layers transfer better, because the top layer over-specializes to the pretraining objective. If that's true, then targeting only the top layer is throwing away signal — different layers of a network extract different kinds of features (low-level near the input, more abstract higher up), and a single layer is a narrow target. So instead of the top layer alone, let me build the target from the *top K blocks* of the teacher and average them: take block l's output at step t, call it a_t^l, and set the target

  y_t = (1/K) Σ_{l=L−K+1}^L â_t^l

(the hat is a normalization I'll get to). Averaging several layers makes the target a richer, multi-scale summary, and it makes the self-supervised task harder and more informative. I'd want to verify this sweeps the right way — vary K from 1 to all layers — and the expectation is that K=1 (top layer only, the BYOL choice) is worst, multi-layer is clearly better in every modality, and using all layers is nearly as good as a carefully tuned K. (An alternative would be to predict each of the K layers with its own projection head, but averaging is about as good and far cheaper, so average.)

Which *feature* inside a block do I take as a_t^l? A Transformer block has a self-attention sub-layer and an FFN sub-layer, each wrapped in a residual connection. Let me reason about the candidates. If I take the output of the self-attention sub-layer *before* its residual connection, that vector is essentially a weighted mix of *other* positions — attention's whole job is to pull in information from elsewhere — so it's heavily biased toward other timesteps and barely represents the current position's own content. As a target that's degenerate; I'd expect a model trained on it to be unusable. The FFN output is better: by the time we're at the FFN (after the attention residual has re-added the position's own representation), the feature includes both the position's own content and the attended context. So I take the FFN output — specifically prior to the last residual connection — as the per-block feature. (Confirmed by ablation: FFN-output target works, self-attention-output target gives a non-usable model.)

The normalization â. This is doing real work, and I want to understand both jobs it has. Job one: prevent collapse. When a network regresses its own EMA representations, the trivial minimizer is for the teacher to emit a constant vector everywhere — then the student predicts the constant and the loss is zero with nothing learned. Normalizing the target across the sequence (so the representations are forced to have spread) removes the constant solution: a constant can't survive a normalization that subtracts the mean and scales by the variance. Job two: prevent one layer from dominating the average. Different blocks have very different activation norms; if I just average raw a_t^l, the highest-norm layer swamps the others and I'm effectively back to a single-layer target. Normalizing each block before averaging puts them on comparable scales so the average actually blends them. The right normalizer is modality-dependent: for speech, adjacent frames are highly correlated (tiny stride over the waveform), and that correlation is itself a collapse risk, so I use instance normalization over the sequence; for vision and NLP a parameter-less LayerNorm over the feature dimension is enough. (One could instead bolt on an explicit variance penalty à la VICReg, but normalization solves it with no extra hyperparameters, so I'll prefer that.)

The loss. I'm regressing continuous vectors, so it's a regression loss, not cross-entropy. Plain L2 is the default but it's sensitive to outliers — a few badly-mispredicted dimensions dominate the gradient. The robust compromise is the Smooth L1 (Huber) loss: quadratic for small errors, linear for large ones, with a knee at β:

  L(y, f) = ½(y−f)²/β   if |y−f| ≤ β,   else  |y−f| − ½β.

Small β behaves like L1 (robust, flatter gradients), large β like L2. The price is one tuned hyperparameter β per modality; the loss-function ablation shows L1, L2, and Smooth L1 are all close, so this isn't load-bearing, but Smooth L1 is the safe default. I compute it only at masked positions, summed over the feature dimension, and I scale by 1/√d (d the feature dimension) to keep the loss magnitude sane as the representation width changes.

The teacher EMA schedule. τ shouldn't be constant from the start. Early in training the student is near-random, so its EMA teacher is also near-garbage and there's no point tracking it slowly — I want the teacher to move fast to keep up. Late in training the student is good and I want a stable target, so the teacher should move slowly. So I linearly anneal τ from a smaller τ0 up to a larger τe over the first τn updates, then hold it constant. And a warning I should heed: if τ is too *low*, the teacher tracks the student so tightly that a student collapse instantly propagates into the teacher and the whole thing dies; if the learning rate is too high or warmup too short, same outcome. Collapse is most dangerous exactly where adjacent targets are most correlated and masking spans are long — speech — which is why speech needs the sequence-normalization most. For vision and NLP the EMA momentum tracking alone is enough to keep targets varied.

A couple of efficiency points fall out. The teacher and student are the same architecture; I can *share* the feature encoder and positional encoder between them (only the Transformer body needs the EMA copy), which is cheaper and slightly more accurate. And for vision I use the *same* augmented image for both teacher and student — there's no augmentation-invariance game here, so I don't need two different views; the only difference between the two passes is that the student's copy is masked.

Now the modality-specific shells, kept minimal and borrowed wholesale. Vision: ViT patch embedding, a 224×224 image as 196 patches of 16×16, and BEiT-style blockwise masking — except I mask 60% rather than BEiT's 40%, because images carry less information per token than text and need a harder task to make masked prediction non-trivial. Speech: the wav2vec 2.0 conv feature encoder (16 kHz → 50 Hz) and span masking (sample 6.5% of frames as span starts, mask the next ten, ~49% masked). Text: BPE embeddings and BERT masking. Everything downstream of the feature encoder is identical.

Let me write the core, mirroring how the regression and target construction actually go.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Data2Vec(nn.Module):
    """One Transformer used as both student (masked input) and EMA teacher (full input).
       Target = normalized average of the teacher's top-K FFN block features at masked positions."""
    def __init__(self, feature_encoder, encoder, dim, K, beta=2.0,
                 ema_decay=0.999, ema_end=0.9999, ema_anneal_steps=30000):
        super().__init__()
        self.feature_encoder = feature_encoder      # modality-specific (patches/conv/embed)
        self.encoder = encoder                      # standard Transformer, returns per-block feats
        self.mask_emb = nn.Parameter(torch.zeros(dim))
        self.final_proj = nn.Linear(dim, dim)       # maps student masked-pos output to target space
        self.K = K
        self.beta = beta
        self.ema = None                             # EMA copy of self.encoder, built lazily
        self.ema_decay, self.ema_end, self.ema_anneal_steps = ema_decay, ema_end, ema_anneal_steps

    def set_num_updates(self, step):                # anneal tau: fast early, slow late
        if step >= self.ema_anneal_steps:
            decay = self.ema_end
        else:
            r = (self.ema_end - self.ema_decay) / self.ema_anneal_steps
            decay = self.ema_decay + r * step
        self.ema.set_decay(decay)
        self.ema.step(self.encoder)                 # Delta <- tau*Delta + (1-tau)*theta

    def forward(self, source, mask):
        feats = self.feature_encoder(source)        # [B, T, D]

        # ---- student: MASKED input ----
        x = feats.clone()
        x[mask] = self.mask_emb                      # learned MASK embedding at masked positions
        x, _ = self.encoder(x, return_layer_results=False)

        # ---- teacher (EMA): UNMASKED full input, contextualized target ----
        with torch.no_grad():
            self.ema.model.eval()
            _, layer_results = self.ema.model(feats, return_layer_results=True)
            # per-block FFN feature, layer-normalized, then averaged over the top K blocks
            blocks = [F.layer_norm(b.float(), b.shape[-1:]) for b in layer_results[-self.K:]]
            y = sum(blocks) / len(blocks)            # y_t = (1/K) sum LN(a_t^l)
            y = F.layer_norm(y.float(), y.shape[-1:])  # normalize the average (anti-collapse)
            y = y[mask]                              # targets only at masked positions

        # ---- student prediction at masked positions ----
        x = self.final_proj(x[mask])

        # ---- Smooth L1 regression, summed over feature dim, scaled by 1/sqrt(d) ----
        d = x.size(-1)
        if self.beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(-1)
        else:
            loss = F.smooth_l1_loss(x.float(), y.float(), reduction="none", beta=self.beta).sum(-1)
        return loss.sum() / math.sqrt(d)
```

So the chain that got me here: I wanted one objective for speech, vision, and text, and the only thing standing in the way was that every modality manufactured its own modality-specific, local prediction target. The most general target available in any modality is the model's own latent representation, and predicting it safely is what the EMA-teacher trick already enables in BYOL/DINO — but they used it for augmentation-invariance, which doesn't port. Swapping the task to masked prediction (student sees the masked input, teacher sees the full input) keeps the one pretext task all three modalities share while making the target *contextualized* — rich, example-specific, open-vocabulary — rather than local. Building that target from a normalized average of the teacher's top-K FFN block features (not the top layer alone) makes it multi-scale and avoids the over-specialized top layer; the normalization and the annealed EMA keep it from collapsing; a robust Smooth L1 regression at masked positions trains the student. The feature encoder and masking are the only modality-specific parts left, and they're borrowed straight from prior work.
