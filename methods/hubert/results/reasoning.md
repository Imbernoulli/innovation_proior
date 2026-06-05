Let me start from the thing that makes self-supervised speech genuinely harder than self-supervised text. In language the masked-prediction recipe is clean: the input is already a sequence of discrete word-pieces, so I hide some and predict their identities with a cross-entropy against a known vocabulary. That works because the *targets exist* — a finite lexicon was handed to me. Speech gives me a waveform: a continuous stream with no tokens, no vocabulary, and — worse — no known boundaries between sound units. So three problems land at once. There are many sounds per utterance, so I can't treat the whole clip as one instance the way vision does with images. There's no discrete inventory to predict against, so I can't directly run a masked-token loss. And I don't even know where one sound ends and the next begins, so I can't cleanly carve the sequence into things to mask and recover. The hardest of the three is the missing vocabulary — if I had even a noisy set of per-frame unit labels, masked prediction would have something concrete to chew on.

So the real question is: where do targets come from? I have no labels. But I don't need *correct* labels to begin with — I need labels that carry phonetic signal. And here's a fact I can lean on: if I take crude frame features, say MFCCs, and just run k-means on them, the resulting clusters are not random. They correlate non-trivially with phones. A cluster will tend to gather, say, fricatives in one bin and certain vowels in another. The assignment is noisy and the cluster identities are arbitrary integers, but the *structure* is real. So I can manufacture targets by clustering: run k-means over frame features, and let each frame's cluster index z_t ∈ {1,…,C} be its pseudo-label. Now I have a discrete target sequence Z = [z₁,…,z_T] for any utterance, with no transcripts used at all.

That immediately suggests the objective. Take the waveform, encode it to frames, mask a span, run a Transformer over the corrupted sequence, and at each frame predict the cluster index z_t with a cross-entropy. The model is a BERT over continuous speech features whose targets are these discovered units. Let me write the per-frame prediction: the encoder emits o_t at frame t, and I want a distribution over the C units. I'll project o_t and compare it to a learned embedding e_c for each unit by cosine similarity, scaled by a temperature, and softmax:

  p(c | corrupted X, t) = exp(sim(A o_t, e_c)/τ) / Σ_{c'} exp(sim(A o_t, e_{c'})/τ),

with τ around 0.1 to sharpen. This is just a learned classifier over units with the unit "class vectors" e_c. The training loss is the negative log-likelihood of the true cluster at each scored frame.

Now the decision that I think actually determines whether this works: *where* do I apply the loss — on masked frames, on unmasked frames, or both? Let me reason through the two extremes. Suppose I compute the loss only on *unmasked* frames — the frames the model can see directly. Then I'm asking the model, "given this frame's own features, output its cluster index." But the cluster index *was computed from* this frame's features. So the model just has to imitate the clustering function — it's a frame-local lookup, and the best it can do is reproduce k-means. It learns nothing about context, and it inherits every error of the clustering exactly. Call that the α=0 case (loss only on the unmasked set). It collapses to mimicking the teacher.

Now the other extreme: loss only on *masked* frames. The model cannot see those frames' features — they've been replaced by a mask embedding. So to predict the unit at a hidden frame it must (a) build a good representation of the *visible* frames around it, that's the acoustic-modeling part, and (b) use the long-range temporal structure of speech to infer what was probably there, that's the language-modeling part. This is exactly analogous to masked language modeling, and it forces both kinds of learning. Crucially, think about what happens to the *noise* in the targets here. The targets are wrong sometimes. But if I'm predicting a hidden frame from context, the model can't just copy a per-frame teacher label — there's no frame to copy from. It has to find whatever is *consistently predictable from context*, and consistency is a property the clustering has even where its absolute correctness fails: the same sound tends to land in the same cluster. So the masked objective should be far more robust to bad targets than the unmasked one, because it rewards modeling the consistent sequential structure rather than memorizing a noisy per-frame map. Call this α=1. This is the case I want.

The general form interpolates: with L_m the loss summed over masked indices M and L_u the same over the unmasked indices, take

  L = α L_m + (1−α) L_u,

where each L is a cross-entropy, e.g. L_m = −Σ_{t∈M} log p_f(z_t | corrupted X, t). My reasoning says push α toward 1 — compute the loss essentially only over the masked frames — because that's the setting that learns context and is resilient to target noise. (The unmasked-only end isn't useless to *understand*, it's just what classical frame-by-frame acoustic modeling does, and it can't bootstrap beyond the teacher.)

For masking I don't want isolated single frames — a single hidden frame is trivially recoverable from its immediate neighbors a few milliseconds away, so the task would teach almost nothing. I want spans: sample a fraction p% of frames as span starts and mask l consecutive frames from each. That's the same span-masking shape that works for masked prediction elsewhere, and it makes each recovered region demand genuine context. And I feed the Transformer the masked *continuous* features (a mask embedding stamped over chosen frames), not quantized tokens — I want to pass as much information as possible into the Transformer, because the limited-capacity convolutional front end shouldn't be the bottleneck on what the context model gets to reason over. That's a deliberate split from the discretize-first systems: the *input* stays rich; only the *targets* are discrete.

Now, k-means on MFCCs gives crude targets. Two independent ways to make them better, and they compose.

First, ensembles. One clustering is a single noisy view. But different clusterings capture different granularities — a k-means with few clusters separates broad manner classes (vowel vs. consonant), one with many clusters splits into sub-phone states. If I have several clusterings {Z^{(k)}}, I can ask the model to predict *all* of them simultaneously, one prediction head per clustering. The masked loss becomes a sum over clusterings,

  L_m = Σ_{t∈M} Σ_k log p_f^{(k)}(z_t^{(k)} | corrupted X, t)

(again, the minimized loss is the negative of this — a sum of per-clustering cross-entropies), with a separate projection A^{(k)} feeding each head. This is multi-task learning where the tasks are free, created by unsupervised clustering at different resolutions; the complementary views give a richer, more robust target than any single one. It also dovetails with product quantization: partition the feature space into subspaces and cluster each separately, so the effective target space is the product of the per-subspace codebooks — a cheap way to get many fine targets.

Second, iterative refinement. The crude targets were built from MFCCs. But once I've trained a model on them, the model's *own* intermediate representations are better than MFCCs — they've absorbed context and the consistent structure. So re-cluster: run k-means over the learned latent representations of the trained model to get a new, better target sequence, and train a fresh generation of the model on those. The features improve the targets, the better targets improve the features, and I can iterate. This is the alternate-clustering-and-predicting bootstrap, applied to speech with a masked-prediction loss in the inner loop — and it's why even starting from something as crude as k-means on MFCCs is fine: the first generation only has to be good enough to seed a better second generation.

Let me settle the architecture, which I'm largely inheriting because it's not where the idea lives. A convolutional waveform encoder — seven strided conv blocks, 512 channels, strides (5,2,2,2,2,2,2), kernels (10,3,3,3,3,2,2) — gives a 20 ms frame rate (320× downsampling at 16 kHz). On top, a stack of identical Transformer blocks with relative positional information; this is the BERT encoder. Then a projection layer A and a unit-embedding table feeding the softmax over units. Sizes scale across Base (12 layers, dim 768, FFN 3072, 8 heads), Large (24 layers, dim 1024), and X-Large (48 layers, dim 1280, ~1B params). The projection dim is small (256–1024).

After pre-training I have a context model whose frame outputs carry phonetic structure. To recognize, I drop the unit-prediction projection(s), attach a fresh randomly-initialized softmax over the character vocabulary (26 letters, a space, an apostrophe, and a CTC blank), and fine-tune the whole model *except* the convolutional encoder, which I freeze — it already learned good low-level features and updating it on small labeled data would overfit. The recognizer is trained with CTC, summing over blank-augmented alignments since I have no frame-level alignment.

Let me write it, mirroring how I'd build it. First the front end and encoder, then the unit predictor, then the masked-prediction loss with α→1, then the offline clustering that creates and refines targets.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ConvFeatureEncoder(nn.Module):                 # waveform -> 20ms latent frames
    def __init__(self, dims=(512,)*7,
                 kernels=(10,3,3,3,3,2,2), strides=(5,2,2,2,2,2,2)):
        super().__init__()
        layers, c_in = [], 1
        for c_out, k, s in zip(dims, kernels, strides):
            layers += [nn.Conv1d(c_in, c_out, k, s),
                       nn.GroupNorm(1, c_out), nn.GELU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
    def forward(self, wav):                          # (B, T_samples)
        return self.conv(wav.unsqueeze(1)).transpose(1, 2)        # (B, T, 512)

class TransformerEncoder(nn.Module):                 # BERT-style context model
    def __init__(self, d=768, layers=12, heads=8, ffn=3072):
        super().__init__()
        self.pos_conv = nn.Conv1d(d, d, 128, padding=64, groups=16)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d, heads, ffn, 0.1, F.gelu, batch_first=True)
            for _ in range(layers))
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        p = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)[:, :x.size(1)]
        x = self.ln(x + F.gelu(p))
        for l in self.layers:
            x = l(x)
        return x
```

The unit predictor scores each frame's output against a unit-embedding table by cosine similarity; one head per clustering when ensembling:

```python
class UnitPredictor(nn.Module):                      # o_t -> distribution over C units
    def __init__(self, d=768, proj=256, n_units=100, tau=0.1):
        super().__init__()
        self.A = nn.Linear(d, proj)                  # A^{(k)} per clustering when ensembling
        self.embed = nn.Embedding(n_units, proj)     # unit "class vectors" e_c
        self.tau = tau
    def forward(self, o):                            # (B, T, d)
        h = F.normalize(self.A(o), dim=-1)           # (B, T, proj)
        e = F.normalize(self.embed.weight, dim=-1)   # (C, proj)
        return (h @ e.t()) / self.tau                # logits (B, T, C) = sim/tau
```

The masked-prediction loss: stamp a mask embedding over sampled spans, feed the corrupted continuous features, and apply cross-entropy only on the masked frames (α≈1), summing across clusterings for the ensemble:

```python
def span_mask(x, mask_emb, p=0.08, l=10):
    B, T, _ = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for b in range(B):
        for s in (torch.rand(T, device=x.device) < p).nonzero().flatten():
            mask[b, s:s + l] = True                  # mask l consecutive frames
    x = x.clone(); x[mask] = mask_emb                # continuous features, masked in spans
    return x, mask

def masked_prediction_loss(enc, heads, x, targets, mask_emb, alpha=1.0):
    # targets: list over clusterings, each (B, T) of cluster ids in [0, C)
    xm, mask = span_mask(x, mask_emb)
    o = enc(xm)
    Lm = Lu = 0.0
    for head, z in zip(heads, targets):              # one head per clustering (ensemble)
        logits = head(o)                             # (B, T, C)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             z.reshape(-1), reduction='none').view(x.size(0), -1)
        Lm = Lm + ce[mask].mean()                    # loss on MASKED frames -> learn context
        Lu = Lu + ce[~mask].mean()                   # loss on unmasked frames -> mimic teacher
    return alpha * Lm + (1 - alpha) * Lu             # alpha -> 1: robust to noisy targets
```

The offline clustering that creates targets and then refines them from the model's own features:

```python
from sklearn.cluster import MiniBatchKMeans

def make_targets(feature_extractor, dataset, n_clusters, layer=None):
    # iteration 1: feature_extractor returns MFCCs.
    # later iterations: feature_extractor returns a HIDDEN LAYER of the trained
    # encoder -> better features -> better, more consistent targets (re-cluster & repeat).
    feats = torch.cat([feature_extractor(x, layer) for x in dataset])      # (N, d)
    km = MiniBatchKMeans(n_clusters).fit(feats.numpy())
    return km                                          # km.predict(...) gives per-frame z_t
```

Fine-tuning: drop the unit heads, attach a fresh CTC softmax, freeze the conv encoder:

```python
class CTCHead(nn.Module):                            # 26 letters + space + apostrophe + blank
    def __init__(self, d=768, n_vocab=29):
        super().__init__()
        self.proj = nn.Linear(d, n_vocab)
    def forward(self, o):
        return self.proj(o).log_softmax(-1)

def finetune_step(wav, enc_conv, enc_tf, head, targets, in_lens, tgt_lens):
    with torch.no_grad():
        z = enc_conv(wav)                            # convolutional encoder FROZEN
    o = enc_tf(z)
    logp = head(o).transpose(0, 1)                   # (T, B, V)
    return F.ctc_loss(logp, targets, in_lens, tgt_lens, blank=0)
```

The causal chain, start to end: I want speech representations from raw audio with no transcripts, but speech hands me no vocabulary to predict; I notice that k-means over crude frame features already produces cluster assignments that carry phonetic structure, so I use those discovered units as discrete targets; I run a BERT over masked *continuous* features and apply the prediction loss on the *masked* frames (α→1), because predicting hidden units from context forces both acoustic and long-range temporal modeling and — since it rewards the *consistency* of the labeling rather than memorizing a noisy per-frame map — is robust to the targets being wrong; I make the targets better by predicting an *ensemble* of clusterings at different granularities at once, and by *re-clustering the model's own improved representations* to seed the next generation; and at fine-tuning I freeze the conv encoder, attach a fresh CTC head, and train for recognition.
