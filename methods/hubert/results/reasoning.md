Let me start from the thing that makes self-supervised speech genuinely harder than self-supervised text. In language the masked-prediction recipe is clean: the input is already a sequence of discrete word-pieces, so I hide some and predict their identities with a cross-entropy against a known vocabulary. That works because the *targets exist* — a finite lexicon was handed to me. Speech gives me a waveform: a continuous stream with no tokens, no vocabulary, and — worse — no known boundaries between sound units. So three problems land at once. There are many sounds per utterance, so I can't treat the whole clip as one instance the way vision does with images. There's no discrete inventory to predict against, so I can't directly run a masked-token loss. And I don't even know where one sound ends and the next begins, so I can't cleanly carve the sequence into things to mask and recover. The hardest of the three is the missing vocabulary — if I had even a noisy set of per-frame unit labels, masked prediction would have something concrete to chew on.

So the real question is: where do targets come from? I have no labels. But I don't need *correct* labels to begin with — I need labels that carry phonetic signal. And here's a fact I can lean on: simple discrete latent-variable models such as k-means or GMMs over acoustic frames already produce hidden units with non-trivial correlation to underlying acoustic units. I can take 39-dimensional MFCCs — 13 coefficients plus first and second derivatives — and run 100-way k-means offline. The assignment is noisy and the cluster identities are arbitrary integers, but the *structure* is real. So I can manufacture the first generation of targets by clustering those frame features into 100 units, and let each frame's cluster index z_t ∈ {1,…,C} be its pseudo-label. Now I have a discrete target sequence Z = [z₁,…,z_T] for any utterance, with no transcripts used at all.

That immediately suggests the objective. Take the waveform, encode it to frames, mask a span, run a Transformer over the corrupted sequence, and at each frame predict the cluster index z_t with a cross-entropy. The model is a BERT over continuous speech features whose targets are these discovered units. Let me write the per-frame prediction: the encoder emits o_t at frame t, and I want a distribution over the C units. I'll project o_t and compare it to a learned embedding e_c for each unit by cosine similarity, scaled by a temperature, and softmax:

  p(c | corrupted X, t) = exp(sim(A o_t, e_c)/τ) / Σ_{c'} exp(sim(A o_t, e_{c'})/τ),

with τ set to 0.1 to sharpen. This is just a learned classifier over units with the unit "class vectors" e_c. The training loss I minimize is the negative log-likelihood of the true cluster at each scored frame; if I write it as a log-likelihood instead, the same choice appears as maximizing log probability.

Now the decision that I think actually determines whether this works: *where* do I apply the loss — on masked frames, on unmasked frames, or both? Let me reason through the two extremes. Suppose I compute the loss only on *unmasked* frames — the frames the model can see directly. Then I'm asking the model, "given this frame's own features, output its cluster index." But the cluster index *was computed from* this frame's features. So the model just has to imitate the clustering function — it's a frame-local lookup, and the best it can do is reproduce k-means. It learns nothing about context. Call that the α=0 case (loss only on the unmasked set).

Now the other extreme: loss only on *masked* frames. The model cannot see those frames' features — they've been replaced by a mask embedding. So to predict the unit at a hidden frame it must (a) build a good representation of the *visible* frames around it, that's the acoustic-modeling part, and (b) use the long-range temporal structure of speech to infer what was probably there, that's the language-modeling part. This is the same shape as masked language modeling, and it forces both kinds of learning.

But the part I actually care about is what each regime does with the *noise* in the targets, and I don't trust my verbal intuition on that — let me put numbers on it with a deliberately tiny model. Say there are two true sounds, A and B, and the k-means teacher is a noisy labeler: when the true sound is A it outputs unit `a` 80% of the time and `b` 20%, symmetric for B. So the teacher's per-frame error against the truth is 20%.

In the unmasked-only regime the model sees the frame's own features and is scored against that frame's own (noisy) teacher label. With enough capacity the predictor that minimizes the loss is just the conditional distribution of the label given the features — i.e. it reproduces the teacher map exactly. So its hard readout disagrees with the *true* sound exactly as often as the teacher does: 20%. The error passes straight through, undiluted.

Now the masked-only regime, on a frame whose true sound is pinned by context (think a short run of the same sound, or strong sequential structure — context, not the frame's own features, says it's A). The frame's features are hidden, so the model can't copy the teacher; it predicts the *target distribution* it expects given everything it can see. Given true sound A that target is the 80/20 mixture over `{a,b}`, so the loss-minimizing soft prediction is `p(a)=0.8, p(b)=0.2`. Reading off a hard unit by argmax gives `a` — which is the unit that A *should* map to. Let me actually compute these two numbers rather than assert them:

```
unmasked-only: optimal predictor = teacher; hard-label error vs TRUTH = 0.20
masked-only:   optimal soft target given true sound = (0.8, 0.2),
               entropy = -(0.8 ln0.8 + 0.2 ln0.2) = 0.500 nats  (irreducible),
               but hard argmax readout vs TRUTH = 0.00
```

That contrast is the whole point, and it survived the check: in the masked regime the 20% teacher noise turns into an *irreducible cross-entropy floor* (0.5 nats here) rather than into a wrong predicted unit, **as long as context genuinely pins the true sound**. That last clause is a real condition, not a guarantee — if context is uninformative, the masked target is just as noisy as the unmasked one and the argmax is a coin flip. So the masked regime isn't magically denoising; it is converting per-frame label noise into loss-floor noise to the extent that the labeling is *consistent* across contexts, and consistency is a property k-means has even where its absolute correctness fails (the same sound tends to land in the same cluster). On the strength of that I'll lean toward α=1, with the caveat that I'd want to confirm empirically that real context is informative enough — the toy assumed it was.

Let me keep the general form so the choice is explicit rather than baked in. With L_m the negative log-likelihood summed over masked indices M and L_u the same over the unmasked indices, take

  L = α L_m + (1−α) L_u,

where each L is a cross-entropy, e.g. L_m = −Σ_{t∈M} log p_f(z_t | corrupted X, t). The toy above pushes me to α=1 — loss only over the masked frames — since that's the end that both learns context and converts teacher noise into a loss floor instead of wrong units. The α=0 end isn't useless to *understand*: it's exactly what classical frame-by-frame acoustic modeling does, and it pins down why that approach can't bootstrap past the teacher. I'll carry α as a knob rather than hard-wiring it, in case real context turns out less informative than the toy assumed.

For masking I don't want isolated single frames. The frame rate matters here, so let me pin it down from the conv stack: strides (5,2,2,2,2,2,2) multiply to 5·2·2·2·2·2·2 = 320, and at 16 kHz that puts one frame every 320/16000 = 20 ms, i.e. a 50 Hz frame rate. So a single masked frame sits 20 ms from each neighbor — and the neighbors are *visible*. The model could recover a one-frame gap by near-interpolation, which teaches almost nothing. I want spans: sample a fraction p% of frames as span starts and mask l consecutive frames from each. Let me sanity-check that p=8%, l=10 gives a usable masked fraction and not, say, everything or nothing. Tracing the masking routine on a realistic 10 s clip (≈500 frames) and averaging over 200 random seeds, the masked fraction comes out at 0.56 — sensible: there are ≈p·T span starts each covering up to l frames, so the crude ceiling is p·l = 0.8, and overlap between nearby spans pulls the realized fraction below that. About half the frames hidden in contiguous chunks of ~200 ms each is exactly the regime where recovery demands real context rather than interpolation. And I feed the Transformer the masked *continuous* features (a mask embedding stamped over chosen frames), not quantized tokens — I want to pass as much information as possible into the Transformer, because the limited-capacity convolutional front end shouldn't be the bottleneck on what the context model gets to reason over. That's a deliberate split from the discretize-first systems: the *input* stays rich; only the *targets* are discrete.

Now, k-means on MFCCs gives crude targets. Two independent ways to make them better, and they compose.

One path is to stop treating a single clustering as the whole teacher. One clustering is one noisy view, and different clusterings capture different granularities — a k-means with few clusters separates broad manner classes (vowel vs. consonant), one with many clusters splits into sub-phone states. If I have several clusterings {Z^{(k)}}, I can ask the model to predict *all* of them simultaneously, one prediction head per clustering. The masked loss becomes a sum over clusterings,

  L_m = −Σ_{t∈M} Σ_k log p_f^{(k)}(z_t^{(k)} | corrupted X, t)

with a separate projection A^{(k)} feeding each head. This is multi-task learning where the tasks are free, created by unsupervised clustering at different resolutions. The hope is that errors in different clusterings are not perfectly correlated, so a frame the coarse view gets wrong the fine view may get right, and the shared representation has to satisfy all heads at once; I'd want to verify on data that the views are actually complementary rather than redundant, but the mechanism is at least plausible. It also dovetails with product quantization: partition the feature space into subspaces and cluster each separately, so the effective target space is the product of the per-subspace codebooks — a cheap way to get many fine targets.

The other path is to stop trusting MFCCs forever. The crude targets were built from MFCCs. But if the first pretrained model has learned useful context-sensitive features, its intermediate representations should be a better space for acoustic unit discovery than raw MFCCs — that's the bet, and it's only a bet until measured. So re-cluster: run 500-way k-means over hidden features from a pretrained model, and train a fresh generation of the model on those new assignments. The practical details matter because the hidden feature matrix is much larger than the MFCC matrix: fit k-means on a random 10% sample, with mini-batches of 10,000 frames, k-means++ initialization, and 20 random starts. For the second Base generation I use the 6th Transformer layer of the first generation; for the larger models I can take the 9th layer from the second Base generation, which makes those labels a third refinement step. The loop is: targets train features, features give a new clustering space, and the new clusters train the next model — an alternate-clustering-and-predicting bootstrap, applied to speech with a masked-prediction loss in the inner loop. The toy above gives me a reason to expect the first generation doesn't need to be good in absolute terms: the masked loss extracts the *consistent* part of even a crude labeling, and the resulting features only have to define a better clustering space than raw MFCCs for the next pass to improve. Whether each pass actually improves — rather than drifting or plateauing — is the thing I can't settle on paper; it has to be checked generation by generation.

Let me settle the architecture, which I'm largely inheriting because the target-generation problem is where the idea lives. A convolutional waveform encoder — seven strided conv blocks, 512 channels, strides (5,2,2,2,2,2,2), kernels (10,3,3,3,3,2,2) — gives the 20 ms / 320× frame rate I worked out above. On top, a stack of identical Transformer blocks gives the BERT encoder. Then a projection layer A and a unit-embedding table feed the softmax over units. Sizes scale across Base (12 layers, dim 768, FFN 3072, 8 heads, projection 256, 95M parameters), Large (24 layers, dim 1024, FFN 4096, 16 heads, projection 768, 317M), and X-Large (48 layers, dim 1280, FFN 5120, 16 heads, projection 1024, 964M).

After pre-training I have a context model whose frame outputs carry phonetic structure. To recognize, I drop the unit-prediction projection(s), attach a fresh randomly-initialized softmax over the character vocabulary (26 letters, a space, an apostrophe, and a CTC blank), and fine-tune with the convolutional waveform encoder fixed. The recognizer is trained with CTC, summing over blank-augmented alignments since I have no frame-level alignment; a freeze-step can also hold the Transformer fixed at the beginning so the new softmax starts from stable features.

The implementation follows the same chain: the front end and encoder feed the unit predictor; the masked-prediction loss uses α=1; the offline clusterer creates MFCC targets first and then refines them from learned hidden features.

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

The masked-prediction loss stamps a mask embedding over sampled spans, feeds the corrupted continuous features, and applies cross-entropy only on the masked frames (α=1), summing across clusterings for the ensemble:

```python
def span_mask(x, mask_emb, p=0.08, l=10):
    B, T, _ = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for b in range(B):
        starts = (torch.rand(T, device=x.device) < p).nonzero(as_tuple=False).flatten()
        for s in starts.tolist():
            mask[b, s:min(s + l, T)] = True          # mask l consecutive frames
    x = x.clone(); x[mask] = mask_emb                # continuous features, masked in spans
    return x, mask

def masked_prediction_loss(enc, heads, x, targets, mask_emb, alpha=1.0):
    # targets: list over clusterings, each (B, T) of cluster ids in [0, C)
    xm, mask = span_mask(x, mask_emb)
    o = enc(xm)
    Lm = o.new_tensor(0.0)
    Lu = o.new_tensor(0.0)
    def selected_mean(values, selected):
        return values[selected].mean() if selected.any() else values.new_tensor(0.0)
    for head, z in zip(heads, targets):              # one head per clustering (ensemble)
        logits = head(o)                             # (B, T, C)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             z.reshape(-1), reduction='none').view(x.size(0), -1)
        Lm = Lm + selected_mean(ce, mask)            # loss on MASKED frames -> learn context
        Lu = Lu + selected_mean(ce, ~mask)           # loss on unmasked frames -> mimic teacher
    return alpha * Lm + (1 - alpha) * Lu             # alpha=1: masked-frame prediction
```

The offline clustering that creates targets and then refines them from the model's own features:

```python
from sklearn.cluster import MiniBatchKMeans

def fit_kmeans_targets(feature_extractor, dataset, n_clusters, layer=None,
                       sample_ratio=1.0, batch_size=10_000):
    # iteration 1: 100 clusters over 39-dim MFCCs.
    # later iterations: 500 clusters over an intermediate hidden layer.
    chunks = []
    for x in dataset:
        f = feature_extractor(x) if layer is None else feature_extractor(x, layer)
        chunks.append(f.reshape(-1, f.size(-1)))
    feats = torch.cat(chunks, dim=0)                       # all frame features, (N, d)
    if sample_ratio < 1.0:
        keep = torch.randperm(feats.size(0))[:int(sample_ratio * feats.size(0))]
        feats = feats[keep]
    km = MiniBatchKMeans(n_clusters=n_clusters, init="k-means++",
                         n_init=20, batch_size=batch_size).fit(
                             feats.detach().cpu().numpy())
    return km                                          # km.predict(...) gives per-frame z_t

def first_generation_targets(mfcc_extractor, dataset):
    return fit_kmeans_targets(mfcc_extractor, dataset, n_clusters=100)

def refined_targets(hidden_extractor, dataset, layer=6):
    # Use layer=9 when clustering the second Base generation for larger models.
    return fit_kmeans_targets(hidden_extractor, dataset, n_clusters=500,
                              layer=layer, sample_ratio=0.10)
```

For fine-tuning I drop the unit heads, attach a fresh CTC softmax, and keep the conv encoder fixed:

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
