We want speech representations learned from raw audio with no transcripts, so that downstream recognition needs only a thin head and a small amount of labeled data. The obstacle unique to speech is that the self-supervision target is not handed to us. Masked-token prediction in language works because the input is already a sequence of discrete word-pieces from a known lexicon; vision's instance-discrimination objectives work because an image is one instance. Speech is a continuous-valued waveform, and three obstacles arrive together. Each utterance contains many sounds, so a clip cannot be treated as one instance. There is no pre-existing inventory of discrete units to predict, so a masked-token loss has no class space. And the boundaries between sound units are unknown, so even a masked-prediction objective has nothing clean to carve up and recover. The existing options each dodge the missing-vocabulary problem rather than solve it cleanly. DiscreteBERT discretizes audio first with a fixed vq-wav2vec quantizer and then runs a BERT over those tokens, which discards whatever the limited-capacity quantizer dropped and freezes the teacher so it can never improve. wav2vec 2.0 keeps continuous inputs but entangles target generation and representation learning into one contrastive InfoNCE objective that needs careful negative sampling, a codebook-diversity loss to avoid collapse, and a Gumbel-softmax temperature schedule. Pseudo-labeling needs labels to begin with and only mimics a label-limited teacher. None of them produce an improvable, label-free, discrete target inventory while keeping that inventory cleanly separated from the model that learns over it.

I propose HuBERT, which stands for Hidden-unit BERT. The idea is to manufacture the missing vocabulary by offline clustering of frame features, and then train a BERT-style model to predict the cluster assignments of masked frames. We do not need correct labels to start; we only need labels that carry phonetic signal, and simple discrete latent models supply exactly that. Run 100-way k-means over 39-dimensional MFCC frame features, built from 13 cepstral coefficients plus their first and second derivatives, to obtain per-frame cluster indices z_t in {1,...,C}. These assignments correlate non-trivially with underlying acoustic units. They are noisy and the cluster identities are arbitrary integers, but the structure is real, so the whole utterance becomes a discrete target sequence Z = [z_1,...,z_T] with no transcripts used. A convolutional waveform encoder, with seven strided conv blocks of 512 channels, strides (5,2,2,2,2,2,2) and kernels (10,3,3,3,3,2,2), gives 320x downsampling and a 20 ms frame rate at 16 kHz. On top of that, a stack of Transformer blocks gives the context model, which emits o_t at frame t. We score o_t against a learned class vector e_c for each unit by cosine similarity at temperature tau, then softmax: p_f(c | masked input, t) = exp(sim(A o_t, e_c) / tau) / sum over c' of exp(sim(A o_t, e_c') / tau). We set tau to 0.1 to sharpen the distribution. This is simply a learned classifier over the discovered units, with e_c as the unit embeddings.

The decision that determines whether the whole thing works is where the loss is applied. Write L_m for the cross-entropy summed over the masked index set M and L_u for the same over the unmasked indices. The general form is L = alpha L_m + (1-alpha) L_u. Consider the two extremes. With alpha = 0, the loss sits only on unmasked frames, whose features the model sees directly. But each frame's cluster index was computed from that frame's own features, so the model only has to imitate the clustering function: a frame-local lookup that reproduces k-means, learns nothing about context, and inherits every clustering error exactly. With alpha = 1, the loss sits only on masked frames, whose features have been replaced by a mask embedding. To predict a hidden frame's unit, the model must build a good representation of the visible surrounding frames, which is the acoustic-modeling part, and exploit the long-range temporal structure of speech to infer what was probably there, which is the language-modeling part. This also tames the target noise: with no frame to copy from, the model cannot memorize a noisy per-frame map and instead must capture whatever is consistently predictable from context, and consistency is a property the clustering retains even where its absolute correctness fails. So I set alpha = 1 and compute the loss only over masked frames. Masking uses spans, not isolated frames, because a single hidden frame is trivially recovered from neighbors a few milliseconds away and would teach almost nothing; we sample a fraction p = 8% of frames as span starts and mask l = 10 consecutive frames from each. The Transformer ingests the masked continuous features, with a mask embedding stamped over the chosen frames, not quantized tokens, so the limited-capacity convolutional front end is never the bottleneck on what the context model gets to reason over. This is the deliberate split from discretize-first systems: the input stays rich, and only the targets are discrete.

K-means on MFCCs gives crude targets, and two composable refinements improve them. First, treat a clustering ensemble as the teacher rather than a single clustering: a coarse k-means separates broad manner classes while a fine one splits sub-phone states, so we predict several clusterings at once with one projection head per clustering, summing the masked loss over all of them. These tasks are free, created by unsupervised clustering at different resolutions, and the complementary views give a richer, more robust target than any single one; this also dovetails with product quantization, where partitioning the feature space into independently clustered subspaces makes the effective target space the product of the per-subspace codebooks. Second, stop trusting MFCCs forever: once the first model has learned context-sensitive features, its intermediate representations are a better space for unit discovery than raw MFCCs, so we re-cluster. We run 500-way k-means over hidden features from the pretrained model and train a fresh generation on the new assignments, fitting on a random 10% sample with 10,000-frame mini-batches, k-means++ initialization, and 20 random starts, because the hidden-feature matrix dwarfs the MFCC matrix. For the second Base generation we cluster the 6th Transformer layer of the first generation; for larger models we cluster the 9th layer of the second Base generation. The loop is the alternate-clustering-and-predicting bootstrap applied to speech with a masked-prediction inner loop: targets train features, features give a better clustering space, and the new clusters train the next model. This is why a teacher as crude as k-means on MFCCs is fine, since the first generation only has to be good enough to seed the next clustering pass. The architecture scales across Base (12 layers, d=768, FFN 3072, 8 heads, projection 256, 95M parameters), Large (24, 1024, 4096, 16, 768, 317M), and X-Large (48, 1280, 5120, 16, 1024, 964M). After pre-training, remove the unit heads, attach a fresh softmax over the character vocabulary, and fine-tune with CTC while keeping the convolutional encoder fixed.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

class ConvFeatureEncoder(nn.Module):
    def __init__(self, dims=(512,)*7,
                 kernels=(10,3,3,3,3,2,2), strides=(5,2,2,2,2,2,2)):
        super().__init__()
        layers, c_in = [], 1
        for c_out, k, s in zip(dims, kernels, strides):
            layers += [nn.Conv1d(c_in, c_out, k, s),
                       nn.GroupNorm(1, c_out), nn.GELU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
    def forward(self, wav):
        return self.conv(wav.unsqueeze(1)).transpose(1, 2)        # (B, T, 512)

class TransformerEncoder(nn.Module):
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

class UnitPredictor(nn.Module):                      # o_t -> logits over C units
    def __init__(self, d=768, proj=256, n_units=100, tau=0.1):
        super().__init__()
        self.A = nn.Linear(d, proj)
        self.embed = nn.Embedding(n_units, proj)
        self.tau = tau
    def forward(self, o):
        h = F.normalize(self.A(o), dim=-1)
        e = F.normalize(self.embed.weight, dim=-1)
        return (h @ e.t()) / self.tau                # (B, T, C)

def span_mask(x, mask_emb, p=0.08, l=10):
    B, T, _ = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for b in range(B):
        starts = (torch.rand(T, device=x.device) < p).nonzero(as_tuple=False).flatten()
        for s in starts.tolist():
            mask[b, s:min(s + l, T)] = True
    x = x.clone(); x[mask] = mask_emb
    return x, mask

def masked_prediction_loss(enc, heads, x, targets, mask_emb, alpha=1.0):
    xm, mask = span_mask(x, mask_emb)
    o = enc(xm)
    Lm = o.new_tensor(0.0)
    Lu = o.new_tensor(0.0)
    def selected_mean(values, selected):
        return values[selected].mean() if selected.any() else values.new_tensor(0.0)
    for head, z in zip(heads, targets):              # one head per clustering
        logits = head(o)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             z.reshape(-1), reduction='none').view(x.size(0), -1)
        Lm = Lm + selected_mean(ce, mask)            # masked frames -> learn context
        Lu = Lu + selected_mean(ce, ~mask)           # unmasked frames -> mimic teacher
    return alpha * Lm + (1 - alpha) * Lu             # alpha=1

def fit_kmeans_targets(feature_extractor, dataset, n_clusters, layer=None,
                       sample_ratio=1.0, batch_size=10_000):
    chunks = []
    for x in dataset:
        f = feature_extractor(x) if layer is None else feature_extractor(x, layer)
        chunks.append(f.reshape(-1, f.size(-1)))
    feats = torch.cat(chunks, dim=0)                    # all frame features, (N, d)
    if sample_ratio < 1.0:
        keep = torch.randperm(feats.size(0))[:int(sample_ratio * feats.size(0))]
        feats = feats[keep]
    return MiniBatchKMeans(n_clusters=n_clusters, init="k-means++",
                           n_init=20, batch_size=batch_size).fit(
                               feats.detach().cpu().numpy())

def first_generation_targets(mfcc_extractor, dataset):
    return fit_kmeans_targets(mfcc_extractor, dataset, n_clusters=100)

def refined_targets(hidden_extractor, dataset, layer=6):
    # Use layer=9 when clustering the second Base generation for larger models.
    return fit_kmeans_targets(hidden_extractor, dataset, n_clusters=500,
                              layer=layer, sample_ratio=0.10)

class CTCHead(nn.Module):                            # fine-tuning: 26 + space + apostrophe + blank
    def __init__(self, d=768, n_vocab=29):
        super().__init__()
        self.proj = nn.Linear(d, n_vocab)
    def forward(self, o):
        return self.proj(o).log_softmax(-1)
```
