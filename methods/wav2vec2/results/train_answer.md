Modern speech recognizers are accurate only because they are fed thousands of hours of *transcribed* audio, and almost no language on Earth has that. Raw, untranscribed speech, by contrast, is everywhere, for nearly every one of the world's languages. The asymmetry I want to exploit is exactly that: unlimited audio, almost no transcripts. If I can learn the *structure* of speech from raw audio alone, then fitting an actual recognizer on top should need only a sliver of labels — tens of minutes instead of thousands of hours. So the goal is to pre-train a representation from waveform with no transcripts, then fine-tune to recognition cheaply. The trouble is what I have to learn from: a one-dimensional stream of samples at 16 kHz, with no segmentation into words, no phoneme boundaries, no tokens. That is what makes speech harder than text or vision. The masked-language-model recipe works for text because the input arrives pre-discretized into word-pieces — hide a token, predict it from a fixed vocabulary, score with cross-entropy. Speech hands me no such vocabulary. The instance-discrimination recipe of vision does not map either, because an utterance contains many sounds, not one instance. The continuous contrastive predecessors (wav2vec, CPC) predict future latents but stay continuous throughout and contextualize with a limited-receptive-field convolution, with no inventory of speech units and no global attention. The two-step quantize-then-BERT systems (vq-wav2vec, DiscreteBERT) do discretize, but they *freeze* the quantizer before the context model ever runs, so the two stages can never co-adapt, and the context model only ever ingests already-lossy tokens. Self-training needs labels to begin with. Whatever pretext task I invent has to manufacture its own targets out of the continuous stream, and it has to do better than freezing a quantizer up front.

I propose wav2vec 2.0. The backbone is straightforward: normalize the waveform and run a stack of seven strided 1-D convolution blocks (512 channels, strides $(5,2,2,2,2,2,2)$, kernels $(10,3,3,3,3,2,2)$, layer norm and GELU) to get a latent frame every $\sim$20 ms — about 49 frames per second with a 25 ms receptive field. Convolution is the natural translation-equivariant local feature extractor and the strides do the downsampling. Call these continuous latents $z_1,\dots,z_T$. A Transformer then contextualizes each frame against the whole utterance, because self-attention is all-to-all and gives every frame a representation conditioned on the entire sequence — call those $c_1,\dots,c_T$. The whole question is the objective that trains this stack from audio alone, and that is where the real design lives.

The obvious BERT move is to mask frames and predict them, but predict them *as what*? Regressing the masked continuous latent $z_t$ with an L2 loss is a trap: a waveform latent carries phonetic content but also pitch, room reverberation, background hum, and the cheapest way to drive an L2 loss down is to nail that low-level acoustic texture, not the linguistic structure. The target is too rich and rewards the wrong thing — the same reason future-sample reconstruction is wasteful. So I do not reconstruct; I predict by *classification*. I never have to name the exact target if I only have to *pick it out* from a set of wrong answers. Given the context vector $c_t$ at a masked frame, I present it with the true target plus $K$ distractors and train it to identify the true one with a noise-contrastive (InfoNCE) loss. With cosine similarity $\mathrm{sim}$ and temperature $\kappa$, the loss at a masked frame $t$ is

$$L_m = -\log \frac{\exp(\mathrm{sim}(c_t, q_t)/\kappa)}{\sum_{\tilde q \in Q_t}\exp(\mathrm{sim}(c_t,\tilde q)/\kappa)},$$

where $Q_t$ holds the true target $q_t$ and $K$ distractors. No generation, only scoring.

The deeper choice is what $q_t$ *is*. If the target is the continuous $z_t$, the nuisance problem returns in a softer form: positive and distractors are all continuous latents from the same utterance, sharing speaker and room, so the discriminator can still latch onto recording-specific artifacts and the learned representation stays narrow. So I *discretize* the target. Quantizing $z_t$ to a code from a small learned inventory is a deliberate lossy bottleneck: it throws away continuous nuisance detail and keeps a coarse, shared identity, so frames that are all the same sound map to the same code regardless of speaker, and the target becomes phonetically meaningful. This is what the two-step systems were groping toward — but I make the asymmetry continuous-in, discrete-out and I learn the quantizer *jointly*, end-to-end. The Transformer's *input* stays the rich continuous latents (so all information remains available), while the contrastive *targets* are quantized, and the codes and the context model shape each other rather than the codes being frozen first.

For the quantizer I need something differentiable. A single codebook large enough for all of speech would be huge and most entries would die, so I use product quantization: split the quantized vector into $G$ groups, each with its own small codebook of $V$ entries, pick one entry per group, and concatenate. That expresses up to $V^G$ codes from only $G\!\cdot\!V$ vectors — for $G=2,\,V=320$, a theoretical 102{,}400 codes from 640 vectors. But "pick one entry" is an argmax with no gradient, so I use the hard Gumbel-softmax. Map $z_t$ to $G\!\cdot\!V$ logits $l_{g,v}$ and form per-group probabilities

$$p_{g,v} = \frac{\exp((l_{g,v}+n_v)/\tau)}{\sum_k \exp((l_{g,k}+n_k)/\tau)}, \qquad n = -\log(-\log u),\; u\sim\mathrm{Uniform}(0,1),$$

with the division by $\tau$ *inside* the exponential — otherwise the temperature would cancel out of the softmax and do nothing. On the forward pass I take the hard choice $i=\arg\max_v p_{g,v}$, a genuinely discrete code; on the backward pass gradients flow through the soft $p$, the straight-through estimator. I anneal $\tau$ from $2$ down toward $0.5$ (or $0.1$ for the larger model) multiplying by $0.999995$ per update: high $\tau$ early keeps the distribution soft so every code gets gradient and the codebook does not collapse, low $\tau$ late makes the choice crisp and close to the true argmax. Concatenating the selected per-group entries and applying a linear map yields $q_t$, and since the Transformer width (768) need not match the product-quantized target width (256), a learned projection sends both branches into a shared final space before the cosine comparison.

A failure mode lurks: nothing forces the model to *use* all the codes. The contrastive task only needs enough codes to tell positives from distractors, so the degenerate solution collapses onto a handful of codes, the task becomes trivial, and the discretization carries no information. The lever is the batch-averaged code distribution. Over a batch I average the *noise-free, temperature-free* softmax probabilities for group $g$ to get $\bar p_g$, and I reward high entropy of $\bar p_g$ — uniform usage means every code fires equally often. Since entropy is maximal at uniform, I maximize $H(\bar p_g)$, i.e. minimize $-H(\bar p_g)$, writing the diversity penalty as

$$L_d = \frac{1}{GV}\sum_g -H(\bar p_g) = \frac{1}{GV}\sum_g \sum_v \bar p_{g,v}\log \bar p_{g,v}.$$

The inner $\sum_v \bar p\log\bar p = -H(\bar p_g)\le 0$ is most negative exactly when $\bar p_g$ is uniform, so minimizing $L_d$ spreads usage across all $V$ entries in every group (equivalently, it maximizes codebook perplexity). The total objective is

$$L = L_m + \alpha\, L_d,$$

with $\alpha\approx 0.1$ — large enough to prevent collapse, small enough not to swamp the real task.

Where I mask matters. I mask in the *latent space*, replacing a chosen subset of $z$ time steps with a single trained mask vector shared across all masked positions, *before* the Transformer. But the targets come from the quantizer applied to the *un-masked* $z$: the quantizer's job is to produce the ground-truth code for each frame, so masking its input would leave nothing to predict. Thus the Transformer sees a partly masked continuous sequence and at each masked position must identify, from context, the quantized code of the hidden frame. Masking single isolated frames is too easy — a frame is nearly determined by its neighbors a few milliseconds away — so I mask *spans*. Each frame is independently chosen as a span start with probability $p=0.065$, and from each start I mask the next $M=10$ consecutive frames, allowing overlaps; over a $\sim$15 s clip this yields a mean masked span around 14.7 frames ($\sim$299 ms) and roughly 49% of frames masked, so each prediction needs genuine long-range reasoning, not local copying. The $K=100$ distractors for a masked frame are drawn uniformly from the *other masked* time steps of the *same* utterance: same-utterance negatives share speaker and channel, so the model cannot win by exploiting those cues and must discriminate on phonetic content — exactly what I want it to encode.

Positions need care too. Fixed sinusoidal or learned-absolute encodings assume a meaningful absolute index, but for speech what matters is the *relative* offset between frames, and utterances vary wildly in length. So instead of an additive positional table I run a grouped 1-D convolution (kernel 128, 16 groups) over the sequence and add its GELU output: a convolution is inherently relative — it sees a frame's neighborhood by offset — and generalizes across lengths.

Then fine-tuning. After pre-training, the context network emits phonetically informed frame representations. To recognize, I add one randomly initialized linear layer mapping each $c_t$ to the output vocabulary — LibriSpeech characters plus a word-boundary token and the CTC blank — and train it with CTC, which handles the missing frame-level alignment by summing over all blank-augmented alignments that collapse to the transcript. I keep the convolutional feature encoder *frozen*: it already learned a good low-level representation, and updating it on tiny labeled data would just overfit. I also apply SpecAugment-style time and channel masking of the encoder outputs to resist overfitting when labels are scarce. The quantizer and diversity loss exist only to *create* targets during pre-training; at fine-tuning time they are gone.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, stride)
        self.norm = nn.LayerNorm(c_out)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.conv(x).transpose(1, 2)
        x = self.act(self.norm(x))
        return x.transpose(1, 2)

class ConvFeatureEncoder(nn.Module):                 # waveform -> latent frames z
    def __init__(self, dims=(512,)*7,
                 kernels=(10,3,3,3,3,2,2), strides=(5,2,2,2,2,2,2)):
        super().__init__()
        layers, c_in = [], 1
        for c_out, k, s in zip(dims, kernels, strides):
            layers.append(ConvBlock(c_in, c_out, k, s))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
    def forward(self, wav):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / wav.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return self.conv(wav.unsqueeze(1)).transpose(1, 2)        # (B, T, 512)

class GumbelProductQuantizer(nn.Module):             # continuous z -> discrete target q
    def __init__(self, in_dim=512, G=2, V=320, code_dim=256, final_dim=256):
        super().__init__()
        self.G, self.V = G, V
        self.code_dim = code_dim
        self.weight   = nn.Linear(in_dim, G * V)
        self.codebook = nn.Parameter(torch.randn(G, V, code_dim // G))
        self.proj     = nn.Linear(code_dim, final_dim)
        self.tau = 2.0
    def forward(self, z):
        B, T, _ = z.shape
        logits = self.weight(z).view(B, T, self.G, self.V)
        oh = F.gumbel_softmax(logits.reshape(B * T * self.G, self.V),
                              tau=self.tau, hard=True).view(B * T, self.G, self.V)
        codes = torch.einsum("bgv,gvd->bgd", oh, self.codebook).reshape(B, T, self.code_dim)
        q = self.proj(codes)
        probs = F.softmax(logits, dim=-1).mean(dim=(0, 1))   # (G,V), no Gumbel noise/tau
        return q, probs

class TransformerContext(nn.Module):
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

def mask_spans(x, mask_emb, p=0.065, M=10):
    B, T, _ = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for b in range(B):
        for s in (torch.rand(T, device=x.device) < p).nonzero().flatten():
            mask[b, s:s + M] = True
    x = x.clone(); x[mask] = mask_emb
    return x, mask

def diversity_loss(probs):                            # probs (G,V), batch-averaged
    return (probs * (probs + 1e-7).log()).sum() / probs.numel()   # = (1/GV) sum -H(p_g)

def contrastive_loss(c, q, mask, K=100, kappa=0.1):
    total, n = c.new_tensor(0.0), 0
    for b in range(c.size(0)):
        idx = mask[b].nonzero().flatten()
        if idx.numel() < 2:
            continue
        cb, qb = c[b, idx], q[b, idx]
        sim = F.cosine_similarity(cb.unsqueeze(1), qb.unsqueeze(0), dim=-1) / kappa
        m = idx.numel()
        row = torch.arange(m, device=c.device).unsqueeze(1)
        negs = torch.randint(0, m - 1, (m, K), device=c.device)
        negs = negs + (negs >= row).long()              # exclude the positive index
        logits = torch.cat([sim.diag().unsqueeze(1), sim.gather(1, negs)], dim=1)
        total += F.cross_entropy(logits, torch.zeros(m, dtype=torch.long, device=c.device), reduction="sum")
        n += m
    return total / max(n, 1)

def pretrain_step(wav, enc, quant, ctx, in_proj, context_proj, mask_emb, alpha=0.1):
    z = enc(wav)
    q, probs = quant(z)                               # targets from UN-masked z
    x, mask = mask_spans(in_proj(z), mask_emb)        # mask the context input only
    c = context_proj(ctx(x))                          # same final dimension as q
    return contrastive_loss(c, q, mask) + alpha * diversity_loss(probs)

class CTCHead(nn.Module):                             # fine-tuning: chars + boundary + blank
    def __init__(self, d=768, n_vocab=31):
        super().__init__()
        self.proj = nn.Linear(d, n_vocab)
    def forward(self, c):
        return self.proj(c).log_softmax(-1)

def apply_finetune_masking(x, mask_emb, time_prob, channel_prob,
                           time_len=10, channel_len=64):
    y = x.clone()
    time_mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
    for b in range(x.size(0)):
        for s in (torch.rand(x.size(1), device=x.device) < time_prob).nonzero().flatten():
            time_mask[b, s:s + time_len] = True
    y[time_mask] = mask_emb
    channel_mask = torch.zeros(x.size(0), x.size(2), dtype=torch.bool, device=x.device)
    for b in range(x.size(0)):
        for s in (torch.rand(x.size(2), device=x.device) < channel_prob).nonzero().flatten():
            channel_mask[b, s:s + channel_len] = True
    return y.masked_fill(channel_mask.unsqueeze(1), 0.0)

def finetune_step(wav, enc, in_proj, ctx, head, mask_emb, time_prob, channel_prob,
                  targets, in_lens, tgt_lens):
    with torch.no_grad():
        z = enc(wav)
    c = ctx(apply_finetune_masking(in_proj(z), mask_emb, time_prob, channel_prob))
    return F.ctc_loss(head(c).transpose(0, 1), targets, in_lens, tgt_lens, blank=0)
```
