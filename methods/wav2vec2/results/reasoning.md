Let me start from what actually hurts. A speech recognizer that works needs thousands of hours of transcribed audio, and almost no language on Earth has that. But raw audio — people talking — is everywhere, for nearly every language. So the asymmetry I want to exploit is stark: unlimited untranscribed speech, almost no transcripts. If I could learn the *structure* of speech from the raw audio alone, then fitting an actual recognizer on top should need only a sliver of labels. That's the goal: pre-train a representation from audio with no transcripts, then fine-tune to recognition with maybe ten minutes of labeled data.

So what do I have to learn from? Just a waveform — a one-dimensional stream of samples at 16 kHz. No segmentation into words, no phoneme boundaries, no tokens. That last part is the thing that makes speech different from text and harder than vision. In language the masked-LM trick works because the input already comes pre-discretized into word-pieces: I can hide a token and ask the model to name it from a fixed vocabulary, scoring with a softmax cross-entropy. Speech has no such vocabulary handed to me. And it isn't a single static image either, so the vision recipe of "treat each image and its augmentations as one class" doesn't map — an utterance contains many sounds, not one instance. I have a continuous-valued *sequence* with no given units. Whatever pretext task I invent has to manufacture its own prediction targets out of the continuous stream.

Raw samples are far too fine-grained to feed a sequence model directly — 16,000 numbers per second — so I normalize each waveform and run a stack of strided 1-D convolutions over it to get a latent frame every ~20 ms. Seven temporal-convolution blocks with 512 channels, strides (5,2,2,2,2,2,2), kernels (10,3,3,3,3,2,2), layer normalization, and GELU give about 49 latent frames per second with a 25 ms receptive field. Convolution is the natural translation-equivariant local feature extractor, and the strides downsample. Call these latents z₁,…,z_T. Then I want a network that contextualizes each frame against the whole utterance — for that I'll use a Transformer, because self-attention is all-to-all and gives each frame a representation conditioned on the entire sequence, which is what a recognizer ultimately needs. Call those contextualized outputs c₁,…,c_T. The unsolved part is the objective that trains this stack from audio alone.

The obvious move is to copy BERT: mask some frames and predict them. But predict them *as what*? In BERT the target is a token id and I score cross-entropy against the vocabulary. Here the "frame" is a continuous vector z_t. I could regress it — predict the masked z_t with an L2 loss. Let me think about whether that's a good target. A waveform latent carries everything: the phonetic content, yes, but also the speaker's pitch, the room reverberation, the background hum. If I make the model regress the exact continuous latent, I'm forcing it to reproduce all of that nuisance detail, and worse, the easiest way to drive an L2 loss down is to nail the low-level acoustic texture, not the linguistic structure. The target is too rich; it rewards the wrong thing. That's the same reason future-sample reconstruction is a bad idea — modeling the raw signal spends all the capacity on detail I don't care about for recognition.

So don't reconstruct. Predict by *classification* instead, the way contrastive predictive coding does: I don't have to name the exact target if I only have to *pick it out* from a set of wrong answers. Given the context vector c_t at a masked frame, present it with the true target for that frame plus a handful of distractor targets drawn from elsewhere, and train it to identify the true one. That's a noise-contrastive / InfoNCE objective, and it sidesteps the "predict an exact continuous vector" trap — I never have to generate anything, I only have to score similarity. Concretely, with cosine similarity sim and a temperature κ, the loss at a masked frame t is

  L = −log[ exp(sim(c_t, q_t)/κ) / Σ_{q̃ ∈ Q_t} exp(sim(c_t, q̃)/κ) ],

where Q_t is the set containing the true target q_t and K distractors. Minimizing this maximizes the similarity of c_t to its true target relative to the distractors.

But what *is* the target q_t? If I make the target the continuous latent z_t itself, I'm back to the nuisance problem in a softer form: the distractors and the positive are all continuous latents from the same utterance, so they all share the same speaker and room, and the easiest discriminator might still latch onto something other than phonetic content. There's a deeper worry too — if targets are continuous, a frame's target can encode fine artifacts specific to *that* moment in *this* recording, which makes the contrastive task too easy and the learned representation less general. Let me think about what would make a better target.

What if I *discretize* the target? Quantize z_t to a code from a small learned inventory, and let the contrastive task be "pick the right code." Discretization is a lossy bottleneck by construction: it throws away the continuous nuisance detail and keeps only a coarse, shared identity. If it works the way I hope, a handful of frames that are all the /s/ sound would map to the same code regardless of speaker, so the target becomes phonetically meaningful instead of recording-specific — though whether the codes actually align with phonetic categories is something I can only confirm empirically later, not assert here. This is also what the two-step systems were groping toward when they vector-quantized audio first — but they froze the quantizer before the context model ever ran, so the two could never co-adapt, and the context model only ever saw the lossy tokens as *input*. I can do better on both counts.

The asymmetry has to be continuous latents as the **input** to the Transformer — all the information stays available to the context network — but discretized latents as the **targets** of the contrastive loss: coarse, denoised, phonetically meaningful things to predict. Continuous in, discrete out. That keeps the context representations rich while making the prediction target robust. And crucially I'll learn the quantizer *jointly*, end-to-end with the rest, not in a frozen first pass — the codes and the context model shape each other.

Now I need a quantizer I can backprop through. A single codebook large enough to cover all of speech would be enormous and most entries would die. Product quantization offers a different shape: split the quantized vector into G groups and give each group its own small codebook of V entries; pick one entry per group and concatenate. The reason this is attractive is a combinatorial one I should make concrete rather than wave at. With G independent groups each choosing one of V entries, the number of distinct concatenated codes is Vᴳ, while the number of stored vectors is only G·V. Plug in the sizes I have in mind, G=2 and V=320: Vᴳ = 320² = 102,400 expressible codes, from G·V = 2·320 = 640 stored vectors. So 640 parameters' worth of codebook span six-figures of codes — the multiplicative blow-up is exactly what lets a small inventory cover the acoustic space without most entries dying. That settles the choice of quantizer structure. But "pick one entry" is an argmax, which has no gradient. The Gumbel-softmax handles that: map z_t to G·V logits l_{g,v}, and form per-group probabilities

  p_{g,v} = exp((l_{g,v} + n_v)/τ) / Σ_k exp((l_{g,k} + n_k)/τ),

with n = −log(−log(u)), u ~ Uniform(0,1) the Gumbel noise, and τ the temperature. The division by τ has to be *inside* the exponential, and I want to be sure I have that right rather than placing τ by reflex. Take logits [2.0, 1.0, 0.1] at τ=0.5. Doing it correctly — softmax of [2.0/.5, 1.0/.5, 0.1/.5] — gives [0.864, 0.117, 0.019], a sharpened distribution, which is what a low temperature should do. Now the wrong placement: compute the ordinary softmax(l) = [0.659, 0.242, 0.099], then scale by 1/τ and renormalize. Scaling every entry by the same 1/τ and renormalizing divides it straight back out, returning [0.659, 0.242, 0.099] — identical to softmax(l), with τ having no effect at all. So an outside-the-exp temperature is inert; it must sit inside. On the forward pass I take the hard choice i = argmax_v p_{g,v} (a real discrete code), but on the backward pass I let gradients flow through the soft p — the straight-through estimator. So the selection is genuinely discrete yet differentiable. I anneal τ from 2 down toward 0.5 for the base-sized model, or 0.1 for the larger one, multiplying by 0.999995 at each update: early on, high τ keeps the distribution soft so every code gets gradient and the codebook doesn't collapse to a few entries; late, low τ makes the choice crisp and close to the true argmax. Select one entry per group, concatenate, and a linear map gives the quantized target q_t. Since the Transformer state and the quantized target need not have the same width — the base Transformer is 768-dimensional while the product-quantized target can be 256-dimensional before the final comparison — I also need a learned projection so both branches meet in the same final contrastive space before cosine similarity.

There's a failure mode lurking: nothing forces the model to *use* all the codes. The contrastive task only needs enough codes to tell positives from distractors. The degenerate solution is to collapse onto a tiny handful of codes — then positives and distractors all share codes, the task becomes trivial-and-useless, and the discretization carries no information. I need a term that pushes toward using the whole codebook. The lever I have is the averaged code distribution: over a batch, average the (noise-free, temperature-free) softmax probabilities for group g to get p̄_g, a length-V vector of how often each code is selected. Spread-out usage means p̄_g is close to uniform; collapse means p̄_g is spiky. So I want a scalar that is small when p̄_g is uniform and large when it is spiky, to minimize. The natural candidate is negative entropy, Σ_v p̄ log p̄ = −H(p̄_g), which I'd add as

  L_d = (1/GV) Σ_g −H(p̄_g) = (1/GV) Σ_g Σ_v p̄_{g,v} log p̄_{g,v}.

I should not just trust the sign of this; let me actually evaluate Σ_v p̄ log p̄ on a few length-4 distributions and check which way it points. Uniform [.25,.25,.25,.25] gives Σ = −1.386. A peaked [.7,.1,.1,.1] gives −0.940. A near-collapsed [.97,.01,.01,.01] gives −0.168. So the value is *most negative* at the uniform distribution and rises toward zero as usage concentrates — which means minimizing L_d does pull p̄_g toward uniform and away from collapse, the direction I wanted. And the uniform number, −1.386, is exactly −log 4 = −log V, so H is maximal there; its exponential, the perplexity exp(H), is 4.0 = V, i.e. all four codes effectively in use, versus 2.56 for the peaked case. So I could equivalently phrase the term as maximizing perplexity via (GV − Σ_g exp(H(p̄_g)))/(GV); same optimum, and the perplexity reading gives me a cheap diagnostic I can log to *watch* for collapse during training. I'll keep the entropy form for the loss. The total objective is the contrastive loss plus a small weight on diversity,

  L = L_contrastive + α L_d,

with α tuned modestly — large enough to prevent collapse, small enough not to swamp the real task. α around 0.1 sits there.

Now the masking, and I have to be careful about *where* in the pipeline I mask. I mask in the **latent space**, not the raw waveform: I take the encoder outputs z and, before feeding them to the Transformer, replace a chosen subset of time steps with a single trained mask vector shared across all masked positions. The targets, though, come from the quantizer applied to the *un-masked* z — I do not mask the input to the quantizer, because the quantizer's job is to produce the ground-truth code for each frame; if I masked its input I'd have nothing to predict. So the Transformer sees a corrupted (partly masked) continuous sequence, and at each masked position it must, from surrounding context, identify the quantized code of the frame that was hidden there.

How much to mask, and in what shape? Masking single isolated frames is too easy — a frame is highly predictable from its immediate neighbors a few milliseconds away, so the task becomes near-trivial interpolation and the representation learns little. I want spans long enough that recovering them demands real context. So I sample starting indices: each frame is independently chosen as a span start with probability p, and from each start I mask the next M consecutive frames; spans may overlap. I need to pick p and M so that a sensible fraction is masked, and I can't read that fraction straight off p — overlapping M-frame spans merge, so the masked fraction is below the naive p·M and the realized spans are longer than M. Let me actually pin the numbers down for a ~15 s clip, which at the encoder's ~49 Hz is about 749 frames, with p = 0.065 and M = 10. The naive start count is 749·0.065 ≈ 49 starts per clip. To get the masked fraction and span length right with the merging, I simulate the sampling a few thousand times and measure: the mean masked fraction comes out to 0.487, and the mean contiguous masked span to 14.6 frames — at 20 ms per frame, about 293 ms. So roughly half the sequence is hidden in spans averaging ~0.3 s, longer than the M=10 I seeded because adjacent starts fuse. That's the band I want — about half the sequence hidden, in chunky spans, so each prediction needs genuine long-range acoustic and temporal reasoning, not local copying. (Were it ever too easy or too hard, p is the dial that moves the fraction and M the dial that moves the span length.) The distractors for a given masked frame I draw uniformly from the *other masked* time steps of the same utterance — same-utterance negatives share speaker and channel, so the model can't win by exploiting those cues; it has to discriminate on phonetic content, which is exactly what I want it to encode. K = 100 distractors gives a hard enough K+1-way decision.

Waveform → conv encoder → continuous z. Quantizer(z) → discrete targets q (this branch is unmasked). Mask spans of z, feed to Transformer → context c, then project c into the same final space as q. At each masked t, the contrastive loss pushes c_t toward q_t and away from 100 same-utterance distractor codes; the diversity loss keeps the codebook fully used. Continuous information flows into the context; coarse phonetic codes are the targets. Nothing is reconstructed; nothing requires a label. Good.

Positions need care. Fixed sinusoidal or learned-absolute positional encodings assume a meaningful absolute index, but for speech what matters is *relative* offset between frames, and utterances vary wildly in length. So instead of an additive positional table I run a grouped 1-D convolution over the input sequence and add its output after GELU; a convolution is inherently relative — it sees a frame's neighborhood by offset — and it generalizes across lengths. Kernel 128 with 16 groups gives a wide relative window cheaply. The conv encoder blocks use layer normalization and GELU; GELU is the smooth activation the Transformer LMs favor.

Then fine-tuning. After pre-training I have a context network that emits phonetically-informed frame representations. To recognize, I add a single randomly-initialized linear layer mapping each c_t to the output vocabulary — LibriSpeech character targets plus a word-boundary token and the CTC blank — and train it with CTC, which handles the lack of frame-level alignment by summing over all blank-augmented alignments that collapse to the transcript. I keep the convolutional feature encoder *frozen* during fine-tuning — it learned a good low-level representation already and updating it on tiny labeled data would just overfit — and I apply a SpecAugment-style masking of the encoder outputs (time and channel masks) to resist overfitting when labels are scarce. The quantizer and diversity loss are only needed to *create* targets during pre-training; at fine-tuning time they're gone.

The pieces line up in code.

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

class ConvFeatureEncoder(nn.Module):                 # raw waveform -> latent frames z
    def __init__(self, dims=(512,)*7,
                 kernels=(10,3,3,3,3,2,2), strides=(5,2,2,2,2,2,2)):
        super().__init__()
        layers, c_in = [], 1
        for c_out, k, s in zip(dims, kernels, strides):
            layers.append(ConvBlock(c_in, c_out, k, s))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
    def forward(self, wav):                          # (B, T_samples)
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / wav.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return self.conv(wav.unsqueeze(1)).transpose(1, 2)   # (B, T, 512), ~20ms frames

class GumbelProductQuantizer(nn.Module):             # continuous z -> discrete TARGET q
    def __init__(self, in_dim=512, G=2, V=320, code_dim=256, final_dim=256):
        super().__init__()
        self.G, self.V = G, V
        self.code_dim = code_dim
        self.weight   = nn.Linear(in_dim, G * V)             # z -> G*V logits
        self.codebook = nn.Parameter(torch.randn(G, V, code_dim // G))
        self.proj     = nn.Linear(code_dim, final_dim)
        self.tau = 2.0                                       # annealed toward ~0.5/0.1
    def forward(self, z):
        B, T, _ = z.shape
        logits = self.weight(z).view(B, T, self.G, self.V)
        flat_logits = logits.reshape(B * T * self.G, self.V)
        # hard Gumbel-softmax: discrete pick forward, soft gradient backward
        oh = F.gumbel_softmax(flat_logits, tau=self.tau, hard=True).view(B * T, self.G, self.V)
        codes = torch.einsum("bgv,gvd->bgd", oh, self.codebook).reshape(B, T, self.code_dim)
        q = self.proj(codes)                                      # concat groups, linear
        # noise/temperature-free averaged usage, for the diversity term
        probs = F.softmax(logits, dim=-1).mean(dim=(0, 1))          # (G, V)
        return q, probs

class TransformerContext(nn.Module):                 # contextualize the (masked) latents
    def __init__(self, d=768, layers=12, heads=8, ffn=3072):
        super().__init__()
        self.pos_conv = nn.Conv1d(d, d, 128, padding=64, groups=16)  # relative positions
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

The masking samples span-starts and stamps a shared trained mask vector over each span; the quantizer runs on the *un*-masked latents so the targets are intact:

```python
def mask_spans(z_proj, mask_emb, p=0.065, M=10):
    B, T, _ = z_proj.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=z_proj.device)
    for b in range(B):
        starts = (torch.rand(T, device=z_proj.device) < p).nonzero().flatten()
        for s in starts:                                   # expand each start to M frames
            mask[b, s:s + M] = True                        # spans may overlap
    x = z_proj.clone()
    x[mask] = mask_emb                                     # shared learned mask vector
    return x, mask
```

The contrastive loss at masked frames, with same-utterance distractors and cosine similarity, plus the diversity penalty:

```python
def contrastive_loss(c, q, mask, K=100, kappa=0.1):
    # c: context outputs (B,T,d_q); q: quantized targets (B,T,d_q); mask: (B,T) bool
    loss, n_targets = c.new_tensor(0.0), 0
    for b in range(c.size(0)):
        idx = mask[b].nonzero().flatten()                  # masked frame indices
        if idx.numel() < 2:
            continue
        cb, qb = c[b, idx], q[b, idx]                      # (m, d), aligned positive at i
        sim = F.cosine_similarity(cb.unsqueeze(1), qb.unsqueeze(0), dim=-1) / kappa  # (m,m)
        m = idx.numel()
        row = torch.arange(m, device=c.device).unsqueeze(1)
        negs = torch.randint(0, m - 1, (m, K), device=c.device)
        negs = negs + (negs >= row).long()                 # sample other masked frames
        logits = torch.cat([sim.diag().unsqueeze(1), sim.gather(1, negs)], dim=1)
        loss = loss + F.cross_entropy(logits, torch.zeros(m, dtype=torch.long, device=c.device), reduction="sum")
        n_targets += m
    return loss / max(n_targets, 1)

def diversity_loss(probs):                                 # probs: (G, V), batch-averaged
    # minimize sum_g sum_v p log p = -sum_g H(p_g): pushes averaged usage to uniform
    return (probs * (probs + 1e-7).log()).sum() / probs.numel()

def pretrain_step(wav, enc, quant, ctx, in_proj, context_proj, mask_emb, alpha=0.1):
    z = enc(wav)                                           # (B,T,512) continuous
    q, probs = quant(z)                                    # discrete targets (unmasked z)
    x = in_proj(z)                                         # project latents to model dim
    x, mask = mask_spans(x, mask_emb)                      # mask the CONTEXT input only
    c = context_proj(ctx(x))                               # shared contrastive dimension
    return contrastive_loss(c, q, mask) + alpha * diversity_loss(probs)
```

For recognition, the feature encoder stays frozen, the encoder outputs get time masking plus channel spans set to zero, and the fresh linear head trains with CTC. The mask probabilities can change with the labeled-data setting; the time span length is ten frames and the channel span length is 64.

```python
class CTCHead(nn.Module):
    def __init__(self, d=768, n_vocab=31):                 # Libri chars + boundary + CTC blank
        super().__init__()
        self.proj = nn.Linear(d, n_vocab)
    def forward(self, c):
        return self.proj(c).log_softmax(-1)

def apply_finetune_masking(x, mask_emb, time_prob, channel_prob,
                           time_len=10, channel_len=64):
    y = x.clone()
    time_mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
    for b in range(x.size(0)):
        starts = (torch.rand(x.size(1), device=x.device) < time_prob).nonzero().flatten()
        for s in starts:
            time_mask[b, s:s + time_len] = True
    y[time_mask] = mask_emb                                # same learned mask vector
    channel_mask = torch.zeros(x.size(0), x.size(2), dtype=torch.bool, device=x.device)
    for b in range(x.size(0)):
        starts = (torch.rand(x.size(2), device=x.device) < channel_prob).nonzero().flatten()
        for s in starts:
            channel_mask[b, s:s + channel_len] = True
    return y.masked_fill(channel_mask.unsqueeze(1), 0.0)

def finetune_step(wav, enc, in_proj, ctx, head, mask_emb, time_prob, channel_prob,
                  targets, in_lens, tgt_lens):
    with torch.no_grad():
        z = enc(wav)                                       # feature encoder FROZEN
    c = ctx(apply_finetune_masking(in_proj(z), mask_emb, time_prob, channel_prob))
    logp = head(c).transpose(0, 1)                         # (T, B, V) for CTC
    return F.ctc_loss(logp, targets, in_lens, tgt_lens, blank=0)
```

Everything now fits the original pressure: I need speech representations from raw audio without transcripts, so I encode the waveform to continuous latent frames and contextualize them with a Transformer; I avoid reconstructing continuous targets because that rewards nuisance acoustic detail, and instead make each masked position identify its true target among same-utterance distractors; I make those targets robust by discretizing the unmasked latents with a jointly learned product quantizer and hard Gumbel-softmax while keeping the Transformer's input continuous; I add the negative-entropy diversity term so the codebooks stay used; I mask overlapping spans so the task cannot collapse into local interpolation; and for recognition I discard the quantizer, freeze the feature encoder, and train a linear CTC head on top of the contextual representations.
