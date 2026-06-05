Let me start from what actually hurts. A speech recognizer that works needs thousands of hours of transcribed audio, and almost no language on Earth has that. But raw audio — people talking — is everywhere, for nearly every language. So the asymmetry I want to exploit is stark: unlimited untranscribed speech, almost no transcripts. If I could learn the *structure* of speech from the raw audio alone, then fitting an actual recognizer on top should need only a sliver of labels. That's the goal: pre-train a representation from audio with no transcripts, then fine-tune to recognition with maybe ten minutes of labeled data.

So what do I have to learn from? Just a waveform — a one-dimensional stream of samples at 16 kHz. No segmentation into words, no phoneme boundaries, no tokens. That last part is the thing that makes speech different from text and harder than vision. In language the masked-LM trick works because the input already comes pre-discretized into word-pieces: I can hide a token and ask the model to name it from a fixed vocabulary, scoring with a softmax cross-entropy. Speech has no such vocabulary handed to me. And it isn't a single static image either, so the vision recipe of "treat each image and its augmentations as one class" doesn't map — an utterance contains many sounds, not one instance. I have a continuous-valued *sequence* with no given units. Whatever pretext task I invent has to manufacture its own prediction targets out of the continuous stream.

Let me get the front end out of the way first, because it's not where the difficulty is. Raw samples are far too fine-grained to feed a sequence model directly — 16,000 numbers per second. I'll run a stack of strided 1-D convolutions over the waveform to get a latent frame every ~20 ms; convolution is the natural translation-equivariant local feature extractor, and the strides downsample. Call these latents z₁,…,z_T. Then I want a network that contextualizes each frame against the whole utterance — for that I'll use a Transformer, because self-attention is all-to-all and gives each frame a representation conditioned on the entire sequence, which is what a recognizer ultimately needs. Call those contextualized outputs c₁,…,c_T. None of that is the contribution; it's the harness. The question is the objective that trains it from audio alone.

The obvious move is to copy BERT: mask some frames and predict them. But predict them *as what*? In BERT the target is a token id and I score cross-entropy against the vocabulary. Here the "frame" is a continuous vector z_t. I could regress it — predict the masked z_t with an L2 loss. Let me think about whether that's a good target. A waveform latent carries everything: the phonetic content, yes, but also the speaker's pitch, the room reverberation, the background hum. If I make the model regress the exact continuous latent, I'm forcing it to reproduce all of that nuisance detail, and worse, the easiest way to drive an L2 loss down is to nail the low-level acoustic texture, not the linguistic structure. The target is too rich; it rewards the wrong thing. That's the same reason future-sample reconstruction is a bad idea — modeling the raw signal spends all the capacity on detail I don't care about for recognition.

So don't reconstruct. Predict by *classification* instead, the way contrastive predictive coding does: I don't have to name the exact target if I only have to *pick it out* from a set of wrong answers. Given the context vector c_t at a masked frame, present it with the true target for that frame plus a handful of distractor targets drawn from elsewhere, and train it to identify the true one. That's a noise-contrastive / InfoNCE objective, and it sidesteps the "predict an exact continuous vector" trap — I never have to generate anything, I only have to score similarity. Concretely, with cosine similarity sim and a temperature κ, the loss at a masked frame t is

  L = −log[ exp(sim(c_t, q_t)/κ) / Σ_{q̃ ∈ Q_t} exp(sim(c_t, q̃)/κ) ],

where Q_t is the set containing the true target q_t and K distractors. Minimizing this maximizes the similarity of c_t to its true target relative to the distractors.

But what *is* the target q_t? If I make the target the continuous latent z_t itself, I'm back to the nuisance problem in a softer form: the distractors and the positive are all continuous latents from the same utterance, so they all share the same speaker and room, and the easiest discriminator might still latch onto something other than phonetic content. There's a deeper worry too — if targets are continuous, a frame's target can encode fine artifacts specific to *that* moment in *this* recording, which makes the contrastive task too easy and the learned representation less general. Let me think about what would make a better target.

What if I *discretize* the target? Quantize z_t to a code from a small learned inventory, and let the contrastive task be "pick the right code." Discretization is exactly a lossy bottleneck: it throws away the continuous nuisance detail and keeps a coarse, shared identity. A handful of frames that are all the /s/ sound should map to the same code regardless of speaker, so the target becomes phonetically meaningful instead of recording-specific. This is also what the two-step systems were groping toward when they vector-quantized audio first — but they froze the quantizer before the context model ever ran, so the two could never co-adapt, and the context model only ever saw the lossy tokens as *input*. I can do better on both counts.

Here's the asymmetry I want. Use *continuous* latents as the **input** to the Transformer — keep all the information there, so the context network has the richest possible signal to reason over. But use *discretized* latents as the **targets** of the contrastive loss — coarse, denoised, phonetically meaningful things to predict. Continuous in, discrete out. That keeps the context representations rich while making the prediction target robust. And crucially I'll learn the quantizer *jointly*, end-to-end with the rest, not in a frozen first pass — the codes and the context model shape each other.

Now I need a quantizer I can backprop through. A single codebook large enough to cover all of speech would be enormous and most entries would die. Product quantization fixes this: split the quantized vector into G groups and give each group its own small codebook of V entries; pick one entry per group and concatenate. G codebooks of V entries express up to Vᴳ codes with only G·V vectors — for G=2, V=320 that's a theoretical 102,400 codes from 640 vectors. But "pick one entry" is an argmax, which has no gradient. The Gumbel-softmax handles that: map z_t to G·V logits l_{g,v}, and form per-group probabilities

  p_{g,v} = exp((l_{g,v} + n_v)/τ) / Σ_k exp((l_{g,k} + n_k)/τ),

with n = −log(−log(u)), u ~ Uniform(0,1) the Gumbel noise, and τ the temperature. On the forward pass I take the hard choice i = argmax_v p_{g,v} (a real discrete code), but on the backward pass I let gradients flow through the soft p — the straight-through estimator. So the selection is genuinely discrete yet differentiable. I anneal τ from high (say 2) down toward a small value over training: early on, high τ keeps the distribution soft so every code gets gradient and the codebook doesn't collapse to a few entries; late, low τ makes the choice crisp and close to the true argmax. Select one entry per group, concatenate, and a linear map gives the quantized target q_t.

There's a failure mode lurking: nothing forces the model to *use* all the codes. The contrastive task only needs enough codes to tell positives from distractors. The degenerate solution is to collapse onto a tiny handful of codes — then positives and distractors all share codes, the task becomes trivial-and-useless, and the discretization carries no information. I need to actively push toward using the whole codebook. The lever is the averaged code distribution: over a batch, average the (noise-free, temperature-free) softmax probabilities for group g to get p̄_g, and reward high entropy of p̄_g — a uniform p̄_g means every code is used equally often. Entropy is maximal when uniform, so I want to *maximize* H(p̄_g), i.e. minimize −H(p̄_g). Write the diversity penalty as

  L_d = (1/GV) Σ_g −H(p̄_g) = (1/GV) Σ_g Σ_v p̄_{g,v} log p̄_{g,v}.

That inner sum Σ_v p̄ log p̄ equals −H(p̄_g) ≤ 0, smallest (most negative) exactly when p̄_g is uniform — so minimizing L_d drives the averaged usage toward uniform, spreading mass across all V entries in every group. (Equivalently one maximizes the perplexity (GV − Σ_g exp(−Σ_v p̄_{gv} log p̄_{gv}))/GV; same thing.) The total objective is the contrastive loss plus a small weight on diversity,

  L = L_contrastive + α L_d,

with α tuned modestly — large enough to prevent collapse, small enough not to swamp the real task. α around 0.1 sits there.

Now the masking, and I have to be careful about *where* in the pipeline I mask. I mask in the **latent space**, not the raw waveform: I take the encoder outputs z and, before feeding them to the Transformer, replace a chosen subset of time steps with a single trained mask vector shared across all masked positions. The targets, though, come from the quantizer applied to the *un-masked* z — I do not mask the input to the quantizer, because the quantizer's job is to produce the ground-truth code for each frame; if I masked its input I'd have nothing to predict. So the Transformer sees a corrupted (partly masked) continuous sequence, and at each masked position it must, from surrounding context, identify the quantized code of the frame that was hidden there.

How much to mask, and in what shape? Masking single isolated frames is too easy — a frame is highly predictable from its immediate neighbors a few milliseconds away, so the task becomes near-trivial interpolation and the representation learns little. I want spans long enough that recovering them demands real context. So I sample starting indices: each frame is independently chosen as a span start with probability p, and from each start I mask the next M consecutive frames; spans may overlap. With p ≈ 0.065 and M = 10 over a ~15 s clip, the overlaps and merges yield a mean masked span around 14.7 frames (~299 ms) and roughly 49% of frames masked overall. That's the band I want — about half the sequence hidden, in chunky spans, so each prediction needs genuine long-range acoustic and temporal reasoning, not local copying. The distractors for a given masked frame I draw uniformly from the *other masked* time steps of the same utterance — same-utterance negatives share speaker and channel, so the model can't win by exploiting those cues; it has to discriminate on phonetic content, which is exactly what I want it to encode. K = 100 distractors gives a hard enough K+1-way decision.

Let me sanity-check the whole loop end to end before coding. Waveform → conv encoder → continuous z. Quantizer(z) → discrete targets q (this branch is unmasked). Mask spans of z, feed to Transformer → context c. At each masked t, the contrastive loss pushes c_t toward q_t and away from 100 same-utterance distractor codes; the diversity loss keeps the codebook fully used. Continuous information flows into the context; coarse phonetic codes are the targets. Nothing is reconstructed; nothing requires a label. Good.

A couple of architectural choices for the Transformer that I should justify rather than wave at. Positions: fixed sinusoidal or learned-absolute positional encodings assume a meaningful absolute index, but for speech what matters is *relative* offset between frames, and utterances vary wildly in length. So instead of an additive positional table I run a grouped 1-D convolution over the input sequence and add its output (after GELU); a convolution is inherently relative — it sees a frame's neighborhood by offset — and it generalizes across lengths. Kernel 128 with 16 groups gives a wide relative window cheaply. The conv encoder blocks use layer normalization and GELU; GELU is the smooth activation the Transformer LMs favor.

Then fine-tuning. After pre-training I have a context network that emits phonetically-informed frame representations. To recognize, I add a single randomly-initialized linear layer mapping each c_t to the output vocabulary — for LibriSpeech characters, 29 letters plus a word-boundary token plus a CTC blank — and train it with CTC, which handles the lack of frame-level alignment by summing over all blank-augmented alignments that collapse to the transcript. I keep the convolutional feature encoder *frozen* during fine-tuning — it learned a good low-level representation already and updating it on tiny labeled data would just overfit — and I apply a SpecAugment-style masking of the encoder outputs (time and channel masks) during fine-tuning, which delays overfitting and matters most when labels are scarce. The quantizer and diversity loss are only needed to *create* targets during pre-training; at fine-tuning time they're gone.

Let me write it, mirroring how I'd actually build it. First the front end and the joint quantizer, then the masked contrastive objective.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ConvFeatureEncoder(nn.Module):                 # raw waveform -> latent frames z
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
        return self.conv(wav.unsqueeze(1)).transpose(1, 2)   # (B, T, 512), ~20ms frames

class GumbelProductQuantizer(nn.Module):             # continuous z -> discrete TARGET q
    def __init__(self, in_dim=512, G=2, V=320, out_dim=256):
        super().__init__()
        self.G, self.V = G, V
        self.weight   = nn.Linear(in_dim, G * V)             # z -> G*V logits
        self.codebook = nn.Parameter(torch.randn(1, G * V, out_dim // G))
        self.proj     = nn.Linear(out_dim, out_dim)
        self.tau = 2.0                                       # annealed toward ~0.5/0.1
    def forward(self, z):
        B, T, _ = z.shape
        logits = self.weight(z).view(B * T * self.G, self.V)
        # hard Gumbel-softmax: discrete pick forward, soft gradient backward
        oh = F.gumbel_softmax(logits, tau=self.tau, hard=True)      # (B*T*G, V)
        codes = (oh.unsqueeze(-1) * self.codebook.view(self.G, self.V, -1)
                 .repeat(B * T, 1, 1)).sum(-2)                       # pick per group
        q = self.proj(codes.view(B, T, -1))                         # concat groups, linear
        # noise/temperature-free averaged usage, for the diversity term
        probs = F.softmax(logits.view(B * T, self.G, self.V), dim=-1).mean(0)  # (G, V)
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
    loss = 0.0
    for b in range(c.size(0)):
        idx = mask[b].nonzero().flatten()                  # masked frame indices
        cb, qb = c[b, idx], q[b, idx]                      # (m, d), aligned positive at i
        sim = F.cosine_similarity(cb.unsqueeze(1), qb.unsqueeze(0), dim=-1) / kappa  # (m,m)
        # for each masked frame, the positive is its own target (diagonal); the
        # other masked frames' targets are the distractor pool -> sample K of them
        m = idx.numel()
        for i in range(m):
            negs = torch.randperm(m, device=c.device)[:K + 1]
            negs = negs[negs != i][:K]
            cols = torch.cat([torch.tensor([i], device=c.device), negs])
            loss += F.cross_entropy(sim[i, cols].unsqueeze(0),
                                    torch.zeros(1, dtype=torch.long, device=c.device))
    return loss / mask.sum().clamp(min=1)

def diversity_loss(probs):                                 # probs: (G, V), batch-averaged
    # minimize sum_g sum_v p log p = -sum_g H(p_g): pushes averaged usage to uniform
    return (probs * (probs + 1e-7).log()).sum() / probs.numel()

def pretrain_step(wav, enc, quant, ctx, in_proj, mask_emb, alpha=0.1):
    z = enc(wav)                                           # (B,T,512) continuous
    q, probs = quant(z)                                    # discrete targets (unmasked z)
    x = in_proj(z)                                         # project latents to model dim
    x, mask = mask_spans(x, mask_emb)                      # mask the CONTEXT input only
    c = ctx(x)                                             # contextual representations
    return contrastive_loss(c, q, mask) + alpha * diversity_loss(probs)
```

And the fine-tuning head — frozen feature encoder, a fresh linear layer, CTC:

```python
class CTCHead(nn.Module):
    def __init__(self, d=768, n_vocab=32):                 # 29 chars + word-boundary + blank
        super().__init__()
        self.proj = nn.Linear(d, n_vocab)
    def forward(self, c):
        return self.proj(c).log_softmax(-1)

def finetune_step(wav, enc, ctx, head, targets, in_lens, tgt_lens):
    with torch.no_grad():
        z = enc(wav)                                       # feature encoder FROZEN
    c = ctx(in_proj_specaugment(z))                        # SpecAugment-style masking
    logp = head(c).transpose(0, 1)                         # (T, B, V) for CTC
    return F.ctc_loss(logp, targets, in_lens, tgt_lens, blank=0)
```

The causal chain, start to end: I want a speech representation learned from raw audio with no transcripts, transferable to recognition with minimal labels; I encode the waveform to continuous latent frames and contextualize them with a Transformer; I refuse to reconstruct the continuous target because that rewards nuisance acoustic detail, so I make the pretext a contrastive *classification* — identify the true target among same-utterance distractors; I make the targets robust by discretizing the latents with a jointly-learned product-quantizer-plus-Gumbel-softmax while keeping the Transformer's *input* continuous, so information is rich but targets are coarse and phonetic; I add an entropy/diversity penalty so the codebook stays fully used instead of collapsing; I mask chunky spans of the latent sequence so each prediction needs real context, drawing negatives from other masked frames of the same utterance; and at fine-tuning I freeze the feature encoder and train one linear layer with CTC.
