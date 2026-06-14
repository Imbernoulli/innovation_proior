Let me start from what I actually have and what I actually want. I have a denoiser eps_theta(x_t, t): a UNet that takes a noisy image and the noise level and predicts the noise. I train it by the simple regression — take a clean image x_0, draw a timestep t and a noise vector eps, form x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps in one shot, and minimize || eps - eps_theta(x_t, t) ||^2 averaged over t, x_0, eps. Sampling runs the thing backwards from noise. Right now it draws *some* image. What I want is to point it: give it a class label c and have it draw an image of that class, i.e. learn eps_theta(x_t, t, c) and minimize || eps - eps_theta(x_t, t, c) ||^2. The whole question is the third argument — by what operation does c get into the network. The denoiser's backbone, its group norms, its schedule, its learning rate are all tuned and I'd rather not disturb them. So I'm really choosing an injection operator and trying not to break what works.

What's already the obvious move? The network already conditions on one scalar, the noise level t, and it does it by FiLM. The integer t becomes a sinusoidal embedding, a small MLP turns that into a per-channel scale and shift, and inside every residual block I do h <- GroupNorm(h)*(1+scale) + shift. That's the established channel for "side information enters the conv stream." And a class label slots right in: embed the class to a vector the same size as the time embedding, *add* the two embeddings, and let the same FiLM path carry the sum. emb = t_emb + class_emb, done. It costs almost nothing, it's stable, and it's exactly why everyone reaches for it first. So let me actually picture what that buys me and where it runs out.

FiLM gives each channel one scale and one shift, computed from the condition, and applies that same affine at *every spatial location*. Think about what that means for steering toward a class. The condition can say "turn channel 37 up and channel 12 down, everywhere." It cannot say "near this corner of the feature map, emphasize this; over there, emphasize that," because the modulation has no spatial address and never looks at the local content of the feature it's modulating — it's the same scale and shift over the whole map, drawn from the label alone. It's a global, content-agnostic gain knob per channel. For a coarse signal like "what noise level am I at," that's plenty; the noise level genuinely is a global property of the whole image. But "which of ten classes" wants to act differently on different parts of the picture, depending on what's already forming there, and FiLM structurally can't. That's the first wall: low bandwidth, spatially uniform, blind to content.

The strongest class-conditional pixel model I know — Dhariwal and Nichol's "Diffusion Models Beat GANs" — basically confirms the diagnosis by how it had to work around it. Their in-network conditioning is adaptive group norm, which is FiLM dressed up: the class-plus-time embedding produces per-channel scale and shift on the group norm. Same global affine. And to actually get *strong* class adherence they bolt on classifier guidance: train a *separate* classifier p_phi(y|x_t) on noised images and, at every sampling step, add grad_{x_t} log p_phi(y|x_t) to the predicted score to nudge the mean toward the class. That works, but read what it's telling me: the in-network affine wasn't enough on its own, so they bought the rest of the control with a whole second model and a backward pass through it on every sampling step. I don't want to pay for an auxiliary classifier and I don't want my conditioning strength to live outside the denoiser at sampling time. I want the denoiser itself to carry strong, content-dependent conditioning. So FiLM/AdaGN by itself is out as the *strong* mechanism, and guidance-as-a-crutch is out as the *cost*.

What about the other thing people do — concatenate the condition to the input? For super-resolution or inpainting that's natural: the condition is a low-res image or a mask, image-shaped and spatially aligned to the output, so you stack it on the input channels and the first conv mixes it in. But a class index has no spatial layout to align; there's nothing to concatenate. And a caption — a variable-length bag of word vectors — has no image shape either. So concatenation just doesn't apply to label- or token-shaped conditions. Dead end for what I'm after, and it also flags something I should want from the *right* mechanism: it should handle conditions that aren't image-shaped, including ones whose size varies.

Let me restate the requirements crisply, because they're starting to point somewhere. I want an operator where (a) each spatial position of the feature map can be conditioned *differently*, as a function of its own content; (b) it can ingest a condition that is a *set* of elements of arbitrary size — one class vector today, seventy-seven caption vectors tomorrow — through the same layer; and (c) inserting it doesn't wreck the tuned base network. Requirement (a) plus (b) is the tell. I need spatial positions to *read* from a condition set, position by position, with the read depending on content. "A set of things read from another set, with content-dependent weights" is exactly what an attention layer does.

Pull up the mechanism. Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V. You have query vectors Q, key vectors K, value vectors V; each query dots with every key, a softmax over those scores gives weights, and the query's output is the weighted sum of the values. Crucially there are *two different roles*: the queries and the keys/values can come from different places. In the encoder-decoder use of attention, the queries come from one sequence and the keys and values come from another, so every query position reads over all positions of the *other* set. That's the shape I need. Map it onto my problem: the queries should be my image feature positions — I want each spatial location to do the reading — and the keys and values should be the condition. So flatten the feature map H x W into N = H*W tokens and let those be Q; let the class embedding (or the caption tokens) be the source of K and V. The output then has one row per query, i.e. per spatial position, so it comes back out shaped like a feature map and slots straight back into the convolutional stream. And because attention couldn't care less how many keys there are — softmax over one key or over seventy-seven, same layer — requirement (b) falls out for free: one class token or a whole caption, identical wiring. This is the operator.

Why this is genuinely different from FiLM, made precise: in FiLM the output at position p is (1+scale_c)*h_p + shift_c, where scale_c, shift_c depend only on the condition c, not on h_p — same affine everywhere. In attention the output at position p is sum_j softmax_j( q_p . k_j ) v_j, and q_p is a projection of h_p, so the *weights* depend on the position's own content through q_p . k_j. Position p decides for itself how much to read from each condition element, based on what p currently holds. That's content-dependent, spatially-varying conditioning — exactly the bandwidth FiLM lacked — and it didn't cost me a second model the way guidance did.

Now build it carefully, because attention has details that bite if you're sloppy. First the projections. The feature map has C channels; the condition lives in some dimension d_tau (the embedding size). I can't dot a C-dim query against a d_tau-dim key directly, and I shouldn't want to — I want learned, separate views for the three roles. So three linear maps: W_Q sends the (C-dim) feature tokens to the attention width, W_K and W_V send the (d_tau-dim) condition to the same width. Q = W_Q phi(z), K = W_K tau(c), V = W_V tau(c), where phi(z) is the flattened feature map and tau(c) is the condition encoder's output. Keys and values both come from the condition; queries from the image. Output width I'll set to C so the result adds back cleanly.

Now the softmax scaling, the 1/sqrt(d) — I want to get this from first principles, not copy it. Suppose the query and key components are roughly independent, mean zero, unit variance. A single logit is q . k = sum_{i=1}^{d_k} q_i k_i, a sum of d_k independent zero-mean unit-variance products, so it has mean 0 and variance d_k. The standard deviation of the logits therefore grows like sqrt(d_k). If d_k is, say, a few tens, the logits are spread over a range of several units; push them through softmax and it saturates toward one-hot, and right where softmax saturates its gradient is nearly zero — so the layer would barely learn. Divide each logit by sqrt(d_k) and the variance goes back to 1 regardless of width: logits stay in a sane range, softmax stays soft, gradients survive. So the scale is 1/sqrt(d_head) where d_head is the per-head key width. Not decoration — it's keeping the softmax in its useful regime.

Should I use one attention head or several? One head has to compress "everything position p might want to read from the condition" into a single softmax over a single set of projected keys. With multiple heads I project Q, K, V down to several lower-dimensional subspaces, run attention in parallel in each, concatenate, and project back — so different heads can read different aspects of the condition simultaneously instead of being forced to average them into one weighting. The averaging a single head does actively inhibits attending to several things at once. So multi-head, with the per-head width = C / num_heads so the total cost stays about that of one full-width head. For a small model I'll take a modest number of heads — four is fine; head_dim = channels // num_heads, and the 1/sqrt(d_head) scale uses that per-head width.

Where do I put this block, and how do I normalize? The UNet uses group normalization everywhere — it doesn't lean on batch statistics, so it's happy at small batch and per image — and it already has self-attention sublayers at low resolution. I'll match that: GroupNorm the feature map before projecting the queries, the way the rest of the network normalizes before its sublayers, which also keeps the query magnitudes controlled going into those dot products. And I'll insert one of these conditioning blocks after each block of the UNet — down, mid, and up — so the condition gets re-read at every scale rather than once.

Now the part that actually decides whether I can insert this without breaking the tuned denoiser. If I drop a fresh, randomly initialized attention sublayer into the middle of a network whose weights and schedule were tuned for the unconditional task, at step zero it injects random garbage into a carefully balanced residual stream and I've thrown away the good initialization — training has to first undo the damage before it can do anything useful. I want the inserted block to be the *identity* at initialization and to grow its effect only as training finds a use for it. The clean way: make the block residual, h <- h + Block(h, c), and zero-initialize the *last* projection of Block — the output projection that writes back into the stream. With W_out = 0, the block's contribution is exactly zero at init, so h <- h + 0 = h, the network is bit-for-bit the original denoiser, and as the output projection's weights move off zero during training the conditioning strength ramps up smoothly from nothing. So: out_proj initialized to zero, and a residual add around the whole thing. That's what lets me staple a brand-new sublayer onto a finished backbone for free.

Let me also settle the time path, because I changed the plan there. Originally I'd have added class_emb into time_emb for FiLM. But now *all* the class information flows through the attention sublayers — that's the whole point of building them. If I also dump the class into the time embedding, I'm splitting the conditioning across two mechanisms and muddying the time signal. Cleaner to leave the time embedding as a pure noise-level signal — prepare_conditioning just returns time_emb unchanged — and let the cross-attention carry "which class" entirely. Time tells the blocks *how noisy*; attention tells them *what to draw*. Clean separation.

Now I have to be honest about the degenerate case I'm actually in, because the task's condition is a single class label, not a caption. The conditioner for a class is one learnable embedding vector — tau(c) is a single token, M = 1. So K and V each have exactly one row. Look at what attention does with one key: softmax over a single logit is identically 1, no matter what the query is. So every spatial query gets weight 1 on the one value, and the attended branch receives the *same* vector v = W_V tau(c) at every position — the content-dependence through q_p that I was so pleased about *vanishes* when there's only one thing to attend to, because there's no competition for the softmax to resolve. Let me not oversell it: with M = 1 the attention branch itself is a learned class vector broadcast across space, projected, and added residually to the feature map. The residual preserves each position's own h_p, and later convolutions can use the class offset together with local features, but this particular softmax is not doing spatially varying routing. So for a bare class label this layer is, in effect, a learned residual injection of the class — not obviously richer than a good FiLM. The *reason* to still build it this way is generality: the very same layer, unchanged, becomes genuinely content-dependent and spatially varying the moment the condition is a set of M > 1 tokens, where the softmax has multiple keys to weigh and each query resolves them differently. A class label is the M = 1 corner of a mechanism whose payoff is that one operator covers class, text, and layout alike. I'm building the general operator and feeding it the single-token condition; that's deliberate, not an accident.

The training objective doesn't change shape, only its third argument. I still draw t, eps, form x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps, and regress the noise — but now the network sees the class too, L = E_{x_0, c, eps, t} || eps - eps_theta(x_t, t, tau(c)) ||^2, and tau and the cross-attention projections are learned jointly with the rest by the same gradient. Sampling is unchanged machinery: run the reverse process (DDIM, deterministic, a few tens of steps) feeding the fixed class index at every step.

Let me assemble the cross-attention conditioning layer concretely. Input feature map x of shape [B, C, H, W] and a condition that I can treat as a token set, with the class case represented by one token. GroupNorm x; flatten the H*W spatial positions into a sequence of C-dim query tokens; project queries from C and keys/values from the condition dimension into the multi-head attention width; reshape into heads; scaled dot product with 1/sqrt(head_dim); softmax over the key axis; weighted sum of values; merge heads; output projection back to C, *zero-initialized*; reshape to a feature map; and add residually to the input.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    # zero-init so the block is the identity at start of training
    for p in module.parameters():
        p.detach().zero_()
    return module


class CrossAttentionLayer(nn.Module):
    """Spatial features (queries) attend to the condition as keys/values.

    Q comes from the image feature map, K and V from the condition (a class
    token, or any token set). One layer of this is inserted after each UNet
    block; at init the zero-initialized output projection makes it the identity.
    """
    def __init__(self, channels, context_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads          # per-head width; total cost ~ one full head
        self.norm = nn.GroupNorm(32, channels)         # match the UNet's group-norm placement
        self.q_proj = nn.Linear(channels, channels)    # queries from image features
        self.k_proj = nn.Linear(context_dim, channels) # keys from the condition
        self.v_proj = nn.Linear(context_dim, channels) # values from the condition
        self.out_proj = zero_module(nn.Linear(channels, channels))  # identity at init -> ramps up

    def forward(self, x, context):
        B, C, H, W = x.shape
        # flatten spatial positions into a sequence of query tokens
        h = self.norm(x).view(B, C, -1).transpose(1, 2)    # [B, H*W, C]
        ctx = context.unsqueeze(1) if context.dim() == 2 else context  # [B, M, context_dim]

        # project, then split into heads: [B, heads, seq, head_dim]
        q = self.q_proj(h).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention: logits scaled by 1/sqrt(head_dim) to keep softmax unsaturated
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)                      # over the key axis (the M condition tokens)
        out = torch.matmul(attn, v)                         # [B, heads, H*W, head_dim]

        out = out.transpose(1, 2).reshape(B, H * W, C)      # merge heads back to C
        out = self.out_proj(out)                            # zero at init
        return x + out.transpose(1, 2).view(B, C, H, W)     # residual: identity at start, grows with training


def prepare_conditioning(time_emb, class_emb):
    # all class information flows through the cross-attention layers, so the time
    # embedding stays a pure noise-level signal
    return time_emb


class ClassConditioner(nn.Module):
    # the operator applied after each UNet block: cross-attention to the class token
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(channels, cond_dim, num_heads=4)

    def forward(self, h, class_emb):
        return self.cross_attn(h, class_emb)
```

So the chain is: FiLM/AdaGN conditions every position with one content-blind per-channel affine, which is enough for the noise level but too low-bandwidth for "which class," and the strongest pixel model could only get strong class control by paying for a separate classifier at sampling time — so I want the denoiser itself to carry the condition. If the condition is a set, each spatial position reading from it with content-dependent weights is exactly attention, with the image features as queries and the condition as keys/values; the 1/sqrt(d_head) scale keeps the softmax learnable, multi-head lets several aspects of the condition be read at once, group-norming before the block matches the backbone, and a zero-initialized output projection inside a residual makes the new sublayer the identity at init so it can be stapled onto the tuned denoiser and grow its effect from zero. For a single class label the mechanism degenerates to a learned residual injection (one key, trivial softmax), but it is the identical operator that, fed a caption or a layout, becomes the general content-dependent, variable-length conditioning channel — one injection operator for every modality.
