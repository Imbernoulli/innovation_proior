Let me start from the cost that's actually hurting. I have one pre-trained BERT, and tasks keep arriving — one customer task, then another, then another. The only transfer recipe that gives me top accuracy is full fine-tuning, and full fine-tuning produces a *complete* new copy of the model per task. So after N tasks I'm storing N full models. That scales linearly in N and shares nothing between tasks. I want the opposite: one shared frozen network plus a tiny per-task increment, so total size stays roughly 1× the base model no matter how many tasks I add, and a new task never forces me to touch the weights a previous task depends on.

Let me write down what "tiny per-task increment" has to mean, abstractly, before I pick an architecture. The pre-trained net is a function φ_w(x) with parameters w. Feature-based transfer keeps w frozen but composes a whole new model on top, χ_v(φ_w(x)) — train v only, but v is a full task model, and accuracy lags fine-tuning. Fine-tuning re-tunes w itself, so the delta per task is the size of w — not compact. The structure I actually want is a third thing: define a new function ψ_{w,v}(x) where w is *copied from pre-training and frozen*, and v is a *small* set of new parameters, |v| ≪ |w|. Then many tasks together cost ≈ |w| (the one shared backbone) plus N·|v| (tiny), and because w never changes, adding a task can't disturb any earlier task — perfect memory, no catastrophic forgetting, because the tasks literally don't interact through w. That's compact and extensible at once. So the whole problem reduces to: *what is ψ, and where do I inject v?*

There's a hard constraint I can't ignore. The moment I inject new randomly-initialized parameters into the middle of a deep pre-trained network and start training, I'm perturbing the activations that every downstream layer was tuned to expect. If the injected module starts as some arbitrary random transform, it corrupts the forward pass through 12 or 24 layers before the gradient has any chance to fix it, and training can diverge. So I need: at initialization, ψ_{w,v₀}(x) ≈ φ_w(x). The adapted network must start out *equal to* the pre-trained network, and only gradually deviate as v learns. That's not optional polish — it's what makes injecting modules into a frozen deep net trainable at all. Hold onto this: near-identity initialization.

Now, where to inject and what shape. The vision precedent (Rebuffi's residual adapters) added small per-domain modules into a frozen convolutional backbone and got one network serving many domains cheaply. The idea — small trainable modules in a frozen backbone — is right; but a text Transformer's structure is different and I have to figure out the placement myself. Each Transformer layer has two sub-layers: multi-head attention and a position-wise feed-forward net. Each ends by projecting back to the model dimension d, adding a residual, and applying layer norm: out = LayerNorm(x + Sublayer(x)). I want to insert my module *inside* this path, at every sub-layer, so it can re-shape the distribution of activations flowing through the network — that's the lever that actually adapts behavior, as opposed to only touching the very top.

What should the module compute? It takes a d-dimensional activation and must return a d-dimensional activation (to stay compatible with the residual/LN that follows). The cheapest dense map d→d is a d×d matrix — but that's d² ≈ 590k parameters per insertion at d=768, two insertions per layer, times 12 layers: on the order of 14M parameters per task. That's not "tiny," and it's most of what I'm trying to avoid. I need far fewer parameters while still mapping d→d.

The trick is a bottleneck. Don't map d→d directly; map d *down* to a small dimension m, apply a nonlinearity, then map back *up* m→d. Down-projection is a d×m matrix, up-projection an m×d matrix, plus biases. Parameter count per adapter: m·d (down) + m (down bias) + d·m (up) + d (up bias) = 2md + d + m. With m ≪ d this is small — e.g. m=64, d=768 gives ~98k+ per adapter, and m is a single dial I can turn down further to trade a little accuracy for a lot of compactness. The nonlinearity in the middle is what lets the module compute something more than a low-rank linear map; I'll use GeLU, matching BERT's own feed-forward nonlinearity. So the module is: down-project d→m, GeLU, up-project m→d. A bottleneck autoencoder-shaped module.

But a bare down-up bottleneck can't be the identity — if I want ψ ≈ φ at init, a module that *replaces* the activation with its own (initially garbage) output breaks the forward pass. Two fixes are needed and they reinforce each other. First, give the module an *internal* skip connection: output = x + (up ∘ GeLU ∘ down)(x). Now the module computes the input *plus* a learned correction. Second, initialize the projection weights to near-zero. With near-zero projections, the correction term up(GeLU(down(x))) ≈ 0, so the module's output ≈ x — an approximate identity. Exactly the near-identity initialization the trainability argument demanded, and the skip connection is what makes "near-zero weights" equal "near-identity function" rather than "near-zero output." As training proceeds, the projections move away from zero and the module starts genuinely reshaping activations. If I initialized too far from identity, the early forward pass would be corrupted and the model could fail to train — so the init scale matters: a zero-mean Gaussian with a small standard deviation, around 1e-2 (truncated to two standard deviations), with the method robust as long as the init stays below ~1e-2 and degrading when it's too large. The smallness is doing real work, not cosmetics.

So I have a module. Where exactly in the sub-layer path does it sit? The sub-layer produces Sublayer(x), projects it back to d, and then there's a residual add and a layer norm. I'll insert the adapter *after* the sub-layer's projection back to d (and after its dropout) but *before* the residual add and the layer norm — the adapter transforms the sub-layer's output, then that output is added to the layer's residual and passed into layer norm. Two adapters per Transformer layer then: one after the attention sub-layer's projection, one after the feed-forward sub-layer's projection. This places the adapter on every information-bearing path while keeping the backbone's residual/LN skeleton intact.

Now, which parameters do I actually train per task? The whole point is to freeze w. So freeze all of BERT's original weights. Train: the adapter modules (the new v), and the final classification layer (unavoidable — the label space and loss differ per task, so the top layer is always new). One more: the layer-normalization parameters. The forward distribution shifts once adapters start modifying activations, and the LN affine parameters (scale and shift, 2d per layer) let the network re-stabilize those shifted statistics cheaply — this echoes the modulation methods (conditional batch-norm, FiLM) that adapt a net purely through normalization affines. It's cheap (2d per layer) and helps the adapters do their job. But I should be clear about why I'm not relying on LN tuning *alone*: tuning only the normalization affines is known to be too weak to reach fine-tuning accuracy — it can rescale and shift activations but can't compute the richer task-specific transformation the bottleneck provides. So LN parameters are a cheap complement to the adapters, not a replacement.

Let me sanity-check the budget. Per task I train: two bottleneck adapters per layer (2md+d+m each), the LN affines (~2d per LN), and one classification head. With m on the order of a few dozen to a couple hundred, this lands around a few percent of BERT's parameters per task — roughly 0.5–8% depending on m — versus 100% for full fine-tuning. That's the one-to-two-orders-of-magnitude compression I was after, and m is the single knob trading accuracy for compactness.

Does this give the extensibility I wanted? Yes, by construction: w is frozen, so each task's adapters + LN + head are an independent island. Adding task 5 doesn't touch tasks 1–4 at all; I can train tasks sequentially with no shared optimization and no forgetting, because the only shared parameters are frozen. And I never need all datasets at once, unlike multi-task learning. Compact, extensible, and — because the bottleneck with a nonlinearity is genuinely expressive while the near-identity init makes it trainable — it should sit within a small margin of full fine-tuning rather than degrading like LN-only or feature-based transfer.

One intuition worth noting about *why* this is enough: a frozen pre-trained net already computes good general features in its lower layers; what a downstream task mostly needs is to re-mix higher-layer features into task-specific ones. A small per-sub-layer corrective module, applied throughout but free to do more work where it matters, can supply exactly that re-mixing without rewriting the whole network — which is why a few percent of parameters can recover near-full-fine-tuning accuracy.

Here is the code these decisions land on.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    """Bottleneck module with an internal skip connection, near-identity at init.
    d -> m (down), GeLU, m -> d (up), then add the input."""
    def __init__(self, d_model, m, init_std=1e-2):
        super().__init__()
        self.down = nn.Linear(d_model, m)     # d -> m
        self.up = nn.Linear(m, d_model)        # m -> d
        # near-zero init so up(GeLU(down(x))) ~ 0  =>  module ~ identity at start
        for lin in (self.down, self.up):
            nn.init.trunc_normal_(lin.weight, std=init_std, a=-2 * init_std, b=2 * init_std)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))   # internal skip => near-identity init


def sublayer_output(sublayer_fn, x, layer_norm, adapter, dropout_p=0.1):
    """Post-LN sub-layer path with the adapter inserted after the sub-layer's
    projection (and dropout), before the residual add and LayerNorm."""
    h = sublayer_fn(x)                  # attention or FFN, projected back to d
    h = F.dropout(h, p=dropout_p)
    h = adapter(h)                       # <-- per-task module, here
    return layer_norm(h + x)             # residual add, then layer norm


class AdaptedTransformerLayer(nn.Module):
    """One frozen Transformer layer with two adapters (after attention, after FFN)."""
    def __init__(self, layer, d_model, m):
        super().__init__()
        self.layer = layer               # frozen pre-trained sub-layers + LayerNorms
        self.attn_adapter = Adapter(d_model, m)
        self.ffn_adapter = Adapter(d_model, m)

    def forward(self, x):
        a = sublayer_output(self.layer.attention, x, self.layer.ln1, self.attn_adapter)
        h = sublayer_output(self.layer.feed_forward, a, self.layer.ln2, self.ffn_adapter)
        return h


def trainable_parameters(model):
    """Freeze pre-trained weights; train adapters, LayerNorm affines, and the head."""
    for p in model.parameters():
        p.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, Adapter) or isinstance(module, nn.LayerNorm):
            for p in module.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():     # new per-task head
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def train(model, loader, opt):
    params = trainable_parameters(model)
    optim = torch.optim.Adam(params, lr=opt.lr)   # linear warmup over first 10%, linear decay
    for batch in loader:
        logits = model(batch["input_ids"])        # uses [CLS] token for classification
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward(); optim.step(); optim.zero_grad()
```

The causal chain: serving many tasks cheaply forces a frozen shared backbone plus a tiny trainable per-task delta (ψ_{w,v}, |v|≪|w|); injecting trainable parameters into a deep frozen net forces near-identity initialization, which forces an internal skip connection plus near-zero projection init; "few parameters but a d→d map" forces a down-project/GeLU/up-project bottleneck whose width m is the accuracy/compactness dial; the post-LN sub-layer structure fixes the insertion point (after each sub-layer's projection, before residual+LN), twice per layer; and since adapters shift the activation statistics and the task's label space is new, I also train the LayerNorm affines and a fresh classification head — everything else frozen, so tasks never interfere and the model is compact, extensible, and within a small margin of full fine-tuning.
