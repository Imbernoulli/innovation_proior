We have one enormous pre-trained Transformer language model and we want to specialize it to many downstream tasks. Everyone does this by full fine-tuning: load the pre-trained weights $\Phi_0$ and run gradient ascent on the conditional log-likelihood $\max_\Phi \sum_{(x,y)} \sum_t \log P_\Phi(y_t \mid x, y_{<t})$, updating every parameter. The quality is excellent, but the object that comes out is the problem. The adapted model is $\Phi_0 + \Delta\Phi$, and $\Delta\Phi$ has exactly as many entries as $\Phi_0$, so each task is a full copy of the model. At the scale of GPT-3, $|\Phi_0| \approx 175$ billion, that is roughly a 350 GB checkpoint per task; ten tasks is ten of those, and switching tasks at serving time means swapping a 350 GB blob in and out. Training is no better: Adam keeps a first and second moment for every trainable parameter, so the optimizer state alone roughly triples the memory beyond the weights, and a gradient must be produced for all 175B numbers. The cost scales with the whole model even though all we want is to nudge it toward one task. So the real question is not how to fine-tune but how to represent the *change* $\Delta\Phi$ cheaply — to find a small task-specific $\Theta$ with $|\Theta| \ll |\Phi_0|$ such that the adapted behavior is $\Phi_0 + \Delta\Phi(\Theta)$, with $\Phi_0$ frozen and shared across all tasks and only $\Theta$ shipped per task.

Each existing efficiency method teaches one constraint by violating it. Adapter layers insert a small bottleneck module *between* the existing Transformer sublayers — a down-projection $d_{\text{model}} \to r$, a nonlinearity, an up-projection $r \to d_{\text{model}}$, plus biases and a residual — at about $2 \cdot d_{\text{model}} \cdot r + r + d_{\text{model}}$ parameters, well under 1% of the model. On storage they are fine, but they add irreducible *depth*: a new layer in the forward path that must be computed sequentially and, because of its internal nonlinearity, cannot be folded back into the existing weights. The tempting dismissal — "it is tiny in FLOPs, so latency is negligible" — is wrong in exactly the regime that matters. Large networks are fast because their arithmetic is spread across massively parallel hardware, so wall-clock time is dominated not by FLOPs but by how many sequential kernel launches you incur. A thin extra layer is almost no FLOPs but a whole extra sequential step you cannot overlap. When there is lots of parallel work to hide behind — big batch, long sequence — you barely notice; but production inference is often batch size one with a short prompt, and there is nothing to hide behind. Measured on GPT-2 medium at batch size 1 and sequence length 128, adapters raise single-pass latency by 20–30%, and it is worse still under model sharding, which adds AllReduce/Broadcast synchronization. Prefix and prompt tuning go the other way and optimize the input activations: prefix-embedding tuning prepends $l_p$ trainable "virtual token" embeddings (count $d_{\text{model}} \cdot (l_p + l_i)$), prefix-layer tuning also replaces the post-layer activations everywhere. No added depth, but the prefix literally occupies positions in the context window, so it permanently taxes the usable input length; and it optimizes badly, moving non-monotonically in the number of prefix tokens and collapsing in the low-data regime. Bias-only tuning is cheaper still but plainly under-powered. The intersection of "tiny footprint, full-fine-tuning quality, no added latency, no stolen sequence length" is unoccupied; that gap is the actual problem.

What licenses a cheap $\Theta$ is the intrinsic-dimension result. Reparameterize the whole parameter vector as $\theta = \theta_0 + P\theta_d$, with $\theta_0$ the frozen pre-trained vector in $\mathbb{R}^D$, $P$ a fixed random projection from a small space $\mathbb{R}^d$ up into $\mathbb{R}^D$, and only the little $\theta_d \in \mathbb{R}^d$ trained; then ask how small $d$ can be and still recover 90% of full-fine-tuning performance, call it $d_{90}$. The answer is shockingly small — for RoBERTa on MRPC, on the order of a couple hundred trainable numbers — and the larger the pre-trained model, the *lower* its intrinsic dimension. The fine-tuning solution does not need the full $D$ dimensions; it lives in a tiny subspace. But that random projection into the *flattened* parameter vector is the wrong construction to copy: it needs a giant implicit projection operator, it is not per-layer, and it gives nothing for latency, since there is no structure that folds the update back into the weights. I want a low-dimensional subspace that is (i) defined per weight matrix so it is local, (ii) directly trainable with no fixed random $P$, and (iii) shaped to merge into the original weight at deploy time. Low intrinsic dimension of a matrix's update, expressed per matrix, is just low *rank* of that update.

I propose LoRA — Low-Rank Adaptation. The hypothesis is that the change $\Delta W$ to any given weight matrix during adaptation has low intrinsic rank, so I should constrain it to be low rank from the start rather than parameterizing a free $d \times k$ matrix. The cleanest way to force rank at most $r$ is to write the update as a product of a tall-skinny and a short-wide matrix: for $W_0 \in \mathbb{R}^{d \times k}$, set $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d,k)$; $BA$ automatically has rank at most $r$. Freeze $W_0$, train only $A$ and $B$, so the forward pass becomes
$$h = W_0 x + \Delta W\, x = W_0 x + \frac{\alpha}{r}\, B A\, x.$$
Instead of $d \cdot k$ numbers per matrix I store $r \cdot (d + k)$. For a square $12288 \times 12288$ projection at $r = 4$ that is $98{,}304$ trainable parameters against $150{,}994{,}944$, a $1{,}536\times$ reduction for that matrix; spend the budget on a few matrices and $|\Theta|$ is a fraction of a percent of $|\Phi_0|$.

Now check the structural constraints, because if this only solves storage I have reinvented adapters. Sequence length is untouched — $BAx$ acts on the same $x$ the layer already receives, costing zero context. Latency is the property I care about most, and here is what makes LoRA different by construction rather than by accident: the adaptation $BAx$ and the base $W_0 x$ act on the *same* input and their outputs are summed coordinate-wise, so this is a *parallel* linear branch, not extra stacked depth, and there is *no nonlinearity* between $W_0$ and $BA$. Both are linear in $x$, so they collapse — $(W_0 + \frac{\alpha}{r}BA)$ is itself one $d \times k$ matrix. At deployment I compute the adapted $W = W_0 + \frac{\alpha}{r}BA$ once and run inference as $h = Wx$, one ordinary matmul of the original shape with zero added latency. That is exactly the merge an adapter can never do, because its internal nonlinearity blocks it. Task switching becomes trivial: subtract the scaled update for the current task and add the scaled update for the next, a few-megabyte operation instead of swapping a 350 GB model. There is also a graceful limit — push $r$ up to $\min(d,k)$ on all matrices and $BA$ can represent an arbitrary $\Delta W$, recovering full fine-tuning, whereas a widened adapter converges to some foreign bolt-on MLP and a longer prefix converges to a model that can no longer take long inputs.

Two design choices make it train. The first is the zero-start initialization. $W_0$ is a known-good pre-trained operator and the intrinsic-dimension story says the solution sits near it, so I do not want to kick it with a random $\Delta W$ at step zero. I want $BA = 0$ at the start, so the adapted model is byte-for-byte the pre-trained model and departs only as gradients dictate. But I cannot simply zero both factors: with $h = W_0 x + s\,BAx$ and $g = \partial L / \partial h$, the differentials give $\partial L/\partial B = s\,g\,(Ax)^{\!\top}$ and $\partial L/\partial A = s\,B^{\!\top} g\, x^{\!\top}$, so if $A = B = 0$ both gradients vanish and nothing ever moves — a dead fixed point. The resolution is to zero exactly one factor. Set $B = 0$ so $BA = 0$ regardless of $A$, and give $A$ an ordinary random (Kaiming-uniform) initialization. On the first backward pass $\partial L/\partial A$ is zero because $B$ is zero, but $\partial L/\partial B = s\,g\,(Ax)^{\!\top}$ is generally nonzero because $Ax$ is not identically zero, so $B$ moves first and once $B$ is nonzero $A$ starts receiving gradients too. The product starts at zero, the model starts unchanged, and the zeroed factor gets the first real update. The second choice is the $\alpha/r$ scaling. Without normalization, changing $r$ changes the strength of the branch and the scale of the gradients, because $BA$ is built from $r$ rank-one channels, so rank secretly doubles as a step-size knob and every rank sweep becomes a learning-rate sweep. Writing the contribution as $\frac{\alpha}{r}BAx$ damps the branch as the number of channels grows, so $r$ means "capacity" rather than "capacity plus hidden gain." This is a convention, not a claim that every norm is exactly invariant; it prevents the obvious linear growth in total channel contribution from becoming the default. With Adam, varying the constant gain $\alpha$ behaves much like varying the effective learning rate up to initialization and optimizer details, so I fix $\alpha$ from the first $r$ I try and keep it fixed while sweeping $r$.

Where to put the modules: in a Transformer the candidates are the four attention projections $W_q, W_k, W_v, W_o$ and the two MLP matrices, and the clean first cut spends the tiny budget on attention projections — they directly decide what is read, written, and mixed at each token — leaving the MLP frozen. If query/key/value are three separate linear layers I replace the chosen ones; if they are one fused $qkv$ projection I need the same idea with a mask over output slices, so the rank factors update, say, the $q$ and $v$ slices while leaving $k$ untouched, choosing the target set by validation without changing the parameterization. Freezing $W_0$ also removes its gradient and, crucially with Adam, its optimizer moments, so the optimizer state is proportional to $|\Theta|$ rather than $|\Phi_0|$; $W_0$ stays resident for the forward and backward passes but costs no gradient or Adam-state memory, and per task I save only the $A$'s and $B$'s.

In code, a small mixin holds the rank, scaling, dropout, and merge state; the ordinary linear layer freezes its base weight, owns $A$ and $B$, handles transposed-weight conventions, and merges or unmerges the scaled product when train/eval mode flips; the fused $qkv$ case does the same with a mask over output slices, building only the enabled slices and zero-padding the rest before adding the update to the original matrix.

```python
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else (lambda x: x)
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged and self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged and self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
            return result
        return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0 and any(self.enable_lora):
            result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
        return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
```

Usage:

```python
# Separate projections:
q_proj = Linear(d_model, d_model, r=8, lora_alpha=16)
v_proj = Linear(d_model, d_model, r=8, lora_alpha=16)

# Fused qkv projection with only q and v adapted:
qkv_proj = MergedLinear(d_model, 3 * d_model, r=8, lora_alpha=16,
                        enable_lora=[True, False, True])

mark_only_lora_as_trainable(model)
# ... train with AdamW ...
torch.save(lora_state_dict(model), 'task.pt')
model.eval()
```
