Let me start from the thing that actually hurts. We have one enormous pre-trained language model, and we want to specialize it to many downstream tasks — summarization, NL-to-SQL, inference, whatever. The way everyone does this is full fine-tuning: load the pre-trained weights Φ₀, then run gradient ascent on the conditional log-likelihood, max over Φ of Σ over (x,y) in the data of Σ_t log P_Φ(y_t | x, y_<t), updating every single parameter. It works, quality is great. But stare at what comes out the other end. The adapted model is Φ₀ + ΔΦ, and ΔΦ has exactly as many entries as Φ₀. So for each task I store a *full copy* of the model. With something the size of GPT-3, |Φ₀| ≈ 175 billion, that's roughly a 350 GB checkpoint per task. Ten tasks, ten times that. Switching tasks at serving time means swapping a 350 GB blob in and out. And the training side is just as bad: Adam keeps a first and second moment for every trainable parameter, so the optimizer state alone roughly triples the memory beyond the weights, and I have to compute a gradient for all 175B numbers. The cost scales with the *whole* model even though all I want is to nudge it toward one task.

So the real question isn't "how do I fine-tune" — it's "how do I represent the *change* ΔΦ cheaply." I want a small set of task-specific parameters Θ with |Θ| ≪ |Φ₀|, such that the adapted behavior is Φ₀ + ΔΦ(Θ): I keep Φ₀ frozen and shared across all tasks, and per task I only ship Θ. Then the optimization becomes max over Θ of Σ log P_{Φ₀+ΔΦ(Θ)}(...). If I can make |Θ| a thousandth or a ten-thousandth of |Φ₀|, the storage problem evaporates, the optimizer-state problem mostly evaporates, and task switching becomes "swap a few megabytes."

People have been chasing parameter-efficient adaptation for years, so let me look at what's already on the table and see precisely where each approach falls short.

First family: adapter layers. Houlsby and colleagues insert a little bottleneck module *between* the existing Transformer sublayers. Each adapter is a down-projection from d_model to some small r, a nonlinearity, an up-projection back from r to d_model, plus biases, wrapped in a residual connection. The original design uses two of these per Transformer block; a later variant uses one per block plus a LayerNorm. Parameter count per adapter is about 2·d_model·r + r + d_model, which with a small bottleneck r is well under 1% of the model. So on the storage axis, adapters are great. And they match quality reasonably. So why not just use them? The trouble isn't a parameter-count problem at all — it's a *latency* problem. The adapter adds *depth*. It's a new layer that sits in the forward path and has to be computed in addition to everything else, sequentially, and there's no way to fold it back into the existing weights — it's got a nonlinearity in the middle, so W₀x and adapter(x) don't combine into a single linear map. Now, the naive instinct is "but it's tiny in FLOPs, so the latency is negligible," and that instinct is wrong in a way that matters. Large networks are fast because their arithmetic is spread across massively parallel hardware; the wall-clock time is dominated not by raw FLOPs but by how many sequential operations you launch and how well each one saturates the hardware. A thin extra layer is almost no FLOPs but it's a whole extra *sequential* step — a kernel launch you can't overlap with the rest. When there's lots of parallel work to hide behind — big batch, long sequence — you barely notice. But production inference is frequently the opposite: batch size one, short prompt, answer me now. There's nothing to hide the extra step behind. And concretely, on GPT-2 medium at batch size 1 and sequence length 128, sticking these bottleneck adapters in raises the single-forward-pass latency by twenty to thirty percent. That's not negligible, that's a deal-breaker for online serving. It gets worse if the model is sharded across GPUs, because the extra depth means extra synchronization — more AllReduce and Broadcast — unless I redundantly replicate the adapter everywhere. So: adapters solve storage but tax latency, by construction, because they add irreducible sequential depth.

Second family: prefix and prompt tuning. Completely different idea — don't touch the weights at all, instead optimize the *input activations*. Prefix-embedding tuning prepends some number of trainable "virtual token" embeddings, l_p of them (maybe some infixed ones l_i too), that aren't real vocabulary; you backprop into those embeddings. The trainable count is just d_model·(l_p+l_i), beautifully small. Prefix-layer tuning goes further and replaces the activations after *every* layer with trainable vectors, count L·d_model·(l_p+l_i). No added depth, so no obvious latency tax. But two things bite. One: the prefix literally occupies positions in the context window. Every virtual token I prepend is a token's worth of sequence length I can no longer spend on the actual task input. For a model with a fixed context budget that's a direct, permanent tax on usable input length. Two: it's just hard to optimize. Performance moves non-monotonically as I add prefix tokens — more trainable parameters can make it *worse* — and in the low-data regime it falls apart; on a tiny MNLI subset prefix-embedding tuning barely clears random chance. So prefix tuning dodges the latency problem but eats my sequence length and is finicky and brittle.

And bias-only tuning — just train the bias vectors — is even cheaper but plainly under-powered; biases alone can't carry the adaptation across tasks.

So now I can write down what a candidate has to aim for, because each ancestor taught me one constraint by violating it: tiny per-task footprint (from the storage pain), enough capacity to reach the full-fine-tuning quality bar, *no added inference latency* (the adapter lesson — no extra sequential depth, the adaptation must be foldable into the existing weights), and *don't steal sequence length* (the prefix lesson — leave the input alone). The intersection of these requirements is unoccupied for everything I've surveyed. That gap is the actual problem.

Let me step back and ask whether ΔΦ even *needs* to be a big object. There's a result I keep coming back to about the intrinsic dimensionality of fine-tuning. The setup: reparameterize the entire parameter vector as θ = θ₀ + P θ_d, where θ₀ is the frozen pre-trained vector living in the huge space R^D, P is a *fixed random* projection from a small space R^d up into R^D, and you only train the little vector θ_d ∈ R^d. Then ask: how small can d be and still recover, say, 90% of full fine-tuning's performance? Call that d₉₀. The answer is shockingly small — for RoBERTa on MRPC, on the order of a couple hundred trainable numbers gets you to 90% of full performance. And the bigger the pre-trained model, the *lower* its intrinsic dimension. So the solution that fine-tuning finds doesn't actually need the full D dimensions; it lives in a tiny subspace. That's the green light. If a few hundred effective degrees of freedom suffice, then encoding ΔΦ with a small Θ isn't a desperate compression — it's matching the true shape of the problem.

But I shouldn't copy that construction directly. A random projection P into the *flattened* parameter vector is a generic low-dimensional subspace; it needs a giant (if implicit, Fastfood-structured) projection operator, it's not naturally per-layer, and crucially it gives me nothing for the latency problem — there's no structure that lets me fold the update back into the weights. So let me hold onto the *content* of the intrinsic-dimension result — the update has few effective degrees of freedom — and throw away its *form*. What I want is a way to spend few degrees of freedom that is (i) defined per weight matrix so it's local and simple, (ii) directly trainable without a fixed random P, and (iii) shaped so I can merge it into the original weight at deploy time.

So fix one weight matrix W₀ ∈ R^{d×k} and ask: how do I parameterize its update ΔW with few numbers, locally, mergeably? "Few effective degrees of freedom in a matrix" is exactly low rank — a rank-r matrix has only r·(d+k) free parameters instead of d·k, it is intrinsic to that one matrix (constraint i), and a low-rank matrix simply *adds* to W₀ to give another d×k matrix (constraint iii looks promising — I'll check the merge carefully below). So the hypothesis I'll commit to: the change ΔW to any given weight matrix during adaptation has low *intrinsic rank*.

If that's true, I shouldn't parameterize ΔW as a free d×k matrix — I should constrain it to be low rank from the start. The cleanest way to force a matrix to have rank at most r is to write it as a product of a tall-skinny and a short-wide matrix. For W₀ ∈ R^{d×k}, write the update as ΔW = B A, with B ∈ R^{d×r} and A ∈ R^{r×k}, and r ≪ min(d,k). Does B A actually have rank ≤ r? rank(BA) ≤ min(rank B, rank A) ≤ r, since A has only r rows and B only r columns. Let me confirm I'm not fooling myself with a quick concrete case: take d=6, k=5, r=2, random B ∈ R^{6×2} and A ∈ R^{2×5}; numerically rank(BA) comes out to 2, the full r. So the product genuinely caps the rank at r and generically attains it — that's the constraint I wanted. Freeze W₀; train only A and B. The forward pass through this layer was h = W₀ x; now it becomes

  h = W₀ x + ΔW x = W₀ x + B A x.

Look at the parameter count: instead of d·k numbers in ΔW, I have d·r + r·k = r·(d+k) numbers in B and A. Let me put the real GPT-3 numbers in rather than wave at "much smaller." A square projection has d = k = 12288, and trying r = 4: the low-rank cost is 4·(12288+12288) = 4·24576 = 98,304 numbers, against 12288² = 150,994,944 for the dense update. Dividing, 150,994,944 / 98,304 = 1,536 exactly — a 1,536× reduction for that one matrix. If I adapt only selected matrices, Θ = {A, B over those matrices}, and |Θ| can become a tiny fraction of a percent of |Φ₀|.

Now let me check the structural constraints, because if this only satisfies storage I've just reinvented adapters with extra steps. Storage: tiny, just shown. Sequence length: I never touch the input — B A x is computed on the same x the layer already gets, so I lose zero context. Good, the prefix problem is gone. Latency — this is the one I care about most, so let me be careful and not just declare it. The adaptation path is B A x, added to W₀ x. Both act on the *same* input x and their outputs are summed coordinate-wise; this is a *parallel* branch, not extra depth stacked on top. There's no nonlinearity between W₀ and B A — both branches are linear in x. Linearity says I should be able to collapse them: ⟨compute both, add⟩ should equal ⟨add the matrices first, then apply⟩, i.e. W₀x + s·(BA)x = (W₀ + s·BA)x for any scalar gain s. That's just the distributive law, but the whole no-latency argument rests on it being an *exact* identity, not an approximation, so let me actually evaluate both sides. With d=6, k=5, r=2, s=0.37, random W₀, B, A and a random x, I compute W₀x + s(BA)x and (W₀ + s·BA)x and subtract: the largest entrywise difference is 8.9e-16 — machine epsilon, i.e. they are the same vector. So the merge is exact: I can fold the branch into the weight and the scalar gain just rides along inside the sum. At deployment I compute the adapted matrix W = W₀ + s·BA once, store it, and run inference as h = W x — one ordinary matmul, exactly the same operation and exactly the same shape as the un-adapted model or a fully fine-tuned model. So there is no added inference latency, and the reason is that exact identity, not an empirical accident. That's precisely the property adapters can never have because their nonlinearity sits *between* the two pieces, so there is no single matrix to fold into — the merge identity I just checked simply doesn't exist for them. And task switching is now trivial: to move from task to task, recover W₀ by subtracting the scaled update for the current task, then add the scaled update for the next task — a cheap operation with negligible memory overhead, far cheaper than swapping a whole 350 GB model. The remaining question is capacity: whether a small rank is enough to reach the quality bar. That I genuinely can't settle from the desk — it's the low-rank hypothesis itself, and the intrinsic-dimension evidence is why it's a reasonable bet rather than wishful thinking, but I'd only know by training and reading the validation numbers.

There's something pleasing here too: this isn't a side hack, it's a generalization of fine-tuning. A more general fine-tuning would let me train some subset of parameters. This goes further — it doesn't even require the accumulated update to be full rank. Does it actually contain full fine-tuning in the limit, though? The claim is that if I push r up to min(d,k), the maximum possible rank of a d×k update, then B A can represent an arbitrary ΔW. Let me make sure that's true and not just plausible: take an arbitrary target ΔW (random 5×7), set r = min(5,7) = 5, and factor it through the inner dimension via its SVD — B = UΣ (5×5), A = Vᵀ (5×7). Reconstructing B A and comparing to the target, the largest error is 1.8e-15, machine zero. So at full rank the product really can hit any matrix exactly; nothing is lost. Apply this to *all* the weight matrices at r = min(d,k) (and let the biases train, they're negligible in count) and I recover the full expressiveness of fine-tuning. So as I add capacity, this method converges back to training the original model — whereas an adapter, as you widen it, converges to some extra MLP bolted on, and prefix tuning, as you add tokens, converges to a model that simply can't take long inputs anymore. This degrades gracefully toward the right limit; the others degrade toward something foreign.

Now the details that make it actually train. How do I initialize A and B? I have a strong prior: W₀ is a carefully pre-trained operator that already does something good, and the intrinsic-dimension story says the solution is *near* it. I do not want to kick the model with a random ΔW at step zero — that would corrupt a known-good starting point and throw away the whole reason I'm starting from a pre-trained model. So I want ΔW = B A = 0 at the very start, so the adapted model is byte-for-byte the pre-trained model at step 0, and then it departs only as gradients dictate. How do I make B A = 0 without making A and B both dead? Let g = ∂L/∂h for the scalar training loss L, and let s be the scalar multiplying B A. Since h = W₀x + s B A x, the differentials are dL = gᵀ s dB A x + gᵀ s B dA x, so ∂L/∂B = s g (A x)ᵀ and ∂L/∂A = s Bᵀ g xᵀ. Read off the structure: ∂L/∂A carries a factor of B, and ∂L/∂B carries a factor of A (through A x). So if I zero *both* factors, both derivatives vanish and nothing ever moves — a dead fixed point. Let me make sure that's a real trap and not a phantom of the algebra: I build a tiny case (d=4, k=3, r=2), set A=B=0, run one backward pass, and indeed ‖∂L/∂A‖ and ‖∂L/∂B‖ both come out exactly 0. So zero-both is genuinely stuck; I can't use it.

The fix is to zero only one factor. Set B = 0 so that B A = 0 regardless of A, and give A an ordinary random initialization. Then the formulas predict ∂L/∂A = 0 at step 0 (it's proportional to B = 0), while ∂L/∂B = s g (A x)ᵀ should be nonzero because A x generally isn't zero. Let me trace it rather than trust it. Same tiny case, now A = Kaiming-uniform and B = 0: at step 0, ‖∂L/∂A‖ = 0 and ‖∂L/∂B‖ ≈ 7.1, and h equals W₀x exactly — so the model truly starts unchanged and only B is free to move first. Then I take one gradient step on B and look at step 1: now ‖∂L/∂A‖ ≈ 7.1 (no longer zero) and ‖∂L/∂B‖ ≈ 5.5. So the moment B leaves zero, A starts receiving gradient through the Bᵀ factor — the two factors hand off exactly as the formulas said, and the whole thing un-sticks itself after one step. The symmetric choice, random B and zero A, would avoid the dead fixed point too by the same argument. The implementation I want mirrors common linear-layer initialization: Kaiming-uniform A, zero B. So: one factor random, the other zero, product zero — the model starts unchanged, and the zeroed factor gets the first nonzero update.

One more knob. I noticed something while thinking about varying r: without any normalization, changing r changes the strength of the branch and the scale of the gradients because B A is built from r rank-one channels. Then rank is secretly also a step-size knob, and every rank sweep becomes a learning-rate sweep. I want r to mean "capacity" as much as possible, not "capacity plus hidden gain." So let me put a rank normalization in the branch and write the contribution as (α/r)·B A x, where α is a fixed branch gain. The 1/r factor is a convention that damps the branch as the number of rank channels grows; it is not a claim that every norm of B A is exactly invariant for every initialization, but it prevents the obvious linear growth in total channel contribution from becoming the default behavior. What about α itself? With Adam, changing a constant branch gain behaves much like changing the effective learning rate, up to initialization and optimizer details, so I do not want to tune α separately for every r. The practical move is to set α from the first rank I try and keep it fixed while r changes. So the final forward pass for an adapted matrix is

  h = W₀ x + (α/r)·B A x.

Where do I put these B A modules? In a Transformer the obvious candidates are the four attention projections W_q, W_k, W_v, W_o and the two MLP matrices. The clean first cut is to spend the tiny parameter budget on attention projections and leave the MLP frozen, because attention projections directly decide what information is read, written, and mixed at each token. If an implementation stores query, key, and value as three separate linear layers, I can replace only the chosen projections. If it stores them as one fused qkv projection, I need the same idea but with a mask over output slices, so the rank factors update, say, the q and v slices while leaving the k slice untouched. I can choose the exact target set by validation without changing the underlying parameterization.

Let me also sanity-check the training-cost win, since I claimed it. Because W₀ is frozen, I don't compute or store gradients for it, and — this is the big one with Adam — I don't keep optimizer moments for it. The optimizer state is now proportional to |Θ|, the size of all the A's and B's, not to |Φ₀|. I still need W₀ resident for the forward and backward computations, but I do not pay gradient or Adam-state memory on it. Storage of the result: I save only the A's and B's per task — megabytes — against the shared frozen W₀.

Let me put this into code. The implementation has a small mixin for the rank, scaling, dropout, and merge state; the ordinary linear layer freezes its base weight, owns A and B, handles transposed-weight conventions, and merges or unmerges the scaled product when train/eval mode changes. The fused qkv case needs the same operation with a mask over output slices, so the merged-weight helper builds only the enabled slices and zero-pads the rest before adding the update to the original matrix.

```python
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer():
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float,
                 merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else (lambda x: x)
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    def __init__(self, in_features: int, out_features: int, r: int = 0,
                 lora_alpha: int = 1, lora_dropout: float = 0.,
                 fan_in_fan_out: bool = False, merge_weights: bool = True,
                 **kwargs):
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
            # A is random, B is zero: BA = 0 at step 0, and B gets
            # the first nonzero low-rank gradient.
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
    def __init__(self, in_features: int, out_features: int, r: int = 0,
                 lora_alpha: int = 1, lora_dropout: float = 0.,
                 enable_lora: List[bool] = [False],
                 fan_in_fan_out: bool = False, merge_weights: bool = True,
                 **kwargs):
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

So the whole chain, end to end: full fine-tuning works but its delta is the size of the entire model, which is unaffordable to store and switch per task at large scale; adapters shrink the delta but add irreducible sequential depth that taxes latency exactly in the batch-size-one serving regime; prefix tuning avoids depth but spends sequence length and optimizes badly. The intrinsic-dimension result says the true update lives in a tiny subspace, so I hypothesize each weight matrix's update is low rank and write it as B A on top of a frozen W₀; that gives a 1,536× smaller update for a 12288×12288 matrix at r = 4, never touches the input sequence, and — because the branch is linear in the same input — folds W = W₀ + (α/r)B A into a single merged matrix at deploy time for zero added latency, with cheap task switching by replacing one scaled update with another. I initialize one factor random and the other zero so the model starts exactly as pre-trained while the zeroed factor can move on the first step, and I scale by α/r so rank changes are less entangled with branch gain. Freezing W₀ removes its gradient and optimizer state. The structural constraints are handled by the parameterization; the quality question is the low-rank-capacity hypothesis I would validate by training.
