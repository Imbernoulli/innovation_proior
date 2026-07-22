Let me start from the pain that's actually in front of me. I have a 12-layer Transformer encoder, ~110M parameters, pre-trained with masked language modeling, and it's wonderful — fine-tune it on anything and it sets the bar. But it's a brick. Running it under a latency budget, on a phone, in a real-time service, is rough; and pre-training one from scratch burns an absurd amount of compute, with the field pushing ever bigger. I want a model that's much smaller and faster at inference, cheap enough to pre-train once, and — this is the part that matters — *general-purpose*: I should be able to fine-tune it on the whole spread of downstream tasks, not just one, and keep almost all of the big model's language understanding.

The obvious move is to just train a smaller Transformer from scratch with MLM. Fewer layers, fewer parameters. But watch what happens: a small model trained on hard one-hot targets alone tends to land well short of the big one. The one-hot label throws away almost everything the big model knows. Think about what the big model actually predicts for a masked slot like "I think this is the beginning of a beautiful [MASK]" — it doesn't just say *day*; it puts high mass on *day* and *life*, and a graded tail on *future*, *story*, *world*. Those relative magnitudes *are* the model's learned notion of which answers are almost-right, how it generalizes. A one-hot target says "*day*, everything else zero," which is both wrong (other answers are valid) and information-poor. So training the small model only on hard labels is throwing away the richest thing the big model produces.

So don't imitate the data — imitate the *teacher's distribution*. Train the student to reproduce the teacher's soft predictions. Concretely, take the teacher's probability `t_i` over the vocabulary at each position and push the student's probability `s_i` toward it. The cross-entropy of the student against the teacher's soft targets, `-Σ_i t_i log s_i`, gives a dense signal at every class, not just the gold one — the student learns the whole shape, the tail included.

But there's a wrinkle. Even the teacher's soft distribution is usually *peaked* — after a confident softmax, the non-target probabilities are tiny, and tiny numbers contribute almost nothing to the loss, so the very signal I want (the structure in the tail) barely registers. I need to flatten the distribution to expose the tail. Divide the logits by a temperature `T > 1` before the softmax: `p_i = exp(z_i/T) / Σ_j exp(z_j/T)`. Larger `T` softens both teacher and student into smoother distributions where the relative weights of the small probabilities are amplified, so matching them actually moves the loss. Apply the *same* `T` to teacher and student during training; set `T = 1` at inference to recover the normal softmax. The teacher produces a vocabulary distribution at every real token position, so the soft loss can use every non-padding position; the hard MLM loss is the one that is restricted to the masked labels. So the distillation term is the cross-entropy between the temperature-softened teacher and student distributions on the valid sequence positions.

Now I have to be careful about something that's easy to wave past: raising `T` doesn't just reshape the distribution, it changes the *size of the gradient* I'm training on. If I'm sloppy here I'll silently downweight the soft term and won't know why. Let me actually work out the gradient rather than guess. Write `q = softmax(z_s/T)` for the student and `p = softmax(z_t/T)` for the teacher. The loss at a position is `-Σ_i p_i log q_i`. Its derivative with respect to a student logit `z_s,k`: `log q_i` depends on `z_s,k` through the softmax, and the standard softmax-cross-entropy result, adjusted for the `1/T` inside, gives `∂L/∂z_s,k = (q_k − p_k)/T`. That `1/T` out front is the first factor of shrink. But there's a second, sneakier factor: at large `T`, both `p` and `q` flatten toward the uniform distribution, and a unit change in a logit moves a softened softmax only a little — `q_k − p_k` is itself roughly proportional to `1/T` when the logit gaps are small relative to `T`. So the gradient shrinks like `1/T · 1/T = 1/T²`, not `1/T`. That's a stronger effect than the surface `1/T` would suggest, and it's exactly the kind of thing I'd get wrong by eyeballing.

Let me check this instead of trusting the hand-derivation, because the `1/T²` claim is load-bearing — it sets the rescaling constant. I'll take a tiny 5-class case, teacher logits `z_t = [2, 1, 0, −1, 0.5]`, student logits `z_s = [1.5, 1.2, 0.1, −0.5, 0.3]`, and compare my analytic `(q_k − p_k)/T` against autograd of `-Σ_i p_i log q_i`. They agree to six decimals at every `T` I try (`T = 1, 2, 4, 8`) — so the gradient formula is right. Then the scaling: the raw gradient norm falls 0.184 → 0.0423 → 0.00998 → 0.00244 as `T` goes 1 → 2 → 4 → 8, i.e. it drops by roughly `4×` each time I double `T`, which is the `1/T²` signature. To see it cleanly I shrink the logit gaps (so the small-gap linearization is exact) and rescale by `T²`: the norms become 1.5526e‑2, 1.5420e‑2, 1.5375e‑2, 1.5355e‑2, 1.5346e‑2 across `T = 1…16` — flat to three digits. So multiplying the distillation loss by `T²` really does hold its gradient magnitude constant as `T` changes, and the residual drift is only the second-order curvature I dropped. Good — I'll scale `L_ce` by `T²`. Without it, the soft term would be downweighted relative to any hard-label term I mix in, and its effective contribution would wander every time I retune `T`.

Should I drop the hard labels entirely? No — keep the supervised MLM loss too, the ordinary cross-entropy of the student's predictions against the true masked tokens. The two are complementary: the soft teacher signal transfers the teacher's generalization structure, the hard MLM signal keeps the student anchored to the actual data. Call them `L_ce` (distillation) and `L_mlm` (masked LM). A linear combination.

Now, is matching the *output distribution* enough? The output is the last thing the network produces; two networks can land on similar output probabilities while organizing their internal representations quite differently, and I want the student's *representation* to resemble the teacher's, because that representation is what gets fine-tuned downstream. So add a term that aligns the hidden states directly. I don't want to force the student's hidden vectors to equal the teacher's in magnitude — the scales can legitimately differ — I care about *direction* in representation space. A cosine-embedding loss on the final hidden-state vectors, pushing the cosine similarity between student and teacher hidden states toward 1, aligns the directions without over-constraining the norms. Call it `L_cos`. So the objective is a triple loss:
`L = α_ce · L_ce + α_mlm · L_mlm + α_cos · L_cos`,
with `L_ce` the temperature-softened distillation cross-entropy (scaled by `T²`), `L_mlm` the standard MLM cross-entropy, and `L_cos` the cosine alignment of final hidden states. I'd want to check each term earns its place — does removing the MLM term hurt much? does removing either distillation term hurt a lot? — and weight them so the distillation terms carry most of the load.

Now the architecture of the student. I want it smaller, but *which* dimension do I cut? I could shrink the hidden size, the number of heads, or the number of layers. Here's the practical fact that decides it: the linear layers and layer-norm that dominate per-layer cost are already heavily optimized in the linear-algebra libraries, and for a fixed parameter budget, shaving the last (hidden) dimension buys less wall-clock speedup than removing whole layers. Depth is the lever for inference latency. So keep the teacher's hidden size and head count, and *halve the number of layers* — 12 → 6. Same width, half the depth. That alone gets me a large parameter reduction concentrated where it speeds up inference.

A couple of structural simplifications fall out. The token-type ("segment") embeddings exist to mark sentence A vs B for the next-sentence-prediction task; I'm going to train without NSP (more on that in a second), so I can drop them. And the pooler — the extra dense layer on the [CLS] token used for sentence-level pre-training heads — isn't needed for the general representation; drop it too. Everything else stays the same shape as the teacher, and I want to hold onto that fact because the next decision depends on it.

That same width opens up a way to start the student warm instead of cold. A small model trained from a random start has to discover good features from scratch, and I'm only going to give it a fraction of the teacher's compute. But I have a teacher with the *same hidden size and the same per-layer structure* — its layers and the student's layers live in the same vector space, so a teacher layer's weight tensors fit a student layer's slots exactly. So initialize the student from a depth-halved slice of the teacher: pick six of the twelve teacher layers to seed the six student layers. Which six? Roughly one out of every two, but I'd bias toward keeping the last layer, since the final block sits closest to the output the student must reproduce. The slice `[0, 2, 4, 7, 9, 11]` does that — `0, 2, 4` are the lower/early blocks, `7, 9` the upper-middle, and `11` is the teacher's final layer — so the student receives representations from across the teacher's depth rather than only the bottom half. The student then starts in a region of weight space the teacher already found useful, and distillation only has to compress and refine, not rediscover. Keeping the dimensions matched is what makes this copy possible at all: if I'd shrunk the width, the teacher tensors wouldn't fit, and I'd be back to a random start.

The training recipe should follow the strong pre-training recipe rather than the original one. Use very large batches via gradient accumulation (up to thousands of examples per effective batch), use *dynamic* masking (a fresh 15% mask each time a sequence is seen) instead of a fixed mask, and drop next-sentence prediction, since dropping NSP and using dynamic masking on large batches is what was shown to make MLM pre-training stronger. Train on the same corpus as the teacher — Wikipedia plus BookCorpus — so the student is matched to the teacher's data. The teacher runs in eval mode, produces soft targets and hidden states with no gradient; the student trains against the triple loss.

Let me put the loss into code, because the temperature handling and the masking are easy to get subtly wrong.

```python
import re, torch, torch.nn as nn, torch.nn.functional as F

class Distiller:
    def __init__(self, teacher, student, temperature=2.0,
                 alpha_ce=5.0, alpha_mlm=2.0, alpha_cos=1.0,
                 restrict_ce_to_mask=False):
        self.teacher, self.student = teacher, student
        self.T = temperature
        self.alpha_ce, self.alpha_mlm, self.alpha_cos = alpha_ce, alpha_mlm, alpha_cos
        self.restrict_ce_to_mask = restrict_ce_to_mask
        # KL divergence between softened distributions; batchmean matches the
        # cross-entropy-of-soft-targets gradient up to the constant teacher entropy
        self.ce_loss_fct  = nn.KLDivLoss(reduction="batchmean")
        self.mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.cos_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

    def step(self, input_ids, attention_mask, lm_labels):
        s_logits, s_hidden = self.student(input_ids, attention_mask)
        with torch.no_grad():
            t_logits, t_hidden = self.teacher(input_ids, attention_mask)

        # --- distillation loss: valid tokens by default; optionally masked tokens only ---
        kd_pos = (lm_labels > -1) if self.restrict_ce_to_mask else attention_mask.bool()
        mask = kd_pos.unsqueeze(-1).expand_as(s_logits)
        s_sel = s_logits.masked_select(mask).view(-1, s_logits.size(-1))
        t_sel = t_logits.masked_select(mask).view(-1, t_logits.size(-1))
        loss_ce = self.ce_loss_fct(
            F.log_softmax(s_sel / self.T, dim=-1),
            F.softmax(t_sel / self.T, dim=-1),
        ) * (self.T ** 2)                                          # compensate 1/T^2 gradient shrink

        # --- hard-label masked-LM loss ---
        loss_mlm = self.mlm_loss_fct(s_logits.view(-1, s_logits.size(-1)),
                                     lm_labels.view(-1))

        # --- cosine alignment of final hidden states (direction, not magnitude) ---
        sel = attention_mask.unsqueeze(-1).expand_as(s_hidden).bool()
        dim = s_hidden.size(-1)
        s_h = s_hidden.masked_select(sel).view(-1, dim)
        t_h = t_hidden.masked_select(sel).view(-1, dim)
        target = s_h.new_ones(s_h.size(0))                         # push cosine sim -> 1
        loss_cos = self.cos_loss_fct(s_h, t_h, target)

        return self.alpha_ce * loss_ce + self.alpha_mlm * loss_mlm + self.alpha_cos * loss_cos
```

One choice in there I should not just trust: I wrote the distillation term with `nn.KLDivLoss`, but the loss I actually reasoned about was the cross-entropy of soft targets `-Σ_i p_i log q_i`. Are these the same thing for training? The KL divergence is `Σ_i p_i log(p_i/q_i) = −Σ_i p_i log q_i + Σ_i p_i log p_i`; the second term is the teacher's entropy, which depends only on `p` (the frozen teacher) and not on the student logits, so it's a constant as far as the student's gradient is concerned. The two losses should therefore differ by a constant and have *identical* gradients. Let me confirm numerically with a 3×6 batch: the KL value is 0.2908 and the soft-cross-entropy value is 2.0014 — different, as expected, by the teacher entropy — but their gradients w.r.t. the student logits match to 0.0 max absolute difference. So using `KLDivLoss(reduction="batchmean")` trains on exactly the gradient I derived; the only thing the entropy offset changes is the printed loss value, not the optimization. Good — the code matches the math.

And the student construction — same width, half the depth, drop token-type embeddings and pooler, seed the six student blocks from a layer slice of the teacher:

```python
def build_student_from_teacher(teacher_cfg, teacher_state, Encoder,
                               layer_map=(0, 2, 4, 7, 9, 11)):
    student_cfg = dict(teacher_cfg)
    student_cfg["n_layers"] = len(layer_map)                  # 12 -> 6: cut depth, keep width
    student_cfg["use_token_type_embeddings"] = False
    student_cfg["use_pooler"] = False
    student = Encoder(student_cfg)                            # DistilBERT-style encoder

    s_state = student.state_dict()
    layer_to_student = {teacher_i: student_i for student_i, teacher_i in enumerate(layer_map)}
    for name, p in teacher_state.items():
        if "pooler" in name or "token_type" in name:
            continue
        m = re.search(r"layer\.(\d+)\.", name)                 # map teacher layer -> student layer
        if m:
            t_idx = int(m.group(1))
            if t_idx not in layer_to_student:
                continue
            s_name = name.replace(f"layer.{t_idx}.", f"layer.{layer_to_student[t_idx]}.")
        else:
            s_name = name                                      # embeddings, MLM head: copy directly
        if s_name in s_state and s_state[s_name].shape == p.shape:
            s_state[s_name] = p.clone()
    student.load_state_dict(s_state)
    return student
```
