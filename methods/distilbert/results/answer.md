# DistilBERT

## Problem

Large MLM-pretrained Transformer encoders (~110M parameters) are too costly to
pre-train and too slow/heavy to deploy under latency or memory constraints. We want a
*general-purpose* pre-trained language model that is much smaller and faster, cheap to
pre-train once, yet retains almost all of the large model's language understanding and
can be fine-tuned across the full range of downstream tasks.

## Key idea

Compress the model *during pre-training* with knowledge distillation: train a smaller
student Transformer to reproduce a frozen large teacher's behavior. Cut depth (not
width), initialize the student from the teacher, and train with a triple loss that
transfers the teacher's soft output distribution and hidden-state directions in
addition to the masked-LM signal.

- **Architecture:** same hidden size / head count as the teacher, half the layers
  (12 → 6). Remove token-type (segment) embeddings and the pooler.
- **Initialization:** copy every other teacher layer into the student (one layer out of
  two), exploiting the shared dimensionality so the student starts where the teacher
  already converged.
- **Triple loss:** `L = α_ce·L_ce + α_mlm·L_mlm + α_cos·L_cos`.
  - `L_ce`: distillation loss — KL/cross-entropy between teacher and student
    distributions, each softened by temperature `T` (`p_i = softmax(z_i/T)`), computed
    over masked positions and scaled by `T²` to compensate the `1/T²` gradient shrink.
    `T=1` at inference.
  - `L_mlm`: standard masked-LM cross-entropy against the true tokens.
  - `L_cos`: cosine-embedding loss pushing the cosine similarity of student and teacher
    final hidden states toward 1 (aligns direction, not magnitude).
- **Recipe:** large batches via gradient accumulation, dynamic masking (15%), no
  next-sentence prediction; same corpus as the teacher (Wikipedia + BookCorpus); AdamW
  with linear warmup; teacher frozen in eval mode.

The published training recipe uses `T=2`, `α_ce=5.0`, `α_mlm=2.0`, `α_cos=1.0`.

## Code

```python
import re, torch, torch.nn as nn, torch.nn.functional as F

class Distiller:
    def __init__(self, teacher, student, temperature=2.0,
                 alpha_ce=5.0, alpha_mlm=2.0, alpha_cos=1.0):
        self.teacher, self.student = teacher, student
        self.T = temperature
        self.alpha_ce, self.alpha_mlm, self.alpha_cos = alpha_ce, alpha_mlm, alpha_cos
        self.ce_loss_fct  = nn.KLDivLoss(reduction="batchmean")
        self.mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.cos_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

    def loss(self, input_ids, attention_mask, lm_labels):
        s_logits, s_hidden = self.student(input_ids, attention_mask)
        with torch.no_grad():
            t_logits, t_hidden = self.teacher(input_ids, attention_mask)

        # distillation loss over masked positions, temperature-softened, T^2-rescaled
        mask  = (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)
        s_sel = s_logits.masked_select(mask).view(-1, s_logits.size(-1))
        t_sel = t_logits.masked_select(mask).view(-1, t_logits.size(-1))
        loss_ce = self.ce_loss_fct(F.log_softmax(s_sel / self.T, dim=-1),
                                   F.softmax(t_sel / self.T, dim=-1)) * (self.T ** 2)

        # hard-label masked-LM cross-entropy
        loss_mlm = self.mlm_loss_fct(s_logits.view(-1, s_logits.size(-1)),
                                     lm_labels.view(-1))

        # cosine alignment of final hidden states
        sel = attention_mask.unsqueeze(-1).expand_as(s_hidden).bool()
        d   = s_hidden.size(-1)
        s_h = s_hidden.masked_select(sel).view(-1, d)
        t_h = t_hidden.masked_select(sel).view(-1, d)
        loss_cos = self.cos_loss_fct(s_h, t_h, s_h.new_ones(s_h.size(0)))

        return self.alpha_ce * loss_ce + self.alpha_mlm * loss_mlm + self.alpha_cos * loss_cos


def build_student_from_teacher(teacher_cfg, teacher_state, Encoder):
    cfg = dict(teacher_cfg); cfg["n_layers"] = teacher_cfg["n_layers"] // 2
    student = Encoder(cfg)                       # no token-type embeddings, no pooler
    s_state = student.state_dict()
    for name, p in teacher_state.items():
        if "pooler" in name or "token_type" in name:
            continue
        m = re.match(r"layer\.(\d+)\.", name)
        if m:
            t = int(m.group(1))
            if t % 2 != 0:                       # take one layer out of two
                continue
            s_name = name.replace(f"layer.{t}.", f"layer.{t // 2}.")
        else:
            s_name = name
        if s_name in s_state and s_state[s_name].shape == p.shape:
            s_state[s_name] = p.clone()
    student.load_state_dict(s_state)
    return student


def train(distiller, corpus, steps, grad_accum=...):
    distiller.teacher.eval()
    opt = torch.optim.AdamW(distiller.student.parameters(), lr=5e-4, eps=1e-6,
                            weight_decay=0.0)
    for step, batch in enumerate(corpus):
        ids, labels = dynamic_mask(batch.input_ids, mask_prob=0.15)   # fresh mask, no NSP
        loss = distiller.loss(ids, batch.attention_mask, labels)
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(distiller.student.parameters(), 5.0)
            opt.step(); opt.zero_grad()
```

The student is then fine-tuned like any pre-trained encoder; with half the layers it is
about 40% smaller and roughly 60% faster at inference while retaining the bulk of the
teacher's downstream quality.
