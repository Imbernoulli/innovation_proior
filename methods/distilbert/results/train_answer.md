Large masked-language-model Transformer encoders have become the default source of transferable text representations, but their scale is a liability. A 12-layer, ~110M-parameter BERT-style model is expensive to pre-train from scratch and too slow or memory-hungry to deploy under tight latency budgets, such as on a phone or in a real-time service. The obvious remedy is to train a smaller Transformer, yet a small model trained only on hard one-hot labels falls well short of the teacher. A one-hot target says the masked token is exactly "day" and nothing else, discarding the teacher's rich output distribution: for a sentence like "I think this is the beginning of a beautiful [MASK]", a strong teacher places high mass on day and life and meaningful tail mass on future, story, and world. Those relative magnitudes encode how the model generalizes, so throwing them away removes the very signal that would let a compact student learn the teacher's function.

Distillation can transfer that signal, but the standard recipes are not enough for this setting. Task-specific distillation only produces a model for one downstream task, not a general-purpose pre-trained representation. Training a small model from scratch purely on soft targets lacks a strong initialization and the grounding of the original masked-LM objective. Pruning and quantization shrink an already-trained model rather than producing a compact pre-trained checkpoint. What is needed is a small encoder that is pre-trained once, cheaply, and remains fine-tunable across the whole range of downstream tasks.

The method is DistilBERT. It compresses the teacher during pre-training by knowledge distillation, training a smaller student Transformer to reproduce a frozen large teacher's behavior. The student keeps the same hidden size and attention-head count as the teacher but halves the depth, going from 12 layers to 6. Depth is the right dimension to cut because linear layers and layer normalization are heavily optimized in modern libraries, so for a fixed parameter budget removing layers reduces wall-clock inference time more than shrinking the hidden dimension. Because the width is unchanged, the student layers live in the same vector space as the teacher layers, which makes it possible to initialize the student from a depth-halved slice of the teacher. The extraction maps teacher layers 0, 2, 4, 7, 9, and 11 to the six student layers, seeding lower, middle, and final blocks so the student starts near a useful region of weight space. Token-type embeddings and the pooler are removed because the model is trained without the next-sentence-prediction auxiliary task.

Training uses a triple loss. The first term is a distillation cross-entropy that pushes the student's output distribution toward the teacher's. Both distributions are softened by dividing logits by a temperature T greater than 1 before the softmax, which flattens the distribution and exposes the informative tail. Because high temperatures shrink soft-target gradients by a factor of 1 over T squared, the loss is scaled by T squared so its contribution is stable. The second term is the standard masked-LM cross-entropy against the true tokens, which keeps the student anchored to the actual data. The third term is a cosine-embedding loss on the final hidden states, pushing the cosine similarity between student and teacher vectors toward one so their representations align in direction without forcing equal magnitudes. The combined objective is L equals alpha_ce times L_ce plus alpha_mlm times L_mlm plus alpha_cos times L_cos. Following the improved pre-training recipe, training uses large batches via gradient accumulation, dynamic masking with 15 percent of tokens corrupted, no next-sentence prediction, and the same Wikipedia plus BookCorpus corpus as the teacher. The default hyperparameters are temperature 2, alpha_ce 5.0, alpha_mlm 2.0, and alpha_cos 1.0.

```python
import re, torch, torch.nn as nn, torch.nn.functional as F

class Distiller:
    def __init__(self, teacher, student, temperature=2.0,
                 alpha_ce=5.0, alpha_mlm=2.0, alpha_cos=1.0):
        self.teacher, self.student = teacher, student
        self.T = temperature
        self.alpha_ce, self.alpha_mlm, self.alpha_cos = alpha_ce, alpha_mlm, alpha_cos
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.cos_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

    def loss(self, input_ids, attention_mask, lm_labels):
        s_logits, s_hidden = self.student(input_ids, attention_mask)
        with torch.no_grad():
            t_logits, t_hidden = self.teacher(input_ids, attention_mask)

        # Distillation loss over valid tokens, softened and T^2-rescaled.
        mask = attention_mask.bool().unsqueeze(-1).expand_as(s_logits)
        s_sel = s_logits.masked_select(mask).view(-1, s_logits.size(-1))
        t_sel = t_logits.masked_select(mask).view(-1, t_logits.size(-1))
        loss_ce = self.ce_loss_fct(
            F.log_softmax(s_sel / self.T, dim=-1),
            F.softmax(t_sel / self.T, dim=-1)
        ) * (self.T ** 2)

        # Hard masked-LM cross-entropy.
        loss_mlm = self.mlm_loss_fct(
            s_logits.view(-1, s_logits.size(-1)),
            lm_labels.view(-1)
        )

        # Cosine alignment of final hidden states.
        sel = attention_mask.unsqueeze(-1).expand_as(s_hidden).bool()
        d = s_hidden.size(-1)
        s_h = s_hidden.masked_select(sel).view(-1, d)
        t_h = t_hidden.masked_select(sel).view(-1, d)
        loss_cos = self.cos_loss_fct(s_h, t_h, s_h.new_ones(s_h.size(0)))

        return (self.alpha_ce * loss_ce +
                self.alpha_mlm * loss_mlm +
                self.alpha_cos * loss_cos)


def build_student_from_teacher(teacher_cfg, teacher_state, Encoder,
                               layer_map=(0, 2, 4, 7, 9, 11)):
    cfg = dict(teacher_cfg)
    cfg["n_layers"] = len(layer_map)
    cfg["use_token_type_embeddings"] = False
    cfg["use_pooler"] = False
    student = Encoder(cfg)
    s_state = student.state_dict()
    layer_to_student = {t: i for i, t in enumerate(layer_map)}
    for name, p in teacher_state.items():
        if "pooler" in name or "token_type" in name:
            continue
        m = re.search(r"layer\.(\d+)\.", name)
        if m:
            t = int(m.group(1))
            if t not in layer_to_student:
                continue
            s_name = name.replace(f"layer.{t}.", f"layer.{layer_to_student[t]}.")
        else:
            s_name = name
        if s_name in s_state and s_state[s_name].shape == p.shape:
            s_state[s_name] = p.clone()
    student.load_state_dict(s_state)
    return student


def dynamic_mask(input_ids, mask_prob=0.15):
    # Returns corrupted input_ids and labels with unmasked positions set to -100.
    labels = input_ids.clone()
    mask = torch.rand(input_ids.shape, device=input_ids.device) < mask_prob
    mask &= input_ids != 0
    labels[~mask] = -100
    masked = input_ids.clone()
    masked[mask] = 103  # [MASK] token id in BERT vocabulary
    return masked, labels


def train(distiller, corpus, steps, grad_accum):
    distiller.teacher.eval()
    opt = torch.optim.AdamW(distiller.student.parameters(), lr=5e-4, eps=1e-6)
    for step, batch in enumerate(corpus):
        if step >= steps:
            break
        ids, labels = dynamic_mask(batch.input_ids, mask_prob=0.15)
        loss = distiller.loss(ids, batch.attention_mask, labels)
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(distiller.student.parameters(), 5.0)
            opt.step()
            opt.zero_grad()
```
