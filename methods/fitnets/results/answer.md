# FitNets: Hint-Based Training of Thin, Deep Students

## Problem

Train a student network that is **thinner and deeper** than a wide teacher — fewer parameters/multiplies per layer (cheap), more layers (expressive) — to a good optimum. Deep thin nets are hard to optimize, and knowledge distillation that matches only the teacher's *output* injects supervision at the top of the network, too weak to steer a much deeper student's interior.

## Key idea

Use the teacher's *intermediate* representation as a **hint** to guide a *middle* layer of the student (the **guided** layer), then train in two stages (a curriculum: easy intermediate objective first, full objective second).

**Hint loss (stage 1).** Because the teacher is wider, its hint layer has more channels than the student's guided layer; add a regressor r to match shapes, and minimize

```
L_HT(W_Guided, W_r) = ½ ‖ u_h(x; W_Hint) − r( v_g(x; W_Guided); W_r ) ‖²,
```

where u_h / v_g are the teacher/student nested functions up to the hint/guided layers. The regressor is **convolutional** (not fully-connected) when the layers are conv: kernel k_i chosen so N_{g,i} − k_i + 1 = N_{h,i}, giving only k_1·k_2·O_h·O_g parameters instead of the dense N_{h,1}·N_{h,2}·O_h·N_{g,1}·N_{g,2}·O_g. Hint and guided layers are placed at the middle of teacher and student (deeper placement over-regularizes).

**Distillation loss (stage 2).** With temperature τ > 1, P^τ = softmax(a/τ):

```
L_KD(W_S) = H(y_true, P_S) + λ · H(P_T^τ, P_S^τ),
```

H cross-entropy, λ balancing the true-label term and the soft-target term.

## Algorithm (stage-wise)

```
Input: trained teacher W_T, random student W_S, indices g (guided), h (hint)
1. W_Hint   ← teacher params up to layer h
2. W_Guided ← student params up to layer g
3. initialize regressor W_r to small random values
4. W_Guided* ← argmin L_HT(W_Guided, W_r)        # stage 1: pretrain student's lower half
5. copy W_Guided* into the student's first g layers (discard regressor)
6. W_S* ← argmin L_KD(W_S)                        # stage 2: distill the whole student
```

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ConvRegressor(nn.Module):                       # k sized so N_g - k + 1 = N_h
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k)
    def forward(self, v):
        return self.conv(v)

def hint_loss(hint, guided, regressor):
    return 0.5 * F.mse_loss(regressor(guided), hint, reduction='sum')

def kd_loss(s_logits, t_logits, y_true, T, lam):
    hard = F.nll_loss(F.log_softmax(s_logits, dim=1), y_true)          # H(y_true, P_S)
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=1),               # H(P_T^τ, P_S^τ)
                    F.softmax(t_logits / T, dim=1), reduction='batchmean')
    return hard + lam * soft

def train_fitnet(student, teacher, loader, guided_idx, hint_idx, T, lam, k_size):
    teacher.eval()
    # Stage 1: hint-based pretraining of the student up to the guided layer
    reg = ConvRegressor(student.channels[guided_idx], teacher.channels[hint_idx], k_size)
    lower = student.params_up_to(guided_idx)
    opt1 = torch.optim.SGD(list(lower) + list(reg.parameters()), lr=0.1, momentum=0.9)
    for x, _ in loader:
        with torch.no_grad():
            hint = teacher.features_up_to(x, hint_idx)
        guided = student.features_up_to(x, guided_idx)
        opt1.zero_grad(); hint_loss(hint, guided, reg).backward(); opt1.step()
    # Stage 2: knowledge distillation of the whole student (regressor discarded)
    opt2 = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9)
    for x, y in loader:
        with torch.no_grad():
            t_logits = teacher(x)
        opt2.zero_grad(); kd_loss(student(x), t_logits, y, T, lam).backward(); opt2.step()
    return student
```
