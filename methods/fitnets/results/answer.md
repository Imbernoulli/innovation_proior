# FitNets

## Problem

Train a student network that is **thinner and deeper** than a wide teacher: fewer parameters/multiplies per layer, more layers for expressiveness, and a good optimization path despite the depth. Deep thin nets are hard to optimize, and knowledge distillation that matches only the teacher's output injects supervision at the top of the network, too weak to steer a much deeper student's interior.

## Key idea

Use the teacher's intermediate representation as a **hint** for a middle layer of the student, the **guided** layer. A regressor on top of the guided activation predicts the teacher hint, then training proceeds in two stages: intermediate pretraining first, full output distillation second.

**Hint loss (stage 1).** Because the teacher is wider, its hint layer can have more outputs than the student's guided layer. Add a regressor r on top of the guided layer so r(v_g(...)) has the teacher hint's shape, and minimize

```
L_HT(W_Guided, W_r) = ½ ‖ u_h(x; W_Hint) − r( v_g(x; W_Guided); W_r ) ‖²,
```

where u_h / v_g are the teacher/student nested functions up to the hint/guided layers. W_Hint is fixed; W_Guided and W_r are optimized. The regressor is **convolutional** rather than fully connected when the matched layers are conv layers: choose kernel k_i so N_{g,i} - k_i + 1 = N_{h,i}, giving a weight tensor with k_1*k_2*O_h*O_g entries instead of the dense N_{h,1}*N_{h,2}*O_h*N_{g,1}*N_{g,2}*O_g. Hint and guided layers are placed near the middle of teacher and student, because placing the guided layer too deep pins too much of the student and over-regularizes it.

**Distillation loss (stage 2).** With temperature tau > 1, P^tau = softmax(a / tau):

```
L_KD(W_S) = H(y_true, P_S) + lambda * H(P_T^tau, P_S^tau),
```

H is cross-entropy. The teacher weight lambda is initialized high and linearly decayed during the KD stage, so confident teacher examples have stronger influence early and progressively less as training proceeds.

## Algorithm (stage-wise)

```
Input: trained teacher W_T, random student W_S, indices g (guided), h (hint)
1. W_Hint   ← teacher params up to layer h
2. W_Guided ← student params up to layer g
3. initialize regressor W_r to small random values
4. W_Guided*, W_r* ← argmin_{W_Guided, W_r} L_HT(W_Guided, W_r)  # pretrain lower student + regressor
5. copy W_Guided* into the student's first g layers; discard W_r*
6. W_S* ← argmin L_KD(W_S)                        # distill the whole student
```

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ConvRegressor(nn.Module):                       # maps guided activation to hint shape
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k)
    def forward(self, v):
        return self.conv(v)

def hint_loss(hint, guided, regressor):
    return 0.5 * F.mse_loss(regressor(guided), hint, reduction='sum')

def soft_cross_entropy(s_logits, teacher_probs, T):
    log_probs = F.log_softmax(s_logits / T, dim=1)
    return -(teacher_probs * log_probs).sum(dim=1).mean()

def linear_decay(step, total_steps, start, end):
    if total_steps <= 1:
        return end
    alpha = min(step / (total_steps - 1), 1.0)
    return start + alpha * (end - start)

def kd_loss(s_logits, t_logits, y_true, T, lam):
    hard = F.cross_entropy(s_logits, y_true)                          # H(y_true, P_S)
    soft = soft_cross_entropy(s_logits, F.softmax(t_logits / T, dim=1), T)
    return hard + lam * soft

def train_fitnet(student, teacher, loader, guided_idx, hint_idx, k_size,
                 T=3.0, lambda_start=4.0, lambda_end=1.0,
                 stage1_epochs=500, stage2_epochs=500, lr=0.005):
    teacher.eval()
    # Pretrain W_Guided and W_r against the fixed teacher hint.
    reg = ConvRegressor(student.channels[guided_idx], teacher.channels[hint_idx], k_size)
    lower = student.params_up_to(guided_idx)
    opt1 = torch.optim.RMSprop(list(lower) + list(reg.parameters()), lr=lr)
    for _ in range(stage1_epochs):
        for x, _ in loader:
            with torch.no_grad():
                hint = teacher.features_up_to(x, hint_idx)
            guided = student.features_up_to(x, guided_idx)
            opt1.zero_grad(); hint_loss(hint, guided, reg).backward(); opt1.step()

    # Discard the regressor; KD trains the whole student on teacher outputs.
    opt2 = torch.optim.RMSprop(student.parameters(), lr=lr)
    total_steps = stage2_epochs * len(loader)
    step = 0
    for _ in range(stage2_epochs):
        for x, y in loader:
            with torch.no_grad():
                t_logits = teacher(x)
            lam = linear_decay(step, total_steps, lambda_start, lambda_end)
            opt2.zero_grad(); kd_loss(student(x), t_logits, y, T, lam).backward(); opt2.step()
            step += 1
    return student
```
