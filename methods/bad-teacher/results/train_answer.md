We need to remove the influence of a forget set from an already-trained deep classifier without retraining from scratch, which is the expensive but otherwise gold-standard solution. The real task is approximate unlearning: nudge the existing weights so the model behaves like one that was trained only on the retained data. That means full accuracy on the retained set, but on the forget set the model should show generalization-level uncertainty, not confident wrong answers. Confidently misclassifying the forget set is itself a privacy leak, because a membership-inference attacker can notice that those samples were specially treated.

Existing ideas all fall short in ways that matter here. Retraining is correct but too costly per request. SISA requires sharding the data and checkpointing many models before training even begins, so it cannot help an already-deployed monolithic network. Fisher scrubbing and its NTK variants need second-order information or an auxiliary linearized model, and they assume the original model was trained with ordinary SGD. UNSIR works only at the class level and pushes forget accuracy exactly to zero, which is the confidently-wrong regime we want to avoid. NegGrad+ actively ascends the forget cross-entropy, but that signal is unbounded: the ascent keeps producing gradient after the retain loss has saturated, so the weights run off and retain accuracy collapses. What is missing is a bounded forget signal that has a fixed destination, namely the behavior of a model that never learned the forget data at all.

The method I propose is Bad Teacher, also called incompetent-teacher unlearning. It is a selective knowledge-distillation procedure that uses two frozen teachers and routes each sample to the right one. The competent teacher is a frozen copy of the original fully trained model; it supplies the target behavior on retained data. The incompetent teacher is a freshly random-initialized copy of the same architecture, never trained; it supplies the target behavior on forget data. The student is the model we are actually producing, and it is initialized to the original trained weights so that retain utility starts intact and only needs to be preserved, not relearned.

The key insight is that distillation makes the student copy whatever distribution a teacher emits, regardless of whether that teacher is good. On a retain sample we therefore minimize KL divergence from the competent teacher's soft distribution to the student's distribution. On a forget sample we minimize KL divergence from the incompetent teacher's random distribution to the student's distribution. Because the forget target is an untrained output, the student is pulled toward genuine uncertainty rather than toward a confident wrong class. Because retain and forget samples are mixed in the same loader, the competent teacher keeps the shared features useful for retained classes while the random target erases only the specific forget-class information. The per-sample loss is simply a single KL against a target constructed from the two teachers using the unlearning label.

Concretely, a small dataset wrapper packs the forget set and the retain set into one pool, tagging each item with a label that is 1 for forget and 0 for retain; a shuffled loader draws mixed mini-batches from this pool, so retain and forget samples train side by side within every batch. For each batch the two frozen teachers are forwarded under no_grad with both set to eval mode, their logits softened to probabilities at temperature T, and combined by the label into the mixed target: label 1 selects the incompetent teacher's diffuse output, label 0 selects the competent teacher's output. The student enters the KL as log-softmax at the same temperature, and I use T = 1 by default, because the competent teacher's natural probabilities are exactly the retain behavior we want to copy and the incompetent teacher is already diffuse; there is no hard-label term mixed in, so no T-squared scaling factor is needed. I leave PyTorch's default mean reduction on the KL rather than batchmean — it still points the gradient in the same direction, just rescaled by the number of classes, and keeping it matches the scale the rest of the routine's learning rate is tuned against. Only the student's parameters receive optimizer updates, one Adam step per mixed batch, repeated for a fixed number of epochs while the running loss is printed each epoch.

```python
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            return self.forget_data[index][0], 1
        return self.retain_data[index - self.forget_len][0], 0


def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample; label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)


def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader,
                    optimizer, device, KL_temperature):
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(
            output=output,
            labels=y,
            full_teacher_logits=full_teacher_logits,
            unlearn_teacher_logits=unlearn_teacher_logits,
            KL_temperature=KL_temperature,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, retain_data, forget_data,
                        epochs=10, optimizer='adam', lr=0.01, batch_size=256, num_workers=32,
                        device='cuda', KL_temperature=1):
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(
        unlearning_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = unlearning_step(
            model=model,
            unlearning_teacher=unlearning_teacher,
            full_trained_teacher=full_trained_teacher,
            unlearn_data_loader=unlearning_loader,
            optimizer=optimizer,
            device=device,
            KL_temperature=KL_temperature,
        )
        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))
```
