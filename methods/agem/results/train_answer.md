The problem is learning from a stream of tasks where each task is seen once, in a single pass, and at the end the model is evaluated on every task it encountered. The central failure is catastrophic forgetting: ordinary gradient descent on the current task's loss has no reason to preserve what it learned about earlier tasks, so accuracy on past tasks collapses once training moves on. The setting comes with hard constraints that rule out the obvious cures. Storing all past data and retraining on it turns the problem back into ordinary multitask learning and violates the bounded-memory, streaming premise. Regularization methods such as Elastic Weight Consolidation add a quadratic penalty that pulls important parameters back to their old values, but they were designed for multi-epoch training, are fragile to their strength hyperparameter in single-pass regimes, and forbid positive backward transfer by clamping coordinates rather than judging moves by their actual effect on past losses. Architectural methods that grow a new module per task eliminate forgetting by construction, but their memory and compute grow with the number of tasks, which is infeasible for a long stream. What is needed is a method that keeps bounded memory, keeps per-step cost nearly that of plain SGD, controls forgetting, and still allows old tasks to improve.

The right diagnostic comes from looking at the geometry of a single gradient step. A step theta <- theta - alpha g changes a past task's loss by approximately -alpha <g, g_k>, where g_k is the gradient of that past task's loss at the current parameters. So the sign of the inner product between the current gradient g and a past-task gradient g_k tells the whole story to first order: a positive inner product means the step decreases the past loss, while a negative inner product means it increases it. The way to avoid forgetting is therefore to police the direction of each gradient step, not to anchor parameters. This requires a small episodic memory of past examples to measure those past-task gradients, but the memory is used only to evaluate a direction, not to rehearse on, and it can be bounded.

The method I propose is A-GEM, Averaged Gradient Episodic Memory. It is the efficient successor to Gradient Episodic Memory. GEM formulates the constraint as one inequality per past task, requiring <g_tilde, g_k> >= 0 for every previous task, which leads to a quadratic program solved through its dual and costs one backward pass per past task at every step. That per-step cost grows with the stream and eventually becomes prohibitive. A-GEM instead replaces the per-task constraints with a single averaged constraint: it requires only that the average loss over the union of all past memories does not increase. Let g_ref be the gradient of the loss on a random minibatch drawn from the combined episodic memory. The constraint collapses to <g_tilde, g_ref> >= 0, a single linear inequality. This trades GEM's worst-case guarantee on every individual past task for an average-loss guarantee that is aligned with the headline average-accuracy metric.

Projecting the current gradient onto this halfspace has a closed form. If the current gradient already satisfies g^T g_ref >= 0, the step is compatible with the past on average and we take it unchanged, so backward transfer is allowed when it occurs naturally. If it violates the constraint, we subtract the component of g along g_ref: g_tilde = g - (g^T g_ref / g_ref^T g_ref) g_ref. In that projected case g_tilde^T g_ref = 0, so the corrected gradient is orthogonal to the reference direction: it does no average harm to past tasks while staying as close as possible to the original update in Euclidean norm. This preserves the current task's signal as much as the constraint allows. The per-step cost is one extra backward pass to compute g_ref and two dot products over the flattened gradient, independent of the number of tasks.

The memory is a bounded buffer shared across completed tasks. At the end of each task a fixed budget of examples is stored, and the reference batch is sampled uniformly from that buffer. The first task trains as plain SGD because there is no past memory yet. The implementation flattens the per-parameter grad fields into two preallocated buffers, computes the projection in flat space, and writes the result back before the optimizer step.

```python
import numpy as np
import torch

from models.gem import overwrite_grad, store_grad
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGem(ContinualModel):
    NAME = 'agem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(AGem, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, dataset):
        samples_per_task = self.args.buffer_size // dataset.N_TASKS
        loader = dataset.train_loader
        cur_y, cur_x = next(iter(loader))[1:]
        self.buffer.add_data(
            examples=cur_x.to(self.device),
            labels=cur_y.to(self.device)
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.zero_grad()
        p = self.net.forward(inputs)
        loss = self.loss(p, labels)
        loss.backward()

        if not self.buffer.is_empty():
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )
            self.net.zero_grad()
            buf_outputs = self.net.forward(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        self.opt.step()
        return loss.item()
```
