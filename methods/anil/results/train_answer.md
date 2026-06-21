Gradient-based meta-learning trains an initialization that can be adapted to a new task from only a few labelled examples. The standard recipe, typified by MAML, runs a differentiable inner optimization over every parameter for each sampled task, then differentiates the query loss back through those inner steps. That design is expensive because each inner step retains computation graph and incurs Hessian-vector products. The deeper question is whether the cost is actually buying representational change: does the body of the network rapidly specialize to each task, or does the inner loop mainly align a task-specific head? The evidence from freezing experiments and representational-similarity diagnostics points strongly to the latter. On MiniImageNet, freezing all convolutional body layers during test-time adaptation barely changes accuracy, and CCA/CKA similarity between body activations before and after the inner loop stays above 0.9, while the head drops below 0.5. The body is reused; the head moves.

The natural conclusion is that the inner loop on the body is a near-no-op, and the principled efficiency cut is to remove it while keeping the head adaptation intact. The head must still adapt because each N-way episode assigns a fresh set of classes to the output neurons, so a fixed classifier cannot align to arbitrary labels. The body, however, can be frozen during task adaptation and learned only through the outer loop. This is not a first-order approximation: the head's inner gradient is computed through a forward pass over the body, so differentiating the query loss with respect to the body still flows through the head update and retains a cross-Hessian second-order term. The method simply shrinks which parameters adapt, not the derivative order through the surviving parameters.

The method is ANIL, short for Almost No Inner Loop. It keeps the standard MAML outer objective but adapts only the final classification head during the inner loop. For a model partitioned into body parameters and head parameters, the inner update for task b is

theta_m^(b) = (theta_1, ..., theta_{l-1}, (theta_l)_{m-1}^(b) - alpha nabla_{(theta_l)_{m-1}^(b)} L_{S_b}(f_{theta_{m-1}^(b)})),

where theta_l denotes the head. The body entries are copied from the meta-initialization at every inner step, while the head takes ordinary gradient-descent steps on the support loss. The meta-loss is unchanged, and the outer loop still updates all parameters, including the body, through the query loss. At test time, the same feature-reuse logic can be pushed even further into the NIL endpoint: discard the head and classify queries by cosine similarity between support and query embeddings. That test-time variant is optional; the training rule keeps the task-specific head adaptation because it is the task-specific pressure that drives the body to learn reusable features.

Implementation requires care because learn2learn's update_module replaces parameter objects at every step, so head tensors must be re-identified by name rather than cached by object identity. Gradients are taken only with respect to the current head parameters with create_graph=True, and update_module receives a full-length update list: -inner_lr * grad for head parameters and zeros_like(p) for body parameters. The zero body updates keep the direct graph from the cloned model back to the meta-initialization while deliberately omitting any body inner update. There is no learned optimizer state.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Set

import learn2learn as l2l

INNER_LR = 0.5


class InnerLoopOptimizer:
    """ANIL: adapt only the final task-specific head in the inner loop."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        self._head_param_names = self._infer_head_param_names(model)
        if not self._head_param_names:
            raise ValueError("could not identify final classifier/head parameters")

    @staticmethod
    def _infer_head_param_names(model: nn.Module) -> Set[str]:
        named = list(model.named_parameters())
        all_names = {name for name, _ in named}

        for token in ("classifier", "head"):
            selected = {
                name for name in all_names
                if token in name.lower().split(".")
            }
            if selected:
                return selected

        last_linear = None
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                direct = list(module.named_parameters(recurse=False))
                if direct:
                    last_linear = (module_name, direct)
        if last_linear is not None:
            module_name, direct = last_linear
            return {
                f"{module_name}.{param_name}" if module_name else param_name
                for param_name, _ in direct
            }

        last_param_module = None
        for module_name, module in model.named_modules():
            direct = list(module.named_parameters(recurse=False))
            if direct:
                last_param_module = (module_name, direct)
        if last_param_module is None:
            return set()
        module_name, direct = last_param_module
        return {
            f"{module_name}.{param_name}" if module_name else param_name
            for param_name, _ in direct
        }

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            named_params = list(model.named_parameters())
            head = [
                (name, param) for name, param in named_params
                if name in self._head_param_names
            ]
            if not head:
                raise RuntimeError("head parameters not found on adapted clone")

            head_names = [name for name, _ in head]
            head_params = [param for _, param in head]
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(loss, head_params, create_graph=True)
            grad_by_name = dict(zip(head_names, grads))

            updates = [
                -self.inner_lr * grad_by_name[name]
                if name in grad_by_name else torch.zeros_like(param)
                for name, param in named_params
            ]
            l2l.update_module(model, updates=updates)
        return model

    def meta_parameters(self) -> List[Tensor]:
        return []
```
