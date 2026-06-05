# Context

## Research question

Designing convolutional architectures for image classification takes enormous manual engineering — the progression from AlexNet through VGG, Inception, and ResNet was driven by expert intuition about how to wire convolutions, nonlinearities, and connections. Automating that design with search is attractive, but running architecture search *directly on a large dataset like ImageNet* is computationally prohibitive: each candidate must be trained on millions of high-resolution images. The question: can we search on a small **proxy dataset** (CIFAR-10) cheaply, and design the search so that whatever is discovered **transfers** to ImageNet and scales to arbitrary input sizes and compute budgets — without re-searching for every target?

## Background

The field state: convolutional architecture progress comes from hand-designed *repeated motifs* — Inception modules, residual blocks — combinations of filter banks, nonlinearities, and a careful choice of connections, stacked many times to form the full network. This is the key observation the method rests on: good networks are built by repeating a small structural unit, so the depth of the network and the size of the inputs are decoupled from the design of that unit.

- **Neural Architecture Search with RL (Zoph & Le, 2016).** A controller RNN samples a full network architecture token by token; each sampled child is trained to convergence; its validation accuracy R is the reward; the controller is updated by policy gradient (REINFORCE) to raise the probability of high-reward architectures. Effective but searches an *entire* network description, which is slow and tied to the target dataset's scale.
- **Policy-gradient / REINFORCE (Williams, 1992).** The controller's expected reward is maximized by scaling the gradient of the log-probability of its choices by the reward (minus a baseline). The joint probability of a sampled architecture is the product of the per-decision softmax probabilities.
- **Proximal Policy Optimization (Schulman et al., 2017).** A more stable, sample-efficient policy-gradient method using a clipped surrogate objective; a drop-in replacement for REINFORCE that trains the controller faster and more stably.
- **Repeated-motif convolutional design (Inception, ResNet; Szegedy et al.; He et al.).** Networks composed of identical-structure blocks with distinct weights, with a common heuristic: when spatial resolution is halved, double the filter count to keep roughly constant per-layer compute.
- **Stochastic depth / DropPath (Huang et al.; Larsson et al., FractalNet).** Regularize multi-branch nets by stochastically dropping whole paths/branches during training and rescaling at test time.

A diagnostic finding that shapes the search space: applying NAS to ImageNet directly would take months, but architectural elements are known to transfer across datasets. So if the search space is constructed so that the *unit* being searched is dataset- and scale-independent, search can be done cheaply on a proxy and reused.

## Baselines

- **Hand-designed architectures (VGG, Inception, ResNet, Inception-ResNet, DenseNet).** State-of-the-art accuracy from manual engineering. Gap: require expert effort; not automatically tailorable to new compute budgets.
- **NAS-RL on full networks (Zoph & Le, 2016).** Searches the whole network with an RNN controller + REINFORCE. Gap: search cost scales with the target dataset; expensive to run on ImageNet; the searched network is specific to one input scale/depth.
- **Random search (in the same search space).** Sample each controller decision uniformly instead of from the learned policy. A surprisingly strong baseline if the search space is well-designed — useful to isolate how much the RL controller actually contributes.

## Evaluation settings

- **Datasets.** CIFAR-10 (32×32, the proxy search dataset) and ImageNet (299×299 or 331×331, the transfer target); features further transferred to COCO object detection via a detection framework.
- **Search protocol.** Hold out 5,000 CIFAR-10 images as the controller's validation set; whiten and augment (random 32×32 crops from 40×40 upsamples, random horizontal flips). Each sampled child is trained for a fixed short schedule (e.g. 20 epochs) with momentum SGD and cosine learning-rate decay; child accuracy on the held-out set is the controller's reward. A small stack (e.g. N=2 normal cells between reductions) is used during search for speed; the discovered cells are then stacked more deeply for final models.
- **Transfer / scaling.** The discovered Normal and Reduction Cells are stacked into larger networks for ImageNet by choosing the number of cell repeats N and the initial filter count to hit a target compute budget.
- **Metrics.** Top-1/top-5 accuracy (ImageNet), error rate (CIFAR-10), FLOPs and parameter count, and detection mAP (COCO) — plus search cost as a headline axis.

## Code framework

The primitives that already exist: an autodiff framework with conv/separable-conv/pooling/identity ops and an LSTM, SGD with momentum, a policy-gradient training routine, a child-training-and-evaluation loop on a proxy dataset, and the convention of stacking identical-structure blocks with filter-doubling at resolution drops. What does *not* yet exist is the *search space* — how a transferable, scale-independent structural unit is encoded as a sequence of controller decisions — and the regularizer for the resulting multi-branch nets. Those are the empty slots.

```python
import torch, torch.nn as nn

OPS = ['identity', 'conv_1x3_3x1', 'conv_1x7_7x1', 'dil_conv_3x3',
       'avg_pool_3x3', 'max_pool_3x3', 'max_pool_5x5', 'max_pool_7x7',
       'conv_1x1', 'conv_3x3', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7']

class ControllerRNN(nn.Module):
    def __init__(self, hidden=100):
        super().__init__()
        self.lstm = nn.LSTMCell(hidden, hidden)
        # TODO: which sequence of softmax decisions encodes one transferable structural unit?
    def sample(self):
        # TODO: emit the discrete choices that define the unit(s) to be stacked
        pass

def build_network(decisions, N, init_filters, image_scale):
    # TODO: turn the controller's decisions into a stackable unit, then stack it N times
    #       with filter-doubling at resolution drops; same unit reused across scales
    pass

def child_drop_regularizer():
    # TODO: a regularizer suited to the multi-branch repeated units
    pass

def search(controller, proxy_loader):
    # sample child -> train child -> reward = val accuracy -> policy-gradient update
    pass
```
