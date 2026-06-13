# Context

## Research question

Designing convolutional architectures for image classification takes enormous manual engineering -- the progression from AlexNet through VGG, Inception, and ResNet was driven by expert intuition about how to wire convolutions, nonlinearities, and connections. Automating that design with search is attractive, but running architecture search *directly on a large dataset like ImageNet* is computationally prohibitive: each candidate must be trained on millions of high-resolution images. The question: can we search on a small **proxy dataset** (CIFAR-10) cheaply, and design the search so that whatever is discovered **transfers** to ImageNet and scales to arbitrary input sizes and compute budgets without re-searching for every target?

## Background

The field state: convolutional architecture progress comes from hand-designed *repeated motifs* -- Inception modules, residual blocks -- combinations of filter banks, nonlinearities, and a careful choice of connections, stacked many times to form the full network.

- **Neural Architecture Search with RL (Zoph & Le, 2016).** A controller RNN samples a full network architecture token by token; each sampled child is trained to convergence; its validation accuracy R is the reward; the controller is updated by policy gradient to raise the probability of high-reward architectures. Effective but searches an *entire* network description, which is slow and tied to the target dataset's scale.
- **Policy-gradient / REINFORCE (Williams, 1992).** The controller's expected reward is maximized by scaling the gradient of the log-probability of its choices by the reward (minus a baseline). The joint probability of a sampled architecture is the product of the per-decision softmax probabilities.
- **Proximal Policy Optimization (Schulman et al., 2017).** A policy-optimization method that can replace the earlier REINFORCE update for the controller when faster and more stable training is needed.
- **Repeated-motif convolutional design (Inception, ResNet; Szegedy et al.; He et al.).** Networks composed of identical-structure blocks with distinct weights, with a common heuristic: when spatial resolution is halved, double the filter count to keep roughly constant per-layer compute.
- **Stochastic depth / DropPath (Huang et al.; Larsson et al., FractalNet).** Regularize multi-branch nets by stochastically dropping whole paths/branches during training and rescaling at test time.

A diagnostic constraint shapes the search problem: applying architecture search directly to ImageNet is computationally expensive, because each candidate must be trained on millions of high-resolution images before its reward is known, and a network found at one input scale and depth has no built-in reason to remain good at another.

## Baselines

- **Hand-designed architectures (VGG, Inception, ResNet, Inception-ResNet, DenseNet).** Strong accuracy from manual engineering. Gap: require expert effort; not automatically tailorable to new compute budgets.
- **NAS-RL on full networks (Zoph & Le, 2016).** Searches the whole network with an RNN controller + REINFORCE. Gap: search cost scales with the target dataset; expensive to run on ImageNet; the searched network is specific to one input scale/depth.
- **Random search in the same decision space.** Sample each controller decision uniformly instead of from the learned policy. A strong baseline when the decision space itself already concentrates useful architectures.

## Evaluation settings

- **Datasets.** CIFAR-10 (32x32, the proxy search dataset) and ImageNet (224x224 in the constrained setting, 299x299 or 331x331 in larger settings, the transfer target); features can also be transferred to COCO object detection through a detection framework.
- **Search protocol.** Hold out 5,000 CIFAR-10 images as the controller's validation set; whiten and augment with random 32x32 crops from 40x40 upsampled images and random horizontal flips. Each sampled child is trained for 20 epochs with a momentum optimizer (momentum 0.9), L2 weight decay, and cosine learning-rate decay; child accuracy on the held-out set is the controller's reward. A deliberately small, fast-to-train child is used during search for speed. The controller updates in minibatches of 20 completed architectures, and the search stops after 20,000 sampled child models before selecting the top 250 for training to convergence on CIFAR-10.
- **Transfer / scaling.** Whatever is searched on the CIFAR proxy must be re-deployable as a larger network for ImageNet at a chosen compute budget, without re-running the search.
- **Metrics.** Top-1/top-5 accuracy (ImageNet), error rate (CIFAR-10), FLOPs and parameter count, and detection mAP (COCO), plus search cost as a headline axis.

## Code framework

The primitives that already exist: an autodiff framework with convolution, depthwise-separable convolution, pooling, identity, BatchNorm, ReLU, and an LSTM; SGD with momentum for child models; PPO for a recurrent sampler; and a child-training-and-evaluation loop on a proxy dataset.

```python
import torch, torch.nn as nn

class RecurrentSampler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)

def search(sampler, train_loader, valid_loader):
    # TODO: design what the sampler emits and how a child network is built
    # from it, then run sample -> train on proxy -> validation reward -> policy update.
    pass
```
