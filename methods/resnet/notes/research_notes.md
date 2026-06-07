# Research notes — ResNet ancestors, field state, explainers, code

## Primary paper (read in full, incl. appendices A/B/C)
He, Zhang, Ren, Sun. "Deep Residual Learning for Image Recognition", arXiv:1512.03385 (Dec 2015), CVPR 2016.
Key eqs verified against src/residual_v1_arxiv_release.tex:
- Eq (1): y = F(x, {W_i}) + x   (identity shortcut, parameter-free)
- Eq (2): y = F(x, {W_i}) + W_s x   (projection shortcut, only to match dims)
- F for 2-layer block: F = W2 σ(W1 x), σ = ReLU; second ReLU applied AFTER addition.
- Bottleneck: 1x1 (reduce) -> 3x3 -> 1x1 (restore), expansion=4.
- ImageNet: conv1 7x7/64 stride2; maxpool 3x3 stride2; conv2_x..conv5_x; global avg pool; 1000-fc.
  18:[2,2,2,2] 34:[3,4,6,3] 50:[3,4,6,3]bottleneck 101:[3,4,23,3] 152:[3,8,36,3].
- CIFAR: 6n+2 layers, filters {16,32,64} on maps {32,16,8}, 2n blocks each, identity (option A) only.
- Training: SGD, BN after each conv before activation, He(2015) init, lr 0.1 /10 on plateau, wd 1e-4, mom 0.9, no dropout, batch 256, up to 60e4 iters.

## Load-bearing ancestors (verified facts)
- VGG (Simonyan & Zisserman, ICLR'15): "very deep" 16-19 layers, stacked 3x3 conv, design rule: same map size -> same #filters; halve map -> double filters. VGG-19 = 19.6 GFLOPs. ResNet borrows the 3x3/doubling philosophy but is thinner (34-layer = 3.6 GFLOPs = 18% of VGG-19). Gap: VGG stopped at ~19 layers; deeper plain stacks degrade.
- GoogLeNet/Inception (Szegedy et al, CVPR'15): 22 layers, inception modules, auxiliary classifiers to fight vanishing gradients, ILSVRC'14 winner ~6.67% top-5. Gap: complex hand-designed modules; aux classifiers are a band-aid for depth.
- Batch Norm (Ioffe & Szegedy, ICML'15): normalize each conv output over the mini-batch to zero-mean/unit-var, learnable scale/shift; lets ~30-layer nets converge, 14x fewer steps. Critical: it solves vanishing/exploding-signal at the START. ResNet relies on it so that the *remaining* failure (degradation) cannot be blamed on vanishing gradients.
- He init / PReLU (He et al, ICCV'15): variance-preserving init for ReLU nets (fan_out kaiming_normal). Used to init all ResNet layers.
- Highway Networks (Srivastava, Greff, Schmidhuber, 2015): y = H(x)·T(x) + x·C(x), gates T,C data-dependent w/ params (LSTM-style gating). CONCURRENT. Gap vs ResNet: gates have parameters and can "close" (then layer is non-residual); had not shown gains beyond ~100 layers. ResNet's identity shortcut is parameter-free, never closes, always passes all info.
- Degradation observation (He & Sun "Constrained time cost" CVPR'15; Srivastava Highway): deeper plain nets get HIGHER training error — not overfitting. This is the pain point ResNet targets.
- AlexNet (Krizhevsky 2012), NIN (Lin 2013, global avg pool + 1x1 conv), ReLU (Nair&Hinton 2010), vanishing/exploding gradients (Bengio 1994, Glorot 2010), Caffe (Jia 2014) framework.
- Residual-representation analogies (motivation, not mechanism): VLAD/Fisher (encode residual to dictionary), Multigrid / hierarchical-basis preconditioning (solve residual between scales -> faster convergence). Used to argue "reformulating around a residual eases optimization".

## Field state at the time (late 2015)
Prevailing wisdom: depth is king (VGG, GoogLeNet, BN-inception all winning by going deeper). Vanishing/exploding gradients considered "largely solved" by normalized init + BN. Open pain: once nets converge, going deeper than ~20-30 layers makes TRAINING error go up — counterintuitive, unexplained. People patched it with auxiliary losses (GoogLeNet, DSN) and gated shortcuts (Highway) but none got clean gains past 100 layers.

## Key intuitions (from explainers, for reasoning.md)
- The "identity by construction" argument: a deeper net can always match a shallower one by setting extra layers = identity, so it should never be worse. SGD fails to find this -> optimization, not representation, problem.
- Residual reformulation: fit F(x)=H(x)-x; if optimal map is near identity, pushing F->0 is easy (drive weights to 0); learning identity from scratch through nonlinear stacks is hard -> preconditioning.
- Gradient flow: with y=F(x)+x, d y/d x = F'(x) + 1; the +1 gives an unimpeded path so gradient never fully vanishes/explodes across blocks. (NOTE: paper itself argues vanishing is NOT the cause of degradation since BN keeps signals healthy; the +1 argument is the complementary explainer view — keep both, attributed correctly.)
- Empirical support in paper: layer-response std (Fig std) shows residual functions have small responses -> identity is good preconditioning; deeper ResNet -> each layer modifies signal less.

## Canonical code
torchvision resnet.py (v0.4.0, clean) saved to code/torchvision_resnet_v0.4.py and (v0.13.0) torchvision_resnet.py.
Structure: conv3x3/conv1x1 helpers, BasicBlock (expansion=1), Bottleneck (expansion=4), ResNet with _make_layer, kaiming_normal_ fan_out init, downsample = conv1x1(stride)+BN. resnet{18,34,50,101,152} = ResNet(block,[..]).

## Injection check
No prompt-injection encountered in fetched pages/searches.
