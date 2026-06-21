## Empirical Risk Minimization

Supervised learning asks for a predictor with low expected loss under an unknown joint distribution over inputs and targets. In practice that distribution is replaced by the uniform empirical distribution over the observed training examples, so the training objective becomes a finite list of point constraints.

## Neural Network Capacity

Modern neural networks have parameter counts that can be comparable to, or far larger than, the number of training examples. Randomization tests show that standard architectures can fit random labels with nearly zero training error. Explicit regularizers such as weight decay, dropout, and batch normalization are widely used alongside these architectures. Classical implicit-regularization arguments do not by themselves account for the generalization behavior observed in practice.

## Robustness to Input Perturbations

Networks that classify ordinary test inputs well can change predictions under small optimized perturbations of the input. This behavior has been studied empirically and theoretically; it motivates interest in training procedures that explicitly supervise how the function behaves away from the observed examples.

## Vicinal Supervision

Vicinal risk minimization gives a statistical language for supervising behavior near the training data. Instead of placing all probability mass on each training example, it replaces each atom with a vicinity distribution and trains on virtual examples drawn from those vicinities.

Classic data augmentation fits this frame. Translations, rotations, tangent directions, or Gaussian perturbations enlarge support around examples. These moves typically keep the original label fixed.

## Interpolation and Soft Targets

Interpolation-based augmentation creates synthetic inputs by combining existing examples. Earlier methods of this kind stay within a class or use nearest neighbors, keeping the associated target unchanged.

Soft-target methods reduce overconfidence by spreading probability mass away from the hard one-hot label. Label smoothing assigns a small uniform mass to all classes. Knowledge distillation trains on the soft output distribution of a larger model.
