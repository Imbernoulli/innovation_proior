## Classical Capacity Picture

The inherited statistical picture says that generalization is controlled by choosing a hypothesis class with the right amount of capacity. If the class is too small, empirical risk stays high because the model underfits. If the class is too large, empirical risk can be driven down by fitting idiosyncrasies of the sample, and test risk rises. This produces the familiar U-shaped curve: increasing complexity helps, then hurts.

That picture also shapes training practice. Early stopping, regularization, and explicit model selection are treated as ways to avoid the region where the model can fit too much. The point where train error approaches zero is read as the line past which the learner has crossed into overfitting.

## Modern Interpolation Tension

Deep learning practice does not fit that story cleanly. Standard networks can have far more parameters than training samples, can be trained until the training set is fit nearly perfectly, and can still perform well on new examples. Zhang, Bengio, Hardt, Recht, and Vinyals made the tension sharper by showing that the same broad families can fit random labels, so the capacity to memorize the finite sample is real rather than merely theoretical.

A static capacity measure that says the class is rich enough to fit arbitrary labels assigns the same value to the real-label run that generalizes and the random-label run that does not. Explicit regularizers, meanwhile, are helpful in some cases and not necessary in others.

## Prior Second-Descent Evidence

Belkin, Hsu, Ma, and Mandal have proposed a broader risk curve. They argue that the textbook U-shape is only the part before the interpolation threshold. In random features, decision-tree ensembles, and small neural networks, test risk can rise near the point where the training set is just fit, then decrease again as the function class becomes richer.

That framing changes the meaning of interpolation. It is no longer automatically the end of generalization; it is a boundary between a classical under-parameterized regime and a modern interpolating regime. The evidence and abstraction are organized around model size or number of parameters.

## Training Knobs

Modern training procedures have several knobs that change fitting behavior without changing the nominal architecture. Training for more epochs can let the same model fit examples it could not fit early in training. Data augmentation can make the fitting task harder. Weight decay and other regularizers can make interpolation harder or easier. The optimizer and learning-rate schedule can also move the point where train error reaches zero.

## Experimental Ingredients

The available ingredients are standard: width-scaled ResNet18s and five-layer CNNs for CIFAR-10/CIFAR-100, width-scaled encoder-decoder Transformers for translation, train/test error traces, controlled label noise, augmentation and regularization choices, and random-feature models as a tractable anchor. The measurement of interest is not just final test error; it is the relationship between test error and the point where training error first becomes approximately zero.

The question is how to relate these ingredients: how a fixed large network behaves over training time, why changing data augmentation or label noise moves the test-error peak, why a barely fitting model behaves differently from one trained well past that boundary, and how increasing the number of training samples affects the curve near that boundary.
