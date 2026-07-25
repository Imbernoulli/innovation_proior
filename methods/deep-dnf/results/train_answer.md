The problem is supervised learning of Boolean functions over binary inputs. The target is a disjunctive normal formula — a disjunction of conjunctions of literals — and the goal is a model that is simultaneously trainable by ordinary gradient descent and interpretable as a symbolic rule after training. Standard additive MLPs fail the second requirement because a threshold plus a learned bias can implement a conjunction only up to arbitrary scaling; once training finishes there is no clean way to read off which variables belong to which clause. The same real-valued weights might represent the same Boolean behavior with shifted or scaled parameters, so a human or downstream system must impose an external thresholding convention that is not part of the learned object. Decision trees and ensembles can represent DNF exactly, but they give up the compact parametric form and cannot be tuned end-to-end with backprop in the same way as a neural network. What is needed is a differentiable architecture whose parameters are themselves the membership decisions of a DNF.

The proposed method is deep-dnf, a neural logic network that implements a differentiable DNF. The idea is to abandon additive thresholds for Boolean operations and instead use product logic on soft truth values in [0,1]. Conjunction is multiplication, so a clause is satisfied only when every selected literal is true. Disjunction is the De Morgan dual, 1 minus the product of one minus the selected inputs, which is the noisy-OR form. Each neuron learns one scalar membership per input through a sharpened sigmoid, $m_i = \mathrm{sigmoid}(c\,w_i)$ with $c \ge 1$: a partially-trained soft membership already ranks a satisfying input above a violating one, and raising $c$ sharpens that ranking toward a clean 0/1 decision as training proceeds. For a conjunction, the factor for input i is 1 - m_i(1 - x_i): if m_i is near zero the variable is skipped and the factor contributes 1, and if m_i is near one the factor becomes x_i and the variable is required. For a disjunction, the contribution is m_i x_i, so the OR fires when any selected input is true. A DNF is built by composing a bank of conjunction neurons, `logic_layer_and`, followed by one disjunction neuron over their outputs, `logic_layer_or`, wired together as `dnf`; the reverse composition gives a CNF.

Training proceeds with ordinary backpropagation and a cross-entropy loss. The gradient for a conjunction membership is nonzero only on examples where the corresponding input is 0, which is exactly when that literal would suppress an overly broad clause; the sign of the loss decides whether the membership should move up or down. The membership logits are initialized sparsely so most clauses start inactive, which avoids the gradient-collapse problem caused by long products of small factors. For numerical stability the long products are evaluated in log space as exp(sum log(factor)), and factors are clamped away from zero. For mixed-polarity rules the input vocabulary is augmented with negated atoms, and the same scalar membership can select either polarity. Because the membership parameters are soft during training but converge toward zero or one, the final formula is recovered by thresholding the memberships and printing the corresponding literals and clauses. Any literal whose removal does not change the loss can be pruned for a cleaner rule. The method is therefore a fully differentiable DNF learner that retains a readable symbolic description.

```python
def sharp_sigmoid(w, c=1.0):
    return tf.sigmoid(c * w)

def logic_layer_and(x, W, c=1.0):
    m = sharp_sigmoid(W, c)                 # [units, n]
    z = tf.expand_dims(m, 0) * (1.0 - tf.expand_dims(x, -2))
    return tf.reduce_prod(1.0 - z, axis=-1)

def logic_layer_or(x, W, c=1.0):
    m = sharp_sigmoid(W, c)                 # [units, n]
    z = tf.expand_dims(m, 0) * tf.expand_dims(x, -2)
    return 1.0 - tf.reduce_prod(1.0 - z, axis=-1)

def dnf(x, W_and, W_or, c=1.0):
    clauses = logic_layer_and(x, W_and, c)
    return logic_layer_or(clauses, W_or, c)
```
