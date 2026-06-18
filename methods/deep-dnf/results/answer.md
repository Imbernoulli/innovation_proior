# Neural Logic Networks: differentiable DNF

Represent Boolean truth values in `[0,1]` and use product logic:

```text
not(x) = 1 - x
and(x, y) = x*y
or(x, y) = 1 - (1 - x)(1 - y)
```

For each neuron, learn one scalar membership per input:

```text
m_i = sigmoid(c*w_i), c >= 1
```

Conjunction neuron:

```text
O_conj(x) = product_i (1 - m_i(1 - x_i))
```

All cases for one factor:

```text
(x_i, m_i) = (0,0) -> 1
(x_i, m_i) = (1,0) -> 1
(x_i, m_i) = (0,1) -> 0
(x_i, m_i) = (1,1) -> 1
```

Disjunction neuron:

```text
O_disj(x) = 1 - product_i (1 - m_i*x_i)
```

All cases for the selected contribution `m_i*x_i`:

```text
(0,0) -> 0
(1,0) -> 0
(0,1) -> 0
(1,1) -> 1
```

A DNF is a conjunction layer followed by a disjunction layer:

```text
DNF(x) = DISJ(CONJ(x))
```

The derivative that controls membership learning is:

```text
d O_conj / d m_i =
  -(1 - x_i) product_{k != i}(1 - m_k(1 - x_k))
```

So only examples with `x_i = 0` move `m_i` directly; the loss sign decides whether this
pushes the membership up or down.

Reference-faithful TensorFlow shape:

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

For long conjunctions, the same product can be evaluated in a stable log form:

```text
product_i factor_i = exp(sum_i log(epsilon + factor_i))
```

For mixed-polarity rules, feed negated atoms or `1 - x_i` as additional input items; each
item still has one scalar membership. In wide layers, initialize most memberships near
zero, either with a small subset near one or with negative-mean logits, so the initial
products do not crush the gradients. After training, read a formula by thresholding
conjunction memberships for literals and disjunction memberships for active clauses;
prune any thresholded-on literal whose removal leaves the loss unchanged.
