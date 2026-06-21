# Context

## Research question

The capacity of a neural network to absorb information from a training corpus is
bounded by its number of parameters. Across text, images, and audio, larger
datasets reward larger models with markedly better accuracy. But for an ordinary
dense network — one in which every parameter participates in every example — the
training cost scales roughly as the product of model size and dataset size, i.e.
quadratically as we grow both together. Compute and inter-device bandwidth are
not keeping pace with that demand.

How can a model's parameter count be made very large while its computation per
example stays roughly fixed, trainable end-to-end, and efficient on GPU clusters
whose compute far outpaces their network bandwidth?

## Background

**Conditional computation.** The idea that only parts of a network should be
active per example was proposed as the route to capacity-without-compute. Davis &
Arel (2013) study low-rank approximations for conditionally activating feedforward
units. Bengio, Léonard & Courville (2013) study how to estimate or propagate
gradients through stochastic/binary neurons (straight-through and REINFORCE-style
estimators), since a hard on/off gate is non-differentiable. Bengio, Bacon,
Pineau & Precup (2015) train boolean gates that switch network blocks on and off
using a REINFORCE-style policy, with several auxiliary losses to control sparsity
and balance. Cho & Bengio (2014) exponentially increase the capacity-to-
computation ratio with a parameterized weight matrix selected per input. Eigen,
Ranzato & Sutskever (2013) stack gated mixtures as components inside a deep net.

**Gate behavior in trainable mixtures.** When a softmax gate over experts is
trained jointly with the experts, it tends to converge to a state where it
assigns large weight to the same few experts on every example. Eigen et al.
(2013) report this and sidestep it with a hard constraint at the start of
training; Bengio et al. (2015) add a soft constraint on the batch-wise average
of each gate.

**Mixtures of experts.** Jacobs, Jordan, Nowlan & Hinton (1991) and Jordan &
Jacobs (1994) introduced adaptive mixtures of local experts: a set of expert
sub-models plus a gating network that produces a distribution over them, with the
mixture output a gate-weighted combination of expert outputs. Trained jointly,
experts specialize to regions of the input space. In these formulations the
mixture *is* the whole model and the gate is dense (every expert is evaluated).

**The text testbed.** Language modeling and machine translation are the domains
known to keep improving with model size. Stacked Long Short-Term Memory networks
(Hochreiter & Schmidhuber, 1997; Gers et al., 2000) are the workhorse sequence
model; Jozefowicz et al. (2016) push LSTM language models to ~10^8 parameters and
show quality climbing with capacity. For translation, the Google NMT system (Wu
et al., 2016) stacks deep LSTM encoder/decoder layers with attention and
wordpiece vocabularies. These models are dense: capacity and compute rise
together.

## Baselines

- **Dense softmax-gated mixture of experts (Jacobs/Jordan).** Gate
  `G(x) = Softmax(x·W_g)` over `n` experts; output `y = Σ_i G(x)_i E_i(x)`.
  Trained end-to-end by backprop. All `n` experts run for every example.

- **Stacked deep MoE components (Eigen et al., 2013).** Use MoEs as layers inside
  a deep network, each with its own gate, rather than as the whole model. Combat
  gate collapse with a hard constraint at the start of training.

- **REINFORCE-trained boolean gates (Bengio et al., 2015).** Hard on/off gates on
  network blocks, trained with a policy-gradient estimator, plus three auxiliary
  losses (per-example sparsity, batch-wise balance, gate diversity). Demonstrated
  on image datasets of up to ~600k examples.

- **Dense LSTM language models / GNMT (Jozefowicz et al., 2016; Wu et al., 2016).**
  Strong sequence baselines whose quality improves with parameter count, at a
  proportional compute cost.

## Evaluation settings

- **1-Billion-Word Language Modeling Benchmark** (Chelba et al., 2013): ~829M
  words of shuffled unique news sentences, vocabulary ~793k. Metric: test
  perplexity (summed over all tokens including end-of-sentence).
- **A ~100-billion-word news corpus**: a much larger shuffled-sentence corpus to
  test whether capacity keeps paying off at greater data scale. Metric: test
  perplexity.
- **WMT'14 English→French (36M pairs) and English→German (5M pairs)**: newstest2014
  as test set; newstest2012+2013 as development set. Metrics: test perplexity and
  tokenized BLEU (multi-bleu.pl). Wordpiece vocabularies; beam search.
- **A multilingual translation setup** over twelve language pairs trained jointly
  in one model. Metrics: per-pair BLEU and dev perplexity.
- Reported as engineering context: compute budget measured in multiply-and-adds
  per token per timestep ("ops/timestep") and realized TFLOPS/GPU on clusters of
  K40 GPUs.

## Code framework

The primitives that already exist: a tensor library with autograd, `nn.Linear`,
`ReLU`, `Softmax`, `Softplus`, a normal distribution with a CDF, and an Adam
optimizer; a sequence model (stacked LSTM layers, embeddings, a softmax output)
into which a new layer can be dropped at one position and applied at every
timestep.

```python
import torch
import torch.nn as nn

class Expert(nn.Module):
    """One sub-network. Same input/output size as its siblings; separate params."""
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ConditionalLayer(nn.Module):
    """A drop-in layer that holds many experts but should cost the compute of only
    a few per example."""
    def __init__(self, input_size, output_size, num_experts, hidden_size, k):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(input_size, output_size, hidden_size) for _ in range(num_experts)]
        )
        # TODO: your approach here
        pass

    def forward(self, x):
        # TODO: your approach here
        pass
```
