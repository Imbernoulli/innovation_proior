# Context: combining many feature sources into one (multi-source / multi-scale feature fusion, circa 2016-2020)

## Research question

A recurring pattern in deep vision and multi-source models: at each spatial location you hold a set of
`V` feature maps coming from *different sources* — different pyramid levels, different resolutions,
different input modalities/variables — and you must collapse them into a single representation per
location to feed onward. The sources are heterogeneous: they were produced by different pathways, carry
different kinds of information, and there is no reason to believe each one is equally useful to the
prediction at hand. The standard move is to resize them to a common shape and add (or average) them, but
that hard-codes a decision — every source counts the same — that nobody actually checked.

The precise problem is to design the aggregation transformation `f: (I_1, ..., I_V) -> O` that combines
`V` aligned feature maps into one, subject to several pressures at once: (1) it should let *different
sources contribute differently*, because they demonstrably do; (2) it should be cheap — adding it on top
of an already-large backbone must cost almost nothing in parameters and FLOPs, ideally negligible
relative to the convolutions/attention around it; (3) it should be *stable to train* end-to-end by
ordinary backprop, with no projection step or special optimizer; and (4) the fused output should stay on
the same scale as its inputs, so the rest of the network sees a representation it can interpret rather
than one with an arbitrary gain baked in. Plain summation gets (2)-(4) for free but fails (1). The
expressive alternatives that get (1) pay heavily on (2). The gap is a fusion rule that buys real
expressiveness over equal-weight combination while staying essentially free.

## Background

By the late 2010s, *multi-scale feature fusion* is the workhorse of dense prediction. A deep ConvNet
already produces a feature hierarchy — feature maps at decreasing spatial resolution and increasing
semantic depth — and the central difficulty of detection/segmentation is that objects appear at many
scales, so predictions want to draw on several of these levels at once. The dominant idea is to build an
in-network *feature pyramid*: take the per-stage feature maps `{C_2, C_3, C_4, C_5}` and combine them
across levels so that every output level has both high resolution and strong semantics.

The combining step inside all of these pyramids is, at bottom, the same primitive: bring several feature
maps to a common resolution (a `1x1` conv to match channels, an upsample or downsample to match spatial
size) and merge them. The merge has almost universally been an *equal-weight* operation — element-wise
addition, occasionally element-wise max, or an unweighted average. That choice is rarely examined; it is
inherited as "the obvious thing."

There is, however, direct diagnostic evidence that equal weighting is the wrong default. When PANet
(below) instrumented *which* pyramid level each pooled feature actually came from, it found that the
importance of a source is only weakly tied to which level it nominally belongs to: for small proposals
nominally assigned to the finest level, on the order of 70% of the useful pooled features came from
*other* levels, and for large proposals assigned to the coarsest level, more than half came from lower
levels. The practical reading is blunt — when you combine features from several sources, those sources
do not contribute equally to the output, and which one matters can shift case by case. The same logic
applies whenever the sources are heterogeneous for any reason (different resolutions, different input
variables): treating them identically throws away a degree of freedom the data is asking you to use.

A second piece of background is the *cost* ledger of the obvious expressive fix. The expressive way to
let sources interact is attention: project each source into queries/keys/values and let a query decide,
per location, how much to read from each source. This genuinely captures source-dependent, even
location-dependent, mixing — but it adds projection matrices (parameters that scale with the channel
dimension) and an attention computation at every location, which on top of a large backbone is a real
tax in both parameters and FLOPs. There is also the search route — letting an automated architecture
search discover a better fusion *topology* — which can find good connection patterns but costs thousands
of GPU-hours and yields irregular networks that are hard to interpret or edit. So the landscape offers
"free but rigidly equal" at one end and "expressive but expensive (or expensively found)" at the other,
with little in between.

A third constraint is easy to miss because it is not a new layer type: the fusion node sits in the
middle of a trained network whose following blocks expect features in a sane numerical range. Plain
addition is predictable, and a mean is scale-stable, but arbitrary learned gains inside the fusion
node could turn it into an uncontrolled amplifier. Any useful middle ground has to expose relative
source importance without making the downstream network absorb a drifting feature scale.

## Baselines

These are the prior fusion mechanisms a new aggregation rule would be measured against and would react
to.

**FPN — Feature Pyramid Networks (Lin, Dollár, Girshick, He, Hariharan & Belongie, CVPR 2017).** Build a
pyramid with a top-down pathway and lateral connections. Each output level merges a lateral feature (the
same-resolution backbone map, passed through a `1x1` conv to fix channel count) with a coarser,
semantically stronger map upsampled from the level above. The merge is *element-wise addition*:

```
P_out_l = Conv( P_in_l + Resize(P_out_{l+1}) )
```

Simple, cheap, end-to-end trainable, and a large accuracy gain over single-scale prediction. **Gap:**
the merge gives both addends weight one — every source counts the same, by fiat. Nothing in the model
can express "this resolution matters more here than that one," even though the inputs being summed are at
different resolutions and different semantic levels. The information flow is also one-way (top-down
only).

**PANet — Path Aggregation Network (Liu, Qi, Qin, Shi & Jia, CVPR 2018).** Add a *bottom-up* path on top
of FPN so strong low-level localization signals can also propagate upward, and pool each proposal's
features from *all* levels (adaptive feature pooling), fusing the per-level grids by an element-wise
operation — max or sum. This is the work that produced the diagnostic above: features from multiple
levels are jointly helpful, and the level a feature "belongs to" predicts its usefulness only weakly.
**Gap:** having shown that sources contribute unequally, the actual combine step is still a *fixed*
equal-treatment operator (max or sum). It surfaces the phenomenon but does not give the network a handle
to act on it — there is no per-source quantity it can learn.

**NAS-FPN (Ghiasi, Lin, Pang & Le, CVPR 2019).** Use neural architecture search to discover the *wiring*
of the fusion network — which levels connect to which — then stack the discovered block repeatedly.
Improves accuracy by finding better cross-scale *connections*. **Gap:** the search decides *which* inputs
a node receives, but each node still merges its inputs by an equal-weight combine (a sum); it optimizes
topology, not the relative contribution of the inputs at a node. And it is enormously expensive —
thousands of GPU-hours — and yields irregular, hard-to-interpret networks.

**Pyramid attention upsampling (Li, Xiong, An & Wang, BMVC 2018).** Introduce global self-attention into
the upsampling path to recover localization. This is the attention route to letting sources interact;
**gap:** it brings the attention cost (extra projections and a global attention computation) into the
fusion step, which is exactly the tax a lightweight aggregator is trying to avoid.

**Equal-weight averaging / cross-attention aggregation (the two ends, in the target setting).** In a
foundation model that tokenizes each input variable independently and must aggregate `V` variable tokens
per spatial location, the two natural baselines are the same two ends of the landscape: a *uniform mean*
over the `V` sources (parameter-free, but it is precisely the equal-weight default that the diagnostic
above argues against), and a *cross-attention* aggregator (a learnable query attends, via multi-head
attention with query/key/value projections, over the `V` source tokens at each location — maximally
expressive, source- and location-dependent, but it carries the projection parameters and the per-location
attention computation). **Gaps:** the mean cannot express unequal contribution at all; the cross-attention
pays full attention cost to do a job — deciding how much each source contributes — that may need far less
machinery.

## Evaluation settings

The natural yardsticks already in use for the fusion question, stated as settings only.

- **COCO object detection** (Lin et al. 2014; 118K training images) with one-stage detectors — the arena
  where FPN/PANet/NAS-FPN fusion variants are compared. Metric: COCO Average Precision (AP). Training:
  SGD with momentum, cosine learning-rate schedule, standard scale-jitter and flip augmentation.
- The fusion module is swapped *in place* inside a fixed detector: same backbone, same prediction heads,
  same training recipe, only the cross-scale feature-combining block changes — isolating the effect of
  the fusion rule.
- **Weather forecasting from reanalysis** as the multi-source-aggregation testbed: a transformer backbone
  that tokenizes each of `V` meteorological variables independently is fine-tuned from pretrained weights
  on ERA5 reanalysis at 5.625-degree resolution. Forecasting targets at several lead times: geopotential
  at 500 hPa (3-day), temperature at 850 hPa (5-day), 10 m wind speed (7-day). Metric: latitude-weighted
  RMSE, which weights each grid cell by the cosine of its latitude to account for the convergence of
  meridians toward the poles. The aggregation module is swapped in place; backbone, tokenization,
  optimizer/schedule, data pipeline, and metric are all held fixed.

## Code framework

The aggregator is a single module dropped into a fixed pipeline: an upstream stage produces `V` aligned
feature maps per spatial location (already brought to a common shape), and the module must return one
fused map per location for the downstream network. Everything around it — the per-source tokenizer, the
backbone, the loss, the optimizer and schedule — already exists and is held fixed. Standard layers are
available (`nn.Linear`, `nn.LayerNorm`, `nn.MultiheadAttention`, `nn.Parameter`, and the functional ops
in `torch.nn.functional`). The one empty slot is the combining rule itself: how the `V` source maps are
collapsed into one.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableAggregator(nn.Module):
    """Collapse V aligned per-source feature maps into one map per location.

    Input  x: [B, V, L, D]  -- V source feature maps, B batch, L spatial locations, D channels.
    Output:   [B, L, D]     -- one fused representation per location.

    The module owns whatever (small) state the combining rule needs; the rule itself
    is what we are here to design.
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim      # D
        self.num_heads = num_heads      # available if the rule wants attention
        self.num_vars = num_vars        # V: number of sources being combined
        # TODO: any parameters the combining rule needs.
        pass

    def forward(self, x):
        # x: [B, V, L, D]
        # TODO: combine the V source maps into one per location -> [B, L, D].
        pass


# fixed surrounding pipeline (already exists, not designed here)
def run(model, aggregator, batch):
    per_source = model.tokenize(batch)     # -> x: [B, V, L, D], one token stream per source
    fused = aggregator(per_source)         # -> [B, L, D]   (the slot above)
    return model.backbone(fused)           # downstream network consumes the fused representation
```

The combining rule is the only thing left to fill in; the surrounding tokenizer, backbone, and loss are
fixed.
