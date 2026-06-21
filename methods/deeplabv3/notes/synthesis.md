# DeepLabv3 Synthesis

This file is retained as a short index. The strict source-grounded reconstruction notes are in `notes/discovery_synthesis.md`, and the evidence table is in `notes/source_matrix.md`.

Key corrected point from this audit: the paper-faithful and official TensorFlow DeepLabv3 ASPP path is image pooling + `1x1` branch + three atrous `3x3` branches, concatenation, `1x1` projection, dropout keep probability 0.9, then final `1x1` logits. The previous draft's `torchvision`-style extra post-ASPP `3x3` refinement block is not part of the target paper's ASPP formulation.
