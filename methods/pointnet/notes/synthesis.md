# PointNet Synthesis

This legacy synthesis note has been superseded by `notes/discovery_synthesis.md` and `notes/source_matrix.md`.

Use the newer files for the strict reconstruction. They correct the previous draft on four points:

- The universal approximation proof should not claim exact binary occupancy from the continuous soft-indicator sketch.
- The official T-Net initializes the final affine layer with zero weights and an identity bias; it does not add identity after a randomly initialized final layer.
- The official classifier applies keep-probability `0.7` dropout after both hidden fully connected layers.
- The released TensorFlow regularizer uses `tf.nn.l2_loss(A A^T - I)`, so the code constant is one half of the summed squared entries before applying weight `0.001`.
