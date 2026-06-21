# NICE Synthesis

This note is superseded by `notes/discovery_synthesis.md`, which was rebuilt from the strict evidence bundle on June 18, 2026.

Key corrections from the earlier draft:

- The canonical released implementation is the first-author Theano/Pylearn2 repository `laurent-dinh/nice`, not the later `fmu2/NICE` PyTorch reimplementation.
- The paper text reports Adam for the experiment description, but the public first-author YAMLs use Pylearn2 `SGD` with custom `RMSPropMomentum`, learning-rate decay, and a momentum adjustor.
- The public YAMLs differ from the paper prose in some dataset hyperparameters; the deliverables now separate mathematical method statements from public-code reproduction details.
- The final result files were regenerated from `notes/discovery_synthesis.md` rather than patched from this older note.
