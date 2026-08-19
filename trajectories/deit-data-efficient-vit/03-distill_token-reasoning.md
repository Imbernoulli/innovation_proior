Hard-label distillation beat soft distillation cleanly and by the same margin at every size: +2.1 on Ti,
+1.1 on S, +1.2 on B at 224², +0.9 on B at 384², with no extra hyperparameters and the identical
architecture and recipe underneath. That settles the loss question for now — hard-label pseudo-labels,
recomputed on the same augmented crop the student sees, are the objective going forward. But the way I
implemented that objective was, on reflection, architecturally lazy: I took one token, the class token,
whose single linear classifier already had a job — predict the true label — and asked the same embedding
to simultaneously satisfy a second, only loosely correlated target, the teacher's hard decision. The
teacher is not omniscient; at 82.9% top-1 it disagrees with the ground truth on roughly one image in six,
and on exactly the images where it disagrees, the class token's shared representation is being pulled by
gradients toward two different answers for the same image, out of the same 768-dimensional vector, read
by the same linear head. That is not obviously the most a single distillation signal can buy. If the
representation feeding the classifier had to compromise between two targets that sometimes conflict, some
of the available signal from each target may simply have been squeezed out fighting the other for the
same output vector.

There is a structural precedent already sitting inside the architecture for how to avoid exactly this
kind of interference: the class token itself. Nothing about the class token is architecturally special —
it is just one more entry in the sequence that passes through the same self-attention layers as every
patch token, distinguished from a patch token only by two facts: it has no fixed spatial content (it is a
free parameter, not a projected image patch), and a classifier is trained to read a specific target off
its final-layer output. That is the entire mechanism by which the class token comes to represent
"whatever the classification objective needs" rather than "the content of some patch." If that mechanism
is what lets one token specialize toward one target purely through attention interactions with the shared
patch sequence, then giving a *second* token of exactly the same kind — a free parameter, injected
alongside the class token before the first block, that attends to and is attended by every patch and every
other special token through the ordinary self-attention computation — but training its own classifier only
against the teacher's hard label, should let the two tokens diverge to occupy different roles without
either one carrying the other's conflicting objective. The class token keeps predicting the true label,
undisturbed by the teacher signal; the new distillation token exists to carry exactly the pseudo-label
signal that was, in the previous rung, sharing space it didn't have to share.

Before committing real training runs to this idea, I want to rule out the boring alternative explanation
in advance rather than discover it only after the fact: maybe any gain from a second token would come
merely from the extra parameters and the extra attention-slot capacity a second free token adds, regardless
of what its classifier is trained on. If that were true, a second token trained on the *same* target as
the class token — a duplicate class token, initialized independently but pointed at the identical true
label — should do just as well. I want that comparison run alongside the real proposal, both under
identical conditions (hard-label objective, same teacher, same recipe, same three sizes), specifically so
that if the two-token idea does show a gain, I can attribute it to the distinct target rather than to bare
capacity. I do not know yet whether the duplicate-target control will behave differently from the
distinct-target design or converge to functionally the same thing — that is exactly the question the
control is there to answer, and I am tracking one diagnostic beyond raw accuracy to help interpret
whichever way it goes: the cosine similarity between the two tokens' learned embeddings, both near the
input (right after the tokens are formed) and at the final layer (right before each classifier reads its
own token). If a second token adds nothing beyond capacity, I expect the two tokens' representations to
converge toward each other regardless of what they were told to predict, since a duplicate free parameter
with no distinguishing signal has no incentive to diverge; if the distinct-target design is doing real
work, I expect measurably lower similarity between the class and distillation tokens than between two
tokens sharing an identical target, since only the distinct-target pair has any incentive from the loss to
represent something different from each other.

The last design choice is how to read a prediction back out once there are two token-specific classifiers
instead of one. Three options are immediately available and none requires new machinery: use only the
class token's classifier (ignoring the distillation token's output entirely at test time, treating it as
purely a training-time auxiliary that shaped the shared representation through attention but contributes
nothing to the final prediction), use only the distillation token's classifier (the mirror case), or
combine both — my working choice for the primary readout is to sum the two classifiers' softmax outputs, a
simple late fusion that costs nothing extra to compute and, if the two tokens really have specialized on
complementary rather than redundant information, should let each contribute whatever the other is
missing. I am reporting all three at every size rather than committing in advance to one, since I do not
yet know whether fusion beats either token alone, whether one token alone already captures most of the
combined benefit, or whether one of the two single-token readouts actually outperforms the fusion for some
size. That is the full test for this rung: two-token architecture, hard-label loss carried over unchanged
from the previous rung, class token trained on the true label, distillation token trained on the teacher's
hard pseudo-label, evaluated three ways at readout time, run against the duplicate-target control to
separate a genuinely useful distinct signal from mere added capacity.
