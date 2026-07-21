I start with the uncomfortable observation. I can train with the usual direct preference loss, watch the chosen-minus-rejected margin grow, and still see the preferred completion's log-probability fall. That is not a separate implementation bug. The objective is a function of one scalar, the reference-relative gap, and a gap can be improved by pushing the loser down faster than the winner falls.

Pin down the signs. For a preference triple `(x, y_w, y_l)`, define `rho(y) = log pi_theta(y|x) - log pi_ref(y|x)`. The standard direct preference loss is `-log sigma(beta * (rho(y_w) - rho(y_l)))`. If I write `z = beta * (rho_w - rho_l)`, then `d[-log sigma(z)]/dz = -sigma(-z)`, so gradient descent moves in the direction that increases `rho_w - rho_l`. That means it pushes up `log pi_theta(y_w|x)` and pushes down `log pi_theta(y_l|x)`, weighted by the same self-paced factor. So far this is exactly the usual story.

The problem is that "push up the chosen sequence and push down the rejected sequence" is not separable when the two sequences share almost all of their tokens. In the low-edit-distance case, the two completions have the same prefix up to some position `m`, then one token differs, then most of the later text may again be shared as text but conditioned on different prefixes. For positions before `m`, the preferred and rejected terms are literally the same and cancel. After `m`, the update depends on how the next-token distributions under the two prefixes differ.

The one-token calculation makes the failure concrete. Suppose the two completions first differ at `m = 1`, and consider a later token `t_k` with vocabulary index `i`. The softmax identity is `d log s_i / d theta_j = 1{i=j} - s_j`. For the DPO log-ratio term, after dropping the leading negative loss sign and looking at the gradient-descent update direction, I get

`d[log pi_theta(t_k|y_w^{<k},x) - log pi_theta(t_k|y_l^{<k},x)] / d theta_j = s_j^{y_l^{<k},x} - s_j^{y_w^{<k},x}`.

If the starting model is already reasonably good, then for the correct token index `i`, the preferred-prefix probability is at least the dispreferred-prefix probability: `s_i^w >= s_i^l`. The update direction on the correct logit is then non-positive, so the correct later-token logit can go down. For other token indices, the assumed inequality reverses, so their logits can go up. This is the exact "wrong-way after the edit" case. It is not that the first differing token is impossible to fix; it is that the contrastive update can make the continuation after that edit worse for the preferred trace.

Now I know what kind of fix is needed. It should not replace the preference gap, because the gap is the useful contrast. It should add pressure only when the preferred completion has fallen below the reference. If I add an unconditional `-rho_w` term to the loss, I turn the objective toward preferred-only SFT and keep pushing even when the policy already makes the preferred completion more likely than the reference. That loses the one-sided character I need. The right shape is a soft floor: zero when `rho_w >= 0`, positive when `rho_w < 0`.

The one-sided term is `max(0, -rho_w)`, equivalently `max(0, log pi_ref(y_w|x) - log pi_theta(y_w|x))`. It is measured in the same sequence-log-prob units as the DPO logit. The important placement is inside the DPO logit, not as a second additive loss outside the `logsigmoid`. So the logit becomes

`rho_w - rho_l - lambda * max(0, -rho_w)`,

and the loss is

`L = -E log sigma(beta * (rho_w - rho_l - lambda * max(0, -rho_w)))`.

This preserves the exact DPO form when `lambda = 0`. It also preserves the exact DPO update on any example where `rho_w >= 0`, because the hinge and its derivative are both zero there. Below the floor, the logit becomes `(1 + lambda) * rho_w - rho_l`, so the preferred sequence receives the ordinary DPO preferred update plus an extra `lambda` times the preferred log-probability update. This is the sign check that matters: the penalty is subtracted from the logit, the outer loss wants the logit larger, and therefore minimizing the loss pushes the positive penalty term down by increasing `log pi_theta(y_w|x)` relative to `log pi_ref(y_w|x)`.

Re-running the one-token case with the hinge active: when `rho_w < 0`, the later-token update direction for token index `i` becomes

`lambda * (1 - s_i^w) + s_i^l - s_i^w`

for the correct token's own logit, and

`-(lambda + 1) * s_j^w + s_j^l`

for `j != i`. Since `s_i^w <= 1`, a large enough `lambda` makes the correct-token direction positive. Since `s_j^w > 0` for softmax probabilities, a large enough `lambda` makes the non-token directions negative. This is the case split I need: below the preferred-likelihood floor, the update can reverse the wrong-way continuation behavior; at or above the floor, it is ordinary DPO.

The hyperparameter interpretation follows from the formula. `beta` scales the whole logit, and `lambda` scales the one-sided preferred-likelihood term before that beta multiplication. The experiments use `beta = 0.3` and `lambda = 50` as the default, and sweep `beta` over `{0.1, 0.3, 1.0}` and `lambda` over `{5, 50, 500}`. I should not describe `lambda` as a hard guarantee that `rho_w` can never be negative. It is a soft restoring pressure. A too-small value makes the objective nearly DPO; a too-large value can dominate the contrast and look more like preferred-only likelihood training.

Now I translate this into the trainer's tensors. The trainer gives me summed completion log-probabilities: `policy_chosen_logps`, `policy_rejected_logps`, `reference_chosen_logps`, and `reference_rejected_logps`. They have to be summed, not length-averaged: `rho(y)` is a difference of sequence log-probabilities feeding a Bradley-Terry gap, and dividing by length would rescale `rho_w` and `rho_l` unequally whenever the two completions differ in token count, which corrupts the very gap the loss is built on. The standard DPO score before beta is

`(policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)`.

The preferred reference-relative log-ratio is `policy_chosen_logps - reference_chosen_logps`, so the active penalty is `relu(reference_chosen_logps - policy_chosen_logps)`. I subtract `lambda_dpop * penalty` from the DPO score, multiply by `beta`, and use the same loss the trainer already had: `-logsigmoid(beta * logits)` when label smoothing is off, or that term mixed with a `-logsigmoid(-beta * logits)` flipped-sign term weighted by the smoothing rate when it is on — neither branch touches the penalty, since the penalty is folded into `logits` before this step. The chosen/rejected reward diagnostics stay the ordinary detached `beta * rho` values used for logging — they are not part of the loss I am changing. That is the entire change to the code: one `clamp` and one subtraction ahead of the existing DPO logit, before `beta` and `logsigmoid` are applied — the full function is in the answer.
