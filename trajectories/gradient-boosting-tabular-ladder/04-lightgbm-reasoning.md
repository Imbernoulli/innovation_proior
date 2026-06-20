The histogram got me to 165.575 seconds per iteration on Higgs, and the cost is now cleanly localized: at
every node, for every feature, I build a histogram by summing the gradient and Hessian of *every*
example into its bin. So one iteration costs (number of data points) × (number of features) histogram
updates, give or take the depth. That product is the whole bill now. The two factors are independent
levers — I can try to use fewer data points per histogram, and fewer features — and the question is
whether I can shrink either without giving up the accuracy the histogram bought.

Take the data factor first. The histogram sums *all* n examples, but they are not all equally
informative for finding splits. Recall what each example contributes to the split objective: its
gradient gᵢ (and Hessian). An example whose gradient is near zero is one the current model already fits
well — it sits near the loss minimum, it barely tugs the objective, and the split that's best for the
large-gradient, under-fit examples is essentially unchanged whether I include this well-fit example in
the histogram or not. So the well-fit examples carry little information about where the next split
should go; the under-fit, large-gradient examples carry most of it. The naive move is to just drop the
small-gradient examples and build the histogram from the large-gradient ones only. But that *changes the
data distribution* — the histogram is supposed to estimate sums over the full data, and if I throw away
all the small-gradient examples the gradient sums Gⱼ are biased, the gain estimate is wrong, and
accuracy suffers.

The fix is to keep *some* of the small-gradient examples and *correct for the ones I dropped*. Sort the
examples by the magnitude of their gradient. Keep the top fraction a — the large-gradient examples, the
informative ones — in full. From the *remaining* small-gradient examples, randomly sample a fraction b
of them, and throw the rest away. Now I've under-sampled the small-gradient tail by a factor, so when I
add a kept small-gradient example's gradient into a histogram, I scale it up by the inverse sampling
rate (1−a)/b to compensate — that constant makes the sampled subset an *unbiased* estimate of the full
small-gradient population's contribution to the gradient and Hessian sums. The large-gradient examples
are kept and counted as-is (weight 1); the sampled small-gradient examples are kept and counted with
weight (1−a)/b. The histogram is now built from only (a + b)·n examples instead of n, but its gradient
and Hessian sums are unbiased, so the split gains are nearly unchanged. This is **Gradient-based
One-Side Sampling** — GOSS — and the per-iteration data factor drops from n to (a+b)·n.

Concretely, per bagging block: form the per-example weight |g·h|, find the threshold that picks out the
top-a fraction, keep everyone above it, and for everyone below it keep them with probability b/(1−a),
multiplying the kept ones' g and h by the amplification (1−a)/b:

```cpp
data_size_t top_k   = std::max(1, (data_size_t)(cnt * config_->top_rate));     // keep top-a
data_size_t other_k = (data_size_t)(cnt * config_->other_rate);               // sample fraction b
ArrayArgs<score_t>::ArgMaxAtK(&tmp_gradients, 0, (int)tmp_gradients.size(), top_k - 1);
score_t threshold = tmp_gradients[top_k - 1];
score_t multiply  = (score_t)(cnt - top_k) / other_k;                          // amplification (1-a)/b
for (data_size_t i = 0; i < cnt; ++i) {
    auto cur_idx = start + i;
    score_t grad = 0.0f;
    for (int t = 0; t < num_tree_per_iteration_; ++t) {
        size_t idx = static_cast<size_t>(t) * num_data_ + cur_idx;
        grad += std::fabs(gradients[idx] * hessians[idx]);
    }
    if (grad >= threshold) {                         // large-gradient: keep at weight 1
        buffer[cur_left_cnt++] = cur_idx;
        ++big_weight_cnt;
    } else {                                          // small-gradient: sample with prob b/(1-a)
        data_size_t sampled = cur_left_cnt - big_weight_cnt;
        data_size_t rest_need = other_k - sampled;
        data_size_t rest_all = (cnt - i) - (top_k - big_weight_cnt);
        double prob = rest_need / static_cast<double>(rest_all);
        if (rand.NextFloat() < prob) {
            buffer[cur_left_cnt++] = cur_idx;
            for (int t = 0; t < num_tree_per_iteration_; ++t) {
                size_t idx = static_cast<size_t>(t) * num_data_ + cur_idx;
                gradients[idx] *= multiply;           // ... and amplify to stay unbiased
                hessians[idx]  *= multiply;
            }
        }
    }
}
```

Now the feature factor. The histogram is also built feature-by-feature, and real tabular data has *many*
features — and crucially, many of them are *sparse*: one-hot columns, indicators, mostly-zero counts.
Two sparse features that are almost never nonzero at the same time carry, between them, about as much
information as one feature — at any given example, at most one of them is "on." If I could merge such
mutually-exclusive features into a single histogram, I'd build one histogram instead of two and lose
nothing, because they never collide. The construction: treat features as vertices of a graph, draw an
edge between two features whenever they take nonzero values on the same example (they "conflict"), and
the goal is to partition the features into bundles such that within a bundle features rarely conflict.
That's graph coloring — NP-hard in general — but I don't need optimal bundles; I need good-enough ones
fast, and I can allow a small budget of conflicts per bundle (a few examples where two bundled features
collide is tolerable, it just adds a little noise to that bundle's histogram). So: greedily order the
features and, for each, try to drop it into an existing bundle whose total conflict count would stay
under a small budget; if none fits, start a new bundle.

Within a bundle I need the merged feature's values to keep the original features separable, so I offset
each feature's bin range — feature A uses bins [0, n_A), feature B uses bins [n_A, n_A+n_B), and so on —
so a single value in the bundled feature unambiguously identifies which original feature was nonzero and
its bin. The histogram over the bundle is then built in one pass, and the number of *effective* features
drops from the raw count to the number of bundles. This is **Exclusive Feature Bundling** — EFB — and it
shrinks the feature factor.

```cpp
const data_size_t single_val_max_conflict_cnt = (data_size_t)(total_sample_cnt / 10000);  // conflict budget
for (auto fidx : find_order) {                          // greedily place each feature
    // collect existing groups whose total count still fits, then search a sample of them
    int best_gid = -1, best_conflict_cnt = -1;
    for (auto gid : search_groups) {
        const data_size_t rest_max_cnt =
            single_val_max_conflict_cnt - group_total_data_cnt[gid] + group_used_row_cnt[gid];
        const data_size_t cnt = GetConflictCount(conflict_marks[gid], sample_indices[fidx],
                                                 num_per_col[fidx], rest_max_cnt);
        if (cnt >= 0 && cnt <= rest_max_cnt && cnt <= cur_non_zero_cnt / 2) {  // fits the budget
            best_gid = gid; best_conflict_cnt = cnt; break;
        }
    }
    if (best_gid >= 0) {                                 // add to an existing bundle
        features_in_group[best_gid].push_back(fidx);
        group_total_data_cnt[best_gid] += cur_non_zero_cnt;
        group_used_row_cnt[best_gid] += cur_non_zero_cnt - best_conflict_cnt;
        MarkUsed(&conflict_marks[best_gid], sample_indices[fidx], num_per_col[fidx]);
    } else {                                             // or open a new bundle
        features_in_group.emplace_back();
        features_in_group.back().push_back(fidx);
        conflict_marks.emplace_back(total_sample_cnt, false);
        MarkUsed(&(conflict_marks.back()), sample_indices[fidx], num_per_col[fidx]);
    }
}
```

There's a third lever, in how the tree itself grows. The histogram learners I've been building grow
trees *level-wise*: split every node at the current depth before going deeper. That keeps the tree
balanced but spends splits on nodes that barely reduce the loss, just to keep the level full. With the
gains now computed cheaply from histograms, I can instead grow *leaf-wise*: keep a priority queue of all
current leaves by their best achievable split gain, and always split the single leaf with the highest
gain, wherever it is in the tree. For a fixed budget of leaves this descends the loss faster, because
every split is the globally most valuable one available rather than the next one in a level-order sweep.
The risk is deeper, lopsided trees that overfit, which I cap by limiting the number of leaves (and
optionally the depth). Controlling complexity by *number of leaves* rather than depth is the natural
pairing with leaf-wise growth.

Put the three together: GOSS cuts the data factor by sampling away well-fit examples while staying
unbiased; EFB cuts the feature factor by bundling mutually-exclusive sparse features; leaf-wise growth
spends each split where it helps most. The per-iteration cost was (#data)×(#features) histogram updates;
GOSS shrinks the first factor and EFB the second, so the product drops on both axes at once — and
because GOSS keeps the histogram unbiased and EFB bundles only near-non-colliding features, the split
gains, and therefore the accuracy, should be preserved while the time per iteration falls below the
histogram baseline. On a ranking task with hundreds of features, the EFB feature-factor reduction and
the leaf-wise growth should also let the learner find sharper splits within the same budget, pushing the
ranking metric up.

This is **LightGBM**: histogram split finding plus Gradient-based One-Side Sampling, Exclusive Feature
Bundling, and leaf-wise tree growth. Against the histogram baseline's 165.575 s/iter on Higgs, the bet is a
further drop in per-iteration time at matched-or-better AUC, and on the ranking benchmark a clear NDCG
gain. The thing I notice it has *not* touched is the one part of the pipeline that has been the same
since AdaBoost: how a *categorical* feature becomes a number a tree can split on. Higgs and the ranking
sets are numeric. But on a dataset whose features are categorical identifiers with thousands of values,
the standard move is to replace each category with a statistic of the target — and that, done naively,
quietly uses each example's own label to encode its own features. That is the next thing to question.
