The histogram cut Higgs to $165.575$ seconds per iteration, and the cost is now cleanly localized: at every node, for every feature, I build a histogram by summing the gradient and Hessian of *every* example into its bin. So one iteration costs (number of data points) $\times$ (number of features) histogram updates, give or take the depth. That product is the whole bill, and its two factors are independent levers — fewer data points per histogram, fewer features — so the question is whether I can shrink either without giving up the accuracy the histogram bought.

I propose **LightGBM**, which attacks both factors while keeping the histogram unbiased, plus a change in how the tree grows. The first piece, for the *data* factor, is **Gradient-based One-Side Sampling (GOSS)**. Recall what each example contributes to the split objective: its gradient $g_i$ (and Hessian). An example whose gradient is near zero is one the current model already fits well — it sits near the loss minimum, barely tugs the objective, and the best split for the large-gradient, under-fit examples is essentially unchanged whether I include it or not. So the well-fit examples carry little split information; the large-gradient examples carry most of it. The naive move — drop the small-gradient examples and build the histogram from the large ones only — *changes the data distribution*: the gradient sums $G_j$ become biased, the gain estimate is wrong, accuracy suffers. The fix is to keep *some* small-gradient examples and *correct for the ones I dropped*. Sort the examples by gradient magnitude, keep the top fraction $a$ in full, randomly sample a fraction $b$ of the rest and discard the others; then, because I have under-sampled the small-gradient tail, I scale each kept small-gradient example's $g$ and $h$ up by the inverse sampling rate

$$\frac{1-a}{b},$$

which makes the sampled subset an *unbiased* estimate of the full small-gradient population's contribution to the gradient and Hessian sums. Large-gradient examples are counted at weight $1$; sampled small-gradient examples at weight $(1-a)/b$. The histogram is now built from only $(a+b)n$ examples, but its sums are unbiased, so the split gains — and the accuracy — are nearly unchanged, and the data factor drops from $n$ to $(a+b)n$.

The second piece, for the *feature* factor, is **Exclusive Feature Bundling (EFB)**. Real tabular data has many features, and many are *sparse*: one-hot columns, indicators, mostly-zero counts. Two sparse features that are almost never nonzero on the same example carry, between them, about as much information as one — at any given example at most one is "on" — so if I merge such mutually-exclusive features into a single histogram, I build one histogram instead of two and lose nothing, because they never collide. The construction treats features as graph vertices, draws an edge between two features whenever they take nonzero values on the same example (they "conflict"), and partitions features into bundles where conflicts are rare. That is graph coloring — NP-hard — but I do not need optimal bundles; I allow a small budget of conflicts per bundle (a few collisions just add a little noise to that bundle's histogram) and place features greedily: order them, and for each try to drop it into an existing bundle whose total conflict count would stay under the budget, else start a new bundle. Within a bundle I keep the original features separable by offsetting each feature's bin range — feature $A$ uses bins $[0, n_A)$, feature $B$ uses $[n_A, n_A+n_B)$, and so on — so a single value in the bundled feature unambiguously identifies which original feature was nonzero and its bin. The histogram over the bundle is built in one pass, and the number of *effective* features drops from the raw count to the number of bundles.

The third piece is in how the tree itself grows. Histogram learners typically grow trees *level-wise* — split every node at the current depth before going deeper — which keeps the tree balanced but spends splits on nodes that barely reduce the loss, just to fill the level. With gains now cheap to read off histograms, I instead grow *leaf-wise*: keep a priority queue of all current leaves by their best achievable split gain and always split the single highest-gain leaf, wherever it is. For a fixed leaf budget this descends the loss faster, because every split is the globally most valuable one available rather than the next in a level-order sweep. The risk is deeper, lopsided, overfitting trees, which I cap by limiting the number of leaves — controlling complexity by *number of leaves* rather than depth is the natural pairing with leaf-wise growth.

The three multiply. The per-iteration cost was (#data) $\times$ (#features) histogram updates; GOSS shrinks the first factor to $(a+b)n$, EFB shrinks the second to #bundles, and because GOSS keeps the histogram unbiased and EFB bundles only near-non-colliding features, the split gains — and the accuracy — are preserved while the time per iteration falls below the histogram baseline. On a ranking task with hundreds of features, the feature-factor reduction and leaf-wise growth should also let the learner find sharper splits within the same budget, pushing the ranking metric up. The one part of the pipeline untouched since AdaBoost is how a *categorical* feature becomes a number a tree can split on; the standard move replaces each category with a statistic of the target, and done naively it quietly uses each example's own label to encode its own features — the next thing to question.

```cpp
// GOSS: keep top-a by |g*h|, sample b/(1-a) of the rest, amplify the kept rest by (1-a)/b
// (LightGBM src/boosting/goss.hpp, GOSSStrategy::Helper)
data_size_t top_k   = std::max(1, static_cast<data_size_t>(cnt * config_->top_rate));
data_size_t other_k = static_cast<data_size_t>(cnt * config_->other_rate);
ArrayArgs<score_t>::ArgMaxAtK(&tmp_gradients, 0, static_cast<int>(tmp_gradients.size()), top_k - 1);
score_t threshold = tmp_gradients[top_k - 1];
score_t multiply  = static_cast<score_t>(cnt - top_k) / other_k;          // (1-a)/b
for (data_size_t i = 0; i < cnt; ++i) {
    auto cur_idx = start + i;
    score_t grad = 0.0f;
    for (int t = 0; t < num_tree_per_iteration_; ++t) {
        size_t idx = static_cast<size_t>(t) * num_data_ + cur_idx;
        grad += std::fabs(gradients[idx] * hessians[idx]);                // |g*h|
    }
    if (grad >= threshold) {                                              // large gradient: keep, weight 1
        buffer[cur_left_cnt++] = cur_idx; ++big_weight_cnt;
    } else {                                                             // small gradient: sample, amplify
        data_size_t rest_need = other_k - (cur_left_cnt - big_weight_cnt);
        data_size_t rest_all  = (cnt - i) - (top_k - big_weight_cnt);
        double prob = rest_need / static_cast<double>(rest_all);
        if (bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() < prob) {
            buffer[cur_left_cnt++] = cur_idx;
            for (int t = 0; t < num_tree_per_iteration_; ++t) {
                size_t idx = static_cast<size_t>(t) * num_data_ + cur_idx;
                gradients[idx] *= multiply;  hessians[idx] *= multiply;   // amplify -> unbiased
            }
        } else {
            buffer[--cur_right_pos] = cur_idx;                            // dropped
        }
    }
}
```

```cpp
// EFB: greedily bundle near-exclusive features under a small conflict budget
// (LightGBM src/io/dataset.cpp, FindGroups)
const data_size_t single_val_max_conflict_cnt =
    static_cast<data_size_t>(total_sample_cnt / 10000);                  // per-bundle conflict budget
for (auto fidx : find_order) {
    int best_gid = -1, best_conflict_cnt = -1;
    for (auto gid : search_groups) {                                     // try existing bundles
        const data_size_t rest_max_cnt =
            single_val_max_conflict_cnt - group_total_data_cnt[gid] + group_used_row_cnt[gid];
        const data_size_t cnt = GetConflictCount(conflict_marks[gid], sample_indices[fidx],
                                                 num_per_col[fidx], rest_max_cnt);
        if (cnt >= 0 && cnt <= rest_max_cnt && cnt <= cur_non_zero_cnt / 2) {  // fits the budget
            best_gid = gid; best_conflict_cnt = cnt; break;
        }
    }
    if (best_gid >= 0) {                                                  // add to existing bundle
        features_in_group[best_gid].push_back(fidx);
        group_total_data_cnt[best_gid] += cur_non_zero_cnt;
        MarkUsed(&conflict_marks[best_gid], sample_indices[fidx], num_per_col[fidx]);
    } else {                                                              // open a new bundle
        features_in_group.emplace_back();
        features_in_group.back().push_back(fidx);
        conflict_marks.emplace_back(total_sample_cnt, false);
        MarkUsed(&(conflict_marks.back()), sample_indices[fidx], num_per_col[fidx]);
    }
}
```
