**Problem (from step 3).** XGBoost's histogram cut per-iteration time to 165.575 s on Higgs, but each
iteration still sums gradients/Hessians over **all** data and **all** features — cost stays proportional
to (#data) × (#features). Both factors are independent levers; neither has been touched.

**Key idea.** **LightGBM**: shrink both factors while keeping the histogram unbiased. (1)
**Gradient-based One-Side Sampling (GOSS)** cuts the *data* factor: examples with near-zero gradient are
already well-fit and carry little split information, so keep the top fraction a of large-gradient
examples in full, randomly sample a fraction b of the rest, and **amplify** the sampled small-gradient
examples' g and h by (1−a)/b so the gradient/Hessian sums stay unbiased — the histogram uses only (a+b)n
examples. (2) **Exclusive Feature Bundling (EFB)** cuts the *feature* factor: sparse features that are
rarely nonzero on the same example never collide, so bundle near-mutually-exclusive features (a
graph-coloring with a small per-bundle conflict budget) into one histogram, offsetting each feature's
bin range so values stay separable — #effective-features drops to #bundles. (3) **Leaf-wise growth**:
always split the leaf with the highest gain (priority queue) rather than level-by-level, capping
complexity by #leaves; for a fixed leaf budget this descends the loss faster.

**Why it works.** GOSS removes the redundant well-fit examples but the (1−a)/b amplification keeps the
sums unbiased, so split gains — and accuracy — are preserved while the data factor falls to (a+b)n. EFB
bundles only features that almost never collide (bounded conflict budget), so the merged histogram loses
almost no information while the feature factor falls to #bundles. The two cuts multiply, dropping
per-iteration cost on both axes; leaf-wise growth spends each split where it helps most, sharpening
accuracy within the leaf budget. Its untouched part: how a *categorical* feature is turned into a number
a tree can split on — replacing a category with a target statistic, done naively, leaks each example's
own label.

**Change / code.** GOSS sampling-and-amplification (`src/boosting/goss.hpp`) and the EFB bundle-finder
(`src/io/dataset.cpp`), real LightGBM source:

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

GOSS skips sub-sampling for the first ⌈1/learning_rate⌉ iterations (so the early model is built on all
data); after that the histogram is built from the (a+b)n kept examples with the amplified gradients.
Leaf-wise growth and complexity control by `num_leaves` are configured around this loop.
