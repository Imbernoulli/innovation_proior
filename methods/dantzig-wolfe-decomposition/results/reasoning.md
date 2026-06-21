我要抓住 Dantzig-Wolfe decomposition 的真正动作，而不是只把它描述成“把大问题拆小”。单纯拆小并不新奇；真正特别的是变量的单位变了。原始 LP 的变量是 `x_k` 的坐标，master 要直接决定每个坐标取多少。Dantzig-Wolfe 说：先承认每个块内部的可行域 `X_k` 已经定义了一整套局部可行结构，不要在 master 里重新暴露这些局部坐标。把 `X_k` 的极点当成可选方案，master 只决定这些方案的凸组合权重。

设原问题是 `min sum_k c_k^T x_k`，约束为 `sum_k A_k x_k = b` 且 `x_k in X_k`。如果 `X_k` 是有界多面体，它的任一点都能写作极点凸组合 `x_k = sum_r lambda_kr p_kr`，并且 `sum_r lambda_kr = 1, lambda_kr >= 0`。代回去之后，原始变量 `x_k` 消失，master 变量变成 `lambda_kr`。一列对应的不是“某个坐标增加 1”，而是“采用子问题第 r 个完整极点方案的一点权重”。这使局部可行性自动内嵌在列里：只要列来自 `X_k`，它就不会违反该块内部约束。

这个 reformulation 看上去甚至更糟，因为极点可能指数多个。这里才出现 column generation 的必要性。simplex 或 LP 对偶性告诉我，求解 restricted master 后，我不需要知道所有没列出的极点；我只需要知道是否存在一个极点的 reduced cost 为负。对第 `k` 个块，master 给出 coupling dual `pi` 和该块 convexity constraint 的 dual `mu_k`。任一极点 `p` 的 reduced cost 是 `c_k^T p - pi^T A_k p - mu_k`，也就是 `(c_k - A_k^T pi)^T p - mu_k`。因此在所有极点上找最负 reduced cost，等价于解子问题 `min_{x in X_k} (c_k - A_k^T pi)^T x`。线性目标在多面体上的最优解自然落在极点上，所以 pricing oracle 会自动返回一列。

这一步说明了为什么“极点太多”不是致命问题。方法没有枚举极点，也没有扫描潜在列；它用一个优化问题构造当前对偶价格下最有价值的列。如果这个子问题最优值仍不能让 reduced cost 为负，那么该块所有极点都不能改进 master。所有块都不能改进时，restricted master 的解已经对完整 master 最优，也就是对原 LP 最优。

因此，Dantzig-Wolfe 的思想转变可以说成：从直接处理变量全集，转向动态生成结构性变量。原始变量全集是静态、巨大、没有经济筛选的；结构性变量则由 master 的对偶价格驱动，只有在它能改善全局目标时才被生成。主问题负责全局资源和凸组合，子问题负责局部可行性和最优响应。信息流也很干净：master 传价格，subproblem 传列。

这个解释还区分了它和普通 decomposition。普通分解可能只是并行解几个子模型再拼接；Dantzig-Wolfe 的 master 是一个精确的凸组合模型，pricing subproblem 是完整 LP 的 reduced-cost separation。停止条件不是经验性的“没有明显更好方案”，而是对偶证书：所有隐含列的 reduced cost 都非负。

我还要避免把它误写成 Benders。Benders 是固定 master 变量，让子问题生成 violated cuts，逐步收紧 master 的可行域；Dantzig-Wolfe 是把子问题可行域的极点变成 columns，逐步扩展 master 的变量空间。一个偏 row generation，一个偏 column generation。Dantzig-Wolfe 的“列”本质上是局部可行方案，所以它特别适合 cutting stock 这种 pattern model、crew scheduling 这种 duty model、vehicle routing 这种 route model：一个列就是一个 pattern、一个 duty、一个 route。

最后要承认边界条件。若 `X_k` 无界，严格的表示还需要极射线而不只是极点；若原问题有整数结构，LP 层面的 Dantzig-Wolfe relaxation 往往给强下界，但整数最优通常需要 branch-and-price。可是这些扩展不改变核心洞察：把局部结构封装进列，把全局协调交给 master，把列的发现交给由对偶价格驱动的 pricing problem。
