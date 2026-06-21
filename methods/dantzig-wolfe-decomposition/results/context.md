## Research question

大规模线性规划常常不是“变量很多”这么简单，而是变量分成若干自然块，每个块内部有一套复杂但局部的可行性约束，块之间只通过少量资源、需求或平衡约束耦合。典型形式是

`min sum_k c_k^T x_k`

subject to

`sum_k A_k x_k = b`, and `x_k in X_k`.

这里 `X_k = {x_k: B_k x_k <= d_k, x_k >= 0}` 是第 `k` 个子系统自己的可行域，`sum_k A_k x_k = b` 是全局耦合约束。把所有 `x_k` 放进一个 LP 可以写出正确模型，求解器会同时面对所有块的内部结构、所有变量和所有耦合关系。研究问题是：对这种 block-angular 结构的 LP，能否让全局协调与每个块的内部可行性在算法上分开处理？

## Background

线性规划的标准求解工具是 simplex 和 interior-point 方法，二者都直接在原始坐标 `x` 上工作，并附带 LP 对偶理论：每个等式约束对应一个对偶价格（shadow price），最优时原始与对偶解满足互补松弛，对偶解构成最优性证书。

一个相关的经典几何事实是凸多面体的极点表示：如果 `X_k` 是非空有界多面体，则每个 `x_k in X_k` 都能写成其极点 `p_kr` 的凸组合：

`x_k = sum_r lambda_kr p_kr`, `sum_r lambda_kr = 1`, `lambda_kr >= 0`.

在一个块内，对线性目标做最优化时，最优值必在某个极点取得，因此在固定的线性目标下，对子多面体 `X_k` 的优化本身就是一个小型 LP，可由 simplex 直接求解。多面体 `X_k` 的极点数目一般可随维数指数增长。

## Baselines

直接求解原始 LP：把所有原始变量和约束一次性交给 simplex 或 interior-point solver。模型直观，求解器同时处理局部块结构和全局耦合结构。

显式枚举法：先枚举每个 `X_k` 的所有极点，用极点权重重写问题再求解。该写法与原 LP 等价，依赖能把极点集合完整列出。

分块启发式：独立求各块局部最优，再用某种修补步骤满足全局耦合约束。计算便宜，按局部目标各自求解后再协调。

Benders decomposition 是相邻的分解思维：它通常固定一部分变量，在子问题中生成 cuts 回到 master，属于 row generation，动态加入的是约束。

## Evaluation settings

适合分解的实例应有 block-angular structure：多个相对独立的子块 `X_k`，再由少量 linking constraints 连接。常见例子包括 cutting stock、vehicle routing set partitioning、crew scheduling、多商品流路径模型、生产计划和大型资源分配模型。

关键评价指标包括：求解过程中处理了多少结构性方案、子层优化是否比直接扫描原始变量便宜、对偶界收敛多快、得到的 LP bound 是否强。若原问题含整数变量，通常还要评估搜索树节点数、整数 gap 和子层难度。

一个小型演示可以设置两个子问题块，每个块有自己的局部多面体 `X_1, X_2`，并用一条共享资源约束 `A_1 x_1 + A_2 x_2 = b` 耦合。块内目标为线性时，对每个 `X_k` 的优化是

`min_{x in X_k} (c_k - A_k^T pi)^T x`,

其中 `pi` 是耦合约束的对偶价格。

## Code framework

现有可用部件有三类。第一，LP 求解器：输入约束与目标，输出 primal 解和对偶价格 `pi`。第二，块内线性优化 oracle：给定每个块的线性目标 `c_k - A_k^T pi`，在 `X_k` 上求最优化，返回极点解。第三，一个把全局 LP 与块内优化串起来的外层循环骨架。

```python
def solve_lp(columns, b):
    """Return an LP solution plus dual prices pi and mu_k."""
    raise NotImplementedError

def optimize_block(block, pi, mu_k):
    """Solve min (c_k - A_k.T @ pi)^T x over X_k and return the extreme-point solution."""
    raise NotImplementedError

def coordinate(blocks, b, initial, eps=1e-8):
    state = list(initial)
    while True:
        sol = solve_lp(state, b)
        updates = []
        for block in blocks:
            r = optimize_block(block, sol.pi, sol.mu[block.id])
            # accept r into state based on its value relative to current duals
            updates.append(r)
        if converged(updates, eps):
            return sol, state
        state.extend(u for u in updates if accept(u))
```

本报告聚焦最常见、也最容易说明核心结构的有界子问题 `X_k` 情形。
