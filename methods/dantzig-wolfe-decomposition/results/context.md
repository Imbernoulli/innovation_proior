## Research question

大规模线性规划常常不是“变量很多”这么简单，而是变量分成若干自然块，每个块内部有一套复杂但局部的可行性约束，块之间只通过少量资源、需求或平衡约束耦合。典型形式是

`min sum_k c_k^T x_k`

subject to

`sum_k A_k x_k = b`, and `x_k in X_k`.

这里 `X_k = {x_k: B_k x_k <= d_k, x_k >= 0}` 是第 `k` 个子系统自己的可行域，`sum_k A_k x_k = b` 是全局耦合约束。直接把所有 `x_k` 放进一个 LP 可以写出正确模型，但求解器会同时面对所有块的内部结构、所有变量和所有耦合关系。问题是：能否只让主问题处理全局协调，而把每个块的内部可行性留给子问题？

Dantzig-Wolfe decomposition 的目标就是把这种 LP 改写成一个等价的 master problem，使 master 的变量不再是原始坐标 `x_k`，而是“选择第 `k` 个子问题的某个极点方案”的权重。这样全局 LP 从直接操纵每个原始变量，转为组合少量结构性方案。

## Background

核心几何事实是凸多面体的极点表示：如果 `X_k` 是非空有界多面体，则每个 `x_k in X_k` 都能写成其极点 `p_kr` 的凸组合：

`x_k = sum_r lambda_kr p_kr`, `sum_r lambda_kr = 1`, `lambda_kr >= 0`.

把这个表示代回原 LP，得到 Dantzig-Wolfe master：

`min sum_k sum_r (c_k^T p_kr) lambda_kr`

subject to

`sum_k sum_r (A_k p_kr) lambda_kr = b`,

`sum_r lambda_kr = 1` for every block `k`,

`lambda_kr >= 0`.

每一列不再代表一个单个原始变量，而代表一个子问题极点 `p_kr`：它自带局部可行性、局部成本 `c_k^T p_kr`，以及对全局耦合约束的贡献 `A_k p_kr`。这就是方法的独特洞察：把巨大 LP 的局部可行域压缩成可组合的“完整局部方案”，然后让 master 在这些方案之间做凸组合。

实际困难是极点可能指数多个，master 变量数可能比原问题还大。因此 Dantzig-Wolfe 必须和 column generation 配套：先只放少量极点列，解 restricted master problem；再用 master 的对偶价格去问每个子问题“有没有更值钱的极点列应该加入”。

## Baselines

直接求解原始 LP：把所有原始变量和约束一次性交给 simplex 或 interior-point solver。优点是模型直观，缺点是求解器必须同时处理局部块结构和全局耦合结构；当每个块内部有大量变量或复杂组合结构时，稀疏性和可分性没有被充分转化为算法优势。

显式完整 master：先枚举每个 `X_k` 的所有极点，再求解 Dantzig-Wolfe master。优点是 LP 等价且理论清楚；缺点是极点集合通常指数大，列无法枚举，矩阵无法形成。

普通分块启发式：独立求各块局部最优，再用某种修补步骤满足全局耦合约束。优点是计算便宜；缺点是没有 LP 对偶证书，也容易因为忽略全局价格而生成局部好、全局差的方案。

Benders decomposition 是相邻但不同的分解思维：它通常固定一部分变量，在子问题中生成 cuts 回到 master，属于 row generation；Dantzig-Wolfe 则把子问题可行解变成 columns 加入 master，属于 column generation。这里的重点不是删约束，而是动态生成结构性变量。

## Evaluation settings

适合 Dantzig-Wolfe 的实例应有 block-angular structure：多个相对独立的子块 `X_k`，再由少量 linking constraints 连接。常见例子包括 cutting stock、vehicle routing set partitioning、crew scheduling、多商品流路径模型、生产计划和大型资源分配模型。

关键评价指标不是“最终 LP 是否等价”这一点，因为完整 master 理论上等价；真正关心的是：求解过程中实际生成了多少列、pricing subproblem 是否比直接扫描所有列便宜、restricted master 的对偶界收敛多快、以及生成列后的 LP bound 是否强。若原问题含整数变量，通常还要评估 branch-and-price 中节点数、整数 gap 和 pricing 难度。

一个小型演示可以设置两个子问题块，每个块有自己的局部多面体 `X_1, X_2`，并用一条共享资源约束 `A_1 x_1 + A_2 x_2 = b` 耦合。初始 master 只含每个块的几个显然可行极点；每轮读取 coupling dual `pi` 和 convexity dual `mu_k`，对子问题求

`min_{x in X_k} (c_k - A_k^T pi)^T x`.

如果某块最优值减去 `mu_k` 为负，就把该极点作为新列加入 master；否则该块没有改进列。

## Code framework

现有实现只需要三个部件。第一，restricted master LP：输入当前列池，每列包含 `(block_id, cost, linking_vector, point)`，输出 primal 解和对偶价格。第二，pricing oracle：给定每个块的 reduced objective `c_k - A_k^T pi`，在 `X_k` 上求线性最优化，返回最小 reduced cost 的极点。第三，column-generation loop：反复解 master、调用 pricing、添加负 reduced cost 列，直到所有块都无改进列。

```python
def solve_restricted_master(columns, b):
    """Return master solution plus duals pi and mu_k."""
    raise NotImplementedError

def price_block(block, pi, mu_k):
    """Solve min (c_k - A_k.T @ pi)^T x over X_k and return a new column if reduced cost < 0."""
    raise NotImplementedError

def dantzig_wolfe(blocks, b, initial_columns, eps=1e-8):
    columns = list(initial_columns)
    while True:
        master = solve_restricted_master(columns, b)
        new_columns = []
        for block in blocks:
            col = price_block(block, master.pi, master.mu[block.id])
            if col.reduced_cost < -eps:
                new_columns.append(col)
        if not new_columns:
            return master, columns
        columns.extend(new_columns)
```

如果 `X_k` 无界，完整理论还要处理极射线列；本报告聚焦最常见、也最容易说明核心洞察的有界子问题情形。
