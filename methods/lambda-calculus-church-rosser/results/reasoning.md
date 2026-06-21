容易误解 Church-Rosser theorem 的地方，是把它说成“无论怎么归约都会停在同一个结果”。这比定理真实含义更强，也不正确。lambda calculus 中有些 term 没有 normal form，例如可以无限自我展开的项；某些归约策略也可能明明存在 normal form 却一直走在发散路径上。Church-Rosser theorem 不是终止性定理，而是 confluence 定理。

正确的核心是：如果从同一个 term 出发，沿两条归约序列到达了两个 term，那么这两个 term 仍然可以继续归约到一个共同 term。这个共同后继可能就是其中一个分支，也可能是更远处的第三个 term。也就是说，归约路径可以分叉，但分叉不是语义灾难，因为系统保证了可汇合性。

这解释了 normal form 唯一性的来源。normal form 是不能再继续 beta-reduce 的 term。假设同一个 term 能归约到两个 normal form `A` 和 `B`。由 Church-Rosser/confluence，`A` 和 `B` 应当还能继续归约到共同后继。但它们已经是 normal form，不能再动，所以共同后继只能分别等于 `A` 和 `B`。因此 `A` 与 `B` 必须是同一个 normal form，至多差一个 alpha-renaming 这样的变量名差异。

这里真正有力量的不是“找到了一个好策略”，而是“证明了策略选择不会制造多个互斥答案”。执行过程仍可能很不一样：normal-order reduction 可能找到 normal form，applicative-order reduction 可能先陷入某个发散子项；不同路径的长度也可以差很多。Church-Rosser 只保证一旦结果以 normal form 的形式存在，它不是由路径任意性捏造出来的。

用图来理解最清楚。term 是节点，单步 beta-reduction 是边。从某个节点出发，如果有两条边或两条路径走向不同位置，confluence 要求下游存在重新汇合的区域。这就是 diamond property 的思想。严格证明中，原始 beta-reduction 的单步关系未必直接满足简单 diamond；常见证明会引入 parallel reduction 或 development 技巧，先证明一个更规整关系具有 diamond，再转回 beta-reduction 的多步 confluence。

这个转化本身就是 Church-Rosser 思想的关键。它不在每一次局部冲突上手工决定谁优先，而是给整个归约关系建立一个几何不变量：所有可达分支都可 join。局部的 redex 选择被放进全局图形里处理，具体执行顺序被降格为路径选择，而不是语义定义本身。

因此，Church-Rosser theorem 对计算观的贡献，是把“程序怎么运行”与“表达式等于什么”分开。程序运行是策略性的、顺序性的、可能非终止的；表达式的 beta-equivalence 和 normal form 唯一性则由重写系统的 confluence 支撑。只要 normal form 存在，它就是这个归约空间中的唯一汇合终点，而不是某个解释器调度规则的副产品。

这也是为什么它在 lambda calculus 中格外重要。lambda calculus 的每个 computation step 都是局部代换，局部代换之间容易重叠、嵌套、互相改变形状。Church-Rosser theorem 说明这种局部复杂性并不会导致全局语义碎裂。它把一个看似混乱的执行图，组织成一个有可汇合结构的空间。

从更一般的重写系统角度看，Church-Rosser theorem 的启发是：研究计算不一定从机器状态和时间顺序出发，也可以从重写关系诱导的空间结构出发。confluence、normalization、joinability 这些概念描述的是整个系统的形状。这个视角后来影响了项重写、证明规范化、函数式语言语义和程序变换正确性。

所以，如果只说“归约顺序不影响结果”，还不够精准。更完整的说法是：Church-Rosser theorem 通过 confluence/diamond property 控制 lambda calculus 中归约路径的分叉，使得 normal form 若存在就具有策略无关的唯一性；它把计算从逐步执行的线性故事，提升为对重写系统全局归约图的几何理解。
