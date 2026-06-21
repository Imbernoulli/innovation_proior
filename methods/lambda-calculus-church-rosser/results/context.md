## Research question

Church-Rosser theorem 要回答的问题不是 lambda calculus 中某个 redex 应该先算哪一个，而是更根本的问题：如果同一个 lambda term 可以沿不同归约路径前进，这些路径会不会把计算带到互相矛盾、无法协调的结果？

在 lambda calculus 里，beta-reduction 把一次函数应用看成一次代换。一个 term 往往含有多个可归约位置，因此执行顺序天然是不唯一的。若不同顺序可能产生不同的 normal form，lambda calculus 就只能描述某种具体求值策略，而不能作为稳定的计算等式理论。Church-Rosser theorem 的核心承诺是：路径可以分叉，但分叉之后仍能重新汇合。

## Background

把 lambda calculus 看成抽象重写系统时，term 是节点，单步归约是有向边。一次计算不再只是线性轨迹，而是在一个由归约关系生成的图中行走。传统执行视角会问“下一步选哪个 redex”；Church-Rosser 视角会问“整个归约图有没有 confluence”。

confluence 的典型图像是 diamond property 的推广：若 `M ->* N1` 且 `M ->* N2`，则存在某个 `P`，使得 `N1 ->* P` 且 `N2 ->* P`。这里 `->*` 是零步或多步归约。局部的单步选择差异，被一个全局可汇合结构控制住。

## Core theorem

对 lambda calculus 的 beta-reduction，Church-Rosser theorem 说明 beta-reduction 是 confluent 的。换言之，从同一个 term 出发，即使选择不同的 redex、采用不同的归约顺序，只要两条归约路径分别到达 `N1` 和 `N2`，仍然可以继续归约到一个共同后继。

一个直接后果是 normal form 的唯一性：若某个 term 存在 normal form，那么这个 normal form 不依赖归约顺序。不同策略可能花费不同步数，甚至某些策略可能在非终止路径上打转；但只要某条路径到达 normal form，任何能到达 normal form 的路径不会得到另一个不兼容的答案。

## Key insight

这个定理的独特洞察，是把“执行顺序是否可靠”的问题提升为“重写系统的全局几何是否汇合”。它没有试图规定唯一正确的求值顺序，而是证明所有局部选择都嵌入同一个可汇合结构中。diamond/confluence 不是一个操作建议，而是对整个归约空间形状的约束。

因此，lambda calculus 中的计算不再只是按步骤执行程序，而是沿着一个归约图移动。Church-Rosser theorem 说明这个图的分叉不会破坏最终可达的规范结果。它把具体执行的偶然性和数学语义的确定性分离开：执行可以有多条路，语义仍能有唯一 normal form。

## Evaluation setting

这个条目的评价对象不是实验指标，而是概念解释是否抓住定理的结构作用。最低要求是说清三点：第一，beta-reduction 的路径分叉来自 redex 选择；第二，Church-Rosser/confluence 保证分叉路径可再汇合；第三，normal form 若存在则与归约顺序无关。

更高层的理解是：Church-Rosser theorem 把 lambda calculus 从一种具体求值机制提升为重写系统理论中的全局几何对象。它解释了为什么函数式语言和证明理论可以区分 reduction strategy 与 equational meaning：策略影响怎样到达结果，confluence 保证结果本身不因策略而分裂。
