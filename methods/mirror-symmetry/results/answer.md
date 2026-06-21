# Mirror Symmetry

Mirror symmetry 的独特洞察，是把一对 Calabi-Yau 的 complex moduli 与 complexified Kahler moduli 互换。也就是说，`X` 上由 Kahler 参数控制的 A-model，可以通过镜像 `X^vee` 上由复结构参数控制的 B-model 来计算。

这一步把 string duality 变成了枚举几何的计算原则。原来在 `X` 上要数有理曲线或求 Gromov-Witten invariants；这些问题直接处理时会遇到曲线模空间、奇异性、紧化和虚计数等困难。镜像后，同一个物理量在 `X^vee` 上表现为全纯三形式的周期积分：

`int_gamma Omega`

这些周期随复结构变化，形成 variation of Hodge structures，并满足 Picard-Fuchs 微分方程。于是计算路线从“直接数曲线”变成“解周期方程、求 mirror map、展开 Yukawa coupling 或 prepotential”。从展开系数中可以读出原 Calabi-Yau 上的枚举不变量。

方法论上的突破在于，mirror symmetry 没有把困难计数问题局部修补成可算问题，而是通过对偶几何改变了问题的表达。A-model 侧的量子枚举数据，在 B-model 侧被 Hodge 理论和周期计算编码。物理中的对偶性因此不仅说明两个理论等价，还能预测具体的枚举几何整数。

最典型的历史信号是 quintic threefold：Candelas、de la Ossa、Green、Parkes 的镜像计算给出了有理曲线计数的强预测，后来推动了 Gromov-Witten theory、mirror theorem 和 homological mirror symmetry 等严格数学发展。它展示的不是一个单次技巧，而是一种新的数学策略：把难算的几何对象转译到一个对偶空间中，让可计算结构显现出来。
