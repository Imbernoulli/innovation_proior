我从 Calabi-Yau 的两种自然变形开始看这个方法。一个方向是 complex structure：改变哪些局部坐标变换被视为全纯。另一个方向是 complexified Kahler structure：改变 Kahler form 和 B-field 的复合参数。普通几何直觉会把这两类数据分开处理，因为前者属于复几何，后者属于 symplectic 或 Kahler 几何。

Mirror symmetry 的反直觉之处，就是它说这两类数据可以在一对不同的 Calabi-Yau 之间互换。`X` 的 Kahler moduli 不是只在 `X` 内部研究；它们可以通过镜像 `X^vee` 的 complex moduli 来研究。物理上，这是同一个 topological string theory 的两种对偶描述。数学上，这提供了一条把问题迁移到另一种几何结构中的通道。

枚举几何的难点正好落在 `X` 的 A-model 侧。数有理曲线或计算 Gromov-Witten invariants，看似是有限计数，但实际要处理曲线模空间、紧化、奇异性、虚计数和 multiple cover 修正。直接在曲线空间里算，每一步都会被几何复杂性拖住。

镜像转译把目标换成 `X^vee` 的 B-model。B-model 不直接数曲线，而研究复结构族上的全纯体形式和它的周期。周期 `int_gamma Omega` 随复结构变化，构成 variation of Hodge structures，并满足 Picard-Fuchs 微分方程。这类对象仍然深，但它们更接近线性代数、微分方程和单值化计算，而不是直接枚举曲线。

关键不是简单地说"另一个问题更容易"。关键是二者由物理对偶识别为同一个量。A-model 的 Yukawa coupling 包含 Gromov-Witten invariants；B-model 的对应 Yukawa coupling 可由周期数据算出。通过 mirror map，把镜像复结构坐标换回原 Calabi-Yau 的 Kahler 坐标，再展开生成函数，原本隐藏在曲线计数中的整数就从周期计算里出现。

这就是为什么 quintic threefold 的例子如此有代表性。经典 Schubert 演算能处理最低几阶：quintic 上有 2875 条直线（次数 1）、609250 条二次曲线（次数 2），这两个数字在镜像对称出现之前就已经由经典代数几何得到。但从三次曲线开始，曲线模空间迅速变得高度奇异，multiple cover 贡献互相纠缠，经典枚举技巧基本失效——难题不是缺少一个更快的算法，而是原始表示把可计算结构藏了起来。镜像计算把问题换到周期积分上后，解出的周期级数首先精确复现了 2875 和 609250 这两个已知数字，这本身就是镜像映射站得住脚的一次检验；随后同一套计算继续给出经典方法当时完全无法触及的高次曲线计数。数字就这样从周期的级数展开里自己冒了出来。
