Every orientation decision in the linear non-Gaussian framework is settled by moments beyond
the second, and moments beyond the second are exactly what heavy tails corrupt. With Laplace
and exponential noise a handful of extreme rows can own the sample estimate of a third or
fourth cumulant outright; the resulting orientation statistic is then a report about
outliers, not about causal direction, and it changes sign when a different handful of rows
lands in the sample. This variant makes tail discipline the organizing principle: no raw
higher-order sample moment may reach an orientation decision without passing through an
influence-bounding stage.

The toolkit is classical robust statistics, aimed at a specific target. Location and scale
come from medians and MAD rather than means and standard deviations; observations are
winsorized, clipped, or reweighted before any cube or fourth power is taken; contrast
functions with bounded influence stand in for polynomial cumulants where possible. The
delicate part — and the substance of the variant — is that non-Gaussianity is the signal as
well as the vulnerability: clip too hard and the uniform-noise setting, whose identifying
information lives in the absence of tails, goes silent. The taming stage must therefore be
calibrated from the data's own measured tail weight, one configuration across all three
noise families, never switched per setting.

The claim to defend is stability with retained power: direction calls that keep their sign
under resampling and row deletion in the heavy-tailed settings; directed-edge precision and
SHD improving there because outlier-driven arrow flips are gone; and a
uniform-noise score demonstrating that the robustness layer knew when to leave light-tailed
data alone.
