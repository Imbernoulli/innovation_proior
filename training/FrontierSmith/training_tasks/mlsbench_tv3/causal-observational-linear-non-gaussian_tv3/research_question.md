Identifiability in the linear non-Gaussian model is purchased entirely with departure from
Gaussianity, and that currency can run short: uniform noise sits closer to Gaussian than
Laplace does, finite samples dilute whatever asymmetry exists, and contamination or mixing
can leave individual variables nearly Gaussian even when the model class formally holds. A
method tuned to vivid non-Gaussianity then starts flipping coins with conviction. This
variant asks for the opposite temperament: orientation aggressiveness that tracks the amount
of identifying signal actually measured in the data at hand.

The required structure is a confidence channel. The method must quantify, per variable or
per pair, how far the relevant distributions sit from Gaussian — negentropy proxies, bounded
contrast functions, cumulant magnitudes carrying uncertainty estimates, or anything that is
itself stable under contamination — and route that measurement into its orientation
decisions, so that vivid asymmetry licenses firm arrows while weak asymmetry demotes the
decision to a fallback that does not lean on high-order moments. The fallback is part of the
design, not an apology: it should be explicit, cheap, and strictly less noise-sensitive than
the primary statistic. Branching on the noise family is unavailable, since the family is
unknown at run time and a single configuration serves all three settings.

The defense is a graceful-degradation curve rather than a single number: as identifying
signal thins, precision should be surrendered slowly and predictably rather than in a cliff,
and the settings rich in non-Gaussianity must not be handicapped by the caution the poor
ones require. A method that earns its score only where the noise is loudly non-Gaussian has
answered the classical LiNGAM question, not this one.
