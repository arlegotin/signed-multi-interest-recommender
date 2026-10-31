# Signed multi-interest recommendation

SignedMax ranks movies by a user's strongest positive interest and penalizes
similarity to explicit dislikes. This repository puts that rule through a complete
MovieLens 1M study and provides the code to compare it with simpler baselines.

[Results](#results) · [Reproduce](REPRODUCE.md) ·
[Retrieval](docs/research/retrieval.md) · [Citation](CITATION.cff)

## What you can reuse

- Deterministic ranking models, including SignedMax, logistic regression, SVM
  and negative-mean baselines.
- Temporal evaluation with separate development and confirmation users, matched
  controls and paired bootstrap comparisons.
- Durable attempt journals and report recovery that keep failed work visible.
- An optional memory-bounded similarity cache for repeated catalog scoring.

## Results

**Explicit dislikes helped. Logistic regression used them better than SignedMax.**

The study evaluated **83 development configurations** and **13 frozen methods**
across **1,776 confirmation users**, with complete coverage.

| Method | AUC ↑ |
| --- | ---: |
| SignedMax | 0.5771 |
| Logistic regression | **0.6046** |
| Negative mean | 0.5961 |

SignedMax trailed the preselected logistic baseline by **0.0275 AUC**, with a
paired 95% interval of **[−0.0326, −0.0223]** for SignedMax minus logistic.
Its gain over the positive-only readout was **0.0687 AUC**, a secondary comparison.
[Full findings and limitations](docs/study-outcome.md).

![Confirmation accuracy and recorded CPU costs](docs/figures/accuracy-cost.png)

The original CPU timings exclude shared encoding; Accelerate's actual thread
count wasn't independently verified.

Separate local benchmarks measured **1.19× full-catalog** and **1.10×
support-excluded** speedups in fast mode against optimized dense scoring on 3,883 movies,
using about 133 MB of persistent cache storage plus workspace. Fast mode can
reorder near ties; default strict mode uses the original calculation.
[Raw measurements and numerical contract](docs/research/retrieval.md).

## Quick check

Verify the published results with Python 3.11+, without installing dependencies
or downloading data:

```sh
python3 scripts/verify_public_results.py
```

[REPRODUCE.md](REPRODUCE.md) covers installation, synthetic tests and offline
benchmarks. MovieLens data and encoder weights are acquired separately.

By **Artem Legotin**. **AGPL-3.0-only** software; **CC BY 4.0** writing and
figures. [Licensing](LICENSING.md) · [Upstream notices](THIRD_PARTY_NOTICES.md).
Cite [CITATION.cff](CITATION.cff) and record the commit used. Development used
AI assistance; README editing used [write-like-me](https://github.com/arlegotin/write-like-me).
