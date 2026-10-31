# Signed multi-interest recommendation

SignedMax scores a movie against a user's positive interests, then subtracts a
penalty for similarity to an explicit dislike. The question is whether that
extra geometry earns its place on top of frozen content embeddings.

The experiment is complete. **Explicit negatives helped; SignedMax lost to
simpler ways of using them.** This repository contains the implementation,
frozen comparison protocol, aggregate results and CPU cost measurements.

## What happened

MovieLens 1M, title/year/genre embeddings from `all-MiniLM-L6-v2`, 83 development
configurations and **1,776 held-out users**. All 13 frozen methods and controls
completed for every confirmation user.

| Method | AUC ↑ | Fit, ms ↓ | Catalog score + rank, ms ↓ |
| --- | ---: | ---: | ---: |
| SignedMax | 0.5771 | 1.369 | 3.670 |
| Logistic regression | **0.6046** | 2.976 | **2.120** |
| Negative mean | 0.5961 | **1.233** | 2.237 |
| RBF SVM | 0.6037 | 4.852 | 43.139 |

SignedMax minus the preselected logistic baseline: **−0.0275 AUC**, with a
paired 95% user-bootstrap interval of **[−0.0326, −0.0223]**. The proposed
advantage didn't survive confirmation. Adding negative feedback did improve
the positive-only readout by 0.0687 AUC; that's a secondary comparison.

![Confirmation accuracy versus fitting and catalog ranking cost](docs/figures/accuracy-cost.png)

SignedMax fit about 2.2× faster than logistic regression, then took about 1.7×
as long to rank the catalog. The negative-mean baseline also had higher AUC and
lower fit and ranking medians. These measurements cover 3,883 movies on an
Apple M3 Max, 32 users and five timed repeats. Shared encoding is excluded;
Accelerate's actual thread count wasn't independently verified.

The findings concern later observed extreme ratings with this encoder and
feedback history. They don't establish online recommendation quality.
[Full results and limitations](docs/study-outcome.md) ·
[Paired comparisons](docs/figures/native-paired-intervals.png)

## Use it

Check the published evidence with Python 3.11+; no downloads or dependencies:

```sh
python3 scripts/verify_public_results.py
```

For experiments, the reusable pieces are the [ranking models](src/signed_study/models.py),
user-disjoint temporal evaluation, matched controls and source-bound protocol.
[REPRODUCE.md](REPRODUCE.md) covers installation, synthetic tests, figure
regeneration and a fresh study. [The blueprint](docs/blueprint.md) records the
research question and protocol. MovieLens data and model weights are acquired
separately; neither is included in the release.

## Credit and license

By Artem Legotin. Cite [CITATION.cff](CITATION.cff) and the commit you use.
Software: **AGPL-3.0-only**. Writing and figures: **CC BY 4.0**.
[Licensing](LICENSING.md) · [Dataset, model and dependency attribution](THIRD_PARTY_NOTICES.md).

Development used AI assistance under the author's direction. README editing
used [write-like-me](https://github.com/arlegotin/write-like-me).
