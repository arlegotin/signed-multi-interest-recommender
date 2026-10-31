# Signed multi-interest recommender

**A Python research toolkit for recommending movies from likes and dislikes.**

It turns movie titles and genres into embeddings, builds user profiles from past
ratings, and scores candidate movies. Use it to prototype a recommender, compare
ways of using negative feedback, or reproduce a complete MovieLens 1M experiment.

The main model, **SignedMax**, keeps several interests per user. It scores each
movie by its best match to those interests, minus a penalty for resembling a movie
the user disliked.

## What you can use

- **Ranking models:** SignedMax, logistic regression, SVM and geometric baselines,
  with Python functions for fitting user histories and scoring your own embeddings.
- **An evaluation pipeline:** chronological splits, separate development and
  confirmation users, fixed model selection and paired statistical comparisons.
- **An auditable workflow:** saved aggregate results, durable records of failed
  and interrupted work, and a verifier that needs no dataset download.

## Try the scoring rule

From a checkout, with Python 3.11–3.12:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

This tiny example uses synthetic unit vectors: two different interests and one
disliked movie.

```python
import numpy as np
from signed_study.models import signed_scores

movies = np.eye(3, dtype=np.float64)
interests = movies[:2]
dislikes = movies[2:]

scores = signed_scores(movies, interests, dislikes, penalty=1.0)
print(scores.tolist())  # [1.0, 1.0, -1.0]
```

Both interests get a match. The disliked example gets penalized.
[Full setup, tests and study reproduction](REPRODUCE.md). Real movie runs acquire
MovieLens data and encoder weights separately.

For repeated catalog queries, an optional cache reuses movie similarities. Strict
mode preserves the original rankings; opt-in fast mode can reorder near ties.
[Benchmarks, memory costs and API](docs/research/retrieval.md).

## What the study found

**Logistic regression beat SignedMax.** The MovieLens 1M experiment compared
**83 configurations on development users**, then evaluated **13 frozen methods
on all 1,776 confirmation users**.

| Method | AUC, higher is better |
| --- | ---: |
| SignedMax | 0.5771 |
| Logistic regression | **0.6046** |

SignedMax minus logistic was **−0.0275 AUC**, with a paired 95% interval of
**[−0.0326, −0.0223]**. The repository keeps that result alongside the reusable
code. [Full findings and limitations](docs/study-outcome.md).

![Confirmation accuracy and recorded CPU costs](docs/figures/accuracy-cost.png)

The recorded CPU timings exclude shared encoding; Accelerate's actual thread
count wasn't independently verified. Faster scoring doesn't establish better
recommendations.

Check the published aggregates with Python 3.11+, without installing dependencies:

```sh
python3 scripts/verify_public_results.py
```

By **Artem Legotin**. **AGPL-3.0-only** software; **CC BY 4.0** writing and
figures. [Licensing](LICENSING.md) · [Upstream notices](THIRD_PARTY_NOTICES.md) ·
[Citation](CITATION.cff). Development used AI assistance; README editing used
[write-like-me](https://github.com/arlegotin/write-like-me).
