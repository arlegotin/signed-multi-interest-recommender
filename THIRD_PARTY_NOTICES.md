# Third-party attribution and boundaries

The release includes original software, synthetic tests and aggregate research
results. It excludes downloaded data, movie metadata and derived text, rating
histories, user identifiers, embeddings, per-item scores, model weights and
private execution artifacts. Public hashes identify local scientific evidence;
they do not supply it or grant rights to upstream assets.

## MovieLens 1M

The study uses [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/),
provided by GroupLens Research at the University of Minnesota. Dataset citation:

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History
and Context*. ACM Transactions on Interactive Intelligent Systems 5(4),
Article 19. [doi:10.1145/2827872](https://doi.org/10.1145/2827872).

The acquired archive's bundled usage license permits research subject to its
conditions, requires acknowledgment, prohibits implied endorsement, and requires
separate permission for redistribution or commercial/revenue-bearing use.
Acquire the dataset separately and review its bundled README. The project
licenses do not relicense it. The official archive used here has SHA-256
`a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.
No endorsement by GroupLens or the University is claimed.

## Frozen encoder

Movie title/year/genre text was encoded locally with
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
at immutable revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
The model card records Apache-2.0; acquisition validates and retains that
evidence locally. Weights, tokenizer files and cached vectors are not bundled.

The Sentence Transformers framework is described in Nils Reimers and Iryna
Gurevych. 2019. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
EMNLP-IJCNLP, pp. 3982–3992.
[doi:10.18653/v1/D19-1410](https://aclanthology.org/D19-1410/).
This framework citation does not identify the later encoder checkpoint as a
model trained in that paper; use the model card and revision above for it.

## Dependencies and research context

Runtime dependencies are installed separately. Their distributions retain their
own licenses and notices. Core libraries include NumPy, SciPy, scikit-learn,
PyTorch, Sentence Transformers, Hugging Face Hub, psutil and Matplotlib. No
vendored library source or binaries are included; project grants do not replace
upstream terms. [requirements.lock.txt](requirements.lock.txt) records the
qualified native environment's versions, not a redistribution of packages.

The [blueprint](docs/blueprint.md) contains the inherited research question,
prior-art references and protocol. Baseline names and citations do not imply
an independently reviewed claim of novelty or universal method superiority.

Development, experiments and reporting used AI assistance under the owner's
direction. README editing used
[write-like-me](https://github.com/arlegotin/write-like-me), inspected at
revision `b7a28e64fed749406a2fd8406d5ebd3eedd584c0`; the skill is not bundled.
Standard license documents retain their own copyright notices.
