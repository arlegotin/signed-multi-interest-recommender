#!/usr/bin/env python3
"""Plot released accuracy/timing aggregates without private study artifacts.

Requires matplotlib; run .venv/bin/python scripts/plot_public_results.py.
The standard-library checker validates inputs before plotting. No model fitting,
encoding, individual outcomes, or bootstrap computation occurs here.
"""

import argparse
from pathlib import Path

from verify_public_results import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='optional alternative PNG destination')
    args = parser.parse_args()
    manifest, metrics, _, timing = verify(args.root)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    methods = ('SignedMax', 'MeanNegative', 'LinearLogistic', 'RBFSVM')
    names = {'SignedMax': 'SignedMax', 'MeanNegative': 'Mean negative',
             'LinearLogistic': 'Linear logistic (B*)', 'RBFSVM': 'RBF SVM'}
    colors = {'SignedMax': '#b54a38', 'MeanNegative': '#567450',
              'LinearLogistic': '#285c8a', 'RBFSVM': '#71568e'}
    offsets = (
        {'SignedMax': (10, 10), 'MeanNegative': (10, -18),
         'LinearLogistic': (-8, 17), 'RBFSVM': (10, -16)},
        {'SignedMax': (10, 10), 'MeanNegative': (10, -18),
         'LinearLogistic': (0, 18), 'RBFSVM': (-10, -16)},
    )
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 10, 'axes.titlesize': 12,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'}):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.4), sharey=True)
        fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.34, wspace=.20)
        for index, (ax, field, title, ticks, limits) in enumerate(zip(
            axes,
            ('fit_seconds', 'score_rank_seconds'),
            ('A   Fit the user readout', 'B   Score and rank the warm catalog'),
            ((1, 2, 3, 5, 8), (2, 5, 10, 20, 50)),
            ((.95, 9), (1.4, 75)),
        )):
            ax.set_xscale('log')
            ax.set_xlim(*limits)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.minorticks_off()
            ax.set_ylim(.570, .617)
            ax.set_yticks((.575, .585, .595, .605, .615))
            ax.grid(axis='both', color='#e5e7eb', linewidth=.7)
            ax.set_axisbelow(True)
            ax.set_title(title, loc='left', pad=14, fontweight='bold')
            ax.set_xlabel('Median time, ms (log scale)', labelpad=10)
            ax.tick_params(length=0, pad=7, colors='#464b53')
            for spine in ax.spines.values():
                spine.set_visible(False)
            for method in methods:
                x = float(timing[method][field]) * 1000
                y = float(metrics[method]['auc'])
                ax.scatter(x, y, s=75, color=colors[method], edgecolors='white',
                           linewidths=1.2, zorder=3)
                dx, dy = offsets[index][method]
                ax.annotate(names[method], (x, y), xytext=(dx, dy),
                            textcoords='offset points', color=colors[method],
                            ha='right' if dx < 0 else 'left', va='center', fontsize=10)
        axes[0].set_ylabel('Confirmation macro user AUC', labelpad=10)
        fig.text(.075, .94, 'Accuracy and warm computation cost',
                 fontsize=19, fontweight='bold', color='#20252d')
        fig.text(.075, .885, 'Four selected methods · descriptive accuracy means · lower time is faster',
                 fontsize=11, color='#505761')
        primary = manifest['primary']
        fig.text(.075, .205,
                 f'Primary result: SignedMax − linear logistic = {primary["mean_difference"]:.5f} AUC '
                 f'(95% paired interval {primary["interval_low"]:.5f} to {primary["interval_high"]:.5f}).',
                 fontsize=10, color='#343a43')
        fig.text(.075, .155,
                 'Accuracy: 1776 confirmation users. Timing: 32 development users × 5 timed repeats; 3883-item catalog.',
                 fontsize=9, color='#505761')
        fig.text(.075, .112,
                 'Time is the median across user repeat medians, after one warmup; score+rank includes scoring. No individual AUC intervals.',
                 fontsize=9, color='#505761')
        fig.text(.075, .069,
                 'Apple M3 Max · configured 1 worker / BLAS thread · Accelerate thread count unverified · background contention.',
                 fontsize=9, color='#505761')
        output = args.output or args.root / manifest['derived_figure']['path']
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, metadata={'Software': 'matplotlib'})
        plt.close(fig)
    print(f'Wrote {output} from verified public aggregates; no study rerun.')


if __name__ == '__main__':
    main()
