"""Appendix line 2 figure for the model chosen on L90 (gradient boosting).

x = e, each Jan 1, 2025 candidate's predicted chance of a routine inspection (line2_propensity.py,
results/line2_scores.csv); y = q, the chosen model's chance of a serious violation. Panels split candidates by
outcome. Dashed line: e cutoff (right = predicted inspected). Dotted line: q cutoff, the score of the K90-th
inspected restaurant, where the model reaches 90% recall. Writes results/l90_line2_scores.csv and
results/line2_scatter.png.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import history
import ml_features
import train_models as tm
from config import RESULTS, TEST_SCORING_DATE

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
HISTORY, NO_HISTORY = '#2a78d6', '#eb6834'


def main():
    chosen = json.loads((RESULTS / 'recall_targets.json').read_text())['chosen']
    follow = json.loads((RESULTS / 'l90_followups.json').read_text())
    e_cut = json.loads((RESULTS / 'line2_propensity.json').read_text())['e_cutoff']
    k = follow['line1']['test']['K90_model']

    ins = history.load_inspections()
    rows = pd.concat([tm.labeled(ins, y) for y in tm.TRAIN_YEARS])
    tr = tm.train_rows(rows, TEST_SCORING_DATE)
    m = tm.GRID[chosen['family']][1](chosen['params']).fit(tr[tm.X], tr.y)
    sc = pd.read_csv(RESULTS / 'line2_scores.csv', dtype={'id': str}).set_index('id')
    fc = ml_features.build(ins, sc.index.tolist(), TEST_SCORING_DATE)
    sc['q'] = m.predict_proba(fc[tm.X])[:, 1]
    q_cut = float(np.sort(sc.q[sc.inspected.eq(1)].to_numpy())[::-1][k - 1])
    sc.to_csv(RESULTS / 'l90_line2_scores.csv')

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED})
    panels = [('Inspected, serious violator', sc.inspected.eq(1) & sc.y.eq(1)),
              ('Inspected, not serious', sc.inspected.eq(1) & sc.y.eq(0)),
              ('Not inspected', sc.inspected.eq(0))]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, (title, mask) in zip(axes, panels):
        g = sc[mask]
        for nohist, color, label in ((False, HISTORY, 'Has routine history'), (True, NO_HISTORY, 'No routine history')):
            h = g[g.n_routine.eq(0) == nohist]
            ax.scatter(h.e, h.q, s=4, c=color, alpha=0.45, linewidths=0, label=label, rasterized=True)
        ax.axvline(e_cut, color=MUTED, lw=0.8, ls='--')
        ax.axhline(q_cut, color=MUTED, lw=0.8, ls=':')
        above = int((g.q >= q_cut).sum())
        ax.set_title(f'{title}\n{above:,} of {len(g):,} above the q cutoff', fontsize=8, color=INK)
        ax.set_yscale('log')
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_xlabel('e: predicted chance of inspection')
    axes[0].set_ylabel('q: predicted chance of serious violation')
    handles, labels = axes[2].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=7, frameon=False, markerscale=3)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(RESULTS / 'line2_scatter.png', dpi=300)
    print('q cutoff', round(q_cut, 5), 'e cutoff', e_cut,
          {t: int((sc[mk].q >= q_cut).sum()) for t, mk in panels})


if __name__ == '__main__':
    main()
