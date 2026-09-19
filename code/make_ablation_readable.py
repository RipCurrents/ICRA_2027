"""Render existing scalar paired effects at one-column publication size.

Run from the repository root. Outputs remain beside this critic-owned script.
"""
from pathlib import Path
import csv
import datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source = Path('train/edge_distill/results/student_bootstrap_cis.tsv')
out = Path(__file__).resolve().parent
rows = list(csv.DictReader(source.open(), delimiter='\t'))
comparisons = [
    ('yolo26s: dense − expert', 'yolo26s_dense', 'yolo26s_canonical'),
    ('yolo26n: dense − expert', 'T1_teacher', 'canonical'),
    ('interpolation − expert', 'interp_control', 'canonical'),
    ('teacher − interpolation', 'T1_teacher', 'interp_control'),
    ('add no-rip negatives', 'T1_teacher', 'no_negatives'),
    ('smooth teacher labels', 'T1_teacher', 'no_smoothing'),
    ('feature KD: expert labels', 'native_KD', 'canonical'),
    ('feature KD: dense labels', 'dense_plus_KD', 'T1_teacher'),
    ('dense → expert fine-tune', 'two_stage_lowLR', 'T1_teacher'),
]
points = []
for label, model, reference in comparisons:
    hits = [r for r in rows if r['task'] == 'bbox_from_instance'
            and r['model'] == model + '[pooled 3 seeds]'
            and r['pair_vs'] == reference + '[pooled 3 seeds]']
    assert len(hits) == 1, (label, len(hits))
    r = hits[0]
    points.append({'label': label, 'model': r['model'], 'reference': r['pair_vs'],
                   'delta': float(r['delta']), 'low': float(r['delta_lo']), 'high': float(r['delta_hi']),
                   'display': f"{Decimal(r['delta']).quantize(Decimal('.001'), rounding=ROUND_HALF_UP):+.3f}"})
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 7,
                     'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})
fig, ax = plt.subplots(figsize=(3.5, 2.65), dpi=250)
for i, r in enumerate(points):
    ax.plot([r['low'], r['high']], [i, i], color='#2a78d6', linewidth=1.25)
    ax.plot(r['delta'], i, 'o', color='#2a78d6', markeredgecolor='white', markeredgewidth=.6, markersize=3.5)
    ax.text(.166, i, r['display'], va='center', fontsize=6.5, color='#52514e', clip_on=False)
ax.axvline(0, color='#777777', linewidth=.8, zorder=0)
ax.set_yticks(range(len(points)), [r['label'] for r in points])
ax.set_ylim(len(points)-.5, -.7)
ax.set_xlim(-.10, .15)
ax.set_xticks([-.10, 0, .10], ['−0.10', '0', '+0.10'])
ax.set_xlabel('ΔF2@0.5 (95% video-bootstrap CI)')
ax.set_title('Paired effects on the test split', fontsize=7.5, pad=6)
ax.tick_params(axis='y', length=0, pad=5)
ax.grid(axis='x', color='#dddddd', linewidth=.4, zorder=0)
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color('#777777')
fig.subplots_adjust(left=.445, right=.84, bottom=.17, top=.88)
for suffix in ('pdf', 'png'):
    fig.savefig(out / ('ablation_deltas.' + suffix))
(out/'ablation_manifest.json').write_text(json.dumps({
    'date': datetime.datetime.now().astimezone().isoformat(),
    'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'points': points, 'scope': 'Same nine stored scalar comparisons; no new evaluation or bootstrap.',
    'display_rounding': 'Three decimals, decimal half-up from the recorded four-decimal TSV, matching existing figure labels.'
}, indent=2)+'\n')
print('Rendered nine recorded comparisons at 3.5 × 2.65 inches.')
