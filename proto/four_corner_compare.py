"""Side-by-side evaluation of the CURRENT four-corner algorithm on real vs
simulated data: one artifact, one table.

Inputs (both produced by the repo's own harnesses with the shipped config, i.e.
no overrides, so `cfg=""`/`pipe=""` == HOLE_DEFAULT / PIPE_DEFAULT):
  benchmark/four_corner_real.json  <- proto/sweep_real.py --seed 0 (500 real imgs)
  benchmark/four_corner_emul.json  <- proto/sweep_emulation.py (data/Image@emul4)

Metric difference that must be kept in view:
  * REAL  — 1076 rule-consistent GT, and the KPI is the GT-corrected metric: an
    unmatched detection is a TRUE FP only if `is_real_unlabeled` says it is not a
    real-but-unlabelled annotation. `fp` below is trueFP.
  * EMUL  — the emulation GT is complete, so no correction: plain precision /
    recall. The harness scores only the RED-class GT (3295 boxes of 9982), which
    is the correct denominator for a red-only detector.

Usage: python proto/four_corner_compare.py
"""
import json

REAL = 'benchmark/four_corner_real.json'
EMUL = 'benchmark/four_corner_emul.json'
OUT_JSON = 'benchmark/four_corner_real_vs_emul.json'
OUT_MD = 'benchmark/four_corner_real_vs_emul.md'
DIST = ('3', '5', '8', '12')


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def row(rec):
    out = {}
    for D in DIST:
        d = rec['per_D'][D]
        fp = d.get('true_fp', d.get('fp', 0))
        out[D] = {'tp': d['tp'], 'fp': fp,
                  'recall': d['recall'], 'precision': d['precision'],
                  'f1': round(f1(d['precision'], d['recall']), 1)}
    return out


def main():
    r = json.load(open(REAL, encoding='utf-8'))[0]
    e = json.load(open(EMUL, encoding='utf-8'))[0]
    R, E = row(r), row(e)

    res = {
        'algorithm': 'pm_v10 shipped (holes + edgefit + side refine), no overrides',
        'real': {'source': REAL, 'dataset': '500 real CARS images',
                 'n_gt': r['ngt'], 'n_det': r['ndet'], 'sec': r['sec'],
                 'metric': 'GT-corrected: fp = trueFP (is_real_unlabeled applied)',
                 'per_D': R},
        'emulation': {'source': EMUL, 'dataset': 'data/Image@emul4 (5000 imgs)',
                      'n_gt': e['ngt'], 'n_det': e['ndet'], 'sec': e['sec'],
                      'metric': 'plain (complete GT); RED-class GT only',
                      'per_D': E},
        'delta_emul_minus_real': {D: {'recall': round(E[D]['recall'] - R[D]['recall'], 1),
                                      'precision': round(E[D]['precision'] - R[D]['precision'], 1),
                                      'f1': round(E[D]['f1'] - R[D]['f1'], 1)}
                                  for D in DIST},
    }
    json.dump(res, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    lines = []
    lines.append('# Current four-corner algorithm — real vs simulated data\n')
    lines.append('Algorithm: **pm_v10 shipped** (holes + edgefit + side refine), no overrides. '
                 'Same point-match metric in both runs (TP = all 4 corners within D px).\n')
    lines.append('| D | real: TP / trueFP | real R / P / F1 | emul: TP / FP | emul R / P / F1 | '
                 'ΔF1 (emul − real) |')
    lines.append('|---|---|---|---|---|---|')
    for D in DIST:
        lines.append('| **%s** | %d / %d | %.1f / %.1f / **%.1f** | %d / %d | %.1f / %.1f / **%.1f** | %+0.1f |'
                     % (D, R[D]['tp'], R[D]['fp'], R[D]['recall'], R[D]['precision'], R[D]['f1'],
                        E[D]['tp'], E[D]['fp'], E[D]['recall'], E[D]['precision'], E[D]['f1'],
                        E[D]['f1'] - R[D]['f1']))
    lines.append('')
    lines.append('Denominators: **real** 1076 rule-consistent GT boxes on 500 images '
                 '(%d detections, %.2f/img); **emulation** 3295 RED-class GT boxes on 5000 '
                 'images (%d detections, %.2f/img). The emulation set has 9982 GT boxes in '
                 'total, but only the red-class subset is the correct denominator for a '
                 'red-only detector.' % (r['ndet'], r['ndet'] / 500.0, e['ndet'], e['ndet'] / 5000.0))
    lines.append('')
    lines.append('**Metric caveat:** the real column uses the GT-corrected metric (its `fp` is '
                 'trueFP); the emulation GT is complete, so its column is plain P/R. '
                 'Do not compare the `fp` counts across the two columns — compare the rates.')
    lines.append('')
    lines.append('Reading: at **D=5** the two datasets agree within ~1.5 points of F1 '
                 '(real %.1f vs emul %.1f), which is why the emulation is a valid proxy for '
                 'D=5 structural tuning. They diverge away from it: **real wins at D=3 by '
                 '%.1f F1** because the hand-drawn GT is precise (best-orthogonal fit p50 '
                 '0.50 px), while the emulation is **optimistic at D>=8 by %.1f (D8) and %.1f '
                 '(D12)** because its boxes are exact rectangles whereas real annotations are '
                 'wobbled and arrow-corrupted.'
                 % (R['5']['f1'], E['5']['f1'],
                    R['3']['f1'] - E['3']['f1'],
                    E['8']['f1'] - R['8']['f1'], E['12']['f1'] - R['12']['f1']))
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

    print('%-4s %-14s %-18s %-14s %-18s %s' %
          ('D', 'real TP/FP', 'real R/P/F1', 'emul TP/FP', 'emul R/P/F1', 'dF1'))
    for D in DIST:
        print('%-4s %-14s %-18s %-14s %-18s %+0.1f' % (
            D, '%d/%d' % (R[D]['tp'], R[D]['fp']),
            '%.1f/%.1f/%.1f' % (R[D]['recall'], R[D]['precision'], R[D]['f1']),
            '%d/%d' % (E[D]['tp'], E[D]['fp']),
            '%.1f/%.1f/%.1f' % (E[D]['recall'], E[D]['precision'], E[D]['f1']),
            E[D]['f1'] - R[D]['f1']))
    print('\nsaved %s and %s' % (OUT_JSON, OUT_MD))


if __name__ == '__main__':
    main()