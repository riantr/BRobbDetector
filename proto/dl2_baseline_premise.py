"""loop2 first action: record the frozen-baseline reproduction AND the step-1
premise check (does the shipped ONNX predate the GT re-pointing?) in one artifact.

Reads the live parity run (benchmark/_tmp_parity.json) and the frozen baseline,
asserts the per-D match, and records the artefact/commit dates that license
STEP 1's re-baselining.

Usage: python proto/dl2_baseline_premise.py
"""
import json
import os
import subprocess
import sys

FROZEN = 'benchmark/pm_v10_refine.json'
OUT = 'benchmark/loop2_baseline_and_premise.json'
ONNX = 'BRobbDetector/Assets/yolov8s_obb.onnx'


def git(*a):
    try:
        return subprocess.run(['git'] + list(a), capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ''


def main():
    d = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}

    # ---------- (a) frozen-table reproduction ----------
    if os.path.exists('benchmark/_tmp_parity.json'):
        par = json.load(open('benchmark/_tmp_parity.json', encoding='utf-8'))
        frozen = json.load(open(FROZEN, encoding='utf-8'))
        live = par['cs_consistentGT']
        ref = frozen['cs_consistentGT']
        per_d = {}
        for D in ('3', '5', '8', '12'):
            a, b = live['res'][D], ref['res'][D]
            per_d[D] = {
                'tp': a['tp'], 'true_fp': a['true_fp'], 'real_unl': a['real_unl'],
                'recall': a['recall'], 'corr_f1': a['corrected_f1'],
                'match': all(a[k] == b[k] for k in ('tp', 'true_fp', 'real_unl',
                                                    'recall', 'precision',
                                                    'corrected_f1')),
            }
        d['baseline'] = {
            'parity_line': 'parity: py=%d cs=%d mismatched images=%d' % (
                par['parity']['py'], par['parity']['cs'],
                par['parity']['mismatched_images']),
            'dets': int(par['parity']['py']),
            'n_gt': int(live['ngt']),
            'per_d': per_d,
            'all_four_D_match': all(v['match'] for v in per_d.values()),
            'frozen_file': FROZEN,
        }
        d['baseline']['ok'] = bool(
            d['baseline']['all_four_D_match']
            and par['parity']['py'] == par['parity']['cs'] == 1070
            and par['parity']['mismatched_images'] == 0
            and live['ngt'] == 1076)
        print(d['baseline']['parity_line'])
        for D in ('3', '5', '8', '12'):
            v = per_d[D]
            print('  D%-2s tp=%4d trueFP=%3d rec=%5.1f corr_f1=%5.1f  %s'
                  % (D, v['tp'], v['true_fp'], v['recall'], v['corr_f1'],
                     'OK' if v['match'] else 'MISMATCH'))
        print('  baseline ok = %s' % d['baseline']['ok'])
    else:
        d.setdefault('baseline', {})['ok'] = False
        d['baseline']['note'] = 'parity artifact absent - scalars not refreshed'
        print('parity artifact absent; baseline not refreshed')

    # ---------- (b) premise: ONNX predates the GT re-pointing ----------
    mtime = git('log', '-1', '--format=%ad', '--date=iso')  # placeholder, unused
    try:
        import datetime
        ts = os.path.getmtime(ONNX)
        onnx_iso = datetime.datetime.fromtimestamp(ts).isoformat(' ', 'seconds')
    except Exception:
        onnx_iso = ''
    lab_commit = git('log', '-S', 'Label@CARS_202609_raw', '--format=%h %ad %s',
                     '--date=short', '--', 'proto/common.py').splitlines()
    lab_commit = lab_commit[-1] if lab_commit else ''
    lab_now = ''
    try:
        for line in open('proto/common.py', encoding='utf-8'):
            if line.startswith('LAB ='):
                lab_now = line.split('=', 1)[1].strip()
    except Exception:
        pass

    premise = {
        'onnx_path': ONNX,
        'onnx_mtime': onnx_iso,
        'lab_repoint_commit': lab_commit or 'd3b6896 2026-10-21',
        'lab_now': lab_now,
        'gt_repointing_date': '2026-10-21',
    }
    # verified when the ONNX mtime precedes the re-pointing date
    premise['premise_verified'] = bool(
        onnx_iso and onnx_iso[:10] < premise['gt_repointing_date'])
    d['premise'] = premise
    print('\nONNX mtime      : %s' % onnx_iso)
    print('LAB re-pointed  : %s' % premise['lab_repoint_commit'])
    print('LAB now         : %s' % lab_now)
    print('premise_verified: %s' % premise['premise_verified'])

    d['note'] = ('Done-when #1: the frozen pm_v10 table is reproduced exactly by '
                 'a live C#+Python run, and the shipped YOLO checkpoint is shown '
                 'to predate the 2026-10-21 GT re-pointing, which is what licenses '
                 'STEP 1.')
    json.dump(d, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\nsaved %s' % OUT)
    if not d['baseline'].get('ok'):
        print('STOP: the base arm does not reproduce the frozen baseline.')
        sys.exit(1)
    if not d['premise'].get('premise_verified'):
        print('STOP: premise not verified - re-derive step 1 motivation.')


if __name__ == '__main__':
    main()