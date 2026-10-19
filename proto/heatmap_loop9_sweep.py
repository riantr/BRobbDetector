"""LOOP-9 tuning sweep: find POS_WEIGHT so the heatmap gets sharp, confident peaks.

The default focal loss (pos_weight=1) makes negatives dominate -> flat low maps.
Sweep pos_weight at low resolution / small subset for a fast signal.

Usage: python proto/heatmap_loop9_sweep.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch
from torch.utils.data import DataLoader, TensorDataset
import heatmap_loop9 as H

torch.set_num_threads(12)
RES = 160
SIGMA = 1.5
EPOCHS = 12
BATCH = 8


def prep(ids):
    ins, hms, oxs, oys, cms = [], [], [], [], []
    for iid in ids:
        img = cv2.imread(os.path.join(H.IMG, iid + ".jpg"))
        m = H.red_mask(img)
        mp, scale = H.resize_pad(m, res=RES)
        corners = [[pt[0] * scale, pt[1] * scale] for b in H.load_gt(iid) for pt in b]
        ins.append(mp.astype(np.float32) / 255.0)
        hms.append(H.make_heatmap(corners, res=RES, sigma=SIGMA))
        ox, oy, cm = H.make_offset(corners, res=RES)
        oxs.append(ox); oys.append(oy); cms.append(cm)
    return (np.stack(ins), np.stack(hms), np.stack(oxs),
            np.stack(oys), np.stack(cms))


def run(pos_weight):
    ids = H.get_ids()[:140]
    tr, va = ids[:120], ids[120:]
    ti, th, tox, toy, tcm = prep(tr)
    vi, vh, vox, voy, vcm = prep(va)

    def aug(inp, hm, ox, oy, cm):
        out = [(inp, hm, ox, oy, cm)]
        out.append((inp[:, ::-1].copy(), hm[:, ::-1].copy(),
                    -ox[:, ::-1].copy(), oy[:, ::-1].copy(), cm[:, ::-1].copy()))
        return out

    AI, AH, AOX, AOY, ACM = [], [], [], [], []
    for i in range(len(ti)):
        for a in aug(ti[i], th[i], tox[i], toy[i], tcm[i]):
            AI.append(a[0]); AH.append(a[1]); AOX.append(a[2]); AOY.append(a[3]); ACM.append(a[4])

    ds = TensorDataset(torch.from_numpy(np.stack(AI)).unsqueeze(1),
                       torch.from_numpy(np.stack(AH)).unsqueeze(1),
                       torch.from_numpy(np.stack(AOX)),
                       torch.from_numpy(np.stack(AOY)),
                       torch.from_numpy(np.stack(ACM)))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)

    model = H.CornerUNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    vi_t = torch.from_numpy(vi).unsqueeze(1)
    vh_t = torch.from_numpy(vh).unsqueeze(1)
    vox_t = torch.from_numpy(vox); voy_t = torch.from_numpy(voy)
    vcm_t = torch.from_numpy(vcm)

    for ep in range(EPOCHS):
        model.train()
        for x, hm, ox, oy, cm in dl:
            opt.zero_grad()
            ph, po = model(x)
            gt_off = torch.stack([ox, oy], dim=1)
            loss = H.focal_loss(ph, hm, pos_weight=pos_weight) + H.offset_loss(po, gt_off, cm)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ph, po = model(vi_t)
    v = ph[:, 0].numpy()
    thr = float((v > 0.3).mean())
    # peak count and how many val images have a peak > 0.3
    n_peak_img = 0
    for i in range(v.shape[0]):
        if v[i].max() > 0.3:
            n_peak_img += 1
    return float(v.max()), float(v.mean()), thr, n_peak_img, len(va)


if __name__ == "__main__":
    for pw in [1.0, 10.0, 30.0, 100.0, 300.0]:
        t0 = time.time()
        mx, mean, frac, npi, n = run(pw)
        print(f"pos_weight={pw:6.1f}  hm_max={mx:.3f} mean={mean:.4f} "
              f"frac>0.3={frac:.5f}  imgs_with_peak={npi}/{n}  ({time.time()-t0:.0f}s)")