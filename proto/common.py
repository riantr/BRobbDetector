import csv, glob, json, os, re, sys
import collections
import numpy as np
import cv2

DATA = r"E:\Xray\CARS\data"
IMG = os.path.join(DATA, "Image@CARS_202609_annotated")
# Ground truth. NOTE: "Label@CARS_202609_annotated" was overwritten by a detector
# run and no longer holds GT; the benchmark GT lives in "Compare-V1@...".
LAB = os.path.join(DATA, "Compare-V1@CARS_202609_annotated")


def load_expected():
    det = {}
    for f in glob.glob(os.path.join(DATA, "2026-09-*_detail.csv")):
        with open(f, encoding="utf-8-sig") as fh:
            r = csv.reader(fh)
            next(r, None)
            for row in r:
                if len(row) >= 3:
                    det[row[0]] = (row[1], row[2])
    exp = {}
    for k, (res, d) in det.items():
        n = 0
        if d:
            for p in d.replace("，", ",").split(","):
                m = re.match(r"^(.*?)(\d+)$", p.strip())
                if m:
                    n += int(m.group(2))
        exp[k] = n
    return exp


def red_mask(img, smin=80, vmin=110):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, smin, vmin), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, smin, vmin), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def rect_corners(rect):
    pts = cv2.boxPoints(rect)
    return pts.astype(np.int32)


def iou_obb(a, b):
    """a,b: 4x2 int arrays. Approximate IoU via mask."""
    xa, ya, wa, ha = cv2.boundingRect(a)
    xb, yb, wb, hb = cv2.boundingRect(b)
    W = max(xa + wa, xb + wb) + 1
    H = max(ya + ha, yb + hb) + 1
    m1 = np.zeros((H, W), np.uint8)
    m2 = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m1, [a], 255)
    cv2.fillPoly(m2, [b], 255)
    inter = np.count_nonzero(cv2.bitwise_and(m1, m2))
    union = np.count_nonzero(cv2.bitwise_or(m1, m2))
    return inter / union if union else 0.0


def nms(boxes, thr=0.4):
    boxes = sorted(boxes, key=lambda r: r[1][0] * r[1][1], reverse=True)
    keep = []
    for r in boxes:
        c = rect_corners(r)
        if all(iou_obb(c, rect_corners(k)) < thr for k in keep):
            keep.append(r)
    return keep