import numpy as np
import cv2
from .energy import saliency_map


def cutting_crop_scale(img, target_w, target_h):
    h, w = img.shape[:2]
    s = saliency_map(img)
    ssum = s.cumsum(0).cumsum(1)

    def win_sum(x0, y0, x1, y1):
        a = ssum[y1 - 1, x1 - 1]
        b = ssum[y0 - 1, x1 - 1] if y0 > 0 else 0
        c = ssum[y1 - 1, x0 - 1] if x0 > 0 else 0
        d = ssum[y0 - 1, x0 - 1] if (x0 > 0 and y0 > 0) else 0
        return a - b - c + d

    ar = target_w / target_h
    if w / h > ar:
        cw, ch = int(round(h * ar)), h
    else:
        cw, ch = w, int(round(w / ar))
    cw, ch = min(cw, w), min(ch, h)

    best, best_xy = -1, (0, 0)
    step = max(1, min(w - cw, h - ch) // 40 + 1)
    for y0 in range(0, h - ch + 1, step):
        for x0 in range(0, w - cw + 1, step):
            v = win_sum(x0, y0, x0 + cw, y0 + ch)
            if v > best:
                best, best_xy = v, (x0, y0)
    x0, y0 = best_xy
    crop = img[y0:y0 + ch, x0:x0 + cw]
    out = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

    kept = np.zeros((h, w), bool)
    kept[y0:y0 + ch, x0:x0 + cw] = True
    return out, kept
