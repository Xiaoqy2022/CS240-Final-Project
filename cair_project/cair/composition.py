import numpy as np
import cv2

from .energy import importance_for, foreground_center, foreground_bbox, saliency_map
from .seam import carve_vertical, carve_horizontal


def _hstack(a, b):
    return np.concatenate([a, b], axis=1)


def _vstack(a, b):
    return np.concatenate([a, b], axis=0)


def _carve_two_regions_h(img, protect, split_col, dL, dR):
    h, w = img.shape[:2]
    split_col = int(np.clip(split_col, 1, w - 1))
    left, right = img[:, :split_col], img[:, split_col:]
    pl, pr = protect[:, :split_col], protect[:, split_col:]

    lout, lorig = carve_vertical(left, pl, dL, recompute=False)
    rout, rorig = carve_vertical(right, pr, dR, recompute=False)
    out = _hstack(lout, rout)

    kept = np.zeros((h, w), bool)
    if dL <= 0: 
        for i in range(h):
            kept[i, lorig[i]] = True
    else:
        kept[:, :split_col] = True
    if dR <= 0:
        for i in range(h):
            kept[i, split_col + rorig[i]] = True
    else:
        kept[:, split_col:] = True
    return out, kept


def _carve_two_regions_v(img, protect, split_row, dU, dD):
    h, w = img.shape[:2]
    split_row = int(np.clip(split_row, 1, h - 1))
    top, bot = img[:split_row], img[split_row:]
    pu, pd = protect[:split_row], protect[split_row:]

    tout, torig = carve_horizontal(top, pu, dU, recompute=False)
    bout, borig = carve_horizontal(bot, pd, dD, recompute=False)
    out = _vstack(tout, bout)

    kept = np.zeros((h, w), bool)
    if dU <= 0:
        for j in range(w):
            kept[torig[:, j], j] = True
    else:
        kept[:split_row, :] = True
    if dD <= 0:
        for j in range(w):
            kept[split_row + borig[:, j], j] = True
    else:
        kept[split_row:, :] = True
    return out, kept


def _semantic_horizon(img, sal):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    gy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)).sum(axis=1)
    h = len(gy)
    band = np.ones(h); band[: h // 8] = 0.3; band[-h // 8:] = 0.3
    l = int(np.argmax(gy * band))
    return l


def _semantic_vertical(img, sal):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).sum(axis=0)
    w = len(gx)
    band = np.ones(w); band[: w // 8] = 0.3; band[-w // 8:] = 0.3
    return int(np.argmax(gx * band))


def resize_thirds(img, target_w, target_h, importance, seg):
    h, w = img.shape[:2]
    xc, yc = foreground_center(seg)
    cands = [target_w / 3.0, 2 * target_w / 3.0]
    x_target = min(cands, key=lambda t: abs(t - xc * target_w / w))
    dL = int(round(x_target - xc))
    dR = (target_w - w) - dL
    out, kept = _carve_two_regions_h(img, importance, int(round(xc)), dL, dR)
    out = _fit_height(out, target_h, importance, kept)
    return out, kept, dict(ref=("fg_center_x", xc), target=("thirds_line", x_target), dL=dL, dR=dR)


def resize_central(img, target_w, target_h, importance, seg):
    h, w = img.shape[:2]
    xc, yc = foreground_center(seg)
    x_target = target_w / 2.0
    dL = int(round(x_target - xc))
    dR = (target_w - w) - dL
    out, kept = _carve_two_regions_h(img, importance, int(round(xc)), dL, dR)
    out = _fit_height(out, target_h, importance, kept)
    return out, kept, dict(ref=("fg_center_x", xc), target=("center", x_target), dL=dL, dR=dR)


def resize_horizontal(img, target_w, target_h, importance, seg):
    h, w = img.shape[:2]
    sal = saliency_map(img)
    l = _semantic_horizon(img, sal)
    frac = l / h
    gold = min([0.382, 0.618], key=lambda g: abs(g - frac))
    y_target = gold * target_h
    dU = int(round(y_target - l))
    dD = (target_h - h) - dU
    out, kept = _carve_two_regions_v(img, importance, l, dU, dD)
    out = _fit_width(out, target_w, importance, kept)
    return out, kept, dict(ref=("horizon_y", l), target=("golden_%.3f" % gold, y_target), dU=dU, dD=dD)


def resize_symmetric(img, target_w, target_h, importance, seg):
    h, w = img.shape[:2]
    sal = saliency_map(img)
    k = _semantic_vertical(img, sal)
    x_target = target_w / 2.0
    dL = int(round(x_target - k))
    dR = (target_w - w) - dL
    out, kept = _carve_two_regions_h(img, importance, k, dL, dR)
    out = _fit_height(out, target_h, importance, kept)
    return out, kept, dict(ref=("axis_x", k), target=("middle", x_target), dL=dL, dR=dR)


def _fit_height(img, target_h, importance, kept):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    prot = cv2.resize(importance, (w, h), interpolation=cv2.INTER_LINEAR)
    out, _ = carve_horizontal(img, prot, target_h - h, recompute=False)
    return out


def _fit_width(img, target_w, importance, kept):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    prot = cv2.resize(importance, (w, h), interpolation=cv2.INTER_LINEAR)
    out, _ = carve_vertical(img, prot, target_w - w, recompute=False)
    return out

def detect_composition(img):

    sal = saliency_map(img)
    from .energy import foreground_mask
    seg = foreground_mask(img, sal)
    h, w = img.shape[:2]
    bbox = foreground_bbox(seg)

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    mirror = np.abs(g - g[:, ::-1]).mean() / (g.mean() + 1e-6)

    gy = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)).sum(axis=1)
    horizon_strength = gy.max() / (gy.mean() + 1e-6)

    fg_ratio = seg.mean()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) / 2 / w
        central = abs(cx - 0.5) < 0.12 and fg_ratio > 0.06
    else:
        central = False

    if mirror < 0.18 and horizon_strength > 6:
        return "symmetric"
    if horizon_strength > 9 and fg_ratio < 0.12:
        return "horizontal"
    if central:
        return "central"
    return "thirds"


RULES = {
    "thirds": resize_thirds,
    "central": resize_central,
    "horizontal": resize_horizontal,
    "symmetric": resize_symmetric,
}


def resize(img, target_w, target_h, comp_type=None, alpha=3.0):
    if comp_type is None:
        comp_type = detect_composition(img)
    importance, seg = importance_for(img, comp_type, alpha=alpha)
    out, kept, info = RULES[comp_type](img, target_w, target_h, importance, seg)
    info["comp_type"] = comp_type
    return out, kept, info
