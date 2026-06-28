import cv2
import numpy as np


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def gradient_energy(img):
    gray = to_gray(img).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    e = np.abs(gx) + np.abs(gy)
    return e


def saliency_map(img):
    sal = cv2.saliency.StaticSaliencyFineGrained_create()
    ok, s = sal.computeSaliency(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    if not ok:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, s = sal.computeSaliency(img)
    s = s.astype(np.float64)
    s = (s - s.min()) / (np.ptp(s) + 1e-8)
    return s

def foreground_mask(img, sal=None):
    from scipy import ndimage
    if sal is None:
        sal = saliency_map(img)
    h, w = sal.shape
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[sal > 0.55 * sal.max()] = cv2.GC_PR_FGD
    gc[sal > 0.80 * sal.max()] = cv2.GC_FGD
    gc[sal < 0.10 * sal.max()] = cv2.GC_BGD
    if not np.any(gc == cv2.GC_FGD):
        gc[sal >= sal.max() - 1e-6] = cv2.GC_FGD
    m_gc = np.zeros((h, w), bool)
    try:
        cv2.grabCut(bgr, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3,
                    cv2.GC_INIT_WITH_MASK)
        m_gc = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    except cv2.error:
        pass

    m_sal = sal > max(sal.mean(), 0.3 * sal.max())

    egr = gradient_energy(img)
    egr_n = (egr - egr.min()) / (np.ptp(egr) + 1e-8)
    edges = (egr_n > 0.25).astype(np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m_fill = ndimage.binary_fill_holes(edges).astype(bool)

    mask = (m_gc | m_sal | m_fill).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = ndimage.binary_fill_holes(mask).astype(np.float64)

    lbl, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(mask), lbl, range(1, n + 1))
        keep = set(np.argsort(sizes)[::-1][:5] + 1)
        big = np.where(sizes >= 0.01 * h * w)[0] + 1
        keep |= set(big.tolist())
        mask = np.isin(lbl, list(keep)).astype(np.float64)

    if mask.sum() < 0.005 * h * w:
        mask = (sal > 0.5 * sal.max()).astype(np.float64)
    return mask


def coseg_importance(img, alpha=3.0):
    egr = gradient_energy(img)
    egr_n = (egr - egr.min()) / (np.ptp(egr) + 1e-8)
    seg = foreground_mask(img)
    et = egr_n + alpha * seg
    return et, seg


def importance_for(img, comp_type, alpha=3.0):
    egr = gradient_energy(img)
    egr_n = (egr - egr.min()) / (np.ptp(egr) + 1e-8)
    sal = saliency_map(img)
    seg = foreground_mask(img, sal)

    if comp_type in ("thirds", "central"):
        F = seg 
    else:
        F = np.maximum(sal, seg)

    et = egr_n + alpha * F
    return et, seg


def foreground_center(mask):
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def foreground_bbox(mask):
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
