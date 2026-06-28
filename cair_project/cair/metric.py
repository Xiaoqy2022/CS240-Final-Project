import numpy as np
import cv2
from .energy import saliency_map, foreground_mask, foreground_bbox


def information_loss(orig, kept_mask):
    s = saliency_map(orig)
    s = s / (s.sum() + 1e-8)
    lost = s[~kept_mask].sum()
    return float(np.clip(lost, 0, 1))


def geometric_distortion(orig, result):
    so = foreground_mask(orig)
    sr = foreground_mask(result)
    bo = foreground_bbox(so)
    br = foreground_bbox(sr)
    if bo is None or br is None:
        return 0.0
    aw = (bo[2] - bo[0]) / max(1, (bo[3] - bo[1]))
    rw = (br[2] - br[0]) / max(1, (br[3] - br[1]))
    gd = abs(np.log((rw + 1e-6) / (aw + 1e-6)))
    return float(np.clip(gd, 0, 1))


def quality_index(orig, result, kept_mask, w_il=0.5, w_gd=0.5):
    il = information_loss(orig, kept_mask)
    gd = geometric_distortion(orig, result)
    q = 1.0 - np.clip(w_il * il + w_gd * gd, 0, 1)
    return float(q), dict(information_loss=il, geometric_distortion=gd)
