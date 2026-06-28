import numpy as np
from .energy import gradient_energy

def _min_vertical_seam(cost):
    h, w = cost.shape
    M = cost.astype(np.float64).copy()
    back = np.zeros((h, w), np.int32)
    for i in range(1, h):
        left = np.r_[np.inf, M[i - 1, :-1]]
        up = M[i - 1, :]
        right = np.r_[M[i - 1, 1:], np.inf]
        choices = np.vstack([left, up, right])
        arg = np.argmin(choices, axis=0)
        back[i] = arg - 1 
        M[i] += np.min(choices, axis=0)
    seam = np.zeros(h, np.int32)
    seam[-1] = int(np.argmin(M[-1]))
    for i in range(h - 2, -1, -1):
        seam[i] = seam[i + 1] + back[i + 1, seam[i + 1]]
        seam[i] = max(0, min(w - 1, seam[i]))
    return seam

def _remove_seam_img(img, seam):
    h, w = img.shape[:2]
    if img.ndim == 3:
        out = np.zeros((h, w - 1, img.shape[2]), img.dtype)
        for i in range(h):
            j = seam[i]
            out[i] = np.delete(img[i], j, axis=0)
    else:
        out = np.zeros((h, w - 1), img.dtype)
        for i in range(h):
            out[i] = np.delete(img[i], seam[i])
    return out


def _insert_seam_img(img, seam):
    h, w = img.shape[:2]
    if img.ndim == 3:
        out = np.zeros((h, w + 1, img.shape[2]), img.dtype)
    else:
        out = np.zeros((h, w + 1), img.dtype)
    for i in range(h):
        j = seam[i]
        if img.ndim == 3:
            left = img[i, max(0, j - 1)]
            new_px = ((img[i, j].astype(np.float64) + left) / 2).astype(img.dtype)
            out[i] = np.insert(img[i], j + 1, new_px, axis=0) if j < w - 1 \
                else np.insert(img[i], j + 1, img[i, j], axis=0)
        else:
            new_px = img[i, j]
            out[i] = np.insert(img[i], j + 1, new_px)
    return out


def carve_vertical(img, protect, n, recompute=True):
    img = img.copy()
    protect = protect.astype(np.float64).copy()
    h, w0 = img.shape[:2]
    origin = np.tile(np.arange(w0, dtype=np.int32), (h, 1))

    if n == 0:
        return img, origin

    if n < 0:
        for _ in range(-n):
            base = gradient_energy(img) if recompute else 0.0
            cost = base + protect
            seam = _min_vertical_seam(cost)
            img = _remove_seam_img(img, seam)
            protect = _remove_seam_img(protect, seam)
            origin = _remove_seam_img(origin, seam)
        return img, origin

    tmp_img = img.copy()
    tmp_protect = protect.copy()
    tmp_origin = origin.copy()
    seams = []
    track = np.arange(w0)[None, :].repeat(h, 0)
    for _ in range(min(n, w0 - 1)):
        base = gradient_energy(tmp_img) if recompute else 0.0
        cost = base + tmp_protect
        seam = _min_vertical_seam(cost)
        # record original-column positions of this seam
        orig_cols = track[np.arange(h), seam]
        seams.append(orig_cols.copy())
        tmp_img = _remove_seam_img(tmp_img, seam)
        tmp_protect = _remove_seam_img(tmp_protect, seam)
        track = _remove_seam_img(track, seam)
    seam_cols = np.stack(seams, axis=0) if seams else np.zeros((0, h), int)  # (n,h)
    return _insert_batch(img, origin, seam_cols)


def _insert_batch(img, origin, seam_cols):
    h, w0 = img.shape[:2]
    n = seam_cols.shape[0]
    new_w = w0 + n
    if img.ndim == 3:
        out = np.zeros((h, new_w, img.shape[2]), img.dtype)
    else:
        out = np.zeros((h, new_w), img.dtype)
    out_origin = np.zeros((h, new_w), np.int32)
    for i in range(h):
        cols = sorted(seam_cols[:, i].tolist())
        row = list(img[i])
        orow = list(origin[i])
        for c in sorted(cols, reverse=True):
            c = int(min(max(c, 0), len(row) - 1))
            if img.ndim == 3:
                left = row[max(0, c - 1)]
                px = ((row[c].astype(np.float64) + left) / 2).astype(img.dtype)
            else:
                px = row[c]
            row.insert(c + 1, px)
            orow.insert(c + 1, orow[c])
        out[i] = np.array(row[:new_w])
        out_origin[i] = np.array(orow[:new_w])
    return out, out_origin


def carve_horizontal(img, protect, n, recompute=True):
    if img.ndim == 3:
        t = np.transpose(img, (1, 0, 2))
    else:
        t = img.T
    pt = protect.T
    out, origin_cols = carve_vertical(t, pt, n, recompute=recompute)
    if out.ndim == 3:
        out = np.transpose(out, (1, 0, 2))
    else:
        out = out.T
    return out, origin_cols.T


def seam_carve_baseline(img, target_w, target_h):
    h, w = img.shape[:2]
    zero = np.zeros((h, w))
    out, origin_cols = carve_vertical(img, zero, target_w - w, recompute=True)
    h2, w2 = out.shape[:2]
    zero2 = np.zeros((h2, w2))
    out2, origin_rows = carve_horizontal(out, zero2, target_h - h2, recompute=True)

    kept = np.zeros((h, w), bool)
    if target_w <= w:
        for i in range(min(h, origin_cols.shape[0])):
            kept[i, origin_cols[i]] = True
    else:
        kept[:] = True
    return out2, kept
