import os, time, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import data

import cair

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

base = data.astronaut()
scales = [96, 128, 192, 256, 320, 384, 448, 512]
FRAC = 0.20

rows = []
for s in scales:
    img = cv2.resize(base, (s, s), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    k = int(round(w * FRAC))
    tw = w - k
    work = w * h * k

    t = time.time()
    out, _ = cair.seam_carve_baseline(img, tw, h)
    dt = time.time() - t

    rows.append(dict(edge=s, W=w, H=h, seams=k, work=work, t_sc=dt))
    print(f"edge={s:4d}  W*H*k={work:>12,d}  seams={k:3d}  t_sc={dt:7.3f}s",
          flush=True)

work = np.array([r["work"] for r in rows], float)
tsc = np.array([r["t_sc"] for r in rows], float)
A = np.vstack([work, np.ones_like(work)]).T
(a, b), *_ = np.linalg.lstsq(A, tsc, rcond=None)
pred = a * work + b
ss_res = np.sum((tsc - pred) ** 2)
ss_tot = np.sum((tsc - tsc.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"\nlinear fit  t = {a:.3e} * (W*H*k) + {b:.3e}   R^2 = {r2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
edges = [r["edge"] for r in rows]
axes[0].plot(edges, tsc, "o-", color="#3060c0")
axes[0].set_xlabel("image edge length (px)")
axes[0].set_ylabel("SC time (s)")
axes[0].set_title("(a) runtime vs. image size\n(20% width reduction)")
axes[0].grid(alpha=.3)

axes[1].plot(work, tsc, "o", color="#c04030", label="measured")
xs = np.linspace(0, work.max(), 100)
axes[1].plot(xs, a * xs + b, "--", color="#555",
             label=f"linear fit  R²={r2:.3f}")
axes[1].set_xlabel("work  =  W · H · k")
axes[1].set_ylabel("SC time (s)")
axes[1].set_title("(b) runtime vs. predicted work")
axes[1].legend()
axes[1].grid(alpha=.3)
fig.tight_layout()
p = os.path.join(OUT, "scalability.png")
fig.savefig(p, dpi=120, bbox_inches="tight")
plt.close(fig)

json.dump({"frac": FRAC, "rows": rows, "fit_a": a, "fit_b": b, "r2": r2},
          open(os.path.join(OUT, "scalability.json"), "w"), indent=2)
print(f"saved -> {p}, outputs/scalability.json")
