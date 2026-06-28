import os
import numpy as np
import cv2
from skimage import data


def synth_landscape(w=600, h=400):
    img = np.zeros((h, w, 3), np.uint8)
    horizon = int(h * 0.45)
    # sky gradient (top)
    for y in range(horizon):
        t = y / horizon
        img[y, :] = (int(135 + 80 * t), int(180 + 50 * t), int(235))
    # sun
    cv2.circle(img, (int(w * 0.72), int(horizon * 0.4)), 26, (180, 240, 255), -1)
    pts = np.array([[0, horizon], [int(w*0.2), int(horizon*0.55)],
                    [int(w*0.4), int(horizon*0.8)], [int(w*0.6), int(horizon*0.5)],
                    [int(w*0.85), int(horizon*0.75)], [w, int(horizon*0.6)],
                    [w, horizon], [0, horizon]], np.int32)
    cv2.fillPoly(img, [pts], (70, 110, 90))
    top = img[:horizon][::-1]
    bh = h - horizon
    water = cv2.resize(top, (w, bh), interpolation=cv2.INTER_LINEAR).astype(np.float64) * 0.6
    water = water.astype(np.uint8)
    water[:, :, 0] = np.clip(water[:, :, 0] + 30, 0, 255)  
    img[horizon:horizon + bh] = water
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def synth_symmetric(w=600, h=420):
    img = np.full((h, w, 3), (200, 225, 245), np.uint8) 
    # ground
    cv2.rectangle(img, (0, int(h*0.72)), (w, h), (150, 170, 185), -1)
    pw = int(w * 0.10)
    for cx in (int(w * 0.22), int(w * 0.78)):
        cv2.rectangle(img, (cx - pw // 2, int(h*0.12)),
                      (cx + pw // 2, int(h*0.80)), (90, 110, 130), -1)
        cv2.rectangle(img, (cx - pw // 2 - 6, int(h*0.10)),
                      (cx + pw // 2 + 6, int(h*0.14)), (70, 90, 110), -1)
    cv2.rectangle(img, (int(w*0.47), int(h*0.45)), (int(w*0.53), int(h*0.80)),
                  (60, 80, 160), -1)
    cv2.circle(img, (w // 2, int(h*0.42)), 20, (40, 60, 150), -1)
    cv2.rectangle(img, (int(w*0.16), int(h*0.10)), (int(w*0.84), int(h*0.16)),
                  (110, 130, 150), -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build(outdir):
    os.makedirs(outdir, exist_ok=True)
    items = []

    def save(name, img, comp, change):
        p = os.path.join(outdir, name + ".png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        items.append(dict(name=name, path=p, comp=comp, change=change,
                          h=img.shape[0], w=img.shape[1]))

    save("thirds_cat", data.chelsea(), "thirds", 0.25)
    save("central_astronaut", data.astronaut(), "central", 0.50)
    save("horizontal_rocket", data.rocket(), "horizontal", 0.25)
    save("symmetric_coffee", data.coffee(), "symmetric", 0.40)
    return items


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    items = build(os.path.join(here, "outputs", "inputs"))
    for it in items:
        print(it)
