import os, time, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cair
from make_inputs import build

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


def targets_for(item):
    h, w, comp, ch = item["h"], item["w"], item["comp"], item["change"]
    if comp == "horizontal":                 # reduce height
        return w, int(round(h * (1 - ch)))
    else:                                    # reduce width
        return int(round(w * (1 - ch))), h


def run_one(item):
    img = cv2.cvtColor(cv2.imread(item["path"]), cv2.COLOR_BGR2RGB)
    tw, th = targets_for(item)
    res = {"name": item["name"], "comp": item["comp"],
           "orig_size": f"{item['w']}x{item['h']}",
           "target_size": f"{tw}x{th}", "change": item["change"]}

    res["detected"] = cair.detect_composition(img)

    t = time.time()
    cut, cut_kept = cair.cutting_crop_scale(img, tw, th)
    res["t_cut"] = time.time() - t
    qc, dc = cair.quality_index(img, cut, cut_kept)
    res["q_cut"] = qc

    t = time.time()
    sc, sc_kept = cair.seam_carve_baseline(img, tw, th)
    res["t_sc"] = time.time() - t
    qs, ds = cair.quality_index(img, sc, sc_kept)
    res["q_sc"] = qs
    res["sc_detail"] = ds

    t = time.time()
    prop, prop_kept, info = cair.resize(img, tw, th, comp_type=item["comp"])
    res["t_prop"] = time.time() - t
    qp, dp = cair.quality_index(img, prop, prop_kept)
    res["q_prop"] = qp
    res["prop_detail"] = dp
    res["prop_info"] = {k: (v if not isinstance(v, tuple) else list(v))
                        for k, v in info.items()}

    imgs = [img, cut, sc, prop]
    labels = ["(a) original", "(b) cutting [22]", "(c) SC [4]",
              "(d) proposed"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, im, lb in zip(axes, imgs, labels):
        ax.imshow(im); ax.set_title(lb, fontsize=12); ax.axis("off")
    fig.suptitle(f"{item['name']}  |  {item['comp']} composition  |  "
                 f"{item['w']}x{item['h']} -> {tw}x{th}", fontsize=13)
    fig.tight_layout()
    p = os.path.join(OUT, f"compare_{item['name']}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    res["panel"] = p

    cv2.imwrite(os.path.join(OUT, f"proposed_{item['name']}.png"),
                cv2.cvtColor(prop, cv2.COLOR_RGB2BGR))
    return res


def main():
    items = build(os.path.join(OUT, "inputs"))
    results = []
    for it in items:
        print(f"=== {it['name']} ({it['comp']}) ===", flush=True)
        r = run_one(it)
        results.append(r)
        print(f"  detected={r['detected']} (gt={it['comp']})")
        print(f"  Q: cutting={r['q_cut']:.3f}  SC={r['q_sc']:.3f}  "
              f"proposed={r['q_prop']:.3f}")
        print(f"  time(s): cut={r['t_cut']:.1f} sc={r['t_sc']:.1f} "
              f"prop={r['t_prop']:.1f}", flush=True)

    print("\n================ TABLE 1 (Quality Index) ================")
    print(f"{'Image':22s}{'Change':>8s}{'SC':>9s}{'Proposed':>11s}")
    rows = []
    for r in results:
        print(f"{r['name']:22s}{int(r['change']*100):>7d}%"
              f"{r['q_sc']:>9.3f}{r['q_prop']:>11.3f}")
        rows.append([r['name'], f"{int(r['change']*100)}%",
                     round(r['q_sc'], 3), round(r['q_prop'], 3)])

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "table1.csv"), "w") as f:
        f.write("image,change,SC,Proposed\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    print("\nsaved -> outputs/results.json, outputs/table1.csv")
    return results


if __name__ == "__main__":
    main()
