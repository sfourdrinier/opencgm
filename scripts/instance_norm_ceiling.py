"""Does per-window instance normalisation impose the ceiling?

The model normalises each 24h window to zero mean and unit variance before anything else, so a
day centred at 105 mg/dL and an identically-shaped day at 205 mg/dL become the same input. If the
downstream phenotypes are carried mainly by absolute glucose level, no amount of pretraining can
recover it, and the ceiling is architectural rather than an optimisation failure.

Split the hand-engineered clinical features into the two groups and probe each alone.
"""

import numpy as np
import pandas as pd

from opencgm_stateevent.eval import baselines, labels
from opencgm_stateevent.eval.probe import HEADLINE, run_probe
from opencgm_stateevent.eval.splits import build_folds
from opencgm_stateevent.eval.windows import build_all

SOURCES_FOR = {
    "cgmacros": ("cgmacros_dexcom", "cgmacros_libre"),
    "hall": ("hall",),
    "stanford": ("stanford",),
    "shanghai_t2dm": ("shanghai_t2dm",),
}
# clinical_metrics column order (see eval/baselines.py)
ABSOLUTE = [0, 3, 4, 5, 6, 7, 8, 9, 13, 14]  # mean, max, min, range fractions, quartiles
SHAPE = [1, 2, 10, 11, 12, 15, 16]  # sd, CV, MAGE, roc mean/sd, density, n

ws_all = build_all()
tables = labels.build_all()
feat = {s: baselines.build("clinical_metrics", w) for s, w in ws_all.items()}


def score(cols, per_task=False):
    out = {}
    for task in labels.TASKS:
        t = tables[task.dataset]
        for src in SOURCES_FOR[task.dataset]:
            ws = ws_all[src]
            ser = t.dropna(subset=[task.name]).set_index("entry")[task.name]
            keys = ws.entries if ws.entries is not None else ws.subjects
            keep = np.array([k in ser.index for k in keys])
            if not keep.any():
                continue
            wl = ser.loc[list(keys[keep])].to_numpy().astype(int)
            wsub = np.asarray(ws.subjects[keep])
            subs, first = np.unique(wsub, return_index=True)
            sl = wl[first]
            if len(np.unique(sl)) < 2:
                continue
            key = f"{task.key}[{src}]"
            r = run_probe(
                build_folds(key, subs, sl, n_repeats=2),
                feat[src][keep][:, cols],
                wsub,
                wl,
                task=key,
                method="x",
                n_classes=task.n_classes,
                cfg=HEADLINE,
            )
            v = r.scores("roc_auc")
            if len(v):
                out[key] = float(v.mean())
    return out if per_task else float(np.mean(list(out.values())))


a = score(ABSOLUTE)
s = score(SHAPE)
both = score(ABSOLUTE + SHAPE)
print(f"{'absolute-level features only':<34}{a:.4f}")
print(f"{'shape-only features (norm-invariant)':<34}{s:.4f}")
print(f"{'both':<34}{both:.4f}")
print()
pa, ps = score(ABSOLUTE, True), score(SHAPE, True)
d = pd.DataFrame({"absolute": pa, "shape_only": ps})
d["loss_from_normalising"] = d.absolute - d.shape_only
print(d.sort_values("loss_from_normalising", ascending=False).round(4).to_string())
