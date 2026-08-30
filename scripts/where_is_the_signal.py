"""Where does the signal actually live? Front end vs Transformer.

If the deterministic front end (masked instance norm, causal Gaussian split, patch statistics,
rate of change, density, circadian position) already carries the linearly-decodable signal, then
the contextual Transformer can only preserve or destroy it, and no training recipe will help.
"""

import numpy as np
import torch

from opencgm_stateevent.eval import labels
from opencgm_stateevent.eval.probe import HEADLINE, run_probe
from opencgm_stateevent.eval.splits import build_folds
from opencgm_stateevent.eval.windows import build_all
from opencgm_stateevent.model.model import OpenCGMStateEvent, architecture_of

SOURCES_FOR = {
    "cgmacros": ("cgmacros_dexcom", "cgmacros_libre"),
    "hall": ("hall",),
    "stanford": ("stanford",),
    "shanghai_t2dm": ("shanghai_t2dm",),
}
ws_all = build_all()
tables = labels.build_all()


def features(ckpt):
    st = torch.load(ckpt, map_location="cuda", weights_only=False)
    # architecture flags come from the checkpoint, not from today's defaults -- see embed.py
    mo = OpenCGMStateEvent(**architecture_of(st))
    mo.load_state_dict(st["model"])
    mo.cuda().eval()
    out = {}
    with torch.no_grad():
        for name, ws in ws_all.items():
            v = torch.from_numpy(ws.values).cuda()
            m = torch.from_numpy(ws.mask).cuda()
            c = torch.from_numpy(ws.circadian).cuda()
            r = mo.encode(v, m, c)
            out[name] = {
                "post_transformer": r.contextual_tokens.mean(1).float().cpu().numpy(),
                "pre_transformer": torch.cat([r.state_tokens, r.event_tokens], -1)
                .mean(1)
                .float()
                .cpu()
                .numpy(),
            }
    return out


def score(feats, method):
    s = []
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
            folds = build_folds(key, subs, sl, n_repeats=2)
            r = run_probe(
                folds,
                feats[src][method][keep],
                wsub,
                wl,
                task=key,
                method=method,
                n_classes=task.n_classes,
                cfg=HEADLINE,
            )
            v = r.scores("roc_auc")
            if len(v):
                s.append(float(v.mean()))
    return float(np.mean(s))


print(f"{'checkpoint':<26}{'pre-Transformer':>17}{'post-Transformer':>18}")
for lab, ck in [
    ("random init", "runs/fixed_seed17/ckpt_ep000.pt"),
    ("strict, epoch 10", "runs/fixed_seed17/ckpt_ep010.pt"),
    ("modern-stable, epoch 10", "runs/stable_seed17/ckpt_ep010.pt"),
]:
    f = features(ck)
    print(f"{lab:<26}{score(f, 'pre_transformer'):>17.4f}{score(f, 'post_transformer'):>18.4f}")
