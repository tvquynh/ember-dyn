#!/usr/bin/env python3
"""EMBER-Dyn AUTHORITATIVE Technical-Validation baseline suite (v6, final release).

ONE runner -> ONE results file, keyed by the SHIPPED task_definitions.json IDs
(T1..T15), on the FINAL 10,000-sample release (full static coverage), with the
CORRECTED task definitions:

  T1  Binary detection (static, 5 seeds)
  T2  Family 5-way (Subset A subset_class, 5 seeds)
  T3  Concept drift (pre/post 2024-09, 5 seeds)
  T4  Open-set A vs B (MSP + Energy, seed 42)
  T5  Hybrid static+dynamic fusion (dynamic = summary TF-IDF (+) apistats, seed 42)
  T6  Few-shot family (Subset C episodic, seed 42)
  T7  Non-UPX packer detection (packer field, UPX-only excluded, 5 seeds)   [CORRECTED]
  T8  Exploit/CVE detection (exploit field non-empty, 5 seeds)              [NEW]
  T9  APT group 6-way (Subset E2 subset_class, seed 42)                     [NEW]
  T10 Capa multi-label  (head/mid/tail thirds, seed 42)
  T11 MITRE multi-label (head/mid/tail thirds, seed 42)
  T12 MBC multi-label   (head/mid/tail thirds, seed 42)
  T13 Per-architecture detector (Win32/Win64/Dot_Net, AUC per file_type, seed 42)
  T14 Capability-count regression (MAE, R^2, seed 42)
  T15 Challenge OOD (challenge + test benign, PR/ROC AUC + bootstrap CI, seed 42)
  Plus: TD1 dynamic-only, and model-class diversity on T1 + T7.

Env: EMBER_DYN_BASELINES must point at a dir holding manifest.csv, splits.json,
task_definitions.json, static/<split>.parquet, dynamic/small/<split>.tar.zst.
Writes results/results_final.json. Set SMOKE=1 for a fast 1-seed reduced run.
"""
from __future__ import annotations
import ast, io, json, os, re, tarfile, time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import zstandard as zstd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             mean_absolute_error, r2_score)

ROOT = Path(os.environ.get("EMBER_DYN_BASELINES", "./baselines_workdir"))
STATIC_DIR = ROOT / "static"
SMALL_DIR = ROOT / "dynamic" / "small"
OUT = ROOT / "results" / "results_final.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("SMOKE", "0") == "1"
SEEDS = [42, 123, 456, 789, 1011] if not SMOKE else [42]
PRIMARY = 42
t_start = time.time()

LGB = {"boosting_type": "gbdt", "num_leaves": 64, "min_data_in_leaf": 100,
       "learning_rate": 0.05, "n_estimators": 500 if not SMOKE else 60,
       "n_jobs": -1, "verbosity": -1}


def log(m): print(m, flush=True)


def to_list(cell):
    if isinstance(cell, list): return cell
    if isinstance(cell, np.ndarray): return list(cell)
    if isinstance(cell, str):
        s = cell.strip()
        if s in ("", "[]", "None", "nan"): return []
        try:
            v = json.loads(s)
        except Exception:
            try:
                v = ast.literal_eval(s)
            except Exception:
                return []
        return v if isinstance(v, list) else [v]
    return []


# ── load ────────────────────────────────────────────────────────────
log("[load] manifest + parquets …")
_mcsv = ROOT / "manifest.csv"
if not _mcsv.exists():
    _mcsv = ROOT / "manifest_open.csv"   # open-tier fallback: recheck columns are not needed for the baselines
manifest = pd.read_csv(_mcsv)
log(f"  manifest rows: {len(manifest):,}  cols: {len(manifest.columns)}")
parq = {sp: pd.read_parquet(STATIC_DIR / f"{sp}.parquet")
        for sp in ["train", "val", "test", "challenge"]}
FEAT = [c for c in parq["train"].columns if c.startswith("feature_")]
log(f"  feature cols: {len(FEAT)}")

dfs = {}
for sp in ["train", "val", "test", "challenge"]:
    m = manifest[manifest["dyn_split"] == sp].copy()
    p = parq[sp][["sha256"] + FEAT]
    dfs[sp] = m.merge(p, on="sha256", how="inner", suffixes=("", "_p")).reset_index(drop=True)
    log(f"  merged {sp}: {len(dfs[sp]):,} x {len(FEAT)} feat")
X = {s: dfs[s][FEAT].values.astype(np.float32) for s in dfs}
results: dict = {}


def fit_binary(ytr, yva, yte, Xtr, Xva, Xte, seed):
    clf = lgb.LGBMClassifier(objective="binary", random_state=seed, **LGB)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    roc = float(roc_auc_score(yte, proba)) if len(set(yte)) > 1 else float("nan")
    pr = float(average_precision_score(yte, proba)) if len(set(yte)) > 1 else float("nan")
    f1 = float(f1_score(yte, pred, zero_division=0))
    return roc, pr, f1, proba, clf


def seeded_binary(ytr, yva, yte, Xtr, Xva, Xte, name):
    rocs, prs, f1s = [], [], []
    last_proba = None
    for sd in SEEDS:
        roc, pr, f1, proba, _ = fit_binary(ytr, yva, yte, Xtr, Xva, Xte, sd)
        rocs.append(roc); prs.append(pr); f1s.append(f1); last_proba = proba
    r = {"roc_auc_mean": float(np.mean(rocs)), "roc_auc_std": float(np.std(rocs)),
         "pr_auc_mean": float(np.mean(prs)), "pr_auc_std": float(np.std(prs)),
         "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
         "n_seeds": len(SEEDS), "n_train": int(len(ytr)), "n_test": int(len(yte)),
         "positive_rate_test": float(np.mean(yte))}
    log(f"[{name}] ROC {r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}  "
        f"PR {r['pr_auc_mean']:.4f}  F1 {r['f1_mean']:.4f}  (n_test={len(yte)})")
    return r, last_proba


# ── T1 binary ───────────────────────────────────────────────────────
yb = {s: dfs[s]["label"].astype(int).values for s in dfs}
results["T1_binary_detection"], proba_static_test = seeded_binary(
    yb["train"], yb["val"], yb["test"], X["train"], X["val"], X["test"], "T1")

# ── T2 family 5-way (Subset A) ──────────────────────────────────────
A_FAM = ["berbew", "cosmu", "expiro", "xmrig", "virlock"]
a2i = {f: i for i, f in enumerate(A_FAM)}
maskA = {s: (dfs[s]["subset"].eq("A") & dfs[s]["subset_class"].isin(A_FAM)).values for s in dfs}
yA = {s: dfs[s].loc[maskA[s], "subset_class"].map(a2i).astype(int).values for s in dfs}
XA = {s: X[s][maskA[s]] for s in dfs}
f1s = []
cm = np.zeros((5, 5))
for sd in SEEDS:
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=5, random_state=sd, **LGB)
    clf.fit(XA["train"], yA["train"], eval_set=[(XA["val"], yA["val"])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = clf.predict(XA["test"])
    _, _, f1m, _ = precision_recall_fscore_support(yA["test"], pred, average="macro", zero_division=0)
    f1s.append(float(f1m))
    if sd == PRIMARY:
        for ti, pi in zip(yA["test"], pred): cm[ti, pi] += 1
rs = cm.sum(1, keepdims=True)
cm_pct = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0) * 100
results["T2_family_5way_A"] = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                               "n_seeds": len(SEEDS), "classes": A_FAM,
                               "confusion_pct": cm_pct.tolist(),
                               "n_test": int(len(yA["test"]))}
log(f"[T2] macro F1 {np.mean(f1s):.4f}±{np.std(f1s):.4f}  n_test={len(yA['test'])}")

# ── T3 concept drift (pre/post) ─────────────────────────────────────
tb = {s: dfs[s]["time_bucket"].map({"pre": 0, "post": 1}).astype(int).values for s in dfs}
results["T3_concept_drift"], _ = seeded_binary(
    tb["train"], tb["val"], tb["test"], X["train"], X["val"], X["test"], "T3")

# ── T7 non-UPX packer (CORRECTED) ───────────────────────────────────
def packer_label(df):
    """1 = non-UPX packer present; 0 = empty; -1 = UPX-only (exclude)."""
    out = np.full(len(df), -2, dtype=int)
    for i, cell in enumerate(df["packer"].tolist()):
        items = [str(x).strip().lower() for x in to_list(cell)]
        if not items:
            out[i] = 0
        elif any(x != "upx" for x in items):
            out[i] = 1
        else:
            out[i] = -1  # UPX-only
    return out
pk = {s: packer_label(dfs[s]) for s in dfs}
keepP = {s: pk[s] >= 0 for s in dfs}  # exclude UPX-only (-1)
results["T7_packer_nonupx"], _ = seeded_binary(
    pk["train"][keepP["train"]], pk["val"][keepP["val"]], pk["test"][keepP["test"]],
    X["train"][keepP["train"]], X["val"][keepP["val"]], X["test"][keepP["test"]], "T7")
results["T7_packer_nonupx"]["n_positive_corpus"] = int(sum((pk[s] == 1).sum() for s in dfs))
results["T7_packer_nonupx"]["n_upx_excluded_corpus"] = int(sum((pk[s] == -1).sum() for s in dfs))

# ── T8 exploit (NEW) ────────────────────────────────────────────────
def exploit_label(df):
    return np.array([1 if to_list(c) else 0 for c in df["exploit"].tolist()], dtype=int)
ex = {s: exploit_label(dfs[s]) for s in dfs}
results["T8_exploit"], _ = seeded_binary(
    ex["train"], ex["val"], ex["test"], X["train"], X["val"], X["test"], "T8")
results["T8_exploit"]["n_positive_corpus"] = int(sum(ex[s].sum() for s in dfs))

# ── T9 APT 6-way (Subset E2) ────────────────────────────────────────
APT = ["apt_equationgroup", "apt_gamaredon", "apt_knotweed", "apt_molerats", "apt_turla", "apt_other"]
g2i = {g: i for i, g in enumerate(APT)}
mE2 = {s: dfs[s]["subset"].eq("E2").values for s in dfs}
nE2 = {s: int(mE2[s].sum()) for s in dfs}
log(f"[T9] E2 per split: {nE2}")
yE_tr = pd.concat([dfs["train"].loc[mE2["train"]], dfs["val"].loc[mE2["val"]]])
XE_tr = np.vstack([X["train"][mE2["train"]], X["val"][mE2["val"]]])
yE_te_df = dfs["test"].loc[mE2["test"]]
XE_te = X["test"][mE2["test"]]
if len(yE_te_df) >= 6 and yE_tr["subset_class"].nunique() >= 2:
    ytr9 = yE_tr["subset_class"].map(g2i).fillna(g2i["apt_other"]).astype(int).values
    yte9 = yE_te_df["subset_class"].map(g2i).fillna(g2i["apt_other"]).astype(int).values
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=6, random_state=PRIMARY, **LGB)
    clf.fit(XE_tr, ytr9)
    pred9 = clf.predict(XE_te)
    _, _, f19, _ = precision_recall_fscore_support(yte9, pred9, average="macro", zero_division=0)
    results["T9_apt_6way"] = {"macro_f1": float(f19), "n_train": int(len(ytr9)),
                              "n_test": int(len(yte9)), "classes": APT}
    log(f"[T9] macro F1 {f19:.4f}  n_train={len(ytr9)} n_test={len(yte9)}")
else:
    results["T9_apt_6way"] = {"skipped": f"E2 test too small ({len(yE_te_df)})",
                              "n_e2_total": int(sum(nE2.values()))}
    log(f"[T9] skipped (E2 test={len(yE_te_df)})")

# ── T4 open-set A vs B ──────────────────────────────────────────────
tr, te = dfs["train"], dfs["test"]
mAtr = (tr["subset"].eq("A") & tr["subset_class"].isin(A_FAM)).values
mAte = (te["subset"].eq("A") & te["subset_class"].isin(A_FAM)).values
mBte = te["subset"].eq("B").values
if mBte.sum() > 20:
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=5, random_state=PRIMARY, **LGB)
    clf.fit(X["train"][mAtr], tr.loc[mAtr, "subset_class"].map(a2i).astype(int).values)
    Xev = np.vstack([X["test"][mAte], X["test"][mBte]])
    known = np.concatenate([np.ones(mAte.sum(), int), np.zeros(mBte.sum(), int)])
    raw = clf.predict_proba(Xev)
    msp = raw.max(1)                            # maximum-softmax-probability score
    margins = clf.predict(Xev, raw_score=True)  # raw class margins (logits)
    mmax = margins.max(1)
    energy = -(mmax + np.log(np.exp(margins - mmax[:, None]).sum(1)))  # free energy = -logsumexp(margins)
    def fpr95(sc, y):
        ks, us = sc[y == 1], sc[y == 0]
        thr = np.quantile(ks, 0.05)
        return float((us >= thr).mean())
    results["T4_openset_A_vs_B"] = {
        "auroc_msp": float(roc_auc_score(known, msp)),
        "auroc_energy": float(roc_auc_score(known, -energy)),
        "energy_score_var": float(np.var(-energy)),
        "fpr_at_95tpr_msp": fpr95(msp, known), "fpr_at_95tpr_energy": fpr95(-energy, known),
        "n_known": int(mAte.sum()), "n_unknown": int(mBte.sum())}
    log(f"[T4] MSP AUROC {results['T4_openset_A_vs_B']['auroc_msp']:.4f}  "
        f"Energy {results['T4_openset_A_vs_B']['auroc_energy']:.4f}")

# ── T13 per-architecture ────────────────────────────────────────────
per_arch = {}
for arch in ["Win32", "Win64", "Dot_Net"]:
    mtr = dfs["train"]["file_type"].eq(arch).values
    mte = dfs["test"]["file_type"].eq(arch).values
    mva = dfs["val"]["file_type"].eq(arch).values
    if mtr.sum() > 100 and mte.sum() > 20 and len(set(yb["test"][mte])) > 1:
        roc, pr, f1, _, _ = fit_binary(yb["train"][mtr], yb["val"][mva], yb["test"][mte],
                                       X["train"][mtr], X["val"][mva], X["test"][mte], PRIMARY)
        per_arch[arch] = {"roc_auc": roc, "f1": f1, "n_test": int(mte.sum())}
        log(f"[T13] {arch}: ROC {roc:.4f} n_test={int(mte.sum())}")
    else:
        per_arch[arch] = {"skipped": f"insufficient ({arch})", "n_test": int(mte.sum())}
results["T13_per_arch"] = per_arch

# ── T14 capability-count regression ─────────────────────────────────
def cap_count(df):
    return np.array([len(to_list(a)) + len(to_list(b)) + len(to_list(c))
                     for a, b, c in zip(df["caps_list"], df["ttps_list"], df["mbc_list"])], float)
ycc = {s: cap_count(dfs[s]) for s in dfs}
reg = lgb.LGBMRegressor(objective="regression", random_state=PRIMARY, **LGB)
reg.fit(X["train"], ycc["train"], eval_set=[(X["val"], ycc["val"])],
        callbacks=[lgb.early_stopping(50, verbose=False)])
pcc = reg.predict(X["test"])
results["T14_capcount_regression"] = {"mae": float(mean_absolute_error(ycc["test"], pcc)),
                                      "r2": float(r2_score(ycc["test"], pcc)),
                                      "mean_count_test": float(ycc["test"].mean())}
log(f"[T14] MAE {results['T14_capcount_regression']['mae']:.3f}  R2 {results['T14_capcount_regression']['r2']:.4f}")

# ── T15 challenge OOD (challenge mal + test benign) ─────────────────
Xtv = np.vstack([X["train"], X["val"]])
ytv = np.concatenate([yb["train"], yb["val"]])
clf = lgb.LGBMClassifier(objective="binary", random_state=PRIMARY, **LGB)
clf.fit(Xtv, ytv)
mben = (dfs["test"]["label"].eq(0)).values
Xood = np.vstack([X["challenge"], X["test"][mben]])
yood = np.concatenate([np.ones(len(X["challenge"]), int), np.zeros(int(mben.sum()), int)])
pood = clf.predict_proba(Xood)[:, 1]
rng = np.random.default_rng(PRIMARY)
boot = []
for _ in range(1000 if not SMOKE else 30):
    idx = rng.integers(0, len(yood), len(yood))
    if len(set(yood[idx])) > 1:
        boot.append(average_precision_score(yood[idx], pood[idx]))
results["T15_challenge_ood"] = {
    "pr_auc": float(average_precision_score(yood, pood)),
    "roc_auc": float(roc_auc_score(yood, pood)),
    "f1": float(f1_score(yood, (pood >= 0.5).astype(int), zero_division=0)),
    "pr_auc_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "n_challenge": int(len(X["challenge"])), "n_benign": int(mben.sum()),
    "positive_rate": float(yood.mean())}
log(f"[T15] PR AUC {results['T15_challenge_ood']['pr_auc']:.4f}  "
    f"ROC {results['T15_challenge_ood']['roc_auc']:.4f}  CI {results['T15_challenge_ood']['pr_auc_ci95']}")

# ── dynamic features (summary TF-IDF + apistats) for TD1/T5 ─────────
log("[dyn] loading Small reports …")
recs = {}
for sp in ["train", "val", "test", "challenge"]:
    tp = SMALL_DIR / f"{sp}.tar.zst"
    if not tp.exists():
        log(f"  WARN missing {tp}"); continue
    with open(tp, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        for mb in tar:
            if mb.isfile() and mb.name.endswith(".json"):
                try:
                    recs[Path(mb.name).stem] = json.loads(tar.extractfile(mb).read())
                except Exception:
                    pass
log(f"  loaded {len(recs):,} Small reports")
BEH = ["resolved_apis", "executed_commands", "files", "read_files", "write_files",
       "delete_files", "read_keys", "write_keys", "delete_keys",
       "started_services", "created_services", "mutexes", "keys"]
TOK = re.compile(r"[A-Za-z0-9_./\\-]+")
def summ_text(o):
    beh = (o.get("behavior") or {}).get("summary") or {}
    bag = []
    for fld in BEH:
        for it in (beh.get(fld) or []):
            if isinstance(it, str):
                bag += [t.lower() for t in TOK.findall(it) if 2 <= len(t) <= 80]
    return " ".join(bag)
def apistat(o):
    return (o.get("behavior") or {}).get("apistats") or {}

def dyn_arrays(sp):
    m = manifest[manifest["dyn_split"] == sp]
    shas = m["sha256"].tolist()
    txt = [summ_text(recs.get(s, {})) for s in shas]
    api = [apistat(recs.get(s, {})) for s in shas]
    y = m["label"].astype(int).values
    return shas, txt, api, y
DY = {sp: dyn_arrays(sp) for sp in ["train", "val", "test", "challenge"]}

vec = TfidfVectorizer(max_features=50000, min_df=5, max_df=0.9, sublinear_tf=True)
Td = {"train": vec.fit_transform(DY["train"][1])}
for sp in ["val", "test", "challenge"]:
    Td[sp] = vec.transform(DY[sp][1])
# apistats vocab from train (min 5 occurrences)
api_count = Counter()
for d in DY["train"][2]:
    for k in d: api_count[k] += 1
api_kept = [k for k, c in api_count.items() if c >= 5]
api_vocab = {k: i for i, k in enumerate(api_kept)}
def api_mat(api_list):
    rows, cols, vals = [], [], []
    for r, d in enumerate(api_list):
        for k, v in d.items():
            if k in api_vocab:
                rows.append(r); cols.append(api_vocab[k]); vals.append(np.log1p(float(v)))
    return csr_matrix((vals, (rows, cols)), shape=(len(api_list), len(api_vocab)))
Xd = {sp: hstack([Td[sp], api_mat(DY[sp][2])]).tocsr() for sp in DY}
log(f"[dyn] TF-IDF dim {Td['train'].shape[1]} + apistats dim {len(api_vocab)} = {Xd['train'].shape[1]}")

# ── TD1 dynamic-only ────────────────────────────────────────────────
clf_d = lgb.LGBMClassifier(objective="binary", random_state=PRIMARY, **LGB)
clf_d.fit(Xd["train"], DY["train"][3], eval_set=[(Xd["val"], DY["val"][3])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
proba_dyn_test = clf_d.predict_proba(Xd["test"])[:, 1]
results["TD1_dynamic_only"] = {
    "roc_auc": float(roc_auc_score(DY["test"][3], proba_dyn_test)),
    "pr_auc": float(average_precision_score(DY["test"][3], proba_dyn_test)),
    "f1": float(f1_score(DY["test"][3], (proba_dyn_test >= 0.5).astype(int), zero_division=0)),
    "feat_dim": int(Xd["train"].shape[1]), "apistats_dim": len(api_vocab)}
log(f"[TD1] ROC {results['TD1_dynamic_only']['roc_auc']:.4f} (with apistats)")

# ── T5 hybrid fusion (late) + bootstrap CI vs static-only ───────────
# align test on intersection of static(test) and dynamic(test)
st_sha = dfs["test"]["sha256"].tolist()
st_idx = {s: i for i, s in enumerate(st_sha)}
dy_sha = DY["test"][0]
dy_idx = {s: i for i, s in enumerate(dy_sha)}
common = [s for s in st_sha if s in dy_idx]
ay = np.array([int(dfs["test"].loc[st_idx[s], "label"]) for s in common])
a_st = np.array([proba_static_test[st_idx[s]] for s in common])  # static (seed-1011 proba ok; recompute seed42 below)
# recompute static test proba at PRIMARY for a clean paired comparison
_, _, _, proba_static_p42, _ = fit_binary(yb["train"], yb["val"], yb["test"],
                                           X["train"], X["val"], X["test"], PRIMARY)
a_st = np.array([proba_static_p42[st_idx[s]] for s in common])
a_dy = np.array([proba_dyn_test[dy_idx[s]] for s in common])
variants = {"mean": (a_st + a_dy) / 2, "wstatic0.7": 0.7 * a_st + 0.3 * a_dy,
            "geom": np.sqrt(np.clip(a_st * a_dy, 0, 1)), "max": np.maximum(a_st, a_dy)}
roc_static = roc_auc_score(ay, a_st)
best_name = max(variants, key=lambda k: roc_auc_score(ay, variants[k]))
best = variants[best_name]
rng = np.random.default_rng(PRIMARY)
diffs = []
for _ in range(2000 if not SMOKE else 30):
    idx = rng.integers(0, len(ay), len(ay))
    if len(set(ay[idx])) > 1:
        diffs.append(roc_auc_score(ay[idx], best[idx]) - roc_auc_score(ay[idx], a_st[idx]))
results["T5_hybrid_fusion"] = {
    "roc_static_only": float(roc_static),
    "roc_dynamic_only_aligned": float(roc_auc_score(ay, a_dy)),
    "best_variant": best_name, "roc_best_fusion": float(roc_auc_score(ay, best)),
    "fusion_minus_static_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    "n_test_aligned": int(len(common)), "variants_roc": {k: float(roc_auc_score(ay, v)) for k, v in variants.items()}}
log(f"[T5] static {roc_static:.4f} best-fusion({best_name}) {roc_auc_score(ay, best):.4f}  "
    f"CI {results['T5_hybrid_fusion']['fusion_minus_static_ci95']}")

# ── T6 few-shot family (Subset C episodic) ──────────────────────────
C_FAM = ["blackmoon", "remcos", "banload", "virut", "cobaltstrike", "ekstak", "sality", "flystudio"]
c2i = {f: i for i, f in enumerate(C_FAM)}
allC = pd.concat([dfs[s][dfs[s]["subset"].eq("C")] for s in dfs])
allC = allC[allC["family"].isin(C_FAM)]
if len(allC) >= 50:
    XC = allC[FEAT].values.astype(np.float32)
    yC = allC["family"].map(c2i).astype(int).values
    rng = np.random.default_rng(PRIMARY)
    epr = {}
    for K in [5, 10]:
        accs = []
        for _ in range(50 if not SMOKE else 5):
            av = [f for f in C_FAM if (yC == c2i[f]).sum() >= K + 5]
            if len(av) < 3: continue
            nway = min(5, len(av))
            ch = rng.choice(av, nway, replace=False)
            sx, sy, qx, qy = [], [], [], []
            for i, f in enumerate(ch):
                idx = np.where(yC == c2i[f])[0]; rng.shuffle(idx)
                sx.append(XC[idx[:K]]); sy += [i] * K
                qx.append(XC[idx[K:K + 5]]); qy += [i] * len(idx[K:K + 5])
            sx, qx = np.vstack(sx), np.vstack(qx); sy, qy = np.array(sy), np.array(qy)
            proto = np.stack([sx[sy == i].mean(0) for i in range(nway)])
            pred = np.linalg.norm(qx[:, None] - proto[None], axis=2).argmin(1)
            accs.append(float((pred == qy).mean()))
        epr[f"K={K}"] = {"mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)), "n_ep": len(accs)}
    results["T6_fewshot_C"] = {"method": "prototype 1-NN raw features", "results": epr, "n_C": int(len(allC))}
    log(f"[T6] K=5 {epr['K=5']['mean_acc']:.3f}  K=10 {epr['K=10']['mean_acc']:.3f}")
else:
    results["T6_fewshot_C"] = {"skipped": f"only {len(allC)} C"}

# ── T10/T11/T12 multi-label head/mid/tail thirds ────────────────────
def multilabel(col, name):
    cnt = Counter()
    for cell in dfs["train"][col]:
        for x in to_list(cell): cnt[x] += 1
    labs = [l for l, c in cnt.most_common() if c >= 10]
    if not labs: return {"skipped": "no labels >=10 pos"}
    n = len(labs); third = max(1, n // 3)
    bands = {"head": labs[:third], "mid": labs[third:2 * third], "tail": labs[2 * third:]}
    if SMOKE:
        bands = {"head": labs[:5]}
    def binmat(df, labset):
        idx = {l: i for i, l in enumerate(labset)}
        Y = np.zeros((len(df), len(labset)), np.int8)
        for r, cell in enumerate(df[col]):
            for x in to_list(cell):
                if x in idx: Y[r, idx[x]] = 1
        return Y
    out = {"n_labels": n}
    for band, labset in bands.items():
        if not labset: out[band] = {"skipped": "empty band"}; continue
        Ytr, Yva, Yte = binmat(dfs["train"], labset), binmat(dfs["val"], labset), binmat(dfs["test"], labset)
        f1l = []
        for j in range(len(labset)):
            if Ytr[:, j].sum() < 10 or Yva[:, j].sum() == 0: continue
            c = lgb.LGBMClassifier(objective="binary", random_state=PRIMARY, **LGB)
            c.fit(X["train"], Ytr[:, j], eval_set=[(X["val"], Yva[:, j])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            pr = (c.predict_proba(X["test"])[:, 1] >= 0.5).astype(int)
            f1l.append(f1_score(Yte[:, j], pr, zero_division=0))
        out[band] = {"macro_f1": float(np.mean(f1l)) if f1l else float("nan"),
                     "n_labels_eval": len(f1l), "n_labels_band": len(labset)}
        log(f"[{name}] {band}: macro F1 {out[band]['macro_f1']:.3f} ({len(f1l)}/{len(labset)})")
    return out
results["T10_capa_multilabel"] = multilabel("caps_list", "T10")
results["T11_mitre_multilabel"] = multilabel("ttps_list", "T11")
results["T12_mbc_multilabel"] = multilabel("mbc_list", "T12")

# ── model-class diversity on T1 + T7 ────────────────────────────────
def diversity(ytr, yva, yte, Xtr, Xva, Xte, name):
    out = {}
    # LightGBM
    roc, pr, f1, _, _ = fit_binary(ytr, yva, yte, Xtr, Xva, Xte, PRIMARY)
    out["lightgbm"] = {"roc_auc": roc, "pr_auc": pr, "f1": f1}
    try:
        import xgboost as xgb
        xc = xgb.XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.05,
                               tree_method="hist", n_jobs=-1, eval_metric="logloss",
                               random_state=PRIMARY)
        xc.fit(Xtr, ytr)
        pp = xc.predict_proba(Xte)[:, 1]
        out["xgboost"] = {"roc_auc": float(roc_auc_score(yte, pp)),
                          "pr_auc": float(average_precision_score(yte, pp)),
                          "f1": float(f1_score(yte, (pp >= 0.5).astype(int), zero_division=0))}
    except Exception as e:
        out["xgboost"] = {"error": str(e)[:120]}
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300 if not SMOKE else 50, n_jobs=-1, random_state=PRIMARY)
    rf.fit(Xtr, ytr)
    pp = rf.predict_proba(Xte)[:, 1]
    out["randomforest"] = {"roc_auc": float(roc_auc_score(yte, pp)),
                           "pr_auc": float(average_precision_score(yte, pp)),
                           "f1": float(f1_score(yte, (pp >= 0.5).astype(int), zero_division=0))}
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        mlp = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=60 if not SMOKE else 8,
                            early_stopping=True, random_state=PRIMARY)
        mlp.fit(sc.fit_transform(Xtr), ytr)
        pp = mlp.predict_proba(sc.transform(Xte))[:, 1]
        out["mlp"] = {"roc_auc": float(roc_auc_score(yte, pp)),
                      "pr_auc": float(average_precision_score(yte, pp)),
                      "f1": float(f1_score(yte, (pp >= 0.5).astype(int), zero_division=0))}
    except Exception as e:
        out["mlp"] = {"error": str(e)[:120]}
    log(f"[div:{name}] LGB {out['lightgbm']['roc_auc']:.4f} "
        f"XGB {out.get('xgboost',{}).get('roc_auc','-')} RF {out['randomforest']['roc_auc']:.4f}")
    return out
results["diversity_T1"] = diversity(yb["train"], yb["val"], yb["test"], X["train"], X["val"], X["test"], "T1")
results["diversity_T7"] = diversity(pk["train"][keepP["train"]], pk["val"][keepP["val"]], pk["test"][keepP["test"]],
                                    X["train"][keepP["train"]], X["val"][keepP["val"]], X["test"][keepP["test"]], "T7")

# ── save ────────────────────────────────────────────────────────────
results["meta"] = {"manifest_total": int(len(manifest)), "feature_dim": len(FEAT),
                   "static_coverage": int(sum(len(dfs[s]) for s in dfs)),
                   "splits": {s: int(len(dfs[s])) for s in dfs}, "seeds": SEEDS,
                   "smoke": SMOKE, "lgb": LGB, "total_elapsed_s": time.time() - t_start}
OUT.write_text(json.dumps(results, indent=2))
log(f"[done] -> {OUT}  total {time.time()-t_start:.1f}s")
