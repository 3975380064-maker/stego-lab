# -*- coding: utf-8 -*-
"""Statistical adequacy of the pilot:
  (1) repeated random splits -> mean/std/95% interval for every AUC
  (2) learning curve           -> does AUC plateau before the max train size?"""
import os
import numpy as np
from experiment import (load_natural, make_synthetic, lsb_match_embed, extract_matrix,
                        SIZE, EMBED_RATE, SEED, OUT)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

CACHE = os.path.join(OUT, "cache_feats.npz")
N = 400
R = 40           # repeated splits
TEST_RATIO = 0.3


def build_cache():
    print("[*] computing features (one-off) ...", flush=True)
    nat = load_natural(N)
    syn, _ = make_synthetic(N, seed=SEED + 1)
    rng_n = np.random.default_rng(SEED + 2)
    rng_s = np.random.default_rng(SEED + 3)
    nat_st = [lsb_match_embed(x, EMBED_RATE, rng_n) for x in nat]
    syn_st = [lsb_match_embed(x, EMBED_RATE, rng_s) for x in syn]
    d = dict(Xn_c=extract_matrix(nat), Xn_s=extract_matrix(nat_st),
             Xs_c=extract_matrix(syn), Xs_s=extract_matrix(syn_st))
    np.savez_compressed(CACHE, **d)
    return d["Xn_c"], d["Xn_s"], d["Xs_c"], d["Xs_s"]


def get_feats():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["Xn_c"], d["Xn_s"], d["Xs_c"], d["Xs_s"]
    return build_cache()


def auc_once(Xc, Xs, seed, ntrain=None):
    rng = np.random.default_rng(seed)
    n = len(Xc)
    perm = rng.permutation(n)
    ntest = int(n * TEST_RATIO)
    te, tr = perm[:ntest], perm[ntest:]
    if ntrain is not None and ntrain < len(tr):
        tr = rng.permutation(tr)[:ntrain]
    Xtr = np.vstack([Xc[tr], Xs[tr]]); ytr = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
    Xte = np.vstack([Xc[te], Xs[te]]); yte = np.r_[np.zeros(len(te)), np.ones(len(te))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def report(name, Xc, Xs, R=R):
    a = np.array([auc_once(Xc, Xs, 1000 + i) for i in range(R)])
    lo, hi = np.percentile(a, [2.5, 97.5])
    print("%-28s mean=%.4f  std=%.4f  95%%=[%.4f, %.4f]  n_test=%d"
          % (name, a.mean(), a.std(), lo, hi, int(len(Xc) * TEST_RATIO)))
    return a


def learning_curve(name, Xc, Xs, sizes, R=15):
    print("\nlearning curve - %s (test set fixed at 30%%, max train=%d):" % (name, int(len(Xc) * (1 - TEST_RATIO))))
    for m in sizes:
        a = np.array([auc_once(Xc, Xs, 5000 + i, ntrain=m) for i in range(R)])
        print("   n_train=%3d   AUC = %.4f +- %.4f" % (m, a.mean(), a.std()))


def main():
    Xn_c, Xn_s, Xs_c, Xs_s = get_feats()
    print("feature dim:", Xn_c.shape[1], " n_per_class:", len(Xn_c), " repeats:", R)
    print("\n=== repeated random splits (70/30) ===")
    report("D_nat  (nat clean vs stego)", Xn_c, Xn_s)
    report("D_syn  (syn clean vs stego)", Xs_c, Xs_s)
    report("D_carrier (nat vs syn, clean)", Xn_c, Xs_c)
    learning_curve("D_nat", Xn_c, Xn_s, [50, 100, 200, 280])
    learning_curve("D_syn", Xs_c, Xs_s, [50, 100, 200, 280])


if __name__ == "__main__":
    main()