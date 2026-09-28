# -*- coding: utf-8 -*-
"""Sanity / falsification harness for the detector pipeline.
If any of these controls come out wrong, the main experiment is invalid."""
import numpy as np
from experiment import load_natural, extract_matrix, train_auc, lsb_match_embed, SEED

N = 400

def main():
    nat = load_natural(N)
    Xc = extract_matrix(nat)
    print("covers:", len(nat), " feature-dim:", Xc.shape[1])

    # control 1: identical images (rate = 0) -> must be ~0.5
    st0 = [x.copy() for x in nat]
    auc0 = train_auc(Xc, extract_matrix(st0), seed=300)
    print("CTRL1  rate=0 (no change)                AUC = %.4f  (expect ~0.50)" % auc0)

    # control 2: two disjoint halves of CLEAN covers -> must be ~0.5
    h = len(nat) // 2
    auc_c = train_auc(Xc[:h], Xc[h:2 * h], seed=301)
    print("CTRL2  clean-half vs clean-half          AUC = %.4f  (expect ~0.50)" % auc_c)

    # control 3: label shuffle inside train_auc? emulate by random rows
    perm = np.random.default_rng(302).permutation(h)
    auc_sh = train_auc(Xc[:h], Xc[h:2 * h][perm], seed=303)
    print("CTRL3  shuffled pairing (clean)          AUC = %.4f  (expect ~0.50)" % auc_sh)

    # rate sweep -> should rise monotonically from ~0.5
    print("\nrate sweep (LSB matching on natural covers):")
    for r in [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]:
        rng = np.random.default_rng(SEED + int(r * 10000))
        st = [lsb_match_embed(x, r, rng) for x in nat]
        auc = train_auc(Xc, extract_matrix(st), seed=400 + int(r * 100))
        print("   rate=%.2f bpp   AUC = %.4f" % (r, auc))

if __name__ == "__main__":
    main()