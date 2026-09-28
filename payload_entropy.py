# -*- coding: utf-8 -*-
"""Does encrypting the payload (i.e. embedding high-entropy random bits) make it HARDER to detect?
Same covers, same embedding positions, same method (LSB matching, 0.4 bpp) - only payload bits differ."""
import numpy as np
from experiment import load_natural, extract_matrix, train_auc, SIZE, EMBED_RATE, SEED

N = 400

def embed_bits(cover, idx, bits, seed=0):
    flat = cover.reshape(-1).astype(np.int16)
    vals = flat[idx]
    need = (vals & 1) != bits
    rng = np.random.default_rng(seed)
    delta = rng.integers(0, 2, size=idx.size) * 2 - 1
    vals = np.clip(vals + np.where(need, delta, 0), 0, 255)
    flat[idx] = vals
    return flat.reshape(cover.shape).astype(np.uint8)

def text_bits(n, phrase=b"The quick brown fox jumps over the lazy dog. "):
    raw = np.frombuffer(phrase * (n // (8 * len(phrase)) + 2), dtype=np.uint8)
    return np.unpackbits(raw)[:n].astype(np.int64)

def main():
    nat = load_natural(N)
    nbits = int(EMBED_RATE * SIZE * SIZE)
    rng = np.random.default_rng(SEED)
    idxs = [rng.choice(SIZE * SIZE * 3, size=nbits, replace=False) for _ in nat]

    rnd = [rng.integers(0, 2, nbits).astype(np.int64) for _ in nat]      # encrypted-like (random)
    zer = [np.zeros(nbits, np.int64) for _ in nat]                        # constant (worst case)
    txt = [text_bits(nbits) for _ in nat]                                 # structured text

    Xc = extract_matrix(nat)
    out = {}
    for name, bits in [("random (encrypted-like)", rnd),
                       ("all-zero (constant)", zer),
                       ("text (structured)", txt)]:
        st = [embed_bits(c, ix, b, seed=7) for c, ix, b in zip(nat, idxs, bits)]
        auc = train_auc(Xc, extract_matrix(st), seed=202)
        out[name] = round(auc, 4)
        print("AUC  cover vs stego   payload=%-24s : %.4f" % (name, auc), flush=True)

    print("\npayload-entropy effect on detectability:", out)

if __name__ == "__main__":
    main()