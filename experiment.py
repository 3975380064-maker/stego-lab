# -*- coding: utf-8 -*-
"""
stego-lab / pilot-1: synthetic(Kakeya/fractal) vs natural carriers
Decisive question: is a synthetic carrier *more* or *less* detectable for the SAME embedding?

Design fix vs naive protocol:
  - naive: train ONE detector on mixed (natural+synthetic) covers/stegos -> it mostly
    learns "natural vs synthetic" (carrier realism), AUC~1, tells us nothing about embedding.
  - here : train PER-CLASS detectors (cover vs stego WITHIN a class) and compare AUCs.
           plus D_carrier (natural vs synthetic clean) to expose the free feature.

Everything is reproducible: fixed seeds, prompt-controlled RNG.
"""
import os, io, math, json, sys, time
import numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

SIZE = 256
OUT = os.environ.get("STEGOLAB_DIR", os.path.dirname(os.path.abspath(__file__)))
NAT_DIR = os.path.join(OUT, "natural")
N_PER_CLASS = 400
EMBED_RATE = 0.4          # bits per pixel
SEED = 20260928
T = 3                     # residual truncation (SPAM)
PREP_JPEG_SYN = False     # if True, jpeg-q95 pass on synthetics to mimic camera pipeline


# ---------------------------------------------------------------- natural covers
def download_natural(n):
    os.makedirs(NAT_DIR, exist_ok=True)

    def fetch(i):
        p = os.path.join(NAT_DIR, "%04d.jpg" % i)
        if os.path.exists(p) and os.path.getsize(p) > 1500:
            return True
        url = "https://picsum.photos/seed/%d/%d/%d" % (i, SIZE, SIZE)
        for _ in range(3):
            try:
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=25) as r:
                    data = r.read()
                if len(data) > 1500:
                    open(p, "wb").write(data)
                    return True
            except Exception:
                pass
        return False

    with ThreadPoolExecutor(max_workers=16) as ex:
        ok = list(ex.map(fetch, range(n)))
    return sum(ok)


def load_natural(n):
    arrs = []
    for i in range(n):
        p = os.path.join(NAT_DIR, "%04d.jpg" % i)
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE))
        arrs.append(np.asarray(im, dtype=np.uint8))
    return arrs


# ---------------------------------------------------------------- synthetic covers
def gen_kakeya(rng):
    """Besicovitch / 'Kakeya-style' star: unit segments through a common center, all directions."""
    img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    d = ImageDraw.Draw(img)
    c = SIZE / 2.0
    nang = int(rng.integers(24, 56))
    for k in range(nang):
        ang = math.pi * k / nang + rng.normal(0, 0.01)
        L = SIZE * (0.45 + 0.45 * rng.random())
        ox, oy = rng.normal(0, SIZE * 0.02, 2)
        x0, y0 = c + ox - L * math.cos(ang), c + oy - L * math.sin(ang)
        x1, y1 = c + ox + L * math.cos(ang), c + oy + L * math.sin(ang)
        col = tuple(int(rng.integers(90, 256)) for _ in range(3))
        d.line([x0, y0, x1, y1], fill=col, width=int(rng.integers(1, 3)))
    return np.asarray(img, dtype=np.uint8)


def gen_sierpinski(rng, n=40000):
    """IFS chaos-game Sierpinski, rasterized to density, then tinted."""
    x = y = 0.0
    xs = np.empty(n); ys = np.empty(n)
    r = rng.random(n)
    for i in range(n):
        v = r[i]
        if v < 1 / 3:
            x, y = 0.5 * x, 0.5 * y
        elif v < 2 / 3:
            x, y = 0.5 * x + 0.5, 0.5 * y
        else:
            x, y = 0.5 * x + 0.25, 0.5 * y + 0.5
        xs[i] = x; ys[i] = y
    img = np.zeros((SIZE, SIZE), np.uint8)
    ix = np.clip((xs * SIZE).astype(int), 0, SIZE - 1)
    iy = np.clip((ys * SIZE).astype(int), 0, SIZE - 1)
    np.add.at(img, (iy, ix), 1)
    img = np.clip(img.astype(np.int32) * 18, 0, 255).astype(np.uint8)
    tint = rng.integers(110, 256, 3)
    rgb = np.stack([np.clip(img.astype(np.int32) * tint[c] // 255, 0, 255) for c in range(3)], -1)
    return rgb.astype(np.uint8)


def gen_fractal_noise(rng, beta=1.4):
    """1/f^beta spectral synthesis - the fairest 'rich-texture fractal' candidate."""
    fy = np.fft.fftfreq(SIZE)[:, None] ** 2 + np.fft.fftfreq(SIZE)[None, :] ** 2
    fy[0, 0] = 1.0
    amp = 1.0 / (fy ** (beta / 2.0))
    chans = []
    for _ in range(3):
        phase = np.exp(2j * np.pi * rng.random((SIZE, SIZE)))
        field = np.real(np.fft.ifft2(amp * phase))
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)
        chans.append((field * 255).astype(np.uint8))
    return np.stack(chans, -1)


SYN_GENS = {"kakeya": gen_kakeya, "sierpinski": gen_sierpinski, "fractal_noise": gen_fractal_noise}


def make_synthetic(n, seed):
    rng = np.random.default_rng(seed)
    names = list(SYN_GENS.keys())
    per = n // len(names)
    arrs, tags = [], []
    for name in names:
        for _ in range(per):
            arr = SYN_GENS[name](rng)
            if PREP_JPEG_SYN:
                arr = jpeg_roundtrip(arr, 95)
            arrs.append(arr); tags.append(name)
    while len(arrs) < n:
        arrs.append(SYN_GENS[names[0]](rng)); tags.append(names[0])
    return arrs, tags


# ---------------------------------------------------------------- embedding
def lsb_match_embed(cover, rate, rng):
    """LSB matching (+-1) at `rate` bits per pixel. Deterministic index/bit plan from rng."""
    h, w, _ = cover.shape
    flat = cover.reshape(-1).astype(np.int16)
    nbits = int(rate * h * w)
    idx = rng.choice(flat.size, size=nbits, replace=False)
    bits = rng.integers(0, 2, nbits)
    vals = flat[idx]
    need = (vals & 1) != bits
    delta = (rng.integers(0, 2, nbits) * 2 - 1)
    vals = np.clip(vals + np.where(need, delta, 0), 0, 255)
    flat[idx] = vals
    return flat.reshape(cover.shape).astype(np.uint8)


# ---------------------------------------------------------------- channel
def jpeg_roundtrip(arr, q):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------- features (SPAM-like)
def spam_feats(img):
    """Truncated residual co-occurrences (order1 & order2, H & V) per RGB channel. 588-d."""
    feats = []
    q = 2 * T + 1
    for c in range(3):
        a = img[:, :, c].astype(np.int32)
        dh = a[:, :-1] - a[:, 1:]
        dv = a[:-1, :] - a[1:, :]
        for d in (dh, dv):
            d = np.clip(d, -T, T) + T
            for off in (1, 2):
                p = d[:, :-off].ravel(); s = d[:, off:].ravel()
                h = np.bincount(p * q + s, minlength=q * q).astype(np.float64)
                feats.append(h)
    f = np.concatenate(feats)
    return f / (f.sum() + 1e-9)


def extract_matrix(arrs):
    return np.asarray([spam_feats(x) for x in arrs], dtype=np.float64)


def transfer_auc(Xc_tr, Xs_tr, Xc_te, Xs_te):
    """Train detector on one carrier class, evaluate cover-vs-stego AUC on another (OOD test)."""
    Xtr = np.vstack([Xc_tr, Xs_tr]); ytr = np.r_[np.zeros(len(Xc_tr)), np.ones(len(Xs_tr))]
    Xte = np.vstack([Xc_te, Xs_te]); yte = np.r_[np.zeros(len(Xc_te)), np.ones(len(Xs_te))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p)


def train_auc(Xc, Xs, seed=0, test_ratio=0.3):
    """covers Xc (label0), stegos Xs (label1). Split by sample, no leakage."""
    rng = np.random.default_rng(seed)
    n = len(Xc)
    perm = rng.permutation(n)
    ntest = int(n * test_ratio)
    te, tr = perm[:ntest], perm[ntest:]

    def rows(X, ids, y):
        return X[ids], np.full(len(ids), y)

    Xtr = np.vstack([Xc[tr], Xs[tr]]); ytr = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
    Xte = np.vstack([Xc[te], Xs[te]]); yte = np.r_[np.zeros(len(te)), np.ones(len(te))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    print("[*] downloading natural covers ...", flush=True)
    got = download_natural(N_PER_CLASS)
    print("    downloaded/available:", got, flush=True)

    nat = load_natural(N_PER_CLASS)
    print("[*] natural loaded:", len(nat), flush=True)

    syn, syn_tags = make_synthetic(N_PER_CLASS, seed=SEED + 1)
    print("[*] synthetic generated:", len(syn), "tags:", {t: syn_tags.count(t) for t in set(syn_tags)}, flush=True)

    # embed
    rng_n = np.random.default_rng(SEED + 2)
    rng_s = np.random.default_rng(SEED + 3)
    nat_st = [lsb_match_embed(x, EMBED_RATE, rng_n) for x in nat]
    syn_st = [lsb_match_embed(x, EMBED_RATE, rng_s) for x in syn]
    print("[*] embedding done", flush=True)

    # features
    Xn_c, Xn_s = extract_matrix(nat), extract_matrix(nat_st)
    Xs_c, Xs_s = extract_matrix(syn), extract_matrix(syn_st)
    print("[*] features done", flush=True)

    # robustness probe: JPEG q70 on both classes (naive LSB extraction BER)
    def ber_after_jpeg(covers, rng):
        bers = []
        for cv in covers[:60]:
            rr = np.random.default_rng(SEED + 9)
            st = lsb_match_embed(cv, EMBED_RATE, rr)
            jp = jpeg_roundtrip(st, 70)
            # re-extract with same plan
            rr2 = np.random.default_rng(SEED + 9)
            flat = jp.reshape(-1)
            nbits = int(EMBED_RATE * SIZE * SIZE)
            idx = rr2.choice(st.size, size=nbits, replace=False)
            bits = rr2.integers(0, 2, nbits)
            got = flat[idx] & 1
            bers.append(float(np.mean(got != bits)))
        return float(np.mean(bers))

    print("[*] jpeg robustness probe ...", flush=True)
    ber_nat = ber_after_jpeg(nat, rng_n)
    ber_syn = ber_after_jpeg(syn, rng_s)

    # AUCs
    auc_nat = train_auc(Xn_c, Xn_s, seed=101)
    auc_syn = train_auc(Xs_c, Xs_s, seed=102)
    auc_carrier = train_auc(Xn_c, Xs_c, seed=103)  # natural cover vs synthetic cover
    # real-world scenario: adversary's model is built on natural covers only
    auc_ood_nat2syn = transfer_auc(Xn_c, Xn_s, Xs_c, Xs_s)
    auc_ood_syn2nat = transfer_auc(Xs_c, Xs_s, Xn_c, Xn_s)

    res = {
        "N_per_class": len(nat),
        "embed_rate_bpp": EMBED_RATE,
        "AUC_natural_cover_vs_stego": round(auc_nat, 4),
        "AUC_synthetic_cover_vs_stego": round(auc_syn, 4),
        "AUC_carrier_natural_vs_synthetic": round(auc_carrier, 4),
        "AUC_xfer_trained_natural_tested_synthetic": round(auc_ood_nat2syn, 4),
        "AUC_xfer_trained_synthetic_tested_natural": round(auc_ood_syn2nat, 4),
        "BER_jpeg70_natural": round(ber_nat, 4),
        "BER_jpeg70_synthetic": round(ber_syn, 4),
        "gpu_time_s": round(time.time() - t0, 1),
    }
    print("\n===== RESULT =====")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(os.path.join(OUT, "result_pilot1.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    main()
