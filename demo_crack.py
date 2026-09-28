# -*- coding: utf-8 -*-
"""Faithful re-implementation of the analyzed APK's container (a/b.java), plus a BLIND
extractor that knows nothing about the app. Purpose: show the container is not protection."""
import io, os
import numpy as np
from PIL import Image

ENC = b"|||ENCRYPT_DELIMITER|||"
FN  = b"|||FILENAME_DELIMITER|||"
FD  = b"|||FILE_DELIMITER|||"

OUT = os.environ.get("STEGOLAB_DIR", os.path.dirname(os.path.abspath(__file__)))

# ---- packer identical in semantics to a/b.java ----
def pack(cover_bytes, files):
    out = bytearray(cover_bytes); out += ENC
    for name, data in files:
        out += FN + name.encode("utf-8") + FN + FD + data
    return bytes(out)

# ---- BLIND extractor: no key, no app, just 'find the markers' ----
def unpack_blind(blob):
    i = blob.find(ENC)
    if i < 0:
        return None
    off = i + len(ENC); res = []
    while True:
        j = blob.find(FN, off)
        if j < 0: break
        ns = j + len(FN); k = blob.find(FN, ns)
        if k < 0: break
        name = blob[ns:k].decode("utf-8", "replace")
        d = blob.find(FD, k + len(FN))
        if d < 0: break
        ds = d + len(FD)
        nxt = blob.find(FN, ds); end = len(blob) if nxt < 0 else nxt
        res.append((name, blob[ds:end])); off = end
    return res

def png_bytes(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()

def main():
    cover = np.asarray(Image.open(os.path.join(OUT, "natural", "0000.jpg")).convert("RGB"), np.uint8)
    png = png_bytes(cover)
    secret = b"BANK-ACCOUNT: 6222-0000-1234  PASSWORD: hunter2\n" * 3
    stego = pack(png, [("secret.txt", secret), ("id_card.png", b"\x89PNG-not-really")])

    print("cover PNG bytes      :", len(png))
    print("stego  bytes         :", len(stego), " (delta =", len(stego)-len(png), ")")
    iend = stego.find(b"IEND") + 8
    print("bytes AFTER PNG IEND :", len(stego)-iend, "  <- any 'trailing data' scanner sees this")
    print("marker 'ENCRYPT' at  :", stego.find(ENC), "(plain ASCII, visible in `strings`)")
    print()

    # blind crack
    got = unpack_blind(stego)
    print("=== BLIND EXTRACTION (no key, no app) ===")
    for name, data in got:
        print("  file:", name, "|", len(data), "bytes |", data[:60])

    # what a platform re-encode does
    jp = io.BytesIO(); Image.fromarray(cover).save(jp, "JPEG", quality=90)
    jp = jp.getvalue()
    print()
    print("After platform JPEG q90 re-encode, markers still present?",
          (ENC in jp), "-> payload gone:", not (ENC in jp))

main()