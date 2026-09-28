# -*- coding: utf-8 -*-
"""Render previews: what a 'Kakeya/fractal' cover actually looks like, vs a natural photo,
and vs the same cover AFTER embedding a payload (should look identical)."""
import os, numpy as np
from PIL import Image, ImageDraw
from experiment import gen_kakeya, gen_sierpinski, gen_fractal_noise, lsb_match_embed, SIZE, OUT as BASE

OUT = os.path.join(BASE, "preview")
os.makedirs(OUT, exist_ok=True)

def label(im, txt):
    tile = Image.new("RGB", (SIZE, SIZE + 28), (25, 25, 25))
    tile.paste(im, (0, 28))
    d = ImageDraw.Draw(tile)
    d.text((6, 8), txt, fill=(235, 235, 235))
    return tile

def main():
    rng = np.random.default_rng(12345)
    kakeya = gen_kakeya(rng)
    sierp = gen_sierpinski(rng)
    frac = gen_fractal_noise(rng)
    nat = np.asarray(Image.open(os.path.join(BASE, "natural", "0001.jpg")).convert("RGB"), np.uint8)

    # embed a payload into kakeya + fractal (0.4 bpp) - visually indistinguishable
    k_stego = lsb_match_embed(kakeya, 0.4, np.random.default_rng(1))
    f_stego = lsb_match_embed(frac, 0.4, np.random.default_rng(2))

    tiles = [label(Image.fromarray(kakeya), "1 KAK EYA star (all directions)"),
             label(Image.fromarray(sierp), "2 Sierpinski IFS"),
             label(Image.fromarray(frac), "3 fractal noise (1/f)"),
             label(Image.fromarray(nat), "4 NATURAL photo (for contrast)"),
             label(Image.fromarray(k_stego), "5 kakeya + hidden payload"),
             label(Image.fromarray(f_stego), "6 fractal + hidden payload")]

    # individual saves
    for i, (name, arr) in enumerate([("kakeya", kakeya), ("sierpinski", sierp),
                                     ("fractal_noise", frac), ("natural", nat),
                                     ("kakeya_with_payload", k_stego),
                                     ("fractal_with_payload", f_stego)], 1):
        Image.fromarray(arr).save(os.path.join(OUT, "%d_%s.png" % (i, name)))

    # montage 3x2
    W, H = SIZE, SIZE + 28
    grid = Image.new("RGB", (W * 3, H * 2), (0, 0, 0))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 3) * W, (i // 3) * H))
    grid.save(os.path.join(OUT, "MONTAGE.png"))
    print("saved to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)

if __name__ == "__main__":
    main()