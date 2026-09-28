# Raw results

All numbers below were produced by the scripts in this repo on the same machine
(see `results/pilot1.json` for the machine-readable pilot-1 output).

## pilot-1 — synthetic vs natural carriers

Config: `N = 400` per class, `256x256` RGB, LSB matching @ `0.4 bpp`,
SPAM-like residual co-occurrence features (`588-d`), `StandardScaler + LogisticRegression`,
train/test split by sample (70/30).

| Metric | Value |
|---|---:|
| AUC natural: clean vs stego | 0.9955 |
| AUC synthetic: clean vs stego | 0.9405 |
| **AUC carrier: natural vs synthetic (clean)** | **1.0000** |
| AUC transfer: natural-trained → synthetic-tested | 0.7218 |
| AUC transfer: synthetic-trained → natural-tested | 0.4904 |
| BER after JPEG q70, natural | 0.4997 |
| BER after JPEG q70, synthetic | 0.5003 |

## payload entropy — does encryption help?

| Payload | AUC |
|---|---:|
| random bits (≈ ciphertext) | 0.9933 |
| structured text | 0.9965 |
| all-zero | 0.9986 |

## controls — must be ≈ 0.5

| Control | AUC |
|---|---:|
| rate = 0 (image untouched) | 0.5000 |
| clean-half vs clean-half | 0.4739 |
| shuffled pairing (clean) | 0.4843 |

## embedding-rate sweep

| rate (bpp) | AUC |
|---:|---:|
| 0.00 | 0.5000 |
| 0.02 | 0.7226 |
| 0.05 | 0.8466 |
| 0.10 | 0.9452 |
| 0.20 | 0.9890 |
| 0.40 | 0.9922 |

## Interpretation

1. The carrier classifier's **AUC = 1.0** is the headline: synthetic covers are trivially
   separable from natural photos, so the "exotic cover" argument is self-defeating.
2. Within-class AUCs are both very high; the small gap does not constitute a stealth advantage.
3. Payload encryption does not reduce detectability (it is orthogonal).
4. LSB matching is detectable even at very low rates and is destroyed by JPEG q70.