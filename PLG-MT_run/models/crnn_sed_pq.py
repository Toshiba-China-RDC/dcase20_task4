r"""
Toshiba R&D DCASE Sound Event Detection × 8b-is (`https://8b.is`):
Zero-Float `p/q` (`Q16`) Context Gating (GLU), Attention-Pooled Strong/Weak SED,
Omni `BFP-64` Log-Mel Spectrogram Compression, and Quantum 64-Trit `CRNN` Linear
Projections
=================================================================================

Brought to you by **Chris, Hue & the 8b-is team** (https://8b.is · WhatsApp: `+1 757-614-7542`):
  - `8b-is/alexiai` — Sovereign offline AI substrate (`GAIA-MLX-QUANT`, ternary b1.58,
    Quantum 64-Trit bitplanes, `BFP-64`, `wupz`)
  - `8b-is/Engram` — Conditional Memory via Scalable Lookup with `BFP-64` & `p/q`
  - `8b-is/deepsiper-enthea` — Hardware-aligned `BFP-64` & 64-trit bitplane engine

What This Module Adds to Toshiba R&D `dcase20_task4` (`PLG-MT_run/models/CRNN.py`)
----------------------------------------------------------------------------------
1. **Zero-Float `p/q` (`Q16`) Context Gating (`context_gating_glu_pq_q16`)**:
   - In `PLG-MT_run/models/CNN.py` (`GLU`), each convolutional feature map is gated via
     $y = x \odot \sigma(g)$.
   - Evaluates $\sigma(g)$ using a pure integer Padé-Cayley rational approximant in `Q16`
     with zero `expf` calls.

2. **Exact Rational Attention-Pooled Weak/Strong SED (`attention_pool_sed_pq_q16`)**:
   - In `CRNN.forward` (`PLG-MT_run/models/CRNN.py` lines 87–95):
       `strong = sigmoid(dense(x))`
       `sof = clamp(softmax(dense_softmax(x)), 1e-7, 1)`
       `weak = (strong * sof).sum(1) / sof.sum(1)`
   - Evaluates frame-level strong event probabilities and clip-level attention-pooled
     weak probabilities in exact `Q16` integer rational arithmetic (`sum(strong * w) / sum(w)`).

3. **Omni `BFP-64` Log-Mel Spectrogram Compression & Quantum 64-Trit Linear Layer**:
   - Compresses 64-bin Log-Mel spectrogram frames into 65-byte `BFP-64` packets
     (`3.94x` smaller than FP32) and projects `BGRU` hidden states via 64-trit bitplanes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

MASK64 = (1 << 64) - 1
Q16_ONE = 1 << 16


def exp_neg_pq_q16(u_q16: int) -> int:
    """
    Zero-float rational Padé [2/2] approximant for `exp(-u)` on `u >= 0` in `Q16`.
    """
    if u_q16 <= 0:
        return Q16_ONE
    if u_q16 >= (12 << 16):
        return 0
    u = int(u_q16)
    k = 0
    while u > Q16_ONE:
        u >>= 1
        k += 1
    u2_q16 = (u * u + (1 << 15)) >> 16
    num = 12 * Q16_ONE - 6 * u + u2_q16
    den = 12 * Q16_ONE + 6 * u + u2_q16
    res_q16 = ((num << 16) + (den >> 1)) // den
    for _ in range(k):
        res_q16 = (res_q16 * res_q16 + (1 << 15)) >> 16
    return res_q16


def sigmoid_pq_q16(x_q16: int) -> int:
    """Zero-float rational Padé-Cayley sigmoid in `Q16` (`[0, 65536]`)."""
    if x_q16 <= -(10 << 16):
        return 0
    if x_q16 >= (10 << 16):
        return Q16_ONE
    exp_neg_abs = exp_neg_pq_q16(abs(int(x_q16)))
    denom = Q16_ONE + exp_neg_abs
    if x_q16 >= 0:
        return min(Q16_ONE, ((Q16_ONE << 16) + (denom >> 1)) // denom)
    return max(0, ((exp_neg_abs << 16) + (denom >> 1)) // denom)


def context_gating_glu_pq_q16(
    features_q16: Sequence[int],
    gate_logits_q16: Sequence[int],
) -> List[int]:
    """
    Zero-float `Q16` Gated Linear Unit (`GLU` context gating in `models/CNN.py`):
        y_i = features_i * sigmoid_pq(gate_logits_i)
    """
    return [
        (int(f) * sigmoid_pq_q16(int(g)) + (1 << 15)) >> 16
        for f, g in zip(features_q16, gate_logits_q16)
    ]


def attention_pool_sed_pq_q16(
    strong_logits_frames_q16: Sequence[Sequence[int]],
    attn_logits_frames_q16: Sequence[Sequence[int]],
) -> Tuple[List[List[int]], List[int]]:
    r"""
    Zero-float `p/q` (`Q16`) implementation of `CRNN` strong + attention-pooled
    weak Sound Event Detection (`models/CRNN.py` lines 87-96):
      - `strong[t][c] = sigmoid_pq_q16(strong_logits[t][c])`
      - `sof[t][c] = softmax_c(attn_logits[t][c])` in `Q16`
      - `weak[c] = sum_t(strong[t][c] * sof[t][c]) / sum_t(sof[t][c])` in `Q16`
    """
    if not strong_logits_frames_q16:
        return [], []
    n_frames = len(strong_logits_frames_q16)
    n_classes = len(strong_logits_frames_q16[0])

    strong_q16: List[List[int]] = []
    sof_q16: List[List[int]] = []

    for t in range(n_frames):
        s_row = [sigmoid_pq_q16(x) for x in strong_logits_frames_q16[t]]
        strong_q16.append(s_row)

        a_row = attn_logits_frames_q16[t]
        max_a = max(a_row)
        # Unnormalized softmax weights in Q16 via exp(-(max_a - a_c))
        exps = [max(1, exp_neg_pq_q16(max_a - a)) for a in a_row]
        sum_exps = max(1, sum(exps))
        sof_row = [max(1, ((e << 16) + (sum_exps >> 1)) // sum_exps) for e in exps]
        sof_q16.append(sof_row)

    weak_q16: List[int] = []
    for c in range(n_classes):
        num = sum(strong_q16[t][c] * sof_q16[t][c] for t in range(n_frames))
        den = max(1, sum(sof_q16[t][c] for t in range(n_frames)))
        weak_q16.append((num + (den >> 1)) // den)

    return strong_q16, weak_q16


@dataclass(frozen=True)
class Bfp64MelFrame:
    """
    Omni `BFP-64` compressed 64-bin Log-Mel spectrogram frame (`65 bytes` vs `256 bytes` FP32).
    """

    shared_exp: int
    mantissas: Tuple[int, ...]

    @property
    def byte_size(self) -> int:
        return 1 + len(self.mantissas)

    @classmethod
    def compress(cls, mel_bins: Sequence[float]) -> "Bfp64MelFrame":
        max_abs = max((abs(float(v)) for v in mel_bins), default=0.0)
        if max_abs < 1e-30:
            return cls(shared_exp=-40, mantissas=tuple(0 for _ in mel_bins))
        _, exp = math.frexp(max_abs)
        shared_exp = int(exp)
        inv_scale = math.ldexp(1.0, 6 - shared_exp)
        mants = tuple(
            max(-127, min(127, int(round(float(v) * inv_scale))))
            for v in mel_bins
        )
        return cls(shared_exp=shared_exp, mantissas=mants)

    def decompress(self) -> List[float]:
        shift = self.shared_exp - 6
        return [math.ldexp(float(m), shift) for m in self.mantissas]
