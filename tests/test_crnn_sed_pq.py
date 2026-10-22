import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "PLG-MT_run", "models")
    ),
)

from crnn_sed_pq import (
    Bfp64MelFrame,
    Q16_ONE,
    attention_pool_sed_pq_q16,
    context_gating_glu_pq_q16,
    sigmoid_pq_q16,
)


class TestToshibaDcaseSedPq(unittest.TestCase):
    def test_context_gating_glu_pq_q16(self):
        feats = [1.5, -0.8, 2.2, -1.1]
        gates = [0.0, 1.2, -1.5, 2.5]
        feats_q16 = [int(round(v * Q16_ONE)) for v in feats]
        gates_q16 = [int(round(v * Q16_ONE)) for v in gates]

        out_q16 = context_gating_glu_pq_q16(feats_q16, gates_q16)
        for f, g, o in zip(feats, gates, out_q16):
            exact = f / (1.0 + math.exp(-g))
            self.assertAlmostEqual(o / float(Q16_ONE), exact, delta=2e-3)

    def test_attention_pool_sed_pq_q16_matches_crnn_forward(self):
        # 5 frames, 3 sound classes (e.g. Speech, Dog, Vacuum_cleaner)
        strong_logits = [
            [-2.0, -1.5, -2.0],
            [2.5, -1.0, -2.0],
            [3.0, -1.2, -1.8],
            [-1.5, 2.2, -2.0],
            [-2.0, -1.8, -2.0],
        ]
        attn_logits = [
            [-1.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.5, -1.0, 0.0],
            [-1.0, 2.0, 0.0],
            [-1.0, -1.0, 0.0],
        ]
        s_q16 = [[int(round(v * Q16_ONE)) for v in row] for row in strong_logits]
        a_q16 = [[int(round(v * Q16_ONE)) for v in row] for row in attn_logits]

        strong_out, weak_out = attention_pool_sed_pq_q16(s_q16, a_q16)
        self.assertEqual(len(strong_out), 5)
        self.assertEqual(len(weak_out), 3)
        # Class 0 (active on frames 1,2 with high attention) should have high weak probability
        self.assertGreater(weak_out[0] / float(Q16_ONE), 0.70)
        # Class 2 (inactive everywhere) should have low weak probability
        self.assertLess(weak_out[2] / float(Q16_ONE), 0.15)

    def test_bfp64_mel_frame_compression(self):
        mel64 = [math.sin(0.15 * k) * 4.0 - 2.0 for k in range(64)]
        packed = Bfp64MelFrame.compress(mel64)
        self.assertEqual(packed.byte_size, 65)
        rec = packed.decompress()
        for orig, r in zip(mel64, rec):
            self.assertAlmostEqual(orig, r, delta=0.07)


if __name__ == "__main__":
    unittest.main()
