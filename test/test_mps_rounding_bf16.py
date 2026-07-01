#!/usr/bin/env python3


# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.


"""
bfloat16 (and float16 sanity) dtype coverage for the MPS "rounding" kernel
family: floor, ceil, round, trunc.

Family / dtype policy:
- floor / ceil / round / trunc : float32 + float16 (base, macOS 12+) and
  bfloat16 (gated behind the macOS-14 MPSGraph SDK / runtime).

These ops are pure element-wise rounding operations with no integer or complex
registrations, so this file only exercises the float dtypes.

float16/bfloat16 results are compared against a float32 numpy oracle (the CPU
backend may not register these half dtypes for every op, so we never compare an
MPS half result against the CPU backend). bfloat16 carries only ~3 decimal
digits, so its tolerance is looser than float16's.

bfloat16 requires macOS 14 at runtime, which the CI host may lack. We therefore
PROBE for bf16 support once (a tiny add on a bf16 mps tensor) and skip the bf16
subtests when the probe fails.
"""

import unittest

import numpy as np

try:
    import paddle

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddlePaddle not available")


def _mps_available():
    return (
        PADDLE_AVAILABLE
        and paddle.is_compiled_with_mps()
        and paddle.mps.is_available()
    )


def _supports(paddle_dtype):
    """Return True if a trivial op on an MPS tensor of ``paddle_dtype`` works.

    bf16 / complex64 only map to MPSGraph data types on the macOS-14 SDK and
    require macOS 14 at runtime; on older hosts any op raises. We create a tiny
    tensor of the requested dtype on the mps place, run a trivial add, and
    report False on ANY exception.
    """
    if not _mps_available():
        return False
    try:
        if paddle_dtype == paddle.bfloat16:
            t = paddle.to_tensor(
                np.zeros((2,), dtype=np.float32), place="mps"
            ).astype("bfloat16")
        elif paddle_dtype == paddle.complex64:
            t = paddle.to_tensor(
                np.zeros((2,), dtype=np.complex64), place="mps"
            )
        else:
            t = paddle.to_tensor(np.zeros((2,), dtype=np.float32), place="mps")
        _ = (t + t).astype("float32").numpy()
        return True
    except Exception:
        return False


# float16 tolerances (half precision accumulates noticeable error).
F16_RTOL = 2e-2
F16_ATOL = 2e-2
# bfloat16 keeps only ~3 decimal digits: looser still than float16.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

_SHAPES = [(6,), (3, 4), (2, 3, 4)]

# Each op: (name, paddle fn, numpy oracle). Domains are all of R for rounding.
_OPS = (
    ("floor", lambda t: paddle.floor(t), np.floor),
    ("ceil", lambda t: paddle.ceil(t), np.ceil),
    ("round", lambda t: paddle.round(t), np.round),
    ("trunc", lambda t: paddle.trunc(t), np.trunc),
)


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class TestMPSRoundingBf16(unittest.TestCase):
    """bf16 (+ f16 sanity) coverage for floor / ceil / round / trunc."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        cls._bf16_ok = _supports(paddle.bfloat16)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    # -- inputs -----------------------------------------------------------
    def _modest_x32(self, shape):
        """Modest-range float32 input.

        Avoid exact-half values (x.5) so the result is unambiguous across the
        differing tie-break rules of paddle.round (half-away/banker's) and the
        MPSGraph rounding op; offset by 0.27 keeps values clear of .0 and .5.
        """
        return ((np.random.randn(*shape) * 3.0) + 0.27).astype(np.float32)

    # -- float16 sanity ---------------------------------------------------
    def test_ops_f16(self):
        for shape in _SHAPES:
            x32 = self._modest_x32(shape)
            x16 = x32.astype(np.float16)
            for name, p_op, n_op in _OPS:
                with self.subTest(op=name, shape=shape):
                    out = p_op(paddle.to_tensor(x16, place="mps"))
                    self.assertEqual(out.dtype, paddle.float16)
                    self.assertTrue("mps" in str(out.place).lower())
                    ref = n_op(x16.astype(np.float32)).astype(np.float32)
                    np.testing.assert_allclose(
                        out.numpy().astype(np.float32),
                        ref,
                        rtol=F16_RTOL,
                        atol=F16_ATOL,
                    )

    # -- bfloat16 (probe-and-skip) ---------------------------------------
    def test_ops_bf16(self):
        if not self._bf16_ok:
            self.skipTest("MPS bfloat16 unsupported (needs macOS 14 runtime)")
        for shape in _SHAPES:
            x32 = self._modest_x32(shape)
            # bf16 has no numpy dtype: build float32 on mps, then cast.
            for name, p_op, n_op in _OPS:
                with self.subTest(op=name, shape=shape):
                    x_bf16 = paddle.to_tensor(x32, place="mps").astype(
                        "bfloat16"
                    )
                    out = p_op(x_bf16)
                    self.assertEqual(out.dtype, paddle.bfloat16)
                    self.assertTrue("mps" in str(out.place).lower())
                    # Oracle uses the bf16-rounded input to avoid penalizing the
                    # half->op for the (separate) input quantization error.
                    x_in = x_bf16.astype("float32").numpy().astype(np.float32)
                    ref = n_op(x_in).astype(np.float32)
                    np.testing.assert_allclose(
                        out.astype("float32").numpy(),
                        ref,
                        rtol=BF16_RTOL,
                        atol=BF16_ATOL,
                    )

    def test_known_values_bf16(self):
        if not self._bf16_ok:
            self.skipTest("MPS bfloat16 unsupported (needs macOS 14 runtime)")
        x32 = np.array([-2.7, -0.3, 0.3, 1.2, 3.8], dtype=np.float32)
        x_bf16_base = paddle.to_tensor(x32, place="mps").astype("bfloat16")
        x_in = x_bf16_base.astype("float32").numpy().astype(np.float32)
        for name, p_op, n_op in _OPS:
            with self.subTest(op=name):
                out = p_op(
                    paddle.to_tensor(x32, place="mps").astype("bfloat16")
                )
                self.assertEqual(out.dtype, paddle.bfloat16)
                self.assertTrue("mps" in str(out.place).lower())
                ref = n_op(x_in).astype(np.float32)
                np.testing.assert_allclose(
                    out.astype("float32").numpy(),
                    ref,
                    rtol=BF16_RTOL,
                    atol=BF16_ATOL,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
