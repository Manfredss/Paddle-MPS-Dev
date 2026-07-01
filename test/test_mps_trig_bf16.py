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
bfloat16 dtype coverage for the MPS "trig" kernel family.

Family / dtype policy:
- Every trig op (sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh,
  acosh, atanh) gains bfloat16 in addition to its existing float32 + float16
  support. No integer and no complex dtypes are added to this family, and no
  graph branch changes; only the kernel registrations grow a bfloat16 token
  behind the macOS-14 SDK guard.

bfloat16 results are compared against a float32 numpy oracle with LOOSE
tolerance: bfloat16 carries only ~3 decimal digits of precision (an 8-bit
mantissa), so it is noticeably coarser than float16. CPU may not register
bfloat16 for these ops, so we never compare MPS bfloat16 against the CPU
backend; the oracle is always a float32 numpy computation.

bfloat16 requires the macOS-14 MetalPerformanceShadersGraph SDK at runtime.
This CI host may be older, so we PROBE-AND-SKIP: a tiny bfloat16 add is run on
an MPS tensor in setUpClass; any exception skips the whole suite.

Per-op input domains (so the math is defined and well-conditioned):
- asin / acos / atanh : |x| <= 1
- acosh               : x >= 1
- all others          : modest range around 0
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
    """Return True iff a trivial op on an MPS tensor of ``paddle_dtype`` works.

    bfloat16 (and complex64) require the macOS-14 MPSGraph SDK at runtime.
    On older hosts constructing/operating on such a tensor raises; we treat
    any exception as "unsupported" so the caller can skip.
    """
    if not _mps_available():
        return False
    try:
        base = np.zeros((2,), dtype=np.float32)
        t = paddle.to_tensor(base, place="mps").astype(paddle_dtype)
        out = paddle.add(t, t)
        # Force the graph to actually execute / materialize.
        _ = out.astype("float32").numpy()
        return True
    except Exception:
        return False


# bfloat16 has ~3 decimal digits (8-bit mantissa); use looser tol than fp16.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSTrigBf16Base(unittest.TestCase):
    """Common setUp + bfloat16 probe for the trig-family dtype tests."""

    BF16_OK = False

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        cls.BF16_OK = _supports(paddle.bfloat16)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)
        if not self.BF16_OK:
            self.skipTest(
                "bfloat16 unsupported at runtime (requires macOS 14 MPSGraph)"
            )

    # -- bfloat16 helpers -------------------------------------------------
    def _assert_bf16_out(self, out):
        """The MPS output must be bfloat16 and live on the mps place."""
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())

    def _run_bf16(
        self, paddle_op, numpy_op, x32, rtol=BF16_RTOL, atol=BF16_ATOL
    ):
        """Run ``paddle_op`` on a bfloat16 MPS tensor; compare to fp32 oracle.

        ``x32`` is a float32 numpy array already restricted to the op's domain.
        bfloat16 has no numpy dtype, so we build a float32 MPS tensor and cast
        it to bfloat16 with ``.astype``. The oracle is ``numpy_op`` evaluated
        on the *bf16-rounded* input (cast there and back through float32) so the
        comparison measures the op error, not the input-quantization error.
        """
        x_mps = paddle.to_tensor(x32, place="mps").astype(paddle.bfloat16)
        out = paddle_op(x_mps)
        self._assert_bf16_out(out)
        # Round the oracle input through bfloat16 to match the kernel input.
        x_bf_round = (
            paddle.to_tensor(x32, place="mps")
            .astype(paddle.bfloat16)
            .astype("float32")
            .numpy()
        )
        ref = numpy_op(x_bf_round).astype(np.float32)
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, rtol=rtol, atol=atol
        )
        return out


# ---------------------------------------------------------------------------
# Per-op specs: (name, paddle_op, numpy_op, domain).
# domain is a callable shape -> float32 numpy array within the op's valid range.
# ---------------------------------------------------------------------------


def _modest(shape):
    # Range around 0 where sin/cos/tan/.../atan are all well behaved.
    return (np.random.uniform(-1.2, 1.2, size=shape)).astype(np.float32)


def _unit(shape):
    # |x| <= 1 for asin / acos / atanh; stay just inside to avoid edge blowups.
    return (np.random.uniform(-0.9, 0.9, size=shape)).astype(np.float32)


def _ge_one(shape):
    # x >= 1 for acosh.
    return (np.random.uniform(1.05, 3.0, size=shape)).astype(np.float32)


_TRIG_OPS = [
    ("sin", lambda t: paddle.sin(t), np.sin, _modest),
    ("cos", lambda t: paddle.cos(t), np.cos, _modest),
    ("tan", lambda t: paddle.tan(t), np.tan, _modest),
    ("asin", lambda t: paddle.asin(t), np.arcsin, _unit),
    ("acos", lambda t: paddle.acos(t), np.arccos, _unit),
    ("atan", lambda t: paddle.atan(t), np.arctan, _modest),
    ("sinh", lambda t: paddle.sinh(t), np.sinh, _modest),
    ("cosh", lambda t: paddle.cosh(t), np.cosh, _modest),
    ("tanh", lambda t: paddle.tanh(t), np.tanh, _modest),
    ("asinh", lambda t: paddle.asinh(t), np.arcsinh, _modest),
    ("acosh", lambda t: paddle.acosh(t), np.arccosh, _ge_one),
    ("atanh", lambda t: paddle.atanh(t), np.arctanh, _unit),
]


class TestMPSTrigBf16(_MPSTrigBf16Base):
    """bfloat16 coverage for the whole trig family."""

    def test_all_trig_bf16(self):
        for name, p_op, n_op, domain in _TRIG_OPS:
            for shape in _SHAPES:
                with self.subTest(op=name, shape=shape):
                    x32 = domain(shape)
                    self._run_bf16(p_op, n_op, x32)

    def test_known_values_bf16(self):
        # sin(0)=0, cos(0)=1, tanh(0)=0; spot-check a couple of exact-ish points
        # that bfloat16 can represent without much rounding.
        x32 = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        for name, p_op, n_op, _ in (
            ("sin", lambda t: paddle.sin(t), np.sin, None),
            ("cos", lambda t: paddle.cos(t), np.cos, None),
            ("tanh", lambda t: paddle.tanh(t), np.tanh, None),
        ):
            with self.subTest(op=name):
                self._run_bf16(p_op, n_op, x32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
