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
Extra dtype coverage for the MPS "elem-float" kernel family.

Family / dtype policy (what this file verifies):
- divide          : float, float16, int8, uint8, int16, int32, int64,
                    bfloat16, complex64.
- elementwise_pow : float, float16, bfloat16.
- pow             : float, float16, bfloat16.

bfloat16 and complex64 require the macOS 14 MetalPerformanceShadersGraph SDK
*and* macOS 14 at runtime. This CI host may lack it, so those subtests
PROBE-AND-SKIP via ``_supports(...)``: a trivial add on a tiny mps tensor of
the dtype; on ANY exception we skip.

- bfloat16 results are compared against a float32 numpy oracle with LOOSE
  tolerance (bf16 carries only ~3 decimal digits).
- complex64 results are compared against the numpy complex oracle.
- integer results (small widths + int32/int64 on divide) are compared EXACTLY
  against BOTH a numpy reference AND the CPU backend.
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
    """Probe whether ``paddle_dtype`` works on the mps place at runtime.

    Creates a tiny float32 zero tensor on mps, casts it to ``paddle_dtype``,
    runs a trivial add, and returns False on ANY exception (e.g. the runtime
    macOS predates the bfloat16 / complex64 MPSGraph data types).
    """
    if not _mps_available():
        return False
    try:
        base = paddle.to_tensor(np.zeros((2,), dtype=np.float32), place="mps")
        t = base.astype(paddle_dtype)
        _ = paddle.add(t, t)
        # Force materialization so lazy graph failures surface here.
        _.numpy()
        return True
    except Exception:
        return False


# bfloat16 has ~3 decimal digits of precision -> looser than fp16.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

# complex64 is plain single precision per component.
C64_RTOL = 1e-4
C64_ATOL = 1e-6

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSElemFloatBase(unittest.TestCase):
    """Common setUp + probe flags for the elem-float-extra dtype tests."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        cls.bf16_ok = _supports(paddle.bfloat16)
        cls.c64_ok = _supports(paddle.complex64)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    # -- bfloat16 helpers -------------------------------------------------
    def _require_bf16(self):
        if not self.bf16_ok:
            self.skipTest("bfloat16 not supported on this MPS runtime")

    def _to_bf16_mps(self, x32):
        """float32 numpy -> paddle bfloat16 tensor on mps.

        bf16 has no numpy dtype, so build float32 on mps then ``.astype``.
        """
        return paddle.to_tensor(x32, place="mps").astype(paddle.bfloat16)

    def _assert_bf16_out(self, out):
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())

    def _check_bf16(self, out, ref32, rtol=BF16_RTOL, atol=BF16_ATOL):
        self._assert_bf16_out(out)
        got = out.astype("float32").numpy()
        np.testing.assert_allclose(got, ref32, rtol=rtol, atol=atol)

    # -- complex64 helper -------------------------------------------------
    def _require_c64(self):
        if not self.c64_ok:
            self.skipTest("complex64 not supported on this MPS runtime")


# ---------------------------------------------------------------------------
# divide: integer widths (exact vs numpy + CPU).
# ---------------------------------------------------------------------------


class TestMPSDivideInt(_MPSElemFloatBase):
    """Integer divide truncates toward zero on both MPSGraph and paddle."""

    def _divide_int(self, np_dtype, paddle_dtype, signed):
        for shape in _SHAPES:
            with self.subTest(shape=shape, dtype=np_dtype):
                lo = -9 if signed else 1
                x = np.random.randint(lo, 10, size=shape).astype(np_dtype)
                # Nonzero divisors; mixed signs where the dtype allows.
                choices = [-3, -2, 2, 3] if signed else [1, 2, 3]
                y = np.random.choice(choices, size=shape).astype(np_dtype)
                out = paddle.divide(
                    paddle.to_tensor(x, place="mps"),
                    paddle.to_tensor(y, place="mps"),
                )
                cpu = paddle.divide(
                    paddle.to_tensor(x, place="cpu"),
                    paddle.to_tensor(y, place="cpu"),
                )
                # paddle integer divide == C truncation toward zero.
                # numpy // floors, so emulate trunc-toward-zero explicitly.
                ref = np.trunc(
                    x.astype(np.float64) / y.astype(np.float64)
                ).astype(np_dtype)
                self.assertEqual(out.dtype, paddle_dtype)
                self.assertTrue("mps" in str(out.place).lower())
                np.testing.assert_array_equal(out.numpy(), ref)
                np.testing.assert_array_equal(out.numpy(), cpu.numpy())

    def test_divide_int8(self):
        self._divide_int(np.int8, paddle.int8, signed=True)

    def test_divide_int16(self):
        self._divide_int(np.int16, paddle.int16, signed=True)

    def test_divide_int32(self):
        self._divide_int(np.int32, paddle.int32, signed=True)

    def test_divide_int64(self):
        self._divide_int(np.int64, paddle.int64, signed=True)

    def test_divide_uint8(self):
        self._divide_int(np.uint8, paddle.uint8, signed=False)


# ---------------------------------------------------------------------------
# divide / add / subtract / multiply: complex64 (probe-and-skip).
# ---------------------------------------------------------------------------


class TestMPSComplex64(_MPSElemFloatBase):
    def _rand_c64(self, shape):
        return (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(
            np.complex64
        )

    def _run_binary_c64(self, p_op, n_op, avoid_zero_y=False):
        self._require_c64()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._rand_c64(shape)
                y = self._rand_c64(shape)
                if avoid_zero_y:
                    # Keep |y| away from 0 for a stable divide reference.
                    y = y + (np.sign(y.real) + 1j * np.sign(y.imag)).astype(
                        np.complex64
                    )
                out = p_op(
                    paddle.to_tensor(x, place="mps"),
                    paddle.to_tensor(y, place="mps"),
                )
                self.assertEqual(out.dtype, paddle.complex64)
                self.assertTrue("mps" in str(out.place).lower())
                ref = n_op(x, y).astype(np.complex64)
                np.testing.assert_allclose(
                    out.numpy(), ref, rtol=C64_RTOL, atol=C64_ATOL
                )

    def test_add_c64(self):
        self._run_binary_c64(paddle.add, np.add)

    def test_subtract_c64(self):
        self._run_binary_c64(paddle.subtract, np.subtract)

    def test_multiply_c64(self):
        self._run_binary_c64(paddle.multiply, np.multiply)

    def test_divide_c64(self):
        self._run_binary_c64(paddle.divide, np.divide, avoid_zero_y=True)


# ---------------------------------------------------------------------------
# divide: bfloat16 (probe-and-skip).
# ---------------------------------------------------------------------------


class TestMPSDivideBf16(_MPSElemFloatBase):
    def test_divide_bf16(self):
        self._require_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                # Divisors bounded away from zero for a stable bf16 result.
                y = (np.random.randn(*shape) * 0.5 + 2.0).astype(np.float32)
                out = paddle.divide(self._to_bf16_mps(x), self._to_bf16_mps(y))
                self._check_bf16(out, x / y)


# ---------------------------------------------------------------------------
# elementwise_pow: bfloat16 (probe-and-skip).
# ---------------------------------------------------------------------------


class TestMPSElementwisePowBf16(_MPSElemFloatBase):
    def test_elementwise_pow_bf16(self):
        self._require_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # Positive base keeps x**y real and in-domain.
                x = (np.random.rand(*shape) * 2.0 + 0.5).astype(np.float32)
                y = (np.random.rand(*shape) * 2.0).astype(np.float32)
                out = paddle.pow(self._to_bf16_mps(x), self._to_bf16_mps(y))
                self._check_bf16(out, np.power(x, y))


# ---------------------------------------------------------------------------
# pow (scalar exponent): bfloat16 (probe-and-skip).
# ---------------------------------------------------------------------------


class TestMPSPowScalarBf16(_MPSElemFloatBase):
    def test_pow_scalar_bf16(self):
        self._require_bf16()
        for factor in (2.0, 3.0, 0.5):
            for shape in _SHAPES:
                with self.subTest(shape=shape, factor=factor):
                    x = (np.random.rand(*shape) * 2.0 + 0.5).astype(np.float32)
                    out = paddle.pow(self._to_bf16_mps(x), factor)
                    self._check_bf16(out, np.power(x, factor))


if __name__ == "__main__":
    unittest.main(verbosity=2)
