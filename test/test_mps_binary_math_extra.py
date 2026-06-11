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
Extra dtype coverage for the MPS "binary-math" kernel family.

This complements the existing float32/float16 coverage by exercising the dtypes
added in this change:

- bfloat16  -> fmax, fmin, floor_divide, remainder, heaviside, atan2 (every
              float-capable binary-math op that gains bf16). bf16 only has ~3
              decimal digits, so it is compared against a float32 numpy oracle
              with LOOSE tolerance.
- complex64 -> add, subtract, multiply, divide ONLY. Compared exactly-ish
              (rtol=1e-4) against the numpy complex oracle.
- small / new integers (int8, int16, uint8, int32, int64) -> sign, scale,
              divide, floor_divide, remainder, heaviside (per each op's CPU
              registration). Integers are compared EXACTLY against both a numpy
              reference and the CPU backend. floor_divide / remainder use
              MIXED-SIGN inputs with non-zero divisors to verify the sign
              conventions, which is the entire point of the integer graphs.

bfloat16 and complex64 require macOS 14 at runtime, which this CI host may lack,
so the bf16 / complex64 subtests PROBE the dtype on the mps place first and
skip themselves on any exception.
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
    """Return True iff a trivial add works on an mps tensor of ``paddle_dtype``.

    bf16 / complex64 only exist in the macOS-14 MPSGraph SDK and only work at
    runtime on macOS 14+, so we probe by building a tiny tensor on the mps
    place and running an add. Any exception means the dtype is unsupported on
    this host and the caller should skip.
    """
    if not _mps_available():
        return False
    try:
        if paddle_dtype == paddle.bfloat16:
            base = paddle.to_tensor(
                np.zeros((2,), dtype=np.float32), place="mps"
            ).astype("bfloat16")
        elif paddle_dtype == paddle.complex64:
            base = paddle.to_tensor(
                np.zeros((2,), dtype=np.complex64), place="mps"
            )
        else:
            base = paddle.to_tensor(
                np.zeros((2,), dtype=np.float32), place="mps"
            ).astype(paddle_dtype)
        _ = paddle.add(base, base)
        # Force materialization so lazy failures surface here.
        _ = _.numpy()
        return True
    except Exception:
        return False


# bf16 has ~3 decimal digits; looser than fp16.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

# complex64 is exact-ish.
C64_RTOL = 1e-4

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSBinaryMathBase(unittest.TestCase):
    """Common setUp + helpers for the binary-math extra-dtype tests."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        cls.bf16_ok = _supports(paddle.bfloat16)
        cls.c64_ok = _supports(paddle.complex64)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    # -- bfloat16 helper --------------------------------------------------
    def _require_bf16(self):
        if not self.bf16_ok:
            self.skipTest(
                "bfloat16 not supported on this MPS host (needs macOS 14+)"
            )

    def _to_bf16(self, x32):
        """float32 numpy -> mps bfloat16 paddle tensor (no numpy bf16 dtype)."""
        return paddle.to_tensor(x32, place="mps").astype("bfloat16")

    def _assert_bf16_out(self, out):
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())

    def _check_bf16_binary(
        self, paddle_fn, np_fn, x32, y32, rtol=BF16_RTOL, atol=BF16_ATOL
    ):
        """Run ``paddle_fn`` on bf16 mps inputs, compare to float32 oracle."""
        self._require_bf16()
        out = paddle_fn(self._to_bf16(x32), self._to_bf16(y32))
        self._assert_bf16_out(out)
        ref = np_fn(x32, y32).astype(np.float32)
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, rtol=rtol, atol=atol
        )

    # -- complex64 helper -------------------------------------------------
    def _require_c64(self):
        if not self.c64_ok:
            self.skipTest(
                "complex64 not supported on this MPS host (needs macOS 14+)"
            )

    def _check_c64_binary(self, paddle_fn, np_fn, x, y):
        self._require_c64()
        out = paddle_fn(
            paddle.to_tensor(x, place="mps"),
            paddle.to_tensor(y, place="mps"),
        )
        self.assertEqual(out.dtype, paddle.complex64)
        self.assertTrue("mps" in str(out.place).lower())
        ref = np_fn(x, y).astype(np.complex64)
        np.testing.assert_allclose(out.numpy(), ref, rtol=C64_RTOL)

    # -- integer helper (exact vs numpy AND cpu backend) ------------------
    def _check_int_binary(self, paddle_fn, np_fn, x, y, paddle_dtype):
        out = paddle_fn(
            paddle.to_tensor(x, place="mps"),
            paddle.to_tensor(y, place="mps"),
        )
        cpu = paddle_fn(
            paddle.to_tensor(x, place="cpu"),
            paddle.to_tensor(y, place="cpu"),
        )
        ref = np_fn(x, y).astype(x.dtype)
        self.assertEqual(out.dtype, paddle_dtype)
        self.assertTrue("mps" in str(out.place).lower())
        np.testing.assert_array_equal(out.numpy(), ref)
        np.testing.assert_array_equal(out.numpy(), cpu.numpy())

    def _check_int_unary(self, paddle_fn, np_fn, x, paddle_dtype):
        out = paddle_fn(paddle.to_tensor(x, place="mps"))
        cpu = paddle_fn(paddle.to_tensor(x, place="cpu"))
        ref = np_fn(x).astype(x.dtype)
        self.assertEqual(out.dtype, paddle_dtype)
        self.assertTrue("mps" in str(out.place).lower())
        np.testing.assert_array_equal(out.numpy(), ref)
        np.testing.assert_array_equal(out.numpy(), cpu.numpy())


# Mapping of integer numpy dtype -> paddle dtype for the parametrized tests.
_SIGNED_INTS = [
    (np.int8, paddle.int8),
    (np.int16, paddle.int16),
    (np.int32, paddle.int32),
    (np.int64, paddle.int64),
]
_UNSIGNED_INTS = [
    (np.uint8, paddle.uint8),
]


def _mixed_sign_xy(np_dtype, shape):
    """x in [-9, 9], y in {-3, -2, 2, 3} (non-zero divisors, mixed signs)."""
    x = np.random.randint(-9, 10, size=shape).astype(np_dtype)
    y = np.random.choice(np.array([-3, -2, 2, 3]), size=shape).astype(np_dtype)
    return x, y


def _nonneg_xy(np_dtype, shape):
    """Non-negative x in [0, 9], y in {2, 3} (for unsigned floor_divide)."""
    x = np.random.randint(0, 10, size=shape).astype(np_dtype)
    y = np.random.choice(np.array([2, 3]), size=shape).astype(np_dtype)
    return x, y


# ---------------------------------------------------------------------------
# fmax / fmin: gain bf16.
# ---------------------------------------------------------------------------


class TestMPSFMaxFMinBf16(_MPSBinaryMathBase):
    def test_fmax_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 3.0).astype(np.float32)
                y = (np.random.randn(*shape) * 3.0).astype(np.float32)
                self._check_bf16_binary(paddle.fmax, np.fmax, x, y)

    def test_fmin_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 3.0).astype(np.float32)
                y = (np.random.randn(*shape) * 3.0).astype(np.float32)
                self._check_bf16_binary(paddle.fmin, np.fmin, x, y)


# ---------------------------------------------------------------------------
# atan2: gains bf16 (float-only).
# ---------------------------------------------------------------------------


class TestMPSAtan2Bf16(_MPSBinaryMathBase):
    def test_atan2_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # paddle.atan2(x, y) == numpy arctan2(x, y) (x numerator).
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                y = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._check_bf16_binary(paddle.atan2, np.arctan2, x, y)


# ---------------------------------------------------------------------------
# heaviside: gains bf16 + int32/int64.
# ---------------------------------------------------------------------------


def _np_heaviside(x, y):
    return np.heaviside(x, y)


class TestMPSHeaviside(_MPSBinaryMathBase):
    def test_heaviside_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # Include exact zeros so the (x == 0 -> y) branch is exercised.
                x = np.random.choice(
                    np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
                    size=shape,
                ).astype(np.float32)
                y = (np.random.randn(*shape)).astype(np.float32)
                self._check_bf16_binary(paddle.heaviside, _np_heaviside, x, y)

    def test_heaviside_int(self):
        for np_dtype, p_dtype in _SIGNED_INTS[2:]:  # int32, int64
            for shape in _SHAPES:
                with self.subTest(dtype=np_dtype, shape=shape):
                    x = np.random.randint(-3, 4, size=shape).astype(np_dtype)
                    y = np.random.randint(-3, 4, size=shape).astype(np_dtype)
                    self._check_int_binary(
                        paddle.heaviside, _np_heaviside, x, y, p_dtype
                    )


# ---------------------------------------------------------------------------
# floor_divide: gains bf16 + uint8/int8/int16/int32/int64.
# ---------------------------------------------------------------------------


class TestMPSFloorDivide(_MPSBinaryMathBase):
    def test_floor_divide_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 5.0).astype(np.float32)
                # Keep |y| away from 0 so float floor(x/y) is well defined.
                y = (
                    np.random.choice(
                        np.array([-3.0, -2.0, 2.0, 3.0], dtype=np.float32),
                        size=shape,
                    )
                ).astype(np.float32)
                self._check_bf16_binary(
                    paddle.floor_divide, np.floor_divide, x, y
                )

    def test_floor_divide_signed_int(self):
        for np_dtype, p_dtype in _SIGNED_INTS:
            for shape in _SHAPES:
                with self.subTest(dtype=np_dtype, shape=shape):
                    x, y = _mixed_sign_xy(np_dtype, shape)
                    self._check_int_binary(
                        paddle.floor_divide, np.floor_divide, x, y, p_dtype
                    )

    def test_floor_divide_uint8(self):
        for np_dtype, p_dtype in _UNSIGNED_INTS:
            for shape in _SHAPES:
                with self.subTest(dtype=np_dtype, shape=shape):
                    x, y = _nonneg_xy(np_dtype, shape)
                    self._check_int_binary(
                        paddle.floor_divide, np.floor_divide, x, y, p_dtype
                    )


# ---------------------------------------------------------------------------
# remainder: gains bf16 + int32/int64.
# ---------------------------------------------------------------------------


class TestMPSRemainder(_MPSBinaryMathBase):
    def test_remainder_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 5.0).astype(np.float32)
                y = (
                    np.random.choice(
                        np.array([-3.0, -2.0, 2.0, 3.0], dtype=np.float32),
                        size=shape,
                    )
                ).astype(np.float32)
                self._check_bf16_binary(paddle.remainder, np.remainder, x, y)

    def test_remainder_int(self):
        for np_dtype, p_dtype in _SIGNED_INTS[2:]:  # int32, int64
            for shape in _SHAPES:
                with self.subTest(dtype=np_dtype, shape=shape):
                    x, y = _mixed_sign_xy(np_dtype, shape)
                    # numpy remainder is Python-style, matching paddle.
                    self._check_int_binary(
                        paddle.remainder, np.remainder, x, y, p_dtype
                    )


# ---------------------------------------------------------------------------
# divide: gains small/new integer widths (int8/int16/uint8 + int32/int64).
# Integer divide is C truncation (a / b), == np.trunc(a / b) for ints.
# ---------------------------------------------------------------------------


def _np_int_divide(x, y):
    # Paddle/MPS integer divide truncates toward zero (C semantics).
    q = x.astype(np.float64) / y.astype(np.float64)
    return np.trunc(q)


class TestMPSDivideInt(_MPSBinaryMathBase):
    def test_divide_signed_int(self):
        for np_dtype, p_dtype in _SIGNED_INTS:
            for shape in _SHAPES:
                with self.subTest(dtype=np_dtype, shape=shape):
                    x, y = _mixed_sign_xy(np_dtype, shape)
                    self._check_int_binary(
                        paddle.divide, _np_int_divide, x, y, p_dtype
                    )

    def test_divide_uint8(self):
        for np_dtype, p_dtype in _UNSIGNED_INTS:
            for shape in _SHAPES:
                with self.subTest(dtype=np_dtype, shape=shape):
                    x, y = _nonneg_xy(np_dtype, shape)
                    self._check_int_binary(
                        paddle.divide, _np_int_divide, x, y, p_dtype
                    )


# ---------------------------------------------------------------------------
# scale: gains small/new integer widths.
# ---------------------------------------------------------------------------


class TestMPSScaleInt(_MPSBinaryMathBase):
    def _check_scale_int(self, np_dtype, p_dtype, lo, hi):
        for shape in _SHAPES:
            with self.subTest(dtype=np_dtype, shape=shape):
                x = np.random.randint(lo, hi, size=shape).astype(np_dtype)
                out = paddle.scale(
                    paddle.to_tensor(x, place="mps"),
                    scale=2.0,
                    bias=1.0,
                    bias_after_scale=True,
                )
                cpu = paddle.scale(
                    paddle.to_tensor(x, place="cpu"),
                    scale=2.0,
                    bias=1.0,
                    bias_after_scale=True,
                )
                ref = (x.astype(np.int64) * 2 + 1).astype(np_dtype)
                self.assertEqual(out.dtype, p_dtype)
                self.assertTrue("mps" in str(out.place).lower())
                np.testing.assert_array_equal(out.numpy(), ref)
                np.testing.assert_array_equal(out.numpy(), cpu.numpy())

    def test_scale_signed_int(self):
        for np_dtype, p_dtype in _SIGNED_INTS:
            self._check_scale_int(np_dtype, p_dtype, -8, 9)

    def test_scale_uint8(self):
        for np_dtype, p_dtype in _UNSIGNED_INTS:
            self._check_scale_int(np_dtype, p_dtype, 0, 40)


# ---------------------------------------------------------------------------
# sign: gains small/new integer widths.
# ---------------------------------------------------------------------------


class TestMPSSignInt(_MPSBinaryMathBase):
    def test_sign_signed_int(self):
        for np_dtype, p_dtype in _SIGNED_INTS:
            for shape in _SHAPES:
                with self.subTest(dtype=np_dtype, shape=shape):
                    x = np.random.randint(-9, 10, size=shape).astype(np_dtype)
                    self._check_int_unary(paddle.sign, np.sign, x, p_dtype)

    def test_sign_uint8(self):
        for np_dtype, p_dtype in _UNSIGNED_INTS:
            for shape in _SHAPES:
                with self.subTest(dtype=np_dtype, shape=shape):
                    x = np.random.randint(0, 10, size=shape).astype(np_dtype)
                    self._check_int_unary(paddle.sign, np.sign, x, p_dtype)


# ---------------------------------------------------------------------------
# add / subtract / multiply / divide: gain complex64.
# ---------------------------------------------------------------------------


class TestMPSComplex64(_MPSBinaryMathBase):
    def _rand_c64(self, shape):
        return (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(
            np.complex64
        )

    def test_add_c64(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._rand_c64(shape)
                y = self._rand_c64(shape)
                self._check_c64_binary(paddle.add, np.add, x, y)

    def test_subtract_c64(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._rand_c64(shape)
                y = self._rand_c64(shape)
                self._check_c64_binary(paddle.subtract, np.subtract, x, y)

    def test_multiply_c64(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._rand_c64(shape)
                y = self._rand_c64(shape)
                self._check_c64_binary(paddle.multiply, np.multiply, x, y)

    def test_divide_c64(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._rand_c64(shape)
                # Keep |y| away from 0 to avoid blow-up in the comparison.
                y = self._rand_c64(shape)
                y = y + np.sign(y.real + (y.real == 0)).astype(np.complex64)
                self._check_c64_binary(paddle.divide, np.divide, x, y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
