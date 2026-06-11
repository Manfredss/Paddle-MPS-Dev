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
Dtype coverage for the MPS 'binary-math' kernel family.

Ops in the family:
  fmax, fmin            -> float16 + int32 + int64
  floor_divide          -> float16
  remainder             -> float16
  heaviside             -> float16
  atan2                 -> float16

float16 coverage (every op): run the op on float16 MPS tensors and compare the
result (cast to float32) to a float32 numpy/oracle reference with LOOSE
tolerance. We do NOT compare against the CPU backend in float16 because CPU may
not register float16 for these ops.

int32/int64 coverage (fmax/fmin only): compare the MPS result EXACTLY against
both a numpy integer reference and the CPU backend (which supports int).
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


_SHAPES = [(6,), (3, 4), (2, 3, 4)]


class _MPSBinaryMathTestBase(unittest.TestCase):
    """Common setUp for the binary-math dtype tests."""

    @classmethod
    def setUpClass(cls):
        if not _mps_available():
            raise unittest.SkipTest(
                "PaddlePaddle is not built with MPS or MPS is unavailable"
            )
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)


# ---------------------------------------------------------------------------
# float16 oracles (computed in float32, the op's natural compute precision).
# ---------------------------------------------------------------------------


def _fmax_np(x, y):
    return np.fmax(x, y)


def _fmin_np(x, y):
    return np.fmin(x, y)


def _floor_divide_np(x, y):
    # Matches floor(x / y) float semantics used by the kernel.
    return np.floor(x / y)


def _remainder_np(x, y):
    # Python-style modulo: result takes the sign of the divisor.
    return x - np.floor(x / y) * y


def _heaviside_np(x, y):
    return np.heaviside(x, y)


def _atan2_np(x, y):
    # Kernel computes atan2(x, y): x is numerator, y is denominator.
    return np.arctan2(x, y)


# Each entry: (name, paddle_fn, numpy_oracle, rtol, atol)
_FP16_OPS = (
    ("fmax", lambda a, b: paddle.fmax(a, b), _fmax_np, 2e-2, 2e-2),
    ("fmin", lambda a, b: paddle.fmin(a, b), _fmin_np, 2e-2, 2e-2),
    (
        "floor_divide",
        lambda a, b: paddle.floor_divide(a, b),
        _floor_divide_np,
        2e-2,
        2e-2,
    ),
    (
        "remainder",
        lambda a, b: paddle.remainder(a, b),
        _remainder_np,
        5e-2,
        5e-2,
    ),
    (
        "heaviside",
        lambda a, b: paddle.heaviside(a, b),
        _heaviside_np,
        2e-2,
        2e-2,
    ),
    ("atan2", lambda a, b: paddle.atan2(a, b), _atan2_np, 2e-2, 2e-2),
)


class TestMPSBinaryMathFloat16(_MPSBinaryMathTestBase):
    """float16 coverage for every op in the binary-math family."""

    def _make_inputs(self, name, shape):
        """Return (x32, y32) float32 arrays respecting each op's domain."""
        x32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
        y32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
        if name in ("floor_divide", "remainder", "atan2"):
            # Keep the divisor / denominator away from zero so the result is
            # well-defined and not dominated by float16 rounding near zero.
            y32 = np.where(
                np.abs(y32) < 0.5, np.sign(y32) + (y32 == 0) + y32, y32
            )
            y32 = y32.astype(np.float32)
        return x32, y32

    def _fp16_check(self, name, paddle_op, numpy_oracle, rtol, atol):
        for shape in _SHAPES:
            with self.subTest(op=name, shape=shape):
                x32, y32 = self._make_inputs(name, shape)
                x16 = x32.astype(np.float16)
                y16 = y32.astype(np.float16)

                out = paddle_op(
                    paddle.to_tensor(x16, place="mps"),
                    paddle.to_tensor(y16, place="mps"),
                )
                self.assertEqual(
                    out.dtype, paddle.float16, f"{name} output dtype"
                )
                self.assertTrue(
                    "mps" in str(out.place).lower(), f"{name} output place"
                )

                out_np = out.numpy().astype(np.float32)
                # Reference computed from the float16-rounded inputs in float32.
                ref = numpy_oracle(
                    x16.astype(np.float32), y16.astype(np.float32)
                ).astype(np.float32)
                np.testing.assert_allclose(
                    out_np,
                    ref,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"{name} float16 vs numpy oracle (shape={shape})",
                )

    def test_fmax_float16(self):
        self._fp16_check(*_FP16_OPS[0])

    def test_fmin_float16(self):
        self._fp16_check(*_FP16_OPS[1])

    def test_floor_divide_float16(self):
        self._fp16_check(*_FP16_OPS[2])

    def test_remainder_float16(self):
        self._fp16_check(*_FP16_OPS[3])

    def test_heaviside_float16(self):
        self._fp16_check(*_FP16_OPS[4])

    def test_atan2_float16(self):
        self._fp16_check(*_FP16_OPS[5])

    def test_fmax_fmin_nan_semantics_float16(self):
        # If one operand is NaN, fmax/fmin return the other operand.
        x = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float16)
        y = np.array([np.nan, 2.0, np.nan, np.nan], dtype=np.float16)
        x_p = paddle.to_tensor(x, place="mps")
        y_p = paddle.to_tensor(y, place="mps")
        out_max = paddle.fmax(x_p, y_p).numpy().astype(np.float32)
        out_min = paddle.fmin(x_p, y_p).numpy().astype(np.float32)
        ref_max = np.fmax(x.astype(np.float32), y.astype(np.float32))
        ref_min = np.fmin(x.astype(np.float32), y.astype(np.float32))
        # Both-NaN entry stays NaN; compare with equal_nan.
        np.testing.assert_allclose(
            out_max, ref_max, rtol=2e-2, atol=2e-2, equal_nan=True
        )
        np.testing.assert_allclose(
            out_min, ref_min, rtol=2e-2, atol=2e-2, equal_nan=True
        )


# ---------------------------------------------------------------------------
# Integer coverage: fmax / fmin only (int32 and int64).
# ---------------------------------------------------------------------------


_INT_OPS = (
    ("fmax", lambda a, b: paddle.fmax(a, b), np.maximum),
    ("fmin", lambda a, b: paddle.fmin(a, b), np.minimum),
)


class TestMPSBinaryMathInt(_MPSBinaryMathTestBase):
    """int32 / int64 coverage for fmax / fmin (no NaN, so plain max/min)."""

    def _int_check(self, name, paddle_op, numpy_op, np_dtype, paddle_dtype):
        for shape in _SHAPES:
            with self.subTest(op=name, shape=shape, dtype=np_dtype):
                x = np.random.randint(-8, 9, size=shape).astype(np_dtype)
                y = np.random.randint(-8, 9, size=shape).astype(np_dtype)

                out = paddle_op(
                    paddle.to_tensor(x, place="mps"),
                    paddle.to_tensor(y, place="mps"),
                )
                self.assertEqual(out.dtype, paddle_dtype, f"{name} dtype")
                self.assertTrue(
                    "mps" in str(out.place).lower(), f"{name} place"
                )
                out_np = out.numpy()

                ref = numpy_op(x, y)
                np.testing.assert_array_equal(
                    out_np,
                    ref,
                    err_msg=f"{name} {np_dtype} vs numpy (shape={shape})",
                )

                out_cpu = paddle_op(
                    paddle.to_tensor(x, place="cpu"),
                    paddle.to_tensor(y, place="cpu"),
                ).numpy()
                np.testing.assert_array_equal(
                    out_np,
                    out_cpu,
                    err_msg=f"{name} {np_dtype} vs cpu (shape={shape})",
                )

    def test_fmax_int32(self):
        self._int_check(
            "fmax", _INT_OPS[0][1], np.maximum, np.int32, paddle.int32
        )

    def test_fmax_int64(self):
        self._int_check(
            "fmax", _INT_OPS[0][1], np.maximum, np.int64, paddle.int64
        )

    def test_fmin_int32(self):
        self._int_check(
            "fmin", _INT_OPS[1][1], np.minimum, np.int32, paddle.int32
        )

    def test_fmin_int64(self):
        self._int_check(
            "fmin", _INT_OPS[1][1], np.minimum, np.int64, paddle.int64
        )

    def test_fmax_fmin_known_int_values(self):
        x = np.array([-3, 0, 5, 7], dtype=np.int32)
        y = np.array([2, -1, 5, 4], dtype=np.int32)
        x_p = paddle.to_tensor(x, place="mps")
        y_p = paddle.to_tensor(y, place="mps")
        np.testing.assert_array_equal(
            paddle.fmax(x_p, y_p).numpy(), np.maximum(x, y)
        )
        np.testing.assert_array_equal(
            paddle.fmin(x_p, y_p).numpy(), np.minimum(x, y)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
