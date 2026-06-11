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
Extra dtype coverage for the MPS "misc" kernel family.

This complements ``test_mps_misc_dtypes.py`` by exercising the dtypes that were
newly registered for the misc family:

- matmul   : + bfloat16 (MPSGraph matmul on macOS 14+/M2+; no int, no complex).
- scale    : + bfloat16 and + uint8/int8/int16/int32/int64 integer widths.
- clip     : + bfloat16 (int32/int64 already covered elsewhere).
- where    : + bfloat16 (condition stays bool).
- isnan/isinf/isfinite : + bfloat16 input (output stays bool).

bfloat16 maps to ``MPSDataTypeBFloat16`` which only exists in the macOS-14 SDK
and only runs on macOS 14+. This CI host may lack it, so every bfloat16 subtest
is guarded by a runtime probe (``_supports``) that creates a tiny bf16 mps tensor
and runs a trivial op; if that raises, the subtest is skipped.

bfloat16 has ~3 decimal digits of precision, so its results are compared against
a float32 numpy oracle with loose tolerance (rtol=atol=4e-2). Integer results are
compared EXACTLY against both a numpy reference and the CPU backend.
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
    """Probe whether a dtype actually works on the MPS device at runtime.

    bfloat16 (and complex64) require the macOS-14 SDK / runtime. Create a tiny
    mps tensor of that dtype, run a trivial add, and return False on ANY
    exception so the caller can skip the corresponding subtest.
    """
    if not _mps_available():
        return False
    try:
        if paddle_dtype == paddle.bfloat16:
            # bf16 has no numpy dtype: build float32 on mps, then cast.
            t = paddle.to_tensor(
                np.zeros((2,), dtype=np.float32), place="mps"
            ).astype("bfloat16")
        else:
            t = paddle.to_tensor(
                np.zeros((2,)).astype("float32"), place="mps"
            ).astype(paddle_dtype)
        _ = (t + t).astype("float32").numpy()
        return True
    except Exception:
        return False


# bf16 tolerance: ~3 decimal digits, looser than fp16.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSMiscExtraBase(unittest.TestCase):
    """Common setUp + helpers for the misc-family extra-dtype tests."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        cls.bf16_ok = _supports(paddle.bfloat16)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    def _skip_if_no_bf16(self):
        if not self.bf16_ok:
            self.skipTest("bfloat16 unsupported on this MPS runtime/SDK")

    def _bf16_tensor(self, x32):
        """A paddle.bfloat16 mps tensor built from a float32 numpy array."""
        return paddle.to_tensor(x32.astype(np.float32), place="mps").astype(
            "bfloat16"
        )

    def _assert_bf16_out(self, out):
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())

    def _run_bf16(self, fn, x32, ref, rtol=BF16_RTOL, atol=BF16_ATOL):
        """Run ``fn`` on a bf16 mps tensor, compare to a float32 numpy oracle.

        ``ref`` is the float32 numpy expected result.
        """
        out = fn(self._bf16_tensor(x32))
        self._assert_bf16_out(out)
        np.testing.assert_allclose(
            out.astype("float32").numpy(),
            ref.astype(np.float32),
            rtol=rtol,
            atol=atol,
        )
        return out


# ---------------------------------------------------------------------------
# matmul: + bfloat16.
# ---------------------------------------------------------------------------


class TestMPSMatmulExtra(_MPSMiscExtraBase):
    def _matmul_bf16(self, x32, y32):
        self._skip_if_no_bf16()
        out = paddle.matmul(self._bf16_tensor(x32), self._bf16_tensor(y32))
        self._assert_bf16_out(out)
        ref = np.matmul(x32, y32).astype(np.float32)
        # bf16 matmul accumulates error; allow a generous absolute floor.
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, rtol=1e-1, atol=2e-1
        )

    def test_matmul_2d_bf16(self):
        x = (np.random.randn(3, 4) * 1.5).astype(np.float32)
        y = (np.random.randn(4, 5) * 1.5).astype(np.float32)
        self._matmul_bf16(x, y)

    def test_matmul_square_bf16(self):
        x = (np.random.randn(4, 4) * 1.5).astype(np.float32)
        y = (np.random.randn(4, 4) * 1.5).astype(np.float32)
        self._matmul_bf16(x, y)

    def test_matmul_batched_bf16(self):
        x = (np.random.randn(2, 3, 4) * 1.5).astype(np.float32)
        y = (np.random.randn(2, 4, 3) * 1.5).astype(np.float32)
        self._matmul_bf16(x, y)


# ---------------------------------------------------------------------------
# scale: + bfloat16 and + integer widths.
# ---------------------------------------------------------------------------


class TestMPSScaleExtra(_MPSMiscExtraBase):
    def test_scale_bias_after_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                self._skip_if_no_bf16()
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                ref = 2.0 * x + 0.5
                self._run_bf16(
                    lambda t: paddle.scale(
                        t, scale=2.0, bias=0.5, bias_after_scale=True
                    ),
                    x,
                    ref,
                )

    def test_scale_bias_before_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                self._skip_if_no_bf16()
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                ref = -1.5 * (x + 1.0)
                self._run_bf16(
                    lambda t: paddle.scale(
                        t, scale=-1.5, bias=1.0, bias_after_scale=False
                    ),
                    x,
                    ref,
                )

    def _scale_int(
        self, np_dtype, paddle_dtype, low, high, scale_v=2, bias_v=1
    ):
        # Integer scale computes T(scale) * x + T(bias) directly in T, matching
        # the CPU EigenScale functor (scale.to<T>()/bias.to<T>()) and the GPU
        # ScaleFunctor (MT == T for integer T). The scalars are cast to the
        # integer type *first*, so a fractional scale/bias (e.g. 2.5/1.5) is
        # truncated to an integer (2/1) before the arithmetic. The numpy oracle
        # mirrors that with ``np_dtype(scale_v)`` / ``np_dtype(bias_v)`` and the
        # result is compared EXACTLY against both numpy and the CPU backend.
        for shape in _SHAPES:
            with self.subTest(
                shape=shape, dtype=np_dtype, scale=scale_v, bias=bias_v
            ):
                x = np.random.randint(low, high, size=shape).astype(np_dtype)
                out = paddle.scale(
                    paddle.to_tensor(x, place="mps"),
                    scale=scale_v,
                    bias=bias_v,
                    bias_after_scale=True,
                )
                cpu = paddle.scale(
                    paddle.to_tensor(x, place="cpu"),
                    scale=scale_v,
                    bias=bias_v,
                    bias_after_scale=True,
                )
                ref = (np_dtype(scale_v) * x + np_dtype(bias_v)).astype(
                    np_dtype
                )
                self.assertEqual(out.dtype, paddle_dtype)
                self.assertTrue("mps" in str(out.place).lower())
                np.testing.assert_array_equal(out.numpy(), ref)
                np.testing.assert_array_equal(out.numpy(), cpu.numpy())

    def test_scale_int8(self):
        self._scale_int(np.int8, paddle.int8, -8, 9)

    def test_scale_int16(self):
        self._scale_int(np.int16, paddle.int16, -8, 9)

    def test_scale_int32(self):
        self._scale_int(np.int32, paddle.int32, -8, 9)

    def test_scale_int64(self):
        self._scale_int(np.int64, paddle.int64, -8, 9)

    def test_scale_uint8(self):
        # uint8 stays non-negative; 2*x+1 with x in [0, 60] stays in range.
        self._scale_int(np.uint8, paddle.uint8, 0, 61)

    def test_scale_int_fractional_scale_bias(self):
        # Fractional scale/bias on integer dtypes must be cast to the integer
        # type BEFORE the arithmetic (scale.to<T>()), so 2.5 -> 2 and 1.5 -> 1.
        # This guards against the MPS kernel feeding a fractional value into an
        # integer-typed constant and relying on implicit float->int rounding,
        # which would diverge from CPU/GPU. ``np_dtype(2.5)`` truncates toward
        # zero, matching the C++ static_cast in the functors.
        self._scale_int(np.int32, paddle.int32, -8, 9, scale_v=2.5, bias_v=1.5)
        self._scale_int(np.int64, paddle.int64, -8, 9, scale_v=2.5, bias_v=1.5)
        # Negative fractional scale: -1.5 -> -1 in integer space.
        self._scale_int(np.int32, paddle.int32, -8, 9, scale_v=-1.5, bias_v=2.5)


# ---------------------------------------------------------------------------
# clip: + bfloat16.
# ---------------------------------------------------------------------------


class TestMPSClipExtra(_MPSMiscExtraBase):
    def test_clip_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                self._skip_if_no_bf16()
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                ref = np.clip(x, -0.5, 0.5)
                self._run_bf16(
                    lambda t: paddle.clip(t, min=-0.5, max=0.5), x, ref
                )


# ---------------------------------------------------------------------------
# where: + bfloat16; condition is bool.
# ---------------------------------------------------------------------------


class TestMPSWhereExtra(_MPSMiscExtraBase):
    def test_where_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                self._skip_if_no_bf16()
                cond = np.random.randint(0, 2, size=shape).astype(np.bool_)
                x32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
                y32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
                cond_p = paddle.to_tensor(cond, place="mps")
                out = paddle.where(
                    cond_p,
                    self._bf16_tensor(x32),
                    self._bf16_tensor(y32),
                )
                self._assert_bf16_out(out)
                ref = np.where(cond, x32, y32).astype(np.float32)
                np.testing.assert_allclose(
                    out.astype("float32").numpy(),
                    ref,
                    rtol=BF16_RTOL,
                    atol=BF16_ATOL,
                )


# ---------------------------------------------------------------------------
# isnan / isinf / isfinite: + bfloat16 input, bool output.
# ---------------------------------------------------------------------------


class TestMPSIsfiniteExtra(_MPSMiscExtraBase):
    _OPS = (
        ("isnan", lambda t: paddle.isnan(t), np.isnan),
        ("isinf", lambda t: paddle.isinf(t), np.isinf),
        ("isfinite", lambda t: paddle.isfinite(t), np.isfinite),
    )

    def test_finite_inputs_bf16(self):
        for shape in _SHAPES:
            self._skip_if_no_bf16()
            x32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
            for name, p_op, n_op in self._OPS:
                with self.subTest(op=name, shape=shape):
                    out = p_op(self._bf16_tensor(x32))
                    self.assertEqual(out.dtype, paddle.bool)
                    self.assertTrue("mps" in str(out.place).lower())
                    ref = n_op(x32)
                    np.testing.assert_array_equal(out.numpy(), ref)

    def test_special_values_bf16(self):
        # Mix of normal, +inf, -inf, nan. bf16 has the same exponent range as
        # float32, so inf/nan survive the cast.
        self._skip_if_no_bf16()
        x32 = np.array(
            [0.0, 1.5, -2.0, np.inf, -np.inf, np.nan], dtype=np.float32
        )
        for name, p_op, n_op in self._OPS:
            with self.subTest(op=name):
                out = p_op(self._bf16_tensor(x32))
                self.assertEqual(out.dtype, paddle.bool)
                ref = n_op(x32)
                np.testing.assert_array_equal(out.numpy(), ref)


if __name__ == "__main__":
    unittest.main(verbosity=2)
