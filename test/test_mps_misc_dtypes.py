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
Dtype coverage for the MPS "misc" kernel family.

Family / dtype policy:
- matmul   : float16 only (MPSGraph matrixMultiplication is float-only).
- scale    : float16 only.
- clip     : float16 + int32 + int64.
- where    : float16 + int32 + int64 (condition is always bool).
- isnan/isinf/isfinite : float16 only on the input; output stays bool.

float16 results are compared against a float32 numpy/oracle reference with loose
tolerance (CPU may not register float16 for these ops, so we never compare MPS
float16 against the CPU backend). int32/int64 results (clip/where only) are
compared EXACTLY against both a numpy reference and the CPU backend, which does
support integer dtypes.
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


# Loose tolerances for float16 (half precision accumulates noticeable error).
F16_RTOL = 2e-2
F16_ATOL = 2e-2
# Looser still for exp/pow/large-magnitude/matmul accumulation.
F16_RTOL_LOOSE = 5e-2
F16_ATOL_LOOSE = 5e-2

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSMiscDtypeBase(unittest.TestCase):
    """Common setUp for the misc-family dtype tests."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    # -- float16 helper ---------------------------------------------------
    def _assert_f16_out(self, out):
        """The MPS output must be float16 and live on the mps place."""
        self.assertEqual(out.dtype, paddle.float16)
        self.assertTrue("mps" in str(out.place).lower())

    def _run_f16(self, fn, x32, rtol=F16_RTOL, atol=F16_ATOL):
        """Run ``fn`` on a float16 MPS tensor; compare to the float32 oracle.

        ``fn`` takes a paddle tensor and returns a paddle tensor.
        Returns the MPS float16 output (as a paddle tensor) for extra asserts.
        """
        x16 = x32.astype(np.float16)
        out = fn(paddle.to_tensor(x16, place="mps"))
        self._assert_f16_out(out)
        ref = fn(paddle.to_tensor(x32, place="cpu")).numpy().astype(np.float32)
        np.testing.assert_allclose(
            out.numpy().astype(np.float32), ref, rtol=rtol, atol=atol
        )
        return out


# ---------------------------------------------------------------------------
# matmul: float16 only.
# ---------------------------------------------------------------------------


class TestMPSMatmulDtypes(_MPSMiscDtypeBase):
    def _matmul_f16(self, x32, y32):
        x16 = x32.astype(np.float16)
        y16 = y32.astype(np.float16)
        out = paddle.matmul(
            paddle.to_tensor(x16, place="mps"),
            paddle.to_tensor(y16, place="mps"),
        )
        self._assert_f16_out(out)
        # float32 numpy oracle; small K keeps the half accumulation error down.
        ref = (x32 @ y32).astype(np.float32)
        np.testing.assert_allclose(
            out.numpy().astype(np.float32),
            ref,
            rtol=F16_RTOL_LOOSE,
            atol=1e-1,
        )

    def test_matmul_2d_f16(self):
        # Modest range and small inner dim K to limit half accumulation error.
        x = (np.random.randn(3, 4) * 1.5).astype(np.float32)
        y = (np.random.randn(4, 5) * 1.5).astype(np.float32)
        self._matmul_f16(x, y)

    def test_matmul_square_f16(self):
        x = (np.random.randn(4, 4) * 1.5).astype(np.float32)
        y = (np.random.randn(4, 4) * 1.5).astype(np.float32)
        self._matmul_f16(x, y)

    def test_matmul_batched_f16(self):
        x = (np.random.randn(2, 3, 4) * 1.5).astype(np.float32)
        y = (np.random.randn(2, 4, 3) * 1.5).astype(np.float32)
        x16 = x.astype(np.float16)
        y16 = y.astype(np.float16)
        out = paddle.matmul(
            paddle.to_tensor(x16, place="mps"),
            paddle.to_tensor(y16, place="mps"),
        )
        self._assert_f16_out(out)
        ref = np.matmul(x, y).astype(np.float32)
        np.testing.assert_allclose(
            out.numpy().astype(np.float32),
            ref,
            rtol=F16_RTOL_LOOSE,
            atol=1e-1,
        )


# ---------------------------------------------------------------------------
# scale: float16 only.
# ---------------------------------------------------------------------------


class TestMPSScaleDtypes(_MPSMiscDtypeBase):
    def test_scale_bias_after_f16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_f16(
                    lambda t: paddle.scale(
                        t, scale=2.0, bias=0.5, bias_after_scale=True
                    ),
                    x,
                )

    def test_scale_bias_before_f16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_f16(
                    lambda t: paddle.scale(
                        t, scale=-1.5, bias=1.0, bias_after_scale=False
                    ),
                    x,
                )

    def test_scale_known_values_f16(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = paddle.scale(
            paddle.to_tensor(x.astype(np.float16), place="mps"),
            scale=3.0,
            bias=1.0,
            bias_after_scale=True,
        )
        self._assert_f16_out(out)
        np.testing.assert_allclose(
            out.numpy().astype(np.float32),
            np.array([4.0, 7.0, 10.0], dtype=np.float32),
            rtol=F16_RTOL,
            atol=F16_ATOL,
        )


# ---------------------------------------------------------------------------
# clip: float16 + int32 + int64.
# ---------------------------------------------------------------------------


class TestMPSClipDtypes(_MPSMiscDtypeBase):
    def test_clip_f16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_f16(lambda t: paddle.clip(t, min=-0.5, max=0.5), x)

    def _clip_int(self, np_dtype, paddle_dtype):
        for shape in _SHAPES:
            with self.subTest(shape=shape, dtype=np_dtype):
                x = np.random.randint(-8, 9, size=shape).astype(np_dtype)
                out = paddle.clip(
                    paddle.to_tensor(x, place="mps"), min=-3, max=3
                )
                cpu = paddle.clip(
                    paddle.to_tensor(x, place="cpu"), min=-3, max=3
                )
                ref = np.clip(x, -3, 3).astype(np_dtype)
                self.assertEqual(out.dtype, paddle_dtype)
                self.assertTrue("mps" in str(out.place).lower())
                np.testing.assert_array_equal(out.numpy(), ref)
                np.testing.assert_array_equal(out.numpy(), cpu.numpy())

    def test_clip_int32(self):
        self._clip_int(np.int32, paddle.int32)

    def test_clip_int64(self):
        self._clip_int(np.int64, paddle.int64)


# ---------------------------------------------------------------------------
# where: float16 + int32 + int64; condition is bool.
# ---------------------------------------------------------------------------


class TestMPSWhereDtypes(_MPSMiscDtypeBase):
    def test_where_f16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                cond = np.random.randint(0, 2, size=shape).astype(np.bool_)
                x32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
                y32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
                cond_p = paddle.to_tensor(cond, place="mps")
                out = paddle.where(
                    cond_p,
                    paddle.to_tensor(x32.astype(np.float16), place="mps"),
                    paddle.to_tensor(y32.astype(np.float16), place="mps"),
                )
                self._assert_f16_out(out)
                ref = np.where(cond, x32, y32).astype(np.float32)
                np.testing.assert_allclose(
                    out.numpy().astype(np.float32),
                    ref,
                    rtol=F16_RTOL,
                    atol=F16_ATOL,
                )

    def _where_int(self, np_dtype, paddle_dtype):
        for shape in _SHAPES:
            with self.subTest(shape=shape, dtype=np_dtype):
                cond = np.random.randint(0, 2, size=shape).astype(np.bool_)
                x = np.random.randint(-8, 9, size=shape).astype(np_dtype)
                y = np.random.randint(-8, 9, size=shape).astype(np_dtype)
                out = paddle.where(
                    paddle.to_tensor(cond, place="mps"),
                    paddle.to_tensor(x, place="mps"),
                    paddle.to_tensor(y, place="mps"),
                )
                cpu = paddle.where(
                    paddle.to_tensor(cond, place="cpu"),
                    paddle.to_tensor(x, place="cpu"),
                    paddle.to_tensor(y, place="cpu"),
                )
                ref = np.where(cond, x, y).astype(np_dtype)
                self.assertEqual(out.dtype, paddle_dtype)
                self.assertTrue("mps" in str(out.place).lower())
                np.testing.assert_array_equal(out.numpy(), ref)
                np.testing.assert_array_equal(out.numpy(), cpu.numpy())

    def test_where_int32(self):
        self._where_int(np.int32, paddle.int32)

    def test_where_int64(self):
        self._where_int(np.int64, paddle.int64)


# ---------------------------------------------------------------------------
# isnan / isinf / isfinite: float16 input, bool output.
# ---------------------------------------------------------------------------


class TestMPSIsfiniteDtypes(_MPSMiscDtypeBase):
    _OPS = (
        ("isnan", lambda t: paddle.isnan(t), np.isnan),
        ("isinf", lambda t: paddle.isinf(t), np.isinf),
        ("isfinite", lambda t: paddle.isfinite(t), np.isfinite),
    )

    def test_finite_inputs_f16(self):
        for shape in _SHAPES:
            x32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
            x16 = x32.astype(np.float16)
            for name, p_op, n_op in self._OPS:
                with self.subTest(op=name, shape=shape):
                    out = p_op(paddle.to_tensor(x16, place="mps"))
                    self.assertEqual(out.dtype, paddle.bool)
                    self.assertTrue("mps" in str(out.place).lower())
                    ref = n_op(x16.astype(np.float32))
                    np.testing.assert_array_equal(out.numpy(), ref)

    def test_special_values_f16(self):
        # Mix of normal, +inf, -inf, nan, all representable in float16.
        x16 = np.array(
            [0.0, 1.5, -2.0, np.inf, -np.inf, np.nan], dtype=np.float16
        )
        for name, p_op, n_op in self._OPS:
            with self.subTest(op=name):
                out = p_op(paddle.to_tensor(x16, place="mps"))
                self.assertEqual(out.dtype, paddle.bool)
                ref = n_op(x16.astype(np.float32))
                np.testing.assert_array_equal(out.numpy(), ref)


if __name__ == "__main__":
    unittest.main(verbosity=2)
