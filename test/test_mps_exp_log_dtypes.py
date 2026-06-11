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
Dtype coverage for the MPS 'exp-log' kernel family.

Family ops (unary):
  exp, expm1, log, log2, log10, log1p, erf, sqrt, rsqrt, reciprocal, square.

These kernels were originally float32-only; they now also register float16 by
plumbing the real tensor dtype through every MPSGraphTensorData wrapper and
constantWithScalar call instead of hardcoding MPSDataTypeFloat32.

float16: run each op on an fp16 input placed on 'mps', cast the result back to
float32, and compare against a float32 numpy oracle with LOOSE tolerances.  We
do NOT compare against the CPU backend in float16 (CPU may not register fp16 for
these ops and would fail to dispatch).

int32/int64: only 'square' marks itself int-capable (it registers `int`).  For
square we additionally compare MPS exactly against numpy AND the CPU backend.
"""

import unittest

import numpy as np

try:
    import paddle

    PADDLE_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    PADDLE_AVAILABLE = False
    print("Warning: PaddlePaddle not available")


def _mps_available():
    return (
        PADDLE_AVAILABLE
        and paddle.is_compiled_with_mps()
        and paddle.mps.is_available()
    )


# Small shapes exercised by every op.
_SHAPES = [(6,), (3, 4), (2, 3, 4)]


class _MPSDtypeTestBase(unittest.TestCase):
    """Common setUp guard for the exp-log dtype tests."""

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


class TestMPSExpLogFloat16(_MPSDtypeTestBase):
    """float16 coverage for every op in the exp-log family."""

    def _check_fp16(self, paddle_op, numpy_ref, make_x32, rtol=2e-2, atol=2e-2):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x32 = make_x32(shape).astype(np.float32)
                x16 = x32.astype(np.float16)

                out = paddle_op(paddle.to_tensor(x16, place="mps"))
                # Output should stay float16 and live on MPS.
                self.assertEqual(out.dtype, paddle.float16)
                self.assertTrue("mps" in str(out.place).lower())

                out32 = out.numpy().astype(np.float32)
                ref = numpy_ref(x32).astype(np.float32)
                np.testing.assert_allclose(out32, ref, rtol=rtol, atol=atol)

    # ---- domain helpers -------------------------------------------------
    @staticmethod
    def _modest(shape):
        # Symmetric, modest magnitude (good for exp/expm1/erf/square).
        return np.random.randn(*shape) * 1.5

    @staticmethod
    def _positive(shape):
        # Strictly positive (log/log2/log10/sqrt/rsqrt).
        return np.abs(np.random.randn(*shape)) * 1.5 + 0.5

    @staticmethod
    def _gt_minus_one(shape):
        # > -1 for log1p; keep comfortably above the singularity.
        return np.abs(np.random.randn(*shape)) * 1.5 + 0.25

    @staticmethod
    def _nonzero(shape):
        # Away from 0 for reciprocal.
        x = np.random.randn(*shape) * 1.5
        x = np.where(np.abs(x) < 0.5, x + np.sign(x + 1e-3) * 0.5, x)
        return x

    # ---- exponentials (looser tolerance) --------------------------------
    def test_exp(self):
        self._check_fp16(paddle.exp, np.exp, self._modest, rtol=5e-2, atol=5e-2)

    def test_expm1(self):
        self._check_fp16(
            paddle.expm1, np.expm1, self._modest, rtol=5e-2, atol=5e-2
        )

    # ---- logarithms -----------------------------------------------------
    def test_log(self):
        self._check_fp16(paddle.log, np.log, self._positive)

    def test_log2(self):
        self._check_fp16(paddle.log2, np.log2, self._positive)

    def test_log10(self):
        self._check_fp16(paddle.log10, np.log10, self._positive)

    def test_log1p(self):
        self._check_fp16(paddle.log1p, np.log1p, self._gt_minus_one)

    # ---- erf ------------------------------------------------------------
    def test_erf(self):
        from math import erf as _erf

        erf_vec = np.vectorize(_erf, otypes=[np.float64])
        self._check_fp16(paddle.erf, lambda x: erf_vec(x), self._modest)

    # ---- roots / reciprocals -------------------------------------------
    def test_sqrt(self):
        self._check_fp16(paddle.sqrt, np.sqrt, self._positive)

    def test_rsqrt(self):
        self._check_fp16(
            paddle.rsqrt, lambda x: 1.0 / np.sqrt(x), self._positive
        )

    def test_reciprocal(self):
        self._check_fp16(paddle.reciprocal, lambda x: 1.0 / x, self._nonzero)

    # ---- square ---------------------------------------------------------
    def test_square(self):
        self._check_fp16(paddle.square, np.square, self._modest)


class TestMPSSquareInteger(_MPSDtypeTestBase):
    """Integer coverage for the one int-capable op in the family: square."""

    def _check_int(self, np_dtype, paddle_dtype):
        for shape in _SHAPES:
            with self.subTest(shape=shape, dtype=np_dtype):
                x = np.random.randint(-8, 9, size=shape).astype(np_dtype)

                out_mps = paddle.square(paddle.to_tensor(x, place="mps"))
                self.assertEqual(out_mps.dtype, paddle_dtype)
                self.assertTrue("mps" in str(out_mps.place).lower())

                out_mps_np = out_mps.numpy()
                out_cpu_np = paddle.square(
                    paddle.to_tensor(x, place="cpu")
                ).numpy()
                ref = np.square(x)

                np.testing.assert_array_equal(out_mps_np, ref)
                np.testing.assert_array_equal(out_mps_np, out_cpu_np)

    def test_square_int32(self):
        self._check_int(np.int32, paddle.int32)

    def test_square_int64(self):
        self._check_int(np.int64, paddle.int64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
