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
Dtype coverage for the MPS 'act-attr2' activation kernel family.

Family ops (all float16-capable, float16 only — no integer variants):
  hardsigmoid, hardswish, hardtanh, hard_shrink, softshrink,
  tanh_shrink, thresholded_relu, stanh, logit.

Each op is exercised in float16 on the MPS backend and compared against a
float32 NumPy oracle with loose tolerance. We do NOT compare against the CPU
backend in float16 (CPU may not register float16 for these ops and would fail
to dispatch). Output dtype and placement are asserted to stay on MPS/float16.
"""

import unittest

import numpy as np

try:
    import paddle
    import paddle.nn.functional as F

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


# ---------------------------------------------------------------------------
# Float32 NumPy oracles for each op.
# ---------------------------------------------------------------------------


def _hardsigmoid_numpy(x, slope=1.0 / 6.0, offset=0.5):
    return np.clip(slope * x + offset, 0.0, 1.0)


def _hardswish_numpy(x):
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def _hardtanh_numpy(x, t_min=-1.0, t_max=1.0):
    return np.clip(x, t_min, t_max)


def _hard_shrink_numpy(x, threshold=0.5):
    return np.where((x > threshold) | (x < -threshold), x, 0.0)


def _softshrink_numpy(x, lambd=0.5):
    out = np.where(x > lambd, x - lambd, 0.0)
    out = np.where(x < -lambd, x + lambd, out)
    return out


def _tanh_shrink_numpy(x):
    return x - np.tanh(x)


def _thresholded_relu_numpy(x, threshold=1.0, value=0.0):
    return np.where(x > threshold, x, value)


def _stanh_numpy(x, scale_a=0.67, scale_b=1.7159):
    return scale_b * np.tanh(scale_a * x)


def _logit_numpy(x, eps=0.0):
    if eps > 0.0:
        x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class TestMPSActAttr2Float16(unittest.TestCase):
    """Float16 coverage for the act-attr2 activation family on MPS."""

    SHAPES = [(6,), (3, 4), (2, 3, 4)]

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    # -- shared float16 driver -------------------------------------------
    def _check_fp16(self, paddle_op, numpy_op, x32, rtol=2e-2, atol=2e-2):
        """Run paddle_op on the float16 cast of x32 on MPS; compare to the
        float32 numpy oracle evaluated on x32."""
        x16 = x32.astype(np.float16)
        out = paddle_op(paddle.to_tensor(x16, place="mps"))
        # dtype / placement invariants
        self.assertEqual(out.dtype, paddle.float16)
        self.assertTrue("mps" in str(out.place).lower())
        out_np = out.numpy().astype(np.float32)
        ref = numpy_op(x32).astype(np.float32)
        np.testing.assert_allclose(out_np, ref, rtol=rtol, atol=atol)

    def _modest(self, shape):
        # Modest magnitude keeps float16 rounding error small.
        return (np.random.randn(*shape) * 1.5).astype(np.float32)

    # -- hardsigmoid ------------------------------------------------------
    def test_hardsigmoid(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x = self._modest(shape)
                self._check_fp16(F.hardsigmoid, _hardsigmoid_numpy, x)

    # -- hardswish --------------------------------------------------------
    def test_hardswish(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x = self._modest(shape)
                self._check_fp16(F.hardswish, _hardswish_numpy, x)

    # -- hardtanh ---------------------------------------------------------
    def test_hardtanh(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x = self._modest(shape)
                self._check_fp16(F.hardtanh, _hardtanh_numpy, x)

    # -- hard_shrink ------------------------------------------------------
    def test_hard_shrink(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x = self._modest(shape)
                self._check_fp16(F.hardshrink, _hard_shrink_numpy, x)

    # -- softshrink -------------------------------------------------------
    def test_softshrink(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x = self._modest(shape)
                self._check_fp16(F.softshrink, _softshrink_numpy, x)

    # -- tanh_shrink ------------------------------------------------------
    def test_tanh_shrink(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x = self._modest(shape)
                self._check_fp16(F.tanhshrink, _tanh_shrink_numpy, x)

    # -- thresholded_relu -------------------------------------------------
    def test_thresholded_relu(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x = self._modest(shape)
                self._check_fp16(F.thresholded_relu, _thresholded_relu_numpy, x)

    # -- stanh ------------------------------------------------------------
    def test_stanh(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x = self._modest(shape)
                self._check_fp16(F.stanh, _stanh_numpy, x)

    # -- logit (domain (0, 1); test the default eps == 0 path) ------------
    def test_logit_no_eps(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                # x in (0, 1), kept away from the 0/1 boundaries so the
                # log(x/(1-x)) stays finite and representable in float16.
                x = np.random.uniform(0.2, 0.8, size=shape).astype(np.float32)
                self._check_fp16(
                    lambda t: paddle.logit(t),
                    lambda a: _logit_numpy(a, eps=0.0),
                    x,
                )

    # -- logit with an eps clamp (eps > 0 builds the clamp constants) -----
    def test_logit_with_eps(self):
        eps = 1e-2
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                # Include values near (and beyond) the boundary so the clamp
                # path is genuinely exercised.
                x = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
                self._check_fp16(
                    lambda t: paddle.logit(t, eps=eps),
                    lambda a: _logit_numpy(a, eps=eps),
                    x,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
