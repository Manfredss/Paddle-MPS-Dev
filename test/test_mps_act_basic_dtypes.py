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
Dtype coverage for the MPS 'act-basic' kernel family.

Family ops: relu, silu, sigmoid, gelu, leaky_relu, softmax, relu6.
Every op in this family is registered for float32 and float16 on the MPS
backend. These tests exercise the float16 path (the kernels now plumb the
real tensor dtype through MPSGraphTensorData wrappers and constantWithScalar
calls instead of hardcoding MPSDataTypeFloat32).

For float16 we compare the MPS result (cast up to float32) against a float32
numpy / oracle reference with LOOSE tolerance. We do NOT compare against the
CPU backend in float16 because CPU may not register float16 for these ops and
would fail to dispatch.
"""

import math
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


# ---------------------------------------------------------------------------
# Numpy float32 oracles for the family ops.
# ---------------------------------------------------------------------------


def _relu_numpy(x):
    return np.maximum(x, 0.0).astype(x.dtype)


def _relu6_numpy(x):
    return np.minimum(np.maximum(x, 0.0), 6.0).astype(x.dtype)


def _sigmoid_numpy(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(x.dtype)


def _silu_numpy(x):
    return (x * (1.0 / (1.0 + np.exp(-x)))).astype(x.dtype)


def _leaky_relu_numpy(x, alpha):
    return np.where(x >= 0, x, alpha * x).astype(x.dtype)


def _softmax_numpy(x, axis):
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x_shift)
    return (ex / np.sum(ex, axis=axis, keepdims=True)).astype(x.dtype)


def _gelu_exact_numpy(x):
    from math import erf

    erf_vec = np.vectorize(erf, otypes=[np.float64])
    return (0.5 * x * (1.0 + erf_vec(x / math.sqrt(2.0)))).astype(x.dtype)


def _gelu_approx_numpy(x):
    beta = math.sqrt(2.0 / math.pi)
    inner = beta * (x + 0.044715 * np.power(x, 3))
    return (0.5 * x * (1.0 + np.tanh(inner))).astype(x.dtype)


_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class TestMPSActBasicFloat16(unittest.TestCase):
    """float16 coverage for relu/silu/sigmoid/gelu/leaky_relu/softmax/relu6."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    def _check_fp16(self, paddle_fn, ref_fn, x32, rtol=2e-2, atol=2e-2, msg=""):
        """Run paddle_fn on a float16 MPS tensor, compare to float32 oracle."""
        x16 = x32.astype(np.float16)
        out = paddle_fn(paddle.to_tensor(x16, place="mps"))
        # dtype / place assertions on the raw output tensor.
        self.assertEqual(out.dtype, paddle.float16, msg=f"{msg} dtype")
        self.assertTrue("mps" in str(out.place).lower(), msg=f"{msg} place")
        out32 = out.numpy().astype(np.float32)
        ref = ref_fn(x32.astype(np.float32)).astype(np.float32)
        np.testing.assert_allclose(
            out32, ref, rtol=rtol, atol=atol, err_msg=msg
        )

    # ---- relu -----------------------------------------------------------
    def test_relu_fp16(self):
        import paddle.nn.functional as F

        for shape in _SHAPES:
            x = (np.random.randn(*shape) * 1.5).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_fp16(F.relu, _relu_numpy, x, msg=f"relu {shape}")

    # ---- relu6 ----------------------------------------------------------
    def test_relu6_fp16(self):
        import paddle.nn.functional as F

        for shape in _SHAPES:
            # widen the range so values cross both the 0 and 6 clamp points.
            x = (np.random.randn(*shape) * 4.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_fp16(F.relu6, _relu6_numpy, x, msg=f"relu6 {shape}")

    # ---- sigmoid --------------------------------------------------------
    def test_sigmoid_fp16(self):
        import paddle.nn.functional as F

        for shape in _SHAPES:
            x = (np.random.randn(*shape) * 1.5).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_fp16(
                    F.sigmoid, _sigmoid_numpy, x, msg=f"sigmoid {shape}"
                )

    # ---- silu -----------------------------------------------------------
    def test_silu_fp16(self):
        import paddle.nn.functional as F

        for shape in _SHAPES:
            x = (np.random.randn(*shape) * 1.5).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_fp16(F.silu, _silu_numpy, x, msg=f"silu {shape}")

    # ---- leaky_relu -----------------------------------------------------
    def test_leaky_relu_fp16(self):
        import paddle.nn.functional as F

        for shape in _SHAPES:
            x = (np.random.randn(*shape) * 1.5).astype(np.float32)
            for alpha in (0.01, 0.1, 0.2):
                with self.subTest(shape=shape, alpha=alpha):
                    self._check_fp16(
                        lambda t, a=alpha: F.leaky_relu(t, negative_slope=a),
                        lambda v, a=alpha: _leaky_relu_numpy(v, a),
                        x,
                        msg=f"leaky_relu {shape} alpha={alpha}",
                    )

    # ---- softmax --------------------------------------------------------
    def test_softmax_fp16(self):
        import paddle.nn.functional as F

        for shape in _SHAPES:
            x = (np.random.randn(*shape) * 1.5).astype(np.float32)
            rank = len(shape)
            for axis in (-1, 0):
                with self.subTest(shape=shape, axis=axis):
                    norm_axis = axis if axis >= 0 else axis + rank
                    self._check_fp16(
                        lambda t, ax=axis: F.softmax(t, axis=ax),
                        lambda v, ax=norm_axis: _softmax_numpy(v, ax),
                        x,
                        msg=f"softmax {shape} axis={axis}",
                    )

    # ---- gelu -----------------------------------------------------------
    def test_gelu_fp16(self):
        import paddle.nn.functional as F

        for shape in _SHAPES:
            x = (np.random.randn(*shape) * 1.5).astype(np.float32)
            for approximate in (False, True):
                ref = _gelu_approx_numpy if approximate else _gelu_exact_numpy
                with self.subTest(shape=shape, approximate=approximate):
                    self._check_fp16(
                        lambda t, ap=approximate: F.gelu(t, approximate=ap),
                        ref,
                        x,
                        msg=f"gelu {shape} approximate={approximate}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
