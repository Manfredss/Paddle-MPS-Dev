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
Dtype coverage for the MPS 'act-attr1' activation kernel family.

Family ops (all attribute-carrying / scalar-constant activations):
    elu, celu, selu, mish, softplus, softsign, logsigmoid, swish

These kernels were float32-only; they now also register float16 by plumbing
the real tensor dtype through every MPSGraphTensorData wrapper and every
constantWithScalar call (instead of hardcoding MPSDataTypeFloat32).

This module exercises the float16 path on MPS. We compare the MPS float16
result (cast back to float32) against a float32 numpy oracle with LOOSE
tolerances. We deliberately do NOT compare against the CPU backend in
float16 -- CPU may not register float16 for these activations and would fail
to dispatch. float32 is also checked as a sanity baseline.
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
# float32 numpy oracles for each family op.
# ---------------------------------------------------------------------------


def _elu_numpy(x, alpha):
    return np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)


def _celu_numpy(x, alpha):
    return np.where(x >= 0.0, x, alpha * (np.exp(x / alpha) - 1.0)).astype(
        np.float32
    )


def _selu_numpy(x):
    # Default scale / alpha values used by paddle.nn.functional.selu.
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return (scale * np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))).astype(
        np.float32
    )


def _softplus_numpy(x, beta, threshold):
    bx = beta * x
    soft = np.log1p(np.exp(bx)) / beta
    return np.where(bx > threshold, x, soft).astype(np.float32)


def _mish_numpy(x):
    sp = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)  # stable softplus
    return (x * np.tanh(sp)).astype(np.float32)


def _softsign_numpy(x):
    return (x / (1.0 + np.abs(x))).astype(np.float32)


def _logsigmoid_numpy(x):
    # -log(1 + exp(-x)); numerically stable form min(x,0) - log(1+exp(-|x|)).
    return (np.minimum(x, 0.0) - np.log1p(np.exp(-np.abs(x)))).astype(
        np.float32
    )


def _swish_numpy(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


_SHAPES = [(6,), (3, 4), (2, 3, 4)]


class TestMPSActAttr1Dtypes(unittest.TestCase):
    """float16 (and float32 baseline) coverage for the act-attr1 family."""

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
        paddle.seed(2026)

    # -- shared check helpers --------------------------------------------

    def _run_mps(self, paddle_op, x_np):
        """Run op on MPS for a given numpy array, return paddle output tensor."""
        return paddle_op(paddle.to_tensor(x_np, place="mps"))

    def _check_fp16(self, name, paddle_op, oracle, x32, rtol=2e-2, atol=2e-2):
        """Run op in float16 on MPS, compare to a float32 numpy oracle."""
        x16 = x32.astype(np.float16)
        out = self._run_mps(paddle_op, x16)
        # dtype / place assertions
        self.assertEqual(
            out.dtype,
            paddle.float16,
            f"{name}: float16 input must yield float16 output",
        )
        self.assertTrue(
            "mps" in str(out.place).lower(), f"{name}: output must live on mps"
        )
        out_np = out.numpy().astype(np.float32)
        ref = oracle(x32)
        np.testing.assert_allclose(
            out_np,
            ref,
            rtol=rtol,
            atol=atol,
            err_msg=f"{name} float16 vs float32 oracle",
        )

    def _check_fp32(self, name, paddle_op, oracle, x32, rtol=1e-4, atol=1e-5):
        """Sanity baseline: float32 MPS vs float32 numpy oracle."""
        out = self._run_mps(paddle_op, x32)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())
        out_np = out.numpy().astype(np.float32)
        ref = oracle(x32)
        np.testing.assert_allclose(
            out_np,
            ref,
            rtol=rtol,
            atol=atol,
            err_msg=f"{name} float32 vs float32 oracle",
        )

    def _modest_x(self, shape):
        """A modest-range float32 array (domain-unrestricted ops)."""
        return (np.random.randn(*shape) * 1.5).astype(np.float32)

    # -- elu --------------------------------------------------------------

    def test_elu_fp16(self):
        alpha = 1.0
        op = lambda t: F.elu(t, alpha=alpha)
        oracle = lambda x: _elu_numpy(x, alpha)
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._modest_x(shape)
                self._check_fp32("elu", op, oracle, x)
                self._check_fp16("elu", op, oracle, x)

    def test_elu_fp16_various_alpha(self):
        for alpha in (0.5, 1.0, 2.0):
            op = lambda t, a=alpha: F.elu(t, alpha=a)
            oracle = lambda x, a=alpha: _elu_numpy(x, a)
            with self.subTest(alpha=alpha):
                x = self._modest_x((3, 4))
                self._check_fp16("elu", op, oracle, x)

    # -- celu -------------------------------------------------------------

    def test_celu_fp16(self):
        alpha = 1.0
        op = lambda t: F.celu(t, alpha=alpha)
        oracle = lambda x: _celu_numpy(x, alpha)
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._modest_x(shape)
                self._check_fp32("celu", op, oracle, x)
                self._check_fp16("celu", op, oracle, x)

    def test_celu_fp16_various_alpha(self):
        for alpha in (0.5, 1.0, 2.0):
            op = lambda t, a=alpha: F.celu(t, alpha=a)
            oracle = lambda x, a=alpha: _celu_numpy(x, a)
            with self.subTest(alpha=alpha):
                x = self._modest_x((3, 4))
                self._check_fp16("celu", op, oracle, x)

    # -- selu -------------------------------------------------------------

    def test_selu_fp16(self):
        op = lambda t: F.selu(t)
        oracle = _selu_numpy
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._modest_x(shape)
                self._check_fp32("selu", op, oracle, x)
                # selu multiplies by scale ~1.05 -> use a slightly looser tol.
                self._check_fp16("selu", op, oracle, x, rtol=5e-2, atol=5e-2)

    # -- mish -------------------------------------------------------------

    def test_mish_fp16(self):
        op = lambda t: F.mish(t)
        oracle = _mish_numpy
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._modest_x(shape)
                self._check_fp32("mish", op, oracle, x)
                self._check_fp16("mish", op, oracle, x)

    # -- softplus ---------------------------------------------------------

    def test_softplus_fp16(self):
        beta, threshold = 1.0, 20.0
        op = lambda t: F.softplus(t, beta=beta, threshold=threshold)
        oracle = lambda x: _softplus_numpy(x, beta, threshold)
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._modest_x(shape)
                self._check_fp32("softplus", op, oracle, x)
                self._check_fp16("softplus", op, oracle, x)

    def test_softplus_fp16_various_beta(self):
        threshold = 20.0
        for beta in (0.5, 1.0, 2.0):
            op = lambda t, b=beta: F.softplus(t, beta=b, threshold=threshold)
            oracle = lambda x, b=beta: _softplus_numpy(x, b, threshold)
            with self.subTest(beta=beta):
                x = self._modest_x((3, 4))
                self._check_fp16("softplus", op, oracle, x)

    # -- softsign ---------------------------------------------------------

    def test_softsign_fp16(self):
        op = lambda t: F.softsign(t)
        oracle = _softsign_numpy
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._modest_x(shape)
                self._check_fp32("softsign", op, oracle, x)
                self._check_fp16("softsign", op, oracle, x)

    # -- logsigmoid -------------------------------------------------------

    def test_logsigmoid_fp16(self):
        op = lambda t: F.log_sigmoid(t)
        oracle = _logsigmoid_numpy
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._modest_x(shape)
                self._check_fp32("logsigmoid", op, oracle, x)
                self._check_fp16("logsigmoid", op, oracle, x)

    # -- swish ------------------------------------------------------------

    def test_swish_fp16(self):
        op = lambda t: F.swish(t)
        oracle = _swish_numpy
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = self._modest_x(shape)
                self._check_fp32("swish", op, oracle, x)
                self._check_fp16("swish", op, oracle, x)

    # -- shared dtype/place assertions -----------------------------------

    def test_all_ops_fp16_dtype_and_place(self):
        x = self._modest_x((3, 4)).astype(np.float16)
        ops = (
            ("elu", lambda t: F.elu(t, alpha=1.0)),
            ("celu", lambda t: F.celu(t, alpha=1.0)),
            ("selu", lambda t: F.selu(t)),
            ("mish", lambda t: F.mish(t)),
            ("softplus", lambda t: F.softplus(t)),
            ("softsign", lambda t: F.softsign(t)),
            ("logsigmoid", lambda t: F.log_sigmoid(t)),
            ("swish", lambda t: F.swish(t)),
        )
        for name, op in ops:
            with self.subTest(op=name):
                out = op(paddle.to_tensor(x, place="mps"))
                self.assertEqual(
                    out.dtype,
                    paddle.float16,
                    f"{name}: output dtype must be float16",
                )
                self.assertTrue(
                    "mps" in str(out.place).lower(),
                    f"{name}: output must be on mps",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
