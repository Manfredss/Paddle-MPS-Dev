#!/usr/bin/env python3
"""Tests for the MPS math / activation / binary kernels added on top of the
original elementwise set.

Each op is checked on the MPS backend against the CPU backend (the reference
implementation these kernels must match) using the same public paddle API, plus
a few hand-rolled numpy value checks. All ops are float32-only, forward-only.

Covered:
- Unary math   : asinh, acosh, atanh, expm1, log1p, trunc
- Activations  : tanhshrink, softsign, log_sigmoid, hardswish, relu6, swish,
                 elu, celu, mish, softshrink, hardshrink, hardtanh, hardsigmoid,
                 thresholded_relu, softplus, log_softmax
- Binary       : atan2, floor_divide, remainder, fmax, fmin, heaviside
"""

import unittest

import numpy as np

try:
    import paddle
    import paddle.nn.functional as F
    PADDLE_AVAILABLE = True
except ImportError:  # pragma: no cover - paddle always present in CI
    PADDLE_AVAILABLE = False


def _mps_available():
    return (
        PADDLE_AVAILABLE
        and paddle.is_compiled_with_mps()
        and paddle.mps.is_available()
    )


class _MPSKernelTestBase(unittest.TestCase):
    """Compares an op run on MPS against the same op on CPU."""

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

    def _check_unary(self, paddle_fn, x_np, rtol=1e-5, atol=1e-6, msg=""):
        out_mps = paddle_fn(paddle.to_tensor(x_np, place="mps")).numpy()
        out_cpu = paddle_fn(paddle.to_tensor(x_np, place="cpu")).numpy()
        np.testing.assert_allclose(
            out_mps, out_cpu, rtol=rtol, atol=atol, err_msg=f"{msg} (mps vs cpu)"
        )
        self.assertEqual(out_mps.shape, out_cpu.shape, msg)

    def _check_binary(self, paddle_fn, x_np, y_np, rtol=1e-5, atol=1e-6, msg=""):
        out_mps = paddle_fn(
            paddle.to_tensor(x_np, place="mps"),
            paddle.to_tensor(y_np, place="mps"),
        ).numpy()
        out_cpu = paddle_fn(
            paddle.to_tensor(x_np, place="cpu"),
            paddle.to_tensor(y_np, place="cpu"),
        ).numpy()
        np.testing.assert_allclose(
            out_mps, out_cpu, rtol=rtol, atol=atol, err_msg=f"{msg} (mps vs cpu)"
        )
        self.assertEqual(out_mps.shape, out_cpu.shape, msg)


_SHAPES = [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]


class TestMPSUnaryMath(_MPSKernelTestBase):
    """asinh, acosh, atanh, expm1, log1p, trunc."""

    def test_asinh(self):
        for shape in _SHAPES:
            self._check_unary(paddle.asinh,
                              np.random.randn(*shape).astype(np.float32) * 3.0,
                              msg=f"asinh {shape}")

    def test_acosh(self):
        # Domain: x >= 1.
        for shape in _SHAPES:
            x = np.abs(np.random.randn(*shape).astype(np.float32)) + 1.0
            self._check_unary(paddle.acosh, x, rtol=1e-4, atol=1e-5,
                              msg=f"acosh {shape}")

    def test_atanh(self):
        # Domain: |x| < 1.
        for shape in _SHAPES:
            x = np.random.uniform(-0.9, 0.9, shape).astype(np.float32)
            self._check_unary(paddle.atanh, x, rtol=1e-4, atol=1e-5,
                              msg=f"atanh {shape}")

    def test_expm1(self):
        for shape in _SHAPES:
            self._check_unary(paddle.expm1,
                              np.random.randn(*shape).astype(np.float32),
                              rtol=1e-4, atol=1e-5, msg=f"expm1 {shape}")

    def test_log1p(self):
        # Domain: x > -1.
        for shape in _SHAPES:
            x = np.random.uniform(-0.9, 3.0, shape).astype(np.float32)
            self._check_unary(paddle.log1p, x, rtol=1e-4, atol=1e-5,
                              msg=f"log1p {shape}")

    def test_trunc(self):
        for shape in _SHAPES:
            self._check_unary(paddle.trunc,
                              np.random.randn(*shape).astype(np.float32) * 5.0,
                              msg=f"trunc {shape}")

    def test_trunc_known_values(self):
        x = np.array([-2.7, -0.5, 0.0, 0.5, 2.7, 3.999], dtype=np.float32)
        out = paddle.trunc(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.trunc(x))

    def test_expm1_known_values(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        out = paddle.expm1(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.expm1(x), rtol=1e-5, atol=1e-6)


class TestMPSActivations(_MPSKernelTestBase):
    """Coverage for the newly added activation kernels."""

    def test_tanhshrink(self):
        for shape in _SHAPES:
            self._check_unary(F.tanhshrink,
                              np.random.randn(*shape).astype(np.float32) * 2.0,
                              rtol=1e-4, atol=1e-5, msg=f"tanhshrink {shape}")

    def test_softsign(self):
        for shape in _SHAPES:
            self._check_unary(F.softsign,
                              np.random.randn(*shape).astype(np.float32) * 3.0,
                              msg=f"softsign {shape}")

    def test_log_sigmoid(self):
        for shape in _SHAPES:
            self._check_unary(F.log_sigmoid,
                              np.random.randn(*shape).astype(np.float32) * 5.0,
                              rtol=1e-4, atol=1e-5, msg=f"log_sigmoid {shape}")

    def test_hardswish(self):
        for shape in _SHAPES:
            self._check_unary(F.hardswish,
                              np.random.randn(*shape).astype(np.float32) * 5.0,
                              msg=f"hardswish {shape}")

    def test_relu6(self):
        for shape in _SHAPES:
            self._check_unary(F.relu6,
                              np.random.randn(*shape).astype(np.float32) * 5.0,
                              msg=f"relu6 {shape}")

    def test_swish(self):
        for shape in _SHAPES:
            self._check_unary(F.swish,
                              np.random.randn(*shape).astype(np.float32) * 4.0,
                              rtol=1e-4, atol=1e-5, msg=f"swish {shape}")

    def test_elu(self):
        for alpha in (1.0, 0.5, 2.0):
            for shape in _SHAPES:
                self._check_unary(lambda x, a=alpha: F.elu(x, a),
                                  np.random.randn(*shape).astype(np.float32) * 3.0,
                                  rtol=1e-4, atol=1e-5,
                                  msg=f"elu alpha={alpha} {shape}")

    def test_celu(self):
        for alpha in (1.0, 0.5, 2.0):
            for shape in _SHAPES:
                self._check_unary(lambda x, a=alpha: F.celu(x, a),
                                  np.random.randn(*shape).astype(np.float32) * 3.0,
                                  rtol=1e-4, atol=1e-5,
                                  msg=f"celu alpha={alpha} {shape}")

    def test_mish(self):
        for shape in _SHAPES:
            self._check_unary(F.mish,
                              np.random.randn(*shape).astype(np.float32) * 4.0,
                              rtol=1e-4, atol=1e-5, msg=f"mish {shape}")

    def test_softshrink(self):
        for threshold in (0.5, 1.0):
            for shape in _SHAPES:
                self._check_unary(lambda x, t=threshold: F.softshrink(x, t),
                                  np.random.randn(*shape).astype(np.float32) * 2.0,
                                  msg=f"softshrink t={threshold} {shape}")

    def test_hardshrink(self):
        for threshold in (0.5, 1.0):
            for shape in _SHAPES:
                self._check_unary(lambda x, t=threshold: F.hardshrink(x, t),
                                  np.random.randn(*shape).astype(np.float32) * 2.0,
                                  msg=f"hardshrink t={threshold} {shape}")

    def test_hardtanh(self):
        for bounds in ((-1.0, 1.0), (-2.0, 3.0)):
            for shape in _SHAPES:
                self._check_unary(
                    lambda x, b=bounds: F.hardtanh(x, b[0], b[1]),
                    np.random.randn(*shape).astype(np.float32) * 4.0,
                    msg=f"hardtanh {bounds} {shape}")

    def test_hardsigmoid(self):
        for shape in _SHAPES:
            self._check_unary(F.hardsigmoid,
                              np.random.randn(*shape).astype(np.float32) * 5.0,
                              msg=f"hardsigmoid {shape}")

    def test_thresholded_relu(self):
        for threshold in (1.0, 0.0, 2.0):
            for shape in _SHAPES:
                self._check_unary(
                    lambda x, t=threshold: F.thresholded_relu(x, t),
                    np.random.randn(*shape).astype(np.float32) * 3.0,
                    msg=f"thresholded_relu t={threshold} {shape}")

    def test_softplus(self):
        for shape in _SHAPES:
            self._check_unary(F.softplus,
                              np.random.randn(*shape).astype(np.float32) * 3.0,
                              rtol=1e-4, atol=1e-5, msg=f"softplus {shape}")

    def test_softplus_beta(self):
        x = np.random.randn(4, 5).astype(np.float32) * 3.0
        self._check_unary(lambda t: F.softplus(t, beta=2.0, threshold=15.0),
                          x, rtol=1e-4, atol=1e-5, msg="softplus beta=2")

    def test_log_softmax(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            x = np.random.randn(*shape).astype(np.float32) * 3.0
            for axis in range(len(shape)):
                self._check_unary(lambda t, a=axis: F.log_softmax(t, a),
                                  x, rtol=1e-4, atol=1e-5,
                                  msg=f"log_softmax axis={axis} {shape}")

    def test_relu6_saturation(self):
        x = np.array([-1.0, 0.0, 3.0, 6.0, 10.0], dtype=np.float32)
        out = F.relu6(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, [0.0, 0.0, 3.0, 6.0, 6.0])


class TestMPSBinary(_MPSKernelTestBase):
    """atan2, floor_divide, remainder, fmax, fmin, heaviside."""

    def test_atan2(self):
        for shape in _SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            y = np.random.randn(*shape).astype(np.float32)
            self._check_binary(paddle.atan2, x, y, rtol=1e-4, atol=1e-5,
                               msg=f"atan2 {shape}")

    def test_atan2_quadrants(self):
        x = np.array([1.0, 1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        y = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        out = paddle.atan2(
            paddle.to_tensor(x, place="mps"), paddle.to_tensor(y, place="mps")
        ).numpy()
        np.testing.assert_allclose(out, np.arctan2(x, y), rtol=1e-5, atol=1e-6)

    def _nonzero(self, shape):
        y = np.random.uniform(0.5, 3.0, shape).astype(np.float32)
        signs = np.random.choice([-1.0, 1.0], size=shape).astype(np.float32)
        return y * signs

    def test_floor_divide(self):
        for shape in _SHAPES:
            x = np.random.randn(*shape).astype(np.float32) * 5.0
            y = self._nonzero(shape)
            self._check_binary(paddle.floor_divide, x, y,
                               msg=f"floor_divide {shape}")

    def test_remainder(self):
        for shape in _SHAPES:
            x = np.random.randn(*shape).astype(np.float32) * 5.0
            y = self._nonzero(shape)
            self._check_binary(paddle.remainder, x, y, rtol=1e-4, atol=1e-5,
                               msg=f"remainder {shape}")

    def test_remainder_sign_follows_divisor(self):
        x = np.array([5.0, -5.0, 5.0, -5.0], dtype=np.float32)
        y = np.array([3.0, 3.0, -3.0, -3.0], dtype=np.float32)
        out = paddle.remainder(
            paddle.to_tensor(x, place="mps"), paddle.to_tensor(y, place="mps")
        ).numpy()
        np.testing.assert_allclose(out, np.mod(x, y), rtol=1e-5, atol=1e-5)

    def test_fmax(self):
        for shape in _SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            y = np.random.randn(*shape).astype(np.float32)
            self._check_binary(paddle.fmax, x, y, msg=f"fmax {shape}")

    def test_fmin(self):
        for shape in _SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            y = np.random.randn(*shape).astype(np.float32)
            self._check_binary(paddle.fmin, x, y, msg=f"fmin {shape}")

    def test_heaviside(self):
        for shape in _SHAPES:
            x = np.random.choice(
                [-1.5, 0.0, 2.0], size=shape).astype(np.float32)
            y = np.random.randn(*shape).astype(np.float32)
            self._check_binary(paddle.heaviside, x, y,
                               msg=f"heaviside {shape}")

    def test_heaviside_known_values(self):
        x = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
        y = np.array([7.0, 7.0, 7.0], dtype=np.float32)
        out = paddle.heaviside(
            paddle.to_tensor(x, place="mps"), paddle.to_tensor(y, place="mps")
        ).numpy()
        np.testing.assert_allclose(out, [0.0, 7.0, 1.0])

    def test_binary_broadcast(self):
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(4, 5).astype(np.float32)
        self._check_binary(paddle.fmax, x, y, msg="fmax broadcast")
        self._check_binary(paddle.atan2, x, y, rtol=1e-4, atol=1e-5,
                           msg="atan2 broadcast")


if __name__ == "__main__":
    unittest.main(verbosity=2)
