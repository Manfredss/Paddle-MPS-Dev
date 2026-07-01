#!/usr/bin/env python3
"""
Test MPS activation kernels against NumPy references and the CPU backend.

Covers the 'activation-a' op family:
- elu          -> TestMPSEluKernel
- relu6        -> TestMPSRelu6Kernel
- hardsigmoid  -> TestMPSHardSigmoidKernel
- hardswish    -> TestMPSHardSwishKernel
- softplus     -> TestMPSSoftplusKernel
- softsign     -> TestMPSSoftsignKernel
- logsigmoid   -> TestMPSLogSigmoidKernel
- swish        -> TestMPSSwishKernel
- hardtanh     -> TestMPSHardTanhKernel
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


SHAPES = [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]

# Mix of zeros, small/large negatives and positives.
EDGE_VALUES = np.array(
    [-100.0, -50.0, -6.5, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 6.5, 50.0, 100.0],
    dtype=np.float32,
)


def _mps_available():
    return (
        PADDLE_AVAILABLE
        and paddle.is_compiled_with_mps()
        and paddle.mps.is_available()
    )


# ---------------------------------------------------------------------------
# NumPy references (computed in float64 for accuracy, cast back to float32).
# ---------------------------------------------------------------------------


def _elu_np(x, alpha=1.0):
    x64 = x.astype(np.float64)
    return np.where(x64 < 0.0, alpha * np.expm1(x64), x64).astype(np.float32)


def _relu6_np(x):
    x64 = x.astype(np.float64)
    return np.minimum(np.maximum(x64, 0.0), 6.0).astype(np.float32)


def _hardsigmoid_np(x, slope=0.1666667, offset=0.5):
    x64 = x.astype(np.float64)
    return np.clip(slope * x64 + offset, 0.0, 1.0).astype(np.float32)


def _hardswish_np(x):
    x64 = x.astype(np.float64)
    return (x64 * np.clip(x64 + 3.0, 0.0, 6.0) / 6.0).astype(np.float32)


def _softplus_np(x, beta=1.0, threshold=20.0):
    x64 = x.astype(np.float64)
    x_beta = beta * x64
    # Avoid overflow warnings: zero out the lanes taken by the linear branch.
    safe = np.where(x_beta > threshold, 0.0, x_beta)
    return np.where(
        x_beta > threshold, x64, np.log1p(np.exp(safe)) / beta
    ).astype(np.float32)


def _softsign_np(x):
    x64 = x.astype(np.float64)
    return (x64 / (1.0 + np.abs(x64))).astype(np.float32)


def _log_sigmoid_np(x):
    x64 = x.astype(np.float64)
    return (np.minimum(x64, 0.0) - np.log1p(np.exp(-np.abs(x64)))).astype(
        np.float32
    )


def _swish_np(x):
    # float64 sigmoid is exact enough for |x| <= ~700 (no overflow).
    x64 = x.astype(np.float64)
    sig = 1.0 / (1.0 + np.exp(-x64))
    return (x64 * sig).astype(np.float32)


def _hardtanh_np(x, t_min=-1.0, t_max=1.0):
    x64 = x.astype(np.float64)
    return np.minimum(np.maximum(x64, t_min), t_max).astype(np.float32)


# ---------------------------------------------------------------------------
# Base class.
# ---------------------------------------------------------------------------


class _MPSKernelTestBase(unittest.TestCase):
    """Common setUp for MPS kernel tests — compares MPS vs NumPy and MPS vs CPU."""

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

    def _check_unary(self, fn, ref, x_np, rtol=1e-4, atol=1e-5, **kwargs):
        """Run fn on MPS and CPU; compare both against the numpy reference."""
        out_mps = fn(paddle.to_tensor(x_np, place="mps"), **kwargs).numpy()
        out_cpu = fn(paddle.to_tensor(x_np, place="cpu"), **kwargs).numpy()
        np.testing.assert_allclose(out_mps, ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol)

    def _check_dtype_and_place(self, fn, **kwargs):
        x = np.random.randn(3, 4).astype(np.float32)
        out = fn(paddle.to_tensor(x, place="mps"), **kwargs)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


# ---------------------------------------------------------------------------
# elu
# ---------------------------------------------------------------------------


class TestMPSEluKernel(_MPSKernelTestBase):
    def test_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 3.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_unary(F.elu, _elu_np(x), x)

    def test_alphas(self):
        x = (np.random.randn(4, 5) * 2.0).astype(np.float32)
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            with self.subTest(alpha=alpha):
                self._check_unary(F.elu, _elu_np(x, alpha), x, alpha=alpha)

    def test_known_values(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        out = F.elu(paddle.to_tensor(x, place="mps"), alpha=1.0).numpy()
        expected = np.array(
            [-0.8646647, -0.6321206, 0.0, 1.0, 2.0], dtype=np.float32
        )
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_edge_cases(self):
        # Large negatives saturate at -alpha; positives pass through.
        self._check_unary(F.elu, _elu_np(EDGE_VALUES), EDGE_VALUES)
        zeros = np.zeros((4, 5), dtype=np.float32)
        self._check_unary(F.elu, _elu_np(zeros), zeros)

    def test_dtype_and_place(self):
        self._check_dtype_and_place(F.elu)


# ---------------------------------------------------------------------------
# relu6
# ---------------------------------------------------------------------------


class TestMPSRelu6Kernel(_MPSKernelTestBase):
    def test_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 5.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_unary(F.relu6, _relu6_np(x), x, rtol=1e-5, atol=1e-6)

    def test_known_values(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 5.9, 6.0, 7.0, 100.0], dtype=np.float32)
        out = F.relu6(paddle.to_tensor(x, place="mps")).numpy()
        expected = np.array(
            [0.0, 0.0, 0.0, 1.0, 5.9, 6.0, 6.0, 6.0], dtype=np.float32
        )
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_edge_cases(self):
        self._check_unary(
            F.relu6, _relu6_np(EDGE_VALUES), EDGE_VALUES, rtol=1e-5, atol=1e-6
        )
        zeros = np.zeros((4, 5), dtype=np.float32)
        self._check_unary(F.relu6, _relu6_np(zeros), zeros, rtol=1e-5, atol=1e-6)

    def test_dtype_and_place(self):
        self._check_dtype_and_place(F.relu6)


# ---------------------------------------------------------------------------
# hardsigmoid
# ---------------------------------------------------------------------------


class TestMPSHardSigmoidKernel(_MPSKernelTestBase):
    def test_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 4.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_unary(F.hardsigmoid, _hardsigmoid_np(x), x)

    def test_slope_offset_variations(self):
        x = (np.random.randn(4, 5) * 4.0).astype(np.float32)
        for slope, offset in [(0.1666667, 0.5), (0.2, 0.5), (0.5, 0.0), (0.25, 0.4)]:
            with self.subTest(slope=slope, offset=offset):
                self._check_unary(
                    F.hardsigmoid,
                    _hardsigmoid_np(x, slope, offset),
                    x,
                    slope=slope,
                    offset=offset,
                )

    def test_known_values(self):
        # Python API default: slope=1/6, offset=0.5.
        x = np.array([-6.0, -3.0, 0.0, 3.0, 6.0], dtype=np.float32)
        out = F.hardsigmoid(paddle.to_tensor(x, place="mps")).numpy()
        expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_edge_cases(self):
        self._check_unary(F.hardsigmoid, _hardsigmoid_np(EDGE_VALUES), EDGE_VALUES)
        zeros = np.zeros((4, 5), dtype=np.float32)
        self._check_unary(F.hardsigmoid, _hardsigmoid_np(zeros), zeros)

    def test_dtype_and_place(self):
        self._check_dtype_and_place(F.hardsigmoid)


# ---------------------------------------------------------------------------
# hardswish
# ---------------------------------------------------------------------------


class TestMPSHardSwishKernel(_MPSKernelTestBase):
    def test_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 4.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_unary(F.hardswish, _hardswish_np(x), x)

    def test_known_values(self):
        x = np.array([-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0], dtype=np.float32)
        out = F.hardswish(paddle.to_tensor(x, place="mps")).numpy()
        expected = np.array(
            [0.0, 0.0, -0.375, 0.0, 1.125, 3.0, 4.0], dtype=np.float32
        )
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_edge_cases(self):
        # Below -3 the output is exactly 0; above +3 it is the identity.
        self._check_unary(F.hardswish, _hardswish_np(EDGE_VALUES), EDGE_VALUES)
        zeros = np.zeros((4, 5), dtype=np.float32)
        self._check_unary(F.hardswish, _hardswish_np(zeros), zeros)

    def test_dtype_and_place(self):
        self._check_dtype_and_place(F.hardswish)


# ---------------------------------------------------------------------------
# softplus
# ---------------------------------------------------------------------------


class TestMPSSoftplusKernel(_MPSKernelTestBase):
    def test_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 3.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_unary(F.softplus, _softplus_np(x), x)

    def test_beta_threshold_variations(self):
        x = (np.random.randn(4, 5) * 3.0).astype(np.float32)
        for beta, threshold in [(1.0, 20.0), (2.0, 20.0), (0.5, 15.0), (4.0, 10.0)]:
            with self.subTest(beta=beta, threshold=threshold):
                self._check_unary(
                    F.softplus,
                    _softplus_np(x, beta, threshold),
                    x,
                    beta=beta,
                    threshold=threshold,
                )

    def test_known_values(self):
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        out = F.softplus(paddle.to_tensor(x, place="mps")).numpy()
        expected = np.array(
            [0.6931472, 1.3132616, 0.3132617], dtype=np.float32
        )
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_linear_region_above_threshold(self):
        # beta*x > threshold -> exact pass-through of x (no overflow).
        x = np.array([21.0, 50.0, 100.0], dtype=np.float32)
        out = F.softplus(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, x, rtol=1e-6, atol=0.0)
        self.assertFalse(np.any(np.isnan(out)))
        self.assertFalse(np.any(np.isinf(out)))

    def test_edge_cases(self):
        self._check_unary(F.softplus, _softplus_np(EDGE_VALUES), EDGE_VALUES)
        zeros = np.zeros((4, 5), dtype=np.float32)
        self._check_unary(F.softplus, _softplus_np(zeros), zeros)

    def test_dtype_and_place(self):
        self._check_dtype_and_place(F.softplus)


# ---------------------------------------------------------------------------
# softsign
# ---------------------------------------------------------------------------


class TestMPSSoftsignKernel(_MPSKernelTestBase):
    def test_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 3.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_unary(F.softsign, _softsign_np(x), x)

    def test_known_values(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
        out = F.softsign(paddle.to_tensor(x, place="mps")).numpy()
        expected = np.array([-0.75, -0.5, 0.0, 0.5, 0.75], dtype=np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_edge_cases(self):
        # Bounded in (-1, 1) even for huge magnitudes.
        self._check_unary(F.softsign, _softsign_np(EDGE_VALUES), EDGE_VALUES)
        out = F.softsign(paddle.to_tensor(EDGE_VALUES, place="mps")).numpy()
        self.assertTrue(np.all(np.abs(out) < 1.0))
        zeros = np.zeros((4, 5), dtype=np.float32)
        self._check_unary(F.softsign, _softsign_np(zeros), zeros)

    def test_dtype_and_place(self):
        self._check_dtype_and_place(F.softsign)


# ---------------------------------------------------------------------------
# logsigmoid
# ---------------------------------------------------------------------------


class TestMPSLogSigmoidKernel(_MPSKernelTestBase):
    def test_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 3.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_unary(F.log_sigmoid, _log_sigmoid_np(x), x)

    def test_known_values(self):
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        out = F.log_sigmoid(paddle.to_tensor(x, place="mps")).numpy()
        expected = np.array(
            [-0.6931472, -0.3132617, -1.3132616], dtype=np.float32
        )
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_numerical_stability(self):
        # Naive log(sigmoid(x)) underflows to -inf for large negative x;
        # the stable form must return ~x instead.
        x = np.array([-100.0, -50.0, 50.0, 100.0], dtype=np.float32)
        out = F.log_sigmoid(paddle.to_tensor(x, place="mps")).numpy()
        self.assertFalse(np.any(np.isnan(out)))
        self.assertFalse(np.any(np.isinf(out)))
        np.testing.assert_allclose(out[:2], x[:2], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[2:], [0.0, 0.0], rtol=0.0, atol=1e-6)

    def test_edge_cases(self):
        self._check_unary(F.log_sigmoid, _log_sigmoid_np(EDGE_VALUES), EDGE_VALUES)
        zeros = np.zeros((4, 5), dtype=np.float32)
        self._check_unary(F.log_sigmoid, _log_sigmoid_np(zeros), zeros)

    def test_dtype_and_place(self):
        self._check_dtype_and_place(F.log_sigmoid)


# ---------------------------------------------------------------------------
# swish
# ---------------------------------------------------------------------------


class TestMPSSwishKernel(_MPSKernelTestBase):
    def test_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 3.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_unary(F.swish, _swish_np(x), x)

    def test_known_values(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        out = F.swish(paddle.to_tensor(x, place="mps")).numpy()
        expected = np.array(
            [-0.26894143, 0.0, 0.7310586], dtype=np.float32
        )
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_edge_cases(self):
        # Large positives approach identity, large negatives approach 0.
        self._check_unary(F.swish, _swish_np(EDGE_VALUES), EDGE_VALUES)
        out = F.swish(paddle.to_tensor(EDGE_VALUES, place="mps")).numpy()
        self.assertFalse(np.any(np.isnan(out)))
        self.assertFalse(np.any(np.isinf(out)))
        zeros = np.zeros((4, 5), dtype=np.float32)
        self._check_unary(F.swish, _swish_np(zeros), zeros)

    def test_dtype_and_place(self):
        self._check_dtype_and_place(F.swish)


# ---------------------------------------------------------------------------
# hardtanh
# ---------------------------------------------------------------------------


class TestMPSHardTanhKernel(_MPSKernelTestBase):
    def test_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 3.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_unary(
                    F.hardtanh, _hardtanh_np(x), x, rtol=1e-5, atol=1e-6
                )

    def test_min_max_variations(self):
        x = (np.random.randn(4, 5) * 5.0).astype(np.float32)
        for t_min, t_max in [(-1.0, 1.0), (-2.5, 2.5), (0.0, 4.0), (-3.0, 0.5)]:
            with self.subTest(t_min=t_min, t_max=t_max):
                self._check_unary(
                    F.hardtanh,
                    _hardtanh_np(x, t_min, t_max),
                    x,
                    rtol=1e-5,
                    atol=1e-6,
                    min=t_min,
                    max=t_max,
                )

    def test_known_values(self):
        x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
        out = F.hardtanh(paddle.to_tensor(x, place="mps")).numpy()
        expected = np.array(
            [-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0], dtype=np.float32
        )
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_edge_cases(self):
        self._check_unary(
            F.hardtanh, _hardtanh_np(EDGE_VALUES), EDGE_VALUES, rtol=1e-5, atol=1e-6
        )
        zeros = np.zeros((4, 5), dtype=np.float32)
        self._check_unary(
            F.hardtanh, _hardtanh_np(zeros), zeros, rtol=1e-5, atol=1e-6
        )

    def test_dtype_and_place(self):
        self._check_dtype_and_place(F.hardtanh)


if __name__ == "__main__":
    unittest.main()
