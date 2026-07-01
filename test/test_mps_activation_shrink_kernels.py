#!/usr/bin/env python3
"""
Test MPS activation kernels of the "shrink / scaled" family by comparing
against hand-written NumPy references and the Paddle CPU backend.

Covers:
- mish               -> TestMPSMishKernel
- hardshrink         -> TestMPSShrinkKernels
- softshrink         -> TestMPSShrinkKernels
- tanhshrink         -> TestMPSShrinkKernels
- thresholded_relu   -> TestMPSThresholdedReluKernel
- stanh              -> TestMPSStanhKernel
- celu               -> TestMPSCeluSeluKernels
- selu               -> TestMPSCeluSeluKernels
- logit              -> TestMPSLogitKernel
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


def _mps_available():
    return (
        PADDLE_AVAILABLE
        and paddle.is_compiled_with_mps()
        and paddle.mps.is_available()
    )


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

    def _check_mps_vs_cpu_and_ref(self, fn, x_np, ref, rtol=1e-4, atol=1e-5):
        """Run fn on MPS and CPU tensors built from x_np; compare both to ref."""
        out_mps = fn(paddle.to_tensor(x_np, place="mps")).numpy()
        out_cpu = fn(paddle.to_tensor(x_np, place="cpu")).numpy()
        np.testing.assert_allclose(out_mps, ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol)

    def _check_dtype_and_place(self, fn, shape=(3, 4)):
        x = np.random.randn(*shape).astype(np.float32)
        out = fn(paddle.to_tensor(x, place="mps"))
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


# ---------------------------------------------------------------------------
# NumPy references
# ---------------------------------------------------------------------------


def _mish_numpy(x, threshold=20.0):
    x64 = x.astype(np.float64)
    sp = np.where(x64 > threshold, x64, np.log1p(np.exp(np.minimum(x64, threshold))))
    return (x64 * np.tanh(sp)).astype(np.float32)


def _hardshrink_numpy(x, threshold):
    keep = (x <= -threshold) | (x >= threshold)
    return np.where(keep, x, 0.0).astype(np.float32)


def _softshrink_numpy(x, threshold):
    return np.where(
        x > threshold,
        x - threshold,
        np.where(x < -threshold, x + threshold, 0.0),
    ).astype(np.float32)


def _tanhshrink_numpy(x):
    x64 = x.astype(np.float64)
    return (x64 - np.tanh(x64)).astype(np.float32)


def _thresholded_relu_numpy(x, threshold, value):
    return np.where(x > threshold, x, value).astype(np.float32)


def _stanh_numpy(x, scale_a, scale_b):
    x64 = x.astype(np.float64)
    return (scale_b * np.tanh(scale_a * x64)).astype(np.float32)


def _celu_numpy(x, alpha):
    x64 = x.astype(np.float64)
    return np.where(x64 < 0.0, alpha * np.expm1(x64 / alpha), x64).astype(np.float32)


def _selu_numpy(x, scale, alpha):
    x64 = x.astype(np.float64)
    return (scale * np.where(x64 <= 0.0, alpha * np.expm1(x64), x64)).astype(
        np.float32
    )


def _logit_numpy(x, eps):
    x64 = x.astype(np.float64)
    if eps:
        x64 = np.clip(x64, eps, 1.0 - eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(x64 / (1.0 - x64))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# mish
# ---------------------------------------------------------------------------


class TestMPSMishKernel(_MPSKernelTestBase):
    """MPS coverage for paddle.nn.functional.mish."""

    def test_mish_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 2.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_mps_vs_cpu_and_ref(F.mish, x, _mish_numpy(x))

    def test_mish_known_values(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        # mish(0) = 0, mish(1) = 1*tanh(ln(1+e)) ~ 0.865098
        expected = _mish_numpy(x)
        out = F.mish(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(out[2], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[3], 0.8650984, rtol=1e-4, atol=1e-5)

    def test_mish_large_magnitudes(self):
        # Above the softplus threshold (20) the identity branch is used.
        x = np.array([-100.0, -25.0, 19.0, 20.0, 25.0, 100.0], dtype=np.float32)
        self._check_mps_vs_cpu_and_ref(F.mish, x, _mish_numpy(x))
        out = F.mish(paddle.to_tensor(x, place="mps")).numpy()
        self.assertFalse(np.any(np.isnan(out)))
        self.assertFalse(np.any(np.isinf(out)))
        # For large positive x, mish(x) -> x.
        np.testing.assert_allclose(out[-1], 100.0, rtol=1e-5, atol=1e-4)

    def test_mish_zeros(self):
        x = np.zeros((4, 5), dtype=np.float32)
        out = F.mish(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.zeros_like(x), atol=1e-6)

    def test_mish_dtype_and_place(self):
        self._check_dtype_and_place(F.mish)


# ---------------------------------------------------------------------------
# hardshrink / softshrink / tanhshrink
# ---------------------------------------------------------------------------


class TestMPSShrinkKernels(_MPSKernelTestBase):
    """MPS coverage for hardshrink, softshrink, tanhshrink."""

    # ---- hardshrink -------------------------------------------------------
    def test_hardshrink_shapes(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_mps_vs_cpu_and_ref(
                    lambda t: F.hardshrink(t, threshold=0.5),
                    x,
                    _hardshrink_numpy(x, 0.5),
                )

    def test_hardshrink_thresholds(self):
        x = np.random.randn(5, 6).astype(np.float32)
        for threshold in [0.0, 0.25, 0.5, 1.0, 2.0]:
            with self.subTest(threshold=threshold):
                self._check_mps_vs_cpu_and_ref(
                    lambda t, th=threshold: F.hardshrink(t, threshold=th),
                    x,
                    _hardshrink_numpy(x, threshold),
                )

    def test_hardshrink_known_values_and_boundary(self):
        # Boundary: |x| == threshold is kept (>= / <=).
        x = np.array([-1.0, -0.5, -0.3, 0.0, 0.3, 0.5, 1.0], dtype=np.float32)
        expected = np.array(
            [-1.0, -0.5, 0.0, 0.0, 0.0, 0.5, 1.0], dtype=np.float32
        )
        out = F.hardshrink(
            paddle.to_tensor(x, place="mps"), threshold=0.5
        ).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)

    def test_hardshrink_large_magnitudes(self):
        x = np.array([-1e6, -100.0, 0.0, 100.0, 1e6], dtype=np.float32)
        self._check_mps_vs_cpu_and_ref(
            lambda t: F.hardshrink(t, threshold=0.5), x, _hardshrink_numpy(x, 0.5)
        )

    def test_hardshrink_dtype_and_place(self):
        self._check_dtype_and_place(F.hardshrink)

    # ---- softshrink -------------------------------------------------------
    def test_softshrink_shapes(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_mps_vs_cpu_and_ref(
                    lambda t: F.softshrink(t, threshold=0.5),
                    x,
                    _softshrink_numpy(x, 0.5),
                )

    def test_softshrink_thresholds(self):
        x = np.random.randn(5, 6).astype(np.float32)
        for threshold in [0.0, 0.25, 0.5, 1.0]:
            with self.subTest(threshold=threshold):
                self._check_mps_vs_cpu_and_ref(
                    lambda t, th=threshold: F.softshrink(t, threshold=th),
                    x,
                    _softshrink_numpy(x, threshold),
                )

    def test_softshrink_known_values_and_boundary(self):
        # Boundary: x == +-threshold maps to 0 (strict > / <).
        x = np.array([-1.5, -0.5, -0.2, 0.0, 0.2, 0.5, 1.5], dtype=np.float32)
        expected = np.array(
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        out = F.softshrink(
            paddle.to_tensor(x, place="mps"), threshold=0.5
        ).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)

    def test_softshrink_large_magnitudes(self):
        x = np.array([-1e6, -50.0, 0.0, 50.0, 1e6], dtype=np.float32)
        self._check_mps_vs_cpu_and_ref(
            lambda t: F.softshrink(t, threshold=1.0), x, _softshrink_numpy(x, 1.0)
        )

    def test_softshrink_dtype_and_place(self):
        self._check_dtype_and_place(F.softshrink)

    # ---- tanhshrink -------------------------------------------------------
    def test_tanhshrink_shapes(self):
        for shape in SHAPES:
            x = (np.random.randn(*shape) * 2.0).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_mps_vs_cpu_and_ref(
                    F.tanhshrink, x, _tanhshrink_numpy(x)
                )

    def test_tanhshrink_known_values(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        expected = _tanhshrink_numpy(x)
        out = F.tanhshrink(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
        # tanhshrink(0) = 0; tanhshrink(1) = 1 - tanh(1) ~ 0.238406
        np.testing.assert_allclose(out[2], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[3], 0.23840582, rtol=1e-4, atol=1e-5)

    def test_tanhshrink_zeros_and_large(self):
        x = np.zeros((3, 4), dtype=np.float32)
        out = F.tanhshrink(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.zeros_like(x), atol=1e-6)

        # For |x| large, tanhshrink(x) -> x -+ 1.
        x = np.array([-50.0, 50.0], dtype=np.float32)
        out = F.tanhshrink(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, [-49.0, 49.0], rtol=1e-5, atol=1e-4)

    def test_tanhshrink_dtype_and_place(self):
        self._check_dtype_and_place(F.tanhshrink)


# ---------------------------------------------------------------------------
# thresholded_relu
# ---------------------------------------------------------------------------


class TestMPSThresholdedReluKernel(_MPSKernelTestBase):
    """MPS coverage for paddle.nn.functional.thresholded_relu."""

    def test_thresholded_relu_shapes(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_mps_vs_cpu_and_ref(
                    F.thresholded_relu, x, _thresholded_relu_numpy(x, 1.0, 0.0)
                )

    def test_thresholded_relu_attrs(self):
        x = (np.random.randn(5, 6) * 2.0).astype(np.float32)
        for threshold, value in [
            (1.0, 0.0),
            (0.0, 0.0),
            (0.5, -1.0),
            (-1.0, 2.5),
            (2.0, 0.5),
        ]:
            with self.subTest(threshold=threshold, value=value):
                self._check_mps_vs_cpu_and_ref(
                    lambda t, th=threshold, v=value: F.thresholded_relu(
                        t, threshold=th, value=v
                    ),
                    x,
                    _thresholded_relu_numpy(x, threshold, value),
                )

    def test_thresholded_relu_known_values_and_boundary(self):
        # Boundary: x == threshold maps to value (strict >).
        x = np.array([-2.0, 0.0, 0.5, 1.0, 1.5, 3.0], dtype=np.float32)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.5, 3.0], dtype=np.float32)
        out = F.thresholded_relu(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)

    def test_thresholded_relu_large_magnitudes(self):
        x = np.array([-1e6, -10.0, 0.0, 10.0, 1e6], dtype=np.float32)
        self._check_mps_vs_cpu_and_ref(
            F.thresholded_relu, x, _thresholded_relu_numpy(x, 1.0, 0.0)
        )

    def test_thresholded_relu_dtype_and_place(self):
        self._check_dtype_and_place(F.thresholded_relu)


# ---------------------------------------------------------------------------
# stanh
# ---------------------------------------------------------------------------


class TestMPSStanhKernel(_MPSKernelTestBase):
    """MPS coverage for paddle.stanh."""

    def test_stanh_shapes_default_attrs(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_mps_vs_cpu_and_ref(
                    paddle.stanh, x, _stanh_numpy(x, 0.67, 1.7159)
                )

    def test_stanh_attrs(self):
        x = np.random.randn(5, 6).astype(np.float32)
        for scale_a, scale_b in [
            (0.67, 1.7159),
            (1.0, 1.0),
            (2.0 / 3.0, 1.7159),
            (0.5, 2.0),
            (2.0, 0.5),
        ]:
            with self.subTest(scale_a=scale_a, scale_b=scale_b):
                self._check_mps_vs_cpu_and_ref(
                    lambda t, a=scale_a, b=scale_b: paddle.stanh(
                        t, scale_a=a, scale_b=b
                    ),
                    x,
                    _stanh_numpy(x, scale_a, scale_b),
                )

    def test_stanh_known_values(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        out = paddle.stanh(
            paddle.to_tensor(x, place="mps"), scale_a=1.0, scale_b=1.0
        ).numpy()
        np.testing.assert_allclose(
            out, np.tanh(x), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(out[1], 0.0, atol=1e-6)

    def test_stanh_saturation(self):
        # tanh saturates: stanh(+-large) -> +-scale_b.
        x = np.array([-100.0, 100.0], dtype=np.float32)
        out = paddle.stanh(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, [-1.7159, 1.7159], rtol=1e-4, atol=1e-5)

    def test_stanh_dtype_and_place(self):
        self._check_dtype_and_place(paddle.stanh)


# ---------------------------------------------------------------------------
# celu / selu
# ---------------------------------------------------------------------------


class TestMPSCeluSeluKernels(_MPSKernelTestBase):
    """MPS coverage for celu and selu."""

    SELU_SCALE = 1.0507009873554804934193349852946
    SELU_ALPHA = 1.6732632423543772848170429916717

    # ---- celu -------------------------------------------------------------
    def test_celu_shapes_default_alpha(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_mps_vs_cpu_and_ref(F.celu, x, _celu_numpy(x, 1.0))

    def test_celu_alphas(self):
        x = (np.random.randn(5, 6) * 2.0).astype(np.float32)
        for alpha in [0.25, 0.5, 1.0, 2.0, -1.0]:
            with self.subTest(alpha=alpha):
                self._check_mps_vs_cpu_and_ref(
                    lambda t, a=alpha: F.celu(t, alpha=a),
                    x,
                    _celu_numpy(x, alpha),
                )

    def test_celu_known_values(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        # celu(-1, alpha=1) = exp(-1) - 1 ~ -0.632121
        out = F.celu(paddle.to_tensor(x, place="mps"), alpha=1.0).numpy()
        np.testing.assert_allclose(out, _celu_numpy(x, 1.0), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(out[1], -0.63212055, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(out[2:], [0.0, 1.0, 2.0], rtol=1e-6, atol=1e-6)

    def test_celu_negative_saturation(self):
        # For very negative x, celu(x) -> -alpha.
        x = np.array([-100.0, -50.0], dtype=np.float32)
        for alpha in [0.5, 1.0, 2.0]:
            with self.subTest(alpha=alpha):
                out = F.celu(paddle.to_tensor(x, place="mps"), alpha=alpha).numpy()
                np.testing.assert_allclose(
                    out, [-alpha, -alpha], rtol=1e-4, atol=1e-5
                )

    def test_celu_zeros(self):
        x = np.zeros((3, 4), dtype=np.float32)
        out = F.celu(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.zeros_like(x), atol=1e-6)

    def test_celu_dtype_and_place(self):
        self._check_dtype_and_place(F.celu)

    # ---- selu -------------------------------------------------------------
    def test_selu_shapes_default_attrs(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_mps_vs_cpu_and_ref(
                    F.selu, x, _selu_numpy(x, self.SELU_SCALE, self.SELU_ALPHA)
                )

    def test_selu_attrs(self):
        x = (np.random.randn(5, 6) * 2.0).astype(np.float32)
        for scale, alpha in [
            (self.SELU_SCALE, self.SELU_ALPHA),
            (2.0, 0.5),
            (1.5, 0.0),
            (1.0001, 1.0),
        ]:
            with self.subTest(scale=scale, alpha=alpha):
                self._check_mps_vs_cpu_and_ref(
                    lambda t, s=scale, a=alpha: F.selu(t, scale=s, alpha=a),
                    x,
                    _selu_numpy(x, scale, alpha),
                )

    def test_selu_known_values(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        out = F.selu(paddle.to_tensor(x, place="mps")).numpy()
        ref = _selu_numpy(x, self.SELU_SCALE, self.SELU_ALPHA)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
        # selu(1) = scale, selu(0) = 0.
        np.testing.assert_allclose(out[3], self.SELU_SCALE, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[2], 0.0, atol=1e-6)

    def test_selu_negative_saturation(self):
        # For very negative x, selu(x) -> -scale * alpha.
        x = np.array([-100.0, -50.0], dtype=np.float32)
        out = F.selu(paddle.to_tensor(x, place="mps")).numpy()
        limit = -self.SELU_SCALE * self.SELU_ALPHA
        np.testing.assert_allclose(out, [limit, limit], rtol=1e-4, atol=1e-5)

    def test_selu_dtype_and_place(self):
        self._check_dtype_and_place(F.selu)


# ---------------------------------------------------------------------------
# logit
# ---------------------------------------------------------------------------


class TestMPSLogitKernel(_MPSKernelTestBase):
    """MPS coverage for paddle.logit."""

    def test_logit_shapes_no_eps(self):
        for shape in SHAPES:
            x = np.random.uniform(0.01, 0.99, size=shape).astype(np.float32)
            with self.subTest(shape=shape):
                self._check_mps_vs_cpu_and_ref(
                    paddle.logit, x, _logit_numpy(x, 0.0)
                )

    def test_logit_eps_values(self):
        # Include values outside [eps, 1-eps] so the clamp path is exercised.
        x = np.random.uniform(0.0, 1.0, size=(5, 6)).astype(np.float32)
        for eps in [1e-3, 1e-2, 0.1]:
            with self.subTest(eps=eps):
                self._check_mps_vs_cpu_and_ref(
                    lambda t, e=eps: paddle.logit(t, eps=e),
                    x,
                    _logit_numpy(x, eps),
                )

    def test_logit_known_values(self):
        x = np.array([0.25, 0.5, 0.75], dtype=np.float32)
        out = paddle.logit(paddle.to_tensor(x, place="mps")).numpy()
        # logit(0.5) = 0; logit(0.75) = -logit(0.25) = ln(3) ~ 1.0986123
        np.testing.assert_allclose(
            out,
            np.array([-1.0986123, 0.0, 1.0986123], dtype=np.float32),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_logit_clamp_with_eps(self):
        # 0 and 1 are clamped to [eps, 1-eps] when eps > 0 -> finite outputs.
        x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        eps = 1e-3
        out = paddle.logit(paddle.to_tensor(x, place="mps"), eps=eps).numpy()
        ref = _logit_numpy(x, eps)
        self.assertFalse(np.any(np.isnan(out)))
        self.assertFalse(np.any(np.isinf(out)))
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

    def test_logit_edge_inf_no_eps(self):
        # Without eps, the domain edges map to -inf / +inf.
        x = np.array([0.0, 1.0], dtype=np.float32)
        out_mps = paddle.logit(paddle.to_tensor(x, place="mps")).numpy()
        out_cpu = paddle.logit(paddle.to_tensor(x, place="cpu")).numpy()
        self.assertTrue(np.isneginf(out_mps[0]))
        self.assertTrue(np.isposinf(out_mps[1]))
        np.testing.assert_array_equal(out_mps, out_cpu)

    def test_logit_out_of_domain_nan_no_eps(self):
        # Without eps, x < 0 or x > 1 yields NaN.
        x = np.array([-0.5, 1.5], dtype=np.float32)
        out_mps = paddle.logit(paddle.to_tensor(x, place="mps")).numpy()
        out_cpu = paddle.logit(paddle.to_tensor(x, place="cpu")).numpy()
        self.assertTrue(np.all(np.isnan(out_mps)))
        self.assertTrue(np.all(np.isnan(out_cpu)))

    def test_logit_dtype_and_place(self):
        x = np.random.uniform(0.05, 0.95, size=(3, 4)).astype(np.float32)
        out = paddle.logit(paddle.to_tensor(x, place="mps"))
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


if __name__ == "__main__":
    unittest.main()
