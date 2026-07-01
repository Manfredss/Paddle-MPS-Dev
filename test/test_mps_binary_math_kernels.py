#!/usr/bin/env python3
"""
Test MPS binary-math kernels against NumPy references and the CPU backend.

Covers:
- floor_divide (Python // semantics for floats)
- remainder    (Python %  semantics: sign follows divisor)
- fmax / fmin  (NaN-ignoring max / min)
- heaviside    (x < 0 -> 0, x == 0 -> y, x > 0 -> 1)
- atan2        (quadrant-aware arctangent, x = numerator, y = denominator)

All tests are float32 only and do not depend on torch.
"""

import unittest

import numpy as np

try:
    import paddle

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddlePaddle not available")


SHAPES = [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]


def _mps_available():
    if not PADDLE_AVAILABLE:
        return False
    # Builds without MPS support may not expose these attributes at all.
    if not hasattr(paddle, "is_compiled_with_mps"):
        return False
    if not paddle.is_compiled_with_mps():
        return False
    mps = getattr(paddle, "mps", None)
    return mps is not None and mps.is_available()


def _safe_divmod_inputs(shape):
    """Random (x, y) float32 pairs that are safe for floor_divide/remainder.

    |y| is bounded away from 0, and x/y is bounded away from exact integers
    so that float32 rounding of the quotient cannot cross an integer
    boundary (where floor division / modulo are discontinuous and the MPS
    composition could legitimately differ from the CPU's exact fmod).
    """
    y_mag = np.random.uniform(0.5, 2.5, size=shape)
    y_sign = np.where(np.random.rand(*shape) < 0.5, -1.0, 1.0)
    y = (y_mag * y_sign).astype(np.float32)
    x = (np.random.randn(*shape) * 5.0).astype(np.float32)
    q = x.astype(np.float64) / y.astype(np.float64)
    frac = q - np.floor(q)
    bad = (frac < 0.05) | (frac > 0.95)
    x = np.where(bad, x + 0.37 * y, x).astype(np.float32)
    return x, y


class _MPSBinaryMathTestBase(unittest.TestCase):
    """Common setUp: skip without MPS, compare MPS vs NumPy and MPS vs CPU."""

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

    def _run_both(self, paddle_op, x_np, y_np):
        out_mps = paddle_op(
            paddle.to_tensor(x_np, place="mps"),
            paddle.to_tensor(y_np, place="mps"),
        ).numpy()
        out_cpu = paddle_op(
            paddle.to_tensor(x_np, place="cpu"),
            paddle.to_tensor(y_np, place="cpu"),
        ).numpy()
        return out_mps, out_cpu

    def _check(self, name, paddle_op, numpy_op, x_np, y_np,
               rtol=1e-5, atol=1e-6):
        out_mps, out_cpu = self._run_both(paddle_op, x_np, y_np)
        ref = numpy_op(x_np, y_np).astype(np.float32)
        np.testing.assert_allclose(
            out_mps, ref, rtol=rtol, atol=atol,
            err_msg="{} vs numpy".format(name),
        )
        np.testing.assert_allclose(
            out_mps, out_cpu, rtol=rtol, atol=atol,
            err_msg="{} vs cpu".format(name),
        )
        self.assertEqual(out_mps.dtype, np.float32,
                         "{} output dtype must be float32".format(name))

    def _check_dtype_and_place(self, paddle_op, x_np, y_np):
        out = paddle_op(
            paddle.to_tensor(x_np, place="mps"),
            paddle.to_tensor(y_np, place="mps"),
        )
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


# ---------------------------------------------------------------------------
# floor_divide / remainder: the sign convention is what matters.
# ---------------------------------------------------------------------------


class TestMPSFloorDivideKernel(_MPSBinaryMathTestBase):
    """MPS coverage for paddle.floor_divide (float32)."""

    def test_shapes(self):
        for shape in SHAPES:
            x, y = _safe_divmod_inputs(shape)
            with self.subTest(shape=shape):
                self._check("floor_divide", paddle.floor_divide,
                            np.floor_divide, x, y)

    def test_sign_combinations(self):
        # Python // semantics: 7//2=3, -7//2=-4, 7//-2=-4, -7//-2=3.
        x = np.array([7.0, -7.0, 7.0, -7.0, 5.5, -5.5], dtype=np.float32)
        y = np.array([2.0, 2.0, -2.0, -2.0, 2.0, 2.0], dtype=np.float32)
        expected = np.array([3.0, -4.0, -4.0, 3.0, 2.0, -3.0],
                            dtype=np.float32)
        out_mps, out_cpu = self._run_both(paddle.floor_divide, x, y)
        np.testing.assert_allclose(out_mps, expected, rtol=0, atol=0)
        np.testing.assert_allclose(out_mps, out_cpu, rtol=0, atol=0)

    def test_fractional_dividend(self):
        x = np.array([0.5, -0.5, 2.3, -2.3], dtype=np.float32)
        y = np.array([2.0, 2.0, 1.5, 1.5], dtype=np.float32)
        expected = np.array([0.0, -1.0, 1.0, -2.0], dtype=np.float32)
        out_mps, _ = self._run_both(paddle.floor_divide, x, y)
        np.testing.assert_allclose(out_mps, expected, rtol=0, atol=0)

    def test_zero_dividend(self):
        x = np.zeros((3, 4), dtype=np.float32)
        y = np.full((3, 4), 1.7, dtype=np.float32)
        self._check("floor_divide", paddle.floor_divide, np.floor_divide,
                    x, y)

    def test_large_magnitudes(self):
        # Quotients far from integer boundaries even at large scale.
        x = np.array([1000.5, -1000.5, 12345.25], dtype=np.float32)
        y = np.array([2.0, 2.0, 8.0], dtype=np.float32)
        self._check("floor_divide", paddle.floor_divide, np.floor_divide,
                    x, y)

    def test_broadcast(self):
        x, _ = _safe_divmod_inputs((3, 4))
        # A fixed safe divisor row broadcast across x.
        y = np.array([1.3, -1.7, 2.1, -0.9], dtype=np.float32)
        q = x.astype(np.float64) / y.astype(np.float64)
        frac = q - np.floor(q)
        bad = (frac < 0.05) | (frac > 0.95)
        x = np.where(bad, x + 0.37 * y, x).astype(np.float32)
        self._check("floor_divide", paddle.floor_divide, np.floor_divide,
                    x, y)

    def test_dtype_and_place(self):
        x, y = _safe_divmod_inputs((3, 4))
        self._check_dtype_and_place(paddle.floor_divide, x, y)


class TestMPSRemainderKernel(_MPSBinaryMathTestBase):
    """MPS coverage for paddle.remainder (float32)."""

    def test_shapes(self):
        for shape in SHAPES:
            x, y = _safe_divmod_inputs(shape)
            with self.subTest(shape=shape):
                self._check("remainder", paddle.remainder, np.remainder,
                            x, y, rtol=1e-4, atol=1e-5)

    def test_sign_follows_divisor(self):
        # Python % semantics: 7%2=1, -7%2=1, 7%-2=-1, -7%-2=-1.
        x = np.array([7.0, -7.0, 7.0, -7.0, 5.5, -5.5], dtype=np.float32)
        y = np.array([2.0, 2.0, -2.0, -2.0, 2.0, 2.0], dtype=np.float32)
        expected = np.array([1.0, 1.0, -1.0, -1.0, 1.5, 0.5],
                            dtype=np.float32)
        out_mps, out_cpu = self._run_both(paddle.remainder, x, y)
        np.testing.assert_allclose(out_mps, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out_mps, out_cpu, rtol=1e-5, atol=1e-6)

    def test_result_in_divisor_range(self):
        # Result must lie in [0, y) for y > 0 and (y, 0] for y < 0.
        x, y = _safe_divmod_inputs((4, 5))
        out_mps, _ = self._run_both(paddle.remainder, x, y)
        pos = y > 0
        self.assertTrue(np.all(out_mps[pos] >= 0.0))
        self.assertTrue(np.all(out_mps[pos] < y[pos]))
        neg = ~pos
        self.assertTrue(np.all(out_mps[neg] <= 0.0))
        self.assertTrue(np.all(out_mps[neg] > y[neg]))

    def test_zero_dividend(self):
        x = np.zeros((7,), dtype=np.float32)
        y = np.array([1.5, -1.5, 2.0, -2.0, 0.7, -0.7, 3.3],
                     dtype=np.float32)
        self._check("remainder", paddle.remainder, np.remainder, x, y,
                    rtol=1e-4, atol=1e-5)

    def test_small_dividend_large_divisor(self):
        # |x| < |y|: remainder is x when signs match, x + y otherwise.
        x = np.array([0.25, -0.25, 0.25, -0.25], dtype=np.float32)
        y = np.array([2.0, 2.0, -2.0, -2.0], dtype=np.float32)
        expected = np.array([0.25, 1.75, -1.75, -0.25], dtype=np.float32)
        out_mps, _ = self._run_both(paddle.remainder, x, y)
        np.testing.assert_allclose(out_mps, expected, rtol=1e-5, atol=1e-6)

    def test_broadcast(self):
        x, _ = _safe_divmod_inputs((2, 3, 4))
        y = np.array([1.3, -1.7, 2.1, -0.9], dtype=np.float32)
        q = x.astype(np.float64) / y.astype(np.float64)
        frac = q - np.floor(q)
        bad = (frac < 0.05) | (frac > 0.95)
        x = np.where(bad, x + 0.37 * y, x).astype(np.float32)
        self._check("remainder", paddle.remainder, np.remainder, x, y,
                    rtol=1e-4, atol=1e-5)

    def test_dtype_and_place(self):
        x, y = _safe_divmod_inputs((3, 4))
        self._check_dtype_and_place(paddle.remainder, x, y)


# ---------------------------------------------------------------------------
# fmax / fmin: NaN-ignoring extrema.
# ---------------------------------------------------------------------------


class TestMPSFMaxFMinKernels(_MPSBinaryMathTestBase):
    """MPS coverage for paddle.fmax / paddle.fmin (float32)."""

    _OPS = (
        ("fmax", lambda a, b: paddle.fmax(a, b), np.fmax),
        ("fmin", lambda a, b: paddle.fmin(a, b), np.fmin),
    )

    def test_shapes(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            y = np.random.randn(*shape).astype(np.float32)
            for name, p_op, n_op in self._OPS:
                with self.subTest(op=name, shape=shape):
                    self._check(name, p_op, n_op, x, y)

    def test_nan_handling(self):
        # If one operand is NaN the other is returned; NaN only when both
        # are NaN. np.testing.assert_allclose treats matching NaNs as equal.
        nan = np.float32(np.nan)
        x = np.array([1.0, nan, nan, 3.0, -2.0], dtype=np.float32)
        y = np.array([2.0, 5.0, nan, -1.0, nan], dtype=np.float32)
        expected = {
            "fmax": np.array([2.0, 5.0, nan, 3.0, -2.0], dtype=np.float32),
            "fmin": np.array([1.0, 5.0, nan, -1.0, -2.0], dtype=np.float32),
        }
        for name, p_op, n_op in self._OPS:
            with self.subTest(op=name):
                out_mps, out_cpu = self._run_both(p_op, x, y)
                np.testing.assert_allclose(out_mps, expected[name],
                                           rtol=0, atol=0)
                np.testing.assert_allclose(out_mps, n_op(x, y),
                                           rtol=0, atol=0)
                np.testing.assert_allclose(out_mps, out_cpu, rtol=0, atol=0)

    def test_known_values(self):
        x = np.array([1.0, -1.0, 0.0, 2.5], dtype=np.float32)
        y = np.array([-1.0, 1.0, 0.0, 2.5], dtype=np.float32)
        out_max, _ = self._run_both(paddle.fmax, x, y)
        out_min, _ = self._run_both(paddle.fmin, x, y)
        np.testing.assert_allclose(
            out_max, np.array([1.0, 1.0, 0.0, 2.5], dtype=np.float32),
            rtol=0, atol=0)
        np.testing.assert_allclose(
            out_min, np.array([-1.0, -1.0, 0.0, 2.5], dtype=np.float32),
            rtol=0, atol=0)

    def test_large_magnitudes_and_inf(self):
        inf = np.float32(np.inf)
        x = np.array([1e30, -1e30, inf, -inf], dtype=np.float32)
        y = np.array([-1e30, 1e30, -inf, inf], dtype=np.float32)
        for name, p_op, n_op in self._OPS:
            with self.subTest(op=name):
                self._check(name, p_op, n_op, x, y, rtol=0, atol=0)

    def test_zeros_and_negatives(self):
        x = np.array([0.0, -0.0, -3.5, -7.25], dtype=np.float32)
        y = np.array([-0.0, 0.0, -3.25, -7.5], dtype=np.float32)
        for name, p_op, n_op in self._OPS:
            with self.subTest(op=name):
                self._check(name, p_op, n_op, x, y, rtol=0, atol=0)

    def test_empty_tensor(self):
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)
        out = paddle.fmax(
            paddle.to_tensor(x, place="mps"),
            paddle.to_tensor(y, place="mps"),
        )
        self.assertEqual(out.shape, [0])

    def test_dtype_and_place(self):
        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(3, 4).astype(np.float32)
        for name, p_op, _ in self._OPS:
            with self.subTest(op=name):
                self._check_dtype_and_place(p_op, x, y)


# ---------------------------------------------------------------------------
# heaviside: x < 0 -> 0, x == 0 -> y, x > 0 -> 1.
# ---------------------------------------------------------------------------


class TestMPSHeavisideKernel(_MPSBinaryMathTestBase):
    """MPS coverage for paddle.heaviside (float32)."""

    def test_shapes(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            y = np.random.randn(*shape).astype(np.float32)
            with self.subTest(shape=shape):
                self._check("heaviside", paddle.heaviside, np.heaviside,
                            x, y, rtol=0, atol=0)

    def test_known_values(self):
        x = np.array([-2.0, -0.5, 0.0, 0.0, 3.0], dtype=np.float32)
        y = np.array([10.0, 20.0, 30.0, -40.0, 50.0], dtype=np.float32)
        expected = np.array([0.0, 0.0, 30.0, -40.0, 1.0], dtype=np.float32)
        out_mps, out_cpu = self._run_both(paddle.heaviside, x, y)
        np.testing.assert_allclose(out_mps, expected, rtol=0, atol=0)
        np.testing.assert_allclose(out_mps, out_cpu, rtol=0, atol=0)

    def test_x_zero_returns_y(self):
        # Entire x == 0 row: output must be y verbatim, including negatives
        # and zeros in y.
        x = np.zeros((2, 5), dtype=np.float32)
        y = np.array([[1.5, -2.5, 0.0, -0.0, 100.0],
                      [-1e10, 1e10, 0.25, -0.25, 7.0]], dtype=np.float32)
        self._check("heaviside", paddle.heaviside, np.heaviside, x, y,
                    rtol=0, atol=0)

    def test_y_nonzero_everywhere(self):
        x = np.random.randn(3, 4).astype(np.float32)
        y = (np.random.uniform(0.5, 2.0, size=(3, 4)) *
             np.where(np.random.rand(3, 4) < 0.5, -1.0, 1.0)).astype(
                 np.float32)
        self._check("heaviside", paddle.heaviside, np.heaviside, x, y,
                    rtol=0, atol=0)

    def test_large_magnitudes(self):
        x = np.array([-1e30, 1e30, -1e-30, 1e-30], dtype=np.float32)
        y = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        expected = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        out_mps, _ = self._run_both(paddle.heaviside, x, y)
        np.testing.assert_allclose(out_mps, expected, rtol=0, atol=0)

    def test_nan_x_matches_cpu(self):
        # Paddle's functor returns 0 for NaN x (NaN fails both x == 0 and
        # x > 0), unlike np.heaviside which propagates NaN. Compare MPS
        # against the CPU backend only.
        x = np.array([np.nan, np.nan, 1.0, -1.0], dtype=np.float32)
        y = np.array([3.0, -3.0, 3.0, 3.0], dtype=np.float32)
        out_mps, out_cpu = self._run_both(paddle.heaviside, x, y)
        np.testing.assert_allclose(out_mps, out_cpu, rtol=0, atol=0)
        np.testing.assert_allclose(
            out_mps[:2], np.zeros(2, dtype=np.float32), rtol=0, atol=0)

    def test_broadcast(self):
        x = np.random.randn(2, 3, 4).astype(np.float32)
        x[0, 0, 0] = 0.0
        y = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        self._check("heaviside", paddle.heaviside, np.heaviside, x, y,
                    rtol=0, atol=0)

    def test_dtype_and_place(self):
        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(3, 4).astype(np.float32)
        self._check_dtype_and_place(paddle.heaviside, x, y)


# ---------------------------------------------------------------------------
# atan2: quadrant-aware arctangent; first arg is the numerator.
# ---------------------------------------------------------------------------


class TestMPSAtan2Kernel(_MPSBinaryMathTestBase):
    """MPS coverage for paddle.atan2 (float32)."""

    _RTOL = 1e-4
    _ATOL = 1e-5

    def test_shapes(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            y = np.random.randn(*shape).astype(np.float32)
            with self.subTest(shape=shape):
                self._check("atan2", paddle.atan2, np.arctan2, x, y,
                            rtol=self._RTOL, atol=self._ATOL)

    def test_four_quadrants(self):
        # One point per quadrant; results must respect the signs of x and y.
        x = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        y = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)
        expected = np.arctan2(x, y).astype(np.float32)  # pi/4, 3pi/4, ...
        out_mps, out_cpu = self._run_both(paddle.atan2, x, y)
        np.testing.assert_allclose(out_mps, expected,
                                   rtol=self._RTOL, atol=self._ATOL)
        np.testing.assert_allclose(out_mps, out_cpu,
                                   rtol=self._RTOL, atol=self._ATOL)

    def test_axes(self):
        # On-axis values: atan2(0, +) = 0, atan2(0, -) = pi,
        # atan2(+, 0) = pi/2, atan2(-, 0) = -pi/2.
        x = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
        y = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        expected = np.array(
            [0.0, np.pi, np.pi / 2.0, -np.pi / 2.0], dtype=np.float32)
        out_mps, out_cpu = self._run_both(paddle.atan2, x, y)
        np.testing.assert_allclose(out_mps, expected,
                                   rtol=self._RTOL, atol=self._ATOL)
        np.testing.assert_allclose(out_mps, out_cpu,
                                   rtol=self._RTOL, atol=self._ATOL)

    def test_known_values(self):
        # atan2(1, 1) = pi/4; atan2(sqrt(3), 1) = pi/3; atan2(1, sqrt(3)) = pi/6.
        s3 = np.float32(np.sqrt(3.0))
        x = np.array([1.0, s3, 1.0], dtype=np.float32)
        y = np.array([1.0, 1.0, s3], dtype=np.float32)
        expected = np.array(
            [np.pi / 4.0, np.pi / 3.0, np.pi / 6.0], dtype=np.float32)
        out_mps, _ = self._run_both(paddle.atan2, x, y)
        np.testing.assert_allclose(out_mps, expected,
                                   rtol=self._RTOL, atol=self._ATOL)

    def test_large_magnitudes(self):
        x = np.array([1e20, -1e20, 1e-20, -1e-20], dtype=np.float32)
        y = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self._check("atan2", paddle.atan2, np.arctan2, x, y,
                    rtol=self._RTOL, atol=self._ATOL)

    def test_result_range(self):
        # atan2 results always lie in [-pi, pi].
        x = np.random.randn(5, 6).astype(np.float32) * 10.0
        y = np.random.randn(5, 6).astype(np.float32) * 10.0
        out_mps, _ = self._run_both(paddle.atan2, x, y)
        self.assertTrue(np.all(out_mps >= -np.pi - 1e-5))
        self.assertTrue(np.all(out_mps <= np.pi + 1e-5))

    def test_broadcast(self):
        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(4).astype(np.float32)
        # Keep y away from 0 to avoid on-axis sign sensitivity.
        y = np.where(np.abs(y) < 0.1, y + 0.5, y).astype(np.float32)
        self._check("atan2", paddle.atan2, np.arctan2, x, y,
                    rtol=self._RTOL, atol=self._ATOL)

    def test_dtype_and_place(self):
        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(3, 4).astype(np.float32)
        self._check_dtype_and_place(paddle.atan2, x, y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
