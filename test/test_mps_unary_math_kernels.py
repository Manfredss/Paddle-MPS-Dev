#!/usr/bin/env python3
"""
Test MPS unary-math kernels against NumPy references and the CPU backend.

Covers:
- asinh / acosh / atanh (inverse hyperbolic functions)
- erf (error function)
- expm1 / log1p (numerically-motivated exp/log variants)
- trunc (round toward zero)

All tests are float32-only and compare the MPS output against BOTH a
hand-written NumPy reference AND the Paddle CPU backend.
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


# ---------------------------------------------------------------------------
# NumPy references.
# ---------------------------------------------------------------------------


def _erf_numpy(x):
    erf_vec = np.vectorize(math.erf, otypes=[np.float64])
    return erf_vec(x).astype(np.float32)


# ---------------------------------------------------------------------------
# Domain-aware random input generators (float32).
# ---------------------------------------------------------------------------


def _rand_real(shape):
    """Unrestricted real inputs (standard normal)."""
    return np.random.randn(*shape).astype(np.float32)


def _rand_acosh(shape):
    """acosh requires x >= 1."""
    return np.random.uniform(1.0, 10.0, size=shape).astype(np.float32)


def _rand_atanh(shape):
    """atanh requires |x| < 1; stay away from the poles."""
    return np.random.uniform(-0.95, 0.95, size=shape).astype(np.float32)


def _rand_log1p(shape):
    """log1p requires x > -1; stay away from the pole."""
    return np.random.uniform(-0.9, 9.0, size=shape).astype(np.float32)


def _rand_trunc(shape):
    """Spread values across several integer bins, both signs."""
    return np.random.uniform(-20.0, 20.0, size=shape).astype(np.float32)


class TestMPSUnaryMathKernels(_MPSKernelTestBase):
    """MPS coverage for paddle.asinh / acosh / atanh / erf / expm1 / log1p /
    trunc."""

    # Each op tuple:
    #   (name, paddle fn, numpy reference, input generator, (rtol, atol), exact)
    _OPS = (
        ("asinh", lambda t: paddle.asinh(t), np.arcsinh, _rand_real, (1e-4, 1e-5), False),
        ("acosh", lambda t: paddle.acosh(t), np.arccosh, _rand_acosh, (1e-4, 1e-5), False),
        ("atanh", lambda t: paddle.atanh(t), np.arctanh, _rand_atanh, (1e-4, 1e-5), False),
        ("erf",   lambda t: paddle.erf(t),   _erf_numpy, _rand_real, (1e-4, 1e-5), False),
        ("expm1", lambda t: paddle.expm1(t), np.expm1,   _rand_real, (1e-4, 1e-5), False),
        ("log1p", lambda t: paddle.log1p(t), np.log1p,   _rand_log1p, (1e-4, 1e-5), False),
        ("trunc", lambda t: paddle.trunc(t), np.trunc,   _rand_trunc, (0.0, 0.0), True),
    )

    def _check(self, name, paddle_op, numpy_op, x_np, tols, exact):
        out_mps = paddle_op(paddle.to_tensor(x_np, place="mps")).numpy()
        out_cpu = paddle_op(paddle.to_tensor(x_np, place="cpu")).numpy()
        ref = np.asarray(numpy_op(x_np)).astype(np.float32)
        if exact:
            np.testing.assert_array_equal(out_mps, ref,
                                          err_msg=f"{name} vs numpy")
            np.testing.assert_array_equal(out_mps, out_cpu,
                                          err_msg=f"{name} vs cpu")
        else:
            rtol, atol = tols
            np.testing.assert_allclose(out_mps, ref, rtol=rtol, atol=atol,
                                       err_msg=f"{name} vs numpy")
            np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol,
                                       err_msg=f"{name} vs cpu")
        self.assertEqual(out_mps.dtype, np.float32,
                         f"{name} output dtype must be float32")
        self.assertEqual(out_mps.shape, ref.shape,
                         f"{name} shape mismatch")

    # ---- shape sweep ------------------------------------------------------
    def test_shapes(self):
        for shape in SHAPES:
            for name, p_op, n_op, gen, tols, exact in self._OPS:
                with self.subTest(op=name, shape=shape):
                    self._check(name, p_op, n_op, gen(shape), tols, exact)

    # ---- known values -----------------------------------------------------
    def test_asinh_known_values(self):
        x = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        expected = np.array(
            [0.0, 0.8813736, -0.8813736, 1.4436355], dtype=np.float32
        )
        out = paddle.asinh(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_acosh_known_values(self):
        x = np.array([1.0, 2.0, 10.0], dtype=np.float32)
        expected = np.array([0.0, 1.3169579, 2.9932228], dtype=np.float32)
        out = paddle.acosh(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_atanh_known_values(self):
        x = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        expected = np.array([0.0, 0.5493061, -0.5493061], dtype=np.float32)
        out = paddle.atanh(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_erf_known_values(self):
        x = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        expected = np.array(
            [0.0, 0.8427008, -0.8427008, 0.9953223], dtype=np.float32
        )
        out = paddle.erf(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_expm1_known_values(self):
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        expected = np.array(
            [0.0, math.e - 1.0, math.exp(-1.0) - 1.0], dtype=np.float32
        )
        out = paddle.expm1(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_log1p_known_values(self):
        x = np.array([0.0, math.e - 1.0, -0.5], dtype=np.float32)
        expected = np.array([0.0, 1.0, -0.6931472], dtype=np.float32)
        out = paddle.log1p(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_trunc_known_values(self):
        x = np.array(
            [1.7, -1.7, 0.5, -0.5, 2.0, -3.9, 0.0], dtype=np.float32
        )
        expected = np.array(
            [1.0, -1.0, 0.0, -0.0, 2.0, -3.0, 0.0], dtype=np.float32
        )
        out = paddle.trunc(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    # ---- identities -------------------------------------------------------
    def test_zero_maps_to_zero(self):
        # asinh(0) = atanh(0) = erf(0) = expm1(0) = log1p(0) = trunc(0) = 0,
        # and acosh(1) = 0.
        z = np.zeros((3, 4), dtype=np.float32)
        for name, p_op in (
            ("asinh", lambda t: paddle.asinh(t)),
            ("atanh", lambda t: paddle.atanh(t)),
            ("erf",   lambda t: paddle.erf(t)),
            ("expm1", lambda t: paddle.expm1(t)),
            ("log1p", lambda t: paddle.log1p(t)),
            ("trunc", lambda t: paddle.trunc(t)),
        ):
            with self.subTest(op=name):
                out = p_op(paddle.to_tensor(z, place="mps")).numpy()
                np.testing.assert_allclose(out, z, rtol=0, atol=1e-7)
        ones = np.ones((3, 4), dtype=np.float32)
        out = paddle.acosh(paddle.to_tensor(ones, place="mps")).numpy()
        np.testing.assert_allclose(out, np.zeros_like(ones), rtol=0, atol=1e-6)

    def test_odd_symmetry(self):
        # asinh, atanh, erf are odd functions: f(-x) == -f(x).
        x = np.random.uniform(0.05, 0.9, size=(4, 5)).astype(np.float32)
        for name, p_op in (
            ("asinh", lambda t: paddle.asinh(t)),
            ("atanh", lambda t: paddle.atanh(t)),
            ("erf",   lambda t: paddle.erf(t)),
        ):
            with self.subTest(op=name):
                pos = p_op(paddle.to_tensor(x, place="mps")).numpy()
                neg = p_op(paddle.to_tensor(-x, place="mps")).numpy()
                np.testing.assert_allclose(neg, -pos, rtol=1e-4, atol=1e-5)

    def test_expm1_log1p_inverse_roundtrip(self):
        # log1p(expm1(x)) == x on a moderate range.
        x = np.random.uniform(-2.0, 2.0, size=(3, 4)).astype(np.float32)
        y = paddle.expm1(paddle.to_tensor(x, place="mps"))
        out = paddle.log1p(y).numpy()
        np.testing.assert_allclose(out, x, rtol=1e-4, atol=1e-5)

    # ---- edge cases -------------------------------------------------------
    def test_negative_inputs(self):
        x = np.array([-0.25, -1.5, -3.0, -7.5], dtype=np.float32)
        # asinh / erf / expm1 accept all negatives; trunc rounds toward zero.
        np.testing.assert_allclose(
            paddle.asinh(paddle.to_tensor(x, place="mps")).numpy(),
            np.arcsinh(x), rtol=1e-4, atol=1e-5,
        )
        np.testing.assert_allclose(
            paddle.erf(paddle.to_tensor(x, place="mps")).numpy(),
            _erf_numpy(x), rtol=1e-4, atol=1e-5,
        )
        np.testing.assert_allclose(
            paddle.expm1(paddle.to_tensor(x, place="mps")).numpy(),
            np.expm1(x), rtol=1e-4, atol=1e-5,
        )
        np.testing.assert_array_equal(
            paddle.trunc(paddle.to_tensor(x, place="mps")).numpy(),
            np.trunc(x),
        )

    def test_large_magnitudes(self):
        # asinh grows like log(2x): well-defined for large |x|.
        x = np.array([1e4, -1e4, 1e6, -1e6], dtype=np.float32)
        out = paddle.asinh(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.arcsinh(x), rtol=1e-4, atol=1e-5)
        # erf saturates to +/-1.
        x = np.array([10.0, -10.0, 30.0, -30.0], dtype=np.float32)
        out = paddle.erf(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(
            out, np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
            rtol=0, atol=1e-6,
        )
        # expm1 approaches -1 for very negative x; grows like exp(x) otherwise.
        x = np.array([-50.0, -20.0, 10.0, 20.0], dtype=np.float32)
        out = paddle.expm1(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.expm1(x), rtol=1e-4, atol=1e-5)
        # log1p of large positive input.
        x = np.array([1e4, 1e6, 3e7], dtype=np.float32)
        out = paddle.log1p(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.log1p(x), rtol=1e-4, atol=1e-5)
        # trunc on large magnitudes (still exactly representable in fp32).
        x = np.array([12345.678, -12345.678, 99999.5], dtype=np.float32)
        out = paddle.trunc(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_array_equal(out, np.trunc(x))

    def test_acosh_near_domain_boundary(self):
        x = np.array([1.0, 1.0001, 1.01, 1.5], dtype=np.float32)
        out = paddle.acosh(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.arccosh(x), rtol=1e-3, atol=1e-3)

    def test_atanh_near_domain_boundary(self):
        x = np.array([0.99, -0.99, 0.999, -0.999], dtype=np.float32)
        out = paddle.atanh(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.arctanh(x), rtol=1e-3, atol=1e-4)

    def test_trunc_integer_inputs_unchanged(self):
        x = np.array([-5.0, -1.0, 0.0, 1.0, 7.0, 100.0], dtype=np.float32)
        out = paddle.trunc(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_array_equal(out, x)

    def test_small_inputs(self):
        # expm1(x) ~ x and log1p(x) ~ x for tiny x (absolute-error check).
        x = np.array([1e-6, -1e-6, 1e-4, -1e-4], dtype=np.float32)
        out = paddle.expm1(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, x, rtol=0, atol=1e-5)
        out = paddle.log1p(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, x, rtol=0, atol=1e-5)
        # asinh(x) ~ x and erf(x) ~ 2x/sqrt(pi) for tiny x.
        out = paddle.asinh(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, x, rtol=0, atol=1e-5)
        out = paddle.erf(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(
            out, x * np.float32(2.0 / math.sqrt(math.pi)), rtol=0, atol=1e-5
        )

    # ---- dtype / place ----------------------------------------------------
    def test_output_dtype_and_place(self):
        for name, p_op, _n_op, gen, _tols, _exact in self._OPS:
            with self.subTest(op=name):
                x = gen((3, 4))
                out = p_op(paddle.to_tensor(x, place="mps"))
                self.assertEqual(out.dtype, paddle.float32)
                self.assertTrue("mps" in str(out.place).lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)
