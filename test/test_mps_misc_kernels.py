#!/usr/bin/env python3
"""
Test MPS 'misc' kernels by comparing with NumPy references and the CPU backend.

Covers:
- scale     (paddle.scale, both bias_after_scale branches)  -> TestMPSScaleKernel
- clip      (paddle.clip)                                   -> TestMPSClipKernel
- where     (paddle.where, same-shape inputs)               -> TestMPSWhereKernel
- pow       (paddle.pow with scalar exponent)               -> TestMPSPowKernel
- isnan / isinf / isfinite (bool outputs)                   -> TestMPSIsfiniteKernels
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
# scale: out = scale * x + bias        (bias_after_scale=True)
#        out = scale * (x + bias)      (bias_after_scale=False)
# ---------------------------------------------------------------------------


def _scale_numpy(x, scale, bias, bias_after_scale):
    scale = np.float32(scale)
    bias = np.float32(bias)
    if bias_after_scale:
        return (scale * x + bias).astype(np.float32)
    return (scale * (x + bias)).astype(np.float32)


class TestMPSScaleKernel(_MPSKernelTestBase):
    """MPS coverage for paddle.scale."""

    def _scale_check(self, x_np, scale, bias, bias_after_scale,
                     rtol=1e-5, atol=1e-6):
        out_mps = paddle.scale(
            paddle.to_tensor(x_np, place="mps"),
            scale=scale, bias=bias, bias_after_scale=bias_after_scale,
        ).numpy()
        out_cpu = paddle.scale(
            paddle.to_tensor(x_np, place="cpu"),
            scale=scale, bias=bias, bias_after_scale=bias_after_scale,
        ).numpy()
        ref = _scale_numpy(x_np, scale, bias, bias_after_scale)
        np.testing.assert_allclose(
            out_mps, ref, rtol=rtol, atol=atol,
            err_msg=f"scale vs numpy (scale={scale}, bias={bias}, "
                    f"bias_after_scale={bias_after_scale})",
        )
        np.testing.assert_allclose(
            out_mps, out_cpu, rtol=rtol, atol=atol,
            err_msg=f"scale vs cpu (scale={scale}, bias={bias}, "
                    f"bias_after_scale={bias_after_scale})",
        )

    def test_scale_shapes_both_branches(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            for bias_after_scale in (True, False):
                with self.subTest(shape=shape, bias_after_scale=bias_after_scale):
                    self._scale_check(x, 2.0, 1.0, bias_after_scale)

    def test_scale_attribute_variations(self):
        x = np.random.randn(4, 5).astype(np.float32)
        for scale in [0.0, 1.0, -1.0, 0.5, 3.75, -2.5]:
            for bias in [0.0, 1.0, -3.0, 0.25]:
                for bias_after_scale in (True, False):
                    with self.subTest(scale=scale, bias=bias,
                                      bias_after_scale=bias_after_scale):
                        self._scale_check(x, scale, bias, bias_after_scale)

    def test_scale_known_values(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        # bias_after_scale=True: 3*x + 1
        out = paddle.scale(
            paddle.to_tensor(x, place="mps"),
            scale=3.0, bias=1.0, bias_after_scale=True,
        ).numpy()
        np.testing.assert_allclose(
            out, np.array([-5.0, -2.0, 1.0, 4.0, 7.0], dtype=np.float32),
            rtol=1e-6, atol=1e-6,
        )
        # bias_after_scale=False: 3*(x + 1)
        out = paddle.scale(
            paddle.to_tensor(x, place="mps"),
            scale=3.0, bias=1.0, bias_after_scale=False,
        ).numpy()
        np.testing.assert_allclose(
            out, np.array([-3.0, 0.0, 3.0, 6.0, 9.0], dtype=np.float32),
            rtol=1e-6, atol=1e-6,
        )

    def test_scale_edge_cases(self):
        zeros = np.zeros((3, 4), dtype=np.float32)
        self._scale_check(zeros, 5.0, -2.0, True)
        self._scale_check(zeros, 5.0, -2.0, False)
        negatives = -np.abs(np.random.randn(3, 4)).astype(np.float32)
        self._scale_check(negatives, -1.5, 0.5, True)
        large = (np.random.randn(3, 4) * 1e6).astype(np.float32)
        self._scale_check(large, 2.0, 100.0, True, rtol=1e-4, atol=1e-2)

    def test_scale_dtype_and_place(self):
        x = np.random.randn(3, 4).astype(np.float32)
        out = paddle.scale(paddle.to_tensor(x, place="mps"), scale=2.0)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


# ---------------------------------------------------------------------------
# clip: out = clamp(x, min, max)
# ---------------------------------------------------------------------------


class TestMPSClipKernel(_MPSKernelTestBase):
    """MPS coverage for paddle.clip."""

    def _clip_check(self, x_np, min_v, max_v, rtol=1e-6, atol=1e-7):
        out_mps = paddle.clip(
            paddle.to_tensor(x_np, place="mps"), min=min_v, max=max_v
        ).numpy()
        out_cpu = paddle.clip(
            paddle.to_tensor(x_np, place="cpu"), min=min_v, max=max_v
        ).numpy()
        ref = np.clip(x_np, min_v, max_v).astype(np.float32)
        np.testing.assert_allclose(
            out_mps, ref, rtol=rtol, atol=atol,
            err_msg=f"clip vs numpy (min={min_v}, max={max_v})",
        )
        np.testing.assert_allclose(
            out_mps, out_cpu, rtol=rtol, atol=atol,
            err_msg=f"clip vs cpu (min={min_v}, max={max_v})",
        )

    def test_clip_shapes(self):
        for shape in SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 3.0).astype(np.float32)
                self._clip_check(x, -1.0, 1.0)

    def test_clip_bound_variations(self):
        x = (np.random.randn(4, 6) * 5.0).astype(np.float32)
        for min_v, max_v in [(-2.0, 2.0), (0.0, 1.0), (-0.5, 0.5),
                             (-10.0, 10.0), (1.0, 1.0), (-3.5, -1.5)]:
            with self.subTest(min=min_v, max=max_v):
                self._clip_check(x, min_v, max_v)

    def test_clip_known_values(self):
        x = np.array([-5.0, -1.0, 0.0, 0.5, 1.0, 5.0], dtype=np.float32)
        out = paddle.clip(
            paddle.to_tensor(x, place="mps"), min=-1.0, max=1.0
        ).numpy()
        np.testing.assert_allclose(
            out, np.array([-1.0, -1.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32),
            rtol=1e-6, atol=1e-7,
        )

    def test_clip_edge_cases(self):
        # All elements below min.
        x = np.full((3, 4), -100.0, dtype=np.float32)
        self._clip_check(x, -1.0, 1.0)
        # All elements above max.
        x = np.full((3, 4), 100.0, dtype=np.float32)
        self._clip_check(x, -1.0, 1.0)
        # Zeros inside the range.
        x = np.zeros((3, 4), dtype=np.float32)
        self._clip_check(x, -1.0, 1.0)
        # Large magnitudes with wide bounds.
        x = (np.random.randn(3, 4) * 1e6).astype(np.float32)
        self._clip_check(x, -1e5, 1e5)

    def test_clip_dtype_and_place(self):
        x = np.random.randn(3, 4).astype(np.float32)
        out = paddle.clip(paddle.to_tensor(x, place="mps"), min=-1.0, max=1.0)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


# ---------------------------------------------------------------------------
# where: out[i] = condition[i] ? x[i] : y[i]   (same-shape inputs only)
# ---------------------------------------------------------------------------


class TestMPSWhereKernel(_MPSKernelTestBase):
    """MPS coverage for paddle.where with same-shape inputs."""

    def _where_check(self, cond_np, x_np, y_np, rtol=1e-6, atol=1e-7):
        out_mps = paddle.where(
            paddle.to_tensor(cond_np, place="mps"),
            paddle.to_tensor(x_np, place="mps"),
            paddle.to_tensor(y_np, place="mps"),
        ).numpy()
        out_cpu = paddle.where(
            paddle.to_tensor(cond_np, place="cpu"),
            paddle.to_tensor(x_np, place="cpu"),
            paddle.to_tensor(y_np, place="cpu"),
        ).numpy()
        ref = np.where(cond_np, x_np, y_np).astype(np.float32)
        np.testing.assert_allclose(out_mps, ref, rtol=rtol, atol=atol,
                                   err_msg="where vs numpy")
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol,
                                   err_msg="where vs cpu")

    def test_where_shapes(self):
        for shape in SHAPES:
            with self.subTest(shape=shape):
                cond = np.random.randint(0, 2, size=shape).astype(np.bool_)
                x = np.random.randn(*shape).astype(np.float32)
                y = np.random.randn(*shape).astype(np.float32)
                self._where_check(cond, x, y)

    def test_where_known_values(self):
        cond = np.array([True, False, True, False], dtype=np.bool_)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
        out = paddle.where(
            paddle.to_tensor(cond, place="mps"),
            paddle.to_tensor(x, place="mps"),
            paddle.to_tensor(y, place="mps"),
        ).numpy()
        np.testing.assert_allclose(
            out, np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32),
            rtol=1e-6, atol=1e-7,
        )

    def test_where_all_true_all_false(self):
        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(3, 4).astype(np.float32)
        all_true = np.ones((3, 4), dtype=np.bool_)
        all_false = np.zeros((3, 4), dtype=np.bool_)
        self._where_check(all_true, x, y)
        self._where_check(all_false, x, y)

    def test_where_edge_cases(self):
        cond = np.random.randint(0, 2, size=(3, 4)).astype(np.bool_)
        zeros = np.zeros((3, 4), dtype=np.float32)
        negatives = -np.abs(np.random.randn(3, 4)).astype(np.float32)
        large = (np.random.randn(3, 4) * 1e6).astype(np.float32)
        self._where_check(cond, zeros, negatives)
        self._where_check(cond, large, zeros)
        self._where_check(cond, negatives, large)

    def test_where_dtype_and_place(self):
        cond = np.random.randint(0, 2, size=(3, 4)).astype(np.bool_)
        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(3, 4).astype(np.float32)
        out = paddle.where(
            paddle.to_tensor(cond, place="mps"),
            paddle.to_tensor(x, place="mps"),
            paddle.to_tensor(y, place="mps"),
        )
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


# ---------------------------------------------------------------------------
# pow: out = x ^ factor (scalar exponent)
# ---------------------------------------------------------------------------


class TestMPSPowKernel(_MPSKernelTestBase):
    """MPS coverage for paddle.pow with a scalar exponent."""

    def _pow_check(self, x_np, factor, rtol=1e-4, atol=1e-5):
        out_mps = paddle.pow(paddle.to_tensor(x_np, place="mps"), factor).numpy()
        out_cpu = paddle.pow(paddle.to_tensor(x_np, place="cpu"), factor).numpy()
        ref = np.power(x_np, np.float32(factor)).astype(np.float32)
        np.testing.assert_allclose(out_mps, ref, rtol=rtol, atol=atol,
                                   err_msg=f"pow vs numpy (factor={factor})")
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol,
                                   err_msg=f"pow vs cpu (factor={factor})")

    def test_pow_shapes_positive_base(self):
        for shape in SHAPES:
            x = np.random.uniform(0.1, 3.0, size=shape).astype(np.float32)
            for factor in [2.0, 0.5, 3.0]:
                with self.subTest(shape=shape, factor=factor):
                    self._pow_check(x, factor)

    def test_pow_factor_variations(self):
        x = np.random.uniform(0.5, 2.0, size=(4, 5)).astype(np.float32)
        for factor in [0.0, 1.0, 2.0, 3.0, 0.5, -1.0, -2.0, 1.5]:
            with self.subTest(factor=factor):
                self._pow_check(x, factor)

    def test_pow_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = paddle.pow(paddle.to_tensor(x, place="mps"), 2.0).numpy()
        np.testing.assert_allclose(
            out, np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32),
            rtol=1e-5, atol=1e-6,
        )
        out = paddle.pow(paddle.to_tensor(x, place="mps"), 0.5).numpy()
        np.testing.assert_allclose(
            out, np.sqrt(x), rtol=1e-4, atol=1e-5,
        )

    def test_pow_factor_zero_gives_ones(self):
        x = np.random.uniform(0.1, 5.0, size=(3, 4)).astype(np.float32)
        out = paddle.pow(paddle.to_tensor(x, place="mps"), 0.0).numpy()
        np.testing.assert_allclose(out, np.ones_like(x), rtol=1e-6, atol=1e-7)

    def test_pow_factor_one_is_identity(self):
        x = np.random.uniform(0.1, 5.0, size=(3, 4)).astype(np.float32)
        out = paddle.pow(paddle.to_tensor(x, place="mps"), 1.0).numpy()
        np.testing.assert_allclose(out, x, rtol=1e-6, atol=1e-7)

    def test_pow_edge_cases(self):
        # Zeros with positive exponent -> zeros.
        zeros = np.zeros((3, 4), dtype=np.float32)
        self._pow_check(zeros, 2.0)
        # Large magnitudes squared.
        large = np.random.uniform(100.0, 1000.0, size=(3, 4)).astype(np.float32)
        self._pow_check(large, 2.0, rtol=1e-4, atol=1e-1)
        # Negative bases with integer exponents.
        negatives = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        out_cpu = paddle.pow(paddle.to_tensor(negatives, place="cpu"), 2.0).numpy()
        out_mps = paddle.pow(paddle.to_tensor(negatives, place="mps"), 2.0).numpy()
        np.testing.assert_allclose(out_mps, out_cpu, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            out_mps, np.array([1.0, 4.0, 9.0], dtype=np.float32),
            rtol=1e-5, atol=1e-6,
        )

    def test_pow_dtype_and_place(self):
        x = np.random.uniform(0.5, 2.0, size=(3, 4)).astype(np.float32)
        out = paddle.pow(paddle.to_tensor(x, place="mps"), 2.0)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


# ---------------------------------------------------------------------------
# isnan / isinf / isfinite: bool outputs.
# ---------------------------------------------------------------------------


class TestMPSIsfiniteKernels(_MPSKernelTestBase):
    """MPS coverage for paddle.isnan / paddle.isinf / paddle.isfinite."""

    _OPS = (
        ("isnan", lambda t: paddle.isnan(t), np.isnan),
        ("isinf", lambda t: paddle.isinf(t), np.isinf),
        ("isfinite", lambda t: paddle.isfinite(t), np.isfinite),
    )

    @staticmethod
    def _special_values_array():
        return np.array(
            [0.0, 1.0, -1.0, np.nan, np.inf, -np.inf, 3.5, -2.25,
             np.nan, -np.inf, 1e30, -1e30],
            dtype=np.float32,
        )

    def _check(self, name, paddle_op, numpy_op, x_np):
        out_mps = paddle_op(paddle.to_tensor(x_np, place="mps")).numpy()
        out_cpu = paddle_op(paddle.to_tensor(x_np, place="cpu")).numpy()
        ref = numpy_op(x_np)
        np.testing.assert_array_equal(out_mps, ref,
                                      err_msg=f"{name} vs numpy")
        np.testing.assert_array_equal(out_mps, out_cpu,
                                      err_msg=f"{name} vs cpu")

    def test_finite_random_shapes(self):
        # Random finite data: isnan/isinf all False, isfinite all True.
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            for name, p_op, n_op in self._OPS:
                with self.subTest(op=name, shape=shape):
                    self._check(name, p_op, n_op, x)

    def test_special_values_1d(self):
        x = self._special_values_array()
        for name, p_op, n_op in self._OPS:
            with self.subTest(op=name):
                self._check(name, p_op, n_op, x)

    def test_special_values_embedded_in_shapes(self):
        for shape in SHAPES:
            x = np.random.randn(*shape).astype(np.float32)
            flat = x.reshape(-1)
            flat[0] = np.nan
            if flat.size > 2:
                flat[1] = np.inf
                flat[2] = -np.inf
            x = flat.reshape(shape)
            for name, p_op, n_op in self._OPS:
                with self.subTest(op=name, shape=shape):
                    self._check(name, p_op, n_op, x)

    def test_known_values(self):
        x = np.array([np.nan, np.inf, -np.inf, 0.0, 1.0], dtype=np.float32)
        x_mps = paddle.to_tensor(x, place="mps")
        np.testing.assert_array_equal(
            paddle.isnan(x_mps).numpy(),
            np.array([True, False, False, False, False]),
        )
        np.testing.assert_array_equal(
            paddle.isinf(x_mps).numpy(),
            np.array([False, True, True, False, False]),
        )
        np.testing.assert_array_equal(
            paddle.isfinite(x_mps).numpy(),
            np.array([False, False, False, True, True]),
        )

    def test_edge_cases(self):
        # Zeros are finite; large magnitudes still finite in float32.
        zeros = np.zeros((3, 4), dtype=np.float32)
        large = np.full((3, 4), 3.0e38, dtype=np.float32)
        negatives = np.full((3, 4), -3.0e38, dtype=np.float32)
        for name, p_op, n_op in self._OPS:
            for x in (zeros, large, negatives):
                with self.subTest(op=name):
                    self._check(name, p_op, n_op, x)
        # float32 overflow to inf.
        overflow = (np.full((4,), 3.0e38, dtype=np.float32) * 2.0).astype(np.float32)
        for name, p_op, n_op in self._OPS:
            with self.subTest(op=name, case="overflow"):
                self._check(name, p_op, n_op, overflow)

    def test_dtype_and_place(self):
        x = np.array([np.nan, 1.0, np.inf], dtype=np.float32)
        for name, p_op, _ in self._OPS:
            with self.subTest(op=name):
                out = p_op(paddle.to_tensor(x, place="mps"))
                self.assertEqual(out.dtype, paddle.bool)
                self.assertTrue("mps" in str(out.place).lower())


if __name__ == "__main__":
    unittest.main()
