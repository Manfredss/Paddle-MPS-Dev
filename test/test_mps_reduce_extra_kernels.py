#!/usr/bin/env python3
"""
Test MPS reduce-extra kernels by comparing with NumPy and the CPU backend.

Covers:
- paddle.prod (float32 input, float32 output)   -> TestMPSProdKernel
- paddle.any  (bool input, bool output)         -> TestMPSAnyKernel
- paddle.all  (bool input, bool output)         -> TestMPSAllKernel
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

    @staticmethod
    def _np_axis(axis):
        return tuple(axis) if isinstance(axis, list) else axis


# ---------------------------------------------------------------------------
# prod: float32 input -> float32 output.
# ---------------------------------------------------------------------------


class TestMPSProdKernel(_MPSKernelTestBase):
    """MPS coverage for paddle.prod."""

    def _rand_input(self, shape):
        # Well-conditioned magnitudes (products stay far from over/underflow)
        # combined with random signs to exercise sign handling.
        mag = np.random.uniform(0.5, 1.5, size=shape)
        sign = np.where(np.random.rand(*shape) < 0.5, -1.0, 1.0)
        return (mag * sign).astype(np.float32)

    def _prod_check(self, x_np, axis=None, keepdim=False, rtol=1e-4, atol=1e-5):
        out_mps = paddle.prod(
            paddle.to_tensor(x_np, place="mps"), axis=axis, keepdim=keepdim
        ).numpy()
        out_cpu = paddle.prod(
            paddle.to_tensor(x_np, place="cpu"), axis=axis, keepdim=keepdim
        ).numpy()
        ref = np.prod(
            x_np.astype(np.float64), axis=self._np_axis(axis), keepdims=keepdim
        ).astype(np.float32)
        np.testing.assert_allclose(
            out_mps, ref, rtol=rtol, atol=atol,
            err_msg=f"prod vs numpy (axis={axis}, keepdim={keepdim})",
        )
        np.testing.assert_allclose(
            out_mps, out_cpu, rtol=rtol, atol=atol,
            err_msg=f"prod vs cpu (axis={axis}, keepdim={keepdim})",
        )
        self.assertEqual(
            out_mps.shape, np.asarray(ref).shape,
            f"prod shape mismatch (axis={axis}, keepdim={keepdim})",
        )

    def test_reduce_all_default(self):
        for shape in SHAPES:
            x = self._rand_input(shape)
            for keepdim in (False, True):
                with self.subTest(shape=shape, keepdim=keepdim):
                    self._prod_check(x, axis=None, keepdim=keepdim)

    def test_single_axis(self):
        x = self._rand_input((2, 3, 4, 5))
        for axis in [0, 1, 2, 3, -1, -2]:
            for keepdim in (False, True):
                with self.subTest(axis=axis, keepdim=keepdim):
                    self._prod_check(x, axis=axis, keepdim=keepdim)

    def test_multiple_axes(self):
        x = self._rand_input((2, 3, 4, 5))
        for axes in [[0, 1], [2, 3], [0, 3], [1, 2, 3]]:
            for keepdim in (False, True):
                with self.subTest(axes=axes, keepdim=keepdim):
                    self._prod_check(x, axis=axes, keepdim=keepdim)

    def test_known_values(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        out = paddle.prod(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.float32(720.0), rtol=1e-5)
        out = paddle.prod(paddle.to_tensor(x, place="mps"), axis=0).numpy()
        np.testing.assert_allclose(out, [4.0, 10.0, 18.0], rtol=1e-5)
        out = paddle.prod(paddle.to_tensor(x, place="mps"), axis=1).numpy()
        np.testing.assert_allclose(out, [6.0, 120.0], rtol=1e-5)

    def test_zeros_propagate(self):
        x = np.array([[1.0, 0.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        out = paddle.prod(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.float32(0.0), atol=1e-6)
        out = paddle.prod(paddle.to_tensor(x, place="mps"), axis=0).numpy()
        np.testing.assert_allclose(out, [4.0, 0.0, 18.0], rtol=1e-5, atol=1e-6)

    def test_negative_values_sign(self):
        # Odd number of negatives -> negative product; even -> positive.
        x = np.array([-2.0, 3.0, -4.0, -5.0], dtype=np.float32)
        out = paddle.prod(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.float32(-120.0), rtol=1e-5)
        self._prod_check(x, axis=None, keepdim=False)

    def test_large_and_small_magnitudes(self):
        x = np.array([1e10, 1e-10, 5.0, 2.0], dtype=np.float32)
        self._prod_check(x, axis=None, keepdim=False)
        x = np.array([[1e8, 1e-8], [2.0, 0.5]], dtype=np.float32)
        self._prod_check(x, axis=0, keepdim=False)
        self._prod_check(x, axis=1, keepdim=True)

    def test_dtype_and_place(self):
        x = self._rand_input((3, 4))
        out = paddle.prod(paddle.to_tensor(x, place="mps"), axis=-1)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())

    def test_empty_tensor(self):
        # prod over an empty tensor is 1, the multiplicative identity
        # (matches np.prod(np.array([])) == 1.0).
        x = np.array([], dtype=np.float32)
        out = paddle.prod(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.float32(np.prod(x)), rtol=1e-6)
        np.testing.assert_allclose(out, np.float32(1.0), rtol=1e-6)
        # Reducing the empty axis of a (0, 3) input fills the output with 1.
        x = np.zeros((0, 3), dtype=np.float32)
        out = paddle.prod(paddle.to_tensor(x, place="mps"), axis=0).numpy()
        ref = np.prod(x, axis=0).astype(np.float32)
        np.testing.assert_allclose(out, ref, rtol=1e-6)
        np.testing.assert_allclose(out, np.ones(3, dtype=np.float32),
                                   rtol=1e-6)


# ---------------------------------------------------------------------------
# any / all: bool input -> bool output. Exact comparisons.
# ---------------------------------------------------------------------------


class _MPSBoolReduceBase(_MPSKernelTestBase):
    """Shared checker for paddle.any / paddle.all."""

    def _rand_bool(self, shape, p_true=0.5):
        return (np.random.rand(*shape) < p_true).astype(np.bool_)

    def _check(self, name, paddle_op, numpy_op, x_np, axis=None, keepdim=False):
        out_mps = paddle_op(
            paddle.to_tensor(x_np, place="mps"), axis=axis, keepdim=keepdim
        ).numpy()
        out_cpu = paddle_op(
            paddle.to_tensor(x_np, place="cpu"), axis=axis, keepdim=keepdim
        ).numpy()
        ref = numpy_op(x_np, axis=self._np_axis(axis), keepdims=keepdim)
        np.testing.assert_array_equal(
            out_mps, ref,
            err_msg=f"{name} vs numpy (axis={axis}, keepdim={keepdim})",
        )
        np.testing.assert_array_equal(
            out_mps, out_cpu,
            err_msg=f"{name} vs cpu (axis={axis}, keepdim={keepdim})",
        )
        self.assertEqual(
            out_mps.dtype, np.bool_, f"{name} output dtype must be bool"
        )
        self.assertEqual(
            out_mps.shape, np.asarray(ref).shape,
            f"{name} shape mismatch (axis={axis}, keepdim={keepdim})",
        )


class TestMPSAnyKernel(_MPSBoolReduceBase):
    """MPS coverage for paddle.any."""

    def test_reduce_all_default(self):
        for shape in SHAPES:
            x = self._rand_bool(shape)
            for keepdim in (False, True):
                with self.subTest(shape=shape, keepdim=keepdim):
                    self._check("any", paddle.any, np.any, x,
                                axis=None, keepdim=keepdim)

    def test_single_axis(self):
        # Sparse True values so per-slice results actually vary.
        x = self._rand_bool((2, 3, 4, 5), p_true=0.1)
        for axis in [0, 1, 2, 3, -1, -2]:
            for keepdim in (False, True):
                with self.subTest(axis=axis, keepdim=keepdim):
                    self._check("any", paddle.any, np.any, x,
                                axis=axis, keepdim=keepdim)

    def test_multiple_axes(self):
        x = self._rand_bool((2, 3, 4, 5), p_true=0.05)
        for axes in [[0, 1], [2, 3], [0, 3], [1, 2, 3]]:
            for keepdim in (False, True):
                with self.subTest(axes=axes, keepdim=keepdim):
                    self._check("any", paddle.any, np.any, x,
                                axis=axes, keepdim=keepdim)

    def test_known_values(self):
        x = np.array([[False, False, False],
                      [False, True, False]], dtype=np.bool_)
        out = paddle.any(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_array_equal(out, np.bool_(True))
        out = paddle.any(paddle.to_tensor(x, place="mps"), axis=0).numpy()
        np.testing.assert_array_equal(out, [False, True, False])
        out = paddle.any(paddle.to_tensor(x, place="mps"), axis=1).numpy()
        np.testing.assert_array_equal(out, [False, True])

    def test_all_false_and_all_true(self):
        for shape in SHAPES:
            with self.subTest(shape=shape, value=False):
                self._check("any", paddle.any, np.any,
                            np.zeros(shape, dtype=np.bool_))
            with self.subTest(shape=shape, value=True):
                self._check("any", paddle.any, np.any,
                            np.ones(shape, dtype=np.bool_))

    def test_single_true_element(self):
        x = np.zeros((2, 3, 5), dtype=np.bool_)
        x[1, 2, 3] = True
        for axis in [None, 0, 1, 2, [0, 2]]:
            with self.subTest(axis=axis):
                self._check("any", paddle.any, np.any, x, axis=axis)

    def test_dtype_and_place(self):
        x = self._rand_bool((3, 4))
        out = paddle.any(paddle.to_tensor(x, place="mps"), axis=-1)
        self.assertEqual(out.dtype, paddle.bool)
        self.assertTrue("mps" in str(out.place).lower())

    def test_empty_tensor(self):
        # any over an empty tensor is False, the identity of logical OR
        # (matches np.any(np.array([], dtype=bool)) == False).
        x = np.array([], dtype=np.bool_)
        out = paddle.any(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_array_equal(out, np.any(x))
        np.testing.assert_array_equal(out, np.bool_(False))
        # Reducing the empty axis of a (0, 3) input fills with False.
        x = np.zeros((0, 3), dtype=np.bool_)
        out = paddle.any(paddle.to_tensor(x, place="mps"), axis=0).numpy()
        np.testing.assert_array_equal(out, np.any(x, axis=0))
        np.testing.assert_array_equal(out, np.zeros(3, dtype=np.bool_))


class TestMPSAllKernel(_MPSBoolReduceBase):
    """MPS coverage for paddle.all."""

    def test_reduce_all_default(self):
        for shape in SHAPES:
            x = self._rand_bool(shape)
            for keepdim in (False, True):
                with self.subTest(shape=shape, keepdim=keepdim):
                    self._check("all", paddle.all, np.all, x,
                                axis=None, keepdim=keepdim)

    def test_single_axis(self):
        # Mostly-True values so per-slice results actually vary.
        x = self._rand_bool((2, 3, 4, 5), p_true=0.9)
        for axis in [0, 1, 2, 3, -1, -2]:
            for keepdim in (False, True):
                with self.subTest(axis=axis, keepdim=keepdim):
                    self._check("all", paddle.all, np.all, x,
                                axis=axis, keepdim=keepdim)

    def test_multiple_axes(self):
        x = self._rand_bool((2, 3, 4, 5), p_true=0.95)
        for axes in [[0, 1], [2, 3], [0, 3], [1, 2, 3]]:
            for keepdim in (False, True):
                with self.subTest(axes=axes, keepdim=keepdim):
                    self._check("all", paddle.all, np.all, x,
                                axis=axes, keepdim=keepdim)

    def test_known_values(self):
        x = np.array([[True, True, True],
                      [True, False, True]], dtype=np.bool_)
        out = paddle.all(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_array_equal(out, np.bool_(False))
        out = paddle.all(paddle.to_tensor(x, place="mps"), axis=0).numpy()
        np.testing.assert_array_equal(out, [True, False, True])
        out = paddle.all(paddle.to_tensor(x, place="mps"), axis=1).numpy()
        np.testing.assert_array_equal(out, [True, False])

    def test_all_false_and_all_true(self):
        for shape in SHAPES:
            with self.subTest(shape=shape, value=False):
                self._check("all", paddle.all, np.all,
                            np.zeros(shape, dtype=np.bool_))
            with self.subTest(shape=shape, value=True):
                self._check("all", paddle.all, np.all,
                            np.ones(shape, dtype=np.bool_))

    def test_single_false_element(self):
        x = np.ones((2, 3, 5), dtype=np.bool_)
        x[0, 1, 4] = False
        for axis in [None, 0, 1, 2, [0, 2]]:
            with self.subTest(axis=axis):
                self._check("all", paddle.all, np.all, x, axis=axis)

    def test_any_all_consistency(self):
        # not(all(x)) == any(not x) — De Morgan over reductions.
        x = self._rand_bool((3, 4, 5), p_true=0.8)
        x_mps = paddle.to_tensor(x, place="mps")
        lhs = paddle.logical_not(paddle.all(x_mps, axis=1)).numpy()
        rhs = paddle.any(paddle.logical_not(x_mps), axis=1).numpy()
        np.testing.assert_array_equal(lhs, rhs)

    def test_dtype_and_place(self):
        x = self._rand_bool((3, 4))
        out = paddle.all(paddle.to_tensor(x, place="mps"), axis=-1)
        self.assertEqual(out.dtype, paddle.bool)
        self.assertTrue("mps" in str(out.place).lower())

    def test_empty_tensor(self):
        # all over an empty tensor is True, the identity of logical AND
        # (matches np.all(np.array([], dtype=bool)) == True).
        x = np.array([], dtype=np.bool_)
        out = paddle.all(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_array_equal(out, np.all(x))
        np.testing.assert_array_equal(out, np.bool_(True))
        # Reducing the empty axis of a (0, 3) input fills with True.
        x = np.zeros((0, 3), dtype=np.bool_)
        out = paddle.all(paddle.to_tensor(x, place="mps"), axis=0).numpy()
        np.testing.assert_array_equal(out, np.all(x, axis=0))
        np.testing.assert_array_equal(out, np.ones(3, dtype=np.bool_))


if __name__ == '__main__':
    unittest.main()
