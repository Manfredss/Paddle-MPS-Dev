#!/usr/bin/env python3
"""
Test MPS kernels by comparing with PyTorch MPS / NumPy outputs.

Covers:
- Elementwise binary ops (add, multiply, subtract, divide, ...) -> TestMPSElementwiseKernels
- Activations (softmax, gelu, leaky_relu)                       -> TestMPSActivationKernels
- Reductions (mean, sum, max, min)                              -> TestMPSReductionKernels
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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available for reference comparison")


class TestMPSElementwiseKernels(unittest.TestCase):
    """Test MPS elementwise kernels against PyTorch reference."""

    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        if not PADDLE_AVAILABLE:
            raise unittest.SkipTest("PaddlePaddle not available")
        
        if not paddle.is_compiled_with_mps():
            raise unittest.SkipTest("PaddlePaddle not compiled with MPS support")
        
        if not paddle.mps.is_available():
            raise unittest.SkipTest("MPS not available on this system")
        
        # Set device to MPS
        paddle.mps.set_device(0)
        cls.device = 'mps'
        
        # Set PyTorch device if available
        if TORCH_AVAILABLE and torch.backends.mps.is_available():
            cls.torch_device = torch.device('mps')
            cls.use_torch_ref = True
        else:
            cls.use_torch_ref = False
            print("Warning: PyTorch MPS not available, using numpy reference only")

    def setUp(self):
        """Set up each test."""
        np.random.seed(42)
        paddle.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)

    def assert_allclose(self, paddle_result, expected, rtol=1e-5, atol=1e-6, msg=""):
        """Assert that paddle result is close to expected."""
        if isinstance(paddle_result, paddle.Tensor):
            paddle_np = paddle_result.numpy()
        else:
            paddle_np = np.array(paddle_result)
        
        if isinstance(expected, torch.Tensor):
            expected_np = expected.cpu().numpy()
        else:
            expected_np = np.array(expected)
        
        np.testing.assert_allclose(
            paddle_np, expected_np, rtol=rtol, atol=atol, err_msg=msg
        )

    def test_add_same_shape(self):
        """Test elementwise add with same shapes."""
        shape = [3, 4, 5]
        x_np = np.random.randn(*shape).astype(np.float32)
        y_np = np.random.randn(*shape).astype(np.float32)
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.add(x_paddle, y_paddle)
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            result_torch = torch.add(x_torch, y_torch)
            self.assert_allclose(result_paddle, result_torch, msg="Add same shape")
        else:
            result_np = x_np + y_np
            self.assert_allclose(result_paddle, result_np, msg="Add same shape")

    def test_add_broadcast(self):
        """Test elementwise add with broadcasting."""
        x_np = np.random.randn(3, 4, 5).astype(np.float32)
        y_np = np.random.randn(4, 5).astype(np.float32)  # Broadcast
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.add(x_paddle, y_paddle)
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            result_torch = torch.add(x_torch, y_torch)
            self.assert_allclose(result_paddle, result_torch, msg="Add broadcast")
        else:
            result_np = x_np + y_np
            self.assert_allclose(result_paddle, result_np, msg="Add broadcast")

    def test_add_scalar(self):
        """Test elementwise add with scalar."""
        shape = [2, 3]
        x_np = np.random.randn(*shape).astype(np.float32)
        scalar = 5.0

        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(scalar, place=self.device)
        result_paddle = paddle.add(x_paddle, y_paddle)

        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            result_torch = torch.add(x_torch, scalar)
            self.assert_allclose(result_paddle, result_torch, msg="Add scalar")
        else:
            result_np = x_np + scalar
            self.assert_allclose(result_paddle, result_np, msg="Add scalar")

    def test_multiply_same_shape(self):
        """Test elementwise multiply with same shapes."""
        shape = [3, 4, 5]
        x_np = np.random.randn(*shape).astype(np.float32)
        y_np = np.random.randn(*shape).astype(np.float32)
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.multiply(x_paddle, y_paddle)
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            result_torch = torch.mul(x_torch, y_torch)
            self.assert_allclose(result_paddle, result_torch, msg="Multiply same shape")
        else:
            result_np = x_np * y_np
            self.assert_allclose(result_paddle, result_np, msg="Multiply same shape")

    def test_multiply_broadcast(self):
        """Test elementwise multiply with broadcasting."""
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        y_np = np.random.randn(3, 1).astype(np.float32)  # Broadcast
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.multiply(x_paddle, y_paddle)
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            result_torch = torch.mul(x_torch, y_torch)
            self.assert_allclose(result_paddle, result_torch, msg="Multiply broadcast")
        else:
            result_np = x_np * y_np
            self.assert_allclose(result_paddle, result_np, msg="Multiply broadcast")

    def test_subtract_same_shape(self):
        """Test elementwise subtract with same shapes."""
        shape = [3, 4]
        x_np = np.random.randn(*shape).astype(np.float32)
        y_np = np.random.randn(*shape).astype(np.float32)
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.subtract(x_paddle, y_paddle)
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            result_torch = torch.sub(x_torch, y_torch)
            self.assert_allclose(result_paddle, result_torch, msg="Subtract same shape")
        else:
            result_np = x_np - y_np
            self.assert_allclose(result_paddle, result_np, msg="Subtract same shape")

    def test_subtract_broadcast(self):
        """Test elementwise subtract with broadcasting."""
        x_np = np.random.randn(5, 3, 4).astype(np.float32)
        y_np = np.random.randn(1, 3, 4).astype(np.float32)  # Broadcast
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.subtract(x_paddle, y_paddle)
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            result_torch = torch.sub(x_torch, y_torch)
            self.assert_allclose(result_paddle, result_torch, msg="Subtract broadcast")
        else:
            result_np = x_np - y_np
            self.assert_allclose(result_paddle, result_np, msg="Subtract broadcast")

    def test_divide_same_shape(self):
        """Test elementwise divide with same shapes."""
        shape = [2, 3, 4]
        x_np = np.random.randn(*shape).astype(np.float32) + 1.0  # Avoid division by zero
        y_np = np.random.randn(*shape).astype(np.float32) + 1.0
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.divide(x_paddle, y_paddle)
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            result_torch = torch.div(x_torch, y_torch)
            self.assert_allclose(result_paddle, result_torch, msg="Divide same shape")
        else:
            result_np = x_np / y_np
            self.assert_allclose(result_paddle, result_np, msg="Divide same shape")

    def test_divide_broadcast(self):
        """Test elementwise divide with broadcasting."""
        x_np = np.random.randn(3, 4, 5).astype(np.float32) + 1.0
        y_np = np.random.randn(4, 1).astype(np.float32) + 1.0  # Broadcast
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.divide(x_paddle, y_paddle)
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            result_torch = torch.div(x_torch, y_torch)
            self.assert_allclose(result_paddle, result_torch, msg="Divide broadcast")
        else:
            result_np = x_np / y_np
            self.assert_allclose(result_paddle, result_np, msg="Divide broadcast")

    def test_add_inplace(self):
        """Test inplace add operation."""
        shape = [2, 3]
        x_np = np.random.randn(*shape).astype(np.float32)
        y_np = np.random.randn(*shape).astype(np.float32)
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        x_paddle.add_(y_paddle)  # Inplace
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            x_torch.add_(y_torch)
            self.assert_allclose(x_paddle, x_torch, msg="Add inplace")
        else:
            expected = x_np + y_np
            self.assert_allclose(x_paddle, expected, msg="Add inplace")

    def test_large_tensor(self):
        """Test operations on large tensors."""
        shape = [100, 200]
        x_np = np.random.randn(*shape).astype(np.float32)
        y_np = np.random.randn(*shape).astype(np.float32)
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.add(x_paddle, y_paddle)
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            result_torch = torch.add(x_torch, y_torch)
            self.assert_allclose(result_paddle, result_torch, msg="Large tensor add")
        else:
            result_np = x_np + y_np
            self.assert_allclose(result_paddle, result_np, msg="Large tensor add")

    def test_edge_case_empty_tensor(self):
        """Test edge case with empty tensors."""
        x_np = np.array([]).astype(np.float32)
        y_np = np.array([]).astype(np.float32)
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.add(x_paddle, y_paddle)
        
        # Should not crash and return empty tensor
        self.assertEqual(result_paddle.shape, [0])

    def test_edge_case_single_element(self):
        """Test edge case with single element tensors."""
        x_np = np.array([5.0]).astype(np.float32)
        y_np = np.array([3.0]).astype(np.float32)
        
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        result_paddle = paddle.add(x_paddle, y_paddle)
        
        # Reference
        expected = np.array([8.0])
        self.assert_allclose(result_paddle, expected, msg="Single element add")

    def test_chained_operations(self):
        """Test chained elementwise operations."""
        shape = [2, 3]
        x_np = np.random.randn(*shape).astype(np.float32)
        y_np = np.random.randn(*shape).astype(np.float32)
        z_np = np.random.randn(*shape).astype(np.float32)
        
        # Paddle: (x + y) * z
        x_paddle = paddle.to_tensor(x_np, place=self.device)
        y_paddle = paddle.to_tensor(y_np, place=self.device)
        z_paddle = paddle.to_tensor(z_np, place=self.device)
        result_paddle = paddle.multiply(paddle.add(x_paddle, y_paddle), z_paddle)
        
        # Reference
        if self.use_torch_ref:
            x_torch = torch.tensor(x_np, device=self.torch_device)
            y_torch = torch.tensor(y_np, device=self.torch_device)
            z_torch = torch.tensor(z_np, device=self.torch_device)
            result_torch = torch.mul(torch.add(x_torch, y_torch), z_torch)
            self.assert_allclose(result_paddle, result_torch, msg="Chained operations")
        else:
            result_np = (x_np + y_np) * z_np
            self.assert_allclose(result_paddle, result_np, msg="Chained operations")


# ---------------------------------------------------------------------------
# Shared helpers for the activation / reduction tests below.
# ---------------------------------------------------------------------------


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
# Activation kernels: softmax, gelu, leaky_relu.
# ---------------------------------------------------------------------------


def _softmax_numpy(x, axis):
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x_shift)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _gelu_exact_numpy(x):
    from math import erf

    erf_vec = np.vectorize(erf, otypes=[np.float64])
    return (0.5 * x * (1.0 + erf_vec(x / math.sqrt(2.0)))).astype(x.dtype)


def _gelu_approx_numpy(x):
    beta = math.sqrt(2.0 / math.pi)
    inner = beta * (x + 0.044715 * np.power(x, 3))
    return (0.5 * x * (1.0 + np.tanh(inner))).astype(x.dtype)


def _leaky_relu_numpy(x, alpha):
    return np.where(x >= 0, x, alpha * x).astype(x.dtype)


class TestMPSActivationKernels(_MPSKernelTestBase):
    """MPS coverage for softmax, gelu, leaky_relu."""

    # ---- softmax --------------------------------------------------------
    def _softmax_check(self, x_np, axis, rtol=1e-4, atol=1e-5):
        import paddle.nn.functional as F
        out_mps = F.softmax(paddle.to_tensor(x_np, place="mps"), axis=axis).numpy()
        out_cpu = F.softmax(paddle.to_tensor(x_np, place="cpu"), axis=axis).numpy()
        ref = _softmax_numpy(x_np, axis)
        np.testing.assert_allclose(out_mps, ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol)

    def test_softmax_shapes_and_axes(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            x = np.random.randn(*shape).astype(np.float32)
            for axis in (-1, 0):
                with self.subTest(shape=shape, axis=axis):
                    self._softmax_check(x, axis)

    def test_softmax_middle_and_negative_axes(self):
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        for axis in [1, 2, -1, -2, -3, -4]:
            with self.subTest(axis=axis):
                self._softmax_check(x, axis)

    def test_softmax_sums_to_one(self):
        import paddle.nn.functional as F
        x = np.random.randn(5, 8).astype(np.float32)
        out = F.softmax(paddle.to_tensor(x, place="mps"), axis=-1).numpy()
        np.testing.assert_allclose(
            out.sum(axis=-1), np.ones(out.shape[0], dtype=np.float32),
            rtol=1e-4, atol=1e-5,
        )

    def test_softmax_numerical_stability(self):
        # Large magnitudes must not overflow; softmax subtracts the row-max.
        import paddle.nn.functional as F
        x = np.array(
            [[1000.0, 1000.0, 1000.0], [-1000.0, -1000.0, 0.0]],
            dtype=np.float32,
        )
        out = F.softmax(paddle.to_tensor(x, place="mps"), axis=-1).numpy()
        self.assertFalse(np.any(np.isnan(out)))
        self.assertFalse(np.any(np.isinf(out)))
        np.testing.assert_allclose(out.sum(axis=-1), [1.0, 1.0], rtol=1e-5, atol=1e-6)

    def test_softmax_dtype_and_place(self):
        import paddle.nn.functional as F
        x = np.random.randn(3, 4).astype(np.float32)
        out = F.softmax(paddle.to_tensor(x, place="mps"), axis=-1)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())

    # ---- gelu -----------------------------------------------------------
    def _gelu_check(self, x_np, approximate, rtol=1e-4, atol=1e-5):
        import paddle.nn.functional as F
        ref = _gelu_approx_numpy(x_np) if approximate else _gelu_exact_numpy(x_np)
        out_mps = F.gelu(
            paddle.to_tensor(x_np, place="mps"), approximate=approximate
        ).numpy()
        out_cpu = F.gelu(
            paddle.to_tensor(x_np, place="cpu"), approximate=approximate
        ).numpy()
        np.testing.assert_allclose(out_mps, ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol)

    def test_gelu_shapes_both_branches(self):
        for shape in [(8,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            x = np.random.randn(*shape).astype(np.float32) * 2.0
            for approximate in (False, True):
                with self.subTest(shape=shape, approximate=approximate):
                    self._gelu_check(x, approximate)

    def test_gelu_zero_in_zero_out(self):
        import paddle.nn.functional as F
        x = np.zeros((4, 5), dtype=np.float32)
        for approximate in (False, True):
            with self.subTest(approximate=approximate):
                out = F.gelu(
                    paddle.to_tensor(x, place="mps"), approximate=approximate
                ).numpy()
                np.testing.assert_allclose(out, np.zeros_like(x), atol=1e-6)

    def test_gelu_saturates_at_extremes(self):
        import paddle.nn.functional as F
        x = np.array([10.0, -10.0], dtype=np.float32)
        for approximate in (False, True):
            with self.subTest(approximate=approximate):
                out = F.gelu(
                    paddle.to_tensor(x, place="mps"), approximate=approximate
                ).numpy()
                np.testing.assert_allclose(out, [10.0, 0.0], rtol=1e-4, atol=1e-3)

    def test_gelu_dtype_and_place(self):
        import paddle.nn.functional as F
        x = np.random.randn(3, 4).astype(np.float32)
        out = F.gelu(paddle.to_tensor(x, place="mps"))
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())

    # ---- leaky_relu -----------------------------------------------------
    def _leaky_relu_check(self, x_np, alpha, rtol=1e-5, atol=1e-6):
        import paddle.nn.functional as F
        ref = _leaky_relu_numpy(x_np, alpha)
        out_mps = F.leaky_relu(
            paddle.to_tensor(x_np, place="mps"), negative_slope=alpha
        ).numpy()
        out_cpu = F.leaky_relu(
            paddle.to_tensor(x_np, place="cpu"), negative_slope=alpha
        ).numpy()
        np.testing.assert_allclose(out_mps, ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol)

    def test_leaky_relu_shapes_default_slope(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            with self.subTest(shape=shape):
                self._leaky_relu_check(np.random.randn(*shape).astype(np.float32), 0.01)

    def test_leaky_relu_various_alphas(self):
        x = np.random.randn(5, 6).astype(np.float32)
        for alpha in [0.0, 0.01, 0.1, 0.2, 0.5, 1.0]:
            with self.subTest(alpha=alpha):
                self._leaky_relu_check(x, alpha)

    def test_leaky_relu_alpha_zero_is_relu(self):
        import paddle.nn.functional as F
        x = np.random.randn(4, 5).astype(np.float32)
        out = F.leaky_relu(
            paddle.to_tensor(x, place="mps"), negative_slope=0.0
        ).numpy()
        np.testing.assert_allclose(out, np.maximum(x, 0.0), rtol=1e-6, atol=1e-7)

    def test_leaky_relu_known_values(self):
        import paddle.nn.functional as F
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        out = F.leaky_relu(
            paddle.to_tensor(x, place="mps"), negative_slope=0.1
        ).numpy()
        np.testing.assert_allclose(
            out, np.array([-0.2, -0.1, 0.0, 1.0, 2.0], dtype=np.float32),
            rtol=1e-5, atol=1e-6,
        )

    def test_leaky_relu_dtype_and_place(self):
        import paddle.nn.functional as F
        x = np.random.randn(3, 4).astype(np.float32)
        out = F.leaky_relu(paddle.to_tensor(x, place="mps"))
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


# ---------------------------------------------------------------------------
# Reduction kernels: mean, sum, max, min.
# ---------------------------------------------------------------------------


class TestMPSReductionKernels(_MPSKernelTestBase):
    """MPS coverage for paddle.mean / sum / max / min."""

    # Each op tuple: (paddle fn, numpy fn, default tolerances)
    _OPS = (
        ("mean", lambda x, **kw: paddle.mean(x, **kw), np.mean, (1e-4, 1e-5)),
        ("sum",  lambda x, **kw: paddle.sum(x,  **kw), np.sum,  (1e-4, 1e-4)),
        ("max",  lambda x, **kw: paddle.max(x,  **kw), np.max,  (1e-5, 1e-6)),
        ("min",  lambda x, **kw: paddle.min(x,  **kw), np.min,  (1e-5, 1e-6)),
    )

    @staticmethod
    def _np_axis(axis):
        return tuple(axis) if isinstance(axis, list) else axis

    def _reduce_check(self, name, paddle_op, numpy_op, x_np, axis, keepdim, tols):
        kwargs = dict(axis=axis, keepdim=keepdim)
        out_mps = paddle_op(paddle.to_tensor(x_np, place="mps"), **kwargs).numpy()
        out_cpu = paddle_op(paddle.to_tensor(x_np, place="cpu"), **kwargs).numpy()
        ref = numpy_op(x_np, axis=self._np_axis(axis), keepdims=keepdim).astype(np.float32)
        rtol, atol = tols
        np.testing.assert_allclose(out_mps, ref, rtol=rtol, atol=atol,
                                   err_msg=f"{name} vs numpy (axis={axis}, keepdim={keepdim})")
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol,
                                   err_msg=f"{name} vs cpu (axis={axis}, keepdim={keepdim})")
        self.assertEqual(out_mps.shape, ref.shape,
                         f"{name} shape mismatch (axis={axis}, keepdim={keepdim})")

    def test_reduce_all_default(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            x = np.random.randn(*shape).astype(np.float32)
            for name, p_op, n_op, tols in self._OPS:
                for keepdim in (False, True):
                    with self.subTest(op=name, shape=shape, keepdim=keepdim):
                        self._reduce_check(name, p_op, n_op, x,
                                           axis=None, keepdim=keepdim, tols=tols)

    def test_single_axis(self):
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        for name, p_op, n_op, tols in self._OPS:
            for axis in [0, 1, 2, 3, -1, -2]:
                for keepdim in (False, True):
                    with self.subTest(op=name, axis=axis, keepdim=keepdim):
                        self._reduce_check(name, p_op, n_op, x,
                                           axis=axis, keepdim=keepdim, tols=tols)

    def test_multiple_axes(self):
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        axes_sets = [[0, 1], [2, 3], [0, 3], [1, 2, 3]]
        for name, p_op, n_op, tols in self._OPS:
            for axes in axes_sets:
                for keepdim in (False, True):
                    with self.subTest(op=name, axes=axes, keepdim=keepdim):
                        self._reduce_check(name, p_op, n_op, x,
                                           axis=axes, keepdim=keepdim, tols=tols)

    def test_known_values(self):
        x = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
        # max along axis 1 -> [3.0, 5.0]; min -> [-2.0, -6.0]
        out = paddle.max(paddle.to_tensor(x, place="mps"), axis=1).numpy()
        np.testing.assert_allclose(out, [3.0, 5.0])
        out = paddle.min(paddle.to_tensor(x, place="mps"), axis=1).numpy()
        np.testing.assert_allclose(out, [-2.0, -6.0])
        # sum-of-all -> -3.0; mean -> -0.5
        out = paddle.sum(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.float32(-3.0), atol=1e-5)
        out = paddle.mean(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.float32(-0.5), atol=1e-6)

    def test_constant_input(self):
        x = np.full((3, 4, 5), 2.5, dtype=np.float32)
        for name, p_op, _, _ in self._OPS:
            with self.subTest(op=name):
                out = p_op(paddle.to_tensor(x, place="mps")).numpy()
                expected = 2.5 if name in ("mean", "max", "min") else 2.5 * x.size
                np.testing.assert_allclose(out, np.float32(expected), rtol=1e-5, atol=1e-5)

    def test_dtype_and_place_preserved(self):
        x = np.random.randn(3, 4).astype(np.float32)
        for name, p_op, _, _ in self._OPS:
            with self.subTest(op=name):
                out = p_op(paddle.to_tensor(x, place="mps"), axis=-1)
                self.assertEqual(out.dtype, paddle.float32)
                self.assertTrue("mps" in str(out.place).lower())


# ---------------------------------------------------------------------------
# Logical kernels: logical_and, logical_or, logical_not, logical_xor.
# Inputs and outputs are all bool tensors.
# ---------------------------------------------------------------------------


class TestMPSLogicalKernels(_MPSKernelTestBase):
    """MPS coverage for paddle.logical_and / or / not / xor."""

    def _rand_bool(self, shape):
        return np.random.randint(0, 2, size=shape).astype(np.bool_)

    def _binary_check(self, name, paddle_op, numpy_op, x_np, y_np):
        out_mps = paddle_op(
            paddle.to_tensor(x_np, place="mps"),
            paddle.to_tensor(y_np, place="mps"),
        ).numpy()
        out_cpu = paddle_op(
            paddle.to_tensor(x_np, place="cpu"),
            paddle.to_tensor(y_np, place="cpu"),
        ).numpy()
        ref = numpy_op(x_np, y_np)
        np.testing.assert_array_equal(out_mps, ref,
                                      err_msg=f"{name} vs numpy")
        np.testing.assert_array_equal(out_mps, out_cpu,
                                      err_msg=f"{name} vs cpu")
        self.assertEqual(out_mps.dtype, np.bool_,
                         f"{name} output dtype must be bool")

    def _unary_check(self, x_np):
        out_mps = paddle.logical_not(paddle.to_tensor(x_np, place="mps")).numpy()
        out_cpu = paddle.logical_not(paddle.to_tensor(x_np, place="cpu")).numpy()
        ref = np.logical_not(x_np)
        np.testing.assert_array_equal(out_mps, ref, err_msg="logical_not vs numpy")
        np.testing.assert_array_equal(out_mps, out_cpu, err_msg="logical_not vs cpu")
        self.assertEqual(out_mps.dtype, np.bool_)

    _BINARY_OPS = (
        ("logical_and", lambda a, b: paddle.logical_and(a, b), np.logical_and),
        ("logical_or",  lambda a, b: paddle.logical_or(a, b),  np.logical_or),
        ("logical_xor", lambda a, b: paddle.logical_xor(a, b), np.logical_xor),
    )

    def test_binary_truth_tables(self):
        # All 4 combinations of two bool inputs.
        x = np.array([False, False, True, True], dtype=np.bool_)
        y = np.array([False, True, False, True], dtype=np.bool_)
        for name, p_op, n_op in self._BINARY_OPS:
            with self.subTest(op=name):
                self._binary_check(name, p_op, n_op, x, y)

    def test_binary_shapes(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            x = self._rand_bool(shape)
            y = self._rand_bool(shape)
            for name, p_op, n_op in self._BINARY_OPS:
                with self.subTest(op=name, shape=shape):
                    self._binary_check(name, p_op, n_op, x, y)

    def test_not_truth_table(self):
        self._unary_check(np.array([False, True], dtype=np.bool_))

    def test_not_shapes(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            with self.subTest(shape=shape):
                self._unary_check(self._rand_bool(shape))

    def test_de_morgan_identity(self):
        # not(a or b) == (not a) and (not b)
        a = self._rand_bool((4, 5))
        b = self._rand_bool((4, 5))
        a_p = paddle.to_tensor(a, place="mps")
        b_p = paddle.to_tensor(b, place="mps")
        lhs = paddle.logical_not(paddle.logical_or(a_p, b_p)).numpy()
        rhs = paddle.logical_and(
            paddle.logical_not(a_p), paddle.logical_not(b_p)
        ).numpy()
        np.testing.assert_array_equal(lhs, rhs)

    def test_self_xor_is_false(self):
        a = self._rand_bool((3, 4, 5))
        a_p = paddle.to_tensor(a, place="mps")
        out = paddle.logical_xor(a_p, a_p).numpy()
        np.testing.assert_array_equal(out, np.zeros_like(a, dtype=np.bool_))

    def test_output_dtype_and_place(self):
        a = self._rand_bool((3, 4))
        b = self._rand_bool((3, 4))
        for name, p_op, _ in self._BINARY_OPS:
            with self.subTest(op=name):
                out = p_op(paddle.to_tensor(a, place="mps"),
                           paddle.to_tensor(b, place="mps"))
                self.assertEqual(out.dtype, paddle.bool)
                self.assertTrue("mps" in str(out.place).lower())
        out = paddle.logical_not(paddle.to_tensor(a, place="mps"))
        self.assertEqual(out.dtype, paddle.bool)
        self.assertTrue("mps" in str(out.place).lower())


# ---------------------------------------------------------------------------
# Comparison kernels: equal, not_equal, less_than, less_equal, greater_than,
# greater_equal. Float32 inputs, bool output.
# ---------------------------------------------------------------------------


class TestMPSComparisonKernels(_MPSKernelTestBase):
    """MPS coverage for paddle.equal / not_equal / less_than / less_equal /
    greater_than / greater_equal."""

    _OPS = (
        ("equal",         lambda a, b: paddle.equal(a, b),         np.equal),
        ("not_equal",     lambda a, b: paddle.not_equal(a, b),     np.not_equal),
        ("less_than",     lambda a, b: paddle.less_than(a, b),     np.less),
        ("less_equal",    lambda a, b: paddle.less_equal(a, b),    np.less_equal),
        ("greater_than",  lambda a, b: paddle.greater_than(a, b),  np.greater),
        ("greater_equal", lambda a, b: paddle.greater_equal(a, b), np.greater_equal),
    )

    def _check(self, name, paddle_op, numpy_op, x_np, y_np):
        out_mps = paddle_op(
            paddle.to_tensor(x_np, place="mps"),
            paddle.to_tensor(y_np, place="mps"),
        ).numpy()
        out_cpu = paddle_op(
            paddle.to_tensor(x_np, place="cpu"),
            paddle.to_tensor(y_np, place="cpu"),
        ).numpy()
        ref = numpy_op(x_np, y_np)
        np.testing.assert_array_equal(out_mps, ref, err_msg=f"{name} vs numpy")
        np.testing.assert_array_equal(out_mps, out_cpu, err_msg=f"{name} vs cpu")
        self.assertEqual(out_mps.dtype, np.bool_,
                         f"{name} output dtype must be bool")

    def test_shapes(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            x = np.random.randn(*shape).astype(np.float32)
            y = np.random.randn(*shape).astype(np.float32)
            for name, p_op, n_op in self._OPS:
                with self.subTest(op=name, shape=shape):
                    self._check(name, p_op, n_op, x, y)

    def test_overlapping_values(self):
        # Force a mix of <, ==, > by sharing some entries between x and y.
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        y = np.array([2.0, 2.0, 2.0, 4.0, 4.0], dtype=np.float32)
        for name, p_op, n_op in self._OPS:
            with self.subTest(op=name):
                self._check(name, p_op, n_op, x, y)

    def test_known_truth_table(self):
        # Hand-rolled expected values, exercising each comparison's edge.
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        expected = {
            "equal":         [False, True, False],
            "not_equal":     [True, False, True],
            "less_than":     [True, False, False],
            "less_equal":    [True, True, False],
            "greater_than":  [False, False, True],
            "greater_equal": [False, True, True],
        }
        for name, p_op, _ in self._OPS:
            with self.subTest(op=name):
                out = p_op(
                    paddle.to_tensor(x, place="mps"),
                    paddle.to_tensor(y, place="mps"),
                ).numpy()
                np.testing.assert_array_equal(
                    out, np.array(expected[name], dtype=np.bool_)
                )

    def test_equal_reflexive(self):
        # equal(x, x) is all-True; not_equal(x, x) is all-False.
        x = np.random.randn(4, 5).astype(np.float32)
        x_p = paddle.to_tensor(x, place="mps")
        np.testing.assert_array_equal(
            paddle.equal(x_p, x_p).numpy(), np.ones_like(x, dtype=np.bool_)
        )
        np.testing.assert_array_equal(
            paddle.not_equal(x_p, x_p).numpy(), np.zeros_like(x, dtype=np.bool_)
        )

    def test_complementary_pairs(self):
        # equal/not_equal, less_than/greater_equal, greater_than/less_equal are
        # logical complements of each other.
        x = np.random.randn(4, 5).astype(np.float32)
        y = np.random.randn(4, 5).astype(np.float32)
        x_p = paddle.to_tensor(x, place="mps")
        y_p = paddle.to_tensor(y, place="mps")
        np.testing.assert_array_equal(
            paddle.equal(x_p, y_p).numpy(),
            np.logical_not(paddle.not_equal(x_p, y_p).numpy()),
        )
        np.testing.assert_array_equal(
            paddle.less_than(x_p, y_p).numpy(),
            np.logical_not(paddle.greater_equal(x_p, y_p).numpy()),
        )
        np.testing.assert_array_equal(
            paddle.greater_than(x_p, y_p).numpy(),
            np.logical_not(paddle.less_equal(x_p, y_p).numpy()),
        )

    def test_output_dtype_and_place(self):
        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(3, 4).astype(np.float32)
        for name, p_op, _ in self._OPS:
            with self.subTest(op=name):
                out = p_op(
                    paddle.to_tensor(x, place="mps"),
                    paddle.to_tensor(y, place="mps"),
                )
                self.assertEqual(out.dtype, paddle.bool)
                self.assertTrue("mps" in str(out.place).lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)

