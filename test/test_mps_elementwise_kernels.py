#!/usr/bin/env python3
"""
Test MPS elementwise kernels by comparing with PyTorch MPS outputs.
This ensures our MPS implementation matches PyTorch's behavior.
"""

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
        result_paddle = paddle.add(x_paddle, scalar)
        
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


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMPSElementwiseKernels)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)

