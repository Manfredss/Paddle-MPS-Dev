#!/usr/bin/env python3
"""
Comprehensive test suite for MPS kernels with detailed PyTorch comparison.
This test file provides more detailed output and edge case testing.
"""

import sys
import traceback
import numpy as np

try:
    import paddle
    PADDLE_AVAILABLE = True
except ImportError as e:
    PADDLE_AVAILABLE = False
    print(f"Error: Failed to import PaddlePaddle: {e}")
    sys.exit(1)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, will use numpy reference only")


def test_setup():
    """Check if MPS is available."""
    print("=" * 70)
    print("MPS Kernel Comprehensive Test Suite")
    print("=" * 70)
    
    if not paddle.is_compiled_with_mps():
        print("✗ PaddlePaddle not compiled with MPS support")
        return False
    
    if not paddle.mps.is_available():
        print("✗ MPS not available on this system")
        return False
    
    paddle.mps.set_device(0)
    print(f"✓ MPS available, using device 0")
    
    if TORCH_AVAILABLE and torch.backends.mps.is_available():
        print(f"✓ PyTorch MPS available for reference comparison")
        return True, torch.device('mps')
    else:
        print("⚠ PyTorch MPS not available, using numpy reference only")
        return True, None


def compare_results(paddle_result, torch_result, numpy_result, op_name, rtol=1e-5, atol=1e-6):
    """Compare results from Paddle, PyTorch, and NumPy."""
    paddle_np = paddle_result.numpy() if hasattr(paddle_result, 'numpy') else np.array(paddle_result)
    
    errors = []
    
    # Compare with PyTorch if available
    if torch_result is not None:
        torch_np = torch_result.cpu().numpy()
        try:
            np.testing.assert_allclose(paddle_np, torch_np, rtol=rtol, atol=atol)
            print(f"  ✓ Paddle matches PyTorch")
        except AssertionError as e:
            errors.append(f"Paddle-PyTorch mismatch: {e}")
            print(f"  ✗ Paddle does not match PyTorch")
            max_diff = np.max(np.abs(paddle_np - torch_np))
            print(f"    Max difference: {max_diff}")
    
    # Compare with NumPy
    try:
        np.testing.assert_allclose(paddle_np, numpy_result, rtol=rtol, atol=atol)
        print(f"  ✓ Paddle matches NumPy")
    except AssertionError as e:
        errors.append(f"Paddle-NumPy mismatch: {e}")
        print(f"  ✗ Paddle does not match NumPy")
        max_diff = np.max(np.abs(paddle_np - numpy_result))
        print(f"    Max difference: {max_diff}")
    
    return len(errors) == 0, errors


def test_operation(op_name, paddle_op, torch_op, numpy_op, x_np, y_np=None, 
                   device='mps', torch_device=None):
    """Test a single operation."""
    print(f"\nTesting {op_name}...")
    print(f"  Input shapes: x={x_np.shape}" + (f", y={y_np.shape}" if y_np is not None else ""))
    
    try:
        # Paddle
        x_paddle = paddle.to_tensor(x_np, place=device)
        if y_np is not None:
            y_paddle = paddle.to_tensor(y_np, place=device)
            result_paddle = paddle_op(x_paddle, y_paddle)
        else:
            result_paddle = paddle_op(x_paddle)
        
        # PyTorch reference
        result_torch = None
        if torch_device is not None and TORCH_AVAILABLE:
            x_torch = torch.tensor(x_np, device=torch_device)
            if y_np is not None:
                y_torch = torch.tensor(y_np, device=torch_device)
                result_torch = torch_op(x_torch, y_torch)
            else:
                result_torch = torch_op(x_torch)
        
        # NumPy reference
        if y_np is not None:
            result_numpy = numpy_op(x_np, y_np)
        else:
            result_numpy = numpy_op(x_np)
        
        # Compare
        success, errors = compare_results(result_paddle, result_torch, result_numpy, op_name)
        
        if success:
            print(f"  ✓ {op_name} test passed")
            return True
        else:
            print(f"  ✗ {op_name} test failed")
            for error in errors:
                print(f"    {error}")
            return False
            
    except Exception as e:
        print(f"  ✗ {op_name} test failed with exception: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    setup_result = test_setup()
    if not setup_result:
        return False
    
    if isinstance(setup_result, tuple):
        success, torch_device = setup_result
        if not success:
            return False
    else:
        torch_device = None
    
    np.random.seed(42)
    paddle.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
    
    all_passed = True
    
    # Test cases
    test_cases = [
        # (name, paddle_op, torch_op, numpy_op, x_shape, y_shape)
        ("Add (same shape)", 
         lambda x, y: paddle.add(x, y),
         lambda x, y: torch.add(x, y),
         lambda x, y: x + y,
         [3, 4, 5], [3, 4, 5]),
        
        ("Add (broadcast)", 
         lambda x, y: paddle.add(x, y),
         lambda x, y: torch.add(x, y),
         lambda x, y: x + y,
         [3, 4, 5], [4, 5]),
        
        ("Multiply (same shape)", 
         lambda x, y: paddle.multiply(x, y),
         lambda x, y: torch.mul(x, y),
         lambda x, y: x * y,
         [2, 3, 4], [2, 3, 4]),
        
        ("Multiply (broadcast)", 
         lambda x, y: paddle.multiply(x, y),
         lambda x, y: torch.mul(x, y),
         lambda x, y: x * y,
         [2, 3, 4], [3, 1]),
        
        ("Subtract (same shape)", 
         lambda x, y: paddle.subtract(x, y),
         lambda x, y: torch.sub(x, y),
         lambda x, y: x - y,
         [3, 4], [3, 4]),
        
        ("Subtract (broadcast)", 
         lambda x, y: paddle.subtract(x, y),
         lambda x, y: torch.sub(x, y),
         lambda x, y: x - y,
         [5, 3, 4], [1, 3, 4]),
        
        ("Divide (same shape)", 
         lambda x, y: paddle.divide(x, y),
         lambda x, y: torch.div(x, y),
         lambda x, y: x / y,
         [2, 3, 4], [2, 3, 4]),
        
        ("Divide (broadcast)", 
         lambda x, y: paddle.divide(x, y),
         lambda x, y: torch.div(x, y),
         lambda x, y: x / y,
         [3, 4, 5], [4, 1]),
    ]
    
    for name, paddle_op, torch_op, numpy_op, x_shape, y_shape in test_cases:
        x_np = np.random.randn(*x_shape).astype(np.float32)
        y_np = np.random.randn(*y_shape).astype(np.float32)
        
        # For divide, avoid zeros
        if "Divide" in name:
            x_np = x_np + 1.0
            y_np = y_np + 1.0
        
        passed = test_operation(name, paddle_op, torch_op, numpy_op, x_np, y_np, 
                               device='mps', torch_device=torch_device)
        all_passed = all_passed and passed
    
    # Edge cases
    print("\n" + "=" * 70)
    print("Testing Edge Cases")
    print("=" * 70)
    
    # Large tensor
    x_large = np.random.randn(100, 200).astype(np.float32)
    y_large = np.random.randn(100, 200).astype(np.float32)
    passed = test_operation("Add (large tensor)", 
                           lambda x, y: paddle.add(x, y),
                           lambda x, y: torch.add(x, y),
                           lambda x, y: x + y,
                           x_large, y_large, device='mps', torch_device=torch_device)
    all_passed = all_passed and passed
    
    # Single element
    x_single = np.array([5.0]).astype(np.float32)
    y_single = np.array([3.0]).astype(np.float32)
    passed = test_operation("Add (single element)", 
                           lambda x, y: paddle.add(x, y),
                           lambda x, y: torch.add(x, y),
                           lambda x, y: x + y,
                           x_single, y_single, device='mps', torch_device=torch_device)
    all_passed = all_passed and passed
    
    # Chained operations
    print("\nTesting Chained Operations...")
    x_chain = np.random.randn(2, 3).astype(np.float32)
    y_chain = np.random.randn(2, 3).astype(np.float32)
    z_chain = np.random.randn(2, 3).astype(np.float32)
    
    try:
        x_p = paddle.to_tensor(x_chain, place='mps')
        y_p = paddle.to_tensor(y_chain, place='mps')
        z_p = paddle.to_tensor(z_chain, place='mps')
        result_p = paddle.multiply(paddle.add(x_p, y_p), z_p)
        
        if torch_device is not None and TORCH_AVAILABLE:
            x_t = torch.tensor(x_chain, device=torch_device)
            y_t = torch.tensor(y_chain, device=torch_device)
            z_t = torch.tensor(z_chain, device=torch_device)
            result_t = torch.mul(torch.add(x_t, y_t), z_t)
            result_n = (x_chain + y_chain) * z_chain
            success, _ = compare_results(result_p, result_t, result_n, "Chained (x+y)*z")
            all_passed = all_passed and success
        else:
            result_n = (x_chain + y_chain) * z_chain
            success, _ = compare_results(result_p, None, result_n, "Chained (x+y)*z")
            all_passed = all_passed and success
    except Exception as e:
        print(f"  ✗ Chained operations failed: {e}")
        traceback.print_exc()
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 70)
    
    return all_passed


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

