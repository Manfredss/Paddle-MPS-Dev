#!/usr/bin/env python3
"""
Test script for MPS unary operators - compares with PyTorch CPU outputs.
Tests: abs, exp, log, sqrt, relu, sigmoid
"""

import sys
import numpy as np

print("=" * 70)
print("MPS Unary Operators Test")
print("=" * 70)

# Import Paddle
try:
    import paddle
    print("✓ PaddlePaddle imported")
except ImportError as e:
    print(f"✗ Failed to import PaddlePaddle: {e}")
    sys.exit(1)

# Check MPS availability
if not paddle.is_compiled_with_mps():
    print("✗ PaddlePaddle not compiled with MPS")
    sys.exit(1)

if not paddle.mps.is_available():
    print("✗ MPS not available")
    sys.exit(1)

paddle.mps.set_device(0)
print("✓ MPS available, device set to 0")

# Import PyTorch for reference (use CPU)
try:
    import torch
    torch_device = torch.device('cpu')
    use_torch = True
    print("✓ PyTorch available for reference (using CPU)")
except ImportError:
    use_torch = False
    print("⚠ PyTorch not available, using numpy reference")

# Set seeds
np.random.seed(42)
paddle.seed(42)
if use_torch:
    torch.manual_seed(42)

print("\n" + "=" * 70)
print("Testing Unary Operations")
print("=" * 70)

def test_unary_op(name, paddle_op, torch_op, numpy_op, x_shape, dtype=np.float32, rtol=1e-5, atol=1e-5):
    """Test a single unary operation."""
    print(f"\n{name}:")
    
    # Generate test data
    if name.lower() in ['log', 'sqrt', 'log10', 'log2']:
        # For log/sqrt operations, use positive values
        x_np = np.random.rand(*x_shape).astype(dtype) + 0.1
    elif name.lower() == 'sigmoid':
        # For sigmoid, use values in reasonable range to avoid overflow
        x_np = (np.random.rand(*x_shape).astype(dtype) - 0.5) * 10.0
    else:
        x_np = (np.random.rand(*x_shape).astype(dtype) - 0.5) * 10.0
    
    try:
        # Paddle
        x_p = paddle.to_tensor(x_np, place='mps')
        result_p = paddle_op(x_p)
        result_p_np = result_p.numpy()
        
        # Reference (PyTorch CPU)
        if use_torch:
            x_t = torch.tensor(x_np, device=torch_device)
            result_t = torch_op(x_t)
            result_t_np = result_t.numpy()
            
            # Compare
            max_diff = np.max(np.abs(result_p_np - result_t_np))
            if np.allclose(result_p_np, result_t_np, rtol=rtol, atol=atol):
                print(f"  ✓ Pass (max diff: {max_diff:.2e})")
                return True
            else:
                print(f"  ✗ Fail (max diff: {max_diff:.2e})")
                print(f"    Sample values - Paddle: {result_p_np.flat[:5]}, PyTorch: {result_t_np.flat[:5]}")
                return False
        else:
            # NumPy reference
            result_n = numpy_op(x_np)
            max_diff = np.max(np.abs(result_p_np - result_n))
            if np.allclose(result_p_np, result_n, rtol=rtol, atol=atol):
                print(f"  ✓ Pass (max diff: {max_diff:.2e})")
                return True
            else:
                print(f"  ✗ Fail (max diff: {max_diff:.2e})")
                return False
                
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test operations
all_passed = True

# Test abs
all_passed = test_unary_op("Abs", 
                          lambda x: paddle.abs(x),
                          lambda x: torch.abs(x),
                          np.abs,
                          [3, 4]) and all_passed

# Test exp
all_passed = test_unary_op("Exp", 
                          lambda x: paddle.exp(x),
                          lambda x: torch.exp(x),
                          np.exp,
                          [2, 3]) and all_passed

# Test log
all_passed = test_unary_op("Log", 
                          lambda x: paddle.log(x),
                          lambda x: torch.log(x),
                          np.log,
                          [3, 4]) and all_passed

# Test sqrt
all_passed = test_unary_op("Sqrt", 
                          lambda x: paddle.sqrt(x),
                          lambda x: torch.sqrt(x),
                          np.sqrt,
                          [2, 3]) and all_passed

# Test relu
all_passed = test_unary_op("ReLU", 
                          lambda x: paddle.nn.functional.relu(x),
                          lambda x: torch.relu(x),
                          lambda x: np.maximum(x, 0),
                          [3, 4]) and all_passed

# Test sigmoid
all_passed = test_unary_op("Sigmoid", 
                          lambda x: paddle.nn.functional.sigmoid(x),
                          lambda x: torch.sigmoid(x),
                          lambda x: 1.0 / (1.0 + np.exp(-x)),
                          [2, 3]) and all_passed

# Summary
print("\n" + "=" * 70)
if all_passed:
    print("✓ All unary operator tests passed!")
    print("=" * 70)
    sys.exit(0)
else:
    print("✗ Some tests failed")
    print("=" * 70)
    sys.exit(1)

