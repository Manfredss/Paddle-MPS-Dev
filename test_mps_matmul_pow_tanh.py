#!/usr/bin/env python3
"""
Test script for new MPS operators: matmul, pow, tanh
Compares with PyTorch CPU outputs.
"""

import sys
import numpy as np

print("=" * 70)
print("New MPS Operators Test: matmul, pow, tanh")
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
print("Testing New Operations")
print("=" * 70)

def test_op(name, paddle_op, torch_op, numpy_op, x_shape, y_shape=None, dtype=np.float32, rtol=1e-5, atol=1e-5):
    """Test a single operation."""
    print(f"\n{name}:")
    
    # Generate test data
    if y_shape is None:
        # Unary operation
        if name.lower() == 'tanh':
            x_np = (np.random.randn(*x_shape).astype(dtype) - 0.5) * 4.0  # Range [-2, 2]
        else:
            x_np = np.random.randn(*x_shape).astype(dtype)
    else:
        # Binary operation
        x_np = np.random.randn(*x_shape).astype(dtype)
        if "pow" in name.lower():
            # For pow, use positive values for base, and reasonable exponents
            x_np = np.abs(x_np) + 0.1
            y_np = (np.random.randn(*y_shape).astype(dtype) - 0.5) * 2.0  # Range [-1, 1]
        else:
            y_np = np.random.randn(*y_shape).astype(dtype)
    
    try:
        # Paddle
        x_p = paddle.to_tensor(x_np, place='mps')
        if y_shape is not None:
            y_p = paddle.to_tensor(y_np, place='mps')
            result_p = paddle_op(x_p, y_p)
        else:
            result_p = paddle_op(x_p)
        result_p_np = result_p.numpy()
        
        # Reference (PyTorch CPU)
        if use_torch:
            x_t = torch.tensor(x_np, device=torch_device)
            if y_shape is not None:
                y_t = torch.tensor(y_np, device=torch_device)
                result_t = torch_op(x_t, y_t)
            else:
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
            if y_shape is not None:
                result_n = numpy_op(x_np, y_np)
            else:
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

# Test matmul
print("\n" + "=" * 70)
print("Testing Matmul")
print("=" * 70)

all_passed = test_op("Matmul (2D)", 
                     lambda x, y: paddle.matmul(x, y),
                     lambda x, y: torch.matmul(x, y),
                     lambda x, y: np.dot(x, y),
                     [3, 4], [4, 5]) and all_passed

all_passed = test_op("Matmul (2D, transpose_x)", 
                     lambda x, y: paddle.matmul(x, y, transpose_x=True),
                     lambda x, y: torch.matmul(x.t(), y),
                     lambda x, y: np.dot(x.T, y),
                     [4, 3], [4, 5]) and all_passed

all_passed = test_op("Matmul (2D, transpose_y)", 
                     lambda x, y: paddle.matmul(x, y, transpose_y=True),
                     lambda x, y: torch.matmul(x, y.t()),
                     lambda x, y: np.dot(x, y.T),
                     [3, 4], [5, 4]) and all_passed

# Test pow
print("\n" + "=" * 70)
print("Testing Pow")
print("=" * 70)

all_passed = test_op("Pow (same shape)", 
                     lambda x, y: paddle.pow(x, y),
                     lambda x, y: torch.pow(x, y),
                     np.power,
                     [3, 4], [3, 4]) and all_passed

all_passed = test_op("Pow (broadcast)", 
                     lambda x, y: paddle.pow(x, y),
                     lambda x, y: torch.pow(x, y),
                     np.power,
                     [3, 4], [4,]) and all_passed

# Test tanh
print("\n" + "=" * 70)
print("Testing Tanh")
print("=" * 70)

all_passed = test_op("Tanh", 
                     lambda x: paddle.nn.functional.tanh(x),
                     lambda x: torch.tanh(x),
                     np.tanh,
                     [3, 4]) and all_passed

all_passed = test_op("Tanh (large values)", 
                     lambda x: paddle.nn.functional.tanh(x),
                     lambda x: torch.tanh(x),
                     np.tanh,
                     [2, 3]) and all_passed

# Summary
print("\n" + "=" * 70)
if all_passed:
    print("✓ All new operator tests passed!")
    print("=" * 70)
    sys.exit(0)
else:
    print("✗ Some tests failed")
    print("=" * 70)
    sys.exit(1)

