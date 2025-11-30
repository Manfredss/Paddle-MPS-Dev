#!/usr/bin/env python3
"""
Quick test script for MPS kernels - compares with PyTorch outputs.
Run this to quickly verify MPS kernels are working correctly.
"""

import sys
import numpy as np

print("=" * 70)
print("Quick MPS Kernel Test")
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

# Import PyTorch for reference (use CPU for comparison as suggested)
try:
    import torch
    torch_device = torch.device('cpu')  # Use CPU for reference comparison
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
print("Testing Operations")
print("=" * 70)

def test_op(name, paddle_op, torch_op, numpy_op, x_shape, y_shape=None):
    """Test a single operation."""
    print(f"\n{name}:")
    
    x_np = np.random.randn(*x_shape).astype(np.float32)
    if y_shape:
        y_np = np.random.randn(*y_shape).astype(np.float32)
        if "divide" in name.lower() or "div" in name.lower():
            x_np = x_np + 1.0
            y_np = y_np + 1.0
    else:
        y_np = None
    
    try:
        # Paddle
        x_p = paddle.to_tensor(x_np, place='mps')
        if y_np is not None:
            y_p = paddle.to_tensor(y_np, place='mps')
            result_p = paddle_op(x_p, y_p)
        else:
            result_p = paddle_op(x_p)
        
        result_p_np = result_p.numpy()
        
        # Reference (PyTorch CPU results)
        if use_torch and y_np is not None:
            x_t = torch.tensor(x_np, device=torch_device)
            y_t = torch.tensor(y_np, device=torch_device)
            result_t = torch_op(x_t, y_t)
            result_t_np = result_t.numpy()  # Already on CPU
            
            # Compare
            max_diff = np.max(np.abs(result_p_np - result_t_np))
            if max_diff < 1e-5:
                print(f"  ✓ Pass (max diff: {max_diff:.2e})")
                return True
            else:
                print(f"  ✗ Fail (max diff: {max_diff:.2e})")
                return False
        else:
            # NumPy reference
            if y_np is not None:
                result_n = numpy_op(x_np, y_np)
            else:
                result_n = numpy_op(x_np)
            
            max_diff = np.max(np.abs(result_p_np - result_n))
            if max_diff < 1e-5:
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

all_passed = test_op("Add (same shape)", 
                     lambda x, y: paddle.add(x, y),
                     lambda x, y: torch.add(x, y),
                     lambda x, y: x + y,
                     [3, 4], [3, 4]) and all_passed

all_passed = test_op("Add (broadcast)", 
                     lambda x, y: paddle.add(x, y),
                     lambda x, y: torch.add(x, y),
                     lambda x, y: x + y,
                     [3, 4, 5], [4, 5]) and all_passed

all_passed = test_op("Multiply (same shape)", 
                     lambda x, y: paddle.multiply(x, y),
                     lambda x, y: torch.mul(x, y),
                     lambda x, y: x * y,
                     [2, 3], [2, 3]) and all_passed

all_passed = test_op("Subtract (same shape)", 
                     lambda x, y: paddle.subtract(x, y),
                     lambda x, y: torch.sub(x, y),
                     lambda x, y: x - y,
                     [3, 4], [3, 4]) and all_passed

all_passed = test_op("Divide (same shape)", 
                     lambda x, y: paddle.divide(x, y),
                     lambda x, y: torch.div(x, y),
                     lambda x, y: x / y,
                     [2, 3], [2, 3]) and all_passed

# Summary
print("\n" + "=" * 70)
if all_passed:
    print("✓ All tests passed!")
    print("=" * 70)
    sys.exit(0)
else:
    print("✗ Some tests failed")
    print("=" * 70)
    sys.exit(1)

