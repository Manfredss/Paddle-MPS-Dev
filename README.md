# Paddle-MPS-Dev: Metal Performance Shaders (MPS) Backend for PaddlePaddle

This document describes the implementation of Metal Performance Shaders (MPS) support for PaddlePaddle, enabling GPU acceleration on Apple Silicon (M1/M2/M3 and later) devices.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Building with MPS Support](#building-with-mps-support)
- [Usage](#usage)
- [Implemented Operators](#implemented-operators)
- [Architecture](#architecture)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)

## Overview

Metal Performance Shaders (MPS) is Apple's framework for high-performance GPU-accelerated computation on Apple Silicon. This implementation brings MPS support to PaddlePaddle, allowing users to leverage the GPU capabilities of their Mac devices for deep learning workloads.

The implementation follows a similar API design to PyTorch's MPS backend, making it familiar to users who have experience with PyTorch on macOS.

## Features

### Core Functionality

- ✅ **Device Management**: Full support for MPS device detection, selection, and management
- ✅ **Memory Management**: Custom MPS allocator with unified memory support
- ✅ **Tensor Operations**: Basic tensor creation and operations on MPS devices
- ✅ **Memory Copy**: Efficient CPU ↔ MPS and MPS ↔ MPS memory transfers
- ✅ **Python API**: PyTorch-style API (`paddle.mps.is_available()`, `paddle.mps.set_device()`, etc.)

### Implemented Operators

#### Elementwise Binary Operations
- `add` - Elementwise addition
- `multiply` - Elementwise multiplication
- `subtract` - Elementwise subtraction
- `divide` - Elementwise division

#### Unary Operations
- `abs` - Absolute value
- `exp` - Exponential function
- `log` - Natural logarithm
- `sqrt` - Square root
- `relu` - Rectified Linear Unit activation
- `sigmoid` - Sigmoid activation function

### Memory Management Features

- **Unified Memory**: Leverages Apple Silicon's unified memory architecture
- **Buddy Allocator**: Efficient memory allocation using buddy allocator strategy
- **Memory Statistics**: Track allocated and reserved memory per device
- **Cache Management**: `empty_cache()` function to release unused memory

## Requirements

### Hardware
- Apple Silicon Mac (M1, M2, M3, or later)
- macOS 12.0 or later (for MPSGraph support)

### Software
- Xcode with Command Line Tools
- CMake 3.15 or later
- Python 3.7 or later

## Building with MPS Support

### CMake Configuration

Enable MPS support when configuring the build:

```bash
cmake .. \
  -DWITH_MPS=ON \
  -DCMAKE_OSX_ARCHITECTURES=arm64
```

The `CMAKE_OSX_ARCHITECTURES=arm64` is required as MPS only supports ARM64 architecture.

### Build Process

```bash
# Configure
mkdir build && cd build
cmake .. -DWITH_MPS=ON -DCMAKE_OSX_ARCHITECTURES=arm64

# Build
make -j$(sysctl -n hw.ncpu)

# Install Python package
pip install -e ../python
```

## Usage

### Basic Usage

```python
import paddle

# Check if MPS is available
if paddle.is_compiled_with_mps() and paddle.mps.is_available():
    print(f"MPS available! Device count: {paddle.mps.device_count()}")
    
    # Set device
    paddle.mps.set_device(0)
    
    # Create tensors on MPS
    x = paddle.randn([2, 3], dtype='float32', place='mps')
    y = paddle.randn([2, 3], dtype='float32', place='mps')
    
    # Perform operations
    z = x + y
    w = paddle.exp(x)
    
    # Convert back to CPU for NumPy conversion
    result = z.numpy()
    print(result)
```

### Device Management

```python
import paddle

# Check MPS availability
print(f"MPS compiled: {paddle.is_compiled_with_mps()}")
print(f"MPS available: {paddle.mps.is_available()}")
print(f"Device count: {paddle.mps.device_count()}")

# Set current device
paddle.mps.set_device(0)

# Get current device
current_device = paddle.mps.current_device()
print(f"Current device: {current_device}")

# Get device properties
props = paddle.mps.get_device_properties()
print(f"Device name: {paddle.mps.get_device_name()}")
print(f"Total memory: {props.total_memory / (1024**3):.2f} GB")
```

### Memory Management

```python
import paddle

# Check memory usage
allocated = paddle.mps.memory_allocated()
reserved = paddle.mps.memory_reserved()
print(f"Allocated: {allocated / (1024**2):.2f} MB")
print(f"Reserved: {reserved / (1024**2):.2f} MB")

# Clear cache
paddle.mps.empty_cache()

# Track peak memory
max_allocated = paddle.mps.max_memory_allocated()
print(f"Peak allocated: {max_allocated / (1024**2):.2f} MB")

# Reset peak statistics
paddle.mps.reset_max_memory_allocated()
```

### Operator Examples

```python
import paddle
import numpy as np

paddle.mps.set_device(0)

# Elementwise operations
x = paddle.to_tensor([1.0, 2.0, 3.0], place='mps')
y = paddle.to_tensor([4.0, 5.0, 6.0], place='mps')

# Addition
z = paddle.add(x, y)  # or z = x + y

# Multiplication
w = paddle.multiply(x, y)  # or w = x * y

# Unary operations
abs_x = paddle.abs(x)
exp_x = paddle.exp(x)
log_x = paddle.log(x + 1.0)  # Add 1 to avoid log(0)
sqrt_x = paddle.sqrt(x)

# Activation functions
relu_x = paddle.nn.functional.relu(x)
sigmoid_x = paddle.nn.functional.sigmoid(x)
```

## Implemented Operators

### Elementwise Binary Operations

All elementwise binary operations support broadcasting:

| Operator | Function | MPSGraph Method |
|----------|----------|-----------------|
| `add` | `paddle.add(x, y)` | `additionWithPrimaryTensor:secondaryTensor:` |
| `multiply` | `paddle.multiply(x, y)` | `multiplicationWithPrimaryTensor:secondaryTensor:` |
| `subtract` | `paddle.subtract(x, y)` | `subtractionWithPrimaryTensor:secondaryTensor:` |
| `divide` | `paddle.divide(x, y)` | `divisionWithPrimaryTensor:secondaryTensor:` |

### Unary Operations

| Operator | Function | MPSGraph Method |
|----------|----------|-----------------|
| `abs` | `paddle.abs(x)` | `absoluteWithTensor:` |
| `exp` | `paddle.exp(x)` | `exponentWithTensor:` |
| `log` | `paddle.log(x)` | `logarithmWithTensor:` |
| `sqrt` | `paddle.sqrt(x)` | `squareRootWithTensor:` |
| `relu` | `paddle.nn.functional.relu(x)` | `maximumWithPrimaryTensor:secondaryTensor:` |
| `sigmoid` | `paddle.nn.functional.sigmoid(x)` | Composite (exp, division) |

## Architecture

### Core Components

#### 1. MPS Backend (`paddle/phi/backends/mps/`)

- **`mps_info.h/mm`**: Device information, device management, memory operations
- **`mps_context.h/mm`**: Device context for MPS operations
- **`mps_device.h`**: MPS device abstraction

#### 2. Memory Management (`paddle/phi/core/memory/`)

- **`mps_allocator.h/mm`**: Custom allocator for MPS memory using `MTLBuffer`
- **`allocator_facade.cc`**: Allocator initialization and management
- **`memcpy.cc`**: Memory copy operations between CPU and MPS

#### 3. Kernels (`paddle/phi/kernels/mps/`)

- **Elementwise kernels**: `elementwise_add_kernel.mm`, `elementwise_multiply_kernel.mm`, etc.
- **Unary kernels**: `abs_kernel.mm`, `exp_kernel.mm`, `log_kernel.mm`, etc.
- **`mps_utils.h/mm`**: Utility functions for MPSGraph operations

#### 4. Python API (`python/paddle/`)

- **`paddle/mps/__init__.py`**: MPS-specific Python API
- **`paddle/base/framework.py`**: `is_compiled_with_mps()` function
- **`paddle/device/__init__.py`**: Device management functions

### Design Decisions

1. **MPSGraph API**: All operations use MPSGraph, Apple's high-level graph API, which provides:
   - Automatic kernel compilation and optimization
   - Efficient memory management
   - Support for complex operations

2. **Unified Memory**: Leverages Apple Silicon's unified memory architecture, allowing:
   - Zero-copy operations in some cases
   - Simplified memory management
   - Better performance for CPU-GPU transfers

3. **PyTorch-style API**: The Python API follows PyTorch's MPS API design for familiarity:
   ```python
   # PyTorch style
   torch.mps.is_available()
   torch.mps.set_device(0)
   
   # PaddlePaddle style (same pattern)
   paddle.mps.is_available()
   paddle.mps.set_device(0)
   ```

4. **Lazy Module Loading**: The `paddle.mps` module uses lazy loading to avoid import errors if MPS is not compiled:
   ```python
   # This works even if paddle.mps module doesn't exist yet
   import paddle
   if paddle.mps.is_available():  # Lazy loaded
       ...
   ```

## Testing

### Quick Test

Run the basic availability test:

```bash
python test_mps_availability.py
```

### Elementwise Operations Test

Test elementwise binary operations:

```bash
python test_mps_kernels_quick.py
```

Or run the comprehensive test:

```bash
python test/test_mps_kernels_comprehensive.py
```

### Unary Operations Test

Test unary operations:

```bash
python test_mps_unary_operators.py
```

### Unit Tests

Run the unittest-based tests:

```bash
python -m pytest test/test_mps_elementwise_kernels.py -v
```

### Test Output Example

```
============================================================
Quick MPS Kernel Test
============================================================
✓ PaddlePaddle imported
✓ MPS available, device set to 0

============================================================
Testing Operations
============================================================

Add (same shape):
  ✓ Pass (max diff: 1.19e-07)

Multiply (same shape):
  ✓ Pass (max diff: 2.38e-07)

...
```

## Known Limitations

1. **Data Types**: Currently only `float32` is supported. Support for other dtypes (float16, int32, etc.) is planned.

2. **Operator Coverage**: Only basic elementwise and unary operations are implemented. More operators (convolution, matrix multiplication, reductions, etc.) are planned.

3. **Gradient Support**: Backward passes (gradients) are not yet implemented for MPS operators.

4. **Multi-device**: Only single device (device 0) is currently supported, though the infrastructure for multi-device is in place.

5. **macOS Version**: Requires macOS 12.0 or later for MPSGraph support.

## Performance Considerations

- **First Run**: The first execution of an operator may be slower due to MPSGraph compilation. Subsequent runs use the cached compiled graph.

- **Memory**: MPS uses unified memory, so memory usage is shared between CPU and GPU. Monitor memory usage with `paddle.mps.memory_allocated()`.

- **Synchronization**: MPS operations are synchronous by default. The `paddle.mps.synchronize()` function is provided for API compatibility but is a no-op.

## Contributing

Contributions are welcome! Areas where contributions would be particularly valuable:

1. **Additional Operators**: Implement more operators (conv, matmul, reductions, etc.)
2. **Gradient Support**: Add backward pass implementations
3. **Performance Optimization**: Optimize existing kernels
4. **Testing**: Add more comprehensive tests
5. **Documentation**: Improve documentation and examples

### Adding a New Operator

To add a new MPS operator:

1. Create a kernel file in `paddle/phi/kernels/mps/` (e.g., `new_op_kernel.mm`)
2. Implement the kernel using MPSGraph API
3. Register the kernel with `PD_REGISTER_KERNEL`
4. Add tests in the test directory
5. Update this README

Example template:

```cpp
// paddle/phi/kernels/mps/new_op_kernel.mm
#ifdef PADDLE_WITH_MPS

#include "paddle/phi/kernels/new_op_kernel.h"
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "paddle/phi/kernels/mps/mps_utils.h"

namespace phi {

template <typename T>
void NewOpKernelImpl(const MPSContext& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
  @autoreleasepool {
    MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);
    MPSGraphTensor* x_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, x, "x");
    
    // Implement operation using MPSGraph
    MPSGraphTensor* result_tensor = [graph ...];
    
    // Execute graph (see existing kernels for pattern)
    ...
  }
}

template <typename T, typename Context>
void NewOpKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    NewOpKernelImpl<T>(*mps_ctx, x, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(new_op,
                   MPS,
                   ALL_LAYOUT,
                   phi::NewOpKernel,
                   float) {}

#endif  // PADDLE_WITH_MPS
```

## References

- [Apple Metal Performance Shaders Documentation](https://developer.apple.com/documentation/metalperformanceshaders)
- [MPSGraph API Reference](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

## License

This implementation follows the same license as PaddlePaddle (Apache 2.0).

## Acknowledgments

This implementation was inspired by PyTorch's MPS backend and follows similar design patterns for consistency and familiarity.

---

**Note**: This is an active development project. Features and APIs may change. Please report issues and contribute improvements!