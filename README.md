# Paddle-MPS-Dev: Metal Performance Shaders (MPS) Backend for PaddlePaddle

<div align="center">

**English** | [简体中文](#简体中文) | [日本語](#日本語)

</div>

---

<a name="english"></a>
# Metal Performance Shaders (MPS) Backend for PaddlePaddle

<div align="right">

[English](#english) | [简体中文](#简体中文) | [↑ Back to Top](#)

</div>

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

<div align="center">

[English](#english) | [简体中文](#简体中文) | [↑ Back to Top](#)

</div>

---

<a name="简体中文"></a>
# PaddlePaddle Metal Performance Shaders (MPS) 后端实现

<div align="right">

[English](#english) | [简体中文](#简体中文) | [↑ 返回顶部](#)

</div>

本文档描述了 PaddlePaddle 中 Metal Performance Shaders (MPS) 支持的实现，使 Apple Silicon (M1/M2/M3 及后续型号) 设备能够使用 GPU 加速。

## 目录

- [概述](#概述-1)
- [功能特性](#功能特性-1)
- [系统要求](#系统要求-1)
- [编译支持 MPS](#编译支持-mps-1)
- [使用方法](#使用方法-1)
- [已实现的算子](#已实现的算子-1)
- [架构设计](#架构设计-1)
- [测试](#测试-1)
- [已知限制](#已知限制-1)
- [贡献指南](#贡献指南-1)

## 概述

Metal Performance Shaders (MPS) 是 Apple 为 Apple Silicon 设备提供的高性能 GPU 加速计算框架。本实现为 PaddlePaddle 带来了 MPS 支持，允许用户在 Mac 设备上利用 GPU 能力进行深度学习工作负载。

该实现遵循与 PyTorch 的 MPS 后端类似的 API 设计，使熟悉 PyTorch 在 macOS 上使用的用户感到熟悉。

## 功能特性

### 核心功能

- ✅ **设备管理**：完整的 MPS 设备检测、选择和管理支持
- ✅ **内存管理**：支持统一内存的自定义 MPS 分配器
- ✅ **张量操作**：在 MPS 设备上创建和执行基本张量操作
- ✅ **内存拷贝**：高效的 CPU ↔ MPS 和 MPS ↔ MPS 内存传输
- ✅ **Python API**：PyTorch 风格的 API (`paddle.mps.is_available()`, `paddle.mps.set_device()` 等)

### 已实现的算子

#### 逐元素二元运算
- `add` - 逐元素加法
- `multiply` - 逐元素乘法
- `subtract` - 逐元素减法
- `divide` - 逐元素除法

#### 一元运算
- `abs` - 绝对值
- `exp` - 指数函数
- `log` - 自然对数
- `sqrt` - 平方根
- `relu` - 修正线性单元激活函数
- `sigmoid` - Sigmoid 激活函数

### 内存管理功能

- **统一内存**：利用 Apple Silicon 的统一内存架构
- **伙伴分配器**：使用伙伴分配器策略进行高效内存分配
- **内存统计**：跟踪每个设备的已分配和保留内存
- **缓存管理**：`empty_cache()` 函数用于释放未使用的内存

## 系统要求

### 硬件要求
- Apple Silicon Mac (M1、M2、M3 或更新型号)
- macOS 12.0 或更高版本（用于 MPSGraph 支持）

### 软件要求
- 带有命令行工具的 Xcode
- CMake 3.15 或更高版本
- Python 3.7 或更高版本

## 编译支持 MPS

### CMake 配置

在配置构建时启用 MPS 支持：

```bash
cmake .. \
  -DWITH_MPS=ON \
  -DCMAKE_OSX_ARCHITECTURES=arm64
```

`CMAKE_OSX_ARCHITECTURES=arm64` 是必需的，因为 MPS 仅支持 ARM64 架构。

### 编译过程

```bash
# 配置
mkdir build && cd build
cmake .. -DWITH_MPS=ON -DCMAKE_OSX_ARCHITECTURES=arm64

# 编译
make -j$(sysctl -n hw.ncpu)

# 安装 Python 包
pip install -e ../python
```

## 使用方法

### 基本使用

```python
import paddle

# 检查 MPS 是否可用
if paddle.is_compiled_with_mps() and paddle.mps.is_available():
    print(f"MPS 可用！设备数量: {paddle.mps.device_count()}")
    
    # 设置设备
    paddle.mps.set_device(0)
    
    # 在 MPS 上创建张量
    x = paddle.randn([2, 3], dtype='float32', place='mps')
    y = paddle.randn([2, 3], dtype='float32', place='mps')
    
    # 执行操作
    z = x + y
    w = paddle.exp(x)
    
    # 转换回 CPU 以便转换为 NumPy
    result = z.numpy()
    print(result)
```

### 设备管理

```python
import paddle

# 检查 MPS 可用性
print(f"MPS 已编译: {paddle.is_compiled_with_mps()}")
print(f"MPS 可用: {paddle.mps.is_available()}")
print(f"设备数量: {paddle.mps.device_count()}")

# 设置当前设备
paddle.mps.set_device(0)

# 获取当前设备
current_device = paddle.mps.current_device()
print(f"当前设备: {current_device}")

# 获取设备属性
props = paddle.mps.get_device_properties()
print(f"设备名称: {paddle.mps.get_device_name()}")
print(f"总内存: {props.total_memory / (1024**3):.2f} GB")
```

### 内存管理

```python
import paddle

# 检查内存使用情况
allocated = paddle.mps.memory_allocated()
reserved = paddle.mps.memory_reserved()
print(f"已分配: {allocated / (1024**2):.2f} MB")
print(f"已保留: {reserved / (1024**2):.2f} MB")

# 清空缓存
paddle.mps.empty_cache()

# 跟踪峰值内存
max_allocated = paddle.mps.max_memory_allocated()
print(f"峰值已分配: {max_allocated / (1024**2):.2f} MB")

# 重置峰值统计
paddle.mps.reset_max_memory_allocated()
```

### 算子示例

```python
import paddle
import numpy as np

paddle.mps.set_device(0)

# 逐元素运算
x = paddle.to_tensor([1.0, 2.0, 3.0], place='mps')
y = paddle.to_tensor([4.0, 5.0, 6.0], place='mps')

# 加法
z = paddle.add(x, y)  # 或 z = x + y

# 乘法
w = paddle.multiply(x, y)  # 或 w = x * y

# 一元运算
abs_x = paddle.abs(x)
exp_x = paddle.exp(x)
log_x = paddle.log(x + 1.0)  # 加 1 以避免 log(0)
sqrt_x = paddle.sqrt(x)

# 激活函数
relu_x = paddle.nn.functional.relu(x)
sigmoid_x = paddle.nn.functional.sigmoid(x)
```

## 已实现的算子

### 逐元素二元运算

所有逐元素二元运算都支持广播：

| 算子 | 函数 | MPSGraph 方法 |
|------|------|---------------|
| `add` | `paddle.add(x, y)` | `additionWithPrimaryTensor:secondaryTensor:` |
| `multiply` | `paddle.multiply(x, y)` | `multiplicationWithPrimaryTensor:secondaryTensor:` |
| `subtract` | `paddle.subtract(x, y)` | `subtractionWithPrimaryTensor:secondaryTensor:` |
| `divide` | `paddle.divide(x, y)` | `divisionWithPrimaryTensor:secondaryTensor:` |

### 一元运算

| 算子 | 函数 | MPSGraph 方法 |
|------|------|---------------|
| `abs` | `paddle.abs(x)` | `absoluteWithTensor:` |
| `exp` | `paddle.exp(x)` | `exponentWithTensor:` |
| `log` | `paddle.log(x)` | `logarithmWithTensor:` |
| `sqrt` | `paddle.sqrt(x)` | `squareRootWithTensor:` |
| `relu` | `paddle.nn.functional.relu(x)` | `maximumWithPrimaryTensor:secondaryTensor:` |
| `sigmoid` | `paddle.nn.functional.sigmoid(x)` | 复合操作（exp、除法） |

## 架构设计

### 核心组件

#### 1. MPS 后端 (`paddle/phi/backends/mps/`)

- **`mps_info.h/mm`**：设备信息、设备管理、内存操作
- **`mps_context.h/mm`**：MPS 操作的设备上下文
- **`mps_device.h`**：MPS 设备抽象

#### 2. 内存管理 (`paddle/phi/core/memory/`)

- **`mps_allocator.h/mm`**：使用 `MTLBuffer` 的自定义 MPS 内存分配器
- **`allocator_facade.cc`**：分配器初始化和管理
- **`memcpy.cc`**：CPU 和 MPS 之间的内存拷贝操作

#### 3. 算子 (`paddle/phi/kernels/mps/`)

- **逐元素算子**：`elementwise_add_kernel.mm`、`elementwise_multiply_kernel.mm` 等
- **一元算子**：`abs_kernel.mm`、`exp_kernel.mm`、`log_kernel.mm` 等
- **`mps_utils.h/mm`**：MPSGraph 操作的实用函数

#### 4. Python API (`python/paddle/`)

- **`paddle/mps/__init__.py`**：MPS 特定的 Python API
- **`paddle/base/framework.py`**：`is_compiled_with_mps()` 函数
- **`paddle/device/__init__.py`**：设备管理函数

### 设计决策

1. **MPSGraph API**：所有操作都使用 MPSGraph，这是 Apple 的高级图 API，提供：
   - 自动内核编译和优化
   - 高效的内存管理
   - 支持复杂操作

2. **统一内存**：利用 Apple Silicon 的统一内存架构，允许：
   - 在某些情况下进行零拷贝操作
   - 简化的内存管理
   - 更好的 CPU-GPU 传输性能

3. **PyTorch 风格 API**：Python API 遵循 PyTorch 的 MPS API 设计，以便熟悉：
   ```python
   # PyTorch 风格
   torch.mps.is_available()
   torch.mps.set_device(0)
   
   # PaddlePaddle 风格（相同模式）
   paddle.mps.is_available()
   paddle.mps.set_device(0)
   ```

4. **延迟模块加载**：`paddle.mps` 模块使用延迟加载，以避免在未编译 MPS 时出现导入错误：
   ```python
   # 即使 paddle.mps 模块尚不存在，这也能工作
   import paddle
   if paddle.mps.is_available():  # 延迟加载
       ...
   ```

## 测试

### 快速测试

运行基本可用性测试：

```bash
python test_mps_availability.py
```

### 逐元素运算测试

测试逐元素二元运算：

```bash
python test_mps_kernels_quick.py
```

或运行综合测试：

```bash
python test/test_mps_kernels_comprehensive.py
```

### 一元运算测试

测试一元运算：

```bash
python test_mps_unary_operators.py
```

### 单元测试

运行基于 unittest 的测试：

```bash
python -m pytest test/test_mps_elementwise_kernels.py -v
```

### 测试输出示例

```
============================================================
快速 MPS 算子测试
============================================================
✓ PaddlePaddle 已导入
✓ MPS 可用，设备设置为 0

============================================================
测试操作
============================================================

加法（相同形状）:
  ✓ 通过 (最大差异: 1.19e-07)

乘法（相同形状）:
  ✓ 通过 (最大差异: 2.38e-07)

...
```

## 已知限制

1. **数据类型**：目前仅支持 `float32`。计划支持其他数据类型（float16、int32 等）。

2. **算子覆盖**：仅实现了基本的逐元素和一元运算。计划实现更多算子（卷积、矩阵乘法、归约等）。

3. **梯度支持**：MPS 算子的反向传播（梯度）尚未实现。

4. **多设备**：目前仅支持单设备（设备 0），尽管多设备的基础设施已就位。

5. **macOS 版本**：需要 macOS 12.0 或更高版本以支持 MPSGraph。

## 性能考虑

- **首次运行**：算子的首次执行可能较慢，因为需要编译 MPSGraph。后续运行使用缓存的编译图。

- **内存**：MPS 使用统一内存，因此 CPU 和 GPU 共享内存使用。使用 `paddle.mps.memory_allocated()` 监控内存使用情况。

- **同步**：MPS 操作默认是同步的。提供 `paddle.mps.synchronize()` 函数用于 API 兼容性，但它是空操作。

## 贡献

欢迎贡献！以下领域特别需要贡献：

1. **更多算子**：实现更多算子（卷积、矩阵乘法、归约等）
2. **梯度支持**：添加反向传播实现
3. **性能优化**：优化现有算子
4. **测试**：添加更全面的测试
5. **文档**：改进文档和示例

### 添加新算子

要添加新的 MPS 算子：

1. 在 `paddle/phi/kernels/mps/` 中创建算子文件（例如 `new_op_kernel.mm`）
2. 使用 MPSGraph API 实现算子
3. 使用 `PD_REGISTER_KERNEL` 注册算子
4. 在测试目录中添加测试
5. 更新本文档

示例模板：

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
    
    // 使用 MPSGraph 实现操作
    MPSGraphTensor* result_tensor = [graph ...];
    
    // 执行图（参考现有算子的模式）
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
        "期望 MPSContext 但得到了不同的上下文类型"));
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

## 参考资料

- [Apple Metal Performance Shaders 文档](https://developer.apple.com/documentation/metalperformanceshaders)
- [MPSGraph API 参考](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [PyTorch MPS 后端](https://pytorch.org/docs/stable/notes/mps.html)

## 许可证

本实现遵循与 PaddlePaddle 相同的许可证（Apache 2.0）。

## 致谢

本实现受到 PyTorch 的 MPS 后端启发，并遵循类似的设计模式以保持一致性和熟悉度。

---

<div align="center">

[English](#english) | [简体中文](#简体中文) | [↑ 返回顶部](#)

</div>

---

**注意**：这是一个活跃的开发项目。功能和 API 可能会发生变化。请报告问题并贡献改进！

