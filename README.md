# Paddle-MPS-Dev: Metal Performance Shaders (MPS) Backend for PaddlePaddle

<div align="center">

**English** | [简体中文](#简体中文) | [日本語](#日本語)

</div>

---

## Overview

This repository implements **Metal Performance Shaders (MPS)** support for PaddlePaddle, enabling GPU acceleration on Apple Silicon (M1/M2/M3 and later) devices. This implementation brings native GPU support to PaddlePaddle on macOS, allowing users to leverage the powerful GPU capabilities of their Mac devices for deep learning workloads.

### About PaddlePaddle

[PaddlePaddle](https://www.paddlepaddle.org.cn/) is an open-source deep learning platform developed by Baidu. It provides a comprehensive set of tools and libraries for building, training, and deploying deep learning models. As the first independent R&D deep learning platform in China, PaddlePaddle has been widely adopted across various industries.

### MPS Implementation

This implementation adds MPS backend support to PaddlePaddle, following a similar API design to PyTorch's MPS backend for familiarity. The implementation includes:

- ✅ **Device Management**: Full MPS device detection, selection, and management
- ✅ **Memory Management**: Custom MPS allocator with unified memory support
- ✅ **Tensor Operations**: Basic tensor creation and operations on MPS devices
- ✅ **Python API**: PyTorch-style API (`paddle.mps.is_available()`, `paddle.mps.set_device()`, etc.)
- ✅ **Elementwise Operations**: Add, multiply, subtract, divide
- ✅ **Unary Operations**: abs, exp, log, sqrt, relu, sigmoid

## Quick Start

### Requirements

- Apple Silicon Mac (M1, M2, M3, or later)
- macOS 12.0 or later
- Xcode with Command Line Tools
- CMake 3.15 or later
- Python 3.7 or later

### Build with MPS Support

```bash
# Configure
mkdir build && cd build
cmake .. -DWITH_MPS=ON -DCMAKE_OSX_ARCHITECTURES=arm64

# Build
make -j$(sysctl -n hw.ncpu)

# Install
pip install -e ../python
```

### Basic Usage

```python
import paddle

# Check MPS availability
if paddle.is_compiled_with_mps() and paddle.mps.is_available():
    paddle.mps.set_device(0)
    
    # Create tensors on MPS
    x = paddle.randn([2, 3], dtype='float32', place='mps')
    y = paddle.randn([2, 3], dtype='float32', place='mps')
    
    # Perform operations
    z = x + y
    w = paddle.exp(x)
    
    print(z.numpy())
```

## Documentation

For detailed documentation, please refer to:

- 📖 **[English Documentation](README_EN.md)** - Complete English documentation
- 📖 **[简体中文文档](README_ZH.md)** - 完整的中文文档

## Features

### Implemented Operators

**Elementwise Binary Operations:**
- `add`, `multiply`, `subtract`, `divide`

**Unary Operations:**
- `abs`, `exp`, `log`, `sqrt`, `relu`, `sigmoid`

### Architecture Highlights

- **MPSGraph API**: All operations use Apple's high-level graph API for automatic optimization
- **Unified Memory**: Leverages Apple Silicon's unified memory architecture
- **PyTorch-style API**: Familiar API design for users experienced with PyTorch MPS
- **Lazy Module Loading**: Graceful handling when MPS is not compiled

## Testing

```bash
# Quick availability test
python test_mps_availability.py

# Elementwise operations test
python test_mps_kernels_quick.py

# Unary operations test
python test_mps_unary_operators.py
```

## Current Status

✅ **Implemented:**
- Device and memory management
- Basic elementwise and unary operations
- Python API with PyTorch-style interface

🚧 **In Progress / Planned:**
- More operators (convolution, matrix multiplication, reductions, etc.)
- Gradient support (backward passes)
- Additional data types (float16, int32, etc.)
- Multi-device support

## Contributing

Contributions are welcome! Please see the detailed documentation for:
- How to add new operators
- Architecture details
- Testing guidelines

For more information, see:
- [English Contributing Guide](README_EN.md#contributing)
- [中文贡献指南](README_ZH.md#贡献指南)

## References

- [Apple Metal Performance Shaders Documentation](https://developer.apple.com/documentation/metalperformanceshaders)
- [MPSGraph API Reference](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [PaddlePaddle Official Website](https://www.paddlepaddle.org.cn/)

## License

This implementation follows the same license as PaddlePaddle (Apache 2.0).

## Acknowledgments

This implementation was inspired by PyTorch's MPS backend and follows similar design patterns for consistency and familiarity.

---

<div align="center">

[English](#overview) | [简体中文](#简体中文) | [↑ Back to Top](#)

</div>

---

<a name="简体中文"></a>
# Paddle-MPS-Dev: PaddlePaddle Metal Performance Shaders (MPS) 后端

<div align="right">

[English](#overview) | [简体中文](#简体中文) | [↑ 返回顶部](#)

</div>

## 概述

本仓库为 PaddlePaddle 实现了 **Metal Performance Shaders (MPS)** 支持，使 Apple Silicon (M1/M2/M3 及后续型号) 设备能够使用 GPU 加速。该实现为 PaddlePaddle 在 macOS 上带来了原生 GPU 支持，允许用户利用 Mac 设备的 GPU 能力进行深度学习工作负载。

### 关于 PaddlePaddle

[PaddlePaddle](https://www.paddlepaddle.org.cn/) 是由百度开发的开源深度学习平台。它提供了一套完整的工具和库，用于构建、训练和部署深度学习模型。作为中国首个独立研发的深度学习平台，PaddlePaddle 已在各个行业得到广泛应用。

### MPS 实现

本实现为 PaddlePaddle 添加了 MPS 后端支持，遵循与 PyTorch 的 MPS 后端类似的 API 设计，以便用户熟悉。实现包括：

- ✅ **设备管理**：完整的 MPS 设备检测、选择和管理
- ✅ **内存管理**：支持统一内存的自定义 MPS 分配器
- ✅ **张量操作**：在 MPS 设备上创建和执行基本张量操作
- ✅ **Python API**：PyTorch 风格的 API (`paddle.mps.is_available()`, `paddle.mps.set_device()` 等)
- ✅ **逐元素运算**：加法、乘法、减法、除法
- ✅ **一元运算**：abs、exp、log、sqrt、relu、sigmoid

## 快速开始

### 系统要求

- Apple Silicon Mac (M1、M2、M3 或更新型号)
- macOS 12.0 或更高版本
- 带有命令行工具的 Xcode
- CMake 3.15 或更高版本
- Python 3.7 或更高版本

### 编译支持 MPS

```bash
# 配置
mkdir build && cd build
cmake .. -DWITH_MPS=ON -DCMAKE_OSX_ARCHITECTURES=arm64

# 编译
make -j$(sysctl -n hw.ncpu)

# 安装
pip install -e ../python
```

### 基本使用

```python
import paddle

# 检查 MPS 是否可用
if paddle.is_compiled_with_mps() and paddle.mps.is_available():
    paddle.mps.set_device(0)
    
    # 在 MPS 上创建张量
    x = paddle.randn([2, 3], dtype='float32', place='mps')
    y = paddle.randn([2, 3], dtype='float32', place='mps')
    
    # 执行操作
    z = x + y
    w = paddle.exp(x)
    
    print(z.numpy())
```

## 文档

详细文档请参考：

- 📖 **[English Documentation](README_EN.md)** - 完整的英文文档
- 📖 **[简体中文文档](README_ZH.md)** - 完整的中文文档

## 功能特性

### 已实现的算子

**逐元素二元运算：**
- `add`、`multiply`、`subtract`、`divide`

**一元运算：**
- `abs`、`exp`、`log`、`sqrt`、`relu`、`sigmoid`

### 架构亮点

- **MPSGraph API**：所有操作使用 Apple 的高级图 API 进行自动优化
- **统一内存**：利用 Apple Silicon 的统一内存架构
- **PyTorch 风格 API**：为熟悉 PyTorch MPS 的用户提供熟悉的 API 设计
- **延迟模块加载**：当未编译 MPS 时优雅处理

## 测试

```bash
# 快速可用性测试
python test_mps_availability.py

# 逐元素运算测试
python test_mps_kernels_quick.py

# 一元运算测试
python test_mps_unary_operators.py
```

## 当前状态

✅ **已实现：**
- 设备和内存管理
- 基本逐元素和一元运算
- PyTorch 风格的 Python API

🚧 **进行中 / 计划中：**
- 更多算子（卷积、矩阵乘法、归约等）
- 梯度支持（反向传播）
- 其他数据类型（float16、int32 等）
- 多设备支持

## 贡献

欢迎贡献！详细文档请参考：
- 如何添加新算子
- 架构详情
- 测试指南

更多信息请查看：
- [English Contributing Guide](README_EN.md#contributing)
- [中文贡献指南](README_ZH.md#贡献指南)

## 参考资料

- [Apple Metal Performance Shaders 文档](https://developer.apple.com/documentation/metalperformanceshaders)
- [MPSGraph API 参考](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [PyTorch MPS 后端](https://pytorch.org/docs/stable/notes/mps.html)
- [PaddlePaddle 官方网站](https://www.paddlepaddle.org.cn/)

## 许可证

本实现遵循与 PaddlePaddle 相同的许可证（Apache 2.0）。

## 致谢

本实现受到 PyTorch 的 MPS 后端启发，并遵循类似的设计模式以保持一致性和熟悉度。

---

<div align="center">

[English](#overview) | [简体中文](#简体中文) | [↑ 返回顶部](#)

</div>

---

**注意**：这是一个活跃的开发项目。功能和 API 可能会发生变化。请报告问题并贡献改进！
