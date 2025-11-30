# Paddle-MPS-Dev: Metal Performance Shaders (MPS) Backend for PaddlePaddle

<div align="center">

📖 **[English Documentation](README_EN.md)** | 📖 **[简体中文文档](README_ZH.md)**

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

**Note**: This is an active development project. Features and APIs may change. Please report issues and contribute improvements!
