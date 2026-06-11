/* Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_MPS

#include "paddle/phi/kernels/isfinite_kernel.h"

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "glog/logging.h"
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/mps/mps_utils.h"

namespace phi {

enum class IsfiniteCheckMode { kIsnan, kIsinf, kIsfinite };

template <typename T>
void IsfiniteCheckKernelImpl(const MPSContext& dev_ctx,
                             const DenseTensor& x,
                             IsfiniteCheckMode mode,
                             const char* op_name,
                             DenseTensor* out) {
  @autoreleasepool {
    MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);

    MPSGraphTensor* x_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, x, "x");

    MPSGraphTensor* result_tensor = nil;
    switch (mode) {
      case IsfiniteCheckMode::kIsnan:
        result_tensor = [graph isNaNWithTensor:x_tensor
                                          name:@"isnan_result"];
        break;
      case IsfiniteCheckMode::kIsinf:
        result_tensor = [graph isInfiniteWithTensor:x_tensor
                                               name:@"isinf_result"];
        break;
      case IsfiniteCheckMode::kIsfinite:
        result_tensor = [graph isFiniteWithTensor:x_tensor
                                             name:@"isfinite_result"];
        break;
    }

    dev_ctx.template Alloc<bool>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for "
              << op_name;
      return;
    }

    auto out_dims = out->dims();
    NSMutableArray<NSNumber*>* out_shape = [NSMutableArray arrayWithCapacity:out_dims.size()];
    for (int i = 0; i < out_dims.size(); ++i) {
      [out_shape addObject:@(out_dims[i])];
    }

    MPSGraphTensorData* out_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:out_buffer
                    shape:out_shape
                 dataType:MPSDataTypeBool];

    id<MTLBuffer> x_buffer = backends::mps::GetMTLBuffer(x);
    if (x_buffer == nil) {
      VLOG(3) << "Input buffer not available, using CPU fallback for "
              << op_name;
      return;
    }

    auto x_dims = x.dims();
    NSMutableArray<NSNumber*>* x_shape = [NSMutableArray arrayWithCapacity:x_dims.size()];
    for (int i = 0; i < x_dims.size(); ++i) {
      [x_shape addObject:@(x_dims[i])];
    }

    MPSGraphTensorData* x_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:x_buffer
                    shape:x_shape
                 dataType:backends::mps::GetMPSDataType(x.dtype())];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      x_tensor: x_data
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      result_tensor: out_data
    };

    id<MTLDevice> device = (__bridge id<MTLDevice>)dev_ctx.device();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    if (@available(macOS 12.0, *)) {
      [graph runWithMTLCommandQueue:commandQueue
                              feeds:feeds
                   targetOperations:nil
                  resultsDictionary:results];
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "MPSGraph requires macOS 12.0 or later"));
    }
  }
}

template <typename T, typename Context>
void IsnanKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    IsfiniteCheckKernelImpl<T>(
        *mps_ctx, x, IsfiniteCheckMode::kIsnan, "isnan", out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

template <typename T, typename Context>
void IsinfKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    IsfiniteCheckKernelImpl<T>(
        *mps_ctx, x, IsfiniteCheckMode::kIsinf, "isinf", out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

template <typename T, typename Context>
void IsfiniteKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    IsfiniteCheckKernelImpl<T>(
        *mps_ctx, x, IsfiniteCheckMode::kIsfinite, "isfinite", out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
PD_REGISTER_KERNEL(isnan,
                   MPS,
                   ALL_LAYOUT,
                   phi::IsnanKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(isinf,
                   MPS,
                   ALL_LAYOUT,
                   phi::IsinfKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(isfinite,
                   MPS,
                   ALL_LAYOUT,
                   phi::IsfiniteKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
#else
PD_REGISTER_KERNEL(isnan,
                   MPS,
                   ALL_LAYOUT,
                   phi::IsnanKernel,
                   float,
                   phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(isinf,
                   MPS,
                   ALL_LAYOUT,
                   phi::IsinfKernel,
                   float,
                   phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(isfinite,
                   MPS,
                   ALL_LAYOUT,
                   phi::IsfiniteKernel,
                   float,
                   phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
#endif

#endif  // PADDLE_WITH_MPS
