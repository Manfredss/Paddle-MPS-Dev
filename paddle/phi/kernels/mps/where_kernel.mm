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

#include "paddle/phi/kernels/where_kernel.h"

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

template <typename T>
void WhereKernelImpl(const MPSContext& dev_ctx,
                     const DenseTensor& condition,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  @autoreleasepool {
    MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);

    auto cond_dims = condition.dims();
    NSMutableArray<NSNumber*>* cond_shape = [NSMutableArray arrayWithCapacity:cond_dims.size()];
    for (int i = 0; i < cond_dims.size(); ++i) {
      [cond_shape addObject:@(cond_dims[i])];
    }

    // condition is a bool tensor.
    MPSGraphTensor* cond_tensor = [graph placeholderWithShape:cond_shape
                                                     dataType:MPSDataTypeBool
                                                         name:@"condition"];
    MPSGraphTensor* x_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, x, "x");
    MPSGraphTensor* y_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, y, "y");

    // out = condition ? x : y
    MPSGraphTensor* result_tensor =
        [graph selectWithPredicateTensor:cond_tensor
                     truePredicateTensor:x_tensor
                    falsePredicateTensor:y_tensor
                                    name:@"where_result"];

    dev_ctx.template Alloc<T>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for where";
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
                 dataType:backends::mps::GetMPSDataType(out->dtype())];

    id<MTLBuffer> cond_buffer = backends::mps::GetMTLBuffer(condition);
    id<MTLBuffer> x_buffer = backends::mps::GetMTLBuffer(x);
    id<MTLBuffer> y_buffer = backends::mps::GetMTLBuffer(y);
    if (cond_buffer == nil || x_buffer == nil || y_buffer == nil) {
      VLOG(3) << "Input buffer not available, using CPU fallback for where";
      return;
    }

    MPSGraphTensorData* cond_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:cond_buffer
                    shape:cond_shape
                 dataType:MPSDataTypeBool];

    auto x_dims = x.dims();
    NSMutableArray<NSNumber*>* x_shape = [NSMutableArray arrayWithCapacity:x_dims.size()];
    for (int i = 0; i < x_dims.size(); ++i) {
      [x_shape addObject:@(x_dims[i])];
    }
    MPSGraphTensorData* x_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:x_buffer
                    shape:x_shape
                 dataType:backends::mps::GetMPSDataType(x.dtype())];

    auto y_dims = y.dims();
    NSMutableArray<NSNumber*>* y_shape = [NSMutableArray arrayWithCapacity:y_dims.size()];
    for (int i = 0; i < y_dims.size(); ++i) {
      [y_shape addObject:@(y_dims[i])];
    }
    MPSGraphTensorData* y_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:y_buffer
                    shape:y_shape
                 dataType:backends::mps::GetMPSDataType(y.dtype())];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      cond_tensor: cond_data,
      x_tensor: x_data,
      y_tensor: y_data,
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      result_tensor: out_data,
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
void WhereKernel(const Context& dev_ctx,
                 const DenseTensor& condition,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 DenseTensor* out) {
  if (condition.numel() == 0 || x.numel() == 0 || y.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    WhereKernelImpl<T>(*mps_ctx, condition, x, y, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
PD_REGISTER_KERNEL(where,
                   MPS,
                   ALL_LAYOUT,
                   phi::WhereKernel,
                   float,
                   phi::dtype::float16,
                   int32_t,
                   int64_t,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(where,
                   MPS,
                   ALL_LAYOUT,
                   phi::WhereKernel,
                   float,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
#endif

#endif  // PADDLE_WITH_MPS
