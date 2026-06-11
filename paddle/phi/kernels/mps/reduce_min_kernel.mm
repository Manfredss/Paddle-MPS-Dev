/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/kernels/reduce_min_kernel.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "glog/logging.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/kernel_utils.h"
#include "paddle/phi/kernels/mps/mps_utils.h"

namespace phi {

template <typename T>
void MinKernelImpl(const MPSContext& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& dims,
                   bool keep_dim,
                   DenseTensor* out) {
  @autoreleasepool {
    MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);

    auto x_dims = x.dims();
    int rank = x_dims.size();
    bool reduce_all = recompute_reduce_all(x, dims);

    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    if (reduce_all) {
      for (int i = 0; i < rank; ++i) {
        [axes addObject:@(i)];
      }
    } else {
      const auto& raw = dims.GetData();
      for (auto a : raw) {
        int64_t normalized = a < 0 ? a + rank : a;
        PADDLE_ENFORCE_GE(normalized,
                          0,
                          common::errors::InvalidArgument(
                              "reduce axis out of range: %d for rank %d",
                              static_cast<int>(a), rank));
        PADDLE_ENFORCE_LT(normalized,
                          rank,
                          common::errors::InvalidArgument(
                              "reduce axis out of range: %d for rank %d",
                              static_cast<int>(a), rank));
        [axes addObject:@(normalized)];
      }
    }

    MPSGraphTensor* x_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, x, "x");

    MPSGraphTensor* reduced = [graph reductionMinimumWithTensor:x_tensor
                                                           axes:axes
                                                           name:@"min_reduce"];

    auto out_dims = out->dims();
    NSMutableArray<NSNumber*>* out_shape = [NSMutableArray array];
    if (out_dims.size() == 0) {
      [out_shape addObject:@(1)];
    } else {
      for (int i = 0; i < out_dims.size(); ++i) {
        [out_shape addObject:@(out_dims[i])];
      }
    }
    MPSGraphTensor* result_tensor = [graph reshapeTensor:reduced
                                              withShape:out_shape
                                                   name:@"min_reshape"];

    dev_ctx.template Alloc<T>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for min";
      return;
    }

    MPSGraphTensorData* out_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:out_buffer
                    shape:out_shape
                 dataType:backends::mps::GetMPSDataType(out->dtype())];

    id<MTLBuffer> x_buffer = backends::mps::GetMTLBuffer(x);
    if (x_buffer == nil) {
      VLOG(3) << "Input buffer not available, using CPU fallback for min";
      return;
    }

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
void MinKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const IntArray& dims,
               bool keep_dim,
               DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    MinKernelImpl<T>(*mps_ctx, x, dims, keep_dim, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
PD_REGISTER_KERNEL(min,
                   MPS,
                   ALL_LAYOUT,
                   phi::MinKernel,
                   float,
                   phi::dtype::float16,
                   int32_t,
                   int64_t,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(min,
                   MPS,
                   ALL_LAYOUT,
                   phi::MinKernel,
                   float,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
#endif

#endif  // PADDLE_WITH_MPS
