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

#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "glog/logging.h"
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/kernel_utils.h"
#include "paddle/phi/kernels/mps/mps_utils.h"

namespace phi {

template <typename T>
void SumKernelImpl(const MPSContext& dev_ctx,
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

    MPSGraphTensor* reduced = [graph reductionSumWithTensor:x_tensor
                                                       axes:axes
                                                       name:@"sum_reduce"];

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
                                                   name:@"sum_reshape"];

    dev_ctx.template Alloc<T>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for sum";
      return;
    }

    MPSGraphTensorData* out_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:out_buffer
                    shape:out_shape
                 dataType:MPSDataTypeFloat32];

    id<MTLBuffer> x_buffer = backends::mps::GetMTLBuffer(x);
    if (x_buffer == nil) {
      VLOG(3) << "Input buffer not available, using CPU fallback for sum";
      return;
    }

    NSMutableArray<NSNumber*>* x_shape = [NSMutableArray arrayWithCapacity:x_dims.size()];
    for (int i = 0; i < x_dims.size(); ++i) {
      [x_shape addObject:@(x_dims[i])];
    }
    MPSGraphTensorData* x_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:x_buffer
                    shape:x_shape
                 dataType:MPSDataTypeFloat32];

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
void SumKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const IntArray& dims,
               DataType out_dtype,
               bool keep_dim,
               DenseTensor* out) {
  // The MPS sum kernel only supports same-dtype reductions (float -> float).
  // out_dtype is allowed to be UNDEFINED (meaning "same as input") or equal
  // to the input dtype; otherwise we bail out because we don't implement cast.
  if (out_dtype != DataType::UNDEFINED && out_dtype != x.dtype()) {
    PADDLE_THROW(common::errors::Unimplemented(
        "MPS sum kernel does not support out_dtype that differs from input "
        "dtype; got input %s and out_dtype %s.",
        DataTypeToString(x.dtype()).c_str(),
        DataTypeToString(out_dtype).c_str()));
  }

  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    SumKernelImpl<T>(*mps_ctx, x, dims, keep_dim, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sum,
                   MPS,
                   ALL_LAYOUT,
                   phi::SumKernel,
                   float) {}

#endif  // PADDLE_WITH_MPS
