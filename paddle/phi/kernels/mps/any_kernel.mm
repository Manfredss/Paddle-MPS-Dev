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

#include "paddle/phi/kernels/reduce_any_kernel.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>

#include "glog/logging.h"
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/kernel_utils.h"
#include "paddle/phi/kernels/mps/mps_utils.h"

namespace phi {

template <typename T>
void AnyKernelImpl(const MPSContext& dev_ctx,
                   const DenseTensor& x,
                   const std::vector<int64_t>& dims,
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
      for (auto a : dims) {
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

    NSMutableArray<NSNumber*>* x_shape = [NSMutableArray arrayWithCapacity:x_dims.size()];
    for (int i = 0; i < x_dims.size(); ++i) {
      [x_shape addObject:@(x_dims[i])];
    }
    MPSGraphTensor* x_tensor = [graph placeholderWithShape:x_shape
                                                  dataType:MPSDataTypeBool
                                                      name:@"x"];

    MPSGraphTensor* x_float = [graph castTensor:x_tensor
                                         toType:MPSDataTypeFloat32
                                           name:@"any_cast"];
    MPSGraphTensor* reduced = [graph reductionMaximumWithTensor:x_float
                                                           axes:axes
                                                           name:@"any_reduce"];
    MPSGraphTensor* half_tensor =
        [graph constantWithScalar:0.5
                            shape:@[@1]
                         dataType:MPSDataTypeFloat32];
    MPSGraphTensor* bool_tensor =
        [graph greaterThanWithPrimaryTensor:reduced
                            secondaryTensor:half_tensor
                                       name:@"any_greater"];

    auto out_dims = out->dims();
    NSMutableArray<NSNumber*>* out_shape = [NSMutableArray array];
    if (out_dims.size() == 0) {
      [out_shape addObject:@(1)];
    } else {
      for (int i = 0; i < out_dims.size(); ++i) {
        [out_shape addObject:@(out_dims[i])];
      }
    }
    MPSGraphTensor* result_tensor = [graph reshapeTensor:bool_tensor
                                              withShape:out_shape
                                                   name:@"any_reshape"];

    dev_ctx.template Alloc<bool>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for any";
      return;
    }

    MPSGraphTensorData* out_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:out_buffer
                    shape:out_shape
                 dataType:MPSDataTypeBool];

    id<MTLBuffer> x_buffer = backends::mps::GetMTLBuffer(x);
    if (x_buffer == nil) {
      VLOG(3) << "Input buffer not available, using CPU fallback for any";
      return;
    }
    MPSGraphTensorData* x_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:x_buffer
                    shape:x_shape
                 dataType:MPSDataTypeBool];

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
TEST_API void AnyKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<int64_t>& dims,
                        bool keep_dim,
                        DenseTensor* out) {
  if (x.numel() == 0) {
    // Empty input: fill with false, the identity element of logical OR
    // (matches CPU semantics: np.any of an empty array is False).
    // MPS allocations use MTLResourceStorageModeShared, so the pointer
    // returned by Alloc is host-accessible unified memory.
    bool* out_data = dev_ctx.template Alloc<bool>(out);
    if (out->numel() > 0) {
      std::fill_n(out_data, out->numel(), false);
    }
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    AnyKernelImpl<T>(*mps_ctx, x, dims, keep_dim, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(any,
                   MPS,
                   ALL_LAYOUT,
                   phi::AnyKernel,
                   bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

#endif  // PADDLE_WITH_MPS
