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

#include "paddle/phi/kernels/activation_kernel.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "glog/logging.h"
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/mps/mps_utils.h"

namespace phi {

template <typename T>
void HardShrinkKernelImpl(const MPSContext& dev_ctx,
                          const DenseTensor& x,
                          float threshold,
                          DenseTensor* out) {
  @autoreleasepool {
    MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);

    MPSGraphTensor* x_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, x, "x");

    // hardshrink(x) = x if x <= -threshold or x >= threshold, else 0.
    MPSGraphTensor* zero = [graph constantWithScalar:0.0f
                                               shape:@[@1]
                                            dataType:MPSDataTypeFloat32];
    MPSGraphTensor* thr = [graph constantWithScalar:threshold
                                              shape:@[@1]
                                           dataType:MPSDataTypeFloat32];
    MPSGraphTensor* neg_thr = [graph constantWithScalar:-threshold
                                                  shape:@[@1]
                                               dataType:MPSDataTypeFloat32];
    MPSGraphTensor* is_low = [graph lessThanOrEqualToWithPrimaryTensor:x_tensor
                                                     secondaryTensor:neg_thr
                                                                name:@"hardshrink_low"];
    MPSGraphTensor* is_high = [graph greaterThanOrEqualToWithPrimaryTensor:x_tensor
                                                         secondaryTensor:thr
                                                                    name:@"hardshrink_high"];
    MPSGraphTensor* keep = [graph logicalORWithPrimaryTensor:is_low
                                            secondaryTensor:is_high
                                                       name:@"hardshrink_keep"];
    MPSGraphTensor* result_tensor =
        [graph selectWithPredicateTensor:keep
                     truePredicateTensor:x_tensor
                    falsePredicateTensor:zero
                                    name:@"hardshrink_result"];

    dev_ctx.template Alloc<T>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for hard_shrink";
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
                 dataType:MPSDataTypeFloat32];

    id<MTLBuffer> x_buffer = backends::mps::GetMTLBuffer(x);
    if (x_buffer == nil) {
      VLOG(3) << "Input buffer not available, using CPU fallback for hard_shrink";
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
void HardShrinkKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      float threshold,
                      DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    HardShrinkKernelImpl<T>(*mps_ctx, x, threshold, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(hard_shrink,
                   MPS,
                   ALL_LAYOUT,
                   phi::HardShrinkKernel,
                   float) {}

#endif  // PADDLE_WITH_MPS
