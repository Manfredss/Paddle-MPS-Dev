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
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/mps/mps_utils.h"

namespace phi {

template <typename T>
void SoftShrinkKernelImpl(const MPSContext& dev_ctx,
                          const DenseTensor& x,
                          float lambda,
                          DenseTensor* out) {
  @autoreleasepool {
    MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);

    MPSGraphTensor* x_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, x, "x");

    // softshrink(x) = x - lambda if x > lambda,
    //                 x + lambda if x < -lambda,
    //                 0 otherwise
    MPSGraphTensor* lambda_tensor =
        [graph constantWithScalar:static_cast<float>(lambda)
                            shape:@[@1]
                         dataType:backends::mps::GetMPSDataType(x.dtype())];
    MPSGraphTensor* neg_lambda_tensor =
        [graph constantWithScalar:static_cast<float>(-lambda)
                            shape:@[@1]
                         dataType:backends::mps::GetMPSDataType(x.dtype())];
    MPSGraphTensor* zero = [graph constantWithScalar:0.0f
                                               shape:@[@1]
                                            dataType:backends::mps::GetMPSDataType(x.dtype())];
    MPSGraphTensor* x_minus_lambda =
        [graph subtractionWithPrimaryTensor:x_tensor
                            secondaryTensor:lambda_tensor
                                       name:@"softshrink_x_minus_lambda"];
    MPSGraphTensor* x_plus_lambda =
        [graph additionWithPrimaryTensor:x_tensor
                         secondaryTensor:lambda_tensor
                                    name:@"softshrink_x_plus_lambda"];
    MPSGraphTensor* gt_pred =
        [graph greaterThanWithPrimaryTensor:x_tensor
                            secondaryTensor:lambda_tensor
                                       name:@"softshrink_gt"];
    MPSGraphTensor* lt_pred =
        [graph lessThanWithPrimaryTensor:x_tensor
                         secondaryTensor:neg_lambda_tensor
                                    name:@"softshrink_lt"];
    MPSGraphTensor* pos_part =
        [graph selectWithPredicateTensor:gt_pred
                     truePredicateTensor:x_minus_lambda
                    falsePredicateTensor:zero
                                    name:@"softshrink_pos_part"];
    MPSGraphTensor* neg_part =
        [graph selectWithPredicateTensor:lt_pred
                     truePredicateTensor:x_plus_lambda
                    falsePredicateTensor:zero
                                    name:@"softshrink_neg_part"];
    MPSGraphTensor* result_tensor =
        [graph additionWithPrimaryTensor:pos_part
                         secondaryTensor:neg_part
                                    name:@"softshrink_result"];

    dev_ctx.template Alloc<T>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for softshrink";
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

    id<MTLBuffer> x_buffer = backends::mps::GetMTLBuffer(x);
    if (x_buffer == nil) {
      VLOG(3) << "Input buffer not available, using CPU fallback for softshrink";
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
void SoftShrinkKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      float lambda,
                      DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    SoftShrinkKernelImpl<T>(*mps_ctx, x, lambda, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
PD_REGISTER_KERNEL(softshrink,
                   MPS,
                   ALL_LAYOUT,
                   phi::SoftShrinkKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(softshrink,
                   MPS,
                   ALL_LAYOUT,
                   phi::SoftShrinkKernel,
                   float,
                   phi::dtype::float16) {}
#endif

#endif  // PADDLE_WITH_MPS
