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

#include "paddle/phi/kernels/gelu_kernel.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cmath>

#include "glog/logging.h"
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/mps/mps_utils.h"

namespace phi {

template <typename T>
void GeluKernelImpl(const MPSContext& dev_ctx,
                    const DenseTensor& x,
                    bool approximate,
                    DenseTensor* out) {
  @autoreleasepool {
    MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);

    MPSGraphTensor* x_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, x, "x");

    MPSGraphTensor* half = [graph constantWithScalar:0.5f
                                                shape:@[@1]
                                             dataType:MPSDataTypeFloat32];
    MPSGraphTensor* one = [graph constantWithScalar:1.0f
                                               shape:@[@1]
                                            dataType:MPSDataTypeFloat32];

    MPSGraphTensor* result_tensor = nil;
    if (approximate) {
      // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      const float kBeta = static_cast<float>(M_2_SQRTPI * M_SQRT1_2);  // sqrt(2/pi)
      const float kKappa = 0.044715f;
      MPSGraphTensor* beta = [graph constantWithScalar:kBeta
                                                  shape:@[@1]
                                               dataType:MPSDataTypeFloat32];
      MPSGraphTensor* kappa = [graph constantWithScalar:kKappa
                                                   shape:@[@1]
                                                dataType:MPSDataTypeFloat32];

      MPSGraphTensor* x_sq = [graph multiplicationWithPrimaryTensor:x_tensor
                                                     secondaryTensor:x_tensor
                                                                name:@"gelu_x_sq"];
      MPSGraphTensor* x_cube = [graph multiplicationWithPrimaryTensor:x_sq
                                                       secondaryTensor:x_tensor
                                                                  name:@"gelu_x_cube"];
      MPSGraphTensor* kx_cube = [graph multiplicationWithPrimaryTensor:kappa
                                                         secondaryTensor:x_cube
                                                                    name:@"gelu_kx_cube"];
      MPSGraphTensor* inner = [graph additionWithPrimaryTensor:x_tensor
                                                secondaryTensor:kx_cube
                                                           name:@"gelu_inner"];
      MPSGraphTensor* scaled = [graph multiplicationWithPrimaryTensor:beta
                                                       secondaryTensor:inner
                                                                  name:@"gelu_scaled"];
      MPSGraphTensor* tanh_t = [graph tanhWithTensor:scaled name:@"gelu_tanh"];
      MPSGraphTensor* one_plus = [graph additionWithPrimaryTensor:one
                                                   secondaryTensor:tanh_t
                                                              name:@"gelu_one_plus"];
      MPSGraphTensor* half_x = [graph multiplicationWithPrimaryTensor:half
                                                       secondaryTensor:x_tensor
                                                                  name:@"gelu_half_x"];
      result_tensor = [graph multiplicationWithPrimaryTensor:half_x
                                              secondaryTensor:one_plus
                                                         name:@"gelu_result"];
    } else {
      // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
      const float kInvSqrt2 = static_cast<float>(M_SQRT1_2);
      MPSGraphTensor* inv_sqrt2 = [graph constantWithScalar:kInvSqrt2
                                                       shape:@[@1]
                                                    dataType:MPSDataTypeFloat32];
      MPSGraphTensor* scaled = [graph multiplicationWithPrimaryTensor:x_tensor
                                                       secondaryTensor:inv_sqrt2
                                                                  name:@"gelu_scaled"];
      MPSGraphTensor* erf_t = [graph erfWithTensor:scaled name:@"gelu_erf"];
      MPSGraphTensor* one_plus = [graph additionWithPrimaryTensor:one
                                                   secondaryTensor:erf_t
                                                              name:@"gelu_one_plus"];
      MPSGraphTensor* half_x = [graph multiplicationWithPrimaryTensor:half
                                                       secondaryTensor:x_tensor
                                                                  name:@"gelu_half_x"];
      result_tensor = [graph multiplicationWithPrimaryTensor:half_x
                                              secondaryTensor:one_plus
                                                         name:@"gelu_result"];
    }

    dev_ctx.template Alloc<T>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for gelu";
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
      VLOG(3) << "Input buffer not available, using CPU fallback for gelu";
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
void GeluKernel(const Context& dev_ctx,
                const DenseTensor& x,
                bool approximate,
                DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    GeluKernelImpl<T>(*mps_ctx, x, approximate, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gelu,
                   MPS,
                   ALL_LAYOUT,
                   phi::GeluKernel,
                   float) {}

#endif  // PADDLE_WITH_MPS
