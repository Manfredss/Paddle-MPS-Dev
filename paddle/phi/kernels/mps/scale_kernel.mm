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

#include "paddle/phi/kernels/scale_kernel.h"

#include <type_traits>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "glog/logging.h"
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/mps/mps_utils.h"

namespace phi {

template <typename T>
void ScaleKernelImpl(const MPSContext& dev_ctx,
                     const DenseTensor& x,
                     const Scalar& scale,
                     const Scalar& bias,
                     bool bias_after_scale,
                     DenseTensor* out) {
  @autoreleasepool {
    MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);

    MPSGraphTensor* x_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, x, "x");

    // Resolve the scale/bias scalars to the kernel element type *before*
    // handing them to MPSGraph, matching the CPU EigenScale functor
    // (scale.to<T>() / bias.to<T>()) and the GPU ScaleFunctor (scale.to<MT>()
    // where MT == T for integer T). For integer T this casts the scalar to the
    // integer type first (e.g. 2.5 -> 2) so the constant is an exact integer,
    // rather than feeding a fractional value into an integer-typed constant and
    // relying on MPSGraph's implicit float->int rounding behaviour. For
    // floating T we keep full double precision.
    double scale_value;
    double bias_value;
    if (std::is_integral<T>::value) {
      scale_value = static_cast<double>(scale.to<T>());
      bias_value = static_cast<double>(bias.to<T>());
    } else {
      scale_value = scale.to<double>();
      bias_value = bias.to<double>();
    }

    MPSGraphTensor* scale_tensor =
        [graph constantWithScalar:scale_value
                            shape:@[@1]
                         dataType:backends::mps::GetMPSDataType(x.dtype())];
    MPSGraphTensor* bias_tensor =
        [graph constantWithScalar:bias_value
                            shape:@[@1]
                         dataType:backends::mps::GetMPSDataType(x.dtype())];

    MPSGraphTensor* result_tensor = nil;
    if (bias_after_scale) {
      // out = scale * x + bias
      MPSGraphTensor* scaled =
          [graph multiplicationWithPrimaryTensor:x_tensor
                                 secondaryTensor:scale_tensor
                                            name:@"scale_mul"];
      result_tensor = [graph additionWithPrimaryTensor:scaled
                                       secondaryTensor:bias_tensor
                                                  name:@"scale_result"];
    } else {
      // out = scale * (x + bias)
      MPSGraphTensor* shifted =
          [graph additionWithPrimaryTensor:x_tensor
                           secondaryTensor:bias_tensor
                                      name:@"scale_add_bias"];
      result_tensor = [graph multiplicationWithPrimaryTensor:shifted
                                             secondaryTensor:scale_tensor
                                                        name:@"scale_result"];
    }

    dev_ctx.template Alloc<T>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for scale";
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
      VLOG(3) << "Input buffer not available, using CPU fallback for scale";
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
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 const Scalar& bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    ScaleKernelImpl<T>(*mps_ctx, x, scale, bias, bias_after_scale, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
PD_REGISTER_KERNEL(scale,
                   MPS,
                   ALL_LAYOUT,
                   phi::ScaleKernel,
                   float,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(scale,
                   MPS,
                   ALL_LAYOUT,
                   phi::ScaleKernel,
                   float,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t) {}
#endif

#endif  // PADDLE_WITH_MPS
