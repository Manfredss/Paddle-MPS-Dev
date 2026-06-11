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

#include "paddle/phi/kernels/elementwise_kernel.h"

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
void RemainderKernelImpl(const MPSContext& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         DenseTensor* out) {
  @autoreleasepool {
    MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);

    MPSGraphTensor* x_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, x, "x");
    MPSGraphTensor* y_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, y, "y");

    // Python-style modulo: the result takes the sign of the divisor y,
    // matching the CPU RemainderFunctor.
    //
    // Float path: remainder(x, y) = x - floor(x / y) * y.
    //
    // Integer path: Metal integer division truncates toward zero, giving a
    // C-style remainder (sign of x). We reproduce RemainderFunctor<integral T>
    // by adding y back when the C-style remainder is non-zero and its sign
    // differs from the divisor's sign.
    const bool is_int =
        (x.dtype() == DataType::INT32 || x.dtype() == DataType::INT64);
    MPSGraphTensor* result_tensor = nil;
    if (is_int) {
      MPSDataType int_type = backends::mps::GetMPSDataType(x.dtype());
      MPSGraphTensor* q =
          [graph divisionWithPrimaryTensor:x_tensor
                           secondaryTensor:y_tensor
                                      name:@"remainder_q"];
      MPSGraphTensor* prod =
          [graph multiplicationWithPrimaryTensor:q
                                 secondaryTensor:y_tensor
                                            name:@"remainder_prod"];
      MPSGraphTensor* rem =
          [graph subtractionWithPrimaryTensor:x_tensor
                              secondaryTensor:prod
                                         name:@"remainder_rem"];
      // Use an integer-valued scalar to match the integer dataType exactly.
      MPSGraphTensor* zero =
          [graph constantWithScalar:0
                              shape:@[ @1 ]
                           dataType:int_type];
      MPSGraphTensor* remNeg =
          [graph lessThanWithPrimaryTensor:rem
                           secondaryTensor:zero
                                      name:@"remainder_rem_neg"];
      MPSGraphTensor* yNeg =
          [graph lessThanWithPrimaryTensor:y_tensor
                           secondaryTensor:zero
                                      name:@"remainder_y_neg"];
      MPSGraphTensor* signDiffer =
          [graph notEqualWithPrimaryTensor:remNeg
                           secondaryTensor:yNeg
                                      name:@"remainder_sign_differ"];
      MPSGraphTensor* remNonzero =
          [graph notEqualWithPrimaryTensor:rem
                           secondaryTensor:zero
                                      name:@"remainder_rem_nonzero"];
      MPSGraphTensor* needAdd =
          [graph logicalANDWithPrimaryTensor:signDiffer
                             secondaryTensor:remNonzero
                                        name:@"remainder_need_add"];
      MPSGraphTensor* remPlusY =
          [graph additionWithPrimaryTensor:rem
                           secondaryTensor:y_tensor
                                      name:@"remainder_rem_plus_y"];
      result_tensor =
          [graph selectWithPredicateTensor:needAdd
                       truePredicateTensor:remPlusY
                      falsePredicateTensor:rem
                                      name:@"remainder_result"];
    } else {
      MPSGraphTensor* quotient_tensor =
          [graph divisionWithPrimaryTensor:x_tensor
                           secondaryTensor:y_tensor
                                      name:@"remainder_quotient"];
      MPSGraphTensor* floor_tensor =
          [graph floorWithTensor:quotient_tensor
                            name:@"remainder_floor_quotient"];
      MPSGraphTensor* prod_tensor =
          [graph multiplicationWithPrimaryTensor:floor_tensor
                                 secondaryTensor:y_tensor
                                            name:@"remainder_prod"];
      result_tensor =
          [graph subtractionWithPrimaryTensor:x_tensor
                              secondaryTensor:prod_tensor
                                         name:@"remainder_result"];
    }

    dev_ctx.template Alloc<T>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for remainder";
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
    id<MTLBuffer> y_buffer = backends::mps::GetMTLBuffer(y);
    if (x_buffer == nil || y_buffer == nil) {
      VLOG(3) << "Input buffer not available, using CPU fallback for "
                 "remainder";
      return;
    }

    auto x_dims = x.dims();
    auto y_dims = y.dims();
    NSMutableArray<NSNumber*>* x_shape = [NSMutableArray arrayWithCapacity:x_dims.size()];
    for (int i = 0; i < x_dims.size(); ++i) {
      [x_shape addObject:@(x_dims[i])];
    }
    NSMutableArray<NSNumber*>* y_shape = [NSMutableArray arrayWithCapacity:y_dims.size()];
    for (int i = 0; i < y_dims.size(); ++i) {
      [y_shape addObject:@(y_dims[i])];
    }

    MPSGraphTensorData* x_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:x_buffer
                    shape:x_shape
                 dataType:backends::mps::GetMPSDataType(x.dtype())];
    MPSGraphTensorData* y_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:y_buffer
                    shape:y_shape
                 dataType:backends::mps::GetMPSDataType(y.dtype())];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
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
void RemainderKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    RemainderKernelImpl<T>(*mps_ctx, x, y, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
PD_REGISTER_KERNEL(remainder,
                   MPS,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   phi::dtype::float16,
                   int32_t,
                   int64_t,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(remainder,
                   MPS,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
#endif

#endif  // PADDLE_WITH_MPS
