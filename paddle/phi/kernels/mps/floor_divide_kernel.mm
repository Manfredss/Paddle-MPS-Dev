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
void FloorDivideKernelImpl(const MPSContext& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           DenseTensor* out) {
  @autoreleasepool {
    MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);

    MPSGraphTensor* x_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, x, "x");
    MPSGraphTensor* y_tensor = backends::mps::CreateMPSGraphTensorWithShape(
        graph, y, "y");

    // floor_divide(x, y) = floor(x / y).
    // Float path (Python // semantics): when y == 0 the division yields IEEE
    // inf/nan and floor passes it through, matching the CPU functor behavior.
    // Integer path: Metal integer division truncates toward zero, so we
    // reproduce FloorDivideFunctor<integral T> by adjusting the truncated
    // quotient down by one when the signs of x and y differ and there is a
    // non-zero remainder. For unsigned types xNeg/yNeg are always false, so
    // the adjustment never fires and trunc == floor (inputs are non-negative).
    const bool is_int =
        (x.dtype() == DataType::INT8 || x.dtype() == DataType::INT16 ||
         x.dtype() == DataType::INT32 || x.dtype() == DataType::INT64 ||
         x.dtype() == DataType::UINT8);
    MPSGraphTensor* result_tensor = nil;
    if (is_int) {
      MPSDataType int_type = backends::mps::GetMPSDataType(x.dtype());
      MPSGraphTensor* q =
          [graph divisionWithPrimaryTensor:x_tensor
                           secondaryTensor:y_tensor
                                      name:@"floor_divide_q"];
      MPSGraphTensor* prod =
          [graph multiplicationWithPrimaryTensor:q
                                 secondaryTensor:y_tensor
                                            name:@"floor_divide_prod"];
      MPSGraphTensor* rem =
          [graph subtractionWithPrimaryTensor:x_tensor
                              secondaryTensor:prod
                                         name:@"floor_divide_rem"];
      // Use integer-valued scalars to match the integer dataType exactly.
      MPSGraphTensor* zero =
          [graph constantWithScalar:0
                              shape:@[ @1 ]
                           dataType:int_type];
      MPSGraphTensor* one =
          [graph constantWithScalar:1
                              shape:@[ @1 ]
                           dataType:int_type];
      MPSGraphTensor* xNeg =
          [graph lessThanWithPrimaryTensor:x_tensor
                           secondaryTensor:zero
                                      name:@"floor_divide_x_neg"];
      MPSGraphTensor* yNeg =
          [graph lessThanWithPrimaryTensor:y_tensor
                           secondaryTensor:zero
                                      name:@"floor_divide_y_neg"];
      MPSGraphTensor* signsDiffer =
          [graph notEqualWithPrimaryTensor:xNeg
                           secondaryTensor:yNeg
                                      name:@"floor_divide_signs_differ"];
      MPSGraphTensor* remNonzero =
          [graph notEqualWithPrimaryTensor:rem
                           secondaryTensor:zero
                                      name:@"floor_divide_rem_nonzero"];
      MPSGraphTensor* needAdjust =
          [graph logicalANDWithPrimaryTensor:signsDiffer
                             secondaryTensor:remNonzero
                                        name:@"floor_divide_need_adjust"];
      MPSGraphTensor* qMinus1 =
          [graph subtractionWithPrimaryTensor:q
                              secondaryTensor:one
                                         name:@"floor_divide_q_minus_1"];
      result_tensor =
          [graph selectWithPredicateTensor:needAdjust
                       truePredicateTensor:qMinus1
                      falsePredicateTensor:q
                                      name:@"floor_divide_result"];
    } else {
      MPSGraphTensor* quotient_tensor =
          [graph divisionWithPrimaryTensor:x_tensor
                           secondaryTensor:y_tensor
                                      name:@"floor_divide_quotient"];
      result_tensor =
          [graph floorWithTensor:quotient_tensor
                            name:@"floor_divide_result"];
    }

    dev_ctx.template Alloc<T>(out);

    id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
    if (out_buffer == nil) {
      VLOG(3) << "MPS buffer not available, using CPU fallback for "
                 "floor_divide";
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
                 "floor_divide";
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
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
  if (mps_ctx != nullptr) {
    FloorDivideKernelImpl<T>(*mps_ctx, x, y, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected MPSContext but got different context type"));
  }
}

}  // namespace phi

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
PD_REGISTER_KERNEL(floor_divide,
                   MPS,
                   ALL_LAYOUT,
                   phi::FloorDivideKernel,
                   float,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(floor_divide,
                   MPS,
                   ALL_LAYOUT,
                   phi::FloorDivideKernel,
                   float,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t) {}
#endif

#endif  // PADDLE_WITH_MPS
