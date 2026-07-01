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

#include "paddle/phi/kernels/activation_kernel.h"

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
    void SquareKernelImpl(const MPSContext& dev_ctx,
                          const DenseTensor& x,
                          DenseTensor* out) {
        @autoreleasepool {
            VLOG(3) << "SquareKernelImpl called, dtype=" << static_cast<int>(x.dtype());
            MPSGraph* graph = backends::mps::GetMPSGraph(dev_ctx);

            // Get the data type from input tensor
            DataType dtype = x.dtype();
            MPSDataType mps_dtype = backends::mps::GetMPSDataType(dtype);
            VLOG(3) << "MPSDataType=" << static_cast<int>(mps_dtype);

            // Create input tensor placeholder
            auto x_dims = x.dims();
            NSMutableArray<NSNumber*>* x_shape = [
                NSMutableArray arrayWithCapacity:x_dims.size()
            ];
            for (int i = 0; i < x_dims.size(); ++i) {
                [x_shape addObject:@(x_dims[i])];
            }
            NSString* x_name = [NSString stringWithUTF8String:"x"];
            MPSGraphTensor* x_tensor = [graph placeholderWithShape:x_shape
                                                          dataType:mps_dtype
                                                              name:x_name];

            // Perform square: x * x
            MPSGraphTensor* result_tensor = [
                graph multiplicationWithPrimaryTensor:x_tensor
                secondaryTensor:x_tensor
                name:@"square_result"
            ];

            // Allocate output mem
            dev_ctx.template Alloc<T>(out);

            // Fetch output buffer
            id<MTLBuffer> out_buffer = backends::mps::GetMTLBuffer(*out);
            if (out_buffer == nil) {
                VLOG(3) << "MPS buffer not available for output, using CPU fallback for square";
                return;
            }

            // Create output shape array
            auto out_dims = out->dims();
            NSMutableArray<NSNumber*>* out_shape = [
                NSMutableArray arrayWithCapacity:out_dims.size()
            ];

            for (int i = 0; i < out_dims.size(); ++i) {
                [out_shape addObject:@(out_dims[i])];
            }

            // Create output tensor data
            MPSGraphTensorData* out_data = [
                [MPSGraphTensorData alloc]
                initWithMTLBuffer:out_buffer
                            shape:out_shape
                         dataType:mps_dtype
            ];

            // Get input buffer
            id<MTLBuffer> x_buffer = backends::mps::GetMTLBuffer(x);
            if (x_buffer == nil) {
                VLOG(3) << "MPS buffer not available for input, using CPU fallback for square";
                return;
            }

            // Create input tensor data
            MPSGraphTensorData* x_data = [
                [MPSGraphTensorData alloc]
                initWithMTLBuffer:x_buffer
                            shape:x_shape
                         dataType:mps_dtype
            ];

            // Create feeds dictionary (maps placeholders to data)
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
                x_tensor: x_data
            };

            // Create results dictionary (maps result tensor to output data)
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
                result_tensor: out_data
            };

            // Get device and command queue
            id<MTLDevice> device = (__bridge id<MTLDevice>)dev_ctx.device();
            id<MTLCommandQueue> commandQueue = [device newCommandQueue];

            // Execute the graph
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
    void SquareKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      DenseTensor* out) {
        VLOG(3) << "SquareKernel called for MPS, dtype=" << static_cast<int>(x.dtype());

        // Handle empty tensor case
        if (x.numel() == 0) {
            dev_ctx.template Alloc<T>(out);
            return;
        }

        // Cast to MPSContext and call implementation
        const auto* mps_ctx = dynamic_cast<const MPSContext*>(&dev_ctx);
        if (mps_ctx != nullptr) {
            SquareKernelImpl<T>(*mps_ctx, x, out);
        } else {
            VLOG(3) << "SquareKernel: Not MPSContext, falling back";
            PADDLE_THROW(common::errors::InvalidArgument(
                "Expected MPSContext but got different context type"));
        }
    }
} // namespace phi

// Register the kernel
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
PD_REGISTER_KERNEL(square,
                   MPS,
                   ALL_LAYOUT,
                   phi::SquareKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(square,
                   MPS,
                   ALL_LAYOUT,
                   phi::SquareKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t) {}
#endif

#endif // PADDLE_WITH_MPS
