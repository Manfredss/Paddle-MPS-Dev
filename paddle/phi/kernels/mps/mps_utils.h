/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_MPS

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace backends {
namespace mps {

// Helper function to get MPSGraph from context
MPSGraph* GetMPSGraph(const MPSContext& ctx);

// Helper function to create MPSGraphTensor from DenseTensor
MPSGraphTensor* CreateMPSGraphTensor(MPSGraph* graph,
                                     const DenseTensor& tensor,
                                     const std::string& name);

// Helper function to execute MPSGraph and write result to output tensor
void ExecuteMPSGraph(MPSGraph* graph,
                     NSArray<MPSGraphTensor*>* feeds,
                     NSArray<MPSGraphTensor*>* results,
                     const MPSContext& ctx);

// Helper function to get MTLBuffer from DenseTensor
id<MTLBuffer> GetMTLBuffer(const DenseTensor& tensor);

// Helper function to create MPSGraphTensor with shape
MPSGraphTensor* CreateMPSGraphTensorWithShape(MPSGraph* graph,
                                              const DenseTensor& tensor,
                                              const std::string& name);

}  // namespace mps
}  // namespace backends
}  // namespace phi

#endif  // PADDLE_WITH_MPS

