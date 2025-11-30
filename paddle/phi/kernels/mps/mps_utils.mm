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

#ifdef PADDLE_WITH_MPS

#include "paddle/phi/kernels/mps/mps_utils.h"

#include <vector>

#include "glog/logging.h"
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/mps_allocator.h"

namespace phi {
namespace backends {
namespace mps {

// Thread-local MPSGraph cache
static thread_local MPSGraph* g_mps_graph = nullptr;

MPSGraph* GetMPSGraph(const MPSContext& ctx) {
  if (g_mps_graph == nullptr) {
    g_mps_graph = [[MPSGraph alloc] init];
  }
  return g_mps_graph;
}

MPSGraphTensor* CreateMPSGraphTensor(MPSGraph* graph,
                                      const DenseTensor& tensor,
                                      const std::string& name) {
  @autoreleasepool {
    // Get the MTLBuffer from tensor
    id<MTLBuffer> buffer = GetMTLBuffer(tensor);
    if (buffer == nil) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Failed to get MTLBuffer from tensor for MPSGraph"));
    }

    // Get tensor shape
    auto dims = tensor.dims();
    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:dims.size()];
    for (int i = 0; i < dims.size(); ++i) {
      [shape addObject:@(dims[i])];
    }

    // Create MPSGraphTensor from buffer
    NSString* ns_name = [NSString stringWithUTF8String:name.c_str()];
    MPSGraphTensor* tensor_node = [graph placeholderWithShape:shape
                                                         dataType:MPSDataTypeFloat32
                                                           name:ns_name];
    
    return tensor_node;
  }
}

MPSGraphTensor* CreateMPSGraphTensorWithShape(MPSGraph* graph,
                                               const DenseTensor& tensor,
                                               const std::string& name) {
  @autoreleasepool {
    auto dims = tensor.dims();
    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:dims.size()];
    for (int i = 0; i < dims.size(); ++i) {
      [shape addObject:@(dims[i])];
    }

    NSString* ns_name = [NSString stringWithUTF8String:name.c_str()];
    MPSGraphTensor* tensor_node = [graph placeholderWithShape:shape
                                                         dataType:MPSDataTypeFloat32
                                                           name:ns_name];
    
    return tensor_node;
  }
}

void ExecuteMPSGraph(MPSGraph* graph,
                     NSArray<MPSGraphTensor*>* feeds,
                     NSArray<MPSGraphTensor*>* results,
                     const MPSContext& ctx) {
  // This function is not used in the current implementation
  // Each kernel handles execution directly
  PADDLE_THROW(common::errors::Unimplemented(
      "ExecuteMPSGraph helper function not implemented"));
}

id<MTLBuffer> GetMTLBuffer(const DenseTensor& tensor) {
  // Get the allocation from tensor
  auto* allocation = tensor.Holder().get();
  if (allocation == nullptr) {
    return nil;
  }

  // Try to cast to MPSAllocation to get the stored MTLBuffer
  // Use fully qualified name and ensure we're in the right namespace context
  using MPSAllocationType = ::paddle::memory::allocation::MPSAllocation;
  auto* mps_allocation = dynamic_cast<MPSAllocationType*>(allocation);
  if (mps_allocation != nullptr) {
    void* buffer_ptr = mps_allocation->buffer();
    if (buffer_ptr != nullptr) {
      return (__bridge id<MTLBuffer>)buffer_ptr;
    }
  }

  // Fallback: create buffer from pointer (for unified memory)
  void* ptr = allocation->ptr();
  if (ptr == nullptr) {
    return nil;
  }

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return nil;
    }
    
    // Create buffer with existing memory (no-copy)
    // This works for unified memory architecture
    NSUInteger length = allocation->size();
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:ptr
                                                      length:length
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    return buffer;
  }
}

}  // namespace mps
}  // namespace backends
}  // namespace phi

#endif  // PADDLE_WITH_MPS

