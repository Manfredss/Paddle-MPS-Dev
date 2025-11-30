// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef PADDLE_WITH_MPS

#include "paddle/phi/core/memory/allocation/mps_allocator.h"

#include <Metal/Metal.h>

#include <string>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/backends/mps/mps_info.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace memory {
namespace allocation {

// MPSAllocation implementation
MPSAllocation::MPSAllocation(void* ptr, size_t size, const phi::Place& place, void* buffer)
    : phi::Allocation(ptr, size, place), buffer_(buffer) {}

void* MPSAllocation::buffer() const { return buffer_; }

bool MPSAllocator::IsAllocThreadSafe() const { return true; }

void MPSAllocator::FreeImpl(phi::Allocation* allocation) {
  PADDLE_ENFORCE_EQ(
      allocation->place(),
      place_,
      common::errors::PermissionDenied(
          "MPS memory is freed in incorrect device. This may be a bug"));
  
  @autoreleasepool {
    // Cast to MPSAllocation to get the buffer reference
    MPSAllocation* mps_allocation = dynamic_cast<MPSAllocation*>(allocation);
    if (mps_allocation != nullptr && mps_allocation->buffer() != nullptr) {
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)mps_allocation->buffer();
      [buffer release];
    }
  }
  delete allocation;
}

phi::Allocation* MPSAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_,
                 [this] { phi::backends::mps::SetMPSDeviceId(place_.GetDeviceId()); });

  @autoreleasepool {
    id<MTLDevice> device = nil;
    
    // Try to get the device from the context if available
    try {
      const auto* dev_ctx = phi::DeviceContextPool::Instance().Get(place_);
      const auto* mps_ctx = dynamic_cast<const phi::MPSContext*>(dev_ctx);
      if (mps_ctx != nullptr && mps_ctx->device() != nullptr) {
        device = (__bridge id<MTLDevice>)mps_ctx->device();
      }
    } catch (...) {
      // If context is not available, fall back to creating a new device
    }
    
    // Fall back to default device if context device is not available
    if (device == nil) {
      // Get device by ID if multiple devices are available
      NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
      int device_id = place_.GetDeviceId();
      if (devices != nil && device_id >= 0 && device_id < static_cast<int>([devices count])) {
        device = devices[device_id];
      } else {
        device = MTLCreateSystemDefaultDevice();
      }
      [devices release];  // Release the devices array
    }
    
    if (device == nil) {
      PADDLE_THROW(common::errors::Unavailable(
          "MPS device not available for memory allocation"));
    }

    // Use Metal buffer for MPS memory allocation
    id<MTLBuffer> buffer = [device newBufferWithLength:size
                                                options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "\n\nOut of memory error on MPS %d. "
          "Cannot allocate %s memory on MPS %d.\n\n",
          place_.GetDeviceId(),
          string::HumanReadableSize(size),
          place_.GetDeviceId()));
    }

    void* ptr = [buffer contents];
    if (ptr == nullptr) {
      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "\n\nOut of memory error on MPS %d. "
          "Cannot allocate %s memory on MPS %d.\n\n",
          place_.GetDeviceId(),
          string::HumanReadableSize(size),
          place_.GetDeviceId()));
    }

    // Retain the buffer so it doesn't get deallocated by autoreleasepool
    [buffer retain];
    
    // Store the buffer reference in a custom allocation class for proper cleanup
    // Use __bridge to convert, then manually retain (we already retained above)
    return new MPSAllocation(ptr, size, phi::Place(place_), (__bridge void*)buffer);
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif  // PADDLE_WITH_MPS

