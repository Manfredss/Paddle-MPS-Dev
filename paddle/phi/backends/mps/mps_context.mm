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

#include "paddle/phi/backends/mps/mps_context.h"

#include <Metal/Metal.h>

#include "glog/logging.h"
#include "paddle/phi/backends/mps/mps_info.h"

namespace phi {

MPSContext::MPSContext() : place_(MPSPlace(0)), device_(nullptr) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      PADDLE_THROW(common::errors::Unavailable(
          "MPS device not available. MPS requires macOS 12.0+ and Apple Silicon."));
    }
    // Retain the device object since we're storing it in a void* pointer
    device_ = (__bridge void*)[device retain];
  }
}

MPSContext::MPSContext(const MPSPlace& place) : place_(place), device_(nullptr) {
  @autoreleasepool {
    id<MTLDevice> device = nil;
    int device_id = place.GetDeviceId();
    
    // Get the device by ID if multiple devices are available
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    if (devices != nil && device_id >= 0 && device_id < static_cast<int>([devices count])) {
      device = devices[device_id];
    } else {
      // Fall back to default device if device_id is invalid or no devices found
      device = MTLCreateSystemDefaultDevice();
    }
    [devices release];  // Release the devices array
    
    if (device == nil) {
      PADDLE_THROW(common::errors::Unavailable(
          "MPS device not available. MPS requires macOS 12.0+ and Apple Silicon."));
    }
    // Retain the device object since we're storing it in a void* pointer
    device_ = (__bridge void*)[device retain];
  }
}

MPSContext::~MPSContext() {
  if (device_ != nullptr) {
    @autoreleasepool {
      id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
      [device release];
      device_ = nullptr;
    }
  }
}

const Place& MPSContext::GetPlace() const { return place_; }

void MPSContext::Wait() const {
  // MPS operations are typically synchronous, but we can add synchronization if needed
  // For now, this is a no-op since MPS uses unified memory
}

}  // namespace phi

#endif  // PADDLE_WITH_MPS

