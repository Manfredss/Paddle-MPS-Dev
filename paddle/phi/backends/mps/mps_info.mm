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

#include "paddle/phi/backends/mps/mps_info.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <mutex>
#include <vector>

#include "glog/logging.h"
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/core/memory/malloc.h"

namespace phi {
namespace backends {
namespace mps {

static int g_mps_device_id = 0;
static std::mutex g_mps_device_mutex;

int GetMPSDeviceCount() {
  @autoreleasepool {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    int count = static_cast<int>([devices count]);
    // Release the devices array since MTLCopyAllDevices returns a retained object
    [devices release];
    return count;
  }
}

void SetMPSDeviceId(int device_id) {
  std::lock_guard<std::mutex> lock(g_mps_device_mutex);
  g_mps_device_id = device_id;
}

int GetMPSCurrentDeviceId() {
  std::lock_guard<std::mutex> lock(g_mps_device_mutex);
  return g_mps_device_id;
}

std::vector<int> GetMPSSelectedDevices() {
  std::vector<int> devices;
  const char *selected_devices = std::getenv("FLAGS_selected_mps");
  if (selected_devices != nullptr) {
    std::string devices_str(selected_devices);
    size_t pos = 0;
    while ((pos = devices_str.find(',')) != std::string::npos) {
      devices.push_back(std::stoi(devices_str.substr(0, pos)));
      devices_str.erase(0, pos + 1);
    }
    if (!devices_str.empty()) {
      devices.push_back(std::stoi(devices_str));
    }
  } else {
    int device_count = GetMPSDeviceCount();
    for (int i = 0; i < device_count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}

void MemcpySyncH2D(void *dst,
                   const void *src,
                   size_t count,
                   const phi::MPSPlace &dst_place,
                   const phi::MPSContext &dev_ctx) {
  // MPS uses unified memory, so we can use memcpy directly
  // The memory is accessible from both CPU and GPU without explicit transfer
  memcpy(dst, src, count);
}

void MemcpySyncD2H(void *dst,
                   const void *src,
                   size_t count,
                   const phi::MPSPlace &src_place,
                   const phi::MPSContext &dev_ctx) {
  // MPS uses unified memory, so we can use memcpy directly
  // The memory is accessible from both CPU and GPU without explicit transfer
  memcpy(dst, src, count);
}

void MemcpySyncD2D(void *dst,
                   const phi::MPSPlace &dst_place,
                   const void *src,
                   const phi::MPSPlace &src_place,
                   size_t count,
                   const phi::MPSContext &dev_ctx) {
  // MPS uses unified memory, so we can use memcpy directly
  // The memory is accessible from both CPU and GPU without explicit transfer
  memcpy(dst, src, count);
}

/***** Memory Management *****/

void EmptyCache() {
  std::vector<int> devices = GetMPSSelectedDevices();
  for (auto device : devices) {
    paddle::memory::Release(phi::MPSPlace(device));
  }
}

std::string GetMPSDeviceName(int device_id) {
  @autoreleasepool {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    if (devices != nil && device_id >= 0 && device_id < static_cast<int>([devices count])) {
      id<MTLDevice> device = devices[device_id];
      NSString *name = [device name];
      std::string device_name = std::string([name UTF8String]);
      [devices release];
      return device_name;
    }
    [devices release];
    return "Unknown MPS Device";
  }
}

int64_t GetMPSDeviceTotalMemory(int device_id) {
  @autoreleasepool {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    if (devices != nil && device_id >= 0 && device_id < static_cast<int>([devices count])) {
      id<MTLDevice> device = devices[device_id];
      // MPS uses unified memory, so we get the recommended working set size
      // This is an approximation of available memory
      // recommendedMaxWorkingSetSize is available on macOS 13.0+
      if (@available(macOS 13.0, *)) {
        uint64_t recommendedMaxWorkingSetSize = [device recommendedMaxWorkingSetSize];
        [devices release];
        return static_cast<int64_t>(recommendedMaxWorkingSetSize);
      } else {
        // Fallback for macOS 12.x: Use a reasonable default or query system memory
        // For unified memory, we can't easily get GPU-specific memory, so return 0
        // to indicate unknown
        [devices release];
        return 0;
      }
    }
    [devices release];
    return 0;
  }
}

}  // namespace mps
}  // namespace backends
}  // namespace phi

#endif  // PADDLE_WITH_MPS

