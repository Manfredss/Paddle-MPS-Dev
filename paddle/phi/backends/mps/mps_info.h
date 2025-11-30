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

#include <string>
#include <vector>

#include "paddle/common/macros.h"
#include "paddle/phi/common/place.h"

namespace phi {
class MPSContext;

namespace backends {
namespace mps {

/***** Device Management *****/

//! Get the total number of MPS devices in system.
PADDLE_API int GetMPSDeviceCount();

//! Set the MPS device id for next execution.
void SetMPSDeviceId(int device_id);

//! Get the current MPS device id in system.
int GetMPSCurrentDeviceId();

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetMPSSelectedDevices();

//! Get the minimum chunk size for MPS buddy allocator.
inline size_t MPSMinChunkSize() {
  // Allow to allocate the minimum chunk size is 64 bytes.
  return 1 << 6;
}

//! Copy memory from address src to dst synchronously.
void MemcpySyncH2D(void *dst,
                   const void *src,
                   size_t count,
                   const phi::MPSPlace &dst_place,
                   const phi::MPSContext &dev_ctx);

void MemcpySyncD2H(void *dst,
                   const void *src,
                   size_t count,
                   const phi::MPSPlace &src_place,
                   const phi::MPSContext &dev_ctx);

void MemcpySyncD2D(void *dst,
                   const phi::MPSPlace &dst_place,
                   const void *src,
                   const phi::MPSPlace &src_place,
                   size_t count,
                   const phi::MPSContext &dev_ctx);

/***** Memory Management *****/

//! Empty idle cached memory held by the allocator.
PADDLE_API void EmptyCache();

//! Get the name of an MPS device.
PADDLE_API std::string GetMPSDeviceName(int device_id);

//! Get the total memory of an MPS device in bytes.
PADDLE_API int64_t GetMPSDeviceTotalMemory(int device_id);

}  // namespace mps
}  // namespace backends
}  // namespace phi

#endif  // PADDLE_WITH_MPS

