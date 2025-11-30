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

#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/backends/mps/mps_info.h"
#include "paddle/phi/backends/mps/mps_context.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/utils/test_macros.h"
#include "glog/logging.h"

namespace paddle::memory {

template <>
PADDLE_API void Copy<phi::MPSPlace, phi::CPUPlace>(phi::MPSPlace dst_place,
                                           void* dst,
                                           phi::CPUPlace src_place,
                                           const void* src,
                                           size_t num) {
  if (UNLIKELY(num == 0)) return;
  const auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(dst_place);
  const auto* mps_ctx = dynamic_cast<const phi::MPSContext*>(dev_ctx);
  PADDLE_ENFORCE_NOT_NULL(
      mps_ctx,
      common::errors::InvalidArgument(
          "Failed to dynamic_cast dev_ctx into phi::MPSContext."));
  phi::backends::mps::MemcpySyncH2D(dst, src, num, dst_place, *mps_ctx);
}

template <>
PADDLE_API void Copy<phi::CPUPlace, phi::MPSPlace>(phi::CPUPlace dst_place,
                                           void* dst,
                                           phi::MPSPlace src_place,
                                           const void* src,
                                           size_t num) {
  if (UNLIKELY(num == 0)) return;
  const auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(src_place);
  const auto* mps_ctx = dynamic_cast<const phi::MPSContext*>(dev_ctx);
  PADDLE_ENFORCE_NOT_NULL(
      mps_ctx,
      common::errors::InvalidArgument(
          "Failed to dynamic_cast dev_ctx into phi::MPSContext."));
  phi::backends::mps::MemcpySyncD2H(dst, src, num, src_place, *mps_ctx);
}

template <>
PADDLE_API void Copy<phi::MPSPlace, phi::MPSPlace>(phi::MPSPlace dst_place,
                                           void* dst,
                                           phi::MPSPlace src_place,
                                           const void* src,
                                           size_t num) {
  if (UNLIKELY(num == 0)) return;
  const auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(dst_place);
  const auto* mps_ctx = dynamic_cast<const phi::MPSContext*>(dev_ctx);
  PADDLE_ENFORCE_NOT_NULL(
      mps_ctx,
      common::errors::InvalidArgument(
          "Failed to dynamic_cast dev_ctx into phi::MPSContext."));
  phi::backends::mps::MemcpySyncD2D(
      dst, dst_place, src, src_place, num, *mps_ctx);
}

}  // namespace paddle::memory

#endif  // PADDLE_WITH_MPS

