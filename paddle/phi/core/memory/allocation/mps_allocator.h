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

#pragma once

#ifdef PADDLE_WITH_MPS

#include <mutex>  // NOLINT

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

// Custom allocation class to store MTLBuffer reference for proper cleanup
class MPSAllocation : public phi::Allocation {
 public:
  MPSAllocation(void* ptr, size_t size, const phi::Place& place, void* buffer);
  void* buffer() const;

 private:
  void* buffer_;  // id<MTLBuffer> stored as void*
};

class MPSAllocator : public Allocator {
 public:
  explicit MPSAllocator(const phi::MPSPlace& place) : place_(place) {}

  bool IsAllocThreadSafe() const override;

 protected:
  void FreeImpl(phi::Allocation* allocation) override;
  phi::Allocation* AllocateImpl(size_t size) override;

 private:
  phi::MPSPlace place_;
  std::once_flag once_flag_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif  // PADDLE_WITH_MPS

