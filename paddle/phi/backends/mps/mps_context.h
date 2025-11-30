/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www/apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_MPS

#include <memory>
#include <mutex>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

class DenseTensor;
class MPSContext : public DeviceContext,
                   public TypeInfoTraits<DeviceContext, MPSContext> {
 public:
  MPSContext();

  explicit MPSContext(const MPSPlace&);

  virtual ~MPSContext();

  /*! \brief  Return place in the device context. */
  const Place& GetPlace() const override;

  /*! \brief  Wait for all operations in the stream to complete. */
  void Wait() const override;

  /*! \brief  Return the type name of the device context. */
  static const char* name() { return "MPSContext"; }

  /*! \brief  Return the MPS device. */
  void* device() const { return device_; }

 private:
  MPSPlace place_;
  void* device_;  // id<MTLDevice>
};

}  // namespace phi

#endif  // PADDLE_WITH_MPS

