# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(WITH_MPS AND APPLE)
  set(MPS_SRCS
    mps_info.mm
    mps_context.mm
  )

  foreach(src ${MPS_SRCS})
    set_source_files_properties(${src} PROPERTIES
      COMPILE_FLAGS "-x objective-c++"
      LANGUAGE OBJCXX
    )
  endforeach()

  collect_srcs(mps_srcs SRCS ${MPS_SRCS})
endif()

