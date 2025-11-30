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

if(WITH_MPS)
  if(NOT APPLE)
    message(FATAL_ERROR "MPS is only supported on Apple platforms (macOS)")
endif()

  # Check or set CMAKE_OSX_ARCHITECTURES for arm64
  if(CMAKE_OSX_ARCHITECTURES)
    # Check if arm64 is in the list
    string(FIND "${CMAKE_OSX_ARCHITECTURES}" "arm64" ARM64_POS)
    if(ARM64_POS EQUAL -1)
      message(FATAL_ERROR "MPS requires arm64 architecture. CMAKE_OSX_ARCHITECTURES is set to '${CMAKE_OSX_ARCHITECTURES}' but must include 'arm64'")
endif()
  else()
    # If not set, set it to arm64
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "OS X architectures" FORCE)
    message(STATUS "Setting CMAKE_OSX_ARCHITECTURES to arm64 for MPS support")
endif()

  # Check for Metal frameworks
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders REQUIRED)
  find_library(METALPERFORMANCESHADERSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

  if(NOT METAL_FRAMEWORK OR NOT METALPERFORMANCESHADERS_FRAMEWORK OR NOT METALPERFORMANCESHADERSGRAPH_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK)
    message(FATAL_ERROR "Required Metal frameworks not found. MPS requires Metal, MetalPerformanceShaders, MetalPerformanceShadersGraph, and Foundation frameworks.")
endif()

  add_definitions(-DPADDLE_WITH_MPS)
  message(STATUS "MPS support enabled")
endif()

