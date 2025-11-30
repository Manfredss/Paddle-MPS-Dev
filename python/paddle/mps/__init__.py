# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# paddle/mps/__init__.py

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
from paddle import base, core, framework
from paddle.device import (
    device,
    device_guard,
    manual_seed,
    manual_seed_all as device_manual_seed_all,
)

if TYPE_CHECKING:
    from paddle import MPSPlace


def is_available() -> bool:
    """
    Check whether MPS (Metal Performance Shaders) is available.

    Returns:
        bool: True if MPS is available and can be used, False otherwise.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> if paddle.mps.is_available():
            ...     print("MPS is available")
            ... else:
            ...     print("MPS is not available")
    """
    if not paddle.is_compiled_with_mps():
        return False
    try:
        return core.get_mps_device_count() > 0
    except Exception:
        return False


def device_count() -> int:
    """
    Get the number of available MPS devices.

    Returns:
        int: The number of available MPS devices.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> count = paddle.mps.device_count()
            >>> print(f"Number of MPS devices: {count}")
    """
    if not is_available():
        return 0
    return core.get_mps_device_count()


def synchronize(device: MPSPlace | int | None = None) -> None:
    """
    Wait for all operations on the given MPS device to complete.

    Args:
        device (MPSPlace | int | None, optional): The MPS device to synchronize.
            - None: Synchronize the current device.
            - int: Device index, e.g., ``0`` means MPS device 0.
            - MPSPlace: A Paddle MPS place object.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.mps.synchronize()
            >>> paddle.mps.synchronize(0)
            >>> paddle.mps.synchronize(paddle.MPSPlace(0))
    """
    if device is None:
        place = paddle.framework._current_expected_place_()
        if not isinstance(place, core.MPSPlace):
            return
        device = place
    elif isinstance(device, int):
        device = core.MPSPlace(device)
    elif not isinstance(device, core.MPSPlace):
        raise ValueError(f"Invalid device type: {type(device)}")

    # MPS operations are synchronous by default, so no explicit synchronization needed
    # This is a placeholder for future implementation if needed
    pass


def set_device(device: MPSPlace | int | str) -> None:
    """
    Set the current MPS device.

    Args:
        device (MPSPlace | int | str): The MPS device to set as current.
            - int: Device index, e.g., ``0`` means MPS device 0.
            - str: Device string, e.g., ``'mps:0'`` or ``'mps'``.
            - MPSPlace: A Paddle MPS place object.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.mps.set_device(0)
            >>> paddle.mps.set_device('mps:0')
            >>> paddle.mps.set_device(paddle.MPSPlace(0))
    """
    if isinstance(device, int):
        device = f"mps:{device}"
    elif isinstance(device, core.MPSPlace):
        device = f"mps:{device.get_device_id()}"
    paddle.device.set_device(device)


def current_device() -> int:
    """
    Get the index of the currently selected MPS device.

    Returns:
        int: The index of the currently selected MPS device.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> device_id = paddle.mps.current_device()
            >>> print(f"Current MPS device: {device_id}")
    """
    place = paddle.framework._current_expected_place_()
    if isinstance(place, core.MPSPlace):
        return place.get_device_id()
    return 0


def manual_seed(seed: int) -> None:
    """
    Set the random seed for the current MPS device.

    Args:
        seed (int): The random seed to set.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.mps.manual_seed(42)
    """
    device_manual_seed_all(seed)


def manual_seed_all(seed: int) -> None:
    """
    Set the random seed for all MPS devices.

    Args:
        seed (int): The random seed to set.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.mps.manual_seed_all(42)
    """
    device_manual_seed_all(seed)


def empty_cache() -> None:
    """
    Release all unoccupied cached memory currently held by the caching allocator.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Create a tensor to allocate memory
            >>> tensor = paddle.randn([1000, 1000], device='mps')
            >>> # Delete the tensor to free memory (but it may still be cached)
            >>> del tensor
            >>> # Release the cached memory
            >>> paddle.mps.empty_cache()
    """
    if not paddle.is_compiled_with_mps():
        raise ValueError(
            "The API paddle.mps.empty_cache is not supported in "
            "PaddlePaddle without MPS support. Please reinstall PaddlePaddle with MPS support "
            "to call this API."
        )
    core.mps_empty_cache()


def memory_allocated(device: MPSPlace | int | None = None) -> int:
    """
    Return the current device memory occupied by tensors in bytes for a given device.

    Args:
        device (MPSPlace | int | None, optional): The device to query. If None, use the current device.
            Can be MPSPlace, int (device index), or None.

    Returns:
        int: The current memory occupied by tensors in bytes.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Get memory allocated for current device
            >>> mem_allocated = paddle.mps.memory_allocated()
            >>> print(f"Memory allocated: {mem_allocated} bytes")

            >>> # Get memory allocated for specific device
            >>> mem_allocated = paddle.mps.memory_allocated(0)
            >>> print(f"Memory allocated on device 0: {mem_allocated} bytes")
    """
    if not paddle.is_compiled_with_mps():
        raise ValueError(
            "The API paddle.mps.memory_allocated is not supported in "
            "PaddlePaddle without MPS support. Please reinstall PaddlePaddle with MPS support "
            "to call this API."
        )
    device_id = 0
    if device is None:
        place = paddle.framework._current_expected_place_()
        if isinstance(place, core.MPSPlace):
            device_id = place.get_device_id()
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.MPSPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(f"Invalid device type: {type(device)}")
    return core.device_memory_stat_current_value("Allocated", device_id)


def memory_reserved(device: MPSPlace | int | None = None) -> int:
    """
    Return the current device memory managed by the caching allocator in bytes for a given device.

    Args:
        device (MPSPlace | int | None, optional): The device to query. If None, use the current device.
            Can be MPSPlace, int (device index), or None.

    Returns:
        int: The current memory managed by the caching allocator in bytes.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Get memory reserved for current device
            >>> mem_reserved = paddle.mps.memory_reserved()
            >>> print(f"Memory reserved: {mem_reserved} bytes")

            >>> # Get memory reserved for specific device
            >>> mem_reserved = paddle.mps.memory_reserved(0)
            >>> print(f"Memory reserved on device 0: {mem_reserved} bytes")
    """
    if not paddle.is_compiled_with_mps():
        raise ValueError(
            "The API paddle.mps.memory_reserved is not supported in "
            "PaddlePaddle without MPS support. Please reinstall PaddlePaddle with MPS support "
            "to call this API."
        )
    device_id = 0
    if device is None:
        place = paddle.framework._current_expected_place_()
        if isinstance(place, core.MPSPlace):
            device_id = place.get_device_id()
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.MPSPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(f"Invalid device type: {type(device)}")
    return core.device_memory_stat_current_value("Reserved", device_id)


def max_memory_allocated(device: MPSPlace | int | None = None) -> int:
    """
    Return the peak size of memory that is allocated to tensor of the given device.

    Args:
        device (MPSPlace | int | None, optional): The device to query. If None, use the current device.
            Can be MPSPlace, int (device index), or None.

    Returns:
        int: The peak size of memory that is allocated to tensor of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> max_memory_allocated_size = paddle.mps.max_memory_allocated(paddle.MPSPlace(0))
            >>> max_memory_allocated_size = paddle.mps.max_memory_allocated(0)
    """
    if not paddle.is_compiled_with_mps():
        raise ValueError(
            "The API paddle.mps.max_memory_allocated is not supported in "
            "PaddlePaddle without MPS support. Please reinstall PaddlePaddle with MPS support "
            "to call this API."
        )
    device_id = 0
    if device is None:
        place = paddle.framework._current_expected_place_()
        if isinstance(place, core.MPSPlace):
            device_id = place.get_device_id()
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.MPSPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(f"Invalid device type: {type(device)}")
    return core.device_memory_stat_peak_value("Allocated", device_id)


def max_memory_reserved(device: MPSPlace | int | None = None) -> int:
    """
    Return the peak size of memory that is held by the allocator of the given device.

    Args:
        device (MPSPlace | int | None, optional): The device to query. If None, use the current device.
            Can be MPSPlace, int (device index), or None.

    Returns:
        int: The peak size of memory that is held by the allocator of the given device, in bytes.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> max_memory_reserved_size = paddle.mps.max_memory_reserved(paddle.MPSPlace(0))
            >>> max_memory_reserved_size = paddle.mps.max_memory_reserved(0)
    """
    if not paddle.is_compiled_with_mps():
        raise ValueError(
            "The API paddle.mps.max_memory_reserved is not supported in "
            "PaddlePaddle without MPS support. Please reinstall PaddlePaddle with MPS support "
            "to call this API."
        )
    device_id = 0
    if device is None:
        place = paddle.framework._current_expected_place_()
        if isinstance(place, core.MPSPlace):
            device_id = place.get_device_id()
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.MPSPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(f"Invalid device type: {type(device)}")
    return core.device_memory_stat_peak_value("Reserved", device_id)


def reset_max_memory_allocated(device: MPSPlace | int | None = None) -> None:
    """
    Reset the peak size of memory that is allocated to tensor of the given device.

    Args:
        device (MPSPlace | int | None, optional): The device to query. If None, use the current device.
            Can be MPSPlace, int (device index), or None.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.mps.reset_max_memory_allocated(paddle.MPSPlace(0))
            >>> paddle.mps.reset_max_memory_allocated(0)
    """
    if not paddle.is_compiled_with_mps():
        raise ValueError(
            "The API paddle.mps.reset_max_memory_allocated is not supported in "
            "PaddlePaddle without MPS support. Please reinstall PaddlePaddle with MPS support "
            "to call this API."
        )
    device_id = 0
    if device is None:
        place = paddle.framework._current_expected_place_()
        if isinstance(place, core.MPSPlace):
            device_id = place.get_device_id()
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.MPSPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(f"Invalid device type: {type(device)}")
    core.device_memory_stat_reset_peak_value("Allocated", device_id)


def reset_max_memory_reserved(device: MPSPlace | int | None = None) -> None:
    """
    Reset the peak size of memory that is held by the allocator of the given device.

    Args:
        device (MPSPlace | int | None, optional): The device to query. If None, use the current device.
            Can be MPSPlace, int (device index), or None.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.mps.reset_max_memory_reserved(paddle.MPSPlace(0))
            >>> paddle.mps.reset_max_memory_reserved(0)
    """
    if not paddle.is_compiled_with_mps():
        raise ValueError(
            "The API paddle.mps.reset_max_memory_reserved is not supported in "
            "PaddlePaddle without MPS support. Please reinstall PaddlePaddle with MPS support "
            "to call this API."
        )
    device_id = 0
    if device is None:
        place = paddle.framework._current_expected_place_()
        if isinstance(place, core.MPSPlace):
            device_id = place.get_device_id()
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.MPSPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(f"Invalid device type: {type(device)}")
    core.device_memory_stat_reset_peak_value("Reserved", device_id)


def get_device_properties(device: MPSPlace | int | None = None):
    """
    Get the properties of an MPS device.

    Args:
        device (MPSPlace | int | None, optional): The device to query. If None, use the current device.
            Can be MPSPlace, int (device index), or None.

    Returns:
        DeviceProperties: An object containing the device properties, such as
        name and total memory.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Get the properties of the current device
            >>> props = paddle.mps.get_device_properties()
            >>> print(props)
    """
    if not paddle.is_compiled_with_mps():
        raise ValueError(
            "The API paddle.mps.get_device_properties is not supported in "
            "PaddlePaddle without MPS support. Please reinstall PaddlePaddle with MPS support "
            "to call this API."
        )
    device_id = 0
    if device is None:
        place = paddle.framework._current_expected_place_()
        if isinstance(place, core.MPSPlace):
            device_id = place.get_device_id()
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.MPSPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(f"Invalid device type: {type(device)}")
    
    # Create a simple device properties object
    # MPS doesn't have compute capability like CUDA, so we use 0, 0
    name = core.get_mps_device_name(device_id)
    total_memory = core.get_mps_device_total_memory(device_id)
    
    # Return a simple object with device properties
    class MPSDeviceProperties:
        def __init__(self, name, total_memory):
            self.name = name
            self.total_memory = total_memory
            self.major = 0  # MPS doesn't have compute capability
            self.minor = 0
            self.multi_processor_count = 0  # Not applicable for MPS
    
    return MPSDeviceProperties(name, total_memory)


def get_device_name(device: MPSPlace | int | None = None) -> str:
    """
    Get the name of an MPS device.

    Args:
        device (MPSPlace | int | None, optional): The device to query. If None, use the current device.
            Can be MPSPlace, int (device index), or None.

    Returns:
        str: The name of the MPS device.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Get the name of the current MPS device
            >>> name = paddle.mps.get_device_name()
            >>> print(name)

            >>> # Get the name of device mps:0
            >>> name0 = paddle.mps.get_device_name(0)
            >>> print(name0)
    """
    if not paddle.is_compiled_with_mps():
        raise ValueError(
            "The API paddle.mps.get_device_name is not supported in "
            "PaddlePaddle without MPS support. Please reinstall PaddlePaddle with MPS support "
            "to call this API."
        )
    device_id = 0
    if device is None:
        place = paddle.framework._current_expected_place_()
        if isinstance(place, core.MPSPlace):
            device_id = place.get_device_id()
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, core.MPSPlace):
        device_id = device.get_device_id()
    else:
        raise ValueError(f"Invalid device type: {type(device)}")
    return core.get_mps_device_name(device_id)


def get_rng_state(device: MPSPlace | int | None = None) -> core.GeneratorState:
    """
    Return the random number generator state of the specified device.

    Args:
        device (MPSPlace | int | None, optional): The device to retrieve the RNG state from.
            If not specified, uses the current default device.
            Can be a device object, integer device ID, or None.

    Returns:
        core.GeneratorState: The current RNG state of the specified device.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.mps.get_rng_state()
    """
    if device is None:
        place = paddle.framework._current_expected_place_()
        if isinstance(place, core.MPSPlace):
            device = place
        else:
            device = core.MPSPlace(0)
    elif isinstance(device, int):
        device = core.MPSPlace(device)
    elif not isinstance(device, core.MPSPlace):
        raise ValueError(f"Invalid device type: {type(device)}")
    
    return paddle.device.get_rng_state(device)


def set_rng_state(
    new_state: core.GeneratorState, device: MPSPlace | int | None = None
) -> None:
    """
    Set the random number generator state of the specified device.

    Args:
        new_state (core.GeneratorState): The desired RNG state to set.
            This should be a state object previously obtained from ``get_rng_state()``.
        device (MPSPlace | int | None, optional): The device to set the RNG state for.
            If not specified, uses the current default device.
            Can be a device object, integer device ID, or None.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Save RNG state
            >>> state = paddle.mps.get_rng_state()
            >>> # Do some random operations
            >>> x = paddle.randn([2, 3])
            >>> # Restore RNG state
            >>> paddle.mps.set_rng_state(state)
    """
    if device is None:
        place = paddle.framework._current_expected_place_()
        if isinstance(place, core.MPSPlace):
            device = place
        else:
            device = core.MPSPlace(0)
    elif isinstance(device, int):
        device = core.MPSPlace(device)
    elif not isinstance(device, core.MPSPlace):
        raise ValueError(f"Invalid device type: {type(device)}")
    
    paddle.device.set_rng_state(new_state, device)


def is_initialized() -> bool:
    """
    Return whether MPS has been initialized.

    Returns:
        bool: True if MPS is compiled and available, False otherwise.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> initialized = paddle.mps.is_initialized()
            >>> print(f"MPS initialized: {initialized}")
    """
    return paddle.is_compiled_with_mps() and is_available()


__all__ = [
    'is_available',
    'device_count',
    'synchronize',
    'set_device',
    'current_device',
    'manual_seed',
    'manual_seed_all',
    'empty_cache',
    'memory_allocated',
    'memory_reserved',
    'max_memory_allocated',
    'max_memory_reserved',
    'reset_max_memory_allocated',
    'reset_max_memory_reserved',
    'get_device_properties',
    'get_device_name',
    'get_rng_state',
    'set_rng_state',
    'is_initialized',
    'device',
    'device_guard',
]

