#!/usr/bin/env python3
"""
Test script to check MPS availability after installation.
Run this after installing the compiled PaddlePaddle wheel.
"""

import sys

def test_mps_availability():
    """Test if MPS is available and working."""
    print("=" * 60)
    print("Testing MPS Availability")
    print("=" * 60)
    
    try:
        import paddle
        print(f"✓ PaddlePaddle imported successfully")
        print(f"  Version: {paddle.__version__ if hasattr(paddle, '__version__') else 'Unknown'}")
    except ImportError as e:
        print(f"✗ Failed to import PaddlePaddle: {e}")
        return False
    
    # Check if MPS is available (PyTorch-style API)
    try:
        is_available = paddle.mps.is_available()
        print(f"\n✓ paddle.mps.is_available() available")
        print(f"  MPS Available: {is_available}")
        
        if not is_available:
            print("\n⚠ Warning: MPS is not available.")
            print("  Make sure you're running on Apple Silicon (M1/M2/M3) with macOS 12.0+")
            print("  and that PaddlePaddle was compiled with MPS support (-DWITH_MPS=ON)")
            return False
    except AttributeError as e:
        print(f"✗ paddle.mps.is_available() not available: {e}")
        return False
    
    # Check MPS device count
    try:
        device_count = paddle.mps.device_count()
        print(f"\n✓ paddle.mps.device_count() available")
        print(f"  MPS Device Count: {device_count}")
        
        if device_count == 0:
            print("\n⚠ Warning: No MPS devices detected.")
            print("  Make sure you're running on Apple Silicon (M1/M2/M3) with macOS 12.0+")
            return False
    except AttributeError as e:
        print(f"✗ paddle.mps.device_count() not available: {e}")
        return False
    
    # Check MPSPlace
    try:
        mps_place = paddle.MPSPlace(0)
        print(f"\n✓ MPSPlace available")
        print(f"  MPSPlace(0): {mps_place}")
        # Try different ways to get device ID
        try:
            device_id = mps_place.get_device_id()
            print(f"  Device ID (get_device_id): {device_id}")
        except:
            try:
                device_id = mps_place.GetDeviceId()
                print(f"  Device ID (GetDeviceId): {device_id}")
            except:
                try:
                    device_id = paddle.base.core.mps_device_id(mps_place)
                    print(f"  Device ID (mps_device_id): {device_id}")
                except:
                    print(f"  Device ID: Unable to retrieve")
    except Exception as e:
        print(f"✗ Failed to create MPSPlace: {e}")
        return False
    
    # Try to create a tensor on MPS
    try:
        print(f"\n✓ Testing tensor creation on MPS...")
        paddle.mps.set_device(0)  # Use PyTorch-style API
        x = paddle.randn([2, 3], dtype='float32')
        print(f"  Created tensor on MPS: {x.place}")
        print(f"  Tensor shape: {x.shape}")
        print(f"  Tensor dtype: {x.dtype}")
        
        # Try a simple operation
        y = x + 1.0
        print(f"  Performed operation: x + 1.0")
        print(f"  Result shape: {y.shape}")
        
        # Test synchronize
        paddle.mps.synchronize()
        print(f"  Synchronized MPS operations")
        
    except Exception as e:
        print(f"✗ Failed to create tensor on MPS: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All MPS tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_mps_availability()
    sys.exit(0 if success else 1)

