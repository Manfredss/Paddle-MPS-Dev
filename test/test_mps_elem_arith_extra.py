#!/usr/bin/env python3


# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

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


"""
Extra dtype coverage for the MPS "elem-arith" kernel family.

Family / dtype policy (this file only exercises the *newly added* dtypes):
- add / subtract / multiply : gained bfloat16 AND complex64.
- maximum / minimum         : gained bfloat16 only (complex has no ordering).

complex128 is intentionally NOT covered: the MPS backend's GetMPSDataType has
no COMPLEX128 -> MPSDataType mapping, so complex128 is not registered for any
MPS elem-arith kernel (only complex64 -> MPSDataTypeComplexFloat32 exists).

bfloat16 and complex64 map to MPSGraph data types that only exist in the
macOS-14 MetalPerformanceShadersGraph SDK and require macOS 14 AT RUNTIME. This
CI host may lack that, so every bf16/complex subtest is gated behind a
probe-and-skip helper (``_supports``) that runs a tiny op on an mps tensor of
the dtype and returns False on any exception.

bf16 results (~3 decimal digits, looser than fp16) are compared to a float32
numpy oracle with LOOSE tolerance. complex64 results are compared to the numpy
complex result. The pre-existing int32/int64 registrations are unchanged by
this task and are not re-tested here.
"""

import unittest

import numpy as np

try:
    import paddle

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddlePaddle not available")


def _mps_available():
    return (
        PADDLE_AVAILABLE
        and paddle.is_compiled_with_mps()
        and paddle.mps.is_available()
    )


def _supports(paddle_dtype):
    """Return True iff a trivial op on an mps tensor of ``paddle_dtype`` works.

    bfloat16/complex64 need the macOS-14 SDK *and* runtime. We can't tell from
    Python whether the SDK was present at build time, so we just try a tiny
    add on an mps tensor of the dtype and treat ANY exception as unsupported.
    """
    if not _mps_available():
        return False
    try:
        base = np.zeros((2,), dtype=np.float32)
        t = paddle.to_tensor(base, place="mps").astype(paddle_dtype)
        out = paddle.add(t, t)
        # Force materialization so lazy failures surface here.
        _ = out.numpy()
        return True
    except Exception:
        return False


# bf16 has ~3 decimal digits of precision; tolerances are looser than fp16.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2
# complex64 is exact-ish float32 arithmetic.
C64_RTOL = 1e-4

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSElemArithBase(unittest.TestCase):
    """Common setUp + helpers for the elem-arith extra-dtype tests."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        cls.bf16_ok = _supports(paddle.bfloat16)
        cls.c64_ok = _supports(paddle.complex64)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    def _require_bf16(self):
        if not self.bf16_ok:
            self.skipTest(
                "bfloat16 unsupported on this MPS host (needs macOS 14)"
            )

    def _require_c64(self):
        if not self.c64_ok:
            self.skipTest(
                "complex64 unsupported on this MPS host (needs macOS 14)"
            )

    # -- bf16 helper ------------------------------------------------------
    def _run_binary_bf16(self, fn, x32, y32):
        """Run binary ``fn`` on bf16 mps tensors; compare to float32 oracle.

        bf16 has no numpy dtype, so we build a float32 mps tensor and cast it
        to paddle.bfloat16 on device.
        """
        self._require_bf16()
        x_bf16 = paddle.to_tensor(x32, place="mps").astype("bfloat16")
        y_bf16 = paddle.to_tensor(y32, place="mps").astype("bfloat16")
        out = fn(x_bf16, y_bf16)
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())
        ref = (
            fn(
                paddle.to_tensor(x32, place="cpu"),
                paddle.to_tensor(y32, place="cpu"),
            )
            .numpy()
            .astype(np.float32)
        )
        np.testing.assert_allclose(
            out.astype("float32").numpy(),
            ref,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )
        return out

    # -- complex64 helper -------------------------------------------------
    def _run_binary_c64(self, paddle_fn, np_fn, x, y):
        """Run binary ``paddle_fn`` on complex64 mps tensors vs numpy oracle."""
        self._require_c64()
        out = paddle_fn(
            paddle.to_tensor(x, place="mps"),
            paddle.to_tensor(y, place="mps"),
        )
        self.assertEqual(out.dtype, paddle.complex64)
        self.assertTrue("mps" in str(out.place).lower())
        ref = np_fn(x, y).astype(np.complex64)
        np.testing.assert_allclose(out.numpy(), ref, rtol=C64_RTOL)
        return out


# ---------------------------------------------------------------------------
# add / subtract / multiply: bfloat16 + complex64.
# ---------------------------------------------------------------------------


class TestMPSAddExtraDtypes(_MPSElemArithBase):
    def test_add_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                y = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_binary_bf16(paddle.add, x, y)

    def test_add_c64(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (
                    np.random.randn(*shape) + 1j * np.random.randn(*shape)
                ).astype(np.complex64)
                y = (
                    np.random.randn(*shape) + 1j * np.random.randn(*shape)
                ).astype(np.complex64)
                self._run_binary_c64(paddle.add, np.add, x, y)


class TestMPSSubtractExtraDtypes(_MPSElemArithBase):
    def test_subtract_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                y = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_binary_bf16(paddle.subtract, x, y)

    def test_subtract_c64(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (
                    np.random.randn(*shape) + 1j * np.random.randn(*shape)
                ).astype(np.complex64)
                y = (
                    np.random.randn(*shape) + 1j * np.random.randn(*shape)
                ).astype(np.complex64)
                self._run_binary_c64(paddle.subtract, np.subtract, x, y)


class TestMPSMultiplyExtraDtypes(_MPSElemArithBase):
    def test_multiply_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                y = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_binary_bf16(paddle.multiply, x, y)

    def test_multiply_c64(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (
                    np.random.randn(*shape) + 1j * np.random.randn(*shape)
                ).astype(np.complex64)
                y = (
                    np.random.randn(*shape) + 1j * np.random.randn(*shape)
                ).astype(np.complex64)
                self._run_binary_c64(paddle.multiply, np.multiply, x, y)


# ---------------------------------------------------------------------------
# maximum / minimum: bfloat16 only (no complex - ordering is undefined).
# ---------------------------------------------------------------------------


class TestMPSMaximumExtraDtypes(_MPSElemArithBase):
    def test_maximum_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                y = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_binary_bf16(paddle.maximum, x, y)


class TestMPSMinimumExtraDtypes(_MPSElemArithBase):
    def test_minimum_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                y = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_binary_bf16(paddle.minimum, x, y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
