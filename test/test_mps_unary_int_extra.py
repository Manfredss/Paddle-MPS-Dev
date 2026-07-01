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
Extra dtype coverage for the MPS "unary-int" kernel family: abs, sign, negative.

Family / dtype policy (this task ADDS dtypes to existing registrations):
- abs      : float, float16, int32, int64 (existing) + bfloat16 (gated).
- negative : float, float16, int32, int64 (existing) + bfloat16 (gated).
- sign     : float, float16, int32, int64 (existing) + uint8, int8, int16
             (newly added small-int widths) + bfloat16 (gated).

bfloat16 maps to MPSDataTypeBFloat16, which only exists in the macOS-14 MPSGraph
SDK and requires macOS 14 AT RUNTIME. This CI host may lack it, so the bf16
subtests PROBE-AND-SKIP: a trivial bf16 op is attempted in a helper and the test
is skipped on any exception.

bf16 outputs are compared to a float32 numpy oracle with LOOSE tolerance
(bf16 has ~3 decimal digits, looser than fp16). Integer outputs (the new sign
widths) are compared EXACTLY against BOTH a numpy reference AND the CPU backend.
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
    """Return True if a trivial op on an MPS tensor of ``paddle_dtype`` works.

    bfloat16 / complex64 require macOS 14 at runtime. Create a tiny tensor of
    the requested dtype on the mps place and run a trivial add; return False on
    ANY exception so callers can skip gracefully on older hosts.
    """
    if not _mps_available():
        return False
    try:
        if paddle_dtype == paddle.bfloat16:
            base = paddle.to_tensor(
                np.zeros((2,), dtype=np.float32), place="mps"
            ).astype("bfloat16")
        elif paddle_dtype == paddle.complex64:
            base = paddle.to_tensor(
                np.zeros((2,), dtype=np.complex64), place="mps"
            )
        else:
            base = paddle.to_tensor(
                np.zeros((2,), dtype=np.float32), place="mps"
            ).astype(paddle_dtype)
        out = paddle.add(base, base)
        # Force materialization so lazy failures surface here.
        _ = out.numpy()
        return True
    except Exception:
        return False


# Loose tolerances for bfloat16 (~3 decimal digits; looser than fp16).
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSUnaryIntBase(unittest.TestCase):
    """Common setUp for the unary-int extra-dtype tests."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        cls.bf16_ok = _supports(paddle.bfloat16)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    # -- bfloat16 helper --------------------------------------------------
    def _require_bf16(self):
        if not self.bf16_ok:
            self.skipTest(
                "bfloat16 unsupported on this MPS host (needs macOS 14)"
            )

    def _assert_bf16_out(self, out):
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())

    def _run_bf16(self, fn, x32, rtol=BF16_RTOL, atol=BF16_ATOL):
        """Run ``fn`` on a bfloat16 MPS tensor; compare to a float32 oracle.

        bf16 has no numpy dtype, so build float32 on mps then .astype(bfloat16).
        ``fn`` takes a paddle tensor and returns a paddle tensor.
        """
        self._require_bf16()
        x_bf16 = paddle.to_tensor(x32, place="mps").astype("bfloat16")
        out = fn(x_bf16)
        self._assert_bf16_out(out)
        ref = fn(paddle.to_tensor(x32, place="cpu")).numpy().astype(np.float32)
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, rtol=rtol, atol=atol
        )
        return out


# ---------------------------------------------------------------------------
# abs: adds bfloat16.
# ---------------------------------------------------------------------------


class TestMPSAbsExtraDtypes(_MPSUnaryIntBase):
    def test_abs_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.abs(t), x)

    def test_abs_bf16_known_values(self):
        x = np.array([-3.0, -0.5, 0.0, 1.25, 4.0], dtype=np.float32)
        out = self._run_bf16(lambda t: paddle.abs(t), x)
        np.testing.assert_allclose(
            out.astype("float32").numpy(),
            np.abs(x),
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )


# ---------------------------------------------------------------------------
# negative: adds bfloat16.
# ---------------------------------------------------------------------------


class TestMPSNegativeExtraDtypes(_MPSUnaryIntBase):
    def test_negative_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.neg(t), x)

    def test_negative_bf16_known_values(self):
        x = np.array([-3.0, -0.5, 0.0, 1.25, 4.0], dtype=np.float32)
        out = self._run_bf16(lambda t: paddle.neg(t), x)
        np.testing.assert_allclose(
            out.astype("float32").numpy(),
            -x,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )


# ---------------------------------------------------------------------------
# sign: adds bfloat16 + small int widths (uint8, int8, int16).
# ---------------------------------------------------------------------------


class TestMPSSignExtraDtypes(_MPSUnaryIntBase):
    def test_sign_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.sign(t), x)

    def _sign_int(self, np_dtype, paddle_dtype, lo, hi):
        for shape in _SHAPES:
            with self.subTest(shape=shape, dtype=np_dtype):
                x = np.random.randint(lo, hi, size=shape).astype(np_dtype)
                out = paddle.sign(paddle.to_tensor(x, place="mps"))
                cpu = paddle.sign(paddle.to_tensor(x, place="cpu"))
                ref = np.sign(x.astype(np.int64)).astype(np_dtype)
                self.assertEqual(out.dtype, paddle_dtype)
                self.assertTrue("mps" in str(out.place).lower())
                np.testing.assert_array_equal(out.numpy(), ref)
                np.testing.assert_array_equal(out.numpy(), cpu.numpy())

    def test_sign_int8(self):
        # Mixed signs to exercise -1 / 0 / +1.
        self._sign_int(np.int8, paddle.int8, -9, 10)

    def test_sign_int16(self):
        self._sign_int(np.int16, paddle.int16, -9, 10)

    def test_sign_uint8(self):
        # Unsigned: only 0 and +1 are possible; include 0 in the range.
        self._sign_int(np.uint8, paddle.uint8, 0, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
