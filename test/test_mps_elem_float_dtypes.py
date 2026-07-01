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
dtype coverage for the MPS 'elem-float' kernel family.

Family ops (float16 only -- no integer tests):
- divide          (paddle.divide,         two tensors)
- elementwise_pow (paddle.pow / **,       two tensors)
- pow             (paddle.pow w/ scalar,  tensor ^ scalar)

Each op is exercised in float16 on the MPS backend; the result is cast back to
float32 and compared against a float32 numpy/oracle reference with loose
tolerance.  We deliberately do NOT compare against the CPU backend in float16,
because CPU may not register float16 for these ops and would fail to dispatch.
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


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class TestMPSElemFloatDtypes(unittest.TestCase):
    """float16 coverage for divide / elementwise_pow / pow on MPS."""

    # Loose tolerances for float16; pow/large-magnitude use a wider band.
    RTOL = 2e-2
    ATOL = 2e-2
    RTOL_POW = 5e-2
    ATOL_POW = 5e-2

    SHAPES = [(6,), (3, 4), (2, 3, 4)]

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    # ---- helpers --------------------------------------------------------
    def _run_f16(self, op, *arrays_f32):
        """Run `op` on float16 MPS tensors built from the given f32 arrays.

        Returns (result_f32_numpy, paddle_out_tensor).
        """
        tensors = [
            paddle.to_tensor(a.astype(np.float16), place="mps")
            for a in arrays_f32
        ]
        out = op(*tensors)
        return out.numpy().astype(np.float32), out

    def _assert_f16_out(self, out):
        self.assertEqual(out.dtype, paddle.float16)
        self.assertTrue("mps" in str(out.place).lower())

    # ---- divide ---------------------------------------------------------
    def test_divide_float16(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x32 = np.random.randn(*shape).astype(np.float32) * 1.5
                # Keep the divisor away from zero to avoid blow-ups in f16.
                y32 = (np.random.randn(*shape).astype(np.float32) * 1.5) + 2.0
                # Reference uses the *rounded-to-f16* inputs so the oracle and
                # the kernel see the same operands.
                xr = x32.astype(np.float16).astype(np.float32)
                yr = y32.astype(np.float16).astype(np.float32)
                ref = xr / yr

                got, out = self._run_f16(paddle.divide, x32, y32)
                np.testing.assert_allclose(
                    got,
                    ref,
                    rtol=self.RTOL,
                    atol=self.ATOL,
                    err_msg=f"divide float16 shape={shape}",
                )
                self._assert_f16_out(out)

    # ---- elementwise_pow (two tensors) ----------------------------------
    def test_elementwise_pow_float16(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                # Positive base, small exponents -> well-behaved in f16.
                x32 = (
                    np.abs(np.random.randn(*shape).astype(np.float32) * 1.5)
                    + 0.5
                )
                y32 = np.random.uniform(0.5, 2.0, size=shape).astype(np.float32)
                xr = x32.astype(np.float16).astype(np.float32)
                yr = y32.astype(np.float16).astype(np.float32)
                ref = np.power(xr, yr)

                got, out = self._run_f16(paddle.pow, x32, y32)
                np.testing.assert_allclose(
                    got,
                    ref,
                    rtol=self.RTOL_POW,
                    atol=self.ATOL_POW,
                    err_msg=f"elementwise_pow float16 shape={shape}",
                )
                self._assert_f16_out(out)

    # ---- pow (tensor ^ scalar) ------------------------------------------
    def test_pow_scalar_float16(self):
        for shape in self.SHAPES:
            for factor in (2.0, 3.0, 0.5):
                with self.subTest(shape=shape, factor=factor):
                    # Positive base so non-integer exponents stay real-valued.
                    x32 = (
                        np.abs(np.random.randn(*shape).astype(np.float32) * 1.5)
                        + 0.5
                    )
                    xr = x32.astype(np.float16).astype(np.float32)
                    ref = np.power(xr, factor)

                    x16 = paddle.to_tensor(x32.astype(np.float16), place="mps")
                    out = paddle.pow(x16, factor)
                    got = out.numpy().astype(np.float32)
                    np.testing.assert_allclose(
                        got,
                        ref,
                        rtol=self.RTOL_POW,
                        atol=self.ATOL_POW,
                        err_msg=f"pow scalar float16 shape={shape} factor={factor}",
                    )
                    self._assert_f16_out(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
