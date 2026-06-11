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
Dtype coverage for the MPS 'rounding' kernel family: floor, ceil, round, trunc.

These kernels now register float16 in addition to float32. MPSGraph
floor/ceil/round/truncate are float-only ops, so this family is float16-only
(no integer coverage).

For float16 we run the op on an MPS tensor and compare the result (cast back to
float32) against a float32 numpy reference with LOOSE tolerance. We do NOT
compare against the CPU backend in float16, because CPU may not register
float16 for these ops and would fail to dispatch.

round() uses banker's rounding (round-half-to-even) on MPSGraph, so the round
test deliberately avoids values near exact .5 in float16.
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


class _MPSRoundingDtypeTestBase(unittest.TestCase):
    """Common setUp for the MPS rounding-family dtype tests."""

    @classmethod
    def setUpClass(cls):
        if not _mps_available():
            raise unittest.SkipTest(
                "PaddlePaddle is not built with MPS or MPS is unavailable"
            )
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)


# Each op: (name, paddle fn, numpy oracle).
# numpy round-half-to-even matches MPSGraph's roundWithTensor (banker rounding).
_ROUNDING_OPS = (
    ("floor", lambda x: paddle.floor(x), np.floor),
    ("ceil", lambda x: paddle.ceil(x), np.ceil),
    ("round", lambda x: paddle.round(x), np.round),
    ("trunc", lambda x: paddle.trunc(x), np.trunc),
)

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


class TestMPSRoundingFloat16(_MPSRoundingDtypeTestBase):
    """float16 coverage for floor / ceil / round / trunc."""

    def _make_inputs(self, shape, op_name):
        """Build a float32 input within a modest range and its float16 copy.

        For 'round' we avoid values near exact .5 so the banker-rounding
        tie-break (and float16 quantization) never makes the comparison
        ambiguous: we draw values and snap any that land too close to a
        half-integer away from it.
        """
        x32 = (np.random.randn(*shape).astype(np.float32)) * 1.5
        if op_name == "round":
            # Shift any value whose fractional part is near +/-0.5 away from
            # the tie so both numpy and MPS agree regardless of tie-breaking.
            frac = x32 - np.floor(x32)
            near_half = np.abs(frac - 0.5) < 0.1
            x32 = np.where(near_half, x32 + 0.2, x32).astype(np.float32)
        x16 = x32.astype(np.float16)
        return x32, x16

    def _check(self, op_name, paddle_op, numpy_op, shape):
        x32, x16 = self._make_inputs(shape, op_name)
        # Reference computed in float32 on the float16-rounded input values so
        # the oracle sees exactly the magnitudes the MPS kernel sees.
        ref = numpy_op(x16.astype(np.float32)).astype(np.float32)

        out = paddle_op(paddle.to_tensor(x16, place="mps"))
        out_np = out.numpy().astype(np.float32)

        np.testing.assert_allclose(
            out_np,
            ref,
            rtol=2e-2,
            atol=2e-2,
            err_msg=f"{op_name} float16 vs numpy (shape={shape})",
        )
        self.assertEqual(
            out.dtype,
            paddle.float16,
            f"{op_name} float16 output dtype must be float16",
        )
        self.assertTrue(
            "mps" in str(out.place).lower(),
            f"{op_name} float16 output must live on mps",
        )

    def test_float16_shapes(self):
        for name, p_op, n_op in _ROUNDING_OPS:
            for shape in _SHAPES:
                with self.subTest(op=name, shape=shape):
                    self._check(name, p_op, n_op, shape)

    def test_float16_known_values(self):
        # Unambiguous values (no exact halves) representable in float16.
        x = np.array([-2.25, -0.75, 0.25, 1.75, 3.5, -3.5], dtype=np.float16)
        expected = {
            # floor / ceil / trunc are unambiguous for these.
            "floor": np.floor(x.astype(np.float32)),
            "ceil": np.ceil(x.astype(np.float32)),
            "trunc": np.trunc(x.astype(np.float32)),
            # round uses banker rounding: 3.5->4, -3.5->-4 (round half to even).
            "round": np.round(x.astype(np.float32)),
        }
        for name, p_op, _ in _ROUNDING_OPS:
            with self.subTest(op=name):
                out = p_op(paddle.to_tensor(x, place="mps"))
                out_np = out.numpy().astype(np.float32)
                np.testing.assert_allclose(
                    out_np,
                    expected[name],
                    rtol=2e-2,
                    atol=2e-2,
                    err_msg=f"{name} known float16 values",
                )
                self.assertEqual(out.dtype, paddle.float16)

    def test_float16_dtype_and_place(self):
        x = np.random.randn(3, 4).astype(np.float16)
        for name, p_op, _ in _ROUNDING_OPS:
            with self.subTest(op=name):
                out = p_op(paddle.to_tensor(x, place="mps"))
                self.assertEqual(out.dtype, paddle.float16)
                self.assertTrue("mps" in str(out.place).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
