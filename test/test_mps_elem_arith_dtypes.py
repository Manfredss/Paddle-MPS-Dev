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
Dtype coverage for the MPS 'elem-arith' kernel family.

Family ops (all integer-capable in MPSGraph):
    add, subtract, multiply, maximum, minimum

This suite exercises the newly added dtypes:
  * float16  -> every op, LOOSE tolerance, compared to a float32 numpy oracle
                (NOT compared to the CPU backend, which may not register fp16).
  * int32    -> every op, compared EXACTLY to numpy AND the CPU backend.
  * int64    -> every op, compared EXACTLY to numpy AND the CPU backend.

float32 is already covered by test_mps_elementwise_kernels.py; we focus on the
non-float32 dtypes here.
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


# Each op tuple: (name, paddle fn, numpy oracle fn)
_OPS = (
    ("add", lambda a, b: paddle.add(a, b), np.add),
    ("subtract", lambda a, b: paddle.subtract(a, b), np.subtract),
    ("multiply", lambda a, b: paddle.multiply(a, b), np.multiply),
    ("maximum", lambda a, b: paddle.maximum(a, b), np.maximum),
    ("minimum", lambda a, b: paddle.minimum(a, b), np.minimum),
)

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class TestMPSElemArithDtypes(unittest.TestCase):
    """float16 / int32 / int64 coverage for add/subtract/multiply/maximum/minimum."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)

    # ------------------------------------------------------------------
    # float16: loose-tolerance vs float32 numpy oracle. We deliberately do
    # NOT compare against the CPU backend because CPU may not register
    # float16 for these ops and would fail to dispatch.
    # ------------------------------------------------------------------
    def _float16_check(
        self, name, paddle_op, numpy_op, x32, y32, rtol=2e-2, atol=2e-2
    ):
        x16 = x32.astype(np.float16)
        y16 = y32.astype(np.float16)

        out = paddle_op(
            paddle.to_tensor(x16, place="mps"),
            paddle.to_tensor(y16, place="mps"),
        )
        # dtype / place assertions on the live tensor.
        self.assertEqual(
            out.dtype, paddle.float16, f"{name} float16 output dtype mismatch"
        )
        self.assertTrue(
            "mps" in str(out.place).lower(), f"{name} float16 output not on mps"
        )

        out_np = out.numpy().astype(np.float32)
        # Oracle computed in float32 from the float16-rounded inputs.
        ref = numpy_op(x16.astype(np.float32), y16.astype(np.float32))
        np.testing.assert_allclose(
            out_np,
            ref,
            rtol=rtol,
            atol=atol,
            err_msg=f"{name} float16 vs float32 numpy oracle",
        )

    def test_float16_all_ops(self):
        for shape in _SHAPES:
            # Modest range to keep float16 rounding error bounded.
            x32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
            y32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
            for name, p_op, n_op in _OPS:
                with self.subTest(op=name, dtype="float16", shape=shape):
                    self._float16_check(name, p_op, n_op, x32, y32)

    def test_float16_broadcast(self):
        # Broadcasting must also preserve the float16 dtype.
        x32 = (np.random.randn(2, 3, 4) * 1.5).astype(np.float32)
        y32 = (np.random.randn(3, 4) * 1.5).astype(np.float32)
        for name, p_op, n_op in _OPS:
            with self.subTest(op=name, dtype="float16", case="broadcast"):
                self._float16_check(name, p_op, n_op, x32, y32)

    # ------------------------------------------------------------------
    # integer dtypes: EXACT comparison vs numpy AND the CPU backend.
    # ------------------------------------------------------------------
    def _int_check(self, name, paddle_op, numpy_op, x, y, paddle_dtype):
        out = paddle_op(
            paddle.to_tensor(x, place="mps"),
            paddle.to_tensor(y, place="mps"),
        )
        self.assertEqual(
            out.dtype, paddle_dtype, f"{name} {x.dtype} output dtype mismatch"
        )
        self.assertTrue(
            "mps" in str(out.place).lower(),
            f"{name} {x.dtype} output not on mps",
        )

        out_mps = out.numpy()
        out_cpu = paddle_op(
            paddle.to_tensor(x, place="cpu"),
            paddle.to_tensor(y, place="cpu"),
        ).numpy()
        ref = numpy_op(x, y)
        np.testing.assert_array_equal(
            out_mps, ref, err_msg=f"{name} {x.dtype} vs numpy"
        )
        np.testing.assert_array_equal(
            out_mps, out_cpu, err_msg=f"{name} {x.dtype} vs cpu"
        )

    def test_int32_all_ops(self):
        for shape in _SHAPES:
            x = np.random.randint(-8, 9, size=shape).astype(np.int32)
            y = np.random.randint(-8, 9, size=shape).astype(np.int32)
            for name, p_op, n_op in _OPS:
                with self.subTest(op=name, dtype="int32", shape=shape):
                    self._int_check(name, p_op, n_op, x, y, paddle.int32)

    def test_int64_all_ops(self):
        for shape in _SHAPES:
            x = np.random.randint(-8, 9, size=shape).astype(np.int64)
            y = np.random.randint(-8, 9, size=shape).astype(np.int64)
            for name, p_op, n_op in _OPS:
                with self.subTest(op=name, dtype="int64", shape=shape):
                    self._int_check(name, p_op, n_op, x, y, paddle.int64)

    def test_int32_broadcast(self):
        x = np.random.randint(-8, 9, size=(2, 3, 4)).astype(np.int32)
        y = np.random.randint(-8, 9, size=(3, 4)).astype(np.int32)
        for name, p_op, n_op in _OPS:
            with self.subTest(op=name, dtype="int32", case="broadcast"):
                self._int_check(name, p_op, n_op, x, y, paddle.int32)

    def test_int_known_values(self):
        # Hand-checked small example exercising signed integer behavior.
        x = np.array([-3, 0, 5, 7], dtype=np.int32)
        y = np.array([2, 0, -5, 4], dtype=np.int32)
        x_p = paddle.to_tensor(x, place="mps")
        y_p = paddle.to_tensor(y, place="mps")
        np.testing.assert_array_equal(
            paddle.add(x_p, y_p).numpy(),
            np.array([-1, 0, 0, 11], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            paddle.subtract(x_p, y_p).numpy(),
            np.array([-5, 0, 10, 3], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            paddle.multiply(x_p, y_p).numpy(),
            np.array([-6, 0, -25, 28], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            paddle.maximum(x_p, y_p).numpy(),
            np.array([2, 0, 5, 7], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            paddle.minimum(x_p, y_p).numpy(),
            np.array([-3, 0, -5, 4], dtype=np.int32),
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
