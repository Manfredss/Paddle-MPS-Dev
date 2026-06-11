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
Dtype coverage for the MPS comparison kernel family.

Ops: equal, not_equal, less_than, less_equal, greater_than, greater_equal.
These take real (non-bool) inputs and produce a BOOL output.

This file complements test_mps_elementwise_kernels.py (which covers float32)
by exercising the newly registered float16 / int32 / int64 input dtypes:

- float16: run on MPS, compare the result (cast nothing -- output is bool) to a
  float32 numpy oracle with LOOSE tolerance. We do NOT compare against the CPU
  backend in float16 because CPU may not dispatch float16 for every op.
- int32 / int64: compare the MPS result EXACTLY against both a numpy integer
  reference AND the CPU backend (which supports integer inputs). The comparison
  output is always paddle.bool.
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


# (name, paddle op, numpy op)
_OPS = (
    ("equal", lambda a, b: paddle.equal(a, b), np.equal),
    ("not_equal", lambda a, b: paddle.not_equal(a, b), np.not_equal),
    ("less_than", lambda a, b: paddle.less_than(a, b), np.less),
    ("less_equal", lambda a, b: paddle.less_equal(a, b), np.less_equal),
    ("greater_than", lambda a, b: paddle.greater_than(a, b), np.greater),
    (
        "greater_equal",
        lambda a, b: paddle.greater_equal(a, b),
        np.greater_equal,
    ),
)

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


class _MPSKernelTestBase(unittest.TestCase):
    """Common setUp for MPS kernel tests."""

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


class TestMPSComparisonFloat16(_MPSKernelTestBase):
    """float16 coverage for the comparison family."""

    def _check_fp16(self, name, paddle_op, numpy_op, x32, y32):
        x16 = x32.astype(np.float16)
        y16 = y32.astype(np.float16)
        out_mps = paddle_op(
            paddle.to_tensor(x16, place="mps"),
            paddle.to_tensor(y16, place="mps"),
        )
        # The comparison reference is computed on the float16 values promoted
        # back to float32 so it matches the rounding the MPS kernel observes.
        ref = numpy_op(x16.astype(np.float32), y16.astype(np.float32))
        np.testing.assert_array_equal(
            out_mps.numpy(), ref, err_msg=f"{name} float16 vs numpy"
        )

    def test_shapes(self):
        for shape in _SHAPES:
            x32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
            y32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
            for name, p_op, n_op in _OPS:
                with self.subTest(op=name, shape=shape):
                    self._check_fp16(name, p_op, n_op, x32, y32)

    def test_overlapping_values(self):
        # Share some entries so each op sees <, ==, and > cases.
        x32 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        y32 = np.array([2.0, 2.0, 2.0, 4.0, 4.0], dtype=np.float32)
        for name, p_op, n_op in _OPS:
            with self.subTest(op=name):
                self._check_fp16(name, p_op, n_op, x32, y32)

    def test_output_dtype_and_place(self):
        x16 = (np.random.randn(3, 4) * 1.5).astype(np.float16)
        y16 = (np.random.randn(3, 4) * 1.5).astype(np.float16)
        for name, p_op, _ in _OPS:
            with self.subTest(op=name):
                out = p_op(
                    paddle.to_tensor(x16, place="mps"),
                    paddle.to_tensor(y16, place="mps"),
                )
                self.assertEqual(out.dtype, paddle.bool)
                self.assertTrue("mps" in str(out.place).lower())


class TestMPSComparisonInteger(_MPSKernelTestBase):
    """int32 / int64 coverage for the comparison family.

    These ops are int-capable. Output is always paddle.bool. Compare MPS
    exactly against both numpy and the CPU backend (which supports ints).
    """

    def _check_int(self, name, paddle_op, numpy_op, x_np, y_np, np_dtype):
        x = x_np.astype(np_dtype)
        y = y_np.astype(np_dtype)
        out_mps = paddle_op(
            paddle.to_tensor(x, place="mps"),
            paddle.to_tensor(y, place="mps"),
        )
        out_cpu = paddle_op(
            paddle.to_tensor(x, place="cpu"),
            paddle.to_tensor(y, place="cpu"),
        ).numpy()
        ref = numpy_op(x, y)
        np.testing.assert_array_equal(
            out_mps.numpy(), ref, err_msg=f"{name} {np_dtype} vs numpy"
        )
        np.testing.assert_array_equal(
            out_mps.numpy(), out_cpu, err_msg=f"{name} {np_dtype} vs cpu"
        )
        self.assertEqual(
            out_mps.numpy().dtype,
            np.bool_,
            f"{name} {np_dtype} output dtype must be bool",
        )

    def test_int32_shapes(self):
        for shape in _SHAPES:
            x = np.random.randint(-8, 9, size=shape).astype(np.int32)
            y = np.random.randint(-8, 9, size=shape).astype(np.int32)
            for name, p_op, n_op in _OPS:
                with self.subTest(op=name, shape=shape):
                    self._check_int(name, p_op, n_op, x, y, np.int32)

    def test_int64_shapes(self):
        for shape in _SHAPES:
            x = np.random.randint(-8, 9, size=shape).astype(np.int64)
            y = np.random.randint(-8, 9, size=shape).astype(np.int64)
            for name, p_op, n_op in _OPS:
                with self.subTest(op=name, shape=shape):
                    self._check_int(name, p_op, n_op, x, y, np.int64)

    def test_int_overlapping_values(self):
        # Guarantee a mix of <, ==, > for every op.
        x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        y = np.array([2, 2, 2, 4, 4], dtype=np.int64)
        for np_dtype in (np.int32, np.int64):
            for name, p_op, n_op in _OPS:
                with self.subTest(op=name, dtype=np_dtype):
                    self._check_int(name, p_op, n_op, x, y, np_dtype)

    def test_int_known_truth_table(self):
        x = np.array([1, 2, 3], dtype=np.int32)
        y = np.array([2, 2, 2], dtype=np.int32)
        expected = {
            "equal": [False, True, False],
            "not_equal": [True, False, True],
            "less_than": [True, False, False],
            "less_equal": [True, True, False],
            "greater_than": [False, False, True],
            "greater_equal": [False, True, True],
        }
        for name, p_op, _ in _OPS:
            with self.subTest(op=name):
                out = p_op(
                    paddle.to_tensor(x, place="mps"),
                    paddle.to_tensor(y, place="mps"),
                ).numpy()
                np.testing.assert_array_equal(
                    out, np.array(expected[name], dtype=np.bool_)
                )

    def test_int_output_dtype_and_place(self):
        x = np.random.randint(-8, 9, size=(3, 4)).astype(np.int32)
        y = np.random.randint(-8, 9, size=(3, 4)).astype(np.int32)
        for name, p_op, _ in _OPS:
            with self.subTest(op=name):
                out = p_op(
                    paddle.to_tensor(x, place="mps"),
                    paddle.to_tensor(y, place="mps"),
                )
                self.assertEqual(out.dtype, paddle.bool)
                self.assertTrue("mps" in str(out.place).lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)
