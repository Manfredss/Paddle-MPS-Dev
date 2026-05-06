#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
from op_test import get_device_place, is_custom_device

import paddle


class TestNegOp(unittest.TestCase):
    def setUp(self):
        self.init_dtype_type()
        self.input = (np.random.random((32, 8)) * 100).astype(self.dtype)

    def init_dtype_type(self):
        self.dtype = np.float64

    def run_imperative(self):
        input = paddle.to_tensor(self.input)
        dy_result = paddle.neg(input)
        expected_result = np.negative(self.input)
        np.testing.assert_allclose(
            dy_result.numpy(), expected_result, rtol=1e-05
        )

    def run_static(self, use_gpu=False):
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data(
                name='input', shape=[32, 8], dtype=self.dtype
            )
            result = paddle.neg(input)

            place = get_device_place() if use_gpu else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            st_result = exe.run(feed={"input": self.input}, fetch_list=[result])
            expected_result = np.negative(self.input)
            np.testing.assert_allclose(
                st_result[0], expected_result, rtol=1e-05
            )

    def test_cpu(self):
        paddle.disable_static(place=paddle.CPUPlace())
        self.run_imperative()
        paddle.enable_static()
        self.run_static()

    def test_gpu(self):
        if not (paddle.base.core.is_compiled_with_cuda() or is_custom_device()):
            return

        paddle.disable_static(place=get_device_place())
        self.run_imperative()
        paddle.enable_static()
        self.run_static(use_gpu=True)


class TestNegOpFp32(TestNegOp):
    def init_dtype_type(self):
        self.dtype = np.float32


class TestNegOpInt64(TestNegOp):
    def init_dtype_type(self):
        self.dtype = np.int64


class TestNegOpInt32(TestNegOp):
    def init_dtype_type(self):
        self.dtype = np.int32


class TestNegOpInt16(TestNegOp):
    def init_dtype_type(self):
        self.dtype = np.int16


class TestNegOpInt8(TestNegOp):
    def init_dtype_type(self):
        self.dtype = np.int8


def _mps_available():
    return (
        hasattr(paddle, "is_compiled_with_mps")
        and paddle.is_compiled_with_mps()
        and getattr(paddle, "mps", None) is not None
        and paddle.mps.is_available()
    )


@unittest.skipUnless(_mps_available(), "Paddle is not built with MPS or MPS is unavailable")
class TestNegOpMPS(unittest.TestCase):
    """MPS-backend coverage for paddle.neg.

    Compares the MPS result against a NumPy ground truth across multiple
    shapes plus a few edge inputs (zero, signed values, single element).
    """

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    def _check(self, x_np, rtol=1e-5, atol=1e-6):
        out = paddle.neg(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(
            out, np.negative(x_np), rtol=rtol, atol=atol,
            err_msg=f"shape={x_np.shape}",
        )

    def test_basic_shapes(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 2, 3, 4)]:
            with self.subTest(shape=shape):
                x = np.random.uniform(-5.0, 5.0, shape).astype(np.float32)
                self._check(x)

    def test_zero_and_signed_inputs(self):
        self._check(np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32))

    def test_single_element(self):
        self._check(np.array([2.5], dtype=np.float32))

    def test_large_tensor(self):
        x = np.random.uniform(-10.0, 10.0, (128, 128)).astype(np.float32)
        self._check(x)


if __name__ == "__main__":
    unittest.main()
