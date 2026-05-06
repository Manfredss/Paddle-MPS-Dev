#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base


def get_places():
    places = []
    if base.is_compiled_with_cuda() or is_custom_device():
        places.append(get_device_place())
    places.append(paddle.CPUPlace())
    return places


class TestCeilAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.places = get_places()
        self.shape = [50]
        self.dtype = "float64"
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        paddle_dygraph_out = []
        # Numpy reference output
        ref_out = np.ceil(self.np_x)
        # Position args (args)
        out1 = paddle.ceil(x)
        paddle_dygraph_out.append(out1)
        # Keywords args (kwargs) for paddle
        out2 = paddle.ceil(x=x)
        paddle_dygraph_out.append(out2)
        # Keywords args for torch compatibility
        out3 = paddle.ceil(input=x)
        paddle_dygraph_out.append(out3)
        # Tensor method args
        out4 = x.ceil()
        paddle_dygraph_out.append(out4)
        # Test 'out' parameter for torch compatibility
        out5 = paddle.empty(ref_out.shape, dtype=x.dtype)
        paddle.ceil(x, out=out5)
        paddle_dygraph_out.append(out5)
        # Check all dygraph results
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            # Define static data placeholders
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            # Position args (args)
            out1 = paddle.ceil(x)
            # Keywords args (kwargs) for paddle
            out2 = paddle.ceil(x=x)
            # Keywords args for torch compatibility
            out3 = paddle.ceil(input=x)
            # Tensor method args
            out4 = x.ceil()
            # Numpy reference output
            ref_out = np.ceil(self.np_x)
            fetch_list = [out1, out2, out3, out4]
            for place in self.places:
                exe = base.Executor(place)
                fetches = exe.run(
                    main,
                    feed={"x": self.np_x},
                    fetch_list=fetch_list,
                )
                for out in fetches:
                    np.testing.assert_allclose(out, ref_out, rtol=1e-05)


def _mps_available():
    return (
        hasattr(paddle, "is_compiled_with_mps")
        and paddle.is_compiled_with_mps()
        and getattr(paddle, "mps", None) is not None
        and paddle.mps.is_available()
    )


@unittest.skipUnless(_mps_available(), "Paddle is not built with MPS or MPS is unavailable")
class TestCeilMPS(unittest.TestCase):
    """MPS-backend coverage for paddle.ceil."""

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    def _check(self, x_np):
        out = paddle.ceil(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(out, np.ceil(x_np), rtol=1e-5, atol=1e-6)

    def test_basic_shapes(self):
        for shape in [(10,), (3, 4), (2, 3, 4)]:
            with self.subTest(shape=shape):
                x = np.random.uniform(-5.0, 5.0, shape).astype(np.float32)
                self._check(x)

    def test_boundary_values(self):
        x = np.array(
            [-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 3.2], dtype=np.float32
        )
        self._check(x)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
