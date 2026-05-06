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

import unittest

import numpy as np

import paddle


class TestFloorOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.uniform(-10, 10, [3, 4]).astype(np.float32)
        self.test_types = ["decorator", "out", "out_decorator"]

    def do_test(self, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        if test_type == 'raw':
            result = paddle.floor(x)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'decorator':
            result = paddle.floor(input=x)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'out':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.floor(x, out=out)
            out.mean().backward()
            return out, x.grad
        elif test_type == 'out_decorator':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.floor(input=x, out=out)
            out.mean().backward()
            return out, x.grad
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def test_all(self):
        out_std, grad_std = self.do_test('raw')
        for test_type in self.test_types:
            out, grad = self.do_test(test_type)
            np.testing.assert_allclose(out.numpy(), out_std.numpy(), rtol=1e-20)
            np.testing.assert_allclose(
                grad.numpy(), grad_std.numpy(), rtol=1e-20
            )


def _mps_available():
    return (
        hasattr(paddle, "is_compiled_with_mps")
        and paddle.is_compiled_with_mps()
        and getattr(paddle, "mps", None) is not None
        and paddle.mps.is_available()
    )


@unittest.skipUnless(_mps_available(), "Paddle is not built with MPS or MPS is unavailable")
class TestFloorMPS(unittest.TestCase):
    """MPS-backend coverage for paddle.floor.

    Includes an explicit boundary case for negatives + .5 values where
    floor / int-cast-style implementations frequently disagree.
    """

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    def _check(self, x_np):
        out = paddle.floor(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(out, np.floor(x_np), rtol=1e-5, atol=1e-6)

    def test_basic_shapes(self):
        for shape in [(10,), (3, 4), (2, 3, 4)]:
            with self.subTest(shape=shape):
                x = np.random.uniform(-5.0, 5.0, shape).astype(np.float32)
                self._check(x)

    def test_boundary_values(self):
        x = np.array(
            [-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 3.7], dtype=np.float32
        )
        self._check(x)


if __name__ == "__main__":
    unittest.main()
