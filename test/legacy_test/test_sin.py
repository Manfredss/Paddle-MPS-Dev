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


class TestSinOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.rand(3, 4).astype(np.float32)
        self.test_types = ["decorator", "out", "out_decorator"]

    def do_test(self, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        if test_type == 'raw':
            result = paddle.sin(x)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'decorator':
            result = paddle.sin(input=x)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'out':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.sin(x, out=out)
            out.mean().backward()
            return out, x.grad
        elif test_type == 'out_decorator':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.sin(input=x, out=out)
            out.mean().backward()
            return out, x.grad
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def test_all(self):
        out_std, grad_std = self.do_test('raw')
        for test_type in self.test_types:
            out, grad = self.do_test(test_type)
            np.testing.assert_allclose(out.numpy(), out_std.numpy(), rtol=1e-7)
            np.testing.assert_allclose(
                grad.numpy(), grad_std.numpy(), rtol=1e-7
            )


def _mps_available():
    return (
        hasattr(paddle, "is_compiled_with_mps")
        and paddle.is_compiled_with_mps()
        and getattr(paddle, "mps", None) is not None
        and paddle.mps.is_available()
    )


@unittest.skipUnless(_mps_available(), "Paddle is not built with MPS or MPS is unavailable")
class TestSinMPS(unittest.TestCase):
    """MPS-backend coverage for paddle.sin."""

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    def _check(self, x_np, atol=1e-6):
        out = paddle.sin(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(out, np.sin(x_np), rtol=1e-5, atol=atol)

    def test_basic_shapes(self):
        for shape in [(7,), (3, 4), (2, 3, 5)]:
            with self.subTest(shape=shape):
                x = np.random.uniform(-np.pi, np.pi, shape).astype(np.float32)
                self._check(x)

    def test_known_values(self):
        x = np.array(
            [0.0, np.pi / 6, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2],
            dtype=np.float32,
        )
        self._check(x, atol=1e-5)

    def test_pythagorean_identity(self):
        x = np.random.uniform(-np.pi, np.pi, (4, 5)).astype(np.float32)
        x_p = paddle.to_tensor(x, place="mps")
        identity = (paddle.sin(x_p) ** 2 + paddle.cos(x_p) ** 2).numpy()
        np.testing.assert_allclose(
            identity, np.ones_like(identity), rtol=1e-5, atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
