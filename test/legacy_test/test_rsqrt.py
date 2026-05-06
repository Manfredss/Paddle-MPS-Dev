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


class TestRsqrtOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.uniform(0.1, 1, [3, 4]).astype(np.float32)
        self.test_types = ["decorator", "out", "out_decorator"]

    def do_test(self, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        if test_type == 'raw':
            result = paddle.rsqrt(x)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'decorator':
            result = paddle.rsqrt(input=x)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'out':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.rsqrt(x, out=out)
            out.mean().backward()
            return out, x.grad
        elif test_type == 'out_decorator':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.rsqrt(input=x, out=out)
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
class TestRsqrtMPS(unittest.TestCase):
    """MPS-backend coverage for paddle.rsqrt.

    rsqrt requires positive inputs; we keep the lower bound away from zero so
    the comparison is numerically stable.
    """

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    @staticmethod
    def _np_rsqrt(x):
        return 1.0 / np.sqrt(x)

    def _check(self, x_np, rtol=1e-5, atol=1e-5):
        out = paddle.rsqrt(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(out, self._np_rsqrt(x_np), rtol=rtol, atol=atol)

    def test_basic_shapes(self):
        for shape in [(8,), (3, 4), (2, 3, 5)]:
            with self.subTest(shape=shape):
                x = np.random.uniform(0.5, 4.0, shape).astype(np.float32)
                self._check(x)

    def test_known_values(self):
        # rsqrt of perfect squares: 1/1, 1/2, 1/3, 1/4, 1/5
        x = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)
        self._check(x, atol=1e-6)

    def test_rsqrt_consistency_with_sqrt(self):
        x = np.random.uniform(0.5, 4.0, (4, 5)).astype(np.float32)
        x_p = paddle.to_tensor(x, place="mps")
        product = (paddle.rsqrt(x_p) * paddle.sqrt(x_p)).numpy()
        np.testing.assert_allclose(
            product, np.ones_like(product), rtol=1e-5, atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
