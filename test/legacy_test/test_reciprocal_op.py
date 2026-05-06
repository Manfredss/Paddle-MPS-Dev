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


class TestReciprocalApi(unittest.TestCase):
    """Lightweight CPU coverage for paddle.reciprocal.

    The full op-level coverage (gradients, dtypes, FP16/BF16) lives in
    test_activation_op.py::TestReciprocal*. This file adds a focused API
    test plus an MPS-backend test class so MPS regressions land in a
    file named after the operator.
    """

    def setUp(self):
        paddle.disable_static()
        np.random.seed(2026)

    def test_reciprocal_float32(self):
        x = np.random.uniform(0.5, 4.0, (3, 4)).astype(np.float32)
        out = paddle.reciprocal(paddle.to_tensor(x)).numpy()
        np.testing.assert_allclose(out, np.reciprocal(x), rtol=1e-5, atol=1e-6)

    def test_reciprocal_negative_inputs(self):
        x = np.random.uniform(-4.0, -0.5, (2, 3)).astype(np.float32)
        out = paddle.reciprocal(paddle.to_tensor(x)).numpy()
        np.testing.assert_allclose(out, np.reciprocal(x), rtol=1e-5, atol=1e-6)


def _mps_available():
    return (
        hasattr(paddle, "is_compiled_with_mps")
        and paddle.is_compiled_with_mps()
        and getattr(paddle, "mps", None) is not None
        and paddle.mps.is_available()
    )


@unittest.skipUnless(_mps_available(), "Paddle is not built with MPS or MPS is unavailable")
class TestReciprocalMPS(unittest.TestCase):
    """MPS-backend coverage for paddle.reciprocal."""

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    def _check(self, x_np):
        out = paddle.reciprocal(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(
            out, np.reciprocal(x_np), rtol=1e-5, atol=1e-6,
        )

    def test_basic_shapes(self):
        # Avoid values too close to zero — both backends would otherwise be
        # numerically unstable rather than the kernel itself being wrong.
        for shape in [(8,), (4, 5), (2, 3, 4)]:
            with self.subTest(shape=shape):
                x = np.random.uniform(0.5, 5.0, shape).astype(np.float32)
                self._check(x)

    def test_negative_inputs(self):
        x = np.random.uniform(-5.0, -0.5, (3, 6)).astype(np.float32)
        self._check(x)

    def test_mixed_inputs(self):
        x = np.array([-2.0, -0.5, 0.5, 2.0, 4.0], dtype=np.float32)
        self._check(x)

    def test_reciprocal_inverse(self):
        # x * reciprocal(x) == 1
        x = np.random.uniform(0.5, 4.0, (3, 7)).astype(np.float32)
        x_p = paddle.to_tensor(x, place="mps")
        product = (x_p * paddle.reciprocal(x_p)).numpy()
        np.testing.assert_allclose(
            product, np.ones_like(product), rtol=1e-5, atol=1e-5,
        )


if __name__ == '__main__':
    unittest.main()
