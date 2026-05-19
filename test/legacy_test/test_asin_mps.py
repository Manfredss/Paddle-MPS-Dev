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


def _mps_available():
    return (
        hasattr(paddle, "is_compiled_with_mps")
        and paddle.is_compiled_with_mps()
        and getattr(paddle, "mps", None) is not None
        and paddle.mps.is_available()
    )


@unittest.skipUnless(_mps_available(), "Paddle is not built with MPS or MPS is unavailable")
class TestAsinMPS(unittest.TestCase):
    """MPS-backend coverage for paddle.asin.

    Compares against NumPy ground truth and against paddle's CPU kernel.
    """

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    def _check_vs_numpy(self, x_np, rtol=1e-4, atol=1e-5):
        out = paddle.asin(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(out, np.arcsin(x_np), rtol=rtol, atol=atol)

    def _check_vs_cpu(self, x_np, rtol=1e-5, atol=1e-6):
        out_mps = paddle.asin(paddle.to_tensor(x_np, place="mps")).numpy()
        out_cpu = paddle.asin(paddle.to_tensor(x_np, place="cpu")).numpy()
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol)

    def test_basic_shapes(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            with self.subTest(shape=shape):
                x = np.random.uniform(-0.95, 0.95, shape).astype(np.float32)
                self._check_vs_numpy(x)
                self._check_vs_cpu(x)

    def test_scalar_tensor(self):
        self._check_vs_numpy(np.array(np.float32(0.5)))

    def test_large_1d(self):
        x = np.random.uniform(-0.99, 0.99, (4096,)).astype(np.float32)
        self._check_vs_numpy(x)
        self._check_vs_cpu(x)

    def test_known_values(self):
        x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        # Endpoints can lose a bit of precision; relax atol slightly.
        self._check_vs_numpy(x, atol=1e-4)

    def test_domain_boundary(self):
        x = np.array([1.0, -1.0], dtype=np.float32)
        out = paddle.asin(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.arcsin(x), rtol=1e-4, atol=1e-4)

    def test_round_trip_with_sin(self):
        x = np.random.uniform(-0.9, 0.9, (4, 5)).astype(np.float32)
        x_p = paddle.to_tensor(x, place="mps")
        result = paddle.sin(paddle.asin(x_p)).numpy()
        np.testing.assert_allclose(result, x, rtol=1e-4, atol=1e-5)

    def test_dtype_and_place_preserved(self):
        x = np.random.uniform(-0.5, 0.5, (3, 4)).astype(np.float32)
        out = paddle.asin(paddle.to_tensor(x, place="mps"))
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())

    def test_input_is_not_mutated(self):
        x = np.random.uniform(-0.5, 0.5, (3, 4)).astype(np.float32)
        t = paddle.to_tensor(x, place="mps")
        _ = paddle.asin(t)
        np.testing.assert_array_equal(t.numpy(), x)


if __name__ == "__main__":
    unittest.main()
