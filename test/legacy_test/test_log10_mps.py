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
class TestLog10MPS(unittest.TestCase):
    """MPS-backend coverage for paddle.log10."""

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    def _check_vs_numpy(self, x_np, rtol=1e-5, atol=1e-5):
        out = paddle.log10(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(out, np.log10(x_np), rtol=rtol, atol=atol)

    def _check_vs_cpu(self, x_np, rtol=1e-5, atol=1e-6):
        out_mps = paddle.log10(paddle.to_tensor(x_np, place="mps")).numpy()
        out_cpu = paddle.log10(paddle.to_tensor(x_np, place="cpu")).numpy()
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol)

    def test_basic_shapes(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            with self.subTest(shape=shape):
                x = np.random.uniform(0.01, 100.0, shape).astype(np.float32)
                self._check_vs_numpy(x)
                self._check_vs_cpu(x)

    def test_scalar_tensor(self):
        self._check_vs_numpy(np.array(np.float32(100.0)))

    def test_large_1d(self):
        x = np.random.uniform(0.01, 1000.0, (4096,)).astype(np.float32)
        self._check_vs_numpy(x)
        self._check_vs_cpu(x)

    def test_exact_powers_of_ten(self):
        x = np.array([1.0, 10.0, 100.0, 1000.0, 0.1, 0.01], dtype=np.float32)
        out = paddle.log10(paddle.to_tensor(x, place="mps")).numpy()
        expected = np.array([0, 1, 2, 3, -1, -2], dtype=np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_log10_of_one_is_zero(self):
        x = np.ones((3, 4), dtype=np.float32)
        out = paddle.log10(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.zeros_like(out), rtol=0, atol=1e-7)

    def test_small_positive_values(self):
        x = np.array([1e-6, 1e-3, 1e-1], dtype=np.float32)
        self._check_vs_numpy(x, atol=1e-4)

    def test_dtype_and_place_preserved(self):
        x = np.random.uniform(0.1, 10.0, (3, 4)).astype(np.float32)
        out = paddle.log10(paddle.to_tensor(x, place="mps"))
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


if __name__ == "__main__":
    unittest.main()
