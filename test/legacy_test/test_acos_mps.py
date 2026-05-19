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
class TestAcosMPS(unittest.TestCase):
    """MPS-backend coverage for paddle.acos."""

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    def _check_vs_numpy(self, x_np, rtol=1e-4, atol=1e-5):
        out = paddle.acos(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(out, np.arccos(x_np), rtol=rtol, atol=atol)

    def _check_vs_cpu(self, x_np, rtol=1e-5, atol=1e-6):
        out_mps = paddle.acos(paddle.to_tensor(x_np, place="mps")).numpy()
        out_cpu = paddle.acos(paddle.to_tensor(x_np, place="cpu")).numpy()
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol)

    def test_basic_shapes(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            with self.subTest(shape=shape):
                x = np.random.uniform(-0.95, 0.95, shape).astype(np.float32)
                self._check_vs_numpy(x)
                self._check_vs_cpu(x)

    def test_scalar_tensor(self):
        self._check_vs_numpy(np.array(np.float32(0.25)))

    def test_large_1d(self):
        x = np.random.uniform(-0.99, 0.99, (4096,)).astype(np.float32)
        self._check_vs_numpy(x)
        self._check_vs_cpu(x)

    def test_known_values(self):
        x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        self._check_vs_numpy(x, atol=1e-4)

    def test_acos_plus_asin_is_pi_over_two(self):
        x = np.random.uniform(-0.9, 0.9, (4, 5)).astype(np.float32)
        x_p = paddle.to_tensor(x, place="mps")
        s = (paddle.asin(x_p) + paddle.acos(x_p)).numpy()
        np.testing.assert_allclose(
            s, np.full_like(s, np.pi / 2.0), rtol=1e-4, atol=1e-5
        )

    def test_domain_boundary(self):
        x = np.array([1.0, -1.0], dtype=np.float32)
        out = paddle.acos(paddle.to_tensor(x, place="mps")).numpy()
        np.testing.assert_allclose(out, np.arccos(x), rtol=1e-4, atol=1e-4)

    def test_dtype_and_place_preserved(self):
        x = np.random.uniform(-0.5, 0.5, (3, 4)).astype(np.float32)
        out = paddle.acos(paddle.to_tensor(x, place="mps"))
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


if __name__ == "__main__":
    unittest.main()
