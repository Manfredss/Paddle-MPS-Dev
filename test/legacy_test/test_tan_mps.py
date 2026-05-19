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
class TestTanMPS(unittest.TestCase):
    """MPS-backend coverage for paddle.tan.

    Compares against NumPy ground truth and against paddle's CPU kernel, exercises
    multiple shapes (including 0-D and 4-D), validates dtype/place are preserved,
    and checks the analytic identity tan(x) = sin(x)/cos(x).
    """

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    def _check_vs_numpy(self, x_np, rtol=1e-4, atol=1e-5):
        out = paddle.tan(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(out, np.tan(x_np), rtol=rtol, atol=atol)

    def _check_vs_cpu(self, x_np, rtol=1e-5, atol=1e-6):
        out_mps = paddle.tan(paddle.to_tensor(x_np, place="mps")).numpy()
        out_cpu = paddle.tan(paddle.to_tensor(x_np, place="cpu")).numpy()
        np.testing.assert_allclose(out_mps, out_cpu, rtol=rtol, atol=atol)

    def test_basic_shapes(self):
        # Stay away from pi/2 multiples (where tan diverges).
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            with self.subTest(shape=shape):
                x = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
                self._check_vs_numpy(x)
                self._check_vs_cpu(x)

    def test_scalar_tensor(self):
        x = np.float32(0.5)
        self._check_vs_numpy(np.array(x))

    def test_large_1d(self):
        x = np.random.uniform(-1.2, 1.2, (4096,)).astype(np.float32)
        self._check_vs_numpy(x)
        self._check_vs_cpu(x)

    def test_known_values(self):
        x = np.array(
            [0.0, np.pi / 6, np.pi / 4, -np.pi / 4, -np.pi / 6],
            dtype=np.float32,
        )
        self._check_vs_numpy(x)

    def test_tan_is_sin_over_cos(self):
        x = np.random.uniform(-1.0, 1.0, (4, 5)).astype(np.float32)
        x_p = paddle.to_tensor(x, place="mps")
        lhs = paddle.tan(x_p).numpy()
        rhs = (paddle.sin(x_p) / paddle.cos(x_p)).numpy()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=1e-5)

    def test_dtype_and_place_preserved(self):
        x = np.random.uniform(-1.0, 1.0, (3, 4)).astype(np.float32)
        out = paddle.tan(paddle.to_tensor(x, place="mps"))
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())

    def test_input_is_not_mutated(self):
        x = np.random.uniform(-1.0, 1.0, (3, 4)).astype(np.float32)
        t = paddle.to_tensor(x, place="mps")
        _ = paddle.tan(t)
        np.testing.assert_array_equal(t.numpy(), x)


if __name__ == "__main__":
    unittest.main()
