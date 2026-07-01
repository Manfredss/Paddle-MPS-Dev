#!/usr/bin/env python3


# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

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


"""
Dtype-coverage tests for the MPS 'unary-int' kernel family.

Family ops (all int-capable in MPSGraph):
- abs       -> absoluteWithTensor
- sign      -> signWithTensor
- negative  -> negativeWithTensor

For each op we test:
- float16: run on MPS in fp16, compare (loose tol) to a float32 numpy oracle.
            CPU is NOT used as a reference here (CPU may not register fp16).
- int32/int64: run on MPS, compare EXACTLY against a numpy integer oracle
                AND against the Paddle CPU backend (which supports int).
"""

import unittest

import numpy as np

try:
    import paddle

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddlePaddle not available")


SHAPES = [(6,), (3, 4), (2, 3, 4)]


def np_sign(x):
    # Matches paddle.sign / torch.sign: -1, 0, +1
    return np.sign(x)


# (op_name, paddle_callable, numpy_oracle)
FAMILY_OPS = [
    ("abs", lambda t: paddle.abs(t), np.abs),
    ("sign", lambda t: paddle.sign(t), np_sign),
    ("negative", lambda t: paddle.neg(t), np.negative),
]


class _MPSUnaryIntDtypeTestBase(unittest.TestCase):
    """Base harness: guards on MPS availability and seeds per test."""

    @classmethod
    def setUpClass(cls):
        if not PADDLE_AVAILABLE:
            raise unittest.SkipTest("PaddlePaddle not available")
        if not paddle.is_compiled_with_mps():
            raise unittest.SkipTest(
                "PaddlePaddle not compiled with MPS support"
            )
        if not paddle.mps.is_available():
            raise unittest.SkipTest("MPS not available on this system")
        paddle.mps.set_device(0)
        cls.device = "mps"

    def setUp(self):
        np.random.seed(42)
        paddle.seed(42)


@unittest.skipUnless(
    PADDLE_AVAILABLE
    and paddle.is_compiled_with_mps()
    and paddle.mps.is_available(),
    "MPS backend not available",
)
class TestMPSUnaryIntFloat16(_MPSUnaryIntDtypeTestBase):
    """float16 coverage for every op in the family."""

    def _run_float16(self, op_name, paddle_op, numpy_oracle):
        for shape in SHAPES:
            with self.subTest(op=op_name, shape=shape, dtype="float16"):
                # Modest range; these ops are valid over all reals.
                x32 = (np.random.randn(*shape) * 1.5).astype(np.float32)
                x16 = x32.astype(np.float16)

                x_mps = paddle.to_tensor(x16, place=self.device)
                out = paddle_op(x_mps)

                self.assertEqual(out.dtype, paddle.float16)
                self.assertIn("mps", str(out.place).lower())

                out_np = out.numpy().astype(np.float32)
                ref = numpy_oracle(x32).astype(np.float32)

                np.testing.assert_allclose(
                    out_np,
                    ref,
                    rtol=2e-2,
                    atol=2e-2,
                    err_msg=f"{op_name} float16 mismatch shape={shape}",
                )

    def test_abs_float16(self):
        self._run_float16(*FAMILY_OPS[0])

    def test_sign_float16(self):
        self._run_float16(*FAMILY_OPS[1])

    def test_negative_float16(self):
        self._run_float16(*FAMILY_OPS[2])


@unittest.skipUnless(
    PADDLE_AVAILABLE
    and paddle.is_compiled_with_mps()
    and paddle.mps.is_available(),
    "MPS backend not available",
)
class TestMPSUnaryIntInteger(_MPSUnaryIntDtypeTestBase):
    """int32/int64 coverage (exact) for every int-capable op."""

    def _run_integer(
        self, op_name, paddle_op, numpy_oracle, np_dtype, pd_dtype
    ):
        for shape in SHAPES:
            with self.subTest(op=op_name, shape=shape, dtype=str(np_dtype)):
                x = np.random.randint(-8, 9, size=shape).astype(np_dtype)

                x_mps = paddle.to_tensor(x, place=self.device)
                out_mps = paddle_op(x_mps)

                self.assertEqual(out_mps.dtype, pd_dtype)
                self.assertIn("mps", str(out_mps.place).lower())

                out_np = out_mps.numpy()

                # Exact match against numpy integer oracle.
                ref = numpy_oracle(x).astype(np_dtype)
                np.testing.assert_array_equal(
                    out_np,
                    ref,
                    err_msg=f"{op_name} {np_dtype} vs numpy mismatch shape={shape}",
                )

                # Exact match against the Paddle CPU backend.
                x_cpu = paddle.to_tensor(x, place="cpu")
                out_cpu = paddle_op(x_cpu).numpy()
                np.testing.assert_array_equal(
                    out_np,
                    out_cpu,
                    err_msg=f"{op_name} {np_dtype} vs CPU mismatch shape={shape}",
                )

    # ---- int32 ----
    def test_abs_int32(self):
        self._run_integer(*FAMILY_OPS[0], np.int32, paddle.int32)

    def test_sign_int32(self):
        self._run_integer(*FAMILY_OPS[1], np.int32, paddle.int32)

    def test_negative_int32(self):
        self._run_integer(*FAMILY_OPS[2], np.int32, paddle.int32)

    # ---- int64 ----
    def test_abs_int64(self):
        self._run_integer(*FAMILY_OPS[0], np.int64, paddle.int64)

    def test_sign_int64(self):
        self._run_integer(*FAMILY_OPS[1], np.int64, paddle.int64)

    def test_negative_int64(self):
        self._run_integer(*FAMILY_OPS[2], np.int64, paddle.int64)


if __name__ == "__main__":
    unittest.main()
