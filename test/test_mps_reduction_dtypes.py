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
Dtype coverage for the MPS reduction kernel family: sum, max, min, prod, mean.

float16 coverage (every op): the MPS kernels now register float16 in addition
to float32, so we run each reduction on a float16 tensor placed on 'mps',
cast the result back to float32, and compare against a float32 numpy oracle
with loose tolerance. We do NOT compare against the CPU backend in float16,
because the CPU kernels may not register float16 for these ops and would fail
to dispatch.

int32 / int64 coverage (sum, max, min, prod only -- mean returns a float
average and is float16-only): the integer-valid reductions register int32 and
int64, so we run them on integer tensors, compare EXACTLY against a numpy
integer reference, and ALSO against the CPU backend (which supports int).
"""

import unittest

import numpy as np

try:
    import paddle

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddlePaddle not available")


def _mps_available():
    return (
        PADDLE_AVAILABLE
        and paddle.is_compiled_with_mps()
        and paddle.mps.is_available()
    )


@unittest.skipUnless(
    _mps_available(), "PaddlePaddle is not built with MPS or MPS is unavailable"
)
class TestMPSReductionDtypes(unittest.TestCase):
    """float16 + int32/int64 coverage for sum/max/min/prod/mean on MPS."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    # Shapes kept small so integer products do not overflow.
    _SHAPES = [(6,), (3, 4), (2, 3, 4)]

    # (name, paddle fn, numpy fn). prod uses np.prod; others map directly.
    _FLOAT_OPS = (
        ("sum", lambda x, **kw: paddle.sum(x, **kw), np.sum),
        ("max", lambda x, **kw: paddle.max(x, **kw), np.max),
        ("min", lambda x, **kw: paddle.min(x, **kw), np.min),
        ("prod", lambda x, **kw: paddle.prod(x, **kw), np.prod),
        ("mean", lambda x, **kw: paddle.mean(x, **kw), np.mean),
    )

    # mean is float16-only; the rest are int-capable.
    _INT_OPS = (
        ("sum", lambda x, **kw: paddle.sum(x, **kw), np.sum),
        ("max", lambda x, **kw: paddle.max(x, **kw), np.max),
        ("min", lambda x, **kw: paddle.min(x, **kw), np.min),
        ("prod", lambda x, **kw: paddle.prod(x, **kw), np.prod),
    )

    @staticmethod
    def _np_axis(axis):
        return tuple(axis) if isinstance(axis, list) else axis

    # ---- float16 --------------------------------------------------------
    def test_float16_all_ops(self):
        # MODEST magnitude so that prod / sum in float16 stay in range and the
        # loose tolerance holds. All these ops accept any real input.
        for shape in self._SHAPES:
            x32 = (np.random.randn(*shape).astype(np.float32)) * 1.5
            x16 = x32.astype(np.float16)
            for name, p_op, n_op in self._FLOAT_OPS:
                for axis in (None, -1):
                    for keepdim in (False, True):
                        with self.subTest(
                            op=name, shape=shape, axis=axis, keepdim=keepdim
                        ):
                            self._float16_check(
                                name, p_op, n_op, x16, x32, axis, keepdim
                            )

    def _float16_check(self, name, p_op, n_op, x16, x32, axis, keepdim):
        x_mps = paddle.to_tensor(x16, place="mps")
        out = p_op(x_mps, axis=axis, keepdim=keepdim)
        # dtype / place assertions on the raw output.
        self.assertEqual(
            out.dtype, paddle.float16, f"{name} float16 output dtype mismatch"
        )
        self.assertTrue(
            "mps" in str(out.place).lower(), f"{name} float16 output not on mps"
        )
        out_np = out.numpy().astype(np.float32)
        # Reference computed in float32 from the SAME float16 values.
        ref = n_op(
            x16.astype(np.float32), axis=self._np_axis(axis), keepdims=keepdim
        ).astype(np.float32)
        # prod / large-magnitude paths get a looser tolerance.
        rtol = atol = 5e-2 if name == "prod" else 2e-2
        np.testing.assert_allclose(
            out_np,
            ref,
            rtol=rtol,
            atol=atol,
            err_msg=f"{name} float16 vs float32 oracle (axis={axis}, keepdim={keepdim})",
        )

    # ---- int32 / int64 --------------------------------------------------
    def test_int_ops(self):
        for np_dtype, pd_dtype in (
            (np.int32, paddle.int32),
            (np.int64, paddle.int64),
        ):
            for shape in self._SHAPES:
                x = np.random.randint(-8, 9, size=shape).astype(np_dtype)
                for name, p_op, n_op in self._INT_OPS:
                    for axis in (None, -1):
                        for keepdim in (False, True):
                            with self.subTest(
                                op=name,
                                dtype=np_dtype.__name__,
                                shape=shape,
                                axis=axis,
                                keepdim=keepdim,
                            ):
                                self._int_check(
                                    name, p_op, n_op, x, pd_dtype, axis, keepdim
                                )

    def _int_check(self, name, p_op, n_op, x, pd_dtype, axis, keepdim):
        out_mps_t = p_op(
            paddle.to_tensor(x, place="mps"), axis=axis, keepdim=keepdim
        )
        out_cpu_t = p_op(
            paddle.to_tensor(x, place="cpu"), axis=axis, keepdim=keepdim
        )
        # dtype / place assertions.
        self.assertEqual(
            out_mps_t.dtype, pd_dtype, f"{name} int output dtype mismatch"
        )
        self.assertTrue(
            "mps" in str(out_mps_t.place).lower(),
            f"{name} int output not on mps",
        )
        out_mps = out_mps_t.numpy()
        out_cpu = out_cpu_t.numpy()
        ref = n_op(x, axis=self._np_axis(axis), keepdims=keepdim)
        # Integer reductions are exact.
        np.testing.assert_array_equal(
            out_mps,
            ref,
            err_msg=f"{name} int vs numpy (axis={axis}, keepdim={keepdim})",
        )
        np.testing.assert_array_equal(
            out_mps,
            out_cpu,
            err_msg=f"{name} int vs cpu (axis={axis}, keepdim={keepdim})",
        )

    def test_known_int_values(self):
        # Hand-checked small case exercising each int-capable reduction.
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        x_mps = paddle.to_tensor(x, place="mps")
        np.testing.assert_array_equal(paddle.sum(x_mps).numpy(), np.int32(21))
        np.testing.assert_array_equal(paddle.max(x_mps).numpy(), np.int32(6))
        np.testing.assert_array_equal(paddle.min(x_mps).numpy(), np.int32(1))
        np.testing.assert_array_equal(paddle.prod(x_mps).numpy(), np.int32(720))
        # along axis 1
        np.testing.assert_array_equal(
            paddle.sum(x_mps, axis=1).numpy(), np.array([6, 15], dtype=np.int32)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
