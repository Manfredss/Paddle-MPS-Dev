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
Dtype coverage for the MPS trigonometric / transcendental kernel family.

Family ops (all float-only on MPSGraph; registered for float32 + float16):
    sin, cos, tan, asin, acos, atan,
    sinh, cosh, tanh, asinh, acosh, atanh

These kernels previously hardcoded MPSDataTypeFloat32 in their
MPSGraphTensorData wrappers; they now plumb the real tensor dtype through
GetMPSDataType, so float16 inputs run end-to-end on MPS.

Policy for this file:
- float16 only (MPSGraph trig is float-only; no integer instantiations).
- Compare the MPS float16 result (cast up to float32) against a float32
  numpy oracle with LOOSE tolerance. We deliberately do NOT compare against
  the CPU backend in float16: CPU may not register float16 for these ops and
  would fail to dispatch.
- Respect each op's mathematical domain when sampling inputs:
    asin/acos      -> |x| <= 1
    atanh          -> |x| <  1 (strictly)
    acosh          -> x  >= 1
  All others accept the full real line; we keep magnitudes modest so float16
  rounding stays well-behaved.
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


# Per-op: (paddle fn, numpy oracle, input sampler, (rtol, atol)).
# The sampler takes a base float32 array of standard-normal noise (already
# scaled to a modest range) and maps it into the op's valid domain.


def _unit_interval(x):
    # Map arbitrary float32 noise into (-1, 1) for asin / acos.
    return np.clip(x, -0.95, 0.95).astype(np.float32)


def _open_unit_interval(x):
    # Strictly inside (-1, 1) for atanh.
    return np.clip(x, -0.9, 0.9).astype(np.float32)


def _ge_one(x):
    # Map into [1, inf) for acosh: 1 + |x|.
    return (1.0 + np.abs(x)).astype(np.float32)


def _modest(x):
    # General-purpose modest range for the rest.
    return x.astype(np.float32)


_TRIG_OPS = (
    ("sin", lambda t: paddle.sin(t), np.sin, _modest, (2e-2, 2e-2)),
    ("cos", lambda t: paddle.cos(t), np.cos, _modest, (2e-2, 2e-2)),
    ("tan", lambda t: paddle.tan(t), np.tan, _modest, (5e-2, 5e-2)),
    ("asin", lambda t: paddle.asin(t), np.arcsin, _unit_interval, (2e-2, 2e-2)),
    ("acos", lambda t: paddle.acos(t), np.arccos, _unit_interval, (2e-2, 2e-2)),
    ("atan", lambda t: paddle.atan(t), np.arctan, _modest, (2e-2, 2e-2)),
    ("sinh", lambda t: paddle.sinh(t), np.sinh, _modest, (2e-2, 2e-2)),
    ("cosh", lambda t: paddle.cosh(t), np.cosh, _modest, (2e-2, 2e-2)),
    ("tanh", lambda t: paddle.tanh(t), np.tanh, _modest, (2e-2, 2e-2)),
    ("asinh", lambda t: paddle.asinh(t), np.arcsinh, _modest, (2e-2, 2e-2)),
    ("acosh", lambda t: paddle.acosh(t), np.arccosh, _ge_one, (2e-2, 2e-2)),
    (
        "atanh",
        lambda t: paddle.atanh(t),
        np.arctanh,
        _open_unit_interval,
        (5e-2, 5e-2),
    ),
)

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


class TestMPSTrigFloat16(unittest.TestCase):
    """float16 coverage for the MPS trig/transcendental kernel family."""

    @classmethod
    def setUpClass(cls):
        if not _mps_available():
            raise unittest.SkipTest(
                "PaddlePaddle is not built with MPS or MPS is unavailable"
            )
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)

    def _make_x32(self, shape, sampler):
        # Modest-magnitude standard normal, then mapped into the op's domain.
        base = (np.random.randn(*shape) * 1.5).astype(np.float32)
        return sampler(base)

    def _check_float16(self, name, paddle_op, numpy_op, sampler, tols, shape):
        rtol, atol = tols
        x32 = self._make_x32(shape, sampler)
        x16 = x32.astype(np.float16)

        # Run on MPS in float16.
        out = paddle_op(paddle.to_tensor(x16, place="mps"))

        # dtype / place invariants.
        self.assertEqual(
            out.dtype,
            paddle.float16,
            f"{name}: expected float16 output, got {out.dtype}",
        )
        self.assertTrue(
            "mps" in str(out.place).lower(),
            f"{name}: expected output on MPS place, got {out.place}",
        )

        out32 = out.numpy().astype(np.float32)

        # float32 numpy oracle computed on the SAME (float16-rounded) inputs,
        # so we only measure the op's own float16 error, not input rounding.
        ref = numpy_op(x16.astype(np.float32)).astype(np.float32)

        # Guard against NaN/Inf leaking from out-of-domain rounding.
        finite = np.isfinite(ref) & np.isfinite(out32)
        self.assertTrue(
            finite.any(),
            f"{name}: no finite reference values to compare (shape={shape})",
        )
        np.testing.assert_allclose(
            out32[finite],
            ref[finite],
            rtol=rtol,
            atol=atol,
            err_msg=f"{name} float16 vs numpy (shape={shape})",
        )

    def test_float16_all_ops_all_shapes(self):
        for name, p_op, n_op, sampler, tols in _TRIG_OPS:
            for shape in _SHAPES:
                with self.subTest(op=name, shape=shape):
                    self._check_float16(name, p_op, n_op, sampler, tols, shape)

    def test_float16_dtype_and_place(self):
        # Explicit, op-by-op dtype/place assertion on a fixed small input.
        for name, p_op, _, sampler, _ in _TRIG_OPS:
            with self.subTest(op=name):
                x32 = self._make_x32((3, 4), sampler)
                x16 = x32.astype(np.float16)
                out = p_op(paddle.to_tensor(x16, place="mps"))
                self.assertEqual(out.dtype, paddle.float16, f"{name} dtype")
                self.assertTrue(
                    "mps" in str(out.place).lower(), f"{name} place"
                )

    def test_float32_still_works(self):
        # Regression: float32 path (the original registration) must still run
        # and match numpy, confirming the dtype plumbing didn't break float32.
        for name, p_op, n_op, sampler, tols in _TRIG_OPS:
            with self.subTest(op=name):
                x32 = self._make_x32((2, 3, 4), sampler)
                out = p_op(paddle.to_tensor(x32, place="mps"))
                self.assertEqual(out.dtype, paddle.float32, f"{name} f32 dtype")
                self.assertTrue(
                    "mps" in str(out.place).lower(), f"{name} f32 place"
                )
                ref = n_op(x32).astype(np.float32)
                out_np = out.numpy()
                finite = np.isfinite(ref) & np.isfinite(out_np)
                np.testing.assert_allclose(
                    out_np[finite],
                    ref[finite],
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"{name} float32 vs numpy",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
