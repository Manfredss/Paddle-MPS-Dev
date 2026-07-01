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
bfloat16 dtype coverage for the MPS "act-basic" activation kernel family.

Family / dtype policy:
- relu, silu, sigmoid, gelu, leaky_relu, softmax, relu6 : float + float16 on
  every macOS, plus bfloat16 ONLY on the macOS-14 SDK (MPSDataTypeBFloat16).

These are unary float activations; none of them gain complex64 or any integer
widths, so this file exercises bfloat16 only.

bfloat16 only exists with the macOS-14 MPSGraph SDK and requires macOS 14 AT
RUNTIME, which this CI host may lack. We therefore probe support once with a
tiny bf16 op on an mps tensor and skip the bf16 subtests when unsupported.

bf16 has ~3 significant decimal digits (8-bit mantissa), so MPS bf16 outputs are
compared to a float32 numpy oracle with LOOSE tolerance (looser than fp16).
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


def _supports(paddle_dtype):
    """Return True iff a trivial op on an mps tensor of ``paddle_dtype`` works.

    bfloat16/complex64 map to MPSGraph data types added only in the macOS-14
    SDK and require macOS 14 at runtime. On older hosts constructing or running
    such a tensor raises; we treat ANY exception as "unsupported".
    """
    if not _mps_available():
        return False
    try:
        base = paddle.to_tensor(np.zeros((2,), dtype=np.float32), place="mps")
        t = base.astype(paddle_dtype)
        out = paddle.add(t, t)
        # Force materialization so lazy failures surface here.
        _ = out.astype("float32").numpy()
        return True
    except Exception:
        return False


# bf16: 8-bit mantissa, ~2-3 decimal digits -> looser than fp16.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

_SUPPORTS_BF16 = _supports("bfloat16") if PADDLE_AVAILABLE else False

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSActBasicBF16Base(unittest.TestCase):
    """Common setUp + bf16 helper for the act-basic family dtype tests."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    def _require_bf16(self):
        if not _SUPPORTS_BF16:
            self.skipTest(
                "bfloat16 on MPS requires the macOS-14 SDK and macOS 14 at "
                "runtime; not supported on this host"
            )

    def _assert_bf16_out(self, out):
        """The MPS output must be bfloat16 and live on the mps place."""
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())

    def _run_bf16(self, fn, x32, rtol=BF16_RTOL, atol=BF16_ATOL):
        """Run ``fn`` on a bfloat16 MPS tensor; compare to a float32 oracle.

        ``fn`` takes a paddle tensor and returns a paddle tensor. bf16 has no
        numpy dtype, so we build a float32 mps tensor and cast it to bfloat16.
        """
        self._require_bf16()
        x_bf16 = paddle.to_tensor(x32, place="mps").astype("bfloat16")
        out = fn(x_bf16)
        self._assert_bf16_out(out)
        # float32 numpy oracle (CPU rarely registers bf16 for these ops, so we
        # never compare against the CPU backend for bf16).
        ref = fn(paddle.to_tensor(x32, place="cpu")).numpy().astype(np.float32)
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, rtol=rtol, atol=atol
        )
        return out


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------


class TestMPSReluBF16(_MPSActBasicBF16Base):
    def test_relu_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.nn.functional.relu(t), x)


# ---------------------------------------------------------------------------
# silu
# ---------------------------------------------------------------------------


class TestMPSSiluBF16(_MPSActBasicBF16Base):
    def test_silu_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.nn.functional.silu(t), x)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------


class TestMPSSigmoidBF16(_MPSActBasicBF16Base):
    def test_sigmoid_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # Keep the range modest so exp(-x) stays well-conditioned.
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.nn.functional.sigmoid(t), x)


# ---------------------------------------------------------------------------
# gelu (both approximate=False and approximate=True)
# ---------------------------------------------------------------------------


class TestMPSGeluBF16(_MPSActBasicBF16Base):
    def test_gelu_exact_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.gelu(t, approximate=False),
                    x,
                )

    def test_gelu_approx_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.gelu(t, approximate=True),
                    x,
                )


# ---------------------------------------------------------------------------
# leaky_relu
# ---------------------------------------------------------------------------


class TestMPSLeakyReluBF16(_MPSActBasicBF16Base):
    def test_leaky_relu_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.leaky_relu(
                        t, negative_slope=0.1
                    ),
                    x,
                )


# ---------------------------------------------------------------------------
# softmax (along the last axis)
# ---------------------------------------------------------------------------


class TestMPSSoftmaxBF16(_MPSActBasicBF16Base):
    def test_softmax_bf16(self):
        # Need at least one reduction axis; use 2D / 3D shapes.
        for shape in [(3, 4), (2, 3, 4)]:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.softmax(t, axis=-1), x
                )


# ---------------------------------------------------------------------------
# relu6
# ---------------------------------------------------------------------------


class TestMPSRelu6BF16(_MPSActBasicBF16Base):
    def test_relu6_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # Span the saturating region [0, 6] plus negatives.
                x = (np.random.randn(*shape) * 4.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.nn.functional.relu6(t), x)


if __name__ == "__main__":
    unittest.main(verbosity=2)
