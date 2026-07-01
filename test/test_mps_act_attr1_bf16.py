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
bfloat16 dtype coverage for the MPS "act-attr1" activation kernel family.

Family / dtype policy (this test):
- elu, celu, selu, mish, softplus, softsign, logsigmoid, swish each gain
  bfloat16 on top of their existing float (float32) + float16 registrations.

bfloat16 only exists in the macOS-14 MetalPerformanceShadersGraph SDK and only
runs on macOS 14+ at runtime. The kernels are gated behind that SDK check, and
the host CI machine may lack macOS 14, so every bf16 subtest is PROBE-AND-SKIPped:
a module-level helper builds a tiny bf16 MPS tensor, runs a trivial add, and
reports whether bf16 is usable. When it is not, the bf16 tests skip.

bfloat16 has only ~3 significant decimal digits (an 8-bit mantissa-ish
effective precision), so MPS bf16 output is compared against a float32 numpy
oracle with LOOSE tolerance (looser than the float16 tests).
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
    """Probe whether ``paddle_dtype`` is usable on the MPS backend.

    bfloat16 (and complex64) require the macOS-14 MPSGraph SDK at build time
    AND macOS 14+ at runtime. We build a tiny tensor of the dtype on the mps
    place, run a trivial elementwise add, and force materialization. Any
    exception (unregistered kernel, unavailable data type, runtime failure)
    means the dtype is not usable here, so we report False and the caller skips.
    """
    if not _mps_available():
        return False
    try:
        if paddle_dtype == paddle.bfloat16:
            base = paddle.to_tensor(
                np.zeros((2,), dtype=np.float32), place="mps"
            ).astype("bfloat16")
        elif paddle_dtype == paddle.complex64:
            base = paddle.to_tensor(
                np.zeros((2,), dtype=np.complex64), place="mps"
            )
        else:
            base = paddle.to_tensor(
                np.zeros((2,), dtype=np.float32), place="mps"
            ).astype(paddle_dtype)
        out = paddle.add(base, base)
        # Force evaluation / readback so lazy failures surface here.
        _ = out.astype("float32").numpy()
        return True
    except Exception:
        return False


# bfloat16 carries ~3 decimal digits; use a loose tolerance, looser than fp16.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


# ---------------------------------------------------------------------------
# float32 numpy oracles for each activation (computed in float32, compared to
# the MPS bf16 output upcast to float32).
# ---------------------------------------------------------------------------


def _np_elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)


def _np_celu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1.0)).astype(
        np.float32
    )


def _np_selu(
    x,
    scale=1.0507009873554804934193349852946,
    alpha=1.6732632423543772848170429916717,
):
    return (scale * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))).astype(
        np.float32
    )


def _np_mish(x, threshold=20.0):
    # Mirror the MPS kernel's direct softplus form exactly:
    #   softplus(x) = x                  if x > threshold
    #               = log(1 + exp(x))    otherwise
    # (The kernel does not use the numerically stable log1p(exp(-|x|)) form;
    # matching it here keeps the bf16 oracle aligned with the graph.)
    sp = np.where(x > threshold, x, np.log(1.0 + np.exp(x)))
    return (x * np.tanh(sp)).astype(np.float32)


def _np_softplus(x, beta=1.0, threshold=20.0):
    # Mirror the MPS kernel's direct form exactly:
    #   softplus(x) = x                              if beta * x > threshold
    #               = log(1 + exp(beta * x)) / beta  otherwise
    bx = beta * x
    sp = np.where(bx > threshold, x, np.log(1.0 + np.exp(bx)) / beta)
    return sp.astype(np.float32)


def _np_softsign(x):
    return (x / (1.0 + np.abs(x))).astype(np.float32)


def _np_logsigmoid(x):
    return (-(np.log1p(np.exp(-np.abs(x))) + np.maximum(-x, 0.0))).astype(
        np.float32
    )


def _np_swish(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSActAttr1Base(unittest.TestCase):
    """Common setUp + bf16 probe for the act-attr1 family dtype tests."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        cls._bf16_ok = _supports(paddle.bfloat16)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    def _require_bf16(self):
        if not self._bf16_ok:
            self.skipTest(
                "bfloat16 not supported on this MPS host "
                "(needs macOS 14+ runtime and SDK)"
            )

    def _assert_bf16_out(self, out):
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())

    def _run_bf16(self, p_op, np_op, x32, rtol=BF16_RTOL, atol=BF16_ATOL):
        """Run ``p_op`` (paddle tensor -> paddle tensor) in bf16 on MPS.

        Compares the bf16 result (upcast to float32) against the float32 numpy
        oracle ``np_op``. Also asserts the output dtype and place.
        """
        self._require_bf16()
        # bf16 has no numpy dtype: build float32 on mps, then cast to bf16.
        x_bf16 = paddle.to_tensor(x32, place="mps").astype("bfloat16")
        out = p_op(x_bf16)
        self._assert_bf16_out(out)
        ref = np_op(x32.astype(np.float32))
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, rtol=rtol, atol=atol
        )
        return out


class TestMPSEluBF16(_MPSActAttr1Base):
    def test_elu_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.elu(t, alpha=1.0),
                    lambda a: _np_elu(a, 1.0),
                    x,
                )


class TestMPSCeluBF16(_MPSActAttr1Base):
    def test_celu_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.celu(t, alpha=1.0),
                    lambda a: _np_celu(a, 1.0),
                    x,
                )


class TestMPSSeluBF16(_MPSActAttr1Base):
    def test_selu_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.selu(t), _np_selu, x
                )


class TestMPSMishBF16(_MPSActAttr1Base):
    def test_mish_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.mish(t), _np_mish, x
                )


class TestMPSSoftplusBF16(_MPSActAttr1Base):
    def test_softplus_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.softplus(t),
                    lambda a: _np_softplus(a, 1.0, 20.0),
                    x,
                )


class TestMPSSoftsignBF16(_MPSActAttr1Base):
    def test_softsign_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.softsign(t), _np_softsign, x
                )


class TestMPSLogSigmoidBF16(_MPSActAttr1Base):
    def test_logsigmoid_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.log_sigmoid(t),
                    _np_logsigmoid,
                    x,
                )


class TestMPSSwishBF16(_MPSActAttr1Base):
    def test_swish_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.swish(t), _np_swish, x
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
