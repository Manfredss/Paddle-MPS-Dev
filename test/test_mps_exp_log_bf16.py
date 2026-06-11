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
bfloat16 dtype coverage for the MPS "exp-log" kernel family.

Family / dtype policy (this task only ADDS bfloat16, gated behind the macOS-14
SDK at build time and macOS-14 at runtime):

    exp, expm1, log, log2, log10, log1p, erf, sqrt, rsqrt, reciprocal, square

Every op above already supported float32 + float16; this file exercises the new
bfloat16 registration. bfloat16 carries only ~3 decimal digits of precision (an
8-bit mantissa), so the MPS bfloat16 output is compared against a float32 numpy
oracle with LOOSE tolerances (rtol/atol = 4e-2) -- looser than float16.

bfloat16 on MPSGraph requires the macOS-14 SDK at build time and macOS 14 at
runtime. This CI host may be older, so each bfloat16 subtest is gated by a
runtime probe (`_supports`) that tries a trivial bfloat16 op on an mps tensor
and skips when it raises.

Domain notes:
- log / log2 / log10 / sqrt / rsqrt require strictly positive inputs.
- log1p requires x > -1.
- reciprocal requires nonzero inputs.
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

    bfloat16 (and complex64) on MPSGraph need the macOS-14 SDK + macOS 14 at
    runtime; on older hosts the registration is absent or the op throws. We
    create a tiny mps tensor of the requested dtype and run a trivial add,
    returning False on ANY exception so the caller can skip.
    """
    if not _mps_available():
        return False
    try:
        base = paddle.to_tensor(np.zeros((2,), dtype=np.float32), place="mps")
        t = base.astype(paddle_dtype)
        _ = (t + t).astype("float32").numpy()
        return True
    except Exception:
        return False


# bfloat16 has ~3 decimal digits; use looser tolerances than float16.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSExpLogBf16Base(unittest.TestCase):
    """Common setUp + bfloat16 helper for the exp-log family."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        # Probe once: does this host support bfloat16 on MPS at runtime?
        cls.bf16_ok = _supports(paddle.bfloat16)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    def _skip_if_no_bf16(self):
        if not getattr(type(self), "bf16_ok", False):
            self.skipTest(
                "bfloat16 not supported on this MPS host "
                "(requires macOS 14 SDK + runtime)"
            )

    def _assert_bf16_out(self, out):
        """The MPS output must be bfloat16 and live on the mps place."""
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())

    def _run_bf16(self, fn, np_op, x32, rtol=BF16_RTOL, atol=BF16_ATOL):
        """Run ``fn`` on a bfloat16 MPS tensor; compare to a float32 oracle.

        ``fn``   : takes a paddle tensor, returns a paddle tensor (the MPS op).
        ``np_op``: numpy reference applied to the float32 input.
        ``x32``  : float32 numpy input already constrained to the op's domain.

        bfloat16 has no numpy dtype, so we build the input as float32 on mps and
        cast to bfloat16 via ``.astype``.
        """
        x_bf16 = paddle.to_tensor(x32, place="mps").astype("bfloat16")
        out = fn(x_bf16)
        self._assert_bf16_out(out)
        ref = np_op(x32).astype(np.float32)
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, rtol=rtol, atol=atol
        )
        return out


# ---------------------------------------------------------------------------
# Per-op test cases. Each op gains bfloat16; inputs respect the op's domain.
# ---------------------------------------------------------------------------


class TestMPSExpBf16(_MPSExpLogBf16Base):
    def test_exp_bf16(self):
        self._skip_if_no_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # Keep magnitude modest so exp() does not overflow bf16.
                x = (np.random.randn(*shape) * 1.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.exp(t), np.exp, x)


class TestMPSExpm1Bf16(_MPSExpLogBf16Base):
    def test_expm1_bf16(self):
        self._skip_if_no_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.expm1(t), np.expm1, x)


class TestMPSLogBf16(_MPSExpLogBf16Base):
    def test_log_bf16(self):
        self._skip_if_no_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # log needs strictly positive inputs.
                x = (np.random.rand(*shape) * 4.0 + 0.5).astype(np.float32)
                self._run_bf16(lambda t: paddle.log(t), np.log, x)


class TestMPSLog2Bf16(_MPSExpLogBf16Base):
    def test_log2_bf16(self):
        self._skip_if_no_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.rand(*shape) * 4.0 + 0.5).astype(np.float32)
                self._run_bf16(lambda t: paddle.log2(t), np.log2, x)


class TestMPSLog10Bf16(_MPSExpLogBf16Base):
    def test_log10_bf16(self):
        self._skip_if_no_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.rand(*shape) * 4.0 + 0.5).astype(np.float32)
                self._run_bf16(lambda t: paddle.log10(t), np.log10, x)


class TestMPSLog1pBf16(_MPSExpLogBf16Base):
    def test_log1p_bf16(self):
        self._skip_if_no_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # log1p needs x > -1; keep well inside the domain.
                x = (np.random.rand(*shape) * 4.0 + 0.1).astype(np.float32)
                self._run_bf16(lambda t: paddle.log1p(t), np.log1p, x)


class TestMPSErfBf16(_MPSExpLogBf16Base):
    def test_erf_bf16(self):
        self._skip_if_no_bf16()
        try:
            from scipy.special import erf as _np_erf
        except ImportError:
            _np_erf = np.vectorize(
                lambda v: float(
                    paddle.erf(
                        paddle.to_tensor(np.array([v], dtype=np.float32))
                    ).numpy()[0]
                )
            )
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(lambda t: paddle.erf(t), _np_erf, x)


class TestMPSSqrtBf16(_MPSExpLogBf16Base):
    def test_sqrt_bf16(self):
        self._skip_if_no_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # sqrt needs non-negative (use strictly positive) inputs.
                x = (np.random.rand(*shape) * 9.0 + 0.1).astype(np.float32)
                self._run_bf16(lambda t: paddle.sqrt(t), np.sqrt, x)


class TestMPSRsqrtBf16(_MPSExpLogBf16Base):
    def test_rsqrt_bf16(self):
        self._skip_if_no_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # rsqrt needs strictly positive inputs.
                x = (np.random.rand(*shape) * 9.0 + 0.5).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.rsqrt(t),
                    lambda a: 1.0 / np.sqrt(a),
                    x,
                )


class TestMPSReciprocalBf16(_MPSExpLogBf16Base):
    def test_reciprocal_bf16(self):
        self._skip_if_no_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # reciprocal needs nonzero inputs; keep away from 0.
                x = (np.random.rand(*shape) * 3.0 + 0.5).astype(np.float32)
                x *= np.where(np.random.rand(*shape) < 0.5, -1.0, 1.0).astype(
                    np.float32
                )
                self._run_bf16(
                    lambda t: paddle.reciprocal(t),
                    lambda a: 1.0 / a,
                    x,
                )


class TestMPSSquareBf16(_MPSExpLogBf16Base):
    def test_square_bf16(self):
        self._skip_if_no_bf16()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.square(t),
                    lambda a: a * a,
                    x,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
