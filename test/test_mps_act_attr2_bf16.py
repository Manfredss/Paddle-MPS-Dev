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
bfloat16 dtype coverage for the MPS "act-attr2" activation kernel family.

Family / dtype policy (every op below):
- BASE  dtypes : float32, float16   (work on macOS 12+)
- GATED dtype  : bfloat16           (registered behind a macOS-14 SDK #if;
                                      also requires macOS 14 AT RUNTIME)

Ops covered (all attribute-carrying elementwise activations):
- hardsigmoid, hardswish, hardtanh
- hard_shrink, softshrink, tanh_shrink
- thresholded_relu, stanh, logit

bfloat16 has no numpy dtype, so the bf16 input is built by creating a float32
tensor on the mps place and calling ``.astype(paddle.bfloat16)``. Because
bfloat16 carries only ~3 decimal digits (8-bit mantissa, looser than fp16), the
mps bf16 output is compared against a *float32 numpy oracle* with a LOOSE
tolerance (rtol/atol = 4e-2) rather than against the CPU backend.

bfloat16 requires macOS 14 at runtime, which this CI host may lack. The module
probes support once (``_supports``) by running a trivial bf16 add on an mps
tensor; if anything raises, every bf16 subtest is skipped.
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
    """Return True iff a trivial mps op on ``paddle_dtype`` succeeds.

    bfloat16 / complex64 map to MPSGraph data types that only exist in the
    macOS-14 SDK and require macOS 14 at runtime. On older hosts constructing
    or operating on such a tensor raises; we treat ANY exception as "this dtype
    is unsupported here" and skip the corresponding subtests.
    """
    if not _mps_available():
        return False
    try:
        base = paddle.to_tensor(np.zeros((2,), dtype=np.float32), place="mps")
        t = base.astype(paddle_dtype)
        _ = paddle.add(t, t)
        # Force the graph to actually execute / materialize.
        _ = _.astype("float32").numpy()
        return True
    except Exception:
        return False


# bfloat16: ~3 decimal digits -> looser than fp16.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSActAttr2BF16Base(unittest.TestCase):
    """Common setup + the bf16 oracle-comparison helper."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        cls._bf16_ok = _supports(paddle.bfloat16)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)
        if not self._bf16_ok:
            self.skipTest(
                "bfloat16 unsupported on this MPS host "
                "(needs the macOS-14 SDK + macOS 14 at runtime)"
            )

    # -- bf16 helpers -----------------------------------------------------
    def _assert_bf16_out(self, out):
        """The MPS output must be bfloat16 and live on the mps place."""
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())

    def _run_bf16(self, fn, oracle, x32, rtol=BF16_RTOL, atol=BF16_ATOL):
        """Run ``fn`` on a bf16 mps tensor; compare to a float32 oracle.

        ``fn``     : paddle tensor -> paddle tensor (the op under test).
        ``oracle`` : numpy float32 array -> numpy float32 array (reference).
        ``x32``    : float32 numpy input (already inside the op's domain).
        """
        # bf16 has no numpy dtype: build float32 on mps, then cast to bf16.
        x_mps = paddle.to_tensor(x32, place="mps").astype("bfloat16")
        out = fn(x_mps)
        self._assert_bf16_out(out)
        ref = oracle(x32).astype(np.float32)
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, rtol=rtol, atol=atol
        )
        return out


# ---------------------------------------------------------------------------
# Numpy float32 oracles (match the phi op definitions used by the kernels).
# ---------------------------------------------------------------------------


def _np_hardsigmoid(x, slope, offset):
    return np.clip(slope * x + offset, 0.0, 1.0).astype(np.float32)


def _np_hardswish(x):
    # x * min(max(x + 3, 0), 6) / 6
    return (x * np.clip(x + 3.0, 0.0, 6.0) / 6.0).astype(np.float32)


def _np_hardtanh(x, t_min, t_max):
    return np.clip(x, t_min, t_max).astype(np.float32)


def _np_hard_shrink(x, threshold):
    out = np.where((x <= -threshold) | (x >= threshold), x, 0.0)
    return out.astype(np.float32)


def _np_softshrink(x, lam):
    out = np.where(
        x > lam,
        x - lam,
        np.where(x < -lam, x + lam, 0.0),
    )
    return out.astype(np.float32)


def _np_tanh_shrink(x):
    return (x - np.tanh(x)).astype(np.float32)


def _np_thresholded_relu(x, threshold, value):
    return np.where(x > threshold, x, value).astype(np.float32)


def _np_stanh(x, scale_a, scale_b):
    return (scale_b * np.tanh(scale_a * x)).astype(np.float32)


def _np_logit(x, eps):
    if eps > 0:
        x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


# ---------------------------------------------------------------------------
# hardsigmoid
# ---------------------------------------------------------------------------


class TestMPSHardSigmoidBF16(_MPSActAttr2BF16Base):
    def test_hardsigmoid_bf16(self):
        slope, offset = 0.2, 0.5
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 3.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.hardsigmoid(
                        t, slope=slope, offset=offset
                    ),
                    lambda a: _np_hardsigmoid(a, slope, offset),
                    x,
                )


# ---------------------------------------------------------------------------
# hardswish
# ---------------------------------------------------------------------------


class TestMPSHardSwishBF16(_MPSActAttr2BF16Base):
    def test_hardswish_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 3.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.hardswish(t),
                    _np_hardswish,
                    x,
                )


# ---------------------------------------------------------------------------
# hardtanh
# ---------------------------------------------------------------------------


class TestMPSHardTanhBF16(_MPSActAttr2BF16Base):
    def test_hardtanh_bf16(self):
        t_min, t_max = -1.0, 1.0
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.hardtanh(
                        t, min=t_min, max=t_max
                    ),
                    lambda a: _np_hardtanh(a, t_min, t_max),
                    x,
                )


# ---------------------------------------------------------------------------
# hard_shrink
# ---------------------------------------------------------------------------


class TestMPSHardShrinkBF16(_MPSActAttr2BF16Base):
    def test_hard_shrink_bf16(self):
        threshold = 0.5
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.hardshrink(
                        t, threshold=threshold
                    ),
                    lambda a: _np_hard_shrink(a, threshold),
                    x,
                )


# ---------------------------------------------------------------------------
# softshrink
# ---------------------------------------------------------------------------


class TestMPSSoftShrinkBF16(_MPSActAttr2BF16Base):
    def test_softshrink_bf16(self):
        lam = 0.5
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.softshrink(t, threshold=lam),
                    lambda a: _np_softshrink(a, lam),
                    x,
                )


# ---------------------------------------------------------------------------
# tanh_shrink
# ---------------------------------------------------------------------------


class TestMPSTanhShrinkBF16(_MPSActAttr2BF16Base):
    def test_tanh_shrink_bf16(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.tanhshrink(t),
                    _np_tanh_shrink,
                    x,
                )


# ---------------------------------------------------------------------------
# thresholded_relu
# ---------------------------------------------------------------------------


class TestMPSThresholdedReluBF16(_MPSActAttr2BF16Base):
    def test_thresholded_relu_bf16(self):
        threshold, value = 1.0, 0.0
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.nn.functional.thresholded_relu(
                        t, threshold=threshold
                    ),
                    lambda a: _np_thresholded_relu(a, threshold, value),
                    x,
                )


# ---------------------------------------------------------------------------
# stanh
# ---------------------------------------------------------------------------


class TestMPSStanhBF16(_MPSActAttr2BF16Base):
    def test_stanh_bf16(self):
        scale_a, scale_b = 0.67, 1.72
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 2.0).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.stanh(t, scale_a=scale_a, scale_b=scale_b),
                    lambda a: _np_stanh(a, scale_a, scale_b),
                    x,
                )


# ---------------------------------------------------------------------------
# logit: domain is (0, 1).
# ---------------------------------------------------------------------------


class TestMPSLogitBF16(_MPSActAttr2BF16Base):
    def test_logit_bf16(self):
        eps = 1e-3
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # Keep inputs strictly inside (eps, 1 - eps) to avoid inf/nan;
                # bf16's coarse mantissa near the edges would blow up the log.
                x = np.random.uniform(0.1, 0.9, size=shape).astype(np.float32)
                self._run_bf16(
                    lambda t: paddle.logit(t, eps=eps),
                    lambda a: _np_logit(a, eps),
                    x,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
