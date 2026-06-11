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
bfloat16 dtype coverage for the MPS "reductions" kernel family.

Family / dtype policy (this task only ADDS bfloat16 to the reductions):
- sum / max / min / prod : float, float16, int32, int64  +  bfloat16 (gated).
- mean                   : float, float16                +  bfloat16 (gated).

bfloat16 maps to MPSDataTypeBFloat16, which exists only in the macOS-14
MetalPerformanceShadersGraph SDK and requires macOS 14 AT RUNTIME. This CI host
may lack it, so every bf16 subtest PROBES support first (a trivial add on a tiny
bf16 mps tensor) and skips on any exception.

bfloat16 has ~3 decimal digits of precision (8-bit mantissa) -- looser than
float16 -- so the mps bf16 output is compared to a float32 numpy oracle with
loose tolerance (rtol=4e-2, atol=4e-2). We never compare bf16 against the CPU
backend (CPU may not register bf16 for these ops).
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
    """Return True if a trivial op on an MPS tensor of ``paddle_dtype`` works.

    bfloat16 (and complex64) require the macOS-14 MPSGraph SDK at runtime;
    probe by building a tiny tensor of that dtype on the mps place and running
    a trivial add. Any exception -> the dtype is unsupported on this host.
    """
    if not _mps_available():
        return False
    try:
        base = paddle.to_tensor(np.zeros((2,), dtype=np.float32), place="mps")
        t = base.astype(paddle_dtype)
        out = t + t
        # Force materialisation so a lazily-thrown failure surfaces here.
        _ = out.astype("float32").numpy()
        return True
    except Exception:
        return False


# bfloat16 has an 8-bit mantissa (~3 decimal digits); use loose tolerances.
BF16_RTOL = 4e-2
BF16_ATOL = 4e-2

_SHAPES = [(6,), (3, 4), (2, 3, 4)]


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSReductionBF16Base(unittest.TestCase):
    """Common setUp + bf16 helpers for the reductions-family dtype tests."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)
        cls.bf16_ok = _supports(paddle.bfloat16)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    def _skip_if_no_bf16(self):
        if not self.bf16_ok:
            self.skipTest(
                "bfloat16 not supported (needs macOS 14 MPSGraph SDK)"
            )

    def _assert_bf16_out(self, out):
        """The MPS output must be bfloat16 and live on the mps place."""
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue("mps" in str(out.place).lower())

    def _run_bf16(self, fn, x32, rtol=BF16_RTOL, atol=BF16_ATOL):
        """Run ``fn`` on a bfloat16 MPS tensor; compare to the float32 oracle.

        bfloat16 has no numpy dtype, so create the input as float32 on the mps
        place and ``.astype(paddle.bfloat16)``. ``fn`` takes a paddle tensor and
        returns a paddle tensor. Returns the MPS bf16 output for extra asserts.
        """
        self._skip_if_no_bf16()
        x_bf16 = paddle.to_tensor(x32, place="mps").astype("bfloat16")
        out = fn(x_bf16)
        self._assert_bf16_out(out)
        ref = fn(paddle.to_tensor(x32, place="cpu")).numpy().astype(np.float32)
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, rtol=rtol, atol=atol
        )
        return out


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


class TestMPSSumBF16(_MPSReductionBF16Base):
    def test_sum_all(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                # Small magnitudes keep the bf16 accumulation error in check.
                x = (np.random.randn(*shape) * 1.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.sum(t), x)

    def test_sum_axis(self):
        for shape in _SHAPES:
            if len(shape) < 2:
                continue
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.sum(t, axis=-1), x)

    def test_sum_keepdim(self):
        x = (np.random.randn(3, 4) * 1.0).astype(np.float32)
        self._run_bf16(lambda t: paddle.sum(t, axis=1, keepdim=True), x)


# ---------------------------------------------------------------------------
# max
# ---------------------------------------------------------------------------


class TestMPSMaxBF16(_MPSReductionBF16Base):
    def test_max_all(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(lambda t: paddle.max(t), x)

    def test_max_axis(self):
        for shape in _SHAPES:
            if len(shape) < 2:
                continue
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(lambda t: paddle.max(t, axis=-1), x)

    def test_max_keepdim(self):
        x = (np.random.randn(3, 4) * 1.5).astype(np.float32)
        self._run_bf16(lambda t: paddle.max(t, axis=0, keepdim=True), x)


# ---------------------------------------------------------------------------
# min
# ---------------------------------------------------------------------------


class TestMPSMinBF16(_MPSReductionBF16Base):
    def test_min_all(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(lambda t: paddle.min(t), x)

    def test_min_axis(self):
        for shape in _SHAPES:
            if len(shape) < 2:
                continue
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.5).astype(np.float32)
                self._run_bf16(lambda t: paddle.min(t, axis=-1), x)

    def test_min_keepdim(self):
        x = (np.random.randn(3, 4) * 1.5).astype(np.float32)
        self._run_bf16(lambda t: paddle.min(t, axis=0, keepdim=True), x)


# ---------------------------------------------------------------------------
# prod
# ---------------------------------------------------------------------------


class TestMPSProdBF16(_MPSReductionBF16Base):
    def test_prod_axis(self):
        # Products grow quickly, so reduce over a single small axis with
        # values near 1 to keep the bf16 result well-conditioned.
        for shape in _SHAPES:
            if len(shape) < 2:
                continue
            with self.subTest(shape=shape):
                x = (0.5 + np.random.rand(*shape)).astype(np.float32)
                self._run_bf16(lambda t: paddle.prod(t, axis=-1), x)

    def test_prod_keepdim(self):
        x = (0.5 + np.random.rand(3, 4)).astype(np.float32)
        self._run_bf16(lambda t: paddle.prod(t, axis=1, keepdim=True), x)

    def test_prod_small_vector(self):
        x = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
        self._run_bf16(lambda t: paddle.prod(t), x)


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------


class TestMPSMeanBF16(_MPSReductionBF16Base):
    def test_mean_all(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.mean(t), x)

    def test_mean_axis(self):
        for shape in _SHAPES:
            if len(shape) < 2:
                continue
            with self.subTest(shape=shape):
                x = (np.random.randn(*shape) * 1.0).astype(np.float32)
                self._run_bf16(lambda t: paddle.mean(t, axis=-1), x)

    def test_mean_keepdim(self):
        x = (np.random.randn(3, 4) * 1.0).astype(np.float32)
        self._run_bf16(lambda t: paddle.mean(t, axis=1, keepdim=True), x)


# ---------------------------------------------------------------------------
# integer (int32 / int64) coverage for sum / max / min / prod
# ---------------------------------------------------------------------------
#
# sum/max/min/prod register int32 and int64 on MPS (matching the CPU/GPU
# backends). Integer reductions must be EXACT, so -- unlike the bf16 tests
# above, which use a float32 oracle -- these compare the MPS integer output
# bit-for-bit against the CPU integer backend. Mixed-sign inputs guard against
# any accidental float intermediate (which would round/truncate differently)
# and against sign/overflow errors. mean is intentionally excluded: the MPS
# mean kernel calls MPSGraph meanOfTensor: directly and is registered for
# float/float16/bfloat16 only (the CPU/GPU MeanKernel cast integers to float32
# and back, a path the MPS kernel does not implement).


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class _MPSReductionIntBase(unittest.TestCase):
    """Common helpers for exact int32/int64 reduction tests."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)

    def _assert_int_out(self, out, np_dtype):
        """The MPS output must keep the integer dtype and live on mps."""
        expected = paddle.to_tensor(
            np.zeros((1,), dtype=np_dtype), place="cpu"
        ).dtype
        self.assertEqual(out.dtype, expected)
        self.assertTrue("mps" in str(out.place).lower())

    def _run_int(self, fn, x_np):
        """Run ``fn`` on an integer MPS tensor; require an EXACT CPU match.

        ``x_np`` is an integer numpy array; its dtype drives the paddle dtype.
        ``fn`` takes a paddle tensor and returns a paddle tensor.
        """
        x_mps = paddle.to_tensor(x_np, place="mps")
        out = fn(x_mps)
        self._assert_int_out(out, x_np.dtype)
        ref = fn(paddle.to_tensor(x_np, place="cpu")).numpy()
        np.testing.assert_array_equal(out.numpy(), ref)
        return out


_INT_DTYPES = [np.int32, np.int64]


class TestMPSSumInt(_MPSReductionIntBase):
    def test_sum_all_mixed_signs(self):
        for dt in _INT_DTYPES:
            with self.subTest(dtype=dt):
                x = np.array([-5, 3, -2, 7], dtype=dt)
                self._run_int(lambda t: paddle.sum(t), x)

    def test_sum_axis_mixed_signs(self):
        for dt in _INT_DTYPES:
            with self.subTest(dtype=dt):
                x = np.array([[-5, 3, -2, 7], [4, -8, 1, -6]], dtype=dt)
                self._run_int(lambda t: paddle.sum(t, axis=-1), x)

    def test_sum_keepdim(self):
        for dt in _INT_DTYPES:
            with self.subTest(dtype=dt):
                x = np.array([[1, -2, 3], [-4, 5, -6]], dtype=dt)
                self._run_int(lambda t: paddle.sum(t, axis=1, keepdim=True), x)


class TestMPSMaxInt(_MPSReductionIntBase):
    def test_max_all_mixed_signs(self):
        for dt in _INT_DTYPES:
            with self.subTest(dtype=dt):
                x = np.array([-5, 3, -2, 7], dtype=dt)
                self._run_int(lambda t: paddle.max(t), x)

    def test_max_axis_mixed_signs(self):
        for dt in _INT_DTYPES:
            with self.subTest(dtype=dt):
                x = np.array([[-5, 3, -2, 7], [4, -8, 1, -6]], dtype=dt)
                self._run_int(lambda t: paddle.max(t, axis=-1), x)


class TestMPSMinInt(_MPSReductionIntBase):
    def test_min_all_mixed_signs(self):
        for dt in _INT_DTYPES:
            with self.subTest(dtype=dt):
                x = np.array([-5, 3, -2, 7], dtype=dt)
                self._run_int(lambda t: paddle.min(t), x)

    def test_min_axis_mixed_signs(self):
        for dt in _INT_DTYPES:
            with self.subTest(dtype=dt):
                x = np.array([[-5, 3, -2, 7], [4, -8, 1, -6]], dtype=dt)
                self._run_int(lambda t: paddle.min(t, axis=-1), x)


class TestMPSProdInt(_MPSReductionIntBase):
    def test_prod_axis_mixed_signs(self):
        for dt in _INT_DTYPES:
            with self.subTest(dtype=dt):
                # Small magnitudes keep the integer product in range.
                x = np.array([[-2, 3, -1], [4, -1, 2]], dtype=dt)
                self._run_int(lambda t: paddle.prod(t, axis=-1), x)

    def test_prod_keepdim(self):
        for dt in _INT_DTYPES:
            with self.subTest(dtype=dt):
                x = np.array([[1, -2, 3], [-1, 2, -1]], dtype=dt)
                self._run_int(lambda t: paddle.prod(t, axis=1, keepdim=True), x)

    def test_prod_empty_fills_one(self):
        # Empty input takes the early-out branch that fills the output with the
        # multiplicative identity via std::fill_n(out_data, .., static_cast<T>(1)).
        # Verify that integer 1 (not 0 / garbage) is produced for int dtypes.
        for dt in _INT_DTYPES:
            with self.subTest(dtype=dt):
                x = np.ones((0,), dtype=dt)
                out = paddle.prod(paddle.to_tensor(x, place="mps"))
                self.assertEqual(int(out.numpy()), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
