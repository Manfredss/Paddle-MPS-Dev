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
bfloat16 dtype coverage for the MPS "comparison" kernel family.

Family: equal / not_equal / less_than / less_equal / greater_than /
greater_equal.

Every comparison op gains bfloat16 input support (in addition to the existing
float32 / float16 / int32 / int64 inputs).  The OUTPUT of every comparison op
is always bool, so:

- The registered MPS kernel plumbs ``GetMPSDataType(x.dtype())`` for the inputs
  (now including ``MPSDataTypeBFloat16`` on the macOS-14 SDK) while the output
  buffer stays ``MPSDataTypeBool``.
- bfloat16 (MPSDataTypeBFloat16) only exists in the macOS-14 SDK and only works
  at RUNTIME on macOS 14+.  This CI host may be older, so we PROBE the dtype with
  a trivial op inside ``try/except`` and ``skipTest`` when it is unsupported.

bf16 has only ~3 significant decimal digits, so comparison inputs are chosen to
be WELL SEPARATED (no near-ties) — that way the bf16 rounding of the inputs does
not flip any comparison relative to the float32 oracle, and the bool result can
be compared EXACTLY.  The float32 numpy oracle is computed on the same values
that bf16 sees (we round the float32 array through bf16 by going via the mps
tensor) so the reference and the kernel agree bit-for-bit on the bool result.
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

    bfloat16 / complex64 map to MPSGraph data types that only exist in the
    macOS-14 SDK and only function on macOS 14+ at runtime.  We create a tiny
    tensor of the dtype on the mps place, run a trivial add, and force
    materialisation; ANY exception means the dtype is unsupported here.
    """
    if not _mps_available():
        return False
    try:
        if paddle_dtype == paddle.bfloat16:
            t = paddle.to_tensor(
                np.zeros((2,), dtype=np.float32), place="mps"
            ).astype("bfloat16")
        elif paddle_dtype == paddle.complex64:
            t = paddle.to_tensor(
                np.zeros((2,), dtype=np.complex64), place="mps"
            )
        else:
            t = paddle.to_tensor(np.zeros((2,), dtype=np.float32), place="mps")
        out = t + t
        # Force the graph to actually run / materialise.
        _ = out.numpy()
        return True
    except Exception:
        return False


_BF16_SUPPORTED = _supports(paddle.bfloat16) if PADDLE_AVAILABLE else False

_SHAPES = [(6,), (3, 4), (2, 3, 4)]

# The six comparison ops: (name, paddle op, numpy op).
_COMPARE_OPS = (
    ("equal", lambda a, b: paddle.equal(a, b), np.equal),
    ("not_equal", lambda a, b: paddle.not_equal(a, b), np.not_equal),
    ("less_than", lambda a, b: paddle.less_than(a, b), np.less),
    ("less_equal", lambda a, b: paddle.less_equal(a, b), np.less_equal),
    ("greater_than", lambda a, b: paddle.greater_than(a, b), np.greater),
    (
        "greater_equal",
        lambda a, b: paddle.greater_equal(a, b),
        np.greater_equal,
    ),
)


@unittest.skipUnless(
    _mps_available(),
    "PaddlePaddle is not built with MPS or MPS is unavailable",
)
class TestMPSComparisonBf16(unittest.TestCase):
    """bfloat16 input coverage for the comparison family (bool output)."""

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()
        paddle.mps.set_device(0)

    def setUp(self):
        np.random.seed(2026)
        paddle.seed(2026)
        if not _BF16_SUPPORTED:
            self.skipTest(
                "bfloat16 not supported on this MPS runtime (needs macOS 14+)"
            )

    def _to_bf16_mps(self, x32):
        """float32 numpy -> bfloat16 mps tensor (rounds through bf16)."""
        return paddle.to_tensor(x32, place="mps").astype("bfloat16")

    def _bf16_roundtrip(self, x32):
        """Round a float32 array through bf16 so the oracle matches the kernel.

        We materialise the bf16 mps tensor back to float32 numpy; that value is
        exactly what the MPSGraph comparison sees, so the numpy oracle computed
        on it agrees bit-for-bit on the bool result.
        """
        return self._to_bf16_mps(x32).astype("float32").numpy()

    def _check_op(self, name, p_op, n_op, x32, y32):
        x_bf = self._to_bf16_mps(x32)
        y_bf = self._to_bf16_mps(y32)
        out = p_op(x_bf, y_bf)

        # Output of a comparison is always bool, on the mps place.
        self.assertEqual(out.dtype, paddle.bool)
        self.assertTrue("mps" in str(out.place).lower())

        # Inputs were genuinely bfloat16.
        self.assertEqual(x_bf.dtype, paddle.bfloat16)
        self.assertEqual(y_bf.dtype, paddle.bfloat16)

        # Oracle on the bf16-rounded values -> exact bool agreement.
        xr = self._bf16_roundtrip(x32)
        yr = self._bf16_roundtrip(y32)
        ref = n_op(xr, yr)
        np.testing.assert_array_equal(out.numpy(), ref)

    def test_compare_random_bf16(self):
        # Well-separated values (>= 0.5 apart after bf16 rounding for the
        # generated magnitudes) so bf16 rounding never flips a comparison.
        for shape in _SHAPES:
            x = (np.random.randn(*shape) * 2.0).astype(np.float32)
            y = (np.random.randn(*shape) * 2.0).astype(np.float32)
            for name, p_op, n_op in _COMPARE_OPS:
                with self.subTest(op=name, shape=shape):
                    self._check_op(name, p_op, n_op, x, y)

    def test_compare_with_equal_elements_bf16(self):
        # Include exactly-equal elements to exercise the == / <= / >= edges.
        # Integer-valued floats are representable exactly in bf16.
        x = np.array([-4.0, -1.0, 0.0, 1.0, 2.0, 5.0], dtype=np.float32)
        y = np.array([-4.0, 1.0, 0.0, -1.0, 3.0, 5.0], dtype=np.float32)
        for name, p_op, n_op in _COMPARE_OPS:
            with self.subTest(op=name):
                self._check_op(name, p_op, n_op, x, y)

    def test_compare_self_bf16(self):
        # x compared to itself: equal/less_equal/greater_equal all True;
        # not_equal/less_than/greater_than all False.
        x = np.array([-3.0, -0.5, 0.0, 0.5, 4.0, 7.0], dtype=np.float32)
        for name, p_op, n_op in _COMPARE_OPS:
            with self.subTest(op=name):
                self._check_op(name, p_op, n_op, x, x)


if __name__ == "__main__":
    unittest.main(verbosity=2)
