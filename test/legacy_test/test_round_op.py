#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, get_device, get_device_place, is_custom_device
from test_activation_op import TestActivation
from utils import dygraph_guard, static_guard

import paddle
from paddle import base
from paddle.base import core

devices = ['cpu', get_device()]


class TestRound(TestActivation):
    def setUp(self):
        self.op_type = "round"
        self.python_api = paddle.round
        self.init_dtype()
        self.init_shape()
        self.init_decimals()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype) * 100
        out = np.round(x, decimals=self.decimals)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {'decimals': self.decimals}
        self.convert_input_output()

    def _get_places(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda() or is_custom_device():
            places.append(get_device_place())
        return places

    def init_shape(self):
        self.shape = [10, 12]

    def init_decimals(self):
        self.decimals = 0

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        pass


class TestRoundEvenTie(TestRound):
    def setUp(self):
        self.op_type = "round"
        self.python_api = paddle.round
        self.init_dtype()
        self.init_shape()
        self.init_decimals()

        np.random.seed(1024)
        x = test_array = np.array(
            [[0.5, 1.5, 2.5], [-0.5, -1.5, -2.5], [1.2, -2.3, 3.0]],
            dtype=np.float32,
        )
        out = np.round(x, decimals=self.decimals)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {'decimals': self.decimals}
        self.convert_input_output()


class TestRound_ZeroDim(TestRound):
    def init_shape(self):
        self.shape = []


class TestRound_decimals1(TestRound):
    def init_decimals(self):
        self.decimals = 2

    def test_round_api(self):
        with dygraph_guard():
            for device in devices:
                if device == 'cpu' or (
                    device == get_device()
                    and (paddle.is_compiled_with_cuda() or is_custom_device())
                ):
                    x_np = (
                        np.random.uniform(-1, 1, self.shape).astype(self.dtype)
                        * 100
                    )
                    out_expect = np.round(x_np, decimals=self.decimals)
                    x_paddle = paddle.to_tensor(
                        x_np, dtype=self.dtype, place=device
                    )
                    y = paddle.round(x_paddle, decimals=self.decimals)
                    np.testing.assert_allclose(y.numpy(), out_expect, rtol=1e-3)


class TestRound_decimals2(TestRound_decimals1):
    def init_decimals(self):
        self.decimals = -1


class TestRoundComplexOp1(TestRound):
    def init_dtype(self):
        self.dtype = np.complex64

    def setUp(self):
        super().setUp()
        x_real = np.random.uniform(-1, 1, self.shape).astype(np.float32) * 100
        x_imag = np.random.uniform(-1, 1, self.shape).astype(np.float32) * 100
        x = x_real + 1j * x_imag
        out = np.round(x, decimals=self.decimals)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'decimals': self.decimals}
        self.convert_input_output()


class TestRoundComplexOp2(TestRoundComplexOp1):
    def init_decimals(self):
        self.decimals = 2


class TestRoundComplexOp3(TestRoundComplexOp1):
    def init_decimals(self):
        self.decimals = -1


class TestRoundComplexOp4(TestRound):
    def init_dtype(self):
        self.dtype = np.complex128

    def setUp(self):
        super().setUp()
        x_real = np.random.uniform(-1, 1, self.shape).astype(np.float64) * 100
        x_imag = np.random.uniform(-1, 1, self.shape).astype(np.float64) * 100
        x = x_real + 1j * x_imag
        out = np.round(x, decimals=self.decimals)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'decimals': self.decimals}
        self.convert_input_output()


class TestRoundComplexOp5(TestRoundComplexOp4):
    def init_decimals(self):
        self.decimals = 2


class TestRoundComplexOp6(TestRoundComplexOp4):
    def init_decimals(self):
        self.decimals = -1


class TestRoundComplexOp7(TestRoundComplexOp4):
    def init_decimals(self):
        self.decimals = -4


class TestRoundComplexOp8(TestRoundComplexOp4):
    def init_decimals(self):
        self.decimals = 4


class TestRoundComplexOp9(TestRoundComplexOp4):
    def init_decimals(self):
        self.decimals = 3


class TestRoundComplexOp10(TestRoundComplexOp4):
    def init_decimals(self):
        self.decimals = -3


class TestRoundInt32(TestRound):
    def init_dtype(self):
        self.dtype = np.int32

    def setUp(self):
        super().setUp()
        x = np.random.randint(-100, 100, self.shape).astype(self.dtype)
        out = np.round(x, decimals=self.decimals)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'decimals': self.decimals}
        self.convert_input_output()


class TestRoundInt64(TestRound):
    def init_dtype(self):
        self.dtype = np.int64

    def setUp(self):
        super().setUp()
        x = np.random.randint(-100, 100, self.shape).astype(self.dtype)
        out = np.round(x, decimals=self.decimals)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'decimals': self.decimals}
        self.convert_input_output()


class TestRoundComplex_ZeroDim(TestRoundComplexOp1):
    def init_shape(self):
        self.shape = []


class TestRoundInt_ZeroDim(TestRoundInt32):
    def init_shape(self):
        self.shape = []


class TestRoundInf(TestRound):
    def setUp(self):
        self.op_type = "round"
        self.python_api = paddle.round
        self.init_dtype()
        self.init_shape()
        self.init_decimals()

        x = np.array(
            [
                np.inf,
                -np.inf,
                *(
                    np.random.uniform(-1, 1, self.shape).astype(self.dtype)
                    * 100
                ),
            ]
        )
        out = np.round(x, decimals=self.decimals)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {'decimals': self.decimals}
        self.convert_input_output()

    def init_shape(self):
        self.shape = [10]

    def init_decimals(self):
        self.decimals = 0

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )


class TestRoundNaN(unittest.TestCase):
    def setUp(self):
        self.op_type = "round"
        self.python_api = paddle.round
        self.init_dtype()
        self.init_shape()
        self.init_decimals()
        self.x = np.array(
            [
                np.nan,
                -np.nan,
                *(
                    np.random.uniform(-1, 1, self.shape).astype(self.dtype)
                    * 100
                ),
            ]
        )
        self.out = np.round(self.x, decimals=self.decimals)

    def init_dtype(self):
        self.dtype = 'float64'

    def init_shape(self):
        self.shape = [10]

    def init_decimals(self):
        self.decimals = 0

    def test_round_nan(self):
        with static_guard():
            places = [core.CPUPlace()]
            if core.is_compiled_with_cuda() or is_custom_device():
                places.append(get_device_place())
            for place in places:
                with paddle.static.program_guard(paddle.static.Program()):
                    input = paddle.static.data(
                        name="input", shape=self.x.shape, dtype=self.x.dtype
                    )
                    output = self.python_api(input, decimals=self.decimals)

                    exe = paddle.static.Executor(place)
                    (result,) = exe.run(
                        feed={'input': self.x}, fetch_list=[output]
                    )
                    nan_mask = np.isnan(self.out)
                    np.testing.assert_array_equal(
                        result[nan_mask], self.out[nan_mask]
                    )
                    np.testing.assert_array_equal(
                        result[~nan_mask], self.out[~nan_mask]
                    )


class TestRoundAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-5, 5, [10, 12]).astype(np.float64)
        self.place = get_device_place()

    def test_dygraph_api(self):
        with dygraph_guard():
            x = paddle.to_tensor(self.x_np)
            out = paddle.round(x)
            out_ref = np.round(self.x_np)
            np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

    def test_static_api(self):
        with static_guard():
            with base.program_guard(base.Program()):
                x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
                out = paddle.round(x)
                exe = base.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            out_ref = np.round(self.x_np)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)


def _mps_available():
    return (
        hasattr(paddle, "is_compiled_with_mps")
        and paddle.is_compiled_with_mps()
        and getattr(paddle, "mps", None) is not None
        and paddle.mps.is_available()
    )


@unittest.skipUnless(_mps_available(), "Paddle is not built with MPS or MPS is unavailable")
class TestRoundMPS(unittest.TestCase):
    """MPS-backend coverage for paddle.round.

    Uses non-halfway values to stay independent of tie-break semantics
    (banker's rounding vs away-from-zero); a dedicated test cross-checks
    against paddle's CPU kernel so any tie-break drift between backends
    surfaces.
    """

    def setUp(self):
        paddle.disable_static()
        paddle.mps.set_device(0)
        np.random.seed(2026)

    def _check_vs_numpy(self, x_np):
        out = paddle.round(paddle.to_tensor(x_np, place="mps")).numpy()
        np.testing.assert_allclose(out, np.round(x_np), rtol=1e-5, atol=1e-6)

    def _check_vs_cpu(self, x_np):
        out_mps = paddle.round(paddle.to_tensor(x_np, place="mps")).numpy()
        out_cpu = paddle.round(paddle.to_tensor(x_np, place="cpu")).numpy()
        np.testing.assert_allclose(out_mps, out_cpu, rtol=0, atol=0)

    def test_basic_shapes(self):
        for shape in [(7,), (3, 4), (2, 3, 5), (2, 3, 4, 5)]:
            with self.subTest(shape=shape):
                # Bias away from exact halfway values so tie-break choice
                # doesn't affect the result.
                x = np.random.uniform(-50.0, 50.0, shape).astype(np.float32)
                x = x + 0.123
                self._check_vs_numpy(x)

    def test_large_1d(self):
        x = np.random.uniform(-1000.0, 1000.0, (4096,)).astype(np.float32) + 0.123
        self._check_vs_numpy(x)

    def test_scalar_tensor(self):
        x = np.array(np.float32(1.7))
        self._check_vs_numpy(x)

    def test_known_values(self):
        x = np.array(
            [0.0, 0.3, -0.3, 1.7, -1.7, 2.4, -2.4, 5.9, -5.9],
            dtype=np.float32,
        )
        self._check_vs_numpy(x)

    def test_integer_inputs_unchanged(self):
        x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        self._check_vs_numpy(x)

    def test_matches_cpu_kernel_on_random(self):
        # Cross-check MPS against the CPU kernel on values that include
        # halfway-adjacent points; any tie-break mismatch will surface here.
        x = np.random.uniform(-10.0, 10.0, (256,)).astype(np.float32)
        self._check_vs_cpu(x)

    def test_dtype_and_place_preserved(self):
        x = np.random.uniform(-5.0, 5.0, (3, 4)).astype(np.float32) + 0.123
        out = paddle.round(paddle.to_tensor(x, place="mps"))
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue("mps" in str(out.place).lower())


if __name__ == "__main__":
    unittest.main()
