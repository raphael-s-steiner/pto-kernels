from pathlib import Path

import pytest
import torch

from cube.jit_util_hadamard_cube import jit_compile as jit_compile_cube
from jit_util_common import normalize_npu_device
from onesided.jit_util_hadamard_onesided import jit_compile as jit_compile_onesided
from standard.jit_util_hadamard import jit_compile


def pytest_addoption(parser):
    parser.addoption(
        "--npu",
        action="store",
        default="npu:0",
        help="NPU device (examples: 0, npu:0, '0', 'npu:0').",
    )


@pytest.fixture(scope="session")
def npu_device(request):
    raw = request.config.getoption("--npu")
    return normalize_npu_device(raw)


@pytest.fixture(scope="session", autouse=True)
def setup_npu_device(npu_device):
    torch.npu.set_device(npu_device)


@pytest.fixture(scope="session")
def hadamard_kernel(npu_device):
    base = Path(__file__).resolve().parent
    src = base / "standard" / "fast_hadamard.cpp"
    return jit_compile(str(src), verbose=True, device=npu_device)


@pytest.fixture(scope="session")
def cube_hadamard_kernel(npu_device):
    base = Path(__file__).resolve().parent
    src = base / "cube" / "fast_hadamard_cube.cpp"
    return jit_compile_cube(str(src), verbose=True, device=npu_device)


@pytest.fixture(scope="session")
def onesided_hadamard_kernel(npu_device):
    base = Path(__file__).resolve().parent
    src = base / "onesided" / "fast_hadamard_onesided.cpp"
    return jit_compile_onesided(str(src), verbose=True, device=npu_device)
