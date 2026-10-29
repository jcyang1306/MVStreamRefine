"""Optional CUDA connected-components extension.

Normal builds do not import torch or require nvcc. To opt in, preinstall torch
and build without isolation:
SAM2_BUILD_CUDA=1 pip install --no-build-isolation .
"""

import os

from setuptools import setup


def optional_cuda_extension():
    if os.environ.get("SAM2_BUILD_CUDA") != "1":
        return [], {}
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as exc:
        raise RuntimeError(
            "SAM2_BUILD_CUDA=1 requires torch in the build environment; "
            "use --no-build-isolation"
        ) from exc
    extension = CUDAExtension(
        "sam2._C",
        ["src/sam2/csrc/connected_components.cu"],
        extra_compile_args={
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        },
    )
    return [extension], {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}


extensions, commands = optional_cuda_extension()
setup(ext_modules=extensions, cmdclass=commands)
