#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import glob

cxx_compiler_flags = []
nvcc_flags = []

if os.name == 'nt':
    # setuptools/distutils on Windows uses its own registry-based MSVC detection
    # (ignoring PATH) and picks x86 cl.exe because plat_name='win32'. This causes
    # ptr_size=4, missing _WIN64/_M_X64, broken intrinsics, and linker x86 lib paths.
    #
    # Fix:
    #   DISTUTILS_USE_SDK=1 + MSSdk=1 → distutils uses cl.exe from PATH instead of
    #     its own vcvars-based detection.
    #   x64 cl.exe prepended to PATH → both distutils and NVCC find the right binary.
    #   --compiler-bindir in nvcc_flags → NVCC uses x64 cl.exe explicitly (belt+suspenders).
    #   LIB env var prepended with x64 paths → linker finds x64 CRT/UCRT/UM symbols.

    _x64_cl_dir = None
    for pattern in [
        "C:/Program Files/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/HostX64/x64",
        "C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/HostX64/x64",
    ]:
        cl_dirs = sorted(glob.glob(pattern))
        if cl_dirs:
            _x64_cl_dir = cl_dirs[-1]
            break

    if _x64_cl_dir:
        # Make distutils use our PATH cl.exe, not registry-detected x86.
        os.environ['DISTUTILS_USE_SDK'] = '1'
        os.environ['MSSdk'] = '1'
        os.environ['PATH'] = _x64_cl_dir + os.pathsep + os.environ.get('PATH', '')

        # Prepend x64 lib dirs so linker finds x64 CRT/UCRT/UM (distutils injects x86 ones).
        msvc_x64_lib = _x64_cl_dir.split('bin')[0] + 'lib/x64'
        x64_libs = [msvc_x64_lib]
        for sdk_pattern in ["C:/Program Files (x86)/Windows Kits/10/lib/*/ucrt/x64"]:
            sdk_ucrt_dirs = sorted(glob.glob(sdk_pattern))
            if sdk_ucrt_dirs:
                ucrt_x64 = sdk_ucrt_dirs[-1]
                um_x64 = ucrt_x64.replace('/ucrt/', '/um/')
                x64_libs += [ucrt_x64, um_x64]
                break
        existing_lib = os.environ.get('LIB', '')
        os.environ['LIB'] = os.pathsep.join(x64_libs) + (os.pathsep + existing_lib if existing_lib else '')

    cxx_compiler_flags = [
        "/wd4624",
        "/Zc:alignedNew-",
        "/std:c++17",
        "/Zc:__cplusplus",
    ]

    nvcc_flags = [
        # Explicitly point NVCC at the x64 cl.exe so it doesn't pick an x86 one from PATH.
        # _WIN64 and _M_X64 are then auto-defined by cl.exe itself; we do NOT add -D_WIN64
        # manually because doing so without _M_X64 triggers __faststorefence undefined errors.
        *([f"--compiler-bindir={_x64_cl_dir}"] if _x64_cl_dir else []),
        "-Xcompiler", "/Zc:alignedNew-",
        "-Xcompiler", "/std:c++17",
        "-Xcompiler", "/Zc:__cplusplus",
    ]

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu",
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": nvcc_flags, "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
