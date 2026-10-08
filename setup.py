import glob
import os
import os.path as osp
import pathlib
import platform
import sys

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())
# Optional PEP 440 local version label, e.g. GSPLAT_LOCAL_VERSION=cu126 builds 1.4.0+cu126, so wheels
# for different backends of the same release get different filenames.
LOCAL_VERSION = os.getenv("GSPLAT_LOCAL_VERSION", "")
if LOCAL_VERSION:
    __version__ = f"{__version__}+{LOCAL_VERSION}"

URL = "https://github.com/nerfstudio-project/gsplat"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"
    print(f"Setting MAX_JOBS to {os.environ['MAX_JOBS']}")


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    import torch
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir_v2 = osp.join("gsplat", "cuda", "csrc")
    sources_v2 = glob.glob(osp.join(extensions_dir_v2, "*.cu")) + glob.glob(
        osp.join(extensions_dir_v2, "*.cpp")
    )
    sources_v2 = [path for path in sources_v2 if "hip" not in path]

    undef_macros = []
    define_macros = []

    if sys.platform == "win32":
        define_macros += [("gsplat_EXPORTS", None)]

    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    if torch.version.hip:
        # hipcc/clang: no --use_fast_math / -diag-suppress / -lineinfo.
        # Native float atomics: without this, hipcc lowers every float atomicAdd to a compare-and-swap loop.
        nvcc_flags += ["-O3", "-munsafe-fp-atomics"]
        if LINE_INFO:
            nvcc_flags += ["-gline-tables-only"]
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    else:
        nvcc_flags += ["-O3", "--use_fast_math"]
        if LINE_INFO:
            nvcc_flags += ["-lineinfo"]
        nvcc_flags += ["--expt-relaxed-constexpr"]
        # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses
        nvcc_flags += ["-diag-suppress", "20012,186"]
    extra_compile_args["nvcc"] = nvcc_flags
    if sys.platform == "win32":
        extra_compile_args["nvcc"] += ["-DWIN32_LEAN_AND_MEAN"]

    current_dir = pathlib.Path(__file__).parent.resolve()
    glm_path = os.path.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm")
    if torch.version.hip:
        # torch's hipify rewrites every header it can find under the project
        # directory, which mangles GLM's relative includes and its compiler
        # detection. Keep GLM outside the project tree so it is left alone.
        import shutil
        import tempfile

        glm_hip_path = os.path.join(tempfile.gettempdir(), f"gsplat-{__version__}-glm")
        if not os.path.exists(os.path.join(glm_hip_path, "glm", "glm.hpp")):
            shutil.copytree(glm_path, glm_hip_path, dirs_exist_ok=True)
        glm_path = glm_hip_path
    extension_v2 = CUDAExtension(
        "gsplat.csrc",
        sources_v2,
        include_dirs=[extensions_dir_v2, glm_path],  # glm lives in v2.
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    return [extension_v2]


setup(
    name="gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.7",
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "build",
            "twine",
        ],
    },
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)

if need_to_unset_max_jobs:
    print("Unsetting MAX_JOBS")
    os.environ.pop("MAX_JOBS")
