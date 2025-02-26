"""setup file for the project."""
# code inspired by https://github.com/navdeep-G/setup.py

import io
import os
import pathlib
import subprocess
import sysconfig
import socket
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build import build

# Package meta-data.
NAME = "mixtera"
DESCRIPTION = "A platform for LLM training data mixture."

URL = "https://github.com/eth-easl/mixtera"
URL_DOKU = "https://github.com/eth-easl/mixtera"
URL_GITHUB = "https://github.com/eth-easl/mixtera"
URL_ISSUES = "https://github.com/eth-easl/mixtera"
EMAIL = "maximilian.boether@inf.ethz.ch"
AUTHOR = "See contributing.md"
REQUIRES_PYTHON = ">=3.10"
KEYWORDS = [""]
REQUIRED = [""]
EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's _version.py module as a dictionary.
about = {}
project_slug = NAME

EXTENSION_BUILD_DIR = pathlib.Path(here) / "libbuild"


def _get_env_variable(name: str, default: str = "OFF") -> str:
    if name not in os.environ.keys():
        return default
    return os.environ[name]


class CustomBuild(build):
    def finalize_options(self):
        # On clusters with an NFS - such as the alps/clariden supercomputer - when running a job with multiple nodes involved,
        # if they all install Mixtera at the same time, we run into concurrency issues if they share the same dierctory.
        # Hence, we have to force a directory based on the hostname
        super().finalize_options()
        hostname = socket.gethostname().strip() or "nohostname"
        self.build_base = f"build_{hostname}"


class CMakeExtension(Extension):
    def __init__(
        self, name: str, cmake_lists_dir: str = ".", sources: list = [], **kwa: dict
    ) -> None:
        Extension.__init__(self, name, sources=sources, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = None
        self.build_temp = None
        self.build_lib = None

    def finalize_options(self):
        super().finalize_options()
        self.set_undefined_options("build", ("build_base", "build_base"))
        plat_spec = ".{0}-{1}".format(
            sysconfig.get_platform(), sysconfig.get_config_var("py_version_nodot")
        )
        self.build_temp = os.path.join(self.build_base, "temp" + plat_spec)
        self.build_lib = os.path.join(self.build_base, "lib" + plat_spec)
        print(f"CMakeBuild.finalize_options:")
        print(f"  build_base: {self.build_base}")
        print(f"  build_temp: {self.build_temp}")
        print(f"  build_lib: {self.build_lib}")

    def build_extension(self, ext):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable")

        cfg = _get_env_variable("MIXTERA_BUILDTYPE", "Release")

        print(f"Using build type {cfg} for Mixtera.")
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        # Get the absolute path to the directory where the extension will be placed
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}"]

        build_temp = pathlib.Path(self.build_temp) / ext.name
        shutil.rmtree(
            build_temp, ignore_errors=True
        )  # Clean directory if exists to avoid caching issues.
        build_temp.mkdir(parents=True, exist_ok=False)
        print(f"Building mixtera at {build_temp}")

        # Config and build the extension
        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=str(build_temp)
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "-j", "8", "--config", cfg], cwd=str(build_temp)
        )


# Where the magic happens:
setup(
    name=NAME,
    version="1.0.0",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    project_urls={"Bug Tracker": URL_ISSUES, "Source Code": URL_GITHUB},
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*", "tests.*.*"]
    ),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points is is required for testing the Python scripts
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    keywords=KEYWORDS,
    scripts=[
        "mixtera/cli/mixtera-cli",
        "mixtera/cli/mixtera",  # Duplication of mixtera-cli
        "mixtera/network/server/mixtera-server",
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[
        CMakeExtension("mixtera.core.query.chunker.chunker_extension"),
    ],
    cmdclass={"build_ext": CMakeBuild, "build": CustomBuild},
)
