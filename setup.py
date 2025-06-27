from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build import build
import subprocess
import pathlib
import os
import sysconfig
import socket
import shutil

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

setup(
    ext_modules=[CMakeExtension("mixtera.core.query.chunker.chunker_extension")],
    cmdclass={"build_ext": CMakeBuild, "build": CustomBuild},
    scripts=[
        "mixtera/cli/mixtera-cli",
        "mixtera/network/server/mixtera-server"
    ],
)
