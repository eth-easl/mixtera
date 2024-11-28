include(FetchContent)

# Configure path to modules (for find_package)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")

################### fmt ####################
message(STATUS "Making fmt available.")
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.0.2
)
FetchContent_MakeAvailable(fmt)

################### spdlog ####################
message(STATUS "Making spdlog available.")
set(SPDLOG_FMT_EXTERNAL ON) # Otherwise, we run into linking errors since the fmt version used by spdlog does not match.
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.0
)
FetchContent_MakeAvailable(spdlog)

################### pybind11 ####################

message(STATUS "Making pybind11 available.")

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

################### indicators ####################

message(STATUS "Making indicators available.")

FetchContent_Declare(
  indicators
  GIT_REPOSITORY https://github.com/p-ranav/indicators.git
  GIT_TAG v2.3
)
FetchContent_MakeAvailable(indicators)
target_compile_options(indicators INTERFACE -Wno-zero-as-null-pointer-constant -Wno-sign-compare)

################### abseil ####################

message(STATUS "Making abseil available.")

FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG        20240722.0
  )
FetchContent_MakeAvailable(absl)

# Required for GCC
target_compile_options(absl_flat_hash_map INTERFACE -Wno-pedantic)
target_compile_options(absl_base INTERFACE -Wno-pedantic)


################### Arrow ####################

message(STATUS "Searching for Python.")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Searching for Arrow.")
find_package(Arrow REQUIRED)
message(STATUS "Getting arrow include path using python: ${Python3_EXECUTABLE}")

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pyarrow, sys; sys.stdout.write(pyarrow.get_include())"
    RESULT_VARIABLE _PYARROW_GET_INCLUDE_RESULT
    OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
    ERROR_QUIET
)

if(NOT _PYARROW_GET_INCLUDE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to get pyarrow include directory")
endif()

message(STATUS "Found Pyarrow include path: ${PYARROW_INCLUDE_DIR}")
target_compile_options(Arrow::arrow_shared INTERFACE -Wno-shadow -Wno-unused-parameter -Wno-shadow-field -Wno-extra-semi -Wno-potentially-evaluated-expression)

message(STATUS "Getting pyarrow library dirctory.")
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pyarrow as pa; print(';'.join(pa.get_library_dirs()))"
    OUTPUT_VARIABLE PYARROW_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Found Pyarrow library directory: ${PYARROW_LIBRARY_DIRS}")

find_library(ARROW_PYTHON_LIB
    NAMES arrow_python
    HINTS ${PYARROW_LIBRARY_DIRS}
)

if(NOT ARROW_PYTHON_LIB)
    message(FATAL_ERROR "Failed to find libarrow_python.so in PyArrow library dirs: ${PYARROW_LIBRARY_DIRS}")
else()
    message(STATUS "Found Arrow Python library: ${ARROW_PYTHON_LIB}")
endif()

target_compile_options(Arrow::arrow_shared INTERFACE -Wno-redundant-move)
