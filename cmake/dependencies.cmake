include(FetchContent)

# Configure path to modules (for find_package)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")

message(STATUS "Making pybind11 available.")

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

find_package(Arrow REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pyarrow, sys; sys.stdout.write(pyarrow.get_include())"
    RESULT_VARIABLE _PYARROW_GET_INCLUDE_RESULT
    OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
    ERROR_QUIET
)

if(NOT _PYARROW_GET_INCLUDE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to get pyarrow include directory")
endif()

message(STATUS "PyArrow include directory: ${PYARROW_INCLUDE_DIR}")

target_compile_options(Arrow::arrow_shared INTERFACE -Wno-shadow -Wno-unused-parameter -Wno-shadow-field -Wno-extra-semi)