# FindArrow.cmake - Find Arrow installation from PyArrow

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Get PyArrow paths
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pyarrow; print(';'.join(pyarrow.get_library_dirs()))"
    OUTPUT_VARIABLE PYARROW_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYARROW_LIB_RESULT
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pyarrow; print(pyarrow.get_include())"
    OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYARROW_INC_RESULT
)

if(NOT PYARROW_LIB_RESULT EQUAL 0 OR NOT PYARROW_INC_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to get PyArrow paths")
endif()

# Find the libraries
find_library(ARROW_LIB
    NAMES arrow
    HINTS ${PYARROW_LIBRARY_DIRS}
    REQUIRED
)

find_library(ARROW_PYTHON_LIB
    NAMES arrow_python
    HINTS ${PYARROW_LIBRARY_DIRS}
    REQUIRED
)

# Find headers
find_path(ARROW_INCLUDE_DIR
    NAMES arrow/api.h
    HINTS ${PYARROW_INCLUDE_DIR}
    REQUIRED
)

# Create imported targets
if(NOT TARGET Arrow::arrow_shared)
    add_library(Arrow::arrow_shared SHARED IMPORTED)
    set_target_properties(Arrow::arrow_shared PROPERTIES
        IMPORTED_LOCATION ${ARROW_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR}
    )
endif()

if(NOT TARGET Arrow::arrow_python_shared)
    add_library(Arrow::arrow_python_shared SHARED IMPORTED)
    set_target_properties(Arrow::arrow_python_shared PROPERTIES
        IMPORTED_LOCATION ${ARROW_PYTHON_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR}
        INTERFACE_LINK_LIBRARIES Arrow::arrow_shared
    )
endif()

# Set standard variables
set(Arrow_FOUND TRUE)
set(ARROW_FOUND TRUE)
set(Arrow_INCLUDE_DIRS ${ARROW_INCLUDE_DIR})
set(Arrow_LIBRARIES ${ARROW_LIB})

message(STATUS "Found Arrow: ${ARROW_LIB}")
message(STATUS "Found Arrow Python: ${ARROW_PYTHON_LIB}")
message(STATUS "Arrow include dir: ${ARROW_INCLUDE_DIR}")
