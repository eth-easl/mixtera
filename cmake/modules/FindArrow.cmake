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

message(STATUS "PyArrow library dirs: ${PYARROW_LIBRARY_DIRS}")
message(STATUS "PyArrow include dir: ${PYARROW_INCLUDE_DIR}")

# List all files in the library directory for debugging
execute_process(
    COMMAND ls -la ${PYARROW_LIBRARY_DIRS}
    OUTPUT_VARIABLE LIB_DIR_CONTENTS
    ERROR_QUIET
)
message(STATUS "Library directory contents:\n${LIB_DIR_CONTENTS}")

# Try to find Arrow libraries with various possible names
find_library(ARROW_LIB
    NAMES 
        arrow 
        libarrow 
        arrow.so 
        libarrow.so
        arrow.dylib
        libarrow.dylib
    HINTS ${PYARROW_LIBRARY_DIRS}
    PATH_SUFFIXES . lib
)

# If standard names don't work, try to find any arrow-related library
if(NOT ARROW_LIB)
    file(GLOB ARROW_CANDIDATES 
        "${PYARROW_LIBRARY_DIRS}/*arrow*"
        "${PYARROW_LIBRARY_DIRS}/lib*arrow*"
    )
    message(STATUS "Arrow library candidates: ${ARROW_CANDIDATES}")
    
    foreach(candidate ${ARROW_CANDIDATES})
        if(candidate MATCHES "\\.(so|dylib|dll)(\\..*)?$")
            set(ARROW_LIB ${candidate})
            message(STATUS "Using Arrow library: ${ARROW_LIB}")
            break()
        endif()
    endforeach()
endif()

# Try to find Arrow Python library
find_library(ARROW_PYTHON_LIB
    NAMES 
        arrow_python 
        libarrow_python
        arrow_python.so
        libarrow_python.so
        arrow_python.dylib
        libarrow_python.dylib
    HINTS ${PYARROW_LIBRARY_DIRS}
    PATH_SUFFIXES . lib
)

# If not found, look for any arrow_python related files
if(NOT ARROW_PYTHON_LIB)
    file(GLOB ARROW_PYTHON_CANDIDATES 
        "${PYARROW_LIBRARY_DIRS}/*arrow_python*"
        "${PYARROW_LIBRARY_DIRS}/lib*arrow_python*"
    )
    message(STATUS "Arrow Python library candidates: ${ARROW_PYTHON_CANDIDATES}")
    
    foreach(candidate ${ARROW_PYTHON_CANDIDATES})
        if(candidate MATCHES "\\.(so|dylib|dll)(\\..*)?$")
            set(ARROW_PYTHON_LIB ${candidate})
            message(STATUS "Using Arrow Python library: ${ARROW_PYTHON_LIB}")
            break()
        endif()
    endforeach()
endif()

# Find headers
find_path(ARROW_INCLUDE_DIR_FOUND
    NAMES arrow/api.h
    HINTS ${PYARROW_INCLUDE_DIR}
    PATH_SUFFIXES . include
)

# Check if we found everything we need
if(NOT ARROW_LIB)
    message(FATAL_ERROR "Could not find Arrow library in ${PYARROW_LIBRARY_DIRS}")
endif()

if(NOT ARROW_INCLUDE_DIR_FOUND)
    message(FATAL_ERROR "Could not find Arrow headers in ${PYARROW_INCLUDE_DIR}")
endif()

# Create imported targets
if(NOT TARGET Arrow::arrow_shared)
    add_library(Arrow::arrow_shared SHARED IMPORTED)
    set_target_properties(Arrow::arrow_shared PROPERTIES
        IMPORTED_LOCATION ${ARROW_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR_FOUND}
    )
endif()

if(ARROW_PYTHON_LIB AND NOT TARGET Arrow::arrow_python_shared)
    add_library(Arrow::arrow_python_shared SHARED IMPORTED)
    set_target_properties(Arrow::arrow_python_shared PROPERTIES
        IMPORTED_LOCATION ${ARROW_PYTHON_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR_FOUND}
        INTERFACE_LINK_LIBRARIES Arrow::arrow_shared
    )
endif()

# Set standard variables
set(Arrow_FOUND TRUE)
set(ARROW_FOUND TRUE)
set(Arrow_INCLUDE_DIRS ${ARROW_INCLUDE_DIR_FOUND})
set(Arrow_LIBRARIES ${ARROW_LIB})

message(STATUS "Found Arrow: ${ARROW_LIB}")
if(ARROW_PYTHON_LIB)
    message(STATUS "Found Arrow Python: ${ARROW_PYTHON_LIB}")
else()
    message(WARNING "Arrow Python library not found - some features may not work")
endif()
message(STATUS "Arrow include dir: ${ARROW_INCLUDE_DIR_FOUND}")
