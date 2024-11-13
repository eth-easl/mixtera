set(MIXTERA_CHUNKER_SOURCES
    ./src/chunker.cpp
)

# Explicitly set all header files so that IDEs will recognize them as part of the project
set(MIXTERA_CHUNKER_HEADERS
    ./include/chunker.hpp
)

pybind11_add_module(chunker_extension ${MIXTERA_CHUNKER_SOURCES}  ${MIXTERA_CHUNKER_HEADERS})
target_include_directories(chunker_extension PUBLIC include  ${PYARROW_INCLUDE_DIR} ${Arrow_INCLUDE_DIR} ${ArrowPython_INCLUDE_DIRS})
target_compile_options(chunker_extension PRIVATE ${MIXTERA_COMPILE_OPTIONS} -Wno-unused-function)
target_link_libraries(chunker_extension PRIVATE Arrow::arrow_shared ${ARROW_PYTHON_LIB}) 
set_target_properties(chunker_extension PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
set_target_properties(chunker_extension PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)