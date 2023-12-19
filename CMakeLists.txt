cmake_minimum_required(VERSION 3.0)
project(IPyM LANGUAGES CXX)

# Set C++17 as the standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -m64 -Wall -shared -fPIC")

# Include directories
include_directories(/usr/include/python3.11)
include_directories($(python3 -m pybind11 --includes))
include_directories(/usr/include/eigen3)
include_directories(/usr/include/suitesparse)

# Find Python, pybind11, and OpenMP
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(OpenMP)

# Add the executable
add_library(ipy_selfdual MODULE ipy_selfdual.cpp)

# Link libraries
target_link_libraries(ipy_selfdual PRIVATE
    -L/usr/lib
    -lgomp -lpthread -lm -ldl -lgfortran
    -lmetis
)

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(ipy_selfdual PRIVATE OpenMP::OpenMP_CXX)
endif()

# Get the Python extension suffix
execute_process(
    COMMAND python3-config --extension-suffix
    OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set output file name with Python extension suffix
set_target_properties(ipy_selfdual PROPERTIES
    PREFIX ""
    SUFFIX ${PYTHON_EXTENSION_SUFFIX}
)
