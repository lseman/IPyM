cmake_minimum_required(VERSION 3.0)
project(IPyM LANGUAGES CXX)

# Set C++17 as the standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -m64 -Wall -shared -fPIC")

# Attempt to include CPM.cmake from provided path
include(cmake/CPM.cmake OPTIONAL)

# Check if CPM was included, if not, fetch and include CPM
if(NOT COMMAND CPMAddPackage)
  # Include FetchContent module
  include(FetchContent)
  # Declare CPM.cmake as a FetchContent
  FetchContent_Declare(
    CPM
    GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
    GIT_TAG v0.40.2)
  # Fetch CPM
  FetchContent_MakeAvailable(CPM)
  # Include CPM.cmake after it has been fetched
  include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()


# Include directories
cpmaddpackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 GIT_TAG v2.13.6)


include_directories(/usr/include/eigen3)
include_directories(/usr/include/suitesparse)

# Find Python, pybind11, and OpenMP
find_package(OpenMP)

# Add the executable
add_library(ipy_selfdual MODULE ipy_selfdual.cpp)

# Link libraries
target_link_libraries(ipy_selfdual PRIVATE
    -L/usr/lib
    -L/usr/lib/x86_64-linux-gnu
    -lgomp -lpthread -lm -ldl -lgfortran
    -lmetis -lcholmod
)

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(ipy_selfdual PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(ipy_selfdual PRIVATE pybind11::module)

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
