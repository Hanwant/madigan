cmake_minimum_required( VERSION 3.1)
project(synth_c)


set(CMAKE_CXX_STANDARD 17)
set(CUSTOM_FLAGS "-O3 -march=native -ffast-math -fopenmp")
set(CMAKE_CXX_FLAG "${CMAKE_CXX_FLAGS} ${CUSTOM_FLAGS}")

set(PYTHON_INCLUDE_PATH "/home/hemu/miniconda3/envs/madigan/include/python3.7m")
set(PYTHON_LIBRARIES "/home/hemu/miniconda3/envs/madigan/lib/libpython3.7m.so")


find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIR})

include_directories(${PYTHON_INCLUDE_PATH})
link_libraries(${PYTHON_LIBRARIES})

message(STATUS "Python Version: ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
pybind11_add_module(synth_c MODULE synth.cpp synth_c.cpp)
