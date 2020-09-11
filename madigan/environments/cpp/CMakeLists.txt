cmake_minimum_required(VERSION 3.10)
project(Env)

set(PYTHON_INCLUDE_PATH "/home/hemu/miniconda3/envs/madigan/include/python3.7m")
set(PYTHON_LIBRARIES "/home/hemu/miniconda3/envs/madigan/lib/libpython3.7m.so")

set(CMAKE_CXX_STANDARD 17)
set(CUSTOM_FLAGS "-O3 -march=native -ffast-math -fopenmp -mavx2 -mfma")# -DNDEBUG")#-g -fopt-info-vec")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUSTOM_FLAGS}")

find_package(pybind11 REQUIRED)

include_directories(${pybind11_INCLUDE_DIR})
# find_package(cctz REQUIRED)
# find_package(Boost 1.73 COMPONENTS filesystem system iostreams program_options REQUIRED)

# include_directories(${Boost_INCLUDE_DIRS})
# include_directories(${Boost_LIBRARY_DIRS})
include_directories(${PYTHON_INCLUDE_PATH})
# link_directories(${PYTHON_INCLUDE_PATH})
# link_libraries(${Boost_LIBRARIES})
# link_libraries(cctz::cctz)
link_libraries(${PYTHON_LIBRARIES})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(env MODULE env.cpp Portfolio.cpp Env.cpp)
# pybind11_add_module(rollers_testing MODULE rollers_testing.cpp)
# add_executable(envTesting envTest.cpp)
# target_include_directories(main PUBLIC "~/Downloads/gnuplot-iostream")

# message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Python Version: ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

#If boost version is newer than CMake version then this is needed
# if(NOT TARGET Boost::filesystem)
#   add_library(Boost::filesystem IMPORTED INTERFACE)
#   set_property(TARGET Boost::filesystem PROPERTY
#     INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
#   set_property(TARGET Boost::filesystem PROPERTY
#     INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
#   message("WARNING - Boost version is newer than CMake version")
# endif()

# enable_testing()
# add_test(NAME envTest
#   COMMAND )
