cmake_minimum_required(VERSION 3.10)
project(env)

set(CMAKE_CXX_STANDARD 17)
set(CUSTOM_FLAGS "-O3 -march=native -ffast-math -fopenmp -mavx2 -mfma -g")# -DNDEBUG")#-g -fopt-info-vec")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUSTOM_FLAGS}")

set(CWD ${CMAKE_CURRENT_SOURCE_DIR})
set(PYTHON_INCLUDE_PATH "/home/hemu/miniconda3/envs/madigan/include/python3.7m")
set(PYTHON_LIBRARIES "/home/hemu/miniconda3/envs/madigan/lib/libpython3.7m.so")
set(HIGHFIVE_INCLUDE_DIR "${CWD}/HighFive/include/")
# set(HDF5_INCLUDE_PATH "/usr/local/hdf5/include")

find_package(pybind11 REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
# find_package(cctz REQUIRED)
# find_package(Boost 1.73 COMPONENTS filesystem system iostreams program_options REQUIRED)

include_directories(${pybind11_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${HIGHFIVE_INCLUDE_DIR})
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${PYTHON_INCLUDE_PATH})
link_libraries(${PYTHON_LIBRARIES})
# link_libraries(${HDF5_HL_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
link_libraries(${HDF5_LIBRARIES})

# include_directories(${Boost_INCLUDE_DIRS})
# include_directories(${Boost_LIBRARY_DIRS})
# link_libraries(${Boost_LIBRARIES})
# link_libraries(cctz::cctz)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(HEADER_FILES Portfolio.h Broker.h Account.h DataSource.h Env.h Config.h)
set(SOURCE_FILES Portfolio.cpp Broker.cpp Account.cpp DataSource.cpp Env.cpp Config.cpp)
set(TESTFOLDER tests/)
# pybind11_add_module(env MODULE env.cpp Portfolio.cpp Env.cpp DataSource.cpp Broker.cpp)
# add_executable(envTest tests/envTest.cpp Portfolio.cpp Env.cpp DataSource.cpp Broker.cpp)

pybind11_add_module(env env.cpp MODULE ${SOURCE_FILES} ${HEADER_FILES})

# Executable for testing from c++
add_executable(envTest tests/envTest.cpp ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(envTest PRIVATE ${TESTFOLDER} ${CWD})

# copy test executable to test folder
add_custom_command(TARGET envTest POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:envTest> ${CWD}/tests/envTest
  COMMENT "Created ${CWD}/tests/envTest"
)

# get_filename_component(PARENT_DIR ${CWD} DIRECTORY)
# set(PY_LIBRARY_DIR ${PARENT_DIR})
# configure_file(${CWD}/build ${PARENT_DIR} COPYONLY)
# target_link_libraries(env PRIVATE ${})
# pybind11_add_module(rollers_testing MODULE rollers_testing.cpp)
# add_executable(envTesting envTest.cpp)
# target_include_directories(main PUBLIC "~/Downloads/gnuplot-iostream")

# message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "HDf INCLUDE DIRS and LIBS ${HDF5_INCLUDE_DIRS} ${HDF5_LIBRARIES}")
message(STATUS "HIGHFIVE INCLUDE DIRS and LIBS ${HIGHFIVE_INCLUDE_DIR}")
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

enable_testing()
add_test(NAME envTest
  COMMAND ../tests/envTest)
add_test(NAME pyenvTest
  COMMAND pytest ../tests/envTest.py)
