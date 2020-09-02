cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(convTest convModel.cpp Synth.cpp Generator.cpp)
target_link_libraries(convTest "${TORCH_LIBRARIES}")
set_property(TARGET convTest PROPERTY CXX_STANDARD 17)
