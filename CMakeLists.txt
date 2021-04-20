cmake_minimum_required(VERSION 3.16)
project(swin_transformer_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
enable_testing()

include_directories(model)
include(CTest)

add_executable(swin_transformer_cpp main.cpp)
