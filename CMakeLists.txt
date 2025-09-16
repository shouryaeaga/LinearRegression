cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


project(machine_learning_algos VERSION 0.1.0 LANGUAGES C CXX)

file(GLOB_RECURSE SOURCES src/*.cpp)

add_executable(machine_learning_algos ${SOURCES})

target_include_directories(machine_learning_algos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

