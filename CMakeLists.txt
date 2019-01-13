cmake_minimum_required(VERSION 2.8)
project(mtcnn C CXX)
SET(CMAKE_BUILD_TYPE "Release")
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(ncnn)
include_directories(
        ${CMAKE_BINARY_DIR}/ncnn/src/
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/src/
        ${CMAKE_CURRENT_LIST_DIR}/ncnn/src/layer
        ${CMAKE_CURRENT_LIST_DIR}/src)

file(GLOB MTCNN_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
        )
set(MTCNN_COMPILE_CODE ${MTCNN_SRC})
add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
add_definitions(-Ofast)
add_definitions(-ffast-math)
add_definitions(-ftree-vectorize)
add_definitions(-fvisibility=hidden -fvisibility-inlines-hidden)
add_executable(mtcnn ${MTCNN_COMPILE_CODE})
target_link_libraries(mtcnn ${CMAKE_BINARY_DIR}/ncnn/src/libncnn.a m)
