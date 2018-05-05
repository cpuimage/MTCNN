cmake_minimum_required(VERSION 2.8)

project(mtcnn C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/src/ncnn
        ${CMAKE_CURRENT_LIST_DIR}/src/ncnn/layer
        ${CMAKE_CURRENT_LIST_DIR}/src)

file(GLOB MTCNN_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
        )
set(MTCNN_COMPILE_CODE ${MTCNN_SRC})
file(GLOB NCNN_SRC
        ${CMAKE_CURRENT_LIST_DIR}/src/ncnn/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/ncnn/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/ncnn/layer/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/ncnn/layer/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/ncnn/layer/x86/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/ncnn/layer/x86/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/ncnn/layer/arm/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/ncnn/layer/arm/*.h
        )
set(NCNN_COMPILE_CODE ${NCNN_SRC})
add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
add_definitions(-Ofast)
add_definitions(-ffast-math)
add_definitions(-ftree-vectorize)
add_definitions(-fvisibility=hidden -fvisibility-inlines-hidden)
add_executable(mtcnn ${MTCNN_COMPILE_CODE} ${NCNN_COMPILE_CODE})
